"""Legged stages (contact model recorded in flags.contact_version). Each calls the entry point the legged scripts ran:

| stage | existing code |
|---|---|
| collect | `python -m rrp.data.legged_latent_collect --body B --seeds S --out D` per shard (options shard_size, workers) |
| train_bc | rrp.training.legged_bc.train (`python -m rrp.learning.legged_bc train`; BC POSITIVE CONTROL, source bc) |
| train_rep | rrp.training.legged_latent_train.train_rep (`python -m rrp.learning.legged_latent_train rep`) |
| probes | rrp.training.legged_latent_train.fit_probe (post-hoc probe for nosem; `... probe`) |
| train_flow, flow_ft | rrp.training.legged_latent_train.train_flow (`... flow`) |
| dagger_collect | `python -m rrp.training.legged_dagger collect` |
| refit | rrp.training.legged_dagger.refit (`... refit --config`) |
| eval_r1, eval_r2, heldout | `python -m rrp.evaluation.legged_latent_eval` over seed chunks + scripts/legged_ladder_summary.py (scripts/legged_ladder.sh) |
| edits | legged_latent_eval --edit per edit + scripts/legged_edit_effects.py (scripts/legged_edit_suite.sh) |

Legged system 0 has no previous-action input (zero_prev_action does not apply). A legged dataset whose manifest
records a contact version different from flags.contact_version is refused (no silent mixing of physics versions);
data without a recorded version counts as contact_v1 (legacy), so it is refused for any other declared version.
Physics selection (W8): every simulating subprocess (collect, DAgger, evaluations, edits) runs with
RRP_CONTACT_MODEL = flags.contact_version, collected shards must record that version, and evaluation rows must
report a scene built with it and checkpoints whose training data (when recorded) used it.
"""
from __future__ import annotations

import json
from pathlib import Path

from rrp.pipelines.base import apply_gate, StageContext, StageError, register

EVAL = "rrp.evaluation.legged_latent_eval"


def _json_safe(x):
    return json.loads(json.dumps(x, default=str))


def _seed_range(spec: str) -> tuple[int, int]:
    a, _, b = str(spec).partition("-")
    return int(a), int(b or a)


def check_contact_version(ctx: StageContext, data_dir: str | None) -> None:
    """Refuse training on data whose recorded contact version differs from the declared one."""
    if not data_dir:
        return
    from rrp.data.manifest import read_manifest
    want = ctx.rc.flags.contact_version
    d = ctx.root / data_dir
    found = set()
    for m in list(d.rglob("*.manifest.json"))[:200] if d.is_dir() else []:
        try:
            prov = read_manifest(m).get("provenance")
            cv = getattr(getattr(prov, "physics", None), "contact_version", None)
            if cv:
                found.add(cv)
        except Exception:   # unreadable/legacy manifests carry no contact version: legacy = contact_v1 (W3)
            continue
    if not found and want != "contact_v1":
        raise StageError(f"{data_dir}: no manifest records a contact version (legacy data = contact_v1), "
                         f"but flags.contact_version is {want!r}")
    if found and found != {want}:
        raise StageError(f"{data_dir}: data contact version(s) {sorted(found)} != flags.contact_version {want!r}")


def declared_actuator_mode(ctx: StageContext) -> str | None:
    """options.actuator_mode (D-126 #14; ideal | v1lat | v2, alias v1 = ideal), canonicalised; None when not declared."""
    am = ctx.opts.get("actuator_mode")
    if am is None:
        return None
    from rrp.physics.actuator import resolve_mode
    return resolve_mode(str(am))


def physics_env(ctx: StageContext, **extra) -> dict:
    """Subprocess env whose simulator uses the declared contact model ($RRP_CONTACT_MODEL, rrp.morphology.contact) and, when
    options.actuator_mode is declared (D-126 #14), the declared actuator mode ($RRP_ACTUATOR_MODE, plus
    $RRP_ACTUATOR_LATENCY_MS from options.actuator_latency_ms). Undeclared -> the environment is exactly as before."""
    cv = ctx.rc.flags.contact_version
    am = declared_actuator_mode(ctx)
    act = {}
    if am is not None:
        act["RRP_ACTUATOR_MODE"] = am
        if ctx.opts.get("actuator_latency_ms") is not None:
            act["RRP_ACTUATOR_LATENCY_MS"] = float(ctx.opts["actuator_latency_ms"])
    return ctx.env(**({"RRP_CONTACT_MODEL": cv} if cv else {}), **act, **extra)


def _row_actuator_mode(r: dict) -> str:
    return ((r.get("actuator_mode") or {}).get("actuator_mode")) or "ideal"


def _ckpt_contact(path: Path, depth: int = 0) -> str | None:
    """Training-data contact version recorded in a legged checkpoint's `_provenance.physics`; a flow checkpoint
    (no data of its own) inherits it from the representation named in its cfg. None = not recorded (legacy)."""
    import torch
    st = torch.load(str(path), map_location="cpu", weights_only=False)
    ph = (st.get("_provenance") or {}).get("physics") or {}
    if ph.get("contact_version"):
        return ph["contact_version"]
    rep = (st.get("cfg") or {}).get("representation")
    if rep and depth < 2 and Path(rep).exists():
        return _ckpt_contact(Path(rep), depth + 1)
    return None


def check_checkpoints_contact(ctx: StageContext, keys=("representation", "flow", "bc", "realizer")) -> dict:
    """Refuse to evaluate a checkpoint trained on data of another contact version (unrecorded = contact_v1)."""
    want = ctx.rc.flags.contact_version
    seen = {}
    for k in keys:
        pth = ctx.inp(k, required=False)
        if not pth:
            continue
        cv = _ckpt_contact(ctx.root / pth) or "contact_v1"
        seen[k] = cv
        if want and cv != want:
            raise StageError(f"{k} {pth} was trained on {cv} data; flags.contact_version is {want!r}")
    return seen


def check_rows_contact(ctx: StageContext, rows: list[dict], where: str) -> dict:
    """Every evaluation row must come from a scene built with flags.contact_version, and every checkpoint that
    records its training-data physics must have been trained on that version. Returns the versions seen."""
    want = ctx.rc.flags.contact_version
    bad, seen = [], set()
    for r in rows:
        got = r.get("contact_version")
        seen.add(got)
        want_lim = ctx.opts.get("actuator_limits")
        if want_lim and r.get("actuator_limits") != want_lim:
            bad.append(f"seed {r.get('seed')}: actuator limits {r.get('actuator_limits')} != declared {want_lim}")
        want_am = declared_actuator_mode(ctx)
        if want_am and _row_actuator_mode(r) != want_am:
            bad.append(f"seed {r.get('seed')}: actuator mode {_row_actuator_mode(r)} != declared {want_am}")
        want_sha = ctx.opts.get("tracker_sha256")
        if want_sha and r.get("tracker_sha256") != want_sha:
            bad.append(f"seed {r.get('seed')}: tracker sha {r.get('tracker_sha256')} != declared {want_sha}")
        if want and got != want:
            bad.append(f"seed {r.get('seed')}: scene {got}")
        for k, pv in (r.get("checkpoint_provenance") or {}).items():
            cv = ((pv or {}).get("physics") or {}).get("contact_version") if isinstance(pv, dict) else None
            if want and cv and cv != want:
                bad.append(f"seed {r.get('seed')}: {k} trained on {cv}")
    if bad:
        raise StageError(f"{where}: contact version != {want!r}: " + "; ".join(bad[:5]))
    return dict(contact_versions=sorted(str(x) for x in seen))


@register("legged", "collect", source="scripted_teacher")
def collect(ctx: StageContext) -> dict:
    """Legged latent dataset shards (waypoint teacher + tracker; DART sigmas cycled over seeds). The seed range is
    split into shards of `shard_size` (default: one shard) run `workers` at a time, as scripts/legged_latent_collect.sh
    (shard files s<a>-<b>.npz). Output: <out>/<body>/ (the `data` root for train_rep/train_bc)."""
    o = ctx.opts
    body = o["body"]
    out = o.get("out_dir", ctx.rc.out)
    a, b = _seed_range(o["seeds"])
    size = int(o.get("shard_size", b - a + 1))
    extra = []
    for k in ("tracker", "sigmas"):
        if k in o:
            extra += [f"--{k}", str(o[k])]
    if o.get("arc_only"):
        extra += ["--arc-only"]
    (ctx.root / out / body).mkdir(parents=True, exist_ok=True)
    jobs = []
    for s in range(a, b + 1, size):
        e = min(s + size - 1, b)
        argv = ["-m", "rrp.data.legged_latent_collect", "--body", body, "--seeds", f"{s}-{e}", "--out", out, *extra]
        jobs.append((argv, physics_env(ctx, OMP_NUM_THREADS=1, CUDA_VISIBLE_DEVICES=""),
                     ctx.root / out / body / f"log_s{s}.txt"))
    ctx.run_parallel(jobs, int(o.get("workers", 1)))
    from rrp.data.manifest import read_manifest
    eps, cvs, trk, shas, lims = [], set(), set(), set(), set()
    for s in range(a, b + 1, size):
        e = min(s + size - 1, b)
        sh = ctx.root / out / body / f"s{s}-{e}.json"
        if not sh.exists():
            raise StageError(f"missing shard {sh}")
        d = json.loads(sh.read_text())
        eps += d["episodes"]
        prov = read_manifest(sh.with_suffix(".manifest.json")).get("provenance")
        cvs.add(getattr(getattr(prov, "physics", None), "contact_version", None))
        trk |= {str(m.get("tracker_version")) for m in d["episodes"]}
        shas |= {str(m.get("tracker_sha256")) for m in d["episodes"]}
        lims |= {str((m.get("physics") or {}).get("actuator_limits")) for m in d["episodes"]}
    want = ctx.rc.flags.contact_version
    if o.get("tracker_sha256") and shas != {o["tracker_sha256"]}:
        raise StageError(f"collected with tracker sha {sorted(shas)}, DAG declares {o['tracker_sha256']}")
    if o.get("actuator_limits") and lims != {o["actuator_limits"]}:
        raise StageError(f"collected with actuator limits {sorted(lims)}, DAG declares {o['actuator_limits']}")
    want_am = declared_actuator_mode(ctx)
    if want_am:
        ams = {_row_actuator_mode(m) for m in eps}
        if ams != {want_am}:
            raise StageError(f"collected with actuator mode(s) {sorted(ams)}, DAG declares {want_am}")
    if cvs != {want}:
        raise StageError(f"collected shards record contact version(s) {sorted(map(str, cvs))} != {want!r}")
    if len(eps) != b - a + 1:
        raise StageError(f"{len(eps)} episodes collected, expected {b - a + 1}")
    st = [m["status"] for m in eps]
    from rrp.evaluation.gates import check_legged_dataset
    gate = apply_gate(ctx, check_legged_dataset(eps, dict(name=ctx.rc.run_id)))       # W6 dataset gate (D-112)
    return dict(outputs={"data": out}, metrics=dict(body=body, seeds=o["seeds"], episodes=len(eps), gate=gate,
                                                    success=st.count("success"), fell=st.count("fell"),
                                                    failure=st.count("failure"), contact_version=want,
                                                    tracker_versions=sorted(trk), tracker_sha256=sorted(shas),
                                                    actuator_limits=sorted(lims), dataset_gate=dataset_gate(eps),
                                                    ticks=int(sum(m["ticks"] for m in eps))),
                source_detail=f"scripted_teacher:waypoint -> {'|'.join(sorted(trk))}")


def dataset_gate(eps: list[dict]) -> dict:
    """D-112 legged dataset gate: stance slip ratio < 0.15 on >= 95% of episodes, and 0 falls at DART noise 0.
    Episodes without a motion record (collected before W8 recorded it) make the slip part `unmeasured`."""
    sl = [(e.get("motion") or {}).get("slip_ratio") for e in eps]
    have = [x for x in sl if x is not None]
    n0 = [e for e in eps if float(e.get("sigma", 0)) == 0.0]
    falls0 = sum(e.get("status") == "fell" for e in n0)
    frac = (sum(x < 0.15 for x in have) / len(have)) if have else None
    slip_ok = None if len(have) < len(eps) else bool(frac >= 0.95)
    return dict(gate="D-112 legged dataset", n=len(eps), slip_measured=len(have),
                slip_lt_0p15_frac=None if frac is None else round(frac, 4),
                slip_median=None if not have else round(float(sorted(have)[len(have) // 2]), 4),
                slip_ok=slip_ok, noise0_episodes=len(n0), noise0_falls=falls0, falls_ok=falls0 == 0,
                passed=None if slip_ok is None else bool(slip_ok and falls0 == 0))


def check_tracker_sha(ctx: StageContext) -> str | None:
    """validate_tracker: refuse when options.tracker_sha256 is declared and the actor file (options.actor, else the installed
    tracker for flags.contact_version) has another sha256. Returns the file's sha256 (None when nothing is declared)."""
    want = ctx.opts.get("tracker_sha256")
    if not want:
        return None
    import hashlib
    from rrp.envs.legged_tracker import tracker_path
    path = Path(ctx.opts["actor"]) if ctx.opts.get("actor") else tracker_path(
        ctx.opts["body"], str(ctx.rc.flags.contact_version).replace("contact_", ""))
    if not path.is_absolute():
        path = ctx.root / path
    if not path.exists():
        raise StageError(f"tracker {path} not found (declared sha256 {want})")
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != str(want):
        raise StageError(f"tracker sha {got} != declared {want} ({path})")
    return got


@register("legged", "validate_tracker", source="learned_tracker")
def validate_tracker(ctx: StageContext) -> dict:
    """Tracker validation (rrp.evaluation.tracker_validation protocol v2 + the W6 robustness check) and the D-112 tracker
    gate. options: body, actor (default: the installed tracker for the contact version), kind (learned|cpg), seeds (5),
    robust (true), tracker_sha256 (optional: the actor file must have this sha256, checked BEFORE validating, so a DAG
    validates exactly the tracker its `collect` node declares). A gate verdict 'fail' fails the node (exit GATE_EXIT,
    reason in gate_report.json)."""
    o = ctx.opts
    body = o["body"]
    check_tracker_sha(ctx)
    out = ctx.out / "validation.json"
    argv = ["-m", "rrp.evaluation.tracker_validation", "--body", body, "--kind", o.get("kind", "learned"),
            "--seeds", str(o.get("seeds", 5)), "--contact", str(ctx.rc.flags.contact_version).replace("contact_", ""),
            "--out", str(out), "--gate-dir", str(ctx.out)]
    if o.get("actor"):
        argv += ["--actor", str(o["actor"])]
    elif ctx.inp("actor", required=False):          # D-126: a freshly trained tracker (train_tracker output)
        argv += ["--actor", str(ctx.inp("actor"))]
    am = declared_actuator_mode(ctx)
    if am is not None:                  # D-126 #14: validate under the declared actuator mode (legacy CLI name v1 = ideal)
        from rrp.physics.actuator import legacy_mode_name
        argv += ["--actuator", legacy_mode_name(am), "--latency-ms", str(float(o.get("actuator_latency_ms", 0.0)))]
    if o.get("robust", True):
        argv += ["--robust"]
    ctx.run(argv, env=physics_env(ctx, OMP_NUM_THREADS=1, CUDA_VISIBLE_DEVICES=""))
    v = json.loads(out.read_text())
    gate = apply_gate(ctx, v["w6_gate"])
    return dict(outputs={"validation": str(out.relative_to(ctx.root))},
                metrics=dict(body=body, gate=gate, contact_gate=v["gate"].get("contact_gate", {}).get("passed"),
                             tracker_version=v.get("tracker_version"), tracker_sha=v.get("tracker_sha"),
                             **({"actuator_mode": am} if am is not None else {})),
                source_detail=v.get("tracker_version"))


@register("legged", "train_tracker", source="learned_tracker")
def train_tracker(ctx: StageContext) -> dict:
    """D-126 #13: tracker training (rrp.training.tracker_training) through run-dag. options: body, recipe (a name in
    rrp.training.tracker_recipes or a JSON path; optional), args ({option dest: value}, explicit overrides; True = bare flag),
    resume (default true: a retried lease continues from checkpoint.pt), actuator_mode (D-126 #14; wins over the recipe).
    Output: actor.pt in the run dir (a NEW tracker; nothing is installed). The contact model is flags.contact_version."""
    o = ctx.opts
    argv = ["-m", "rrp.training.tracker_training", "--out", ctx.rc.out,
            "--contact", str(ctx.rc.flags.contact_version).replace("contact_", "")]
    if o.get("body"):
        argv += ["--body", str(o["body"])]
    if o.get("recipe"):
        argv += ["--recipe", str(o["recipe"])]
    for k, v in (o.get("args") or {}).items():
        flag = "--" + str(k).replace("_", "-")
        if v is True:
            argv.append(flag)
        elif v not in (False, None):
            argv += [flag, str(v)]
    am = declared_actuator_mode(ctx)
    if am is not None:
        from rrp.physics.actuator import legacy_mode_name
        argv += ["--actuator", legacy_mode_name(am)]
    if o.get("resume", True):
        argv.append("--resume")
    ctx.run(argv, env=physics_env(ctx, OMP_NUM_THREADS=1, **({} if o.get("gpu") else {"CUDA_VISIBLE_DEVICES": ""})))
    import hashlib
    actor = ctx.out / "actor.pt"
    if not actor.exists():
        raise StageError(f"{actor} missing after training")
    meta = json.loads((ctx.out / "meta.json").read_text())
    rel = str(Path(ctx.rc.out) / "actor.pt")
    return dict(outputs={"actor": rel}, metrics=dict(body=meta["body"], actor_sha256=hashlib.sha256(actor.read_bytes()).hexdigest(),
                                                     recipe=(meta.get("recipe") or {}).get("name") or (meta.get("recipe") or {}).get("path"),
                                                     reward_options=meta.get("reward_options"),
                                                     terrain_curriculum=bool(meta.get("terrain_curriculum")),
                                                     actuator=meta.get("actuator"), contact_model=meta.get("contact_model")),
                source_detail=f"learned_tracker:{meta['body']}:{rel}")


@register("legged", "train_bc", source="bc")
def train_bc(ctx: StageContext) -> dict:
    """Legged BC POSITIVE CONTROL (no packet; behaviour cloning of the scripted teacher -> tracker targets)."""
    from rrp.training.legged_bc import train as fn
    cfg = ctx.native
    check_contact_version(ctx, cfg.get("data"))
    res = fn(cfg, ctx.out)
    pol = str(Path(ctx.rc.out) / "policy.pt")
    return dict(outputs={"policy": pol}, metrics=_json_safe(res), source_detail=pol)


@register("legged", "train_rep", source="learned")
def train_rep(ctx: StageContext) -> dict:
    """Legged Stage A (E, R, P)."""
    from rrp.training.legged_latent_train import train_rep as fn
    cfg = ctx.native
    check_contact_version(ctx, cfg.get("data"))
    res = fn(cfg, ctx.out)
    rep = str(Path(ctx.rc.out) / "representation.pt")
    return dict(outputs={"representation": rep}, metrics=_json_safe(res), source_detail=rep)


@register("legged", "probes", source="learned")
def probes(ctx: StageContext) -> dict:
    """Post-hoc measurement probe on the frozen encoder (nosem diagnostic)."""
    from rrp.training.legged_latent_train import fit_probe
    cfg = dict(representation=ctx.inp("representation"), steps=int(ctx.opts.get("steps", 4000)))
    (ctx.out / "probe_cfg.json").write_text(json.dumps(cfg))
    res = fit_probe(cfg, ctx.out)
    return dict(outputs={"probe": str(Path(ctx.rc.out) / "probe_posthoc.pt")}, metrics=_json_safe(res),
                source_detail=cfg["representation"])


def _flow(ctx: StageContext) -> dict:
    from rrp.training.legged_latent_train import train_flow as fn
    res = fn(ctx.native, ctx.out)
    pol = str(Path(ctx.rc.out) / "policy.pt")
    return dict(outputs={"policy": pol}, metrics=_json_safe(res), source_detail=pol)


@register("legged", "train_flow", source="learned")
def train_flow(ctx: StageContext) -> dict:
    """Legged system i."""
    return _flow(ctx)


@register("legged", "flow_ft", source="learned")
def flow_ft(ctx: StageContext) -> dict:
    """Legged system i continued/fine-tuned (config carries its init)."""
    return _flow(ctx)


@register("legged", "dagger_collect", source="bc")
def dagger_collect(ctx: StageContext) -> dict:
    """Legged DAgger buffer: rollouts (route bc | oracle_bc | generated), labels = stateless BC expert."""
    o = ctx.opts
    argv = ["-m", "rrp.training.legged_dagger", "collect", "--route", o.get("route", "oracle_bc"), "--body", o["body"],
            "--seeds", o["seeds"], "--out", ctx.rc.out, "--bc", ctx.inp("bc")]
    for k in ("rep", "flow", "realizer"):
        v = ctx.inp(k, required=False)
        if v:
            argv += [f"--{k}", v]
    if "max_s" in o:
        argv += ["--max-s", str(o["max_s"])]
    ctx.run(argv, env=physics_env(ctx, CUDA_VISIBLE_DEVICES=""))
    return dict(outputs={}, metrics=dict(body=o["body"], seeds=o["seeds"]), source_detail=ctx.inp("bc"))


@register("legged", "refit", source="learned")
def refit(ctx: StageContext) -> dict:
    """Legged system-0 refit on the frozen encoder from DAgger buffers."""
    from rrp.training.legged_dagger import refit as fn
    res = fn(ctx.native, ctx.out)
    return dict(outputs={}, metrics=_json_safe(res), source_detail=ctx.rc.out)


def _route_args(ctx: StageContext, route: str) -> list[str]:
    a = []
    get = lambda k: ctx.inp(k, required=False)
    if route == "teacher":
        return ["--arc-only", "g1"] if ctx.opts.get("body") == "g1" else []
    if route == "bc":
        return ["--bc", ctx.inp("bc")]
    if route in ("r1", "r1qd0"):
        a = ["--rep", ctx.inp("representation"), "--bc", ctx.inp("bc"), "--oracle-bc"] + (["--zero-qd"] if route == "r1qd0" else [])
    elif route == "r1t":
        a = ["--rep", ctx.inp("representation"), "--oracle"]
    elif route == "r2":
        a = ["--flow", ctx.inp("flow")]
    else:
        raise StageError(f"unknown legged route {route!r}")
    if get("realizer"):
        a += ["--realizer", get("realizer")]
    if get("posthoc_probe"):
        a += ["--posthoc-probe", get("posthoc_probe")]
    return a


def _ladder(ctx: StageContext, default_route: str) -> dict:
    """scripts/legged_ladder.sh: split the seed range into `par` chunks, run in parallel, concatenate, summarize."""
    o = ctx.opts
    body, tag, par = o["body"], o["tag"], int(o.get("workers", 1))
    route = o.get("route", default_route)
    ck = check_checkpoints_contact(ctx)
    a, b = _seed_range(o.get("seeds", "10000-10029"))
    n = (b - a + 1 + par - 1) // par
    out = ctx.out / body
    out.mkdir(parents=True, exist_ok=True)
    jobs, parts = [], []
    for s in range(a, b + 1, n):
        e = min(s + n - 1, b)
        part = out / f"{route}_{tag}.part{s}.jsonl"
        parts.append(part)
        argv = ["-m", EVAL, *_route_args(ctx, route), "--bodies", body, "--seeds", f"{s}-{e}", "--out", str(part)]
        jobs.append((argv, physics_env(ctx, OMP_NUM_THREADS=1, CUDA_VISIBLE_DEVICES=""), ctx.out / f"{route}_{tag}.part{s}.log"))
    ctx.run_parallel(jobs, par)
    rows = out / f"{route}_{tag}.jsonl"
    rows.write_text("".join(p.read_text() for p in parts))
    for p in parts:
        p.unlink()
    rl = [json.loads(x) for x in rows.read_text().splitlines() if x.strip()]
    cv = check_rows_contact(ctx, rl, str(rows))
    cv["tracker_sha256"] = sorted({str(r.get("tracker_sha256")) for r in rl})
    cv["trackers"] = sorted({str(r.get("tracker")) for r in rl})
    ctx.run(["scripts/legged_ladder_summary.py", str(rows)])
    summ = json.loads(rows.with_suffix(".summary.json").read_text())
    summ.update(cv, checkpoint_contact=ck)
    if summ["n"] != b - a + 1:
        raise StageError(f"{rows}: {summ['n']} rows, expected {b - a + 1}")
    rel = str(Path(ctx.rc.out) / body / rows.name)
    src = {"teacher": "scripted_teacher", "bc": "bc", "r1": "oracle", "r1qd0": "oracle", "r1t": "privileged_teacher",
           "r2": "learned"}[route]
    return dict(outputs={"rows": rel, "summary": rel.replace(".jsonl", ".summary.json")},
                metrics=_json_safe(summ), source=src, source_detail=ctx.inp("flow", required=False) or ctx.inp("bc", required=False) or rel)


@register("legged", "eval_r1", source="oracle")
def eval_r1(ctx: StageContext) -> dict:
    """R1 ORACLE DIAGNOSTIC (stateless BC chunk encoded -> system 0)."""
    return _ladder(ctx, "r1")


@register("legged", "eval_r2", source="learned")
def eval_r2(ctx: StageContext) -> dict:
    """R2 deployable route (flow -> system 0)."""
    return _ladder(ctx, "r2")


@register("legged", "heldout", source="learned")
def heldout(ctx: StageContext) -> dict:
    """R2 on a held-out body (the body must not be in the training bodies given in options.train_bodies)."""
    if ctx.opts["body"] in ctx.opts.get("train_bodies", []):
        raise StageError("heldout body is a training body")
    return _ladder(ctx, "r2")


@register("legged", "edits", source="learned")
def edits(ctx: StageContext) -> dict:
    """Causal packet edits with irrelevant-edit controls (scripts/legged_edit_suite.sh) + paired effects."""
    o = ctx.opts
    body, route = o["body"], o.get("route", "r2")
    ck = check_checkpoints_contact(ctx)
    eds = o.get("edits") or ["none", "probe_yaw:0.6", "probe_yaw:-0.6", "probe_goal_mirror", "probe_halt", "contact:0:1",
                             "contact:0:0", "rand_norm:1", "rand_norm:2", "rand_norm:4", "rand_norm:8", "rand_norm:12", "zero"]
    out = ctx.out / o.get("suite", "edits")
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for ed in eds:
        f = out / f"{ed.replace(':', '_')}.jsonl"
        f.unlink(missing_ok=True)
        argv = ["-m", EVAL, *_route_args(ctx, route), "--bodies", body, "--seeds", o.get("seeds", "10000-10019"),
                "--edit", ed, "--t-edit", str(o.get("t_edit", 2.0)), "--max-s", str(o.get("max_s", 5.0)), "--out", str(f)]
        jobs.append((argv, physics_env(ctx, OMP_NUM_THREADS=1, CUDA_VISIBLE_DEVICES=""), out / f"{ed.replace(':', '_')}.log"))
    ctx.run_parallel(jobs, int(o.get("workers", 1)))
    for ed in eds:
        f = out / f"{ed.replace(':', '_')}.jsonl"
        check_rows_contact(ctx, [json.loads(x) for x in f.read_text().splitlines() if x.strip()], str(f))
    ctx.run(["scripts/legged_edit_effects.py", str(out)])
    if o.get("mirror_effect"):         # scripts/legged_fixrep_eval.sh: paired effect toward the mirrored goal side
        ctx.run(["scripts/legged_mirror_effect.py", str(out), *o["mirror_effect"]])
    return dict(outputs={"suite": str(Path(ctx.rc.out) / out.name)}, metrics=dict(edits=eds, contact_version=ctx.rc.flags.contact_version, checkpoint_contact=ck),
                source_detail=ctx.inp("flow", required=False) or "")
