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

from rrp.pipelines.base import StageContext, StageError, register

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


def physics_env(ctx: StageContext, **extra) -> dict:
    """Subprocess env whose simulator uses the declared contact model ($RRP_CONTACT_MODEL, rrp.morphology.contact)."""
    cv = ctx.rc.flags.contact_version
    return ctx.env(**({"RRP_CONTACT_MODEL": cv} if cv else {}), **extra)


def check_rows_contact(ctx: StageContext, rows: list[dict], where: str) -> dict:
    """Every evaluation row must come from a scene built with flags.contact_version, and every checkpoint that
    records its training-data physics must have been trained on that version. Returns the versions seen."""
    want = ctx.rc.flags.contact_version
    bad, seen = [], set()
    for r in rows:
        got = r.get("contact_version")
        seen.add(got)
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
    eps, cvs, trk = [], set(), set()
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
    want = ctx.rc.flags.contact_version
    if cvs != {want}:
        raise StageError(f"collected shards record contact version(s) {sorted(map(str, cvs))} != {want!r}")
    if len(eps) != b - a + 1:
        raise StageError(f"{len(eps)} episodes collected, expected {b - a + 1}")
    st = [m["status"] for m in eps]
    return dict(outputs={"data": out}, metrics=dict(body=body, seeds=o["seeds"], episodes=len(eps),
                                                    success=st.count("success"), fell=st.count("fell"),
                                                    failure=st.count("failure"), contact_version=want,
                                                    tracker_versions=sorted(trk),
                                                    ticks=int(sum(m["ticks"] for m in eps))),
                source_detail=f"scripted_teacher:waypoint -> {'|'.join(sorted(trk))}")


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
    cv = check_rows_contact(ctx, [json.loads(x) for x in rows.read_text().splitlines() if x.strip()], str(rows))
    ctx.run(["scripts/legged_ladder_summary.py", str(rows)])
    summ = json.loads(rows.with_suffix(".summary.json").read_text())
    summ.update(cv)
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
    return dict(outputs={"suite": str(Path(ctx.rc.out) / out.name)}, metrics=dict(edits=eds, contact_version=ctx.rc.flags.contact_version),
                source_detail=ctx.inp("flow", required=False) or "")
