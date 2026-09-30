"""Legged stages (contact model recorded in flags.contact_version). Each calls the entry point the legged scripts ran:

| stage | existing code |
|---|---|
| collect | `python -m rrp.cli data legged-latent-collect --body B --seeds S --out D` per shard (options shard_size, workers) |
| train_bc | rrp.training.legged_bc.train (`python -m rrp.cli train legged-bc train`; BC POSITIVE CONTROL, source bc) |
| train_rep | rrp.training.legged_latent_train.train_rep (`python -m rrp.cli train legged-latent rep`) |
| probes | rrp.training.legged_latent_train.fit_probe (post-hoc probe for nosem; `... probe`) |
| train_flow, flow_ft | rrp.training.legged_latent_train.train_flow (`... flow`) |
| dagger_collect | `python -m rrp.cli train legged-dagger collect` |
| refit | rrp.training.legged_dagger.refit (`... refit --config`) |
| eval_r1, eval_r2, heldout | `python -m rrp.cli suite legged` over seed chunks + `rrp suite legged-summary` |
| edits | `rrp suite legged --edit` per edit + `rrp suite legged-edit-effects` |

Legged system 0 has no previous-action input (zero_prev_action does not apply). A legged dataset whose manifest
records a contact version different from flags.contact_version is refused (no silent mixing of physics versions);
data without a recorded version counts as contact_v1 (legacy), so it is refused for any other declared version.
Physics selection (W8): every simulating subprocess (collect, DAgger, evaluations, edits) runs with
RRP_CONTACT_MODEL = flags.contact_version, collected shards must record that version, and evaluation rows must
report a scene built with it and checkpoints whose training data (when recorded) used it.
Sealed split (D-138, D-146 H5): every data / training stage calls the `rrp.core.sealed.SealedSplit` guard (sealed bodies only on
target-adaptation seeds, evaluation / development seeds never in training); a sealed body is evaluated only by eval_tracker /
eval_r1 / eval_r2 / heldout under options `sealed_cell: {method, train_seed}` (100 evaluation-range scenes, run once, logged in
artifacts/runs/humanoid/sealed_log.jsonl); validate_tracker and edits refuse it.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

from rrp.core.sealed import SealedSplit
from rrp.harness.pipelines.base import apply_gate, StageContext, StageError, register_stage

EVAL = ["-m", "rrp.cli", "suite", "legged"]


def _json_safe(x):
    return json.loads(json.dumps(x, default=str))


def _seed_list(spec: str) -> list[int]:
    from rrp.core.runs import parse_seed_spec
    return parse_seed_spec(str(spec))


def _seed_range(spec: str) -> tuple[int, int]:
    a, _, b = str(spec).partition("-")
    return int(a), int(b or a)


def sealed_train_guard(bodies, seeds=(), what: str = "training") -> None:
    SealedSplit.load().assert_train_allowed(bodies, seeds, what=what)


def sealed_data_guard(root, bodies) -> None:
    """Refuse a trainer whose dataset shards (root/<body>/s*.json) hold seeds it may not read for these bodies."""
    SealedSplit.load().assert_dataset_allowed(root, bodies)


def _tracker_bodies(o: dict) -> list[str]:
    """Bodies a train_tracker node trains on: options body / args.body, and the recipe's body / groups."""
    out = [str(x) for x in (o.get("body"), (o.get("args") or {}).get("body")) if x]
    if o.get("recipe"):
        from rrp.harness.train.tracker_recipes import recipe_record
        r, _ = recipe_record(str(o["recipe"]))
        out += ([str(r["body"])] if r.get("body") else []) + [b for g in r.get("groups") or [] for b in g[0]]
    return sorted(set(out))


def sealed_evaluation(scope):
    """Stage decorator: a stage that evaluates `scope(ctx) -> (body, scene_seeds)`. Non-sealed bodies run as before; a sealed
    body needs options `sealed_cell: {method, train_seed}` and runs inside SealedSplit.sealed_eval (once, logged)."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapped(ctx: StageContext):
            body, scenes = scope(ctx)
            split = SealedSplit.load()
            if not split.is_sealed_body(body):
                return fn(ctx)
            sc = ctx.opts.get("sealed_cell")
            if not sc or "method" not in sc or "train_seed" not in sc:
                raise StageError(f"{body} is sealed: options.sealed_cell = {{method, train_seed}} is required (run once)")
            with split.sealed_eval(dict(body=body, method=sc["method"], train_seed=sc["train_seed"], scenes=list(scenes))):
                return fn(ctx)
        return wrapped
    return deco


def _range_scope(ctx: StageContext):
    o = ctx.opts
    a, b = _seed_range(o.get("seeds", "10000-10029"))
    return o["body"], range(a, b + 1)


def _tracker_scope(ctx: StageContext):
    o = ctx.opts
    s0 = int(o.get("seed0", 7200))
    return o["body"], range(s0, s0 + int(o.get("n", 20)))


def check_contact_version(ctx: StageContext, data_dir: str | None) -> None:
    """Refuse training on data whose recorded contact version differs from the declared one."""
    if not data_dir:
        return
    from rrp.harness.data.manifest import read_manifest
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
    from rrp.bodies.actuator import resolve_mode
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


@register_stage("legged", "collect", source="scripted_teacher")
def collect(ctx: StageContext) -> dict:
    """Legged latent dataset shards (waypoint teacher + tracker; DART sigmas cycled over seeds). The seed range is
    split into shards of `shard_size` (default: one shard) run `workers` at a time, as scripts/legged_latent_collect.sh
    (shard files s<a>-<b>.npz). Output: <out>/<body>/ (the `data` root for train_rep/train_bc)."""
    o = ctx.opts
    body = o["body"]
    out = o.get("out_dir", ctx.rc.out)
    a, b = _seed_range(o["seeds"])
    sealed_train_guard([body], range(a, b + 1), "collect")
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
        argv = ["-m", "rrp.cli", "data", "legged-latent-collect", "--body", body, "--seeds", f"{s}-{e}", "--out", out, *extra]
        jobs.append((argv, physics_env(ctx, OMP_NUM_THREADS=1, CUDA_VISIBLE_DEVICES=""),
                     ctx.root / out / body / f"log_s{s}.txt"))
    ctx.run_parallel(jobs, int(o.get("workers", 1)))
    from rrp.harness.data.manifest import read_manifest
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
    from rrp.harness.eval.gates import check_legged_dataset
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
    from rrp.envs.mujoco.legged_tracker import tracker_path
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


@register_stage("legged", "validate_tracker", source="learned_tracker")
def validate_tracker(ctx: StageContext) -> dict:
    """Tracker validation (rrp.evaluation.tracker_validation protocol v2 + the W6 robustness check) and the D-112 tracker
    gate. options: body, actor (default: the installed tracker for the contact version), kind (learned|cpg), seeds (5),
    robust (true), tracker_sha256 (optional: the actor file must have this sha256, checked BEFORE validating, so a DAG
    validates exactly the tracker its `collect` node declares). A gate verdict 'fail' fails the node (exit GATE_EXIT,
    reason in gate_report.json)."""
    o = ctx.opts
    body = o["body"]
    SealedSplit.load().assert_eval_allowed(body)
    check_tracker_sha(ctx)
    out = ctx.out / "validation.json"
    argv = ["-m", "rrp.cli", "suite", "tracker-validation", "--body", body, "--kind", o.get("kind", "learned"),
            "--seeds", str(o.get("seeds", 5)), "--contact", str(ctx.rc.flags.contact_version).replace("contact_", ""),
            "--out", str(out), "--gate-dir", str(ctx.out)]
    if o.get("actor"):
        argv += ["--actor", str(o["actor"])]
    elif ctx.inp("actor", required=False):          # D-126: a freshly trained tracker (train_tracker output)
        argv += ["--actor", str(ctx.inp("actor"))]
    am = declared_actuator_mode(ctx)
    if am is not None:                  # D-126 #14: validate under the declared actuator mode (legacy CLI name v1 = ideal)
        from rrp.bodies.actuator import legacy_mode_name
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


@register_stage("legged", "train_tracker", source="learned_tracker")
def train_tracker(ctx: StageContext) -> dict:
    """D-126 #13: tracker training through run-dag. options: body, recipe (a name in rrp.harness.train.tracker_recipes:
    CPU_RECIPES or WARP_RECIPES, or a JSON path; optional), args ({option dest: value}, explicit overrides; True = bare flag),
    resume (default true: a retried lease continues from checkpoint.pt), actuator_mode (D-126 #14; wins over the recipe),
    engine (cpu = `rrp train tracker-cpu`, default; warp = `rrp train tracker-warp`, GPU MuJoCo Warp, W13; declare
    resources.gpu), pythonpath (extra PYTHONPATH entries, e.g. the isolated Warp install), resume_from (a run directory
    whose checkpoint.pt / actor.pt / meta.json seed this run's directory once, so a resume segment continues an earlier run
    under a new run id). Output: actor.pt in the run dir (a NEW tracker; nothing is installed). The contact model is
    flags.contact_version."""
    o = ctx.opts
    sealed_train_guard(_tracker_bodies(o), (), "tracker training")
    warp = o.get("engine", "cpu") == "warp"
    if o.get("resume_from") and not (ctx.out / "checkpoint.pt").exists():
        import shutil
        src = Path(o["resume_from"]).expanduser()
        src = src if src.is_absolute() else ctx.root / src
        if not (src / "checkpoint.pt").exists():
            raise StageError(f"resume_from {src}: no checkpoint.pt (restore the run directory first)")
        ctx.out.mkdir(parents=True, exist_ok=True)
        for f in ("checkpoint.pt", "actor.pt", "meta.json"):
            if (src / f).exists():
                shutil.copy2(src / f, ctx.out / f)
    argv = ["-m", "rrp.cli", "train", "tracker-warp" if warp else "tracker-cpu", "--out", ctx.rc.out]
    if not warp:
        argv += ["--contact", str(ctx.rc.flags.contact_version).replace("contact_", "")]
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
        from rrp.bodies.actuator import legacy_mode_name
        argv += ["--actuator", legacy_mode_name(am)]
    if o.get("resume", True):
        argv.append("--resume")
    extra = {} if (o.get("gpu") or warp) else {"CUDA_VISIBLE_DEVICES": ""}
    env = physics_env(ctx, OMP_NUM_THREADS=1, **extra)
    if o.get("pythonpath"):
        env["PYTHONPATH"] = ":".join([str(Path(p).expanduser()) for p in o["pythonpath"]] + [env["PYTHONPATH"]])
    ctx.run(argv, env=env)
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


@register_stage("legged", "eval_tracker", source="learned_tracker")
@sealed_evaluation(_tracker_scope)
def eval_tracker(ctx: StageContext) -> dict:
    """W13: C-MuJoCo evaluation of a tracker / task expert (full self-collision; never the training simulator). options: task
    (waypoint | steps | gap | gap_smoke), body, actor (else input `actor`), seed0, n, pythonpath (as train_tracker), plus per task: steps `h_fracs` (list; one
    result file per step height), gap `level`. task waypoint with input `validation` (a validate_tracker output) also writes
    lab_gate.json (rrp.harness.eval.humanoid_eval.lab_gate: no-fall 1.0, forward >= 0.8, turn >= 0.5, slip < 0.15, D-112 verdict,
    waypoint success). Verdicts are data (exit 0); the source label of every result is in its JSON."""
    o = ctx.opts
    task, body = o["task"], o["body"]
    actor = str(o["actor"]) if o.get("actor") else ctx.inp("actor")
    seed0, n = int(o.get("seed0", 7200)), int(o.get("n", 20))
    env = physics_env(ctx, OMP_NUM_THREADS=1, **({} if task == "gap_smoke" else {"CUDA_VISIBLE_DEVICES": ""}))
    if o.get("pythonpath"):
        env["PYTHONPATH"] = ":".join([str(Path(p).expanduser()) for p in o["pythonpath"]] + [env["PYTHONPATH"]])
    tool = {"waypoint": "contact-waypoint", "steps": "humanoid-steps", "gap": "humanoid-gap", "gap_smoke": "humanoid-gap-smoke"}[task]
    base = ["-m", "rrp.cli", "suite", tool, body, actor]
    results = {}
    if task == "gap_smoke":
        res = ctx.out / "gap_smoke.json"
        ctx.run(base + ["--out", str(res)], env=env)
        results["gap_smoke"] = res
    elif task == "steps":
        for hf in o.get("h_fracs", [None]):
            res = ctx.out / (f"steps_h{hf:.2f}.json" if hf is not None else "steps.json")
            ctx.run(base + ["--seed0", str(seed0), "--n", str(n), "--out", str(res)] + (["--h-frac", str(hf)] if hf is not None else []), env=env)
            results[res.stem] = res
    else:
        res = ctx.out / f"{task}.json"
        ctx.run(base + ["--seed0", str(seed0), "--n", str(n), "--out", str(res)] + (["--level", str(o["level"])] if task == "gap" and "level" in o else []), env=env)
        results[task] = res
    metrics = {}
    for k, r in results.items():
        d = json.loads(r.read_text())
        metrics[k] = {x: d[x] for x in ("n", "success", "fell", "status", "source") if x in d} or d
    outs = {k: str(Path(ctx.rc.out) / r.name) for k, r in results.items()}
    if task == "waypoint" and ctx.inp("validation", required=False):
        from rrp.harness.eval.humanoid_eval import lab_gate
        vpath = ctx.root / ctx.inp("validation")
        rep = vpath.parent / "gate_report.json"
        lab = lab_gate(json.loads(vpath.read_text()), json.loads(rep.read_text()) if rep.exists() else None,
                       json.loads(results["waypoint"].read_text()))
        (ctx.out / "lab_gate.json").write_text(json.dumps(lab, indent=1, default=str))
        outs["lab_gate"] = str(Path(ctx.rc.out) / "lab_gate.json")
        metrics["lab_gate"] = lab
    return dict(outputs=outs, metrics=metrics, source_detail=f"{task}:{body}:{actor}")


@register_stage("legged", "train_bc", source="bc")
def train_bc(ctx: StageContext) -> dict:
    """Legged BC POSITIVE CONTROL (no packet; behaviour cloning of the scripted teacher -> tracker targets)."""
    from rrp.harness.train.legged_bc import train as fn
    cfg = ctx.native
    check_contact_version(ctx, cfg.get("data"))
    res = fn(cfg, ctx.out)
    pol = str(Path(ctx.rc.out) / "policy.pt")
    return dict(outputs={"policy": pol}, metrics=_json_safe(res), source_detail=pol)


@register_stage("legged", "train_rep", source="learned")
def train_rep(ctx: StageContext) -> dict:
    """Legged Stage A (E, R, P)."""
    from rrp.harness.train.legged_latent_train import train_rep as fn
    cfg = ctx.native
    sealed_data_guard(cfg["data"], cfg["bodies"])
    check_contact_version(ctx, cfg.get("data"))
    res = fn(cfg, ctx.out)
    rep = str(Path(ctx.rc.out) / "representation.pt")
    return dict(outputs={"representation": rep}, metrics=_json_safe(res), source_detail=rep)


@register_stage("legged", "probes", source="learned")
def probes(ctx: StageContext) -> dict:
    """Post-hoc measurement probe on the frozen encoder (nosem diagnostic)."""
    from rrp.harness.train.legged_latent_train import fit_probe
    cfg = dict(representation=ctx.inp("representation"), steps=int(ctx.opts.get("steps", 4000)))
    (ctx.out / "probe_cfg.json").write_text(json.dumps(cfg))
    res = fit_probe(cfg, ctx.out)
    return dict(outputs={"probe": str(Path(ctx.rc.out) / "probe_posthoc.pt")}, metrics=_json_safe(res),
                source_detail=cfg["representation"])


def _flow(ctx: StageContext) -> dict:
    from rrp.harness.train.legged_latent_train import train_flow as fn
    res = fn(ctx.native, ctx.out)
    pol = str(Path(ctx.rc.out) / "policy.pt")
    return dict(outputs={"policy": pol}, metrics=_json_safe(res), source_detail=pol)


@register_stage("legged", "train_flow", source="learned")
def train_flow(ctx: StageContext) -> dict:
    """Legged system i."""
    return _flow(ctx)


@register_stage("legged", "flow_ft", source="learned")
def flow_ft(ctx: StageContext) -> dict:
    """Legged system i continued/fine-tuned (config carries its init)."""
    return _flow(ctx)


@register_stage("legged", "dagger_collect", source="bc")
def dagger_collect(ctx: StageContext) -> dict:
    """Legged DAgger buffer: rollouts (route bc | oracle_bc | generated), labels = stateless BC expert."""
    o = ctx.opts
    sealed_train_guard([o["body"]], _seed_list(o["seeds"]), "dagger collect")
    argv = ["-m", "rrp.cli", "train", "legged-dagger", "collect", "--route", o.get("route", "oracle_bc"), "--body", o["body"],
            "--seeds", o["seeds"], "--out", ctx.rc.out, "--bc", ctx.inp("bc")]
    for k in ("rep", "flow", "realizer"):
        v = ctx.inp(k, required=False)
        if v:
            argv += [f"--{k}", v]
    if "max_s" in o:
        argv += ["--max-s", str(o["max_s"])]
    ctx.run(argv, env=physics_env(ctx, CUDA_VISIBLE_DEVICES=""))
    return dict(outputs={}, metrics=dict(body=o["body"], seeds=o["seeds"]), source_detail=ctx.inp("bc"))


@register_stage("legged", "refit", source="learned")
def refit(ctx: StageContext) -> dict:
    """Legged system-0 refit on the frozen encoder from DAgger buffers."""
    from rrp.harness.train.legged_dagger import refit as fn
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
    """The former legged_ladder driver: split the seed range into `par` chunks, run in parallel, concatenate, summarize."""
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
        argv = [*EVAL, *_route_args(ctx, route), "--bodies", body, "--seeds", f"{s}-{e}", "--out", str(part)]
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
    ctx.run(["-m", "rrp.cli", "suite", "legged-summary", str(rows)])
    summ = json.loads(rows.with_suffix(".summary.json").read_text())
    summ.update(cv, checkpoint_contact=ck)
    if summ["n"] != b - a + 1:
        raise StageError(f"{rows}: {summ['n']} rows, expected {b - a + 1}")
    rel = str(Path(ctx.rc.out) / body / rows.name)
    src = {"teacher": "scripted_teacher", "bc": "bc", "r1": "oracle", "r1qd0": "oracle", "r1t": "privileged_teacher",
           "r2": "learned"}[route]
    return dict(outputs={"rows": rel, "summary": rel.replace(".jsonl", ".summary.json")},
                metrics=_json_safe(summ), source=src, source_detail=ctx.inp("flow", required=False) or ctx.inp("bc", required=False) or rel)


@register_stage("legged", "eval_r1", source="oracle")
@sealed_evaluation(_range_scope)
def eval_r1(ctx: StageContext) -> dict:
    """R1 ORACLE DIAGNOSTIC (stateless BC chunk encoded -> system 0)."""
    return _ladder(ctx, "r1")


@register_stage("legged", "eval_r2", source="learned")
@sealed_evaluation(_range_scope)
def eval_r2(ctx: StageContext) -> dict:
    """R2 deployable route (flow -> system 0)."""
    return _ladder(ctx, "r2")


@register_stage("legged", "heldout", source="learned")
@sealed_evaluation(_range_scope)
def heldout(ctx: StageContext) -> dict:
    """R2 on a held-out body (the body must not be in the training bodies given in options.train_bodies)."""
    if ctx.opts["body"] in ctx.opts.get("train_bodies", []):
        raise StageError("heldout body is a training body")
    return _ladder(ctx, "r2")


@register_stage("legged", "edits", source="learned")
def edits(ctx: StageContext) -> dict:
    """Causal packet edits with irrelevant-edit controls (scripts/legged_edit_suite.sh) + paired effects."""
    o = ctx.opts
    body, route = o["body"], o.get("route", "r2")
    SealedSplit.load().assert_eval_allowed(body)
    ck = check_checkpoints_contact(ctx)
    eds = o.get("edits") or ["none", "probe_yaw:0.6", "probe_yaw:-0.6", "probe_goal_mirror", "probe_halt", "contact:0:1",
                             "contact:0:0", "rand_norm:1", "rand_norm:2", "rand_norm:4", "rand_norm:8", "rand_norm:12", "zero"]
    out = ctx.out / o.get("suite", "edits")
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for ed in eds:
        f = out / f"{ed.replace(':', '_')}.jsonl"
        f.unlink(missing_ok=True)
        argv = [*EVAL, *_route_args(ctx, route), "--bodies", body, "--seeds", o.get("seeds", "10000-10019"),
                "--edit", ed, "--t-edit", str(o.get("t_edit", 2.0)), "--max-s", str(o.get("max_s", 5.0)), "--out", str(f)]
        jobs.append((argv, physics_env(ctx, OMP_NUM_THREADS=1, CUDA_VISIBLE_DEVICES=""), out / f"{ed.replace(':', '_')}.log"))
    ctx.run_parallel(jobs, int(o.get("workers", 1)))
    for ed in eds:
        f = out / f"{ed.replace(':', '_')}.jsonl"
        check_rows_contact(ctx, [json.loads(x) for x in f.read_text().splitlines() if x.strip()], str(f))
    ctx.run(["-m", "rrp.cli", "suite", "legged-edit-effects", str(out)])
    if o.get("mirror_effect"):         # scripts/legged_fixrep_eval.sh: paired effect toward the mirrored goal side
        ctx.run(["-m", "rrp.cli", "suite", "legged-mirror-effect", str(out), *o["mirror_effect"]])
    return dict(outputs={"suite": str(Path(ctx.rc.out) / out.name)}, metrics=dict(edits=eds, contact_version=ctx.rc.flags.contact_version, checkpoint_contact=ck),
                source_detail=ctx.inp("flow", required=False) or "")
