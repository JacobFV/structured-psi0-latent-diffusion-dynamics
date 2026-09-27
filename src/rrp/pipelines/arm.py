"""Arm (single-manipulator pick_place) stages. Each calls the code the chain scripts ran:

| stage | existing code (what the chain scripts ran) |
|---|---|
| collect | rrp.data.generate.generate (`rrp data generate`) |
| pack | rrp.data.packed.pack_dataset (`rrp data pack`) |
| train_rep | rrp.training.latent_train.train_representation (`rrp latent train-representation`) |
| probes | rrp.training.latent_train.fit_probes_on_frozen (`rrp latent fit-probes`) |
| train_flow, flow_ft | rrp.training.latent_train.train_latent_flow (`rrp latent train-flow`) |
| refit | rrp.training.latent_train.refit_realizer (scripts/ladder_refit.py) |
| dagger_collect | scripts/ladder.py --collect-dagger (scripts/ladder_dagger_collect.sh, EXPERT=bc [FLOW=] [GENCTX=]) |
| eval_r1 | scripts/ladder.py --route oracle --oracle-expert bc (scripts/ladder_eval_orcbc.sh; ORACLE DIAGNOSTIC) |
| eval_r2, heldout | scripts/ladder.py --route generated (chain r2eval) |
| edits | `rrp latent semantic-edits --route generated` (chain semedit) |

scripts/ladder.py keeps its logic in the script's main(), so the evaluation stages run it as a subprocess with the
same arguments (byte-identical behaviour; moving that main into rrp.evaluation is a follow-up). Outputs go to the
stage's own out dir, never to the shared artifacts/runs/ladder_v1/.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

from rrp.pipelines.base import StageContext, StageError, register

LADDER = "scripts/ladder.py"
TRAIN_BODIES = ("panda_pg2", "parm5_pg2", "parm5_tf3", "parm5l_tf3", "parm5s_pg2", "parm6_pg2", "parm6_tf3",
                "parm7_pg2", "parm7_tf3", "sawyer_pg2", "sawyer_tf3", "ur5e_pg2", "ur5e_tf3")
TARGET_BODIES = ("xarm7_pg2", "xarm7_tf3", "panda_tf3")      # never in the ladder (scripts/ladder.py refuses them)
DEFAULT_BC = ("artifacts/runs/baselines_bc_ckpts/direct1701_u12000.pt", "direct1701_u12000")
SEMEDIT_CONDITIONS = "control,goal_shift,rebind_desc,irrelevant_distractor,orthogonal_matched,control_replay"


def _prev_action(ctx: StageContext) -> str:
    return "zero" if ctx.rc.flags.zero_prev_action else "own"


def _rel(ctx: StageContext, *parts) -> str:
    return str(Path(ctx.rc.out, *parts))


def _json_safe(x):
    return json.loads(json.dumps(x, default=str))


def _versions(res: dict) -> dict:
    return {k: res[k] for k in ("latent_space_version", "realizer_compat_version") if isinstance(res.get(k), str)}


# ------------------------------------------------------------------------------------------------ data
@register("arm", "collect", source="scripted_teacher")
def collect(ctx: StageContext) -> dict:
    """Scripted-teacher dataset generation (privileged teacher; public featurizer)."""
    cfg = ctx.native
    p = ctx.out / "data_config.json"
    p.write_text(json.dumps(cfg, indent=1))
    argv = ["-m", "rrp.cli", "data", "generate", "--config", str(p)]
    if ctx.opts.get("workers"):
        argv += ["--workers", str(ctx.opts["workers"])]
    ctx.run(argv)
    return dict(outputs={"manifest": str(Path(cfg["out_dir"]) / "manifest.json")})


@register("arm", "pack", source="scripted_teacher")
def pack(ctx: StageContext) -> dict:
    """Pack a dataset into memory-mapped training chunks (rrp data pack)."""
    cfg = ctx.native
    out = cfg.get("out_dir", ctx.rc.out)
    p = ctx.out / "pack_config.json"
    p.write_text(json.dumps(cfg, indent=1))
    ctx.run(["-m", "rrp.cli", "data", "pack", "--config", str(p), "--out", out])
    return dict(outputs={"meta": str(Path(out) / "meta.json")})


# ------------------------------------------------------------------------------------------------ training
@register("arm", "train_rep", source="learned")
def train_rep(ctx: StageContext) -> dict:
    """Stage A: encoder E + system 0 R + probes P."""
    from rrp.training.latent_train import train_representation
    cfg = ctx.native
    d = Path(cfg["out_dir"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg, indent=1))            # as `rrp latent train-representation`
    res = train_representation(cfg, d)
    rep = str(d / "representation.pt")
    return dict(outputs={"representation": rep}, metrics=_json_safe(res), versions=_versions(res), source_detail=rep)


@register("arm", "probes", source="learned")
def probes(ctx: StageContext) -> dict:
    """Measurement probes on the frozen, detached packet (diagnostic)."""
    from rrp.training.latent_train import fit_probes_on_frozen
    o = ctx.opts
    out = ctx.out / o.get("out_name", "probes.json")
    res = fit_probes_on_frozen(Path(ctx.inp("representation")), Path(ctx.inp("packed_dir")), out,
                               steps=int(o.get("steps", 6000)), metadata_only=bool(o.get("metadata_only", False)),
                               binding_cf=float(o.get("binding_cf", 0.0)))
    return dict(outputs={}, metrics=_json_safe(res), source_detail=ctx.inp("representation"))


def _flow(ctx: StageContext) -> dict:
    from rrp.training.latent_train import train_latent_flow
    cfg = ctx.native
    d = Path(cfg["out_dir"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg, indent=1))            # as `rrp latent train-flow`
    res = train_latent_flow(cfg, d)
    if res.get("interrupted"):
        raise StageError("flow training was interrupted (policy_interrupted.pt); rerun resumes from policy_last.pt")
    outs = {"policy": str(d / "policy.pt")}
    alias = ctx.opts.get("final_snapshot_alias")
    if alias:   # the chains' F0 step: snap_final_s20000.pt = copy of the final policy.pt (warm-start path of ft/gdag1)
        shutil.copyfile(d / "policy.pt", d / alias)
        outs["snapshot_alias"] = str(d / alias)
    return dict(outputs=outs, metrics=_json_safe(res), versions=_versions(res), source_detail=outs["policy"])


@register("arm", "train_flow", source="learned")
def train_flow(ctx: StageContext) -> dict:
    """System i: conditional flow over the packet (from scratch)."""
    return _flow(ctx)


@register("arm", "flow_ft", source="learned")
def flow_ft(ctx: StageContext) -> dict:
    """System i fine-tune (init_from; optional generator-DAgger contexts gen_dagger)."""
    if "init_from" not in ctx.rc.inputs:
        raise StageError("flow_ft needs inputs.init_from")
    return _flow(ctx)


@register("arm", "refit", source="learned")
def refit(ctx: StageContext) -> dict:
    """System-0 refit on the frozen encoder from DAgger buffers (scripts/ladder_refit.py)."""
    from rrp.training.latent_train import refit_realizer
    cfg = ctx.native
    res = refit_realizer(cfg, Path(cfg["out_dir"]))
    if res.get("interrupted"):
        raise StageError("refit was interrupted; rerun resumes from rz_last.pt")
    rep = str(Path(cfg["out_dir"]) / "representation.pt")
    return dict(outputs={"representation": rep}, metrics=_json_safe(res), versions=_versions(res), source_detail=rep)


# ------------------------------------------------------------------------------------------------ rollouts
def _bc(ctx: StageContext) -> tuple[str, str]:
    p = ctx.inp("bc_policy", required=False) or DEFAULT_BC[0]
    return p, ctx.opts.get("bc_label", DEFAULT_BC[1] if p == DEFAULT_BC[0] else Path(p).stem)


def _robots(ctx: StageContext, default=TRAIN_BODIES) -> list[str]:
    rs = list(ctx.opts.get("robots") or default)
    bad = [r for r in rs if r in TARGET_BODIES]
    if bad:
        raise StageError(f"target bodies are not allowed in the ladder: {bad}")
    return rs


def _threads_env(ctx: StageContext, threads: int | None):
    return ctx.env(CUDA_VISIBLE_DEVICES="", **({"OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads} if threads else {}))


@register("arm", "dagger_collect", source="bc")
def dagger_collect(ctx: StageContext) -> dict:
    """System-0 DAgger buffers: rollouts of the CURRENT system 0 (mode bc: R1 packets E(BC chunk); mode gen: system i's
    own packets drive the rollout), labels = the stateless BC expert's plan rows (ladder_dagger_collect.sh EXPERT=bc)."""
    o = ctx.opts
    mode = o.get("mode", "bc")
    if mode not in ("bc", "gen"):
        raise StageError(f"dagger_collect mode {mode!r} (bc|gen)")
    if o.get("expert", "bc") != "bc":
        raise StageError("only the stateless BC expert is supported (the shadow-teacher expert is confounded, D-050)")
    genctx = bool(o.get("genctx", False))
    if genctx and mode != "gen":
        raise StageError("genctx needs mode gen (generated route)")
    rep = ctx.inp("representation")
    flow = ctx.inp("flow") if mode == "gen" else None
    bc, bcl = _bc(ctx)
    n, seed = int(o["episodes"]), int(o["seed_start"])
    jobs, bufs = [], {}
    for r in _robots(ctx):
        buf = ctx.out / f"{r}.npz"
        bufs[r] = _rel(ctx, f"{r}.npz")
        if genctx:
            bufs[r + ".genctx"] = _rel(ctx, f"{r}.npz.genctx.pkl")
        if buf.exists() and (not genctx or Path(str(buf) + ".genctx.pkl").exists()):
            continue                                                      # as the script: existing buffers are kept
        route = ["--route", "generated", "--flow", flow] if flow else ["--route", "oracle"]
        argv = [LADDER, *route, "--oracle-expert", "bc", "--policy", bc, "--policy-label", bcl,
                "--prev-action", _prev_action(ctx), "--robot", r, "--n", str(n), "--seed-start", str(seed),
                "--rep", rep, "--out", ctx.rc.out, "--tag", r]
        argv += [] if flow else ["--no-compare"]
        argv += ["--collect-gen-ctx"] if genctx else []
        argv += ["--collect-dagger", bufs[r]]
        jobs.append((argv, _threads_env(ctx, o.get("threads")), ctx.out / f"{r}.log"))
    ctx.run_parallel(jobs, int(o.get("workers", 1)))
    missing = [p for p in bufs.values() if not (ctx.root / p).exists()]
    if missing:
        raise StageError(f"missing buffers: {missing}")
    return dict(outputs=bufs, metrics=dict(robots=len(_robots(ctx)), episodes=n, seed_start=seed, mode=mode,
                                           genctx=genctx), source_detail=f"{bcl}")


def _source_counts(rows_path: Path) -> dict:
    """Eval rows keep their legacy free-string source; the manifest records the canonical Source enum per row
    (contracts.provenance.parse_source) so reports can group by it."""
    from rrp.contracts.provenance import parse_source
    c = Counter()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if line.strip():
                s = json.loads(line).get("source")
                if isinstance(s, str) and "(" in s:        # ladder rows: "learned(system-i flow)", "target_encoder_oracle(...)"
                    s = s.split("(", 1)[0]
                try:
                    c[str(parse_source(s).kind.value)] += 1
                except (ValueError, AttributeError):
                    c["unparsed"] += 1
    return dict(c)


def _ladder_eval(ctx: StageContext, route: str, robots: list[str], seed_starts: list[int], tag_fn, extra: list[str]):
    o = ctx.opts
    n = int(o.get("episodes", 30))
    jobs, results = [], []
    for s in seed_starts:
        for r in robots:
            tag = tag_fn(s)
            out = _rel(ctx, r)
            argv = [LADDER, "--route", route, *extra, "--prev-action", _prev_action(ctx), "--robot", r, "--n", str(n),
                    "--seed-start", str(s), "--tag", tag, "--out", out]
            jobs.append((argv, _threads_env(ctx, o.get("threads")), ctx.out / f"{r}_{tag}.log"))
            results.append((r, s, Path(out) / f"{route}_{tag}.summary.json", Path(out) / f"{route}_{tag}.jsonl"))
    ctx.run_parallel(jobs, int(o.get("workers", 1)))
    metrics, outputs = {}, {}
    for r, s, sp, rp in results:
        summ = json.loads((ctx.root / sp).read_text())
        if summ.get("n") != n:
            raise StageError(f"{sp}: n={summ.get('n')} != {n}")
        key = f"{r}/s{s}"
        metrics[key] = dict(n=summ["n"], success=summ["success"], rate=summ["rate"], wilson95=summ["wilson95"],
                            failed_stage=summ.get("failed_stage"), source_kinds=_source_counts(ctx.root / rp))
        outputs[key] = str(sp)
        outputs[key + "/rows"] = str(rp)
    k = sum(m["success"] for m in metrics.values())
    tot = sum(m["n"] for m in metrics.values())
    metrics["_total"] = dict(success=k, n=tot)
    return outputs, metrics


def _r2(ctx: StageContext, heldout: bool) -> dict:
    o = ctx.opts
    robots = _robots(ctx, default=())
    if not robots:
        raise StageError("options.robots is required for evaluations")
    if heldout and set(robots) & set(o.get("train_robots", TRAIN_BODIES)):
        raise StageError(f"heldout robots overlap the training bodies: {sorted(set(robots) & set(TRAIN_BODIES))}")
    flow, rep = ctx.inp("flow"), ctx.inp("representation")
    tag = o["tag"]
    outputs, metrics = _ladder_eval(ctx, "generated", robots, [int(s) for s in o.get("seed_starts", [3000000])],
                                    lambda s: f"zero_{tag}_s{s}" if _prev_action(ctx) == "zero" else f"{tag}_s{s}",
                                    ["--flow", flow, "--rep", rep] + (["--nfe", str(o["nfe"])] if "nfe" in o else []))
    return dict(outputs=outputs, metrics=metrics, source="learned", source_detail=flow)


@register("arm", "eval_r2", source="learned")
def eval_r2(ctx: StageContext) -> dict:
    """R2 (deployable): generated packet from system i -> system 0, on training bodies / dev seeds."""
    return _r2(ctx, heldout=False)


@register("arm", "heldout", source="learned")
def heldout(ctx: StageContext) -> dict:
    """R2 on held-out bodies (not in the training set)."""
    return _r2(ctx, heldout=True)


@register("arm", "eval_r1", source="oracle")
def eval_r1(ctx: StageContext) -> dict:
    """R1 ORACLE DIAGNOSTIC with the stateless BC expert: packet = E(BC chunk) -> system 0 (ladder_eval_orcbc.sh)."""
    o = ctx.opts
    rep = ctx.inp("representation")
    bc, bcl = _bc(ctx)
    tag = o["tag"]
    outputs, metrics = _ladder_eval(ctx, "oracle", _robots(ctx, default=("panda_pg2", "parm6_tf3")),
                                    [int(s) for s in o.get("seed_starts", [3000000])],
                                    lambda s: f"zero_{tag}_orcbc" if _prev_action(ctx) == "zero" else f"{tag}_orcbc",
                                    ["--oracle-expert", "bc", "--policy", bc, "--policy-label", bcl, "--rep", rep])
    return dict(outputs=outputs, metrics=metrics, source="oracle", source_detail=f"E({rep})+bc:{bcl}")


@register("arm", "edits", source="learned")
def edits(ctx: StageContext) -> dict:
    """Task-context semantic-edit suite on the deployable route (`rrp latent semantic-edits --route generated`),
    sharded over (robot, seed range)."""
    o = ctx.opts
    flow, rep = ctx.inp("flow"), ctx.inp("representation")
    if _prev_action(ctx) != "zero":
        raise StageError("semantic-edits runs with the deployment input (zero previous action) only")
    jobs, outs = [], {}
    for sh in o["shards"]:
        name = sh["name"]
        out = _rel(ctx, name)
        if (ctx.root / out / "semantic_summary_generated.json").exists():
            outs[name] = str(Path(out) / "semantic_summary_generated.json")
            continue
        argv = ["-m", "rrp.cli", "latent", "semantic-edits", "--route", o.get("route", "generated"),
                "--checkpoint", flow, "--representation", rep, "--robots", sh["robot"],
                "--episodes", str(sh["episodes"]), "--seed-start", str(sh["seed_start"]),
                "--max-steps", str(o.get("max_steps", 400)), "--conditions", o.get("conditions", SEMEDIT_CONDITIONS),
                "--out", out]
        jobs.append((argv, _threads_env(ctx, 1), ctx.out / f"{name.replace('/', '_')}.log"))
        outs[name] = str(Path(out) / "semantic_summary_generated.json")
    ctx.run_parallel(jobs, int(o.get("workers", 1)))
    missing = [v for v in outs.values() if not (ctx.root / v).exists()]
    if missing:
        raise StageError(f"incomplete shards: {missing}")
    return dict(outputs=outs, metrics=dict(shards=len(outs)), source_detail=flow)
