"""Dual-arm (M=2) stages (D-126 #33; W12 phase B/C). Each calls existing code; nothing is reimplemented.

| stage | code |
|---|---|
| collect | rrp.data.collect_dual (`python -m rrp.cli data collect-dual --config`), then the dual dataset gate |
|         | (rrp.evaluation.gates.check_dual_dataset; needs `record_quality: true` in the data config, else "incomplete") |
| pack | rrp.data.dual_latent.pack_dual (= `rrp latent pack-dual`) |
| train_rep | rrp.training.latent_train.train_representation on the multi-assembly pack (arm stage) |
| probes | rrp.training.latent_train.fit_probes_on_frozen (arm stage; per-slot probes on a multi pack) |
| train_flow / flow_ft | rrp.training.latent_train.train_latent_flow (arm stages) |
| refit | rrp.training.latent_train.refit_realizer (arm stage) on DAgger buffers given as inputs |
| eval_r2 / heldout | `rrp latent evaluate-dual` per (task, pair) shard, dev pairs / held-out pairs |
| edits | `rrp latent evaluate-dual --packet-edit <edit>` per edit x shard (control = no edit, same seeds) |
| dagger_collect | NOT AVAILABLE: raises. Dual DAgger needs a state-feedback label source (a dual BC expert, as the arm's |
|                | BC-DAgger); the v2/v3 scripted teachers are stateful FSMs that cannot label off-policy states. |

Physics: set the generic stage option `grasp_contact` (rrp.pipelines.base; e.g. v2.1) on collect/eval stages; the dual
stages do not choose a grasp contact version themselves.

Every dual training config must set zero_prev_action: true (B-1 fix, D-045). The v1 dual configs
(configs/latent/rep-dualarm_latent_*_v1.json) predate it and are NOT used by recipes/templates/dual_lineage.yaml;
`train_rep`/`train_flow`/`refit` refuse zero_prev_action false for new (non-legacy) dual configs.
"""
from __future__ import annotations

import json
from pathlib import Path

from rrp.harness.pipelines import arm
from rrp.harness.pipelines.base import StageContext, StageError, apply_gate, register

DUAL_TASKS = ("support_insert", "handover")


def _b1(ctx: StageContext):
    if ctx.rc.legacy is None and not ctx.rc.flags.zero_prev_action:
        raise StageError("dual training needs flags.zero_prev_action: true (B-1 fix; the v1 dual configs predate it)")


# ------------------------------------------------------------------------------------------------ data
@register("dual", "collect", source="scripted_teacher")
def collect(ctx: StageContext) -> dict:
    """Scripted dual teacher collection (v2 default; teacher_version v3 opt-in) + the dual dataset gate."""
    cfg = ctx.native
    p = ctx.out / "data_config.json"
    p.write_text(json.dumps(cfg, indent=1))
    argv = ["-m", "rrp.cli", "data", "collect-dual", "--config", str(p)]
    if ctx.opts.get("workers"):
        argv += ["--workers", str(ctx.opts["workers"])]
    ctx.run(argv)                     # grasp contact version: the generic stage option `grasp_contact` (pipelines.base)
    man_p = ctx.root / cfg["out_dir"] / "manifest.json"
    from rrp.harness.eval.gates import check_dual_dataset
    man = json.loads(man_p.read_text())
    gate = apply_gate(ctx, check_dual_dataset(man.get("episodes") or [], man))
    return dict(outputs={"manifest": str(Path(cfg["out_dir"]) / "manifest.json")}, metrics=dict(gate=gate))


@register("dual", "pack", source="scripted_teacher")
def pack(ctx: StageContext) -> dict:
    """Multi-assembly pack of one or more dual datasets (rrp.data.dual_latent.pack_dual)."""
    from rrp.harness.data.dual_latent import pack_dual
    cfg = ctx.native
    ds = cfg.pop("datasets", None)             # inputs.datasets (DAG refs to collect nodes) fill sources[k].dataset
    if ds is not None:
        ds = ds if isinstance(ds, list) else [ds]
        if len(ds) != len(cfg["sources"]):
            raise StageError(f"inputs.datasets has {len(ds)} entries for {len(cfg['sources'])} sources")
        for src, d in zip(cfg["sources"], ds):
            src.setdefault("dataset", d)
    (ctx.out / "pack_config.json").write_text(json.dumps(cfg, indent=1))
    meta = pack_dual(cfg, keep_parts=bool(ctx.opts.get("keep_parts", False)), log=ctx.log)
    return dict(outputs={"meta": str(Path(cfg["out_dir"]) / "meta.json")},
                metrics=dict(n=meta["n"], robots=meta.get("robots")))


# ------------------------------------------------------------------------------------------------ training
@register("dual", "train_rep", source="learned")
def train_rep(ctx: StageContext) -> dict:
    """Dual Stage A: the arm trainer on a multi-assembly pack."""
    _b1(ctx)
    return arm.train_rep(ctx)


@register("dual", "probes", source="learned")
def probes(ctx: StageContext) -> dict:
    """Per-slot measurement probes on the frozen packet (diagnostic)."""
    return arm.probes(ctx)


@register("dual", "train_flow", source="learned")
def train_flow(ctx: StageContext) -> dict:
    """Dual system i: the arm flow trainer on a multi-assembly pack."""
    _b1(ctx)
    return arm._flow(ctx)


@register("dual", "flow_ft", source="learned")
def flow_ft(ctx: StageContext) -> dict:
    _b1(ctx)
    return arm.flow_ft(ctx)


@register("dual", "refit", source="learned")
def refit(ctx: StageContext) -> dict:
    """System-0 refit on the frozen encoder from DAgger buffers (inputs), as the arm refit."""
    _b1(ctx)
    return arm.refit(ctx)


@register("dual", "dagger_collect", source="bc")
def dagger_collect(ctx: StageContext) -> dict:
    """Not available yet (see module doc): refuses before any compute."""
    raise StageError("dual dagger_collect: no dual state-feedback label source (dual BC expert) exists; "
                     "train_bc is not implemented for the dual family (D-126 #33 follow-up)")


# ------------------------------------------------------------------------------------------------ rollouts
def _shards(ctx: StageContext, key: str) -> list[dict]:
    o = ctx.opts
    sh = o.get(key) or o.get("shards")
    if not sh:
        raise StageError(f"options.{key} (list of {{task, pair, episodes, seed_start}}) required")
    for s in sh:
        if s["task"] not in DUAL_TASKS:
            raise StageError(f"unknown dual task {s['task']}")
    return sh


def _evaluate(ctx: StageContext, shards: list[dict], edit: str | None, tag: str) -> dict:
    o = ctx.opts
    flow = ctx.inp("flow")
    jobs, outs = [], {}
    for sh in shards:
        name = f"{tag}/{sh['task']}_{sh['pair']}_s{sh['seed_start']}"
        out = str(Path(ctx.rc.out, name + ".jsonl"))
        outs[name] = out
        if (ctx.root / out).exists() and o.get("resume", True):
            continue
        (ctx.root / out).parent.mkdir(parents=True, exist_ok=True)
        argv = ["-m", "rrp.cli", "latent", "evaluate-dual", "--checkpoint", flow, "--task", sh["task"],
                "--pairs", sh["pair"], "--episodes", str(sh["episodes"]), "--seed-start", str(sh["seed_start"]),
                "--max-steps", str(sh.get("max_steps", o.get("max_steps", 800))), "--out", out]
        if o.get("cpu", True):
            argv.append("--cpu")
        if ctx.inp("probe", required=False):
            argv += ["--probe", ctx.inp("probe")]
        if edit:
            argv += ["--packet-edit", edit]
        env = ctx.env(OMP_NUM_THREADS=1, MKL_NUM_THREADS=1)          # grasp contact: stage option `grasp_contact`
        jobs.append((argv, env, ctx.out / f"{name.replace('/', '_')}.log"))
    ctx.run_parallel(jobs, int(o.get("workers", 1)))
    missing = [v for v in outs.values() if not (ctx.root / v).exists()]
    if missing:
        raise StageError(f"incomplete shards: {missing[:5]}")
    return outs


@register("dual", "eval_r2", source="learned")
def eval_r2(ctx: StageContext) -> dict:
    """Deployable route (system i packets -> system 0) on dev pairs/seeds."""
    outs = _evaluate(ctx, _shards(ctx, "shards"), None, "r2")
    return dict(outputs=outs, metrics=dict(shards=len(outs)), source_detail=ctx.inp("flow"))


@register("dual", "heldout", source="learned")
def heldout(ctx: StageContext) -> dict:
    """Budget-0 transfer to held-out pairs (xarm7 family, panda_tf3 attachment, aloha)."""
    outs = _evaluate(ctx, _shards(ctx, "heldout_shards"), None, "heldout")
    return dict(outputs=outs, metrics=dict(shards=len(outs)), source_detail=ctx.inp("flow"))


@register("dual", "edits", source="learned")
def edits(ctx: StageContext) -> dict:
    """Packet edits (control = no edit) on the same seeds; options.packet_edits (default [swap_slots])."""
    shards = _shards(ctx, "shards")
    outs = dict(_evaluate(ctx, shards, None, "control"))
    for e in ctx.opts.get("packet_edits", ["swap_slots"]):
        outs.update(_evaluate(ctx, shards, e, e))
    return dict(outputs=outs, metrics=dict(shards=len(outs)), source_detail=ctx.inp("flow"))
