"""ComputerWorld pointer stages (track pointer, D-142; D-145 P4c). Each stage calls existing code; nothing is reimplemented.

| stage | code |
|---|---|
| collect | `rrp train pointer collect` per task (scripted teacher demos, DART pointer noise) -> `<task>.npz` (+ .json); peer store |
| train_rep | `rrp train pointer rep --variant semfix|nosem` (E encoder + R learned system 0 + P packet probe) -> `rep.pt` |
| train_flow | `rrp train pointer flow --target latent|eng` (system i on the frozen representation, or on the engineered encoding) -> `flow.pt` |
| train_bc | `rrp train pointer bc` (BC baseline, same public inputs and demos) -> `bc.pt` |
| probes | `rrp train pointer probe` (post-hoc packet probes + metadata-only control; `--flow` = probe generated packets) -> `probe.json` |
| edits | `rrp train pointer edit` (probe-guided packet retargeting realised by system 0 in closed loop) -> `edit.json` |
| eval_r1 | `rrp eval --policy pointer_oracle={representation}`: teacher packets -> frozen E -> LEARNED system 0 (ORACLE, diagnostic) |
| eval_r2 | `rrp eval --policy pointer_latent|pointer_bc|pointer_oracle` on the split's seeds (`options.seed_set`, default dev) |

Options: `tasks` (default all four cw/* tasks), `seed_set` (dev | sealed_id | sealed_heldout of the split file; sealed sets are
evaluated ONCE, D-142), `workers`. Params of the training stages are passed through as `--<name>` flags (steps, batch, lr, dz,
w_sem, lv_min, ...); `variant` is the factor set (semfix | nosem; `params.w_sem` must agree, see runconfig._check_variant).
Inputs: `data` (a collect run), `representation` (`@rep:rep.pt`), `flow` (`@flow:flow.pt`), `checkpoint` (`@bc:bc.pt`).
Every trained model reads only `rrp.policies.pointer.public_features`; weights and datasets stay in the peer store.
"""
from __future__ import annotations

import json
from pathlib import Path

from rrp.harness.pipelines.base import StageContext, StageError, register

TRAIN = ["-m", "rrp.cli", "train", "pointer"]
SPLIT_PATH = "research/splits/cworld_pointer_v1.json"   # = rrp.harness.train.pointer.SPLIT_PATH (that module needs torch)
TASKS = ("cw/calc_sum", "cw/open_type", "cw/drag_window", "cw/fill_form")


def _tasks(ctx: StageContext) -> list[str]:
    ts = list(ctx.opts.get("tasks") or TASKS)
    bad = [t for t in ts if t not in TASKS]
    if bad:
        raise StageError(f"unknown pointer tasks {bad} (known: {list(TASKS)})")
    return ts


def _split(ctx: StageContext) -> dict:
    return json.loads((ctx.root / SPLIT_PATH).read_text())


def _slug(task: str) -> str:
    return task.split("/", 1)[1]


def _flags(params: dict, skip=()) -> list[str]:
    """{steps: 20000, w_sem: 0.5} -> ["--steps", "20000", "--w-sem", "0.5"] (argparse names of `rrp train pointer`)."""
    argv: list[str] = []
    for k, v in params.items():
        if k in skip or v is None:
            continue
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return argv


def _data(ctx: StageContext) -> list[str]:
    d = ctx.root / ctx.inp("data")
    files = sorted(str(p.relative_to(ctx.root)) for p in d.glob("*.npz"))
    if not files:
        raise StageError(f"no *.npz demos under {d}")
    return files


def _train(ctx: StageContext, cmd: str, out_name: str, extra: list[str], *, use_variant: bool = False) -> Path:
    out = ctx.out / out_name
    argv = [*TRAIN, cmd, "--data", *_data(ctx), "--out", str(out), "--seed", str(ctx.rc.seed), *extra]
    if use_variant:
        argv += ["--variant", ctx.rc.variant]
    ctx.run(argv, log_to=ctx.out / f"{cmd}.log")
    return out


def _outputs(out: Path, ctx: StageContext, *, meta: bool = True) -> dict:
    o = {out.name: str(out.relative_to(ctx.root))}
    if meta and out.with_suffix(".json").exists():
        o["meta"] = str(out.with_suffix(".json").relative_to(ctx.root))
    return o


# ------------------------------------------------------------------------------------------------ data
@register("pointer", "collect", source="scripted_teacher")
def collect(ctx: StageContext) -> dict:
    """Scripted-teacher demos per task (held-out variants and eval seeds skipped by the split guard)."""
    p = dict(ctx.rc.params)
    n = int(p.pop("episodes", 6000))
    jobs, outs = [], {}
    for t in _tasks(ctx):
        out = ctx.out / f"{_slug(t)}.npz"
        outs[_slug(t)] = str(out.relative_to(ctx.root))
        if out.exists() and ctx.opts.get("resume", True):
            continue
        argv = [*TRAIN, "collect", "--task", t, "--episodes", str(n), "--seed", str(ctx.rc.seed), *_flags(p, skip=("seed",)),
                "--out", str(out)]
        jobs.append((argv, None, ctx.out / f"collect_{_slug(t)}.log"))
    ctx.run_parallel(jobs, int(ctx.opts.get("workers", 2)))
    return dict(outputs=outs, metrics=dict(tasks=len(outs), episodes_per_task=n))


# ------------------------------------------------------------------------------------------------ training
@register("pointer", "train_rep", source="learned")
def train_rep(ctx: StageContext) -> dict:
    """E + R + P on demo chunks; variant semfix (probe loss on z, bounded NLL) | nosem (weights 0)."""
    if ctx.rc.variant not in ("semfix", "nosem"):
        raise StageError("pointer train_rep: variant must be semfix or nosem")
    out = _train(ctx, "rep", "rep.pt", _flags(ctx.rc.params), use_variant=True)
    return dict(outputs=_outputs(out, ctx), source_detail=str(out.relative_to(ctx.root)))


@register("pointer", "train_flow", source="learned")
def train_flow(ctx: StageContext) -> dict:
    """System i (rectified flow): on the frozen representation's posterior means (target latent) or on the engineered
    encoding (target eng, realised by the SCRIPTED engineered system 0)."""
    target = ctx.rc.params.get("target", "latent")
    extra = _flags(ctx.rc.params)
    if target == "latent":
        extra += ["--representation", ctx.inp("representation")]
    out = _train(ctx, "flow", "flow.pt", extra)
    return dict(outputs=_outputs(out, ctx), source_detail=str(out.relative_to(ctx.root)))


@register("pointer", "train_bc", source="bc")
def train_bc(ctx: StageContext) -> dict:
    """BC baseline: context -> 7-tick chunk, the same public inputs and demos."""
    out = _train(ctx, "bc", "bc.pt", _flags(ctx.rc.params))
    return dict(outputs=_outputs(out, ctx), source_detail=str(out.relative_to(ctx.root)))


@register("pointer", "probes", source="learned")
def probes(ctx: StageContext) -> dict:
    """Post-hoc packet probes + the metadata-only control (diagnostic); `inputs.flow` probes generated packets."""
    extra = _flags(ctx.rc.params) + ["--representation", ctx.inp("representation")]
    if ctx.inp("flow", required=False):
        extra += ["--flow", ctx.inp("flow")]
    out = _train(ctx, "probe", "probe.json", extra)
    return dict(outputs=_outputs(out, ctx, meta=False))


@register("pointer", "edits", source="learned")
def edits(ctx: StageContext) -> dict:
    """Causal packet edits: probe-guided retargeting of a received packet, realised by system 0 in closed loop."""
    extra = _flags(ctx.rc.params) + ["--representation", ctx.inp("representation"), "--flow", ctx.inp("flow")]
    out = _train(ctx, "edit", "edit.json", extra)
    return dict(outputs=_outputs(out, ctx, meta=False))


# ------------------------------------------------------------------------------------------------ rollouts
def _seeds(ctx: StageContext, seed_set: str, task: str) -> str:
    s = _split(ctx)["seeds"][task].get(seed_set)
    if not s:
        raise StageError(f"{task}: split {SPLIT_PATH} has no seed set {seed_set!r}")
    return ",".join(str(x) for x in s)


def _policy(ctx: StageContext, kind: str) -> str:
    kw = {}
    if kind == "pointer_oracle":
        if ctx.inp("representation", required=False):
            kw["representation"] = ctx.inp("representation")
    elif kind == "pointer_latent":
        kw["flow"] = ctx.inp("flow")
        if ctx.inp("representation", required=False):
            kw["representation"] = ctx.inp("representation")
    elif kind == "pointer_bc":
        kw["checkpoint"] = ctx.inp("checkpoint")
    else:
        raise StageError(f"options.policy {kind!r}: pointer_oracle | pointer_latent | pointer_bc")
    return kind + ("=" + json.dumps(kw) if kw else "")


def _evaluate(ctx: StageContext, kind: str) -> dict:
    o = ctx.opts
    seed_set = o.get("seed_set", "dev")
    pol = _policy(ctx, kind)
    jobs, outs = [], {}
    for t in _tasks(ctx):
        if seed_set not in _split(ctx)["seeds"][t]:
            continue                       # e.g. no held-out variants for cw/drag_window
        out = ctx.out / f"{_slug(t)}_{seed_set}.jsonl"
        outs[f"{_slug(t)}_{seed_set}"] = str(out.relative_to(ctx.root))
        if out.exists() and o.get("resume", True):
            continue
        argv = ["-m", "rrp.cli", "eval", "--policy", pol, "--env", "computerworld", "--task", t, "--body", "cw_pointer",
                "--seeds", _seeds(ctx, seed_set, t), "--batch", str(o.get("batch", 16)), "--out", str(out)]
        jobs.append((argv, None, ctx.out / f"eval_{_slug(t)}.log"))
    ctx.run_parallel(jobs, int(o.get("workers", 1)))
    metrics = {}
    for name, rel in outs.items():
        summ = ctx.root / Path(rel).with_suffix(".summary.json")
        if summ.exists():
            s = json.loads(summ.read_text())
            metrics[name] = f"{s.get('successes')}/{s.get('attempted')}"
    return dict(outputs=outs, metrics=metrics)


@register("pointer", "eval_r1", source="oracle")
def eval_r1(ctx: StageContext) -> dict:
    """R1 rung (ORACLE, diagnostic): teacher chunk -> encoder -> system 0. With inputs.representation the system 0 is the
    LEARNED realizer, without it the SCRIPTED engineered one."""
    return _evaluate(ctx, "pointer_oracle")


@register("pointer", "eval_r2", source="learned")
def eval_r2(ctx: StageContext) -> dict:
    """Deployable route on the split's seeds: options.policy pointer_latent (default) | pointer_bc."""
    kind = ctx.opts.get("policy", "pointer_latent")
    res = _evaluate(ctx, kind)
    res["source_detail"] = ctx.inp("checkpoint" if kind == "pointer_bc" else "flow")
    if kind == "pointer_bc":
        res["source"] = "bc"
    return res
