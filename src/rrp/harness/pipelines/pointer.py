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

Options: `tasks` (default all four cw/* tasks), `split` (the split file, default `cworld_pointer_v1`; every stage of one lineage
passes it to the trainers as `--split`, so the no-leak guard and the eval seeds come from the same file), `seed_sets` (a list of
dev | sealed_id | sealed_heldout of the split; `seed_set` = one), `workers`. Params of the training stages are passed through as
`--<name>` flags (steps, batch, lr, dz, w_sem, lv_min, factors, ...; a list becomes repeated values); `params.factors` is the
relation-factor list of the net (`rrp.policies.relations.resolve` items, e.g. `["preset:ui"]`; absent = the empty preset);
`variant` is the semantic factor set (semfix | nosem; `params.w_sem` must agree, see runconfig._check_variant).

Sealed guard (D-142, D-146 C3, round 2 PC). A sealed seed set is evaluated at most ONCE per (split, seed set, task, method), where
a method is the policy kind plus its frozen inputs (flow / representation / checkpoint paths) and the cell's train seed. The cell
goes through `core.sealed.SealedSplit("cworld_pointer_v2").sealed_eval(cell)`: the pinned split file, the run-once log
(`artifacts/runs/pointer/sealed_log.jsonl`: start before the eval jobs, done after a clean exit, an exception leaves the attempt open
on purpose) and `rrp suite sealed-log infra-failure` to re-open a cell after an infrastructure failure. Only splits `core.sealed`
pins can be sealed-evaluated: `cworld_pointer_v1` was consumed once (D-142), so a new question needs the new split.
Defaults. `options.split` = `SPLIT_PATH_V2` (procedural strings: the split's `env_kw` reach the collector, the trainers and every
eval job). `train_flow --target eng` and `train_bc` default to the v2 packet encoding (`eng_version: cw_pointer_eng.v2`, 7-bit key
code) and the instruction-copy key head (`key_head: copy`); a recipe that wants another arm says so (`eng_version:
cw_pointer_eng.v1`, `key_head: free`).
Inputs: `data` (a collect run), `representation` (`@rep:rep.pt`), `flow` (`@flow:flow.pt`), `checkpoint` (`@bc:bc.pt`).
Every trained model reads only `rrp.policies.pointer.public_features`; weights and datasets stay in the peer store.
"""
from __future__ import annotations

import contextlib
from rrp.core.provenance import json_digest
import json
import os
from pathlib import Path

from rrp.core.sealed import SPLITS, SealedSplit, SealedSplitError
from rrp.harness.pipelines.base import StageContext, StageError, register_stage
from rrp.harness.train.pointer.split import SPLIT_PATH_V2

TRAIN = ["-m", "rrp.cli", "train", "pointer"]
TASKS = ("cw/calc_sum", "cw/open_type", "cw/drag_window", "cw/fill_form")
SEAL_SETS = ("sealed_id", "sealed_heldout")
CONSUMED = {"cworld_pointer_v1": "D-142 (sealed_id and sealed_heldout were evaluated once with the frozen checkpoints)"}
V2_EVAL_DEFAULTS = {"eng_version": "cw_pointer_eng.v2", "key_head": "copy"}     # train_flow --target eng, train_bc


def _tasks(ctx: StageContext) -> list[str]:
    ts = list(ctx.opts.get("tasks") or TASKS)
    bad = [t for t in ts if t not in TASKS]
    if bad:
        raise StageError(f"unknown pointer tasks {bad} (known: {list(TASKS)})")
    return ts


def _split_path(ctx: StageContext) -> str:
    return str(ctx.opts.get("split") or SPLIT_PATH_V2)


def _split(ctx: StageContext) -> dict:
    p = ctx.root / _split_path(ctx)
    if not p.exists():
        raise StageError(f"pointer split {_split_path(ctx)} does not exist: declare the split (seed ranges, held-out variants, "
                         "seed lists) before any demo is collected")
    return json.loads(p.read_text())


def _seed_sets(ctx: StageContext) -> list[str]:
    o = ctx.opts
    sets = o.get("seed_sets") or [o.get("seed_set", "dev")]
    return [sets] if isinstance(sets, str) else list(sets)


def _slug(task: str) -> str:
    return task.split("/", 1)[1]


def _flags(params: dict, skip=()) -> list[str]:
    """{steps: 20000, w_sem: 0.5} -> ["--steps", "20000", "--w-sem", "0.5"] (argparse names of `rrp train pointer`)."""
    argv: list[str] = []
    for k, v in params.items():
        if k in skip or v is None:
            continue
        vals = list(v) if isinstance(v, (list, tuple)) else [v]                 # nargs="*" options (--factors)
        if not vals:
            continue                                                             # an empty list is the option's absent default
        argv += [f"--{k.replace('_', '-')}", *[json.dumps(x) if isinstance(x, dict) else str(x) for x in vals]]
    return argv


def _data(ctx: StageContext) -> list[str]:
    d = ctx.root / ctx.inp("data")
    files = sorted(str(p.relative_to(ctx.root)) for p in d.glob("*.npz"))
    if not files:
        raise StageError(f"no *.npz demos under {d}")
    return files


def _env(ctx: StageContext) -> dict:
    """The child env with the stage's own PYTHONPATH kept: `StageContext.env` replaces a PYTHONPATH that already holds the
    repo `src` by `src` alone, which drops the optional `computerworld` wheel dir (`RRP_PEER_PYTHONPATH`) from every
    stage job (asked of the pipeline base owner; until then the pointer stages pass the inherited path on)."""
    e = ctx.env()
    if inherited := os.environ.get("PYTHONPATH"):
        e["PYTHONPATH"] = inherited
    return e


def _train(ctx: StageContext, cmd: str, out_name: str, extra: list[str], *, use_variant: bool = False) -> Path:
    out = ctx.out / out_name
    argv = [*TRAIN, cmd, "--data", *_data(ctx), "--out", str(out), "--seed", str(ctx.rc.seed), "--split", _split_path(ctx),
            *extra]
    if use_variant:
        argv += ["--variant", ctx.rc.variant]
    ctx.run(argv, env=_env(ctx), log_to=ctx.out / f"{cmd}.log")
    return out


def _outputs(out: Path, ctx: StageContext, *, meta: bool = True) -> dict:
    o = {out.name: str(out.relative_to(ctx.root))}
    if meta and out.with_suffix(".json").exists():
        o["meta"] = str(out.with_suffix(".json").relative_to(ctx.root))
    return o


# ------------------------------------------------------------------------------------------------ data
@register_stage("pointer", "collect", source="scripted_teacher")
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
        argv = [*TRAIN, "collect", "--task", t, "--episodes", str(n), "--seed", str(ctx.rc.seed), "--split", _split_path(ctx),
                *_flags(p, skip=("seed",)), "--out", str(out)]
        jobs.append((argv, _env(ctx), ctx.out / f"collect_{_slug(t)}.log"))
    ctx.run_parallel(jobs, int(ctx.opts.get("workers", 2)))
    return dict(outputs=outs, metrics=dict(tasks=len(outs), episodes_per_task=n))


# ------------------------------------------------------------------------------------------------ training
@register_stage("pointer", "train_rep", source="learned")
def train_rep(ctx: StageContext) -> dict:
    """E + R + P on demo chunks; variant semfix (probe loss on z, bounded NLL) | nosem (weights 0)."""
    if ctx.rc.variant not in ("semfix", "nosem"):
        raise StageError("pointer train_rep: variant must be semfix or nosem")
    out = _train(ctx, "rep", "rep.pt", _flags(ctx.rc.params), use_variant=True)
    return dict(outputs=_outputs(out, ctx), source_detail=str(out.relative_to(ctx.root)))


@register_stage("pointer", "train_flow", source="learned")
def train_flow(ctx: StageContext) -> dict:
    """System i (rectified flow): on the frozen representation's posterior means (target latent) or on the engineered
    encoding (target eng, realised by the SCRIPTED engineered system 0)."""
    target = ctx.rc.params.get("target", "latent")
    extra = _flags({**(V2_EVAL_DEFAULTS if target == "eng" else {}), **ctx.rc.params})
    if target == "latent":
        extra += ["--representation", ctx.inp("representation")]
    out = _train(ctx, "flow", "flow.pt", extra)
    return dict(outputs=_outputs(out, ctx), source_detail=str(out.relative_to(ctx.root)))


@register_stage("pointer", "train_bc", source="bc")
def train_bc(ctx: StageContext) -> dict:
    """BC baseline: context -> 7-tick chunk, the same public inputs and demos."""
    out = _train(ctx, "bc", "bc.pt", _flags({"key_head": V2_EVAL_DEFAULTS["key_head"], **ctx.rc.params}))
    return dict(outputs=_outputs(out, ctx), source_detail=str(out.relative_to(ctx.root)))


@register_stage("pointer", "probes", source="learned")
def probes(ctx: StageContext) -> dict:
    """Post-hoc packet probes + the metadata-only control (diagnostic); `inputs.flow` probes generated packets."""
    extra = _flags(ctx.rc.params, skip=("factors",)) + ["--representation", ctx.inp("representation")]
    if ctx.inp("flow", required=False):
        extra += ["--flow", ctx.inp("flow")]
    out = _train(ctx, "probe", "probe.json", extra)
    return dict(outputs=_outputs(out, ctx, meta=False))


@register_stage("pointer", "edits", source="learned")
def edits(ctx: StageContext) -> dict:
    """Causal packet edits: probe-guided retargeting of a received packet, realised by system 0 in closed loop."""
    extra = _flags(ctx.rc.params, skip=("factors",)) + ["--representation", ctx.inp("representation"), "--flow", ctx.inp("flow")]
    out = _train(ctx, "edit", "edit.json", extra)
    return dict(outputs=_outputs(out, ctx, meta=False))


# ------------------------------------------------------------------------------------------------ rollouts
def _seeds(ctx: StageContext, seed_set: str, task: str) -> str:
    s = _split(ctx)["seeds"][task].get(seed_set)
    if not s:
        raise StageError(f"{task}: split {_split_path(ctx)} has no seed set {seed_set!r}")
    if (n := ctx.opts.get("max_seeds")) is not None:          # smoke recipes: the first n declared dev seeds, never a sealed list
        if seed_set != "dev":
            raise StageError(f"options.max_seeds truncates only the dev seeds, not {seed_set!r}")
        s = s[:int(n)]
    return ",".join(str(x) for x in s)


# ------------------------------------------------------------------------------------------------ sealed guard
def _method_id(ctx: StageContext, kind: str) -> str:
    """The frozen method under test: the policy kind and the digest of its input checkpoints (a second lineage that
    evaluates the same checkpoints is the same method)."""
    paths = {k: v for k, v in sorted(ctx.rc.input_paths(ctx.index).items()) if k in ("flow", "representation", "checkpoint")}
    return f"{kind}:{json_digest(paths)}"


def _sealed_split(ctx: StageContext, split: dict, seed_set: str) -> SealedSplit:
    """The pinned `core.sealed` split of this stage's split file; a split `core.sealed` does not pin (the consumed v1, or an
    undeclared one) has no sealed evaluation."""
    sid = split["split_id"]
    if sid in CONSUMED:
        raise SealedSplitError(f"{sid}: {seed_set} is sealed and consumed by {CONSUMED[sid]}; a new comparison needs a new split "
                               "declared before any demo", code="sealed_split_consumed")
    if sid not in SPLITS:
        raise SealedSplitError(f"{sid}: {seed_set} cannot be evaluated: core.sealed pins no such split "
                               f"(pinned: {', '.join(SPLITS)})", code="sealed_split_unknown")
    return SealedSplit.load(sid, path=ctx.root / _split_path(ctx))


def _sealed_cells(ctx: StageContext, sp: SealedSplit, split: dict, seed_set: str, tasks: list[str], kind: str) -> list[dict]:
    """One `sealed_eval` cell per (task, method): body cw_pointer, this stage's train seed, the split's declared seed list."""
    return [dict(body="cw_pointer", method=_method_id(ctx, kind), train_seed=ctx.rc.seed, task=t, seed_set=seed_set,
                 scenes=[int(x) for x in split["seeds"][t][seed_set]]) for t in tasks]


@contextlib.contextmanager
def _sealed_run(ctx: StageContext, sp: SealedSplit, cells: list[dict]):
    """Start every cell (run-once log, `SealedSplit.sealed_eval`) or none: all cells are checked before the first start, so a
    refused cell never leaves the others open. `done` is written after a clean exit of the block."""
    log = ctx.root / SPLITS[sp.name]["log"]
    for c in cells:
        sp.check_cell(c)
        cid = sp.cell_id(c)
        if (st := SealedSplit.cell_state(cid, log)) not in ("none", "infrastructure_failure"):
            raise SealedSplitError(f"sealed cell {cid} is '{st}' in {log}: it already ran (a rerun needs a recorded "
                                   "infrastructure failure: `rrp suite sealed-log infra-failure`)", code="sealed_cell_rerun")
    with contextlib.ExitStack() as stack:
        for c in cells:
            stack.enter_context(sp.sealed_eval(c, log))
        yield [sp.cell_id(c) for c in cells]


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
    pol = _policy(ctx, kind)
    split = _split(ctx)
    outs, metrics = {}, {}
    for seed_set in _seed_sets(ctx):
        sealed = seed_set in SEAL_SETS
        if not sealed and seed_set != "dev":
            raise StageError(f"seed set {seed_set!r}: dev | {' | '.join(SEAL_SETS)}")
        sp = _sealed_split(ctx, split, seed_set) if sealed else None
        jobs, todo = [], []
        for t in _tasks(ctx):
            if seed_set not in split["seeds"][t]:
                continue                       # e.g. no held-out variants for cw/drag_window
            out = ctx.out / f"{_slug(t)}_{seed_set}.jsonl"
            outs[f"{_slug(t)}_{seed_set}"] = str(out.relative_to(ctx.root))
            if out.exists() and o.get("resume", True):
                continue
            argv = ["-m", "rrp.cli", "eval", "--policy", pol, "--env", "computerworld", "--task", t, "--body", "cw_pointer",
                    "--seeds", _seeds(ctx, seed_set, t), "--batch", str(o.get("batch", 16)), "--out", str(out)]
            for k, v in sorted(split.get("env_kw", {}).items()):     # the split's env kwargs (v2: strings=procedural)
                argv += ["--env-kw", f"{k}={v if isinstance(v, str) else json.dumps(v)}"]
            jobs.append((argv, _env(ctx), ctx.out / f"eval_{_slug(t)}_{seed_set}.log"))
            todo.append(t)
        if sealed:                             # a raised exception in the block leaves the attempts open on purpose
            with _sealed_run(ctx, sp, _sealed_cells(ctx, sp, split, seed_set, todo, kind)):
                ctx.run_parallel(jobs, int(o.get("workers", 1)))
        else:
            ctx.run_parallel(jobs, int(o.get("workers", 1)))
    for name, rel in outs.items():
        summ = ctx.root / Path(rel).with_suffix(".summary.json")
        if summ.exists():
            s = json.loads(summ.read_text())
            metrics[name] = f"{s.get('successes')}/{s.get('attempted')}"
    return dict(outputs=outs, metrics=metrics)


@register_stage("pointer", "eval_r1", source="oracle")
def eval_r1(ctx: StageContext) -> dict:
    """R1 rung (ORACLE, diagnostic): teacher chunk -> encoder -> system 0. With inputs.representation the system 0 is the
    LEARNED realizer, without it the SCRIPTED engineered one."""
    return _evaluate(ctx, "pointer_oracle")


@register_stage("pointer", "eval_r2", source="learned")
def eval_r2(ctx: StageContext) -> dict:
    """Deployable route on the split's seeds: options.policy pointer_latent (default) | pointer_bc."""
    kind = ctx.opts.get("policy", "pointer_latent")
    res = _evaluate(ctx, kind)
    res["source_detail"] = ctx.inp("checkpoint" if kind == "pointer_bc" else "flow")
    if kind == "pointer_bc":
        res["source"] = "bc"
    return res
