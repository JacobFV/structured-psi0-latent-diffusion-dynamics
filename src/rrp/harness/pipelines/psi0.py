"""Ψ₀ / SIMPLE stages (track psi0, D-140; D-145 P4c). Each stage calls existing code (`rrp.policies.psi0.{data,train}`).

| stage | code |
|---|---|
| collect (`options.data: features`, the default) | `rrp data psi0-features`: frozen-VLM feature cache of a SIMPLE task's training data -> `feat/` (peer GPU) |
| collect (`options.data: labels`) | `rrp data psi0-labels`: probe labels, `psi0_replay` x `simple` (MuJoCo only) through `harness.rollout` + `LabelRecorder` -> `labels/` (SOURCE `privileged_teacher:sim_replay`; labels only, never a policy input) |
| train_rep | `rrp train psi0 --arm stageA`: Stage A (E, R over the 36-d chunk) -> `stage_a.pt`, `z_stats.pt`, `summary.json` |
| train_bc | `rrp train psi0 --arm direct`: the matched DIRECT arm (Ψ₀ action head fine-tuned on the cached features) -> `final.pt` |
| train_flow | `rrp train psi0 --arm structured`: system i samples the packet z[5, 6, 64], the frozen Stage A realises it -> `final.pt`; refuses to start (`GateFailed`) unless its `gate` input (the gate node's `heldout.json`) passes |
| probes | `rrp train psi0 probes`: packet probes + metadata-only control on the training run's train/val split -> `probes.json` |
| heldout | `rrp train psi0 heldout`: held-out open-loop L1 per action group, incl. the DIAGNOSTIC oracle route R(E(chunk)) -> `heldout.json`. Run WITHOUT a `structured` input it is the pre-head GATE (architecture 14.5): `heldout.json["gate"] = {gap, margin}` must hold gap >= margin, else the node fails (`GateFailed`) |
| eval_r2 | closed loop `rrp eval --policy psi0_direct|psi0_structured --env simple` (Isaac Sim, `simple` venv, peer) |

Options: `task` (SIMPLE task name, e.g. G1WholebodyTabletopGraspMP-v0; selects the released run for the data transform), `arm`
(eval_r2: direct | structured | released), `seeds`, `data` (collect: features | labels), `episodes` (collect labels: A:B or a comma list; default every cached episode),
`data_root` (collect labels), `workers`. Inputs: `features` (collect), `features` (collect labels: the cache whose episodes are labelled), `labels` (consumed by the fits and probes), `gate`
(train_flow). Params pass through as `--<name>` flags (steps, batch, lr, w_sem, w_kl, lv_min, z_noise, ...).
`RRP_PSI0_EXT` / `PSI_HOME` locate the third-party stack (one resolver: `rrp.envs.simple.compat.ext_dir`).
Labelled sources: the released checkpoint is UPSTREAM (`learned:psi0-released/...`, not trained by us); every other checkpoint is
ours (`learned:<final.pt>`), and the oracle route in `heldout` is a diagnostic that reads the target actions.
"""
from __future__ import annotations

import json
from pathlib import Path

from rrp.harness.pipelines.base import StageContext, StageError, apply_gate, register

TRAIN = ["-m", "rrp.cli", "train", "psi0"]


def _task(ctx: StageContext) -> str:
    t = ctx.opts.get("task")
    if not t:
        raise StageError("psi0: options.task (a SIMPLE task name) is required")
    return t.split("/", 1)[-1]


def _run_dir(ctx: StageContext) -> str:
    from rrp.policies.psi0 import released_run
    return str(released_run(_task(ctx)))


def read_gate(path: Path) -> dict:
    """The heldout gate of a `heldout.json` (architecture 14.5): {gap, margin}. A file without it is an error, never a pass."""
    g = json.loads(Path(path).read_text()).get("gate")
    if not isinstance(g, dict) or not {"gap", "margin"} <= g.keys():
        raise StageError(f"psi0: {path} has no gate block with gap and margin (rrp train psi0 heldout writes it)")
    return g


def gate_report(g: dict, subject: str) -> dict:
    """W6-style report of the heldout gate: pass iff gap >= margin (the packet must matter: err(R(z_perm-or-mean)) beats
    the demonstrated-chunk route by at least the margin)."""
    from rrp.harness.eval.gates import _crit, _report
    gap, m = float(g["gap"]), float(g["margin"])
    return _report("psi0_heldout_gate", subject, [_crit("heldout_gap_min", gap, m, gap >= m,
                                                         note="err gap of the packet route; the structured head trains only above the margin")])


def _flags(params: dict, skip=()) -> list[str]:
    argv: list[str] = []
    for k, v in params.items():
        if k in skip or v is None:
            continue
        if v is True:
            argv.append(f"--{k.replace('_', '-')}")
        elif v is not False:
            argv += [f"--{k.replace('_', '-')}", str(v)]
    return argv


def _fit(ctx: StageContext, arm: str, *, extra=(), outputs: tuple[str, ...]) -> dict:
    out = ctx.out
    argv = [*TRAIN, "--arm", arm, "--feat-dir", ctx.inp("features"), "--run-dir", _run_dir(ctx), "--out", str(out),
            "--seed", str(ctx.rc.seed), *extra, *_flags(ctx.rc.params, skip=("seed",))]
    labels = ctx.inp("labels", required=arm != "stageA")
    if labels:
        argv += ["--labels-dir", labels]
    ctx.run(argv, log_to=out / "train.log")
    outs = {n: str((out / n).relative_to(ctx.root)) for n in outputs}
    summ = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    return dict(outputs=outs, metrics={k: summ[k] for k in ("steps", "val_loss", "gpu_seconds") if k in summ},
                source_detail=outs.get("final.pt") or outs.get("stage_a.pt"))


def _collect_features(ctx: StageContext) -> dict:
    """Frozen-VLM feature cache (Qwen3-VL hidden states of the task's training frames); no training happens."""
    o = ctx.opts
    from rrp.policies.psi0 import base_vlm
    out = ctx.out / "feat"
    argv = ["-m", "rrp.cli", "data", "psi0-features", "--vlm", str(Path(o["vlm"]).expanduser() if o.get("vlm") else base_vlm()), "--run-dir", _run_dir(ctx),
            "--repo-id", str(o.get("repo_id", _task(ctx))), "--out", str(out),
            *_flags(ctx.rc.params)]
    ctx.run(argv, log_to=ctx.out / "features.log")
    return dict(outputs={"feat": str(out.relative_to(ctx.root))}, metrics={})


def _collect_labels(ctx: StageContext) -> dict:
    """Probe labels of the training episodes: recorded rows replayed through harness.rollout on `simple`, no learned policy."""
    o = ctx.opts
    eps = o.get("episodes")
    if eps is None:
        eps = f"0:{json.loads((ctx.root / ctx.inp('features') / 'meta.json').read_text())['episodes']}"
    out = ctx.out / "labels"
    argv = ["-m", "rrp.cli", "data", "psi0-labels", "--task", _task(ctx), "--out", str(out), "--episodes", str(eps),
            "--batch", str(o.get("batch", 1))]
    if o.get("data_root"):
        argv += ["--data-root", str(Path(o["data_root"]).expanduser())]
    ctx.run(argv, log_to=ctx.out / "labels.log")
    return dict(outputs={"labels": str(out.relative_to(ctx.root))}, metrics={},
                source="privileged_teacher:sim_replay")


@register("psi0", "collect", source="learned")
def collect(ctx: StageContext) -> dict:
    """Training data of the task: `options.data` features (frozen-VLM feature cache; default) or labels (probe labels)."""
    kind = ctx.opts.get("data", "features")
    if kind not in ("features", "labels"):
        raise StageError(f"psi0 collect: options.data must be features or labels, got {kind!r}")
    return (_collect_features if kind == "features" else _collect_labels)(ctx)


@register("psi0", "train_rep", source="learned")
def train_rep(ctx: StageContext) -> dict:
    """Stage A: E, R and the packet statistics; the structured arm freezes it."""
    return _fit(ctx, "stageA", outputs=("stage_a.pt", "z_stats.pt", "summary.json"))


@register("psi0", "train_bc", source="learned")
def train_bc(ctx: StageContext) -> dict:
    """Direct arm: same features, same split, same steps as the structured arm (matched control)."""
    return _fit(ctx, "direct", outputs=("final.pt", "summary.json"))


@register("psi0", "train_flow", source="learned")
def train_flow(ctx: StageContext) -> dict:
    """Structured arm: packet z from system i, realised by the frozen Stage A. Blocked unless the heldout gate passed."""
    apply_gate(ctx, gate_report(read_gate(ctx.root / ctx.inp("gate")), ctx.inp("gate")))
    return _fit(ctx, "structured", extra=["--stage-a", ctx.inp("stage_a")], outputs=("final.pt", "summary.json"))


def _diag(ctx: StageContext, cmd: str, name: str, extra: list[str]) -> dict:
    out = ctx.out / name
    ctx.run([*TRAIN, cmd, "--feat-dir", ctx.inp("features"), "--out", str(out), *extra], log_to=ctx.out / f"{cmd}.log")
    return dict(outputs={name: str(out.relative_to(ctx.root))}, metrics={})


@register("psi0", "probes", source="learned")
def probes(ctx: StageContext) -> dict:
    """Packet probes on E means / generated packets and the metadata-only control (diagnostic)."""
    extra = ["--stage-a", ctx.inp("stage_a"), "--train-summary", ctx.inp("summary"), "--labels-dir", ctx.inp("labels"),
             *_flags(ctx.rc.params)]
    if ctx.inp("structured", required=False):
        extra += ["--structured", ctx.inp("structured"), "--run-dir", _run_dir(ctx)]
    return _diag(ctx, "probes", "probes.json", extra)


@register("psi0", "heldout", source="oracle")
def heldout(ctx: StageContext) -> dict:
    """Held-out open-loop L1 per action group; the R(E(chunk)) route is an ORACLE diagnostic. With no `structured` input this is
    the pre-head gate: heldout.json must carry gate {gap, margin} and gap >= margin (GateFailed otherwise)."""
    summ = json.loads((ctx.root / ctx.inp("summary")).read_text())
    extra = ["--run-dir", _run_dir(ctx), "--val-episodes", ",".join(str(e) for e in summ["val_eps"]), *_flags(ctx.rc.params)]
    for k, flag in (("direct", "--direct"), ("structured", "--structured"), ("stage_a", "--stage-a")):
        if ctx.inp(k, required=False):
            extra += [flag, ctx.inp(k)]
    res = _diag(ctx, "heldout", "heldout.json", extra)
    if ctx.inp("structured", required=False) is None:
        out = ctx.out / "heldout.json"
        res["metrics"] = apply_gate(ctx, gate_report(read_gate(out), str(out.relative_to(ctx.root))))
    return res


@register("psi0", "eval_r2", source="learned")
def eval_r2(ctx: StageContext) -> dict:
    """Closed loop on the SIMPLE task (Isaac Sim): options.arm released (upstream checkpoint) | direct | structured."""
    o = ctx.opts
    arm = o.get("arm", "structured")
    kw: dict = dict(task=f"simple/{_task(ctx)}")
    if arm == "released":
        kind = "psi0_direct"
    else:
        kind = f"psi0_{arm}"
        kw["weights"] = ctx.inp("checkpoint")
        if arm == "structured":
            kw["stage_a"] = ctx.inp("stage_a")
    for k in ("rtc", "nfe", "vlm"):
        if k in o:
            kw[k] = o[k]
    out = ctx.out / f"{arm}.jsonl"
    argv = ["-m", "rrp.cli", "eval", "--policy", f"{kind}={json.dumps(kw)}", "--env", "simple", "--task", f"simple/{_task(ctx)}",
            "--body", "g1_simple", "--seeds", str(o.get("seeds", "0:10")), "--batch", str(o.get("batch", 1)), "--out", str(out)]
    ctx.run(argv, log_to=ctx.out / "eval.log")
    metrics = {}
    sp = out.with_suffix(".summary.json")
    if sp.exists():
        s = json.loads(sp.read_text())
        metrics = dict(success=f"{s.get('successes')}/{s.get('attempted')}")
    return dict(outputs=dict(rows=str(out.relative_to(ctx.root)), summary=str(sp.relative_to(ctx.root))), metrics=metrics,
                source_detail=f"psi0-{arm}")
