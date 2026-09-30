"""Arm (single-manipulator pick_place) stages. Each calls the code the chain scripts ran:

| stage | existing code (what the chain scripts ran) |
|---|---|
| collect | rrp.data.generate.generate (`rrp data generate`) |
| pack | rrp.data.packed.pack_dataset (`rrp data pack`) |
| train_rep | rrp.training.latent_train.train_representation (`rrp latent train-representation`) |
| probes | rrp.training.latent_train.fit_probes_on_frozen (`rrp latent fit-probes`) |
| train_flow, flow_ft | rrp.training.latent_train.train_latent_flow (`rrp latent train-flow`) |
| refit | rrp.training.latent_train.refit_realizer  |
| dagger_collect | `rrp suite ladder --collect-dagger` (EXPERT=bc [FLOW=] [GENCTX=]) |
| eval_r1 | `rrp suite ladder --route oracle --oracle-expert bc` (ORACLE DIAGNOSTIC) |
| eval_r2, heldout | `rrp suite ladder --route generated` |
| edits | `rrp latent semantic-edits --route generated` (chain semedit) |
| grpo (D-126 #6) | rrp.training.latent_grpo.train_latent_grpo (latent) / rrp.training.adapt.run method grpo (BC), + anchors |
| target_eval (D-126 #9/#10) | python -m rrp.cli suite target (sealed protocol scenes; ladder routes generated / learned) |
| target_adapt (D-126 #9/#10) | sft_latent_flow (flow_sft) / refit_realizer + episode_budget (system0_refit) / sft_packed (bc_sft) |
| train_bc (D-126 #10) | rrp.training.behavior.train_policy with baseline_campaign.source_config (direct-action BC source) |

The evaluation stages run the ladder (rrp.harness.eval.ladder_cli) as a subprocess `rrp suite ladder ...`. Outputs go to
the stage's own out dir, never to the shared artifacts/runs/ladder_v1/.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

from rrp.harness.pipelines.base import apply_gate, StageContext, StageError, register

LADDER = ["-m", "rrp.cli", "suite", "ladder"]
TRAIN_BODIES = ("panda_pg2", "parm5_pg2", "parm5_tf3", "parm5l_tf3", "parm5s_pg2", "parm6_pg2", "parm6_tf3",
                "parm7_pg2", "parm7_tf3", "sawyer_pg2", "sawyer_tf3", "ur5e_pg2", "ur5e_tf3")
TARGET_BODIES = ("xarm7_pg2", "xarm7_tf3", "panda_tf3")      # never in the ladder (the ladder refuses them)
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
    man_p = ctx.root / cfg["out_dir"] / "manifest.json"
    from rrp.harness.eval.gates import check_arm_dataset
    man = json.loads(man_p.read_text())
    gate = apply_gate(ctx, check_arm_dataset(man.get("episodes") or [], man))        # W6 dataset gate (D-112)
    return dict(outputs={"manifest": str(Path(cfg["out_dir"]) / "manifest.json")}, metrics=dict(gate=gate))


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
    from rrp.harness.train.latent_train import train_representation
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
    from rrp.harness.train.latent_train import fit_probes_on_frozen
    o = ctx.opts
    out = ctx.out / o.get("out_name", "probes.json")
    # D-144 sweep-flags: the `options.binding_cf` fallback is gone (confirmed by grep: no dag/config sets either
    # key for this stage's options; `cf_mix` is the only live options key now, matching cli/latent.py's --cf-mix).
    res = fit_probes_on_frozen(Path(ctx.inp("representation")), Path(ctx.inp("packed_dir")), out,
                               steps=int(o.get("steps", 6000)), metadata_only=bool(o.get("metadata_only", False)),
                               cf_mix=float(o.get("cf_mix", 0.0)))
    return dict(outputs={}, metrics=_json_safe(res), source_detail=ctx.inp("representation"))


def _flow(ctx: StageContext) -> dict:
    from rrp.harness.train.latent_train import train_latent_flow
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
    """System-0 refit on the frozen encoder from DAgger buffers ."""
    from rrp.harness.train.latent_train import refit_realizer
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
    from rrp.bodies.armdiv import is_armdiv_sealed
    bad = [r for r in rs if r in TARGET_BODIES or is_armdiv_sealed(r)]
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
        argv = [*LADDER, *route, "--oracle-expert", "bc", "--policy", bc, "--policy-label", bcl,
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
    """Eval rows keep their legacy free-string source; the manifest records the canonical Source enum per row so
    reports can group by it. Reads both formats (contracts.provenance.row_source: sl-1 `source_label` first, else
    the legacy `source`, including the ladder's "learned(system-i flow)" / "target_encoder_oracle(...)" forms)."""
    from rrp.core.provenance import row_source
    c = Counter()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if line.strip():
                try:
                    c[str(row_source(json.loads(line)).kind.value)] += 1
                except (ValueError, TypeError, AttributeError):
                    c["unparsed"] += 1
    return dict(c)


def _blend_args(ctx: StageContext) -> list[str]:
    """D-126 #7: options.chunk_blend (none|crossfade|ensemble), blend_ticks, blend_decay -> ladder CLI; [] when unset."""
    o = ctx.opts
    mode = o.get("chunk_blend", "none")
    if mode == "none":
        return []
    return ["--chunk-blend", mode, "--blend-ticks", str(int(o.get("blend_ticks", 4))),
            "--blend-decay", str(float(o.get("blend_decay", 0.0)))]


def _ladder_eval(ctx: StageContext, route: str, robots: list[str], seed_starts: list[int], tag_fn, extra: list[str]):
    o = ctx.opts
    extra = list(extra) + _blend_args(ctx)
    n = int(o.get("episodes", 30))
    jobs, results = [], []
    for s in seed_starts:
        for r in robots:
            tag = tag_fn(s)
            out = _rel(ctx, r)
            argv = [*LADDER, "--route", route, *extra, "--prev-action", _prev_action(ctx), "--robot", r, "--n", str(n),
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
    tag = o["tag"]
    tag_fn = lambda s: f"zero_{tag}_s{s}" if _prev_action(ctx) == "zero" else f"{tag}_s{s}"
    nfe = ["--nfe", str(o["nfe"])] if "nfe" in o else []
    if o.get("route", "generated") == "learned":      # D-126 #6/#10: direct-action BC on the same sets and harness
        bc = ctx.inp("bc_policy")
        bcl = o.get("bc_label") or Path(bc).stem
        outputs, metrics = _ladder_eval(ctx, "learned", robots, [int(s) for s in o.get("seed_starts", [3000000])],
                                        tag_fn, ["--policy", bc, "--policy-label", bcl] + nfe)
        return dict(outputs=outputs, metrics=metrics, source="bc", source_detail=bcl)
    flow, rep = ctx.inp("flow"), ctx.inp("representation")
    outputs, metrics = _ladder_eval(ctx, "generated", robots, [int(s) for s in o.get("seed_starts", [3000000])],
                                    tag_fn, ["--flow", flow, "--rep", rep] + nfe)
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


# ------------------------------------------------------------------------------------------------ D-126 stages
PROTOCOL = "configs/eval/latent_slice1.json"


def _protocol(ctx: StageContext) -> tuple[dict, str]:
    from rrp.harness.eval.target_eval import load_protocol
    path = ctx.opts.get("protocol", PROTOCOL)
    proto, sha = load_protocol(ctx.root / path)
    return proto, sha


def _anchor(ctx: StageContext) -> dict | None:
    a = ctx.opts.get("anchor")
    if not a:
        return None
    a = dict(a)
    a.setdefault("prev_action", _prev_action(ctx))
    if a["prev_action"] != _prev_action(ctx):
        raise StageError("anchor prev_action must equal the flag zero_prev_action (deployment input)")
    from rrp.harness.train.grpo_anchor import AnchorConfig
    AnchorConfig.from_dict(a)                          # validate early (targets refused, dev seeds, action)
    return a


@register("arm", "grpo", source="learned")
def grpo(ctx: StageContext) -> dict:
    """D-126 #6: GRPO fine-tuning with anchor / forgetting evaluations. options.method latent (system i on the packet,
    system 0 = inputs.representation, frozen) or bc (direct-action BC through rrp.training.adapt, the matched-budget
    baseline). Budget = new control transitions (params.budget_env_steps for latent; params.budgets for bc).
    options.anchor = rrp.training.grpo_anchor.AnchorConfig fields (robots, seed_start, episodes, max_drop, action)."""
    import copy
    from dataclasses import asdict
    o = ctx.opts
    method = o.get("method", "latent")
    anchor = _anchor(ctx)
    p = copy.deepcopy(dict(ctx.rc.params))
    out = ctx.rc.out
    if method == "latent":
        from rrp.harness.train.flow_sde import SDEConfig
        from rrp.harness.train.grpo import GRPOConfig
        from rrp.harness.train.latent_grpo import LatentGRPORunConfig, RewardConfig, train_latent_grpo
        from rrp.core.runconfig import overlay
        dflt = asdict(LatentGRPORunConfig(checkpoint="", robot="", out_dir="").grpo)
        g = overlay(dflt, p.pop("grpo", {}) or {})
        sde = SDEConfig(**g.pop("sde"))
        if p.get("budget_env_steps") is None:
            raise StageError("latent grpo stage: params.budget_env_steps is required (matched-budget comparison)")
        cfg = LatentGRPORunConfig(checkpoint=ctx.inp("flow"), out_dir=out, representation=ctx.inp("representation"),
                                  anchor=anchor, snapshot_evals=bool(p.pop("snapshot_evals", True)),
                                  reward=RewardConfig(**(p.pop("reward", {}) or {})), grpo=GRPOConfig(sde=sde, **g), **p)
        res = train_latent_grpo(cfg)
        outs = {"policy": str(Path(out) / "policy.pt"), "result": str(Path(out) / "result.json")}
        src = f"{ctx.inp('flow')}+latent_grpo"
    elif method == "bc":
        from rrp.harness.train.adapt import run
        if not p.get("budgets"):
            raise StageError("bc grpo stage: params.budgets is required (matched-budget comparison)")
        cfg = dict(p, method="grpo", checkpoint=ctx.inp("bc_policy"), out_dir=out, name=ctx.rc.run_id.replace("/", "_"))
        if anchor:
            cfg["anchor"] = anchor
        res = run(cfg)
        b = max(cfg["budgets"])
        outs = {"result": str(Path(out) / "result.json")}
        if (ctx.root / out / f"policy_b{b}.pt").exists():
            outs["policy"] = str(Path(out) / f"policy_b{b}.pt")
        src = f"{ctx.inp('bc_policy')}+bc_grpo"
    else:
        raise StageError(f"grpo method {method!r} (latent|bc)")
    if anchor:
        outs["anchor_report"] = str(Path(out) / "anchor_report.json")
    metrics = dict(method=method, anchor=res.get("anchor"), budget=res.get("budget"),
                   accounting=res.get("accounting") or res.get("counters"),
                   evals=[{k: e.get(k) for k in ("tag", "successes", "episodes", "attempted", "rate", "success_rate",
                                                  "budget")} for e in res.get("evals", [])])
    return dict(outputs=outs, metrics=_json_safe(metrics), source_detail=src)


@register("arm", "target_eval", source="learned")
def target_eval(ctx: StageContext) -> dict:
    """D-126 #9/#10: sealed-protocol evaluation (python -m rrp.cli suite target) of the latent route (route
    generated: inputs flow + representation) or direct-action BC (route learned: inputs bc_policy) on options.robot.
    Target bodies need options.sealed_run: true; options.smoke: true runs a <= 3-episode plumbing check on a NON-target
    options.robot with dev seeds (labelled smoke)."""
    o = ctx.opts
    proto, sha = _protocol(ctx)
    robot, route = o["robot"], o.get("route", "generated")
    kind = o.get("kind", "zero_shot")
    tag = o.get("tag", kind.replace(":", "_"))
    argv = ["-m", "rrp.cli", "suite", "target", "--protocol", o.get("protocol", PROTOCOL), "--robot", robot,
            "--route", route, "--prev-action", _prev_action(ctx), "--kind", kind, "--tag", tag, "--out", ctx.rc.out]
    if route == "generated":
        argv += ["--flow", ctx.inp("flow"), "--rep", ctx.inp("representation")]
        src, detail = "learned", ctx.inp("flow")
    elif route == "learned":
        bc = ctx.inp("bc_policy")
        argv += ["--policy", bc, "--policy-label", o.get("bc_label") or Path(bc).stem]
        src, detail = "bc", o.get("bc_label") or Path(bc).stem
    else:
        raise StageError(f"target_eval route {route!r} (generated|learned)")
    if o.get("smoke"):
        argv += ["--smoke", "--smoke-episodes", str(int(o.get("smoke_episodes", 2)))]
    elif o.get("sealed_run") is True:
        argv += ["--sealed-run"]
    argv += _blend_args(ctx)
    ctx.run(argv, env=_threads_env(ctx, o.get("threads")), log_to=ctx.out / f"{robot}_{tag}.log")
    sp = Path(ctx.rc.out) / f"{route}_{tag}.summary.json"
    summ = json.loads((ctx.root / sp).read_text())
    metrics = {k: summ.get(k) for k in ("n", "success", "rate", "wilson95", "failed_stage", "scene_kind", "smoke", "kind",
                                        "seeds", "chunk_vel_step_max_mean", "chunk_vel_step_flag_frac")}
    metrics["protocol"] = dict(id=proto.get("id"), sha256=sha)
    return dict(outputs={"summary": str(sp), "rows": str(Path(ctx.rc.out) / f"{route}_{tag}.jsonl")}, metrics=metrics,
                source=src, source_detail=detail)


SFT_STEPS = {5: 150, 20: 300, 100: 600}      # = rrp.training.baseline_campaign.SFT_STEPS / run_latent_cell
SFT_LR = 1e-4


@register("arm", "target_adapt", source="learned")
def target_adapt(ctx: StageContext) -> dict:
    """D-126 #9/#10: few-shot adaptation on target demos with the sealed protocol's budgets and acquisition (nested
    episode choice by adapt_seed, update counts SFT_STEPS[budget], lr 1e-4): options.method flow_sft (system i only;
    inputs flow, packed_dir) | system0_refit (system 0 only; inputs representation, packed_dir) | bc_sft (direct-action
    BC; inputs bc_policy, packed_dir). packed_dir = the target body's stride-1 demo pack."""
    o = ctx.opts
    proto, sha = _protocol(ctx)
    target, budget, method = o["target"], int(o["budget"]), o["method"]
    seed = int(o.get("adapt_seed", ctx.rc.seed))
    smoke = bool(o.get("smoke"))
    if target not in proto["targets"]:
        raise StageError(f"{target} is not a protocol target")
    if not smoke and (budget not in proto["sft_budgets"] or budget <= 0):
        raise StageError(f"budget {budget} not in the protocol's sft_budgets {proto['sft_budgets']} (> 0)")
    if not smoke and seed not in proto["seeds"]:
        raise StageError(f"adapt_seed {seed} not in the protocol seeds {proto['seeds']}")
    steps = int(o.get("steps", SFT_STEPS.get(budget, 0))) if smoke else SFT_STEPS[budget]
    lr = SFT_LR
    pack = Path(ctx.inp("packed_dir"))
    meta_p = ctx.root / pack / "meta.json"
    if meta_p.exists():
        mj = json.loads(meta_p.read_text())
        robots = set(mj.get("robots") or mj.get("train_robots") or [])
        if robots and robots != {target}:
            raise StageError(f"packed_dir holds {sorted(robots)}, not only {target}")
    out = ctx.out
    if method == "flow_sft":
        from rrp.harness.train.latent_train import sft_latent_flow
        res = sft_latent_flow(Path(ctx.inp("flow")), pack, budget, seed=seed, out_dir=out, steps=steps, lr=lr)
        outs, detail = {"policy": str(Path(ctx.rc.out) / "policy.pt")}, str(Path(ctx.rc.out) / "policy.pt")
    elif method == "bc_sft":
        from rrp.harness.train.sft import sft_packed
        res = sft_packed(Path(ctx.inp("bc_policy")), pack, budget, seed=seed, out_dir=out, steps=steps, lr=lr)
        outs, detail = {"policy": str(Path(ctx.rc.out) / "policy.pt")}, str(Path(ctx.rc.out) / "policy.pt")
    elif method == "system0_refit":
        from rrp.policies.nets.checkpoint import load_checkpoint
        from rrp.harness.train.latent_train import refit_realizer
        rep = ctx.inp("representation")
        src_cfg = load_checkpoint(ctx.root / rep, map_location="cpu")["config"]
        cfg = dict(representation=rep, packed_dir=str(pack), steps=steps, batch_size=int(o.get("batch_size", 128)),
                   lr=lr, seed=seed, init="old", episode_budget=budget, budget_seed=seed, prefetch_workers=0,
                   zero_prev_action=bool(src_cfg.get("zero_prev_action", False)),
                   realizer_anchor=bool(src_cfg.get("realizer_anchor", False)),
                   realizer_drop_qd=bool(src_cfg.get("realizer_drop_qd", False)),
                   name=ctx.rc.run_id.replace("/", "_"))
        (out / "refit_config.json").write_text(json.dumps(cfg, indent=1))
        res = refit_realizer(cfg, out)
        if res.get("interrupted"):
            raise StageError("system-0 adaptation interrupted; rerun resumes from rz_last.pt")
        outs = {"representation": str(Path(ctx.rc.out) / "representation.pt")}
        detail = outs["representation"]
    elif method == "joint_adapt":           # D-136 (added AFTER D-135): flow + system 0 at the SAME total update count
        from rrp.harness.train.joint_adapt import joint_adapt
        res = joint_adapt(Path(ctx.inp("flow")), Path(ctx.inp("representation")), pack, budget, seed=seed, out_dir=out,
                          steps=steps, lr=lr, mode=o.get("joint_mode", "joint"), gen_frac=float(o.get("gen_frac", 0.0)))
        outs = {"policy": str(Path(ctx.rc.out) / "policy.pt"),
                "representation": str(Path(ctx.rc.out) / "representation.pt")}
        detail = outs["policy"]
    else:
        raise StageError(f"target_adapt method {method!r} (flow_sft|system0_refit|bc_sft|joint_adapt)")
    metrics = dict(method=method, target=target, budget=budget, adapt_seed=seed, steps=steps, lr=lr, smoke=smoke,
                   protocol=dict(id=proto.get("id"), sha256=sha), result=_json_safe(res))
    return dict(outputs=outs, metrics=metrics, source="bc" if method == "bc_sft" else "learned", source_detail=detail,
                versions=_versions(res) if isinstance(res, dict) else {})


@register("arm", "train_bc", source="bc")
def train_bc(ctx: StageContext) -> dict:
    """D-126 #10: direct-action BC source training (the stateless BC expert recipe: baseline_campaign.source_config =
    policy-small-structured, stride-2 rows of the stride-1 pack, zero_prev_action) on inputs.packed_dir, seed = rc.seed;
    params overlay the recipe (recorded). options.snapshot_steps keeps extra snapshots (e.g. [12000])."""
    from rrp.core.runconfig import overlay
    from rrp.harness.train.baseline_campaign import source_config
    from rrp.harness.train.behavior import train_policy
    o = ctx.opts
    method = o.get("method", "baseline_direct_action")
    if method != "baseline_direct_action":
        raise StageError("train_bc implements baseline_direct_action (the codec baseline needs its codec stage)")
    cfg = source_config(method, ctx.rc.seed, ctx.out, source_pack=ctx.inp("packed_dir"),
                        snapshot_steps=o.get("snapshot_steps"))
    cfg = overlay(cfg, dict(ctx.rc.params))
    cfg.update(out_dir=ctx.rc.out, zero_prev_action=bool(ctx.rc.flags.zero_prev_action))
    (ctx.out / "config.json").write_text(json.dumps(cfg, indent=1))
    res = train_policy(cfg, ctx.out)
    if res.get("interrupted"):
        raise StageError("BC training interrupted (checkpointed; rerun resumes)")
    outs = {"policy": str(Path(ctx.rc.out) / "policy.pt")}
    for st in o.get("snapshot_steps") or []:
        sp = Path(ctx.rc.out) / f"policy_u{int(st)}.pt"
        if (ctx.root / sp).exists():
            outs[f"policy_u{int(st)}"] = str(sp)
    return dict(outputs=outs, metrics=_json_safe({k: res.get(k) for k in ("steps", "train_chunks", "wall_s")}),
                source_detail=f"bc_direct{ctx.rc.seed}")
