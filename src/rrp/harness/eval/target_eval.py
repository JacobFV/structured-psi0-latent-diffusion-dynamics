"""Sealed-protocol evaluation on the target / held-out source bodies with the ladder's deployable routes (D-126 #9, #10).

`python -m rrp.cli suite target --protocol recipes/presets/eval-latent_slice1.json --robot xarm7_pg2 --route generated
     --flow F --rep R --sealed-run --out DIR --tag T`

The same machinery as the lineage R2 evaluations (rrp.harness.eval.ladder.run_ladder: route `generated` = system-i flow
-> deployed system 0; route `learned` = direct-action BC chunks), so latent routes and BC baselines are compared with
identical inputs, replan period, NFE, horizon and evaluator. Scenes follow the sealed protocol (latent_slice1):
eval seeds range(eval.seed_start, eval.seed_start + eval.episodes), infeasible scenes excluded and counted (the
baseline_campaign convention), max_steps / replan_ticks / nfe from the protocol; they cannot be overridden.

Access rules (protocol dev_rule "targets / eval scenes untouched until the sealed runs"):
  * target bodies and the protocol's held-out source bodies on eval seeds need --sealed-run (an explicit, recorded
    acknowledgement that this is a sealed run);
  * --smoke runs plumbing checks ONLY on a non-target body with dev seeds (>= 3,000,000) and <= 3 episodes; the
    summary is labelled smoke and must never be reported as a result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

TARGET_EVAL_VERSION = "target_eval_v1"


def load_protocol(path: str | Path) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def plan_scenes(protocol: dict, robot: str, *, sealed_run: bool, smoke: bool, smoke_episodes: int = 2) -> dict:
    """Which scenes may be evaluated (raises on a protocol violation)."""
    ev = protocol["eval"]
    if smoke:
        from rrp.bodies.armdiv import is_sealed_target
        if is_sealed_target(robot) or robot in protocol.get("targets", []):
            raise ValueError("smoke evaluations never touch a target body (protocol dev_rule)")
        if not 1 <= smoke_episodes <= 3:
            raise ValueError("smoke: 1-3 episodes")
        return dict(kind="smoke", seed_start=int(ev.get("dev_seed_start", 3_000_000)), episodes=int(smoke_episodes))
    if robot in protocol["targets"]:
        kind = "target"
    elif robot in protocol.get("source_heldout_eval_robots", []):
        kind = "source_heldout"
    else:
        raise ValueError(f"{robot} is neither a protocol target nor a held-out source body")
    if not sealed_run:
        raise ValueError("sealed evaluation scenes need --sealed-run (protocol dev_rule)")
    return dict(kind=kind, seed_start=int(ev["seed_start"]), episodes=int(ev["episodes"]))


def _feasible(robot_key: str, seeds) -> tuple[list[int], list[int]]:
    from rrp.bodies.catalog import workbench_robots
    from rrp.envs.mujoco.session import Session
    from rrp.envs.mujoco.scenario import BUILDERS
    from rrp.policies.teachers.arm import PickPlaceTeacher
    robot = workbench_robots()[robot_key]()
    ok, bad = [], []
    for sd in seeds:
        s = Session(BUILDERS["pick_place"](robot, sd, n_distractors=sd % 3), seed=sd)
        (ok if PickPlaceTeacher(s).feasibility()["feasible"] else bad).append(sd)
    return ok, bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="recipes/presets/eval-latent_slice1.json")
    ap.add_argument("--robot", required=True)
    ap.add_argument("--route", choices=["generated", "learned"], required=True)
    ap.add_argument("--flow")
    ap.add_argument("--rep")
    ap.add_argument("--policy", help="route learned: direct-action BC checkpoint")
    ap.add_argument("--policy-label")
    ap.add_argument("--prev-action", choices=["zero", "own"], default="zero")
    ap.add_argument("--kind", default="zero_shot", help="label: zero_shot | adapted:<method>:b<budget> | source")
    ap.add_argument("--sealed-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-episodes", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chunk-blend", choices=["none", "crossfade", "ensemble"], default="none")
    ap.add_argument("--blend-ticks", type=int, default=4)
    ap.add_argument("--blend-decay", type=float, default=0.0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        from rrp.ops.workload import apply_cap
        apply_cap()
    proto, sha = load_protocol(a.protocol)
    sc = plan_scenes(proto, a.robot, sealed_run=a.sealed_run, smoke=a.smoke, smoke_episodes=a.smoke_episodes)
    ev = proto["eval"]
    seeds, infeasible = _feasible(a.robot, range(sc["seed_start"], sc["seed_start"] + sc["episodes"]))
    if a.route == "generated" and not (a.flow and a.rep):
        sys.exit("route generated needs --flow and --rep")
    if a.route == "learned" and not a.policy:
        sys.exit("route learned needs --policy")
    from rrp.harness.eval.ladder import LadderConfig, load_models, run_ladder, summarize
    cfg = LadderConfig(route=a.route, robot=a.robot, seeds=seeds, representation=a.rep if a.route == "generated" else None,
                       flow=a.flow if a.route == "generated" else None, policy=a.policy, policy_label=a.policy_label,
                       replan_ticks=int(ev["replan_ticks"]), max_steps=int(ev["max_steps"]), nfe=int(ev["nfe"]),
                       compare_oracle=False, device=dev, prev_action=a.prev_action, chunk_blend=a.chunk_blend,
                       blend_ticks=a.blend_ticks, blend_decay=a.blend_decay)
    models, ids = load_models(cfg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{a.route}_{a.tag}"
    p = out / f"{name}.jsonl"
    if p.exists():
        p.unlink()
    rows = []
    for i in range(0, len(seeds), a.batch):
        cfg.seeds = seeds[i:i + a.batch]
        rows += run_ladder(cfg, p, models, ids)
        print(f"{len(rows)}/{len(seeds)} success={sum(r['privileged_success'] for r in rows)}", flush=True)
    cvs = [r["motion"].get("chunk_vel_step_max") for r in rows if isinstance(r.get("motion"), dict)]
    cvs = [x for x in cvs if x is not None]
    src = (f"learned:{a.flow}" if a.route == "generated" else f"learned:{a.policy_label or a.policy} (bc)")
    summ = dict(summarize(rows), version=TARGET_EVAL_VERSION, route=a.route, robot=a.robot, kind=a.kind,
                scene_kind=sc["kind"], smoke=a.smoke, sealed_run=a.sealed_run, controller_source=src,
                protocol=dict(id=proto.get("id"), status=proto.get("status"), path=a.protocol, sha256=sha),
                seeds=dict(start=sc["seed_start"], requested=sc["episodes"], feasible=len(seeds),
                           infeasible=len(infeasible)),
                replan=int(ev["replan_ticks"]), nfe=int(ev["nfe"]), max_steps=int(ev["max_steps"]),
                prev_action=a.prev_action, chunk_vel_step_max_mean=float(np.mean(cvs)) if cvs else None,
                chunk_vel_step_flag_frac=float(np.mean([x > 1.5 for x in cvs])) if cvs else None, checkpoints=ids)
    if a.chunk_blend != "none":
        summ["chunk_blend"] = ids.get("chunk_blend")
    if a.smoke:
        summ["label"] = "SMOKE (plumbing check on a non-target body; not a result)"
    (out / f"{name}.summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps({k: v for k, v in summ.items() if k != "checkpoints"}, default=str))
