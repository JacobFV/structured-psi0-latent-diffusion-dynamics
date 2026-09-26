"""Closed-loop failure-localization ladder (rrp.evaluation.ladder). Examples:
  ladder.py --route teacher   --robot panda_pg2 --n 30 --out artifacts/runs/ladder_v1/panda_pg2
  ladder.py --route oracle    --robot panda_pg2 --n 30 --rep artifacts/runs/latent_sem_v1/representation.pt --out ...
  ladder.py --route generated --robot panda_pg2 --n 30 --flow artifacts/runs/flow_latent_sem_v2/policy.pt --out ...
  ladder.py --disturbance --route oracle ...   (fixed-packet joint disturbance via latent_eval.disturbance_test)
Writes <out>/<route>[_tag].jsonl (one row per episode) and <out>/<route>[_tag].summary.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", choices=["teacher", "oracle", "generated", "learned"], required=True)
    ap.add_argument("--robot", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed-start", type=int, default=3_000_000)
    ap.add_argument("--rep")
    ap.add_argument("--flow")
    ap.add_argument("--policy", help="route learned: LearnedPolicy checkpoint (baseline FlowPolicy)")
    ap.add_argument("--policy-label", help="label, e.g. baseline_direct_action_s1701@11000")
    ap.add_argument("--replan", type=int, default=8)
    ap.add_argument("--prev-action", choices=["zero", "own"], default="zero")
    ap.add_argument("--nfe", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--object-shift", help="tick,dx,dy")
    ap.add_argument("--no-compare", action="store_true")
    ap.add_argument("--keep-ticks", action="store_true")
    ap.add_argument("--reanchor", action="store_true", help="R1: re-anchor the expert reference at each replan")
    ap.add_argument("--collect-dagger", help="R1 only: write a system-0 DAgger buffer (.npz) of learner-visited states")
    ap.add_argument("--disturbance", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--render-seeds", help="comma list: render these seeds (one episode each) instead of evaluating")
    ap.add_argument("--video-out", default="artifacts/video")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.seed_start < 3_000_000:
        sys.exit("dev seeds must be >= 3,000,000")
    if a.robot in ("xarm7_pg2", "xarm7_tf3", "panda_tf3"):
        sys.exit("target bodies are not allowed in the ladder")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        from rrp.ops.gpu import apply_cap
        apply_cap()
    from rrp.evaluation.ladder import LadderConfig, load_models, run_ladder, summarize, OraclePacketPolicy, install_prev_action
    from rrp.learning.latent_grpo import feasible_seeds
    seeds = feasible_seeds(a.robot, a.seed_start, a.n)
    cfg = LadderConfig(route=a.route, robot=a.robot, seeds=seeds, representation=a.rep, flow=a.flow, policy=a.policy, policy_label=a.policy_label,
                       replan_ticks=a.replan, max_steps=a.max_steps, nfe=a.nfe, compare_oracle=not a.no_compare,
                       device=dev, prev_action=a.prev_action, keep_ticks=a.keep_ticks, oracle_reanchor=a.reanchor, object_shift=tuple(float(x) for x in a.object_shift.split(",")) if a.object_shift else None)
    if cfg.object_shift:
        cfg.object_shift = (int(cfg.object_shift[0]),) + cfg.object_shift[1:]
    models, ids = load_models(cfg) if (a.route != "teacher" or a.rep) else (dict(E=None, R=None, P=None, lcfg=None, res=None, flow=None), {})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = a.route + (f"_{a.tag}" if a.tag else "")
    if a.render_seeds:
        render(a, cfg, models, ids)
        return
    if a.disturbance:
        from rrp.evaluation.latent_eval import disturbance_test
        pol = models["flow"] if a.route == "generated" else OraclePacketPolicy(models["E"], models["lcfg"], models["res"], dev, reanchor=a.reanchor)
        rows = []
        for ji in (1, 3):
            rows += [dict(r, joint_index=ji, route=a.route, checkpoints=ids)
                     for r in disturbance_test(pol, models["R"], a.robot, seeds, joint_index=ji, device=dev,
                                               session_hook=lambda s: install_prev_action(s, a.prev_action))]
        p = out / f"disturbance_{name}.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        import numpy as np
        summ = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k.startswith("final_dev")}
        (out / f"disturbance_{name}.summary.json").write_text(json.dumps(dict(n=len(rows), **summ, checkpoints=ids), indent=1))
        print(json.dumps(summ))
        return
    p = out / f"{name}.jsonl"
    if p.exists():
        p.unlink()
    rows = []
    collect = {} if a.collect_dagger else None
    for i in range(0, len(seeds), a.batch):
        cfg.seeds = seeds[i:i + a.batch]
        rows += run_ladder(cfg, p, models, ids, collect=collect)
        print(f"{len(rows)}/{len(seeds)} success={sum(r['privileged_success'] for r in rows)}", flush=True)
    if collect is not None:
        from rrp.evaluation.ladder import save_dagger
        save_dagger(collect, Path(a.collect_dagger), dict(robot=a.robot, seeds=seeds, route=a.route, rep=a.rep,
                                                           reanchor=a.reanchor, prev_action=a.prev_action, ids=ids))
    summ = dict(summarize(rows), route=a.route, robot=a.robot, seeds=[seeds[0], seeds[-1], len(seeds)],
                replan=a.replan, nfe=a.nfe, prev_action=a.prev_action, reanchor=a.reanchor, keep_ticks=a.keep_ticks, oracle_reanchor=a.reanchor, object_shift=a.object_shift, checkpoints=ids)
    (out / f"{name}.summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps({k: v for k, v in summ.items() if k != "checkpoints"}, default=str))


def render(a, cfg, models, ids):
    import datetime as dt
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    import imageio
    import mujoco
    from rrp.evaluation.ladder import run_ladder
    sys.path.insert(0, str(Path(__file__).parent))
    from render_episode import caption
    label = dict(teacher="R0 SCRIPTED TEACHER (privileged)",
                 oracle="R1 ORACLE DIAGNOSTIC: E(teacher future actions) -> sys-0",
                 generated="R2 LEARNED sys-i flow -> sys-0",
                 learned=f"LEARNED plain BC learned:{a.policy_label or a.policy}")[a.route]
    ck = (a.policy_label or Path(a.policy).stem) if a.policy else Path(a.flow).stem if a.flow else (Path(a.rep).parent.name if a.rep else "-")
    out = Path(a.video_out)
    out.mkdir(parents=True, exist_ok=True)
    for sd in [int(x) for x in a.render_seeds.split(",")]:
        cfg.seeds = [sd]
        frames, rend = [], {}

        def cb(k, s, step, phase):
            if "r" not in rend:
                rend["r"] = mujoco.Renderer(s.model, 360, 480)
            r = rend["r"]
            r.update_scene(s.data, camera="front")
            st = " ".join(f"{e}:{v.status}" for e, v in s.runtime.instances.items())
            frames.append(caption(r.render().copy(), [f"{label} | {ck} | prev-action={a.prev_action}",
                                                      f"{a.robot} pick_place seed {sd} | t={s.data.time:.1f}s teacher-phase={phase}",
                                                      st]))
        row = run_ladder(cfg, None, models, ids, frame_cb=cb)[0]
        tag = "success" if row["privileged_success"] else f"failure-{row['failed_stage']}"
        name = f"{dt.date.today()}_ladder_{a.route}_{a.robot}_s{sd}_{a.tag or 'x'}_{tag}.mp4"
        imageio.mimsave(out / name, frames, fps=20, quality=6)
        with open(out / "INDEX.md", "a") as fh:
            fh.write(f"- `{name}` — ladder rung {a.route} ({row['source']}) ckpt={ck} prev_action={a.prev_action} "
                     f"robot={a.robot} task=pick_place seed={sd} outcome={tag} (privileged evaluator)\n")
        print(name, tag, flush=True)


if __name__ == "__main__":
    main()
