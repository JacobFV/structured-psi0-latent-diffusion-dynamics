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
    ap.add_argument("--route", choices=["teacher", "oracle", "generated"], required=True)
    ap.add_argument("--robot", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed-start", type=int, default=3_000_000)
    ap.add_argument("--rep")
    ap.add_argument("--flow")
    ap.add_argument("--replan", type=int, default=8)
    ap.add_argument("--prev-action", choices=["zero", "own"], default="zero")
    ap.add_argument("--nfe", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--object-shift", help="tick,dx,dy")
    ap.add_argument("--no-compare", action="store_true")
    ap.add_argument("--disturbance", action="store_true")
    ap.add_argument("--tag", default="")
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
    from rrp.evaluation.ladder import LadderConfig, load_models, run_ladder, summarize, OraclePacketPolicy
    from rrp.learning.latent_grpo import feasible_seeds
    seeds = feasible_seeds(a.robot, a.seed_start, a.n)
    cfg = LadderConfig(route=a.route, robot=a.robot, seeds=seeds, representation=a.rep, flow=a.flow,
                       replan_ticks=a.replan, max_steps=a.max_steps, nfe=a.nfe, compare_oracle=not a.no_compare,
                       device=dev, prev_action=a.prev_action, object_shift=tuple(float(x) for x in a.object_shift.split(",")) if a.object_shift else None)
    if cfg.object_shift:
        cfg.object_shift = (int(cfg.object_shift[0]),) + cfg.object_shift[1:]
    models, ids = load_models(cfg) if (a.route != "teacher" or a.rep) else (dict(E=None, R=None, P=None, lcfg=None, res=None, flow=None), {})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = a.route + (f"_{a.tag}" if a.tag else "")
    if a.disturbance:
        from rrp.evaluation.latent_eval import disturbance_test
        pol = models["flow"] if a.route == "generated" else OraclePacketPolicy(models["E"], models["lcfg"], models["res"], dev)
        rows = []
        for ji in (1, 3):
            rows += [dict(r, joint_index=ji, route=a.route, checkpoints=ids)
                     for r in disturbance_test(pol, models["R"], a.robot, seeds, joint_index=ji, device=dev)]
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
    for i in range(0, len(seeds), a.batch):
        cfg.seeds = seeds[i:i + a.batch]
        rows += run_ladder(cfg, p, models, ids)
        print(f"{len(rows)}/{len(seeds)} success={sum(r['privileged_success'] for r in rows)}", flush=True)
    summ = dict(summarize(rows), route=a.route, robot=a.robot, seeds=[seeds[0], seeds[-1], len(seeds)],
                replan=a.replan, nfe=a.nfe, prev_action=a.prev_action, object_shift=a.object_shift, checkpoints=ids)
    (out / f"{name}.summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps({k: v for k, v in summ.items() if k != "checkpoints"}, default=str))


if __name__ == "__main__":
    main()
