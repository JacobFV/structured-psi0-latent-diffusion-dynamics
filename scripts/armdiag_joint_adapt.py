"""armdiag (D-135 follow-up): JOINT system-i flow + system-0 adaptation on target demos with a TOTAL update budget
matched to BC SFT (default 600 = 300 flow SFT + 300 system-0 refit updates; same episodes chosen by the protocol's
nested_budget_indices with the adapt seed, lr 1e-4). NON-SEALED DIAGNOSTIC probe: outputs are evaluated on DEV seeds
only (scripts/armdiag_closedloop.sh with EXTRA_FLOW / EXTRA_REP and route gen_X_X); never a sealed cell.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rrp.models.checkpoint import load_checkpoint
from rrp.training.latent_train import refit_realizer, sft_latent_flow


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", required=True)
    ap.add_argument("--rep", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--flow-steps", type=int, default=300)
    ap.add_argument("--refit-steps", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    r1 = sft_latent_flow(Path(a.flow), Path(a.pack), a.budget, seed=a.seed, out_dir=out / "flow", steps=a.flow_steps,
                         lr=1e-4)
    src = load_checkpoint(Path(a.rep), map_location="cpu")["config"]
    cfg = dict(representation=a.rep, packed_dir=a.pack, steps=a.refit_steps, batch_size=128, lr=1e-4, seed=a.seed,
               init="old", episode_budget=a.budget, budget_seed=a.seed, prefetch_workers=0,
               zero_prev_action=bool(src.get("zero_prev_action")), realizer_anchor=bool(src.get("realizer_anchor")),
               realizer_drop_qd=bool(src.get("realizer_drop_qd")), name=f"armdiag_joint_{out.name}")
    (out / "rz").mkdir(parents=True, exist_ok=True)
    r2 = refit_realizer(cfg, out / "rz")
    res = dict(label="NON-SEALED DIAGNOSTIC joint adaptation", total_updates=a.flow_steps + a.refit_steps,
               flow_sft=r1, refit=dict(steps_refit=r2.get("steps_refit"), target_adaptation=r2.get("target_adaptation"),
                                       realize_mse=(r2.get("eval") or {}).get("realize_mse")))
    (out / "joint_result.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, default=str)[:800])


if __name__ == "__main__":
    main()
