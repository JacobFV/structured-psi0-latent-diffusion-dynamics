"""Legged train/deploy feature parity (lesson B-1, D-044/45).

Re-simulates stored teacher episodes (deterministic: same seed, DART sigma and teacher gains) and, at every native
tick, computes the features the DEPLOYED featurizers produce (System0Adapter.dyn_batch + static_batch for system 0;
public_context for system i / BC). Compares them with what TRAINING feeds the models for the same row
(LeggedData.ctx_batch on the stored shard). Any systematic difference = a train/deploy mismatch.

usage: PYTHONPATH=src python scripts/legged_feature_parity.py --body go2 --shard artifacts/datasets/legged_latent_v1/go2/s0-99.npz --eps 0,1,2 --out X.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rrp.control.legged_latent import LeggedMorph, public_context, active_event, TICK_DT
from rrp.data import legged_latent_collect as C
from rrp.evaluation.legged_latent_eval import System0Adapter, static_batch
from rrp.learning.legged_latent_train import LeggedData


class _Ctl:
    dev = torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default="go2")
    ap.add_argument("--data", default="artifacts/datasets/legged_latent_v1")
    ap.add_argument("--eps", default="0,1,2")
    ap.add_argument("--max-ticks", type=int, default=400)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = LeggedData(Path(a.data), [a.body], torch.device("cpu"), max_eps=max(int(e) for e in a.eps.split(",")) + 1)
    res = dict(body=a.body, episodes=[])
    for e in [int(x) for x in a.eps.split(",")]:
        em = data.ep_meta[e]
        meta = json.loads(sorted((Path(a.data) / a.body).glob("s*.json"))[0].read_text())["episodes"][e]
        deployed = []

        class Probe(C.RecordingTracker):
            def act(self, d, cmd):
                if self.rec is not None and len(deployed) < a.max_ticks:
                    ad = System0Adapter.__new__(System0Adapter)
                    ad.c, ad.s, ad.m, ad.ticks = _Ctl(), self.s, self.m, self.ticks
                    ad.c.static = static_batch(self.m, ad.c.dev)
                    b = ad.dyn_batch()
                    b["ctx"] = torch.from_numpy(self.ctx)[None] if self.ctx is not None else None
                    b["ctx_deploy"] = torch.from_numpy(public_context(self.s, ad.osc()))[None]
                    deployed.append(b)
                return super().act(d, cmd)

        orig = C.RecordingTracker
        C.RecordingTracker = Probe
        try:
            arr, m2, morph = C.collect_episode(a.body, meta["seed"], meta["sigma"], "auto",
                                               arc_only=meta.get("teacher_variant") == "arc_only")
        finally:
            C.RecordingTracker = orig
        n = min(len(deployed), em["T"])
        idx = torch.arange(em["start"], em["start"] + n)
        tb = data.ctx_batch(idx)
        diffs = {}
        for k in ("q", "qd", "imu", "asm_touch", "osc", "ctx"):
            dep = torch.cat([d[k].reshape(1, -1) for d in deployed[:n]]).float()
            diffs[k] = float((dep - tb[k].float().reshape(n, -1)).abs().max())
        for k in ("node_static", "node_asm", "asm_static", "node_mask", "asm_mask"):
            diffs[k] = float((deployed[0][k][0].float() - tb[k][0].float()).abs().max())
        # the ctx system i sees at a replan (deploy) vs the stored step-start ctx used in training, on step-aligned ticks
        cd = torch.cat([d["ctx_deploy"] for d in deployed[:n]])
        diffs["ctx_deploy_vs_train_all_ticks"] = float((cd - tb["ctx"].float()).abs().max())
        al = torch.arange(0, n, 5)                     # step-start ticks (5 native ticks per 10 Hz step)
        diffs["ctx_deploy_vs_train_step_aligned"] = float((cd[al] - tb["ctx"].float()[al]).abs().max())
        diffs["ctx_deploy_vs_train_osc_cols_excluded"] = float(torch.cat([(cd - tb["ctx"].float())[:, :6],
                                                                          (cd - tb["ctx"].float())[:, 8:]], 1).abs().max())
        rec_equal = {k: float(np.abs(arr[k][:n].astype(np.float64) - data.A[k][idx].numpy()[..., :arr[k].shape[-1]]
                                     .reshape(arr[k][:n].shape)).max()) for k in ("a", "q", "qd")}
        res["episodes"].append(dict(ep=e, seed=meta["seed"], sigma=meta["sigma"], ticks=n,
                                    resim_vs_stored_maxabs=rec_equal, deploy_vs_train_maxabs=diffs))
        print(json.dumps(res["episodes"][-1]), flush=True)
    res["system0_inputs"] = "q, qd (local_state), imu, per-foot touch, osc-v1 (own tick counter), morphology tokens, z, phase"
    res["teacher_command_or_prev_action_in_inputs"] = False
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
