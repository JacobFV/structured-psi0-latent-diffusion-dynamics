"""Latent-training view of a packed dataset (W4: `LatentData` moved unchanged from rrp.learning.latent_train, which
re-exports it): row i = (episode, t); system-0 targets use row i+j of the same episode."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from rrp.harness.data.packed import PackedChunkDataset


class LatentData:
    """Row i = (episode, t). Realizer targets use row i+j of the same episode (stride-1 packing required)."""

    def __init__(self, packed_dir: Path, zero_prev_action: bool = False, anchor: bool = False, drop_qd: bool = False):
        if anchor and not zero_prev_action:
            raise ValueError("realizer_anchor reuses node column 28: requires zero_prev_action")
        self.ds = PackedChunkDataset(packed_dir, zero_prev_action=zero_prev_action)
        self.anchor = anchor
        self.drop_qd = drop_qd          # realizer input without the joint-velocity column (velocity-copy causal confusion)
        if self.ds.meta["stride"] != 1:
            raise ValueError("latent training needs stride-1 packing (state at t+j)")
        self.ep = np.asarray(self.ds.arr["ep_idx"])
        self.t = np.asarray(self.ds.arr["t"])
        self.n = len(self.ds)

    def sample(self, B: int, rng: random.Random, max_j: int, idx_pool=None):
        pool = idx_pool if idx_pool is not None else range(self.n)
        sel = np.array(sorted(rng.sample(pool, B)) if not isinstance(pool, range) else
                       sorted(rng.sample(range(self.n), B)))
        j = np.array([rng.randint(0, max_j) for _ in range(B)])
        tgt = np.minimum(sel + j, self.n - 1)
        same = self.ep[tgt] == self.ep[sel]
        tgt = np.where(same, tgt, sel)            # phase beyond episode end -> j=0
        j = np.where(same, j, 0)
        return sel, tgt, j

    def fetch(self, sel, tgt, dev):
        batch, a, v, lab, eff = self.ds.collate(sel)
        A = self.ds.arr
        order = np.argsort(tgt)
        inv = np.argsort(order)
        ts = tgt[order]
        nodes = np.asarray(A["node"][ts]).astype(np.float32)[inv]
        if self.ds.zero_prev_action:                    # bug B-1 fix (see rrp.learning.packed.PREV_ACTION_COL)
            from rrp.harness.data.packed import PREV_ACTION_COL
            nodes[..., PREV_ACTION_COL] = 0
        nn_ = np.asarray(A["n_nodes"][ts])[inv]
        if self.ds.kinfeat:                             # D-137 ablation (rrp.features.kinfeat)
            nodes = self.ds.kinfeat_nodes(nodes, np.asarray(A["robot_id"][ts])[inv], nn_)
        a1 = np.asarray(A["a"][ts][:, 0]).astype(np.float32)[inv]           # 1-step teacher command at t+j
        v1 = np.asarray(A["valid"][ts][:, 0])[inv]
        loc = np.asarray(A["local"][ts]).astype(np.float32)[inv]
        N = batch.node_feats.shape[1]
        if self.anchor:            # anchored realizer: col 28 = normalized joint displacement since the packet state (t)
            from rrp.policies.system0 import Q_COL, ANCHOR_COL
            nodes[:, :N, ANCHOR_COL] = nodes[:, :N, Q_COL] - batch.node_feats[:, :, Q_COL].numpy()
        if self.drop_qd:
            from rrp.policies.system0 import QD_COL
            nodes[:, :, QD_COL] = 0
        lab["subtask"] = torch.from_numpy(np.asarray(A["subtask"][sel]).astype(np.int64))
        r = dict(node=torch.from_numpy(nodes[:, :N]), node_mask=torch.from_numpy(np.arange(N)[None] < nn_[:, None]),
                 a1=torch.from_numpy(a1[:, :N]), v1=torch.from_numpy(v1[:, :N]), local=torch.from_numpy(loc))
        if "held_m" in A:          # multi-assembly pack (rrp.learning.dual_latent)
            S = lab["held"].shape[1]
            for k in ("held_m", "contact_m", "rel_tcp_m"):
                x = np.asarray(A[k][sel])[:, :S]
                lab[k] = torch.from_numpy(x.astype(np.float32) if x.dtype == np.float16 else x)
            lab["subtask_m"] = torch.from_numpy(np.asarray(A["subtask_m"][sel]).astype(np.int64))
            na = np.asarray(A["node_asm"][ts])[inv][:, :N].astype(np.int64)            # slot of each node at t+j
            locm = np.asarray(A["local_m"][ts]).astype(np.float32)[inv]                 # [B,M,4] at t+j
            r["node_asm"] = torch.from_numpy(na)
            r["local"] = torch.from_numpy(np.take_along_axis(locm, na[..., None].clip(0, locm.shape[1] - 1), 1))  # per node
        from rrp.policies.nets.binding_aug import goal_effect_from_batch
        lab["goal_effect"] = goal_effect_from_batch(batch)          # task-goal label (public spec + estimates)
        mv = lambda x: x.to(dev, non_blocking=True)
        return batch.to(dev), mv(a[..., 0]), mv(v), {k: mv(x) for k, x in lab.items()}, {k: mv(x) for k, x in r.items()}
