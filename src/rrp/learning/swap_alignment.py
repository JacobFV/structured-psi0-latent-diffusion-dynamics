"""Compatible-swap alignment (R23 / hypothesis h2).

Pairs: same arm, same scene seed, different compatible gripper module, same teacher phase
(privileged phase label used only to PAIR training samples). Loss aligns a low-rank
semantic projection of the system-i slot readouts (role/task/effect subspace), not native
joint coordinates or the full latent, so capability differences can persist elsewhere.
A contrastive term pushes apart pairs from DIFFERENT scenes (different goals).
"""
from __future__ import annotations

import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from rrp.learning.data import Sample, collate_samples


class SwapProjector(nn.Module):
    def __init__(self, D: int, rank: int = 16):
        super().__init__()
        self.p = nn.Linear(D, rank, bias=False)

    def forward(self, r):
        return F.normalize(self.p(r), dim=-1)


def build_pairs(episodes: list[tuple[dict, dict]], H: int) -> list[tuple[Sample, Sample]]:
    from rrp.learning.data import episode_samples
    by_key = defaultdict(dict)
    for pub, prv in episodes:
        m = pub["meta"]
        rk = m.get("robot_key", "")
        arm, _, grip = rk.rpartition("_")
        by_key[(arm, m["seed"])][grip] = (pub, prv)
    pairs = []
    for (arm, seed), d in by_key.items():
        if len(d) < 2:
            continue
        (g1, e1), (g2, e2) = sorted(d.items())[:2]
        s1, s2 = episode_samples(*e1, H, stride=4), episode_samples(*e2, H, stride=4)
        ph1 = [e1[1]["phases"][s.meta["t"]] for s in s1]
        ph2 = [e2[1]["phases"][s.meta["t"]] for s in s2]
        for i, p in enumerate(ph1):
            js = [j for j, q in enumerate(ph2) if q == p]
            if js:
                pairs.append((s1[i], s2[js[len(js) // 2]]))
    return pairs


def swap_alignment_loss(model, proj: SwapProjector, pairs: list[tuple[Sample, Sample]], dev, temperature=0.1):
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    outs = []
    for side in (a, b):
        batch, act, v, lab, eff = collate_samples(side)
        batch = batch.to(dev)
        cache = model.prepare(batch)
        z = torch.randn(act.shape, device=dev) * 0 + act.to(dev)   # clean action tokens at tau=1
        _, hidden = model.velocity(z, torch.ones(act.shape[0], device=dev), cache, return_hidden=True)
        layer = len(hidden) // 2
        h = hidden[layer].mean(1)                                  # [B, N, D] -> pool nodes below
        m = batch.node_mask.float()[..., None]
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        outs.append(proj(pooled))
    za, zb = outs
    logits = za @ zb.T / temperature
    target = torch.arange(za.shape[0], device=dev)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
