"""Unit-scale synthetic control sanity run for flow-SDE GRPO (no simulator).

Context o in {-1, +1}; a 1-D "action" a = z_K is produced by a small velocity MLP v(z, tau, o).
The behaviour-cloned start is broad (flow-matched to N(0, 1.5^2) regardless of o); the sparse reward is
success = |a - 1.5 o| < 0.35. GRPO with the exact path likelihood (same code path as the robot runs:
sample_sde / path_log_prob / group_advantages / clipped_surrogate) must raise the success rate.
"""
from __future__ import annotations

import json
import time

import torch
import torch.nn as nn

from rrp.learning.flow_sde import SDEConfig, sample_sde, path_log_prob, SDEPath
from rrp.learning.grpo import group_advantages, clipped_surrogate


class TinyVel(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))

    def forward(self, z, tau, o):
        return self.net(torch.cat([z, tau[:, None], o], -1))


def run_synthetic(iters=60, G=16, contexts=8, lr=3e-3, clip=0.2, epochs=2, seed=0, sde=None, eval_n=2000):
    torch.manual_seed(seed)
    sde = sde or SDEConfig(nfe=8, noise_level=0.7)
    m = TinyVel().double()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(600):                       # "BC" pretraining: broad, context-independent
        x = 1.5 * torch.randn(256, 1, dtype=torch.float64)
        o = torch.randint(0, 2, (256, 1)).double() * 2 - 1
        e = torch.randn_like(x)
        t = torch.rand(256, dtype=torch.float64)
        z = (1 - t[:, None]) * e + t[:, None] * x
        loss = ((m(z, t, o) - (x - e)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    gen = torch.Generator().manual_seed(seed + 1)

    def success(a, o):
        return ((a[:, 0] - 1.5 * o[:, 0]).abs() < 0.35).double()

    @torch.no_grad()
    def evaluate():
        o = torch.randint(0, 2, (eval_n, 1), generator=gen).double() * 2 - 1
        valid = torch.ones(eval_n, 1, dtype=torch.bool)
        p = sample_sde(lambda z, t: m(z, t, o), torch.randn(eval_n, 1, generator=gen, dtype=torch.float64), valid,
                       sde, behavior_version="eval", generator=gen)
        return float(success(p.action, o).mean())

    opt = torch.optim.Adam(m.parameters(), lr=lr)
    hist = [dict(iteration=0, success=evaluate())]
    t0 = time.time()
    for it in range(1, iters + 1):
        o = (torch.randint(0, 2, (contexts, 1), generator=gen).double() * 2 - 1).repeat_interleave(G, 0)
        valid = torch.ones(contexts * G, 1, dtype=torch.bool)
        path = sample_sde(lambda z, t: m(z, t, o), torch.randn(contexts * G, 1, generator=gen, dtype=torch.float64),
                          valid, sde, behavior_version=f"it{it}", generator=gen)
        R = success(path.action, o).reshape(contexts, G)
        adv = torch.cat([group_advantages(R[c])[0] for c in range(contexts)])
        first = None
        for _ in range(epochs):
            lp = path_log_prob(lambda z, t: m(z, t, o), path)
            obj, ratio = clipped_surrogate(lp, path.old_log_prob, adv, clip)
            if first is None:
                first = float((ratio.detach() - 1).abs().max())
            loss = -obj.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if it % 10 == 0 or it == iters:
            hist.append(dict(iteration=it, success=evaluate(), train_success=float(R.mean()),
                             first_pass_max_abs_ratio_minus_1=first))
    return dict(history=hist, before=hist[0]["success"], after=hist[-1]["success"], wall_s=time.time() - t0,
                sde=sde.__dict__, reward="sparse |a-1.5o|<0.35", label="synthetic (not robot)")


if __name__ == "__main__":
    print(json.dumps(run_synthetic(), indent=1))
