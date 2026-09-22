"""Chunk-level Q ensemble and tanh-Gaussian edit policy (EXPO / EXPO-FT ingredients).

Q_k(s, a): s = frozen public-observation embedding, a = flattened normalized executed chunk [C*n].
Ensemble of E LayerNorm MLPs evaluated in one batched einsum; target = soft (Polyak) copy.
TD target: y = R + gamma^n (1 - done) * min over 2 random target members Q'(s', a*),
a* = argmax_{a in base candidates U edited candidates} min2 Q'(s', a)  (on-the-fly policy).
Edit: a_hat = beta * tanh(u), u ~ N(mu(s, a), sigma(s, a)); exact log-density of the unit-scale
squashed variable tanh(u) (the constant -d log beta is dropped: it shifts neither the edit gradient nor,
with the entropy target defined on the unit scale, the temperature update).
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleLinear(nn.Module):
    def __init__(self, E, d_in, d_out):
        super().__init__()
        self.w = nn.Parameter(torch.empty(E, d_in, d_out))
        self.b = nn.Parameter(torch.zeros(E, 1, d_out))
        bound = 1 / math.sqrt(d_in)
        nn.init.uniform_(self.w, -bound, bound)

    def forward(self, x):            # x [E, B, d_in]
        return torch.baddbmm(self.b, x, self.w)


class QEnsemble(nn.Module):
    def __init__(self, state_dim, action_dim, E=10, hidden=256, layers=3):
        super().__init__()
        self.E = E
        dims = [state_dim + action_dim] + [hidden] * layers
        self.lin = nn.ModuleList([EnsembleLinear(E, a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.out = EnsembleLinear(E, hidden, 1)

    def forward(self, s, a):          # [B, S], [B, A] -> [E, B]
        x = torch.cat([s, a], -1)[None].expand(self.E, -1, -1)
        for lin, n in zip(self.lin, self.norms):
            x = F.relu(n(lin(x)))
        return self.out(x).squeeze(-1)


def min_of_random_pair(q: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
    """q [E, B] -> min over 2 randomly chosen distinct members [B] (REDQ-style target)."""
    E = q.shape[0]
    if E < 2:
        return q[0]
    idx = torch.randperm(E, generator=gen)[:2].to(q.device)
    return q[idx].min(0).values


def soft_update(target: nn.Module, online: nn.Module, tau: float):
    with torch.no_grad():
        for pt, p in zip(target.parameters(), online.parameters()):
            pt.mul_(1 - tau).add_(p, alpha=tau)


def td_target(reward, n_steps, done, gamma, next_q_min2):
    """y = R + gamma^n (1 - done) * min2 Q'(s', a*)."""
    return reward + (gamma ** n_steps) * (1.0 - done) * next_q_min2


class EditPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, beta=0.05, hidden=256, layers=3, log_std_bounds=(-5.0, 2.0)):
        super().__init__()
        self.beta = beta
        dims = [state_dim + action_dim] + [hidden] * layers
        mods = []
        for a, b in zip(dims[:-1], dims[1:]):
            mods += [nn.Linear(a, b), nn.LayerNorm(b), nn.ReLU()]
        self.trunk = nn.Sequential(*mods)
        self.head = nn.Linear(hidden, 2 * action_dim)
        self.lo, self.hi = log_std_bounds

    def dist(self, s, a):
        mu, log_std = self.head(self.trunk(torch.cat([s, a], -1))).chunk(2, -1)
        log_std = self.lo + 0.5 * (self.hi - self.lo) * (torch.tanh(log_std) + 1)
        return mu, log_std

    def sample(self, s, a, deterministic=False, gen=None):
        """-> edit [B, A] in [-beta, beta], log-prob of the unit-scale squashed sample [B]."""
        mu, log_std = self.dist(s, a)
        if deterministic:
            u = mu
        else:
            u = mu + log_std.exp() * torch.randn(mu.shape, generator=gen, device=mu.device, dtype=mu.dtype)
        logp = squashed_log_prob(u, mu, log_std)
        return self.beta * torch.tanh(u), logp


def squashed_log_prob(u, mu, log_std):
    """log density of y = tanh(u), u ~ N(mu, exp(log_std)^2), summed over dims (numerically stable)."""
    base = -0.5 * (((u - mu) / log_std.exp()) ** 2 + 2 * log_std + math.log(2 * math.pi))
    log_det = 2 * (math.log(2) - u - F.softplus(-2 * u))       # log(1 - tanh(u)^2)
    return (base - log_det).sum(-1)


def make_target(m: nn.Module) -> nn.Module:
    t = copy.deepcopy(m)
    for p in t.parameters():
        p.requires_grad_(False)
    return t
