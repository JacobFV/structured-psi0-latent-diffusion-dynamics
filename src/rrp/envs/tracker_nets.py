"""Actor-critic networks + running normaliser for legged trackers."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


# ------------------------------------------------------------------ networks
class RunningNorm(nn.Module):
    def __init__(self, dim, clip=5.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def forward(self, x):
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -self.clip, self.clip)


def mlp(i, hs, o):
    layers, d = [], i
    for h in hs:
        layers += [nn.Linear(d, h), nn.ELU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, priv_dim, act_dim, hidden=(256, 128, 64), init_std=0.6):
        super().__init__()
        self.obs_norm = RunningNorm(obs_dim)
        self.priv_norm = RunningNorm(priv_dim)
        self.actor = mlp(obs_dim, hidden, act_dim)
        self.critic = mlp(obs_dim + priv_dim, (256, 128, 128), 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), math.log(init_std)))
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def dist(self, obs):
        mu = self.actor(self.obs_norm(obs))
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    def value(self, obs, priv):
        return self.critic(torch.cat([self.obs_norm(obs), self.priv_norm(priv)], -1)).squeeze(-1)


