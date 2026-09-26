"""Bounded semantic NLL option (D-085): the log-variance floor bounds the probe NLL; default keeps old versions."""
import math

import torch

from rrp.model.latent_probes import gaussian_nll
from rrp.model.semantic_latent import LatentConfig


def test_floor_bounds_nll():
    p = torch.zeros(1, 2, 6); p[..., 3:] = -10.0
    t = torch.zeros(1, 2, 3); m = torch.ones(1, 2, dtype=torch.bool)
    lo = lambda f: 1.5 * (f + math.log(2 * math.pi))
    assert abs(float(gaussian_nll(p, t, m)) - lo(-8.0)) < 1e-4
    assert abs(float(gaussian_nll(p, t, m, -4.0)) - lo(-4.0)) < 1e-4


def test_default_version_unchanged():
    assert LatentConfig(name="x").version() == LatentConfig(name="x", probe_lv_min=-8.0).version()
    assert LatentConfig(name="x").version() != LatentConfig(name="x", probe_lv_min=-4.0).version()
