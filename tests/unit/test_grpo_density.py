import math

import pytest
import torch

from rrp.learning.flow_sde import (SDEConfig, SDEPath, diffusion_schedule, gaussian_transition_log_prob,
                                   path_log_prob, sample_sde, transition_mean)
from rrp.learning.grpo import GRPOConfig, GRPOLearner, group_advantages
from rrp.learning.synthetic import TinyVel


def test_gaussian_log_density_is_not_flow_mse():
    x = torch.tensor([[0.]], dtype=torch.float64)
    lp = gaussian_transition_log_prob(x, mean=x, std=torch.ones_like(x),
                                      valid=torch.ones_like(x, dtype=torch.bool))
    assert torch.allclose(lp, torch.tensor([-0.5*math.log(2*math.pi)],
                                          dtype=torch.float64))


def test_analytic_1d_and_exact_dimension_sum():
    x, mu, sd = torch.tensor([[1.0, 3.0]], dtype=torch.float64), torch.zeros(1, 2, dtype=torch.float64), \
        torch.tensor([[2.0, 0.5]], dtype=torch.float64)
    lp = gaussian_transition_log_prob(x, mu, sd, torch.ones(1, 2, dtype=torch.bool))
    ref = sum(-0.5 * ((xi / si) ** 2 + 2 * math.log(si) + math.log(2 * math.pi)) for xi, si in [(1, 2), (3, .5)])
    assert lp.item() == pytest.approx(ref, abs=1e-12)       # summed, not dimension-averaged


def test_padding_has_no_effect_and_invalid_variance_rejected():
    x = torch.tensor([[0.3, float("nan")]], dtype=torch.float64)
    sd = torch.tensor([[0.7, 0.0]], dtype=torch.float64)
    valid = torch.tensor([[True, False]])
    lp = gaussian_transition_log_prob(x, torch.zeros_like(x), sd, valid)
    ref = gaussian_transition_log_prob(x[:, :1], torch.zeros(1, 1, dtype=torch.float64), sd[:, :1],
                                       torch.ones(1, 1, dtype=torch.bool))
    assert torch.isfinite(lp).all() and torch.allclose(lp, ref)
    with pytest.raises(ValueError):
        gaussian_transition_log_prob(x[:, :1], x[:, :1], torch.zeros(1, 1, dtype=torch.float64),
                                     torch.ones(1, 1, dtype=torch.bool))
    with pytest.raises(ValueError):
        SDEConfig(nfe=1).validate()


def test_ratio_is_one_at_identical_weights_and_padding_invariant():
    torch.manual_seed(0)
    m = TinyVel().double()
    o = torch.ones(6, 1, dtype=torch.float64)
    cfg = SDEConfig(nfe=6, noise_level=0.5)
    g = torch.Generator().manual_seed(0)
    valid = torch.ones(6, 1, dtype=torch.bool)
    p = sample_sde(lambda z, t: m(z, t, o), torch.randn(6, 1, dtype=torch.float64, generator=g), valid, cfg,
                   behavior_version="v0", generator=g)
    lp = path_log_prob(lambda z, t: m(z, t, o), p)
    assert torch.allclose(lp, p.old_log_prob, atol=1e-10)
    # add a padded coordinate carrying garbage: likelihood unchanged
    lat = torch.cat([p.latents, torch.full_like(p.latents, 7.0)], -1)
    pp = SDEPath(lat, p.taus, p.g, p.std, p.stochastic, torch.cat([valid, torch.zeros_like(valid)], -1),
                 p.old_log_prob, p.old_step_log_prob, "v0", cfg)
    vel2 = lambda z, t: torch.cat([m(z[:, :1], t, o), 100 * torch.ones_like(z[:, 1:])], -1)
    assert torch.allclose(path_log_prob(vel2, pp), lp, atol=1e-10)


def test_endpoint_schedule():
    taus, g, st = diffusion_schedule(SDEConfig(nfe=4, noise_level=0.5, first_step="clamp", last_step="stochastic"))
    assert g[0].item() == pytest.approx(0.5 * math.sqrt(4))          # clamp: a sqrt((1-0)/tau_1)
    assert (g * math.sqrt(0.25))[0].item() == pytest.approx(0.5)      # s_0 = a
    assert st.all() and taus[-2] == 0.75                               # tau=1 never evaluated
    _, g2, st2 = diffusion_schedule(SDEConfig(nfe=4, first_step="deterministic", last_step="deterministic"))
    assert st2.tolist() == [False, True, True, False] and g2[0] == 0 and g2[-1] == 0


def test_mean_shift_direction():
    """A path whose transitions landed above the mean must raise d logp / d(velocity offset) > 0."""
    cfg = SDEConfig(nfe=5, noise_level=0.6)
    taus, g, st = diffusion_schedule(cfg)
    h = 1 / cfg.nfe
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    z = [torch.zeros(1, 1, dtype=torch.float64)]
    for k in range(cfg.nfe):
        mu = transition_mean(z[-1], torch.zeros_like(z[-1]), float(taus[k]), h, float(g[k]))
        z.append(mu + 0.1)
    std = g * math.sqrt(h)
    p = SDEPath(torch.stack(z), taus, g, std, st, torch.ones(1, 1, dtype=torch.bool), torch.zeros(1),
                torch.zeros(cfg.nfe, 1), "v", cfg)
    lp = path_log_prob(lambda zz, t: b.expand_as(zz), p)
    lp.sum().backward()
    assert b.grad.item() > 0


def test_forward_sde_preserves_gaussian_marginal():
    """Noise/time conversion check: exact v for x ~ N(2, 0.5^2); sign flip of the correction would explode."""
    m_, s_ = 2.0, 0.5

    def v_exact(z, t):
        t = t[:, None]
        var_z = (1 - t) ** 2 + t ** 2 * s_ ** 2
        xh = m_ + t * s_ ** 2 / var_z * (z - t * m_)
        eh = (z - t * xh) / (1 - t)
        return xh - eh
    g = torch.Generator().manual_seed(0)
    n = 6000
    p = sample_sde(v_exact, torch.randn(n, 1, dtype=torch.float64, generator=g), torch.ones(n, 1, dtype=torch.bool),
                   SDEConfig(nfe=100, noise_level=0.8), behavior_version="x", generator=g)
    a = p.action[:, 0]
    assert abs(a.mean().item() - m_) < 0.05 and abs(a.std().item() - s_) < 0.05


def test_zero_variance_groups_and_staleness():
    adv, ok = group_advantages(torch.tensor([1.0, 1.0, 1.0]))
    assert not ok and torch.all(adv == 0)
    adv, ok = group_advantages(torch.tensor([1.0, 0.0, 0.0, 1.0]))
    assert ok and adv.tolist() == pytest.approx([1, -1, -1, 1], abs=1e-5)
    lr = GRPOLearner(TinyVel(), GRPOConfig(trainable="all"), "cpu", version_prefix="g")
    p = SDEPath(torch.zeros(3, 1, 1), torch.tensor([0, .5, 1.]), torch.ones(2), torch.ones(2),
                torch.ones(2, dtype=torch.bool), torch.ones(1, 1, dtype=torch.bool), torch.zeros(1),
                torch.zeros(2, 1), "g@7")
    with pytest.raises(ValueError):
        lr.update([dict(pi=None, path=p, adv=1.0)])
