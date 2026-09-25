"""Likelihood math for packet-policy GRPO (system i flow-SDE on the latent packet)."""
import math

import torch

from rrp.learning.flow_sde import SDEConfig, path_log_prob, gaussian_path_kl
from rrp.learning.latent_grpo import LatentSDEPolicy, latent_collate
from rrp.model.flow import FlowPolicy, PolicyConfig
from rrp.sim.fixtures import make_pick_place_session

DZ, K = 8, 4


def _policy(seed=0):
    torch.manual_seed(seed)
    m = FlowPolicy(PolicyConfig(width=32, heads=2, ctx_layers=1, blocks=1, horizon=K, latent_dim=DZ, aux=False))
    with torch.no_grad():
        m.out.weight.normal_(0, 0.2)
        m.set_target_norm(torch.linspace(-1, 1, DZ), torch.linspace(0.5, 2, DZ))
    m.eval()
    return m


def _actor(m, sde):
    return LatentSDEPolicy(m, sde=sde, version="v0", knot_times=(0.1, 0.3, 0.5, 0.7), latent_space_version="ls-t",
                           realizer_compat_version="rz-t", device="cpu", seed=3)


def _vel(m, rec):
    b = latent_collate([rec["pi"]])
    cache = m.prepare(b)
    return lambda z, t: m.velocity(z, t, cache)


def test_latent_path_likelihood_ratio_padding_and_kl():
    if True:
        s = make_pick_place_session(seed=3)
        m = _policy()
        sde = SDEConfig(nfe=4, noise_level=0.5, first_step="clamp", last_step="deterministic")
        a = _actor(m, sde)
        pk = a.packets([s])[0]
        rec = a.last_records[0]
        p = rec["path"]
        assert p.latents.shape == (5, 1, K, 2, DZ) and p.valid[0, :, 1].sum() == 0   # single-gripper body: M=1 of 2
        # emitted packet == denormalized final state of the recorded path (flow runs in standardized space)
        assert torch.allclose(torch.from_numpy(pk.z).float(), m.denormalize(p.latents[-1])[0, :, :1], atol=1e-4)
        vel = _vel(m, rec)
        lp = path_log_prob(vel, p)
        assert torch.allclose(lp, p.old_log_prob, atol=1e-3)                 # ratio == 1 at identical weights
        # independent density: per stochastic step, Normal over the VALID coords only
        ref = 0.0
        h = 1.0 / sde.nfe
        with torch.no_grad():
            for k in range(sde.nfe):
                if not p.stochastic[k]:
                    continue
                t = float(p.taus[k]); g = float(p.g[k]); z = p.latents[k]
                v = vel(z, torch.full((1,), t))
                mu = z + h * v - g * g * h / (2 * (1 - t)) * (z - t * v)
                sd = g * math.sqrt(h)
                ref += torch.distributions.Normal(mu[p.valid], sd).log_prob(p.latents[k + 1][p.valid]).sum()
        assert not p.stochastic[-1] and p.stochastic[0]
        assert abs(float(lp) - float(ref)) < 1e-3
        assert int(p.valid.sum()) == K * 1 * DZ
        # garbage in the padded assembly slot changes nothing
        lat = p.latents.clone(); lat[:, :, :, 1] = 1e3
        p2 = type(p)(lat, p.taus, p.g, p.std, p.stochastic, p.valid, p.old_log_prob, p.old_step_log_prob, "v0", sde)
        assert torch.allclose(path_log_prob(vel, p2), lp, atol=1e-3)
        # exact same-variance path KL: 0 vs itself, sum_k ||dmu||^2 / (2 s_k^2) after a known velocity offset
        _, mA = path_log_prob(vel, p, return_means=True)
        assert float(gaussian_path_kl(mA, mA, p)) == 0.0
        off = 0.1
        vel_b = lambda z, t: vel(z, t) + off * p.valid[0].float()
        _, mB = path_log_prob(vel_b, p, return_means=True)
        kl = float(gaussian_path_kl(mA, mB, p))
        exp = 0.0
        for k in range(sde.nfe):
            if p.stochastic[k]:
                t = float(p.taus[k]); g = float(p.g[k])
                dmu = off * (h + g * g * h * t / (2 * (1 - t)))       # d mu / d v = h + g^2 h tau / (2(1-tau))
                exp += K * DZ * dmu ** 2 / (2 * float(p.std[k]) ** 2)
        assert abs(kl - exp) < 1e-4 * max(1, exp)
