"""Flow-SDE path likelihood for GRPO (see research/methods/grpo-derivation.md).

Project convention: tau=0 noise, tau=1 data, z_tau = (1-tau) eps + tau x, v = x - eps, dz/dtau = v.
The marginal-preserving forward SDE (Flow-GRPO/piRL schedule mapped to our time direction) is

    dz = [ v - g(tau)^2 / (2 (1-tau)) * (z - tau v) ] dtau + g(tau) dW,   g(tau) = a sqrt((1-tau)/tau)

Euler-Maruyama step k (h = tau_{k+1} - tau_k > 0, g_k = g(tau_k)):

    mu_k = z_k + h v_k - g_k^2 h / (2 (1 - tau_k)) (z_k - tau_k v_k),   s_k = g_k sqrt(h)

Endpoints: tau_0 = 0 has g = inf. Default "clamp" uses g_0 = a sqrt((1-tau_0)/tau_1) (RLinf/Flow-GRPO
clamp) with the GENERAL mean form; "deterministic" makes step 0 an ODE step that gets NO density.
tau = 1 is never evaluated (last step starts at 1-h). The last step may be kept stochastic (default)
or deterministic (excluded from the likelihood).

The objective is the augmented PATH likelihood sum_k log p(z_{k+1} | z_k, obs) over stochastic steps
(prior N(0,I) of z_0 omitted: parameter-free, cancels in ratios). It is NOT the marginal density of the
final action. Log-densities are exact sums over valid coordinates (no dimension normalization).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch

LOG_2PI = math.log(2 * math.pi)


def gaussian_transition_log_prob(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                                 valid: torch.Tensor) -> torch.Tensor:
    """log N(x; mean, diag(std^2)) summed over VALID coordinates of all non-batch dims -> [B].

    Padding (valid=False) contributes exactly 0 whatever its values (even NaN/inf or std=0).
    Non-positive / non-finite std on a valid coordinate raises ValueError (no silent clamping).
    """
    x, mean, std = torch.broadcast_tensors(x, mean, std)
    valid = valid.to(torch.bool).expand_as(x)
    if valid.any():
        sv = std[valid]
        if not torch.isfinite(sv).all() or (sv <= 0).any():
            raise ValueError("invalid transition std on a valid coordinate (must be finite and > 0)")
    safe_std = torch.where(valid, std, torch.ones_like(std))
    diff = torch.where(valid, (x - mean) / safe_std, torch.zeros_like(x))
    term = diff ** 2 + 2 * torch.log(safe_std) + LOG_2PI
    term = torch.where(valid, term, torch.zeros_like(term))
    return -0.5 * term.reshape(term.shape[0], -1).sum(1)


# ------------------------------------------------------------------ schedule
@dataclass
class SDEConfig:
    nfe: int = 8                     # K denoising steps on a uniform grid tau_k = k/K
    noise_level: float = 0.5         # a  (piRL/RLinf default 0.5)
    first_step: str = "clamp"        # clamp | deterministic
    last_step: str = "stochastic"    # stochastic | deterministic

    def validate(self):
        if self.nfe < 2:
            raise ValueError("flow-SDE needs K >= 2 (K=1 is degenerate)")
        if not (self.noise_level > 0 and math.isfinite(self.noise_level)):
            raise ValueError("noise_level must be finite and > 0")
        if self.first_step not in ("clamp", "deterministic") or self.last_step not in ("stochastic", "deterministic"):
            raise ValueError("unknown endpoint treatment")
        return self


def time_grid(K: int, dtype=torch.float64) -> torch.Tensor:
    return torch.arange(K + 1, dtype=dtype) / K


def diffusion_schedule(cfg: SDEConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> taus [K+1], g [K] (0 on deterministic steps), stochastic mask [K]. float64."""
    cfg.validate()
    K, a = cfg.nfe, cfg.noise_level
    taus = time_grid(K)
    g = torch.zeros(K, dtype=torch.float64)
    stoch = torch.ones(K, dtype=torch.bool)
    for k in range(K):
        t = float(taus[k])
        if k == 0:
            if cfg.first_step == "deterministic":
                stoch[k] = False
                continue
            g[k] = a * math.sqrt((1 - t) / float(taus[1]))      # clamp tau -> tau_1 in the denominator
        else:
            g[k] = a * math.sqrt((1 - t) / t)
        if k == K - 1 and cfg.last_step == "deterministic":
            stoch[k] = False
            g[k] = 0.0
    return taus, g, stoch


def transition_mean(z: torch.Tensor, v: torch.Tensor, tau: float, h: float, g: float) -> torch.Tensor:
    """General EM mean (valid for clamped g as well): z + h v - g^2 h/(2(1-tau)) (z - tau v)."""
    return z + h * v - (g * g * h / (2 * (1 - tau))) * (z - tau * v)


# ------------------------------------------------------------------ recorded path
@dataclass
class SDEPath:
    latents: torch.Tensor          # [K+1, B, *shape]  z_0 .. z_K (z_K is the generated latent/action)
    taus: torch.Tensor             # [K+1]
    g: torch.Tensor                # [K] diffusion coefficient used (0 on deterministic steps)
    std: torch.Tensor              # [K] transition std s_k = g_k sqrt(h)
    stochastic: torch.Tensor       # [K] bool
    valid: torch.Tensor            # [B, *shape] bool
    old_log_prob: torch.Tensor     # [B] sum over stochastic steps
    old_step_log_prob: torch.Tensor  # [K, B] (0 on deterministic steps)
    behavior_version: str
    config: SDEConfig = field(default_factory=SDEConfig)

    @property
    def action(self) -> torch.Tensor:
        return self.latents[-1]

    def select(self, idx) -> "SDEPath":
        return SDEPath(self.latents[:, idx], self.taus, self.g, self.std, self.stochastic, self.valid[idx],
                       self.old_log_prob[idx], self.old_step_log_prob[:, idx], self.behavior_version, self.config)

    def to(self, device) -> "SDEPath":
        return SDEPath(self.latents.to(device), self.taus, self.g, self.std, self.stochastic, self.valid.to(device),
                       self.old_log_prob.to(device), self.old_step_log_prob.to(device), self.behavior_version,
                       self.config)

    @staticmethod
    def cat(paths: list["SDEPath"]) -> "SDEPath":
        p0 = paths[0]
        for p in paths[1:]:
            if p.behavior_version != p0.behavior_version or not torch.equal(p.std, p0.std):
                raise ValueError("cannot concatenate paths from different behavior versions/schedules")
        return SDEPath(torch.cat([p.latents for p in paths], 1), p0.taus, p0.g, p0.std, p0.stochastic,
                       torch.cat([p.valid for p in paths], 0), torch.cat([p.old_log_prob for p in paths]),
                       torch.cat([p.old_step_log_prob for p in paths], 1), p0.behavior_version, p0.config)


VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]   # (z [B,...], tau [B]) -> v


@torch.no_grad()
def sample_sde(vel: VelocityFn, noise: torch.Tensor, valid: torch.Tensor, cfg: SDEConfig, *,
               behavior_version: str, generator: torch.Generator | None = None) -> SDEPath:
    """Sample z_0=noise -> z_K with the recorded EM chain; padded coords stay exactly 0."""
    taus, g, stoch = diffusion_schedule(cfg)
    K = cfg.nfe
    h = 1.0 / K
    std = g * math.sqrt(h)
    m = valid.to(noise.dtype)
    z = noise * m
    B = z.shape[0]
    zs = [z]
    step_lp = torch.zeros(K, B, dtype=noise.dtype, device=noise.device)
    for k in range(K):
        t = float(taus[k])
        v = vel(z, torch.full((B,), t, dtype=z.dtype, device=z.device))
        mu = transition_mean(z, v, t, h, float(g[k]))
        if stoch[k]:
            eps = torch.randn(z.shape, generator=generator, device=z.device, dtype=z.dtype)
            z_next = (mu + float(std[k]) * eps) * m
            step_lp[k] = gaussian_transition_log_prob(z_next, mu, torch.full_like(mu, float(std[k])), valid)
        else:
            z_next = mu * m
        zs.append(z_next)
        z = z_next
    return SDEPath(torch.stack(zs), taus, g, std, stoch, valid.clone(), step_lp.sum(0), step_lp, behavior_version,
                   cfg)


def path_log_prob(vel: VelocityFn, path: SDEPath, return_means: bool = False):
    """Recompute log p_theta of the SAME recorded path (with grad through vel). -> [B] (and means)."""
    K = path.latents.shape[0] - 1
    h = 1.0 / K
    B = path.latents.shape[1]
    total = torch.zeros(B, dtype=path.latents.dtype, device=path.latents.device)
    means = []
    for k in range(K):
        if not path.stochastic[k]:
            means.append(None)
            continue
        t = float(path.taus[k])
        z = path.latents[k]
        v = vel(z, torch.full((B,), t, dtype=z.dtype, device=z.device))
        mu = transition_mean(z, v, t, h, float(path.g[k]))
        means.append(mu)
        total = total + gaussian_transition_log_prob(path.latents[k + 1], mu, torch.full_like(mu, float(path.std[k])),
                                                     path.valid)
    return (total, means) if return_means else total


def gaussian_path_kl(means_a: list, means_b: list, path: SDEPath) -> torch.Tensor:
    """Exact per-path sum of KL(N(mu_a, s^2) || N(mu_b, s^2)) over stochastic steps (same std). [B]"""
    B = path.latents.shape[1]
    total = torch.zeros(B, dtype=path.latents.dtype, device=path.latents.device)
    m = path.valid.to(path.latents.dtype)
    for k, (ma, mb) in enumerate(zip(means_a, means_b)):
        if ma is None:
            continue
        s2 = float(path.std[k]) ** 2
        total = total + (((ma - mb) ** 2) * m).reshape(B, -1).sum(1) / (2 * s2)
    return total
