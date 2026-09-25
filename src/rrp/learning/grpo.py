"""Flow-SDE GRPO update (Z-1 §3.1.2 objective on the piRL/Flow-GRPO SDE; see grpo-derivation.md).

  A_i = (R_i - mean R) / (std_pop R + eps)                       (group-relative, per rollout)
  rho = exp(log p_new(path) - log p_old(path))                   (per trainable chunk, same path+obs)
  L   = -1/sum|B_i| * sum_i sum_{chunk in B_i} min(rho A_i, clip(rho, 1-eta, 1+eta) A_i),  eta = 0.2

No KL / reference policy by default (Z-1 uses none). An optional exact Gaussian path KL to a frozen
reference is available (kl_coef > 0) and must be reported as a labelled departure.
Codec and controller are frozen: latents are the optimized variables and the codec decoder / joint
controller are outside the graph. Zero-variance groups carry no signal and are EXCLUDED explicitly
(counted in stats), unless `all_equal="z1_min"` for decayed rewards (A_i = R_i - min R).
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, asdict

import torch

from rrp.learning.flow_sde import SDEConfig, SDEPath, path_log_prob, gaussian_path_kl
from rrp.model.batch import collate_inputs


@dataclass
class GRPOConfig:
    group_size: int = 8
    clip: float = 0.2
    lr: float = 1e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    epochs: int = 4                  # passes over each iteration's rollouts (Z-1: 4 rollout epochs)
    minibatch: int = 32              # chunks per optimizer step
    adv_eps: float = 1e-6
    zero_var_tol: float = 1e-9
    all_equal: str = "skip"          # skip | z1_min (only meaningful with success-time decay)
    kl_coef: float = 0.0             # 0 = no KL (faithful to Z-1); >0 = labelled departure
    trainable: str = "action_expert"  # action_expert (context encoder + aux readout frozen) | all
    # "sum" = exact augmented-path log-likelihood (default). "mean_dims" = DIMENSION-NORMALIZED SURROGATE
    # (log-ratio divided by #valid coords x #stochastic steps, as in Flow-GRPO/RLinf code); not an exact
    # likelihood ratio and always reported under that name.
    logprob_reduction: str = "sum"
    sde: SDEConfig = field(default_factory=SDEConfig)


def group_advantages(returns: torch.Tensor, eps: float = 1e-6, zero_var_tol: float = 1e-9,
                     all_equal: str = "skip") -> tuple[torch.Tensor, bool]:
    """-> (advantages [G], informative). Population std (1/G) as in Z-1 Eq. 2."""
    r = returns.to(torch.float64)
    if r.numel() < 2:
        return torch.zeros_like(r), False
    std = r.std(unbiased=False)
    if std <= zero_var_tol:
        return torch.zeros_like(r), False          # no ordinary mean-normalised signal
    if all_equal == "z1_min" and bool((r > 0).all()):
        return r - r.min(), True
    return (r - r.mean()) / (std + eps), True


def clipped_surrogate(logp_new: torch.Tensor, logp_old: torch.Tensor, adv: torch.Tensor, clip: float):
    """Per-chunk clipped objective (to be MAXIMIZED); returns (objective [n], ratio [n])."""
    ratio = torch.exp(logp_new - logp_old)
    obj = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv)
    return obj, ratio


def set_trainable(model, mode: str) -> list[str]:
    for p in model.parameters():
        p.requires_grad_(True)
    frozen = []
    if mode == "action_expert":
        for name in ("context", "readout"):
            m = getattr(model, name, None)
            if m is not None:
                for p in m.parameters():
                    p.requires_grad_(False)
                frozen.append(name)
    elif mode != "all":
        raise ValueError(mode)
    return frozen


class GRPOLearner:
    def __init__(self, model, cfg: GRPOConfig, device, version_prefix: str = "grpo", collate_fn=None):
        self.model = model
        self.collate_fn = collate_fn or collate_inputs    # latent path: assembly_batch(collate_inputs(.))
        self.cfg = cfg
        self.device = device
        self.frozen = set_trainable(model, cfg.trainable)
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.ref = None
        if cfg.kl_coef > 0:
            self.ref = copy.deepcopy(model).eval()
            for p in self.ref.parameters():
                p.requires_grad_(False)
        self.iteration = 0
        self.version_prefix = version_prefix
        self.opt_steps = 0
        self.update_velocity_evals = 0

    @property
    def version(self) -> str:
        return f"{self.version_prefix}@{self.iteration}"

    def module_report(self) -> dict:
        n_tr = sum(p.numel() for p in self.params)
        n_all = sum(p.numel() for p in self.model.parameters())
        return dict(trainable_params=n_tr, total_params=n_all, frozen_modules=sorted(self.frozen) + ["codec",
                    "controller"], changed_modules=sorted({n.split(".")[0] for n, p in self.model.named_parameters()
                                                          if p.requires_grad}))

    def _vel(self, model, batch):
        cache = model.prepare(batch)
        return lambda z, t: model.velocity(z, t, cache)

    def update(self, samples: list[dict]) -> dict:
        """samples: dicts with pi (PolicyInput), path (SDEPath, batch 1), adv (float), behavior_version.
        Paths must come from the CURRENT behavior version (stale rollouts are rejected)."""
        cfg = self.cfg
        for s in samples:
            if s["path"].behavior_version != self.version:
                raise ValueError(f"stale rollout: behavior {s['path'].behavior_version} != learner {self.version}")
        stats = dict(chunks=len(samples), opt_steps=0, first_pass_max_abs_ratio_minus_1=None, clip_frac=[],
                     ratio_mean=[], obj=[], kl=[], grad_norm=[])
        if not samples:
            self.iteration += 1
            return stats
        n_total = len(samples)
        self.model.train()
        g = torch.Generator().manual_seed(1000 + self.iteration)
        for ep in range(cfg.epochs):
            order = torch.randperm(n_total, generator=g).tolist()
            for i in range(0, n_total, cfg.minibatch):
                idx = order[i:i + cfg.minibatch]
                mb = [samples[j] for j in idx]
                batch = self.collate_fn([s["pi"] for s in mb]).to(self.device)
                path = SDEPath.cat([s["path"] for s in mb]).to(self.device)
                if path.valid.shape[2] != batch.node_mask.shape[1]:
                    raise ValueError("path/node padding mismatch")
                adv = torch.tensor([s["adv"] for s in mb], dtype=path.latents.dtype, device=self.device)
                logp, means = path_log_prob(self._vel(self.model, batch), path, return_means=True)
                self.update_velocity_evals += len(mb) * int(path.stochastic.sum())
                if cfg.logprob_reduction == "mean_dims":
                    n = path.valid.reshape(path.valid.shape[0], -1).sum(1).to(logp.dtype) * int(path.stochastic.sum())
                    obj, ratio = clipped_surrogate(logp / n, path.old_log_prob / n, adv, cfg.clip)
                elif cfg.logprob_reduction == "sum":
                    obj, ratio = clipped_surrogate(logp, path.old_log_prob, adv, cfg.clip)
                else:
                    raise ValueError(cfg.logprob_reduction)
                # Z-1 normalisation: sum over chunks / total trainable chunks of the iteration batch
                loss = -obj.sum() / len(mb)
                if self.ref is not None:
                    with torch.no_grad():
                        _, rmeans = path_log_prob(self._vel(self.ref, batch), path, return_means=True)
                    kl = gaussian_path_kl(means, rmeans, path).mean()
                    loss = loss + cfg.kl_coef * kl
                    stats["kl"].append(float(kl))
                if ep == 0 and i == 0:
                    stats["first_pass_max_abs_ratio_minus_1"] = float((ratio.detach() - 1).abs().max())
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(self.params, cfg.grad_clip)
                if not torch.isfinite(gn):
                    raise FloatingPointError("non-finite GRPO gradient")
                self.opt.step()
                self.opt_steps += 1
                stats["opt_steps"] += 1
                r = ratio.detach()
                stats["clip_frac"].append(float(((r < 1 - cfg.clip) | (r > 1 + cfg.clip)).float().mean()))
                stats["ratio_mean"].append(float(r.mean()))
                stats["obj"].append(float(obj.detach().mean()))
                stats["grad_norm"].append(float(gn))
        self.iteration += 1
        for k in ("clip_frac", "ratio_mean", "obj", "kl", "grad_norm"):
            v = stats[k]
            stats[k] = sum(v) / len(v) if v else None
        return stats


def build_samples(groups: list[dict], cfg: GRPOConfig) -> tuple[list[dict], dict]:
    """groups: [{returns: [G], members: [[chunk records...] x G]}] -> trainable samples + stats.
    Only suffix/trainable chunk records are in `members` (shared prefix excluded upstream)."""
    out, info = [], dict(iter_groups=len(groups), informative_groups=0, zero_variance_groups=0, excluded_chunks=0)
    for gr in groups:
        adv, ok = group_advantages(torch.tensor(gr["returns"], dtype=torch.float64), cfg.adv_eps, cfg.zero_var_tol,
                                   cfg.all_equal)
        if not ok:
            info["zero_variance_groups"] += 1
            info["excluded_chunks"] += sum(len(m) for m in gr["members"])
            continue
        info["informative_groups"] += 1
        for a, recs in zip(adv.tolist(), gr["members"]):
            for r in recs:
                out.append(dict(pi=r["pi"], path=r["path"], adv=a, behavior_version=r["behavior_version"]))
    return out, info
