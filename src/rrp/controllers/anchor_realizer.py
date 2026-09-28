"""System-0 input variant with anchor-relative inputs (W12, opt-in; realizer arch tag "rz-anchor-in-v1").

`AnchorLatentRealizer` = LatentRealizer + one extra input: per packet slot m the deploy-time anchor features of
rrp.features.anchor_frame.anchor_inputs (TCP pose relative to the own / other contact anchor and the frame estimate,
anchor normal in the tool frame, validity, age, covariance), computed ONLY from FK of measured joints and the task
runtime's receipts. Each action node receives the features of its own slot (node_asm). The added projection is
zero-initialized, so at initialization the variant computes exactly what the base realizer computes on the same
weights (tested); the base class and every existing bundle are unchanged.

It also accepts per-sample knot times [B, K] (event-aligned knots, rrp.data.contact_segments "knots-contact-v1");
the packet contract already carries knot_times per packet.

`session_anchor_inputs(session, manipulators)` is the runtime adapter: it reads session.tcp_pose (FK of measured
joints, rrp.envs.dual) and session.runtime.receipts, nothing else.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from rrp.controllers.latent_realizer import LatentRealizer
from rrp.features.anchor_frame import ANCHOR_INPUT_DIM, ANCHOR_INPUT_VERSION, anchor_inputs
from rrp.models.flow import sinusoidal

ANCHOR_REALIZER_ARCH = f"rz-{ANCHOR_INPUT_VERSION}"


class AnchorLatentRealizer(LatentRealizer):
    def __init__(self, dz: int, width: int = 192, heads: int = 4, layers: int = 2, z_norm: bool = False,
                 anchor_dim: int = ANCHOR_INPUT_DIM):
        super().__init__(dz, width=width, heads=heads, layers=layers, z_norm=z_norm)
        self.anchor_in = nn.Linear(anchor_dim, width)
        nn.init.zeros_(self.anchor_in.weight)
        nn.init.zeros_(self.anchor_in.bias)
        self.arch_tag = ANCHOR_REALIZER_ARCH

    @classmethod
    def from_base(cls, base: LatentRealizer, anchor_dim: int = ANCHOR_INPUT_DIM) -> "AnchorLatentRealizer":
        """Variant initialized from a trained base realizer (warm start for the phase-C system-0 refit)."""
        dz = base.z_in.in_features
        net = cls(dz, width=base.D, heads=base.blocks[0]["x"].h,
                  layers=len(base.blocks), z_norm=base.z_norm, anchor_dim=anchor_dim)
        missing, unexpected = net.load_state_dict(base.state_dict(), strict=False)
        assert not unexpected and all(k.startswith("anchor_in.") for k in missing), (missing, unexpected)
        return net

    def forward(self, z, zmask, knot_times, phase, node_feats, node_mask, local, node_asm=None, anchor=None):
        """As LatentRealizer.forward, plus anchor [B, M, anchor_dim] (per packet slot; None = zeros) and knot_times
        either [K] or [B, K]."""
        B, K, M, _ = z.shape
        N = node_feats.shape[1]
        kt = knot_times if knot_times.dim() == 2 else knot_times[None, :].expand(B, -1)
        rel_t = kt - phase[:, None]
        if self.z_norm:
            z = (z - self.z_mean) / self.z_std
        tok = self.z_in(z) + self.dt_in(sinusoidal(rel_t, self.D))[:, :, None]
        tok = tok.reshape(B, K * M, -1)
        if node_asm is None:
            node_asm = torch.zeros(B, N, dtype=torch.long, device=z.device)
        knot_asm = torch.arange(M, device=z.device).repeat(K)
        own = (node_asm[:, :, None] == knot_asm[None, None, :]) & zmask[:, None, :].repeat(1, 1, K)
        bias = torch.zeros(B, 1, N, K * M, device=z.device, dtype=tok.dtype).masked_fill(~own[:, None], float("-inf"))
        loc = self.local(local)
        x = self.node(node_feats) + (loc if local.dim() == 3 else loc[:, None])
        if anchor is not None:
            per_node = torch.gather(anchor, 1, node_asm.clamp(0, anchor.shape[1] - 1)[..., None]
                                    .expand(-1, -1, anchor.shape[-1]))                     # [B,N,F]
            x = x + self.anchor_in(per_node)
        for L in self.blocks:
            x = x + L["x"](L["n1"](x), kv=tok, bias=bias)
            x = x + L["s"](L["n2"](x), key_mask=node_mask)
            x = x + L["m"](L["n3"](x))
        return self.out(x).squeeze(-1) * node_mask


def session_anchor_inputs(session, manipulators: list[str], max_m: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Deploy-time anchor inputs for a DualSession: FK of measured joints + runtime receipts only."""
    tcp = {e: session.tcp_pose(e) for e in manipulators if e is not None and e in session.handles}
    now = float(session.runtime.clock())
    return anchor_inputs(tcp, session.runtime.receipts.all(), manipulators, now, max_m=max_m)
