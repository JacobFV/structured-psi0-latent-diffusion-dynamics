"""System 0: online, morphology-conditioned realization of the received latent packet (R38, section 8).

Every control tick (20 Hz) system 0 computes native joint references from
    z (the received LatentActionChunk, unchanged) + morphology/proprio node features (public FK of CURRENT measured
    state) + declared local sensors (touch summary, gripper width) + elapsed phase since packet.valid_from,
and hands them to the existing joint-target tracker (500 Hz physics underneath).
It never sees the task graph, instructions, scene/object estimates, system-i hidden state or its context cache.
Newly sensed facts (current proprio/touch) are inputs every tick, so a stale planning-time belief in z cannot
override what the sensors report now. Local recurrent state: none in v1 (declared; history enters via phase).
Knot consumption: learned attention over knot tokens keyed by (knot_time - phase); no interpolation is assumed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from rrp.contracts.action import NativeCommand
from rrp.contracts.latent_action import LatentActionChunk, check_packet
from rrp.contracts.errors import ControllerRejection, StaleActionError
from rrp.model.attention import MHA
from rrp.model.batch import NODE_DIM
from rrp.model.flow import MLP, sinusoidal
import hashlib

REALIZER_RECURRENT_STATE = "none-v1"


class LatentRealizer(nn.Module):
    def __init__(self, dz: int, width: int = 192, heads: int = 4, layers: int = 2):
        super().__init__()
        D = width
        self.node = MLP(NODE_DIM, D)
        self.local = nn.Linear(4, D)
        self.z_in = nn.Linear(dz, D)
        self.dt_in = MLP(D, D)
        self.blocks = nn.ModuleList([nn.ModuleDict(dict(n1=nn.LayerNorm(D), x=MHA(D, heads), n2=nn.LayerNorm(D),
                                                        s=MHA(D, heads), n3=nn.LayerNorm(D), m=MLP(D, D, 4 * D)))
                                     for _ in range(layers)])
        self.out = nn.Linear(D, 1)
        self.D = D

    def forward(self, z, zmask, knot_times, phase, node_feats, node_mask, local, node_asm=None):
        """z [B,K,M,dz]; knot_times [K] (s); phase [B] (s since valid_from); node_feats [B,N,F]; local [B,4] or,
        for multi-assembly bodies, per node [B,N,4] (each node sees its own assembly's touch/width);
        node_asm [B,N] packet assembly index per node (default 0). Returns normalized 1-step actions [B,N]."""
        B, K, M, _ = z.shape
        N = node_feats.shape[1]
        rel_t = knot_times[None, :] - phase[:, None]                                  # [B,K]
        kt = self.z_in(z) + self.dt_in(sinusoidal(rel_t, self.D))[:, :, None]         # [B,K,M,D]
        kt = kt.reshape(B, K * M, -1)
        if node_asm is None:
            node_asm = torch.zeros(B, N, dtype=torch.long, device=z.device)
        knot_asm = torch.arange(M, device=z.device).repeat(K)                          # [K*M]
        own = (node_asm[:, :, None] == knot_asm[None, None, :]) & zmask[:, None, :].repeat(1, 1, K)  # [B,N,K*M]
        bias = torch.zeros(B, 1, N, K * M, device=z.device, dtype=kt.dtype).masked_fill(~own[:, None], float("-inf"))
        loc = self.local(local)
        x = self.node(node_feats) + (loc if local.dim() == 3 else loc[:, None])     # local [B,4] or per node [B,N,4]
        for L in self.blocks:
            x = x + L["x"](L["n1"](x), kv=kt, bias=bias)
            x = x + L["s"](L["n2"](x), key_mask=node_mask)
            x = x + L["m"](L["n3"](x))
        return self.out(x).squeeze(-1) * node_mask


@dataclass
class System0Stats:
    ticks: int = 0
    packets: int = 0
    rejected: int = 0
    fallback_holds: int = 0


class LatentSystem0:
    """Runtime wrapper: holds the current packet, realizes it every tick from fresh local state."""

    def __init__(self, realizer: LatentRealizer, featurizer, *, latent_space_version: str,
                 realizer_compat_version: str, device="cpu", fallback: str = "hold_measured"):
        self.net = realizer.eval()
        self.f = featurizer
        # the realizer's OWN frozen-bundle fingerprint (set by load_representation) is authoritative; the
        # caller-supplied IDs are only used for legacy realizers without one
        self.lsv, self.rcv = getattr(realizer, "bundle_versions", None) or (latent_space_version, realizer_compat_version)
        self.device = device
        self.packet: LatentActionChunk | None = None
        self.fallback = fallback
        self.stats = System0Stats()
        self.log: list[dict] = []

    def receive(self, packet: LatentActionChunk, *, now: float, graph_version: int | None = None):
        try:
            check_packet(packet, latent_space_version=self.lsv, realizer_compat_version=self.rcv,
                         robot_spec_hash=self.f.spec.spec_hash, now=now, graph_version=graph_version)
        except (ControllerRejection, StaleActionError) as e:
            self.stats.rejected += 1
            self.log.append(dict(t=now, event="packet_rejected", code=e.code))
            raise
        self.packet = packet
        self.stats.packets += 1
        self.log.append(dict(t=now, event="packet_accepted", obs=packet.observation_id))

    def invalidate(self, reason: str, now: float):
        if self.packet is not None:
            self.log.append(dict(t=now, event="packet_invalidated", reason=reason))
        self.packet = None

    def local_inputs(self, obs):
        """Only proprio/FK node features and declared local sensors — no scene/task tokens."""
        from rrp.learning.packed import local_sensors
        pi = self.f(obs)
        return pi, local_sensors(pi)

    @torch.no_grad()
    def tick(self, session, controller_version: str) -> NativeCommand | None:
        now = float(session.data.time)
        # runtime validation (allowed): a graph edit invalidates the packet -> fallback hold until replanned.
        if self.packet is not None and session.runtime.graph_version != self.packet.graph_version:
            self.invalidate("graph_edit", now)
        obs = session.observe()
        if self.packet is None or now > self.packet.valid_until:
            if self.packet is not None:
                self.invalidate("expired", now)
            self.stats.fallback_holds += 1
            return None                                   # declared fallback: controller holds measured state
        pi, loc = self.local_inputs(obs)
        dev = self.device
        z = torch.from_numpy(np.asarray(self.packet.z, np.float32))[None].to(dev)
        zm = torch.tensor([self.packet.assembly_mask], device=dev)
        kt = torch.tensor(self.packet.knot_times, dtype=torch.float32, device=dev)
        ph = torch.tensor([now - self.packet.valid_from], dtype=torch.float32, device=dev)
        nf = torch.from_numpy(pi.act_node_feats.astype(np.float32))[None].to(dev)
        nm = torch.ones(1, nf.shape[1], dtype=torch.bool, device=dev)
        lc = torch.from_numpy(loc)[None].to(dev)
        a = self.net(z, zm, kt, ph, nf, nm, lc)[0].cpu().numpy()
        groups = self.f.aspace.denormalize(np.clip(a, -6, 6)[None], pi.q0)[0]
        self.stats.ticks += 1
        return NativeCommand(controller_version=controller_version, groups=groups, source="learned")


def weights_digest(state_dict: dict) -> str:
    """sha256 over parameter/buffer names and raw bytes (dtype-exact), sorted by name."""
    h = hashlib.sha256()
    for k in sorted(state_dict):
        t = state_dict[k].detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()[:12]


def bundle_versions(config_version: str, encoder_state: dict, realizer_state: dict) -> tuple[str, str]:
    """Compatibility IDs of a FROZEN bundle: the latent space is defined by the encoder weights (not only its
    config); system-0 compatibility additionally by the realizer weights. Retraining with the same config yields
    different IDs, so stale packets/generators are rejected instead of silently reinterpreted."""
    lsv = f"{config_version}-w{weights_digest(encoder_state)}"
    return lsv, f"rz-{lsv}-r{weights_digest(realizer_state)}-{REALIZER_RECURRENT_STATE}"


def is_fingerprinted(latent_space_version: str) -> bool:
    return "-w" in latent_space_version
