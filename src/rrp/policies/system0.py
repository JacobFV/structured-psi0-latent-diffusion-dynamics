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

import numpy as np
import torch
import torch.nn as nn

from rrp.core.action import NativeCommand
from rrp.core.errors import ControllerRejection, StaleActionError
from rrp.core.latent_action import check_packet
from rrp.policies.features.multi import assembly_handles, local_sensors_multi, node_slots
from rrp.policies.nets.attention import MHA, RelBlock
from rrp.policies.nets.batch import NODE_DIM
from rrp.policies.nets.flow import MLP, sinusoidal
from rrp.policies.relations.base import RelCtx, TokenSet, resolve
from rrp.policies.relations.ops import FactorSite
import hashlib

REALIZER_RECURRENT_STATE = "none-v1"


def make_realizer(dz: int, default_layers: int, arch: dict | None = None) -> "LatentRealizer":
    """Realizer from a bundle's optional `realizer_arch` (ladder refits: layers/width/z_norm); default = Stage-A arch."""
    a = arch or {}
    return LatentRealizer(dz, width=a.get("width", 192), layers=a.get("layers", default_layers),
                          z_norm=a.get("z_norm", False))


class LatentRealizer(nn.Module):
    def __init__(self, dz: int, width: int = 192, heads: int = 4, layers: int = 2, z_norm: bool = False,
                 factors=None):
        """factors: run config `factors:` entries for the node>knot routing site (default preset `s0-arm` =
        `route.own_assembly`, D-144 R3). Parameter paths / init order are unchanged from the pre-R3 net: `route`
        (the routing `FactorSite`) owns no parameters and draws nothing from the RNG (mask-form `same`), so its
        placement here does not shift any Linear/LayerNorm init draw."""
        super().__init__()
        D = width
        self.z_norm = z_norm
        if z_norm:              # per-dim standardization of the received z (stats of E's posterior mean on the pack)
            self.register_buffer("z_mean", torch.zeros(dz))
            self.register_buffer("z_std", torch.ones(dz))
        self.node = MLP(NODE_DIM, D)
        self.local = nn.Linear(4, D)
        self.z_in = nn.Linear(dz, D)
        self.dt_in = MLP(D, D)
        self.route = FactorSite(heads, D, "node>knot", resolve(factors, default="s0-arm"), ("assembly_id",))
        self.blocks = nn.ModuleList([RelBlock(D, heads) for _ in range(layers)])
        self.out = nn.Linear(D, 1)
        self.D = D

    def route_bias(self, zmask, node_mask, K, node_asm=None):
        """`route.own_assembly` (preset `s0-arm`, D-144 R3): [B,1,N,K*M] -inf off a node's own packet assembly.
        Shared with `AnchorLatentRealizer` (system0_anchor.py), which has the same node>knot routing."""
        B, M = zmask.shape
        N = node_mask.shape[1]
        if node_asm is None:
            node_asm = torch.zeros(B, N, dtype=torch.long, device=zmask.device)
        knot_asm = torch.arange(M, device=zmask.device).repeat(K)                      # [K*M]
        knot_ids = knot_asm[None, :].expand(B, -1).masked_fill(~zmask.repeat(1, K), -1)  # invalid knots: id -1
        rc = RelCtx(sets={"node": TokenSet("node", node_mask, fields={"assembly_id": node_asm[..., None]}),
                          "knot": TokenSet("knot", zmask.repeat(1, K), fields={"assembly_id": knot_ids[..., None]})})
        return self.route.bias(rc)                                          # route.own_assembly: -inf off own assembly

    def forward(self, z, zmask, knot_times, phase, node_feats, node_mask, local, node_asm=None):
        """z [B,K,M,dz]; knot_times [K] (s); phase [B] (s since valid_from); node_feats [B,N,F]; local [B,4] or,
        for multi-assembly bodies, per node [B,N,4] (each node sees its own assembly's touch/width);
        node_asm [B,N] packet assembly index per node (default 0). Returns normalized 1-step actions [B,N]."""
        B, K, M, _ = z.shape
        N = node_feats.shape[1]
        rel_t = knot_times[None, :] - phase[:, None]                                  # [B,K]
        if self.z_norm:
            z = (z - self.z_mean) / self.z_std
        kt = self.z_in(z) + self.dt_in(sinusoidal(rel_t, self.D))[:, :, None]         # [B,K,M,D]
        kt = kt.reshape(B, K * M, -1)
        bias = self.route_bias(zmask, node_mask, K, node_asm)
        loc = self.local(local)
        x = self.node(node_feats) + (loc if local.dim() == 3 else loc[:, None])     # local [B,4] or per node [B,N,4]
        for L in self.blocks:
            x = L(x, kt, bias_x=bias, q_mask=node_mask)
        return self.out(x).squeeze(-1) * node_mask


from rrp.core.system0 import System0Base


Q_COL, QD_COL, ANCHOR_COL = 26, 27, 28     # node-feature layout: normalized joint position; (formerly prev-action, bug B-1) column


def realizer_node_feats(s0, pi) -> np.ndarray:
    """Node features handed to the realizer. For anchored realizers (`net.anchor`, ladder track), column 28 carries the
    joint displacement since the packet's anchor state (the measured state at the first tick of this packet, i.e. at
    valid_from), in the same normalized-position units as column 26; this is local proprio memory of system 0 (declared),
    no scene/task information. Otherwise column 28 is left as the featurizer wrote it (0 since D-021)."""
    nf = pi.act_node_feats.astype(np.float32)
    if getattr(s0.net, "drop_qd", False):         # realizer trained without the joint-velocity input (ladder sprint)
        nf = nf.copy()
        nf[:, QD_COL] = 0
    if not getattr(s0.net, "anchor", False):
        return nf
    if getattr(s0, "_anchor_for", None) is not s0.packet:
        s0._anchor_for, s0._anchor = s0.packet, nf[:, Q_COL].copy()
    nf = nf.copy()
    nf[:, ANCHOR_COL] = nf[:, Q_COL] - s0._anchor
    return nf


class LatentSystem0(System0Base):
    """Runtime wrapper: holds the current packet, realizes it every tick from fresh local state.
    receive / invalidate / stats: rrp.contracts.system0.System0Base (the shared acceptance protocol)."""

    def __init__(self, realizer: LatentRealizer, featurizer, *, latent_space_version: str,
                 realizer_compat_version: str, device="cpu", fallback: str = "hold_measured"):
        self.net = realizer.eval()
        self.f = featurizer
        # the realizer's OWN frozen-bundle fingerprint (set by load_representation) is authoritative; the
        # caller-supplied IDs are only used for legacy realizers without one
        lsv, rcv = getattr(realizer, "bundle_versions", None) or (latent_space_version, realizer_compat_version)
        super().__init__(latent_space_version=lsv, realizer_compat_version=rcv, fallback=fallback)
        self.device = device
        self.blender = None          # D-126 #7 chunk blending (rrp.controllers.chunk_blend); None = historical ticks

    def configure_blend(self, mode: str = "none", ticks: int = 4, decay: float = 0.0):
        """Overlapping-packet blending (default none). Set before the first packet is received."""
        from rrp.policies.chunk_blend import BlendConfig, PacketBlender
        cfg = BlendConfig(mode, ticks, decay)
        self.blender = PacketBlender(cfg) if cfg.on else None
        return self

    def receive(self, packet, *, now: float, graph_version: int | None = None):
        super().receive(packet, now=now, graph_version=graph_version)
        if self.blender is not None:
            self.blender.accepted(packet, now)

    def invalidate(self, reason: str, now: float):
        super().invalidate(reason, now)
        if self.blender is not None:        # never blend across an invalidation (graph edit, expiry)
            self.blender.clear()

    @torch.no_grad()
    def realize_packet(self, packet, now: float, nf, nm, lc) -> np.ndarray:
        """Normalized realizer output for `packet` at this tick's inputs (nf, nm, lc: batch-1 tensors) and its own phase."""
        dev = self.device
        z = torch.from_numpy(np.asarray(packet.z, np.float32))[None].to(dev)
        zm = torch.tensor([packet.assembly_mask], device=dev)
        kt = torch.tensor(packet.knot_times, dtype=torch.float32, device=dev)
        ph = torch.tensor([now - packet.valid_from], dtype=torch.float32, device=dev)
        return self.net(z, zm, kt, ph, nf, nm, lc)[0].cpu().numpy()

    @property
    def robot_spec_hash(self) -> str:
        return self.f.spec.spec_hash

    def local_inputs(self, obs):
        """Only proprio/FK node features and declared local sensors — no scene/task tokens."""
        from rrp.policies.features.derived import local_sensors
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
        nf = torch.from_numpy(realizer_node_feats(self, pi))[None].to(dev)
        nm = torch.ones(1, nf.shape[1], dtype=torch.bool, device=dev)
        lc = torch.from_numpy(loc)[None].to(dev)
        a = self.net(z, zm, kt, ph, nf, nm, lc)[0].cpu().numpy()
        if self.blender is not None:
            a = self.blender.blend(a, now, session.dt, lambda p_: self.realize_packet(p_, now, nf, nm, lc))
        groups = self.f.aspace.denormalize(np.clip(a, -6, 6)[None], pi.q0)[0]
        self.stats.ticks += 1
        return NativeCommand(controller_version=controller_version, groups=groups, source="learned")


from rrp.core.provenance import weights_digest  # noqa: E402,F401  (moved; same algorithm)


def bundle_versions(config_version: str, encoder_state: dict, realizer_state: dict) -> tuple[str, str]:
    """Compatibility IDs of a FROZEN bundle: the latent space is defined by the encoder weights (not only its
    config); system-0 compatibility additionally by the realizer weights. Retraining with the same config yields
    different IDs, so stale packets/generators are rejected instead of silently reinterpreted."""
    lsv = f"{config_version}-w{weights_digest(encoder_state)}"
    return lsv, f"rz-{lsv}-r{weights_digest(realizer_state)}-{REALIZER_RECURRENT_STATE}"


def is_fingerprinted(latent_space_version: str) -> bool:
    return "-w" in latent_space_version


# W4: moved unchanged from rrp.learning.latent_grpo (re-exported there); batched system-0 ticks.
@torch.no_grad()
def batched_ticks(s0s, sessions) -> list:
    """LatentSystem0.tick for several sessions with ONE realizer forward (same inputs/rules per session: graph-edit
    invalidation, expiry fallback hold, fresh proprio/local sensors every tick). Returns NativeCommand|None each."""
    from rrp.core.action import NativeCommand
    cmds, work = [None] * len(sessions), []
    for i, (s0, s) in enumerate(zip(s0s, sessions)):
        now = float(s.data.time)
        if s0.packet is not None and s.runtime.graph_version != s0.packet.graph_version:
            s0.invalidate("graph_edit", now)
        obs = s.observe()
        if s0.packet is None or now > s0.packet.valid_until:
            if s0.packet is not None:
                s0.invalidate("expired", now)
            s0.stats.fallback_holds += 1
            continue
        pi, loc = s0.local_inputs(obs)
        work.append((i, now, pi, loc))
    if not work:
        return cmds
    s0 = s0s[work[0][0]]
    dev, net = s0.device, s0.net
    B = len(work)
    Nmax = max(w[2].act_node_feats.shape[0] for w in work)
    F = work[0][2].act_node_feats.shape[1]
    zs = [np.asarray(s0s[i].packet.z, np.float32) for i, *_ in work]
    Kk, Mmax, dz = zs[0].shape[0], max(z.shape[1] for z in zs), zs[0].shape[2]
    z = np.zeros((B, Kk, Mmax, dz), np.float32); zm = np.zeros((B, Mmax), bool)
    nf = np.zeros((B, Nmax, F), np.float32); nm = np.zeros((B, Nmax), bool)
    for b, (i, now, pi, loc) in enumerate(work):
        z[b, :, :zs[b].shape[1]] = zs[b]; zm[b] = False; zm[b, :zs[b].shape[1]] = s0s[i].packet.assembly_mask
        from rrp.policies.system0 import realizer_node_feats
        n = pi.act_node_feats.shape[0]; nf[b, :n] = realizer_node_feats(s0s[i], pi); nm[b, :n] = True
    kt = torch.tensor(s0s[work[0][0]].packet.knot_times, dtype=torch.float32, device=dev)
    ph = torch.tensor([now - s0s[i].packet.valid_from for i, now, _, _ in work], dtype=torch.float32, device=dev)
    lc = torch.from_numpy(np.stack([w[3] for w in work]).astype(np.float32)).to(dev)
    a = net(torch.from_numpy(z).to(dev), torch.from_numpy(zm).to(dev), kt, ph, torch.from_numpy(nf).to(dev),
            torch.from_numpy(nm).to(dev), lc).cpu().numpy()
    for b, (i, now, pi, loc) in enumerate(work):
        n = pi.act_node_feats.shape[0]
        ab = a[b, :n]
        if getattr(s0s[i], "blender", None) is not None:          # D-126 #7 (default None: unchanged)
            s0b = s0s[i]
            nf1 = torch.from_numpy(nf[b:b + 1, :n]).to(dev)
            nm1 = torch.ones(1, n, dtype=torch.bool, device=dev)
            lc1 = lc[b:b + 1]
            ab = s0b.blender.blend(ab, now, sessions[i].dt,
                                   lambda p_, s0b=s0b, nf1=nf1, nm1=nm1, lc1=lc1, now=now:
                                   s0b.realize_packet(p_, now, nf1, nm1, lc1))
        groups = s0s[i].f.aspace.denormalize(np.clip(ab, -6, 6)[None], pi.q0)[0]
        s0s[i].stats.ticks += 1
        cmds[i] = NativeCommand(controller_version=sessions[i].controller_version(), groups=groups, source="learned")
    return cmds


class DualLatentSystem0(LatentSystem0):
    """System 0 for a multi-robot scene: per-node slot routing + per-assembly local sensors."""

    def receive(self, packet, *, now, graph_version=None):
        try:
            check_packet(packet, latent_space_version=self.lsv, realizer_compat_version=self.rcv,
                         robot_spec_hash=self.f.spec_hash, now=now, graph_version=graph_version,
                         owned_assemblies={h.handle for h in assembly_handles(self.f, len(packet.assemblies))[0]})
        except (ControllerRejection, StaleActionError) as e:
            self.stats.rejected += 1
            self.log.append(dict(t=now, event="packet_rejected", code=getattr(e, "code", "stale")))
            raise
        self.packet = packet
        self.stats.packets += 1
        self.log.append(dict(t=now, event="packet_accepted", obs=packet.observation_id))

    @torch.no_grad()
    def tick(self, session, controller_version=None):
        now = float(session.data.time)
        if self.packet is not None and session.runtime.graph_version != self.packet.graph_version:
            self.invalidate("graph_edit", now)
        if self.packet is None or now > self.packet.valid_until:
            if self.packet is not None:
                self.invalidate("expired", now)
            self.stats.fallback_holds += 1
            return None
        pi = self.f(session.observe())
        M = len(self.packet.assemblies)
        na = node_slots(self.f, pi, M)
        loc = local_sensors_multi(pi, M)[na]                               # [N,4] own assembly's sensors
        dev = self.device
        z = torch.from_numpy(np.asarray(self.packet.z, np.float32))[None].to(dev)
        zm = torch.tensor([self.packet.assembly_mask], device=dev)
        kt = torch.tensor(self.packet.knot_times, dtype=torch.float32, device=dev)
        ph = torch.tensor([now - self.packet.valid_from], dtype=torch.float32, device=dev)
        nf = torch.from_numpy(realizer_node_feats(self, pi))[None].to(dev)    # anchored realizers: col 28 (ladder)
        nm = torch.ones(1, nf.shape[1], dtype=torch.bool, device=dev)
        a = self.net(z, zm, kt, ph, nf, nm, torch.from_numpy(loc)[None].to(dev),
                     node_asm=torch.from_numpy(na)[None].to(dev))[0].cpu().numpy()
        flat = self.f.aspace.denormalize(np.clip(a, -6, 6)[None], pi.q0)[0]
        self.stats.ticks += 1
        return session.command_from_flat(flat, "learned")
