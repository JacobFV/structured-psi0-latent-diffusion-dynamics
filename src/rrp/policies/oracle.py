"""Oracle packet sources (PRIVILEGED diagnostics, source "target_encoder_oracle" / "oracle"): z = E(public context,
demonstrated native chunk a[t:t+H]) with the frozen target encoder E, so system 0 is evaluated on packets that encode a
known-good demonstration. One encoder path (`encode_demos`) for every arm / dual oracle (the ladder's shadow-teacher
and BC look-ahead oracles, the semantic-edit OracleSource); the legged oracle is rrp.policies.legged.OracleShadow.
`make_oracle` puts OraclePacketPolicy behind the Policy interface with the frozen system 0 (LatentStackPolicy).
"""
from __future__ import annotations

import numpy as np
import torch

from rrp.policies.features.featurizer import cached_featurizer
from rrp.policies.packets import arm_packet


@torch.no_grad()
def encode_demos(E, items, H: int, device):
    """items: [(featurizer, PolicyInput at t, demo commands [<= H group dicts])] -> (posterior means z [K, M, dz] per
    item, log-variances). Demos shorter than H are padded with their last row and masked (valid = demonstrated rows);
    actions are stored as in the packed data (fp16)."""
    from rrp.policies.nets.batch import collate_inputs
    from rrp.policies.nets.semantic_latent import assembly_tokens
    feats, A, V = [], [], []
    for f, pi, cmds in items:
        n = len(cmds)
        a = f.aspace.normalize(cmds + [cmds[-1]] * (H - n), pi.q0).astype(np.float16).astype(np.float32)
        v = np.zeros_like(a, bool)
        v[:n] = True
        feats.append(pi); A.append(a); V.append(v)
    b = collate_inputs(feats).to(device)
    N = b.node_feats.shape[1]
    a = np.zeros((len(A), H, N), np.float32); v = np.zeros((len(A), H, N), bool)
    for i in range(len(A)):
        a[i, :, :A[i].shape[1]] = A[i]; v[i, :, :V[i].shape[1]] = V[i]
    af, am, ai = assembly_tokens(b)
    mu, lv = E(b, torch.from_numpy(a).to(device), torch.from_numpy(v).to(device), af, am, ai)
    mu, lv = mu.float().cpu().numpy(), lv.float().cpu().numpy()
    M = [int(am[i].sum()) for i in range(len(items))]
    return [mu[i][:, :M[i]] for i in range(len(items))], [lv[i][:, :M[i]] for i in range(len(items))]


class ShadowTeacher:
    """Per-session scripted teacher (PRIVILEGED). label(s) advances it once at the current state; lookahead(s, H)
    returns the teacher's next H commands executed from the current state, then restores session + teacher."""

    def __init__(self, s, reanchor: bool = False):
        from rrp.policies.teachers.arm import PickPlaceTeacher
        self.t = PickPlaceTeacher(s)
        self.synced = s.step_count
        self.reanchor = reanchor

    def reanchor_now(self, s):
        """Re-anchor the expert's internal TCP reference (and IK seed) to the MEASURED arm, keeping its phase: the
        expert then demonstrates a smooth continuation from where the arm actually is (as in clean demonstrations)
        instead of a jump back to its own run-away reference."""
        t = self.t
        t.tcp_cmd = s._fk_site(t.r, t.tcp_site)[0].copy()
        t.q_arm = s.data.qpos[t.r.qadr[:len(t.q_arm)]].copy()

    def catch_up(self, s):
        while self.synced < s.step_count:       # missed ticks (e.g. disturbance_test warmup): advance at current state
            self.t.act()
            self.synced += 1

    def label(self, s):
        self.catch_up(s)
        c = self.t.act()
        self.synced = s.step_count + 1
        return c

    def lookahead(self, s, H: int):
        """The teacher's next H commands from the current state, on a snapshot that is restored. The one session tick
        made outside harness.rollout in the owned files (D-146 RP3): this is a policy-internal look-ahead in the
        policies layer, which may not import harness (tests/unit/test_layering.py); the equivalent harness-layer
        look-ahead is latent_semantic_edits._teacher_demo. Reported to X1 (lint allowlist or a layer-correct home)."""
        self.catch_up(s)
        if self.reanchor:
            self.reanchor_now(s)
        snap, st = s.snapshot(), self.t.state()
        cmds = []
        for _ in range(H):
            c = self.t.act()
            cmds.append(c.groups)
            s.step(c)
            if self.t.done:
                break
        s.restore(snap)
        self.t.load(st)
        return cmds


class BCLookahead:
    """Stateless expert for the oracle route: the H-step chunk a learned BC policy emits at the CURRENT state (no
    stepping, no FSM). Valid off the teacher trajectory, unlike the shadow teacher (see sprint_bc / D-050)."""

    def __init__(self, lp):
        self.lp = lp

    def lookahead(self, s, H: int):
        return self.lookahead_batch([s], H)[0]

    def lookahead_batch(self, sessions, H: int):
        return [[{g.group: np.asarray(g.values[t]).tolist() for g in ch.command_groups} for t in range(min(H, ch.horizon))]
                for ch in self.lp.chunks(sessions)]


class OraclePacketPolicy:
    """ORACLE DIAGNOSTIC: z = E(public context at t, teacher chunk a[t:t+H]) (posterior mean). Same interface as
    LatentPolicy (packets/featurizer/lsv/rcv/calls) so evaluate/disturbance code can drive it."""
    name = "target_encoder_oracle"

    def __init__(self, E, cfg, res, device, validity_s: float = 0.8, reanchor: bool = False):
        self.reanchor = reanchor
        self.E, self.cfg, self.device, self.validity = E, cfg, device, validity_s
        self.lsv, self.rcv = res["latent_space_version"], res["realizer_compat_version"]
        self.shadows: dict[int, ShadowTeacher] = {}
        self.calls = 0

    def featurizer(self, s):
        return cached_featurizer(s)

    def shadow(self, s) -> ShadowTeacher:
        k = id(s)
        if k not in self.shadows:
            self.shadows[k] = ShadowTeacher(s, reanchor=self.reanchor)
        return self.shadows[k]

    @torch.no_grad()
    def encode(self, sessions) -> list[np.ndarray]:
        H = self.cfg.horizon
        items = []
        pre = {}
        bcs = [s for s in sessions if isinstance(self.shadow(s), BCLookahead)]
        if bcs:                                           # one batched BC call for all stateless-expert sessions
            for s, rows in zip(bcs, self.shadow(bcs[0]).lookahead_batch(bcs, H)):
                pre[id(s)] = rows
        self.last_cmds = getattr(self, "last_cmds", {})
        for s in sessions:
            pi = self.featurizer(s)(s.observe())
            cmds = pre[id(s)] if id(s) in pre else self.shadow(s).lookahead(s, H)
            self.last_cmds[id(s)] = cmds
            items.append((self.featurizer(s), pi, cmds))
        zs, self.last_logvar = encode_demos(self.E, items, H, self.device)
        return zs

    def packets(self, sessions):
        zs = self.encode(sessions)
        self.calls += len(sessions)
        return [make_packet(s, self.featurizer(s), z, self.cfg.knot_times, self.lsv, self.rcv, self.validity,
                            source="target_encoder_oracle", policy_version=self.name) for s, z in zip(sessions, zs)]


def make_packet(s, f, z, knot_times, lsv, rcv, validity, *, source, policy_version):
    """Single-robot packet at the session's current observation (W5: delegates to evaluation.packets.arm_packet)."""
    from rrp.policies.packets import arm_packet
    return arm_packet(f, s, s.observe(), z, lsv=lsv, rcv=rcv, knot_times=knot_times, source=source,
                      name=policy_version, validity=validity)


def make_oracle(*, representation: str, device: str = "cpu", replan_ticks: int = 8, reanchor: bool = False,
                name: str = "oracle"):
    """ORACLE DIAGNOSTIC behind the Policy interface: shadow-teacher look-ahead -> E -> packet -> frozen system 0."""
    from rrp.policies.bundles import load_representation
    from rrp.policies.latent import LatentStackPolicy
    lcfg, E, R, _, res = load_representation(representation, device)
    src = OraclePacketPolicy(E, lcfg, res, device, reanchor=reanchor)
    return LatentStackPolicy(src, R, replan_ticks=replan_ticks, device=device, name=name, source="oracle",
                             version=f"oracle:E({representation})", privileged=True)
