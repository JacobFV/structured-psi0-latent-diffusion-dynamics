"""Morphology-conditioned tracker observation `morph_v1` (W13 P1c, D-138): one actor for many humanoids.

Legs are mapped to 14 CANONICAL SLOTS (per side: hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll, toe; left then
right) by actuator-name rules (`slot_of`), with a per-slot SIGN s = sign(axis_world . canonical_axis) at the default pose
(yaw +z, roll +x, pitch/knee/ankle-pitch/toe +y, in the pelvis frame), so a positive slot value always means the same
physical rotation. Missing slots are zero with present = 0; their actions are ignored.

Deployable observation (all public: IMU, encoders, command, clock; morphology is a static, known body description):
    gyro * 0.25 (3), gravity in IMU frame (3), command * CMD_SCALE (3), slot s (q - q0) (14), slot s qdot * 0.05 (14),
    last slot action (14), clock sin/cos (2)                                              -> DYN_DIM = 53
    static context: per slot [present, s(lo - q0), s(hi - q0), effort / (M g L_leg), joint position in the pelvis frame
    / L_leg (3)] (14 x 7 = 98) + body [L_leg, log M, gait period, action_scale, vx_max] (5) -> CTX_DIM = 103
Action: 14 slot actions a; joint target = clip(q0 + action_scale * s * a, lo, hi) for present slots.
"""
from __future__ import annotations

import math
import re

import mujoco
import numpy as np

OBS_FORMAT = "morph_v1"
SLOTS = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll", "toe")
NS = 2 * len(SLOTS)
DYN_DIM = 3 + 3 + 3 + 3 * NS + 2
CTX_DIM = NS * 7 + 5
OBS_DIM = DYN_DIM + CTX_DIM
_AXIS = dict(hip_yaw=(0, 0, 1), hip_roll=(1, 0, 0), hip_pitch=(0, 1, 0), knee=(0, 1, 0), ankle_pitch=(0, 1, 0),
             ankle_roll=(1, 0, 0), toe=(0, 1, 0))
_TALOS = {1: "hip_yaw", 2: "hip_roll", 3: "hip_pitch", 4: "knee", 5: "ankle_pitch", 6: "ankle_roll"}
_BERKELEY = dict(hr="hip_yaw", haa="hip_roll", hfe="hip_pitch", kfe="knee", ffe="ankle_pitch", faa="ankle_roll")


def slot_of(name: str) -> tuple[str, str] | None:
    """(side, slot) for a leg actuator name, or None."""
    n = name.lower()
    m = re.match(r"leg_(left|right)_(\d)_", n)                            # talos
    if m and int(m.group(2)) in _TALOS:
        return m.group(1), _TALOS[int(m.group(2))]
    m = re.match(r"l([lr])_(hr|haa|hfe|kfe|ffe|faa)$", n)                  # berkeley humanoid
    if m:
        return ("left" if m.group(1) == "l" else "right"), _BERKELEY[m.group(2)]
    side = "left" if ("left" in n or n.startswith("l_")) else ("right" if ("right" in n or n.startswith("r_")) else None)
    if side is None:
        return None
    n = n.replace("_", "")
    for slot, pats in (("hip_yaw", ("hipyaw", "hipie")), ("hip_roll", ("hiproll", "hipaa")), ("hip_pitch", ("hippitch", "hipfe")),
                       ("knee", ("knee",)), ("ankle_pitch", ("anklepitch", "ankpitch", "anklepd")),
                       ("ankle_roll", ("ankleroll", "ankroll", "ankleie")), ("toe", ("toe",))):
        if any(p in n for p in pats):
            return side, slot
    if re.search(r"ankle(act|joint|link)?$", n):                             # h1: single (pitch) ankle
        return side, "ankle_pitch"
    return None


class MorphSpec:
    """Slot mapping, signs and static context for one body (compiled model + LeggedBinding)."""

    def __init__(self, model: mujoco.MjModel, b, meta: dict):
        acts = meta["legged"]["policy_actuators"]
        self.idx = np.full(NS, -1)                      # slot -> policy actuator index
        for k, a in enumerate(acts):
            so = slot_of(a)
            if so is None:
                raise ValueError(f"{meta['name']}: no canonical slot for leg actuator {a}")
            j = (0 if so[0] == "left" else len(SLOTS)) + SLOTS.index(so[1])
            if self.idx[j] >= 0:
                raise ValueError(f"{meta['name']}: two actuators map to slot {so}")
            self.idx[j] = k
        self.present = self.idx >= 0
        d = mujoco.MjData(model)
        b.set_default(d)
        mujoco.mj_kinematics(model, d)
        Rp = d.xmat[b.root_bid].reshape(3, 3)
        pp = d.xpos[b.root_bid]
        self.sign = np.ones(NS)
        pos = np.zeros((NS, 3))
        leg_len = float(b.nominal_height())
        M = float(model.body_subtreemass[b.root_bid])
        for j in np.nonzero(self.present)[0]:
            k = self.idx[j]
            jid = model.actuator_trnid[b.pol_act[k], 0]
            ax = Rp.T @ (d.xmat[model.jnt_bodyid[jid]].reshape(3, 3) @ model.jnt_axis[jid])
            s = float(np.dot(ax, _AXIS[SLOTS[j % len(SLOTS)]]))
            self.sign[j] = 1.0 if s >= 0 else -1.0
            pos[j] = Rp.T @ (d.xanchor[jid] - pp) / leg_len
        ctx = np.zeros((NS, 7))
        for j in np.nonzero(self.present)[0]:
            k, s = self.idx[j], self.sign[j]
            lo, hi = s * (b.lo[k] - b.q0[k]), s * (b.hi[k] - b.q0[k])
            ctx[j] = [1.0, min(lo, hi), max(lo, hi), b.effort[k] / (M * 9.81 * leg_len), *pos[j]]
        vx = float(b.cmd_ranges["vx"][1])
        self.ctx = np.concatenate([ctx.ravel(), [leg_len, math.log(M), float(b.period), float(b.action_scale), vx]]).astype(np.float32)
        self.q0 = b.q0
        self.n = b.n

    def to_slots(self, x_native: np.ndarray) -> np.ndarray:
        out = np.zeros(NS)
        out[self.present] = self.sign[self.present] * x_native[self.idx[self.present]]
        return out

    def from_slots(self, a_slot: np.ndarray) -> np.ndarray:
        a = np.zeros(self.n)
        a[self.idx[self.present]] = self.sign[self.present] * a_slot[self.present]
        return a

    def obs(self, b, d: mujoco.MjData, cmd, last_slot_action, phase, clock_gate: bool = False) -> np.ndarray:
        """numpy deployment observation (morph_v1) from a C-MuJoCo MjData."""
        from rrp.envs.mujoco.legged_core import CMD_SCALE, clock_features, quat_rotate_inv
        quat, gyro = b.imu(d)
        g = quat_rotate_inv(quat, np.array([0, 0, -1.0]))
        return np.concatenate([gyro * 0.25, g, np.asarray(cmd) * CMD_SCALE, self.to_slots(d.qpos[b.pol_qadr] - self.q0),
                               self.to_slots(d.qvel[b.pol_dadr] * 0.05), last_slot_action,
                               clock_features(phase, cmd, clock_gate), self.ctx]).astype(np.float32)
