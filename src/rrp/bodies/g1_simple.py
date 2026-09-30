"""Body `g1_simple`: Unitree G1 (+ 3-finger Dex3 hands) as seen through the Ψ₀ / SIMPLE action interface (W10, D-098;
moved from psi1z `body_g1.py`, D-140).

Psi0's SIMPLE policies emit a 36-dim command per 20 ms tick (chunk of 30) that a decoupled whole-body controller
(GR00T WBC ONNX walk/balance policies) realizes on the legs:

    [0:3]   left thumb 0..2        [3:5] left middle 0..1     [5:7] left index 0..1
    [7:14]  right thumb 0..2, right index 0..1, right middle 0..1
    [14:21] left arm (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw)
    [21:28] right arm
    [28:30] waist roll, pitch      [30] waist yaw
    [31]    base height (m)        [32] vx  [33] vy (m/s, base frame)   [34] turning flag   [35] target yaw (world, rad)

Source of truth for the order: SIMPLE `scripts/postprocess_psi0.py` STATE_SLICES/ACTION_SLICES (the data generator).
The 32 real state dims are the same first 28 joints, then the previously commanded torso rpy (3) and base height (1).

This module is numpy-only: it gives per-dimension morphology tokens (assembly, kind, side, chain depth, fingertip flag,
mirror partner) and the relation structure (kinematic parent/child, same assembly, left/right mirror) that the
structured Ψ₀ variant (`rrp.policies.psi0.nets`) consumes. Legs are not commanded joint-by-joint through this interface; they appear as the
`base` assembly (height + planar velocity + heading), realized by the WBC.
"""
from __future__ import annotations

import numpy as np

# assemblies (M = 6); order is part of the packet contract
ASSEMBLIES = ("left_hand", "right_hand", "left_arm", "right_arm", "torso", "base")
ASM_INDEX = {n: i for i, n in enumerate(ASSEMBLIES)}
MANIPULATOR_ASSEMBLIES = ("left_hand", "right_hand")

# kinds of command dimension
KINDS = ("finger_joint", "arm_joint", "waist_joint", "height", "planar_velocity", "turn_flag", "heading")

# (name, G1 joint name or None, assembly, kind, parent dim or -1, is_fingertip_link)
_DIMS: list[tuple[str, str | None, str, str, int, bool]] = [
    ("l_thumb0", "left_hand_thumb_0_joint", "left_hand", "finger_joint", 20, False),
    ("l_thumb1", "left_hand_thumb_1_joint", "left_hand", "finger_joint", 0, False),
    ("l_thumb2", "left_hand_thumb_2_joint", "left_hand", "finger_joint", 1, True),
    ("l_middle0", "left_hand_middle_0_joint", "left_hand", "finger_joint", 20, False),
    ("l_middle1", "left_hand_middle_1_joint", "left_hand", "finger_joint", 3, True),
    ("l_index0", "left_hand_index_0_joint", "left_hand", "finger_joint", 20, False),
    ("l_index1", "left_hand_index_1_joint", "left_hand", "finger_joint", 5, True),
    ("r_thumb0", "right_hand_thumb_0_joint", "right_hand", "finger_joint", 27, False),
    ("r_thumb1", "right_hand_thumb_1_joint", "right_hand", "finger_joint", 7, False),
    ("r_thumb2", "right_hand_thumb_2_joint", "right_hand", "finger_joint", 8, True),
    ("r_index0", "right_hand_index_0_joint", "right_hand", "finger_joint", 27, False),
    ("r_index1", "right_hand_index_1_joint", "right_hand", "finger_joint", 10, True),
    ("r_middle0", "right_hand_middle_0_joint", "right_hand", "finger_joint", 27, False),
    ("r_middle1", "right_hand_middle_1_joint", "right_hand", "finger_joint", 12, True),
    ("l_shoulder_pitch", "left_shoulder_pitch_joint", "left_arm", "arm_joint", 29, False),
    ("l_shoulder_roll", "left_shoulder_roll_joint", "left_arm", "arm_joint", 14, False),
    ("l_shoulder_yaw", "left_shoulder_yaw_joint", "left_arm", "arm_joint", 15, False),
    ("l_elbow", "left_elbow_joint", "left_arm", "arm_joint", 16, False),
    ("l_wrist_roll", "left_wrist_roll_joint", "left_arm", "arm_joint", 17, False),
    ("l_wrist_pitch", "left_wrist_pitch_joint", "left_arm", "arm_joint", 18, False),
    ("l_wrist_yaw", "left_wrist_yaw_joint", "left_arm", "arm_joint", 19, False),
    ("r_shoulder_pitch", "right_shoulder_pitch_joint", "right_arm", "arm_joint", 29, False),
    ("r_shoulder_roll", "right_shoulder_roll_joint", "right_arm", "arm_joint", 21, False),
    ("r_shoulder_yaw", "right_shoulder_yaw_joint", "right_arm", "arm_joint", 22, False),
    ("r_elbow", "right_elbow_joint", "right_arm", "arm_joint", 23, False),
    ("r_wrist_roll", "right_wrist_roll_joint", "right_arm", "arm_joint", 24, False),
    ("r_wrist_pitch", "right_wrist_pitch_joint", "right_arm", "arm_joint", 25, False),
    ("r_wrist_yaw", "right_wrist_yaw_joint", "right_arm", "arm_joint", 26, False),
    ("waist_roll", "waist_roll_joint", "torso", "waist_joint", 30, False),
    ("waist_pitch", "waist_pitch_joint", "torso", "waist_joint", 28, False),
    ("waist_yaw", "waist_yaw_joint", "torso", "waist_joint", 31, False),
    ("base_height", None, "base", "height", -1, False),
    ("base_vx", None, "base", "planar_velocity", 31, False),
    ("base_vy", None, "base", "planar_velocity", 31, False),
    ("base_turn_flag", None, "base", "turn_flag", 31, False),
    ("base_target_yaw", None, "base", "heading", 31, False),
]
ACTION_DIM = len(_DIMS)          # 36
STATE_DIM = 32                   # real state dims (padded to 36 by Psi0)
DIM_NAMES = tuple(d[0] for d in _DIMS)
JOINT_NAMES = tuple(d[1] for d in _DIMS)
DIM_ASM = np.array([ASM_INDEX[d[2]] for d in _DIMS], dtype=np.int64)
DIM_KIND = np.array([KINDS.index(d[3]) for d in _DIMS], dtype=np.int64)
DIM_PARENT = np.array([d[4] for d in _DIMS], dtype=np.int64)
DIM_FINGERTIP = np.array([d[5] for d in _DIMS], dtype=bool)

# index of each G1 joint in SIMPLE's 43-dim `joint_qpos` observation (for the first 31 dims)
SIMPLE_QPOS_INDEX = np.array([29, 30, 31, 34, 35, 32, 33, 36, 37, 38, 39, 40, 41, 42,
                              15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 13, 14, 12], dtype=np.int64)


def _side(name: str) -> float:
    return -1.0 if name.startswith("l_") else (1.0 if name.startswith("r_") else 0.0)


def _mirror(i: int) -> int:
    n = DIM_NAMES[i]
    if n[:2] in ("l_", "r_"):
        other = ("r_" if n[:2] == "l_" else "l_") + n[2:]
        return DIM_NAMES.index(other)
    return -1


DIM_SIDE = np.array([_side(n) for n in DIM_NAMES], dtype=np.float32)
DIM_MIRROR = np.array([_mirror(i) for i in range(ACTION_DIM)], dtype=np.int64)


def _depth(i: int) -> int:
    d = 0
    while DIM_PARENT[i] >= 0 and d < 64:
        i = int(DIM_PARENT[i])
        d += 1
    return d


DIM_DEPTH = np.array([_depth(i) for i in range(ACTION_DIM)], dtype=np.float32)
MAX_DEPTH = float(DIM_DEPTH.max())

NODE_STATIC_DIM = len(ASSEMBLIES) + len(KINDS) + 1 + 1 + 1   # asm one-hot, kind one-hot, side, depth, fingertip


def node_static() -> np.ndarray:
    """[ACTION_DIM, NODE_STATIC_DIM] morphology token features (no task or state information)."""
    x = np.zeros((ACTION_DIM, NODE_STATIC_DIM), dtype=np.float32)
    x[np.arange(ACTION_DIM), DIM_ASM] = 1.0
    x[np.arange(ACTION_DIM), len(ASSEMBLIES) + DIM_KIND] = 1.0
    o = len(ASSEMBLIES) + len(KINDS)
    x[:, o] = DIM_SIDE
    x[:, o + 1] = DIM_DEPTH / MAX_DEPTH
    x[:, o + 2] = DIM_FINGERTIP.astype(np.float32)
    return x


RELATIONS = ("same_node", "kin_parent", "kin_child", "same_assembly", "mirror")   # edge vocabulary g1-dim-rel-v1 (D-144)


def relation_matrix() -> np.ndarray:
    """[R, ACTION_DIM, ACTION_DIM] boolean relation tensor over command dims (kinematic graph + assemblies)."""
    n = ACTION_DIM
    R = np.zeros((len(RELATIONS), n, n), dtype=bool)
    R[0] = np.eye(n, dtype=bool)
    for i in range(n):
        p = int(DIM_PARENT[i])
        if p >= 0:
            R[1, i, p] = True
            R[2, p, i] = True
        if DIM_MIRROR[i] >= 0:
            R[4, i, DIM_MIRROR[i]] = True
    R[3] = DIM_ASM[:, None] == DIM_ASM[None, :]
    return R


def asm_static() -> np.ndarray:
    """[M, M + 3] assembly tokens: one-hot, side, is_manipulator, is_base."""
    M = len(ASSEMBLIES)
    x = np.zeros((M, M + 3), dtype=np.float32)
    x[np.arange(M), np.arange(M)] = 1.0
    for i, n in enumerate(ASSEMBLIES):
        x[i, M] = -1.0 if n.startswith("left") else (1.0 if n.startswith("right") else 0.0)
        x[i, M + 1] = float(n in MANIPULATOR_ASSEMBLIES)
        x[i, M + 2] = float(n == "base")
    return x


def assembly_dims() -> dict[str, np.ndarray]:
    return {n: np.nonzero(DIM_ASM == i)[0] for i, n in enumerate(ASSEMBLIES)}


# system-0 routing (D-144 R5): which assemblies' knots each assembly's command dims may read directly in the
# structured Ψ₀ Realizer (own assembly + kinematic neighbours). Canonical source for the `route.assembly_reads`
# factor's `params.reads` (`rrp.policies.relations.catalog`) and for `psi0.nets.read_mask` (legacy-shape helper,
# kept for its golden digest); declared once here so neither module duplicates the mapping.
READS: dict[str, tuple[str, ...]] = {
    "left_hand": ("left_hand", "left_arm"),
    "right_hand": ("right_hand", "right_arm"),
    "left_arm": ("left_arm", "left_hand", "torso"),
    "right_arm": ("right_arm", "right_hand", "torso"),
    "torso": ("torso", "base", "left_arm", "right_arm"),
    "base": ("base", "torso"),
}


def reads_table() -> np.ndarray:
    """[M, M] bool: reads_table[a, b] = dims of assembly `a` may cross-attend directly to knots of assembly `b`."""
    M = len(ASSEMBLIES)
    t = np.zeros((M, M), dtype=bool)
    for a, allowed in READS.items():
        for b in allowed:
            t[ASM_INDEX[a], ASM_INDEX[b]] = True
    return t


def spec_hash() -> str:
    """Stable identity of this command-space morphology (dims, joints, assemblies, relations); the G1 itself is simulated
    inside SIMPLE, not compiled by rrp, so there is no full `RobotSpec` for it."""
    import hashlib
    h = hashlib.sha256()
    for a in (node_static(), relation_matrix(), asm_static(), DIM_PARENT, SIMPLE_QPOS_INDEX):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update("|".join(f"{n}:{j}" for n, j in zip(DIM_NAMES, JOINT_NAMES)).encode())
    return "g1_simple:" + h.hexdigest()[:16]
