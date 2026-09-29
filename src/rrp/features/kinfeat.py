"""`kinfeat` (armdiv D-137 ablation flag; default OFF = legacy features, byte-identical).

With $RRP_KINFEAT=v1:
  (a) the static joint-axis columns of every action node (node feature cols 2:5, the joint's LOCAL axis in the legacy
      features) become the joint axis in the robot BASE frame at the robot's declared home pose, so the static
      morphology describes the kinematic chain (D-136 diagnosis: the local axis is z for every menagerie hinge);
  (b) the dynamic world-axis column is rotated into the base frame like anchor/jp/jr (a no-op at mount yaw 0, which is
      every scene so far).
The same replacement is applied at load time to packed rows (by robot key; idempotent), so one pack serves both the
flagged and unflagged lineages. Passive (mimic) joint tokens keep their local axes.
"""
from __future__ import annotations

import os
from functools import lru_cache

import mujoco
import numpy as np

ENV = "RRP_KINFEAT"
VERSION = "kinfeat_v1"
AXIS_COLS = slice(2, 5)


def enabled() -> bool:
    v = os.environ.get(ENV, "").strip().lower()
    if v in ("", "0", "off", "none"):
        return False
    if v in ("1", "v1", VERSION):
        return True
    raise ValueError(f"${ENV}={v!r}: expected v1 or unset")


def home_axes(model: mujoco.MjModel, node_joint_names: list[str], meta: dict, arm_joint_names: list[str],
              base_yaw: float = 0.0) -> np.ndarray:
    """[N, 3] base-frame axes of the node joints with the arm at meta['home'] (other joints at qpos0)."""
    d = mujoco.MjData(model)
    d.qpos[:] = model.qpos0
    for jn, v in zip(arm_joint_names, meta.get("home") or []):
        d.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)]] = v
    mujoco.mj_kinematics(model, d)
    c, s = np.cos(-base_yaw), np.sin(-base_yaw)
    Rb = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    out = [Rb @ d.xaxis[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in node_joint_names]
    return np.asarray(out, np.float32)


@lru_cache(maxsize=None)
def table_for_robot(robot_key: str) -> np.ndarray:
    """[N, 3] node axes of a workbench robot (the order the Featurizer uses: command groups, arm first)."""
    from rrp.bodies.catalog import workbench_robots
    from rrp.envs.scenario import BUILDERS
    from rrp.features.featurizer import featurizer_for
    from rrp.envs.native import Session
    prev = os.environ.get(ENV)
    os.environ[ENV] = "v1"
    try:
        s = Session(BUILDERS["pick_place"](workbench_robots()[robot_key](), 0, n_distractors=0), seed=0)
        return featurizer_for(s).node_static[:, AXIS_COLS].copy()
    finally:
        if prev is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = prev


def apply_rows(nodes: np.ndarray, robot_id: np.ndarray, id_to_key: dict[int, str], n_nodes: np.ndarray) -> np.ndarray:
    """In-place replacement of the axis columns of packed node rows [B, N, F] (rows of robots in id_to_key)."""
    for rid in np.unique(robot_id):
        tab = table_for_robot(id_to_key[int(rid)])
        rows = np.nonzero(robot_id == rid)[0]
        n = tab.shape[0]
        if (n_nodes[rows] != n).any():
            raise ValueError(f"kinfeat: node count mismatch for {id_to_key[int(rid)]}")
        nodes[rows[:, None], np.arange(n)[None, :], AXIS_COLS] = tab[None]
    return nodes
