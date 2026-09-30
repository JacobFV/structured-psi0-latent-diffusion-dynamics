"""`kinfeat` / `feat.base_axes` (D-137 ablation, migrated to a run-scoped resolved value by D-144/R12+):

With base_axes resolved True:
  (a) the static joint-axis columns of every action node (node feature cols 2:5, the joint's LOCAL axis in the legacy
      features) become the joint axis in the robot BASE frame at the robot's declared home pose, so the static
      morphology describes the kinematic chain (D-136 diagnosis: the local axis is z for every menagerie hinge);
  (b) the dynamic world-axis column is rotated into the base frame like anchor/jp/jr (a no-op at mount yaw 0, which is
      every scene so far).
The same replacement is applied at load time to packed rows (by robot key; idempotent), so one pack serves both the
flagged and unflagged lineages. Passive (mimic) joint tokens keep their local axes.

R12/R12c (D-144, docs/relations.md sections 8, 10): `$RRP_KINFEAT` is GONE from `src/` -- no live code path reads
`os.environ` for this flag any more. `Featurizer` / `MultiFeaturizer` take an explicit `base_axes: bool | None`
kwarg (`None` = ambient default via `resolved()`, `True` / `False` pins it regardless of the ambient value).
`harness/pipelines/base.py::apply_run_context(rc)` resolves the run's factor list (`options.kinfeat: v1` or a
`feat.base_axes` factor item) and calls `set_base_axes(...)` at stage entry -- in the stage's own process AND, through
`StageContext.run` -> `child_main`, in every subprocess the stage spawns (`rrp data pack`, `rrp suite ladder`, the
DAgger / edit workers, target_eval, ...): the rendered RunConfig (`<out>/run_context.json`) is the one channel, so a
kinfeat lineage's children featurize -- and load checkpoints -- with the same value (docs/architecture.md 14.2).
`harness/data/packed.py` / `harness/data/latent.py` and `policies/nets/checkpoint.py` read `resolved()` the same
way `enabled()` used to be read. `versions["kinfeat"]` as a standalone checkpoint key is gone too: `resolved()`
folds into `policies/nets/checkpoint.save_checkpoint`'s combined `versions["factors"]` hash instead of its own key.

The only remnant of the old `$RRP_KINFEAT=v1/0/off/...` string vocabulary is `legacy_bool` below, a pure mapping
table used ONLY when reading an old on-disk config/checkpoint value (never `os.environ`) -- e.g. a DAG YAML's
`options: {kinfeat: v1}` stage option, or an old checkpoint's `versions["kinfeat"]` entry.
"""
from __future__ import annotations

from functools import lru_cache

import mujoco
import numpy as np

VERSION = "kinfeat_v1"
FACTOR_NAME = "feat.base_axes"        # the run-config-facing name of this featurizer option
AXIS_COLS = slice(2, 5)

_current: bool | None = None          # process-ambient resolved value, set only by harness.pipelines.base
                                       # (apply_run_context; parent stage and child); None = "unset" (resolved() then False)


def legacy_bool(v) -> bool:
    """Pure mapping of the old `$RRP_KINFEAT` / `options.kinfeat` string vocabulary to a bool. Used only by loaders
    (on-disk configs / checkpoints), never by live code deciding its OWN behaviour."""
    s = ("" if v is None else str(v)).strip().lower()
    if s in ("", "0", "off", "none"):
        return False
    if s in ("1", "v1", VERSION):
        return True
    raise ValueError(f"kinfeat value {v!r}: expected v1 or unset")


def set_base_axes(value: bool | None) -> bool | None:
    """Set the process-ambient resolved value for the duration of a stage; returns the previous value so the
    caller can restore it (matches the old `RRP_KINFEAT` set/restore dance, without `os.environ`)."""
    global _current
    prev = _current
    _current = value
    return prev


def resolved(explicit: bool | None = None) -> bool:
    """The effective base_axes value: `explicit` wins when given; otherwise the ambient value set by
    `set_base_axes` (a stage's `options.kinfeat`); otherwise False (today's default, byte-identical goldens)."""
    if explicit is not None:
        return bool(explicit)
    if _current is not None:
        return bool(_current)
    return False


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
    from rrp.envs.mujoco.scenario import BUILDERS
    from rrp.policies.features.featurizer import featurizer_for
    from rrp.envs.mujoco.session import Session
    s = Session(BUILDERS["pick_place"](workbench_robots()[robot_key](), 0, n_distractors=0), seed=0)
    return featurizer_for(s, base_axes=True).node_static[:, AXIS_COLS].copy()


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
