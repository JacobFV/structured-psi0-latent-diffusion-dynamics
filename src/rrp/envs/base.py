"""The one environment interface (docs/architecture.md section 2).

An environment is a world with one or more bodies. It accepts native commands at declared rates (`EnvSpec.action_spaces`)
and returns public observations; privileged truth, snapshots, chunk execution and rendering exist only when the env
declares the capability. Implementations: `rrp.envs.mujoco` (Session / DualSession / LeggedSession), `rrp.envs.warp`
(batched GPU legged env), `rrp.envs.simple` (Ψ₀ SIMPLE, optional extra), `rrp.envs.computerworld` (optional extra).
numpy / pydantic only: importing this module never imports a simulator.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

import numpy as np
from pydantic import Field

from rrp.core.action import NativeCommand
from rrp.core.base import Strict
from rrp.core.observation import PolicyObservation
from rrp.core.robot import CommandGroup, RobotSpec

# CommandGroup.semantic plus the non-joint kinds of UI bodies (rrp.core.robot keeps the two lists equal)
ActionKind = Literal["joint_position", "joint_velocity", "joint_torque", "gripper", "base_velocity", "wholebody_command",
                     "ee_pose", "cartesian_position", "button", "discrete", "psi0"]
Capability = Literal["privileged_truth", "snapshot", "render", "task_graph", "chunk_executor", "reward", "deterministic",
                     "batched", "images", "language", "object_descriptors", "predicates", "proprio", "vector_obs"]
LEGGED_FAMILIES = frozenset({"quadruped", "biped", "humanoid", "hexapod", "multipod"})


class CapabilityError(RuntimeError):
    """An optional Env/Policy method was called on an implementation that does not declare the capability."""


class ActionSpace(Strict):
    group: str                      # command group name in NativeCommand.groups ("arm", "gripper", "base_velocity", "legs", ...)
    kind: ActionKind
    width: int = Field(ge=1)
    robot: int = 0                  # body index (multi-robot envs: NativeCommand dict keyed by robot)
    rate_hz: float = Field(gt=0)
    low: list[float] | None = None
    high: list[float] | None = None
    units: str = ""
    vocab: list[str] | None = None  # discrete spaces: value i means vocab[i]; -1 = no event

    @classmethod
    def from_group(cls, g: CommandGroup, *, robot: int, rate_hz: float) -> "ActionSpace":
        return cls(group=g.name, kind=g.semantic, width=g.width, robot=robot, rate_hz=rate_hz, low=list(g.lower),
                   high=list(g.upper), units=g.units)


class BodyInfo(Strict):
    robot: int
    family: str                     # RobotSpec.family ("arm", "quadruped", "humanoid", ...) or "g1_simple", "pointer"
    key: str                        # catalog key / name
    robot_spec_hash: str

    @classmethod
    def from_spec(cls, robot: int, spec: RobotSpec, key: str | None = None) -> "BodyInfo":
        return cls(robot=robot, family=spec.family, key=key or spec.name, robot_spec_hash=spec.spec_hash)


class EnvSpec(Strict):
    env_id: str                     # registry key ("mujoco/arm", "mujoco/dual", "mujoco/legged", "warp/legged", "simple", "computerworld")
    backend: Literal["mujoco", "mujoco_warp", "mjx", "isaac_simple", "computerworld"]
    task: str
    bodies: list[BodyInfo]
    batch: int = Field(default=1, ge=1)
    control_hz: float = Field(gt=0)
    action_spaces: list[ActionSpace]
    capabilities: list[Capability]
    frame: dict = Field(default_factory=lambda: {"units": "m", "up": "+z"})
    provenance: dict = Field(default_factory=dict)

    def has(self, cap: str) -> bool:
        return cap in self.capabilities

    def action_kinds(self) -> set[str]:
        return {a.kind for a in self.action_spaces}

    def space(self, group: str, robot: int = 0) -> ActionSpace:
        for a in self.action_spaces:
            if a.group == group and a.robot == robot:
                return a
        raise KeyError(f"{self.env_id}: no action space {group!r} for robot {robot}")


@dataclass
class StepResult:
    observation: Any                 # PolicyObservation (single world) | VectorObservation (batched)
    qpos: np.ndarray | None
    time: float
    rejected: str | None = None      # controller rejection code; never silent
    source: str | None = None
    command: dict | None = None      # groups actually executed for robot 0 (exact replay)
    commands: dict | None = None     # robot index -> groups executed (multi-robot replay)
    reward: Any = None               # capability "reward" only (training envs); privileged, never a policy input


@dataclass
class VectorObservation:
    """Batched public observation (capability "batched"): vec [N, D] with a named layout; privileged state is truth()."""
    vec: Any                         # np.ndarray or torch.Tensor [N, D]
    layout: list[tuple[str, int]]
    time: float
    robot_spec_hashes: list[str]


@dataclass
class BatchCommand:
    """Command for a batched env: group -> [N, width] array/tensor (NativeCommand stays the per-world wire type)."""
    groups: dict[str, Any]
    source: str


Observation = PolicyObservation | VectorObservation


@runtime_checkable
class Env(Protocol):
    """Required methods. Capability-gated (raise CapabilityError when absent): truth() ["privileged_truth"],
    snapshot()/restore(s) ["snapshot"], submit_chunk(chunk, robot=0, execute_prefix=None) ["chunk_executor"],
    render(camera=None, width=, height=) ["render"]."""

    @property
    def spec(self) -> EnvSpec: ...

    def reset(self, seed: int | None = None) -> Observation: ...

    def observe(self) -> Observation: ...

    def step(self, command: NativeCommand | Mapping[int, NativeCommand] | BatchCommand | None) -> StepResult: ...

    def close(self) -> None: ...


# ------------------------------------------------------------------ registry (lazy: "module:factory")
ENVS: dict[str, str] = {
    "mujoco/arm": "rrp.envs.mujoco.session:make_arm_env",
    "mujoco/dual": "rrp.envs.mujoco.dual:make_dual_env",
    "mujoco/legged": "rrp.envs.mujoco.legged:make_legged_env",
    "warp/legged": "rrp.envs.warp.tracker_env:make_warp_env",
    "simple": "rrp.envs.simple:make_env",                   # Ψ₀ migration (architecture.md section 9)
    "computerworld": "rrp.envs.computerworld:make_env",     # ComputerWorld (architecture.md sections 5, 9)
}


def register_env(env_id: str, target: str) -> None:
    """target "module:factory"; factory(task=, body=, seed=, **kw) -> Env. Re-registering a different target raises."""
    if ENVS.get(env_id, target) != target:
        raise ValueError(f"env {env_id} is already registered as {ENVS[env_id]}")
    ENVS[env_id] = target


def make_env(env_id: str, *, task: str, body: str | list[str], seed: int = 0, **kw) -> Env:
    if env_id not in ENVS:
        raise KeyError(f"unknown env {env_id!r}; registered: {sorted(ENVS)}")
    mod, fn = ENVS[env_id].split(":")
    try:
        m = importlib.import_module(mod)
    except ModuleNotFoundError as e:
        if e.name == mod:
            raise NotImplementedError(f"env {env_id!r} is declared but not implemented yet ({mod}; "
                                      "docs/architecture.md section 9)") from e
        raise
    return getattr(m, fn)(task=task, body=body, seed=seed, **kw)
