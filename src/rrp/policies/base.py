"""The one policy interface (docs/architecture.md section 3).

A policy maps a batch of observations (one per parallel episode) to native commands for one control tick. Internals
(chunk planning, system i packets realized by system 0, scripted teachers) are the policy's business; what it emitted is
reported in `Act`. `negotiate` is the capability check the harness runs before any episode: a declined pair is recorded
with its reasons, never silently skipped.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Literal, Mapping, Protocol, Sequence, runtime_checkable

from rrp.core.action import ActionChunk, NativeCommand
from rrp.core.latent_action import LatentActionChunk
from rrp.envs.base import EnvSpec

ObsField = Literal["proprio", "images", "object_descriptors", "predicates", "task_graph", "language", "vector"]
# observation fields -> the env capability that provides them
_OBS_CAP = {"proprio": "proprio", "images": "images", "object_descriptors": "object_descriptors",
            "predicates": "predicates", "task_graph": "task_graph", "language": "language", "vector": "vector_obs"}


@dataclass(frozen=True)
class Requirements:
    action_kinds: frozenset[str]                  # every kind the policy emits must be offered by the env
    groups: frozenset[str] = frozenset()          # required group names; empty = any group of those kinds
    observations: frozenset[str] = frozenset({"proprio"})
    body_families: frozenset[str] | None = None   # None = any family whose action spaces match
    bodies: frozenset[str] | None = None          # checkpoint trained on specific body keys, else None
    tasks: frozenset[str] | None = None           # trained for / scripted for specific tasks, else None
    privileged: bool = False                      # teachers / oracles: needs env.truth() and env internals
    env_capabilities: frozenset[str] = frozenset()
    batched_env: bool = False                     # consumes VectorObservation batches (capability "batched")


@dataclass(frozen=True)
class PolicyInfo:
    name: str                    # registry key ("bc", "latent", "legged_latent", "tracker", "teacher:pick_place", "psi0_direct", ...)
    source: str                  # rrp.core.provenance.Source kind: scripted_teacher | privileged | oracle | learned | bc | random | mock ...
    version: str                 # weights digest / bundle compatibility IDs / teacher version
    requires: Requirements
    variant: str | None = None   # "semfix" | "nosem" | "sem" for the latent family


@dataclass
class Act:
    command: NativeCommand | dict[int, NativeCommand] | None   # applied this tick (None = env hold / queued chunk row)
    packet: LatentActionChunk | None = None     # emitted this tick by system i (after any hook edit)
    chunk: ActionChunk | None = None            # submitted this tick (chunk policies; env capability "chunk_executor")
    info: dict = field(default_factory=dict)    # latencies, rejections, probe readouts, SDE records ...


@runtime_checkable
class Policy(Protocol):
    info: PolicyInfo

    def reset(self, spec: EnvSpec, task, seeds: Sequence[int], *, envs: Sequence | None = None) -> None:
        """Start len(seeds) parallel episodes on `envs` (one per seed). A policy without requires.privileged reads only
        the env's public surface: observe() through its featurizer (the ONLY definition of what a policy may see,
        rrp.policies.features), the body model for public FK, controller/task-graph versions and the clock."""

    def act(self, obs: Mapping[int, object]) -> dict[int, Act]:
        """One control tick for the RUNNING episodes only: obs maps episode index (position in reset's seeds) to its
        observation; per-episode state and random streams are keyed by that index."""


@dataclass(frozen=True)
class Compat:
    ok: bool
    reasons: tuple[str, ...] = ()


def negotiate(info: PolicyInfo, spec: EnvSpec, task=None) -> Compat:
    """Can `info` run on an env with `spec` (and task)? Reasons name what is missing, in the env's own vocabulary."""
    r, why = info.requires, []
    offered = spec.action_kinds()
    missing = sorted(set(r.action_kinds) - offered)
    if missing:
        why.append(f"needs {', '.join(missing)}; env offers {', '.join(sorted(offered))}")
    groups = {a.group for a in spec.action_spaces}
    if r.groups - groups:
        why.append(f"needs command groups {sorted(r.groups - groups)}; env has {sorted(groups)}")
    for o in sorted(r.observations):
        if not spec.has(_OBS_CAP.get(o, o)):
            why.append(f"needs observation {o!r}")
    fams = {b.family for b in spec.bodies}
    if r.body_families is not None and not fams <= set(r.body_families):
        why.append(f"body family {', '.join(sorted(fams))} not in {sorted(r.body_families)}")
    keys = {b.key for b in spec.bodies}
    if r.bodies is not None and not keys <= set(r.bodies):
        why.append(f"trained on {sorted(r.bodies)}; env body {sorted(keys)}")
    tname = getattr(task, "name", task)
    if r.tasks is not None and tname is not None and tname not in r.tasks:
        why.append(f"task {tname!r} not in {sorted(r.tasks)}")
    if task is not None and hasattr(task, "envs") and spec.env_id not in task.envs:
        why.append(f"task {tname!r} does not exist in {spec.env_id}")
    if r.privileged and not spec.has("privileged_truth"):
        why.append("privileged policy; env has no privileged_truth")
    for c in sorted(r.env_capabilities):
        if not spec.has(c):
            why.append(f"needs env capability {c!r}")
    if r.batched_env != spec.has("batched"):
        why.append("needs a batched env" if r.batched_env else "needs a single-world env; env is batched")
    return Compat(not why, tuple(why))


# ------------------------------------------------------------------ registry (lazy: "module:factory")
POLICIES: dict[str, str] = {
    # scripted_teacher, privileged; one key per task (TaskSpec.teacher names it); the factory gets arg = task
    "teacher:pick_place": "rrp.policies.teachers:make_arm_teacher_policy",
    "teacher:support_insert": "rrp.policies.teachers:make_dual_teacher_policy",
    "teacher:handover": "rrp.policies.teachers:make_dual_teacher_policy",
    "teacher:assign_left": "rrp.policies.teachers:make_dual_teacher_policy",
    "teacher:assign_right": "rrp.policies.teachers:make_dual_teacher_policy",
    "teacher:pivot_against_surface": "rrp.policies.teachers:make_dual_teacher_policy",
    "teacher:carry_tray_level": "rrp.policies.teachers:make_dual_teacher_policy",
    "teacher:waypoint_contact": "rrp.policies.teachers:make_waypoint_teacher_policy",
    "teacher:loco_pick": "rrp.policies.teachers:make_loco_pick_teacher_policy",
    "teacher:h_steps": "rrp.policies.teachers:make_steps_teacher_policy",
    "teacher:h_gap": "rrp.policies.teachers:make_gap_teacher_policy",
    "teacher:cw/calc_sum": "rrp.policies.teachers:make_cw_teacher_policy",
    "teacher:cw/open_type": "rrp.policies.teachers:make_cw_teacher_policy",
    "teacher:cw/drag_window": "rrp.policies.teachers:make_cw_teacher_policy",
    "teacher:cw/fill_form": "rrp.policies.teachers:make_cw_teacher_policy",
    "rl_expert": "rrp.policies.teachers.humanoid:make_rl_expert",   # make_policy("rl_expert", tracker="<body>:<version>")
    "rl_expert:*": "rrp.policies.teachers.humanoid:make_rl_expert",  # make_policy("rl_expert:<body>:<version>")
    "bc": "rrp.policies.bc:make_bc",
    "latent": "rrp.policies.latent:make_latent",
    "legged_latent": "rrp.policies.legged:make_legged_latent",
    "legged_bc": "rrp.policies.legged:make_legged_bc",
    "psi0_direct": "rrp.policies.psi0:make_direct",          # Ψ₀ migration (architecture.md section 11 glossary)
    "psi0_structured": "rrp.policies.psi0:make_structured",
    "psi0_replay": "rrp.policies.psi0:make_replay",
    "oracle": "rrp.policies.oracle:make_oracle",                # privileged diagnostic, source oracle
    "pointer_oracle": "rrp.policies.pointer:make_pointer_oracle",  # ComputerWorld: teacher packets -> scripted system 0
    "pointer_latent": "rrp.policies.pointer:make_pointer_latent",  # learned system i -> learned (or scripted eng) system 0
    "pointer_bc": "rrp.policies.pointer:make_pointer_bc",
}


def register_policy(name: str, target: str) -> None:
    if POLICIES.get(name, target) != target:
        raise ValueError(f"policy {name} is already registered as {POLICIES[name]}")
    POLICIES[name] = target


def make_policy(name: str, **kw) -> Policy:
    """name: a registry key. A name "<family>:<arg>" not registered itself resolves to the key "<family>:*" (factory gets
    `arg`); a registered "<family>:<arg>" key (the teachers) gets `arg` too."""
    key = name if name in POLICIES else f"{name.split(':', 1)[0]}:*"
    if key not in POLICIES:
        raise KeyError(f"unknown policy {name!r}; registered: {sorted(POLICIES)}")
    mod, fn = POLICIES[key].split(":")
    try:
        m = importlib.import_module(mod)
    except ModuleNotFoundError as e:
        if e.name == mod:
            raise NotImplementedError(f"policy {name!r} is declared but not implemented yet ({mod})") from e
        raise
    if ":" in name:
        kw["arg"] = name.split(":", 1)[1]
    return getattr(m, fn)(**kw)
