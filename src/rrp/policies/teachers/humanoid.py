"""W13 humanoid task command layers (label scripted_teacher, PRIVILEGED truth) and the `rl_expert` policy.

StepsHeadingTeacher (h_steps): walk +x at 0.6 vx_max, steering yaw and lateral offset to 0 -- the same law as the GPU expert
env (rrp.envs.warp.task_env.WarpStepsEnv._command). GapTeacher is the h_gap counterpart.

`rl_expert` (POLICIES key, `make_rl_expert(arg="<body>:<version>")`, architecture 14.3): the joint-level expert is a registered
LearnedTracker actor (rrp.envs.mujoco.legged_tracker.TRACKERS), loaded by the session (`make_legged_env(..., tracker=spec)`),
under the scripted command layer of the task. Its source label states what the ACTOR consumes: `learned:rl_expert:<sha>` when
its extra inputs are public (none, or the D-146 terrain scan), `privileged_teacher:rl_expert:<sha>` when they are task truth.
The command layer reads the base pose truth either way, so the policy still requires the env's privileged truth.
"""
from __future__ import annotations

import math

import numpy as np

from rrp.core.action import NativeCommand
from rrp.core.provenance import source_label
from rrp.policies.base import Act, PolicyInfo, Requirements


class StepsHeadingTeacher:
    source = "scripted_teacher"          # the scripted command layer (contract literal)
    privileged = True

    def __init__(self, session):
        self.s = session
        L = session.robots[0].meta["legged"]
        self.vx = 0.6 * L["command_ranges"]["vx"][1]
        self.L = float(session.scenario.meta["L"])
        self.done = False

    def command_values(self) -> np.ndarray:
        rt = self.s.runtime
        if rt.succeeded() or any(i.status == "failed" for i in rt.instances.values()):
            self.done = True
            return np.zeros(3)
        x, y, yaw = self.s.base_pose_truth()            # PRIVILEGED
        tgt = 0.5 * math.atan(-y / self.L)
        head = math.atan2(math.sin(tgt - yaw), math.cos(tgt - yaw))
        return np.array([self.vx, 0.0, float(np.clip(1.5 * head, -0.5, 0.5))])

    def act(self) -> NativeCommand:
        return NativeCommand(controller_version=self.s.controller_version(), groups={"base_velocity": self.command_values().tolist()},
                             source="scripted_teacher")


class GapTeacher:
    """h_gap_sidestep command layer (scripted_teacher, PRIVILEGED truth): the WarpGapEnv._command law. Phase 1: keep yaw 0,
    walk +x at 0.5 vx_max and sidestep onto the gap centre; phase 2 (0.5 L past the wall): stop and turn to psi_f."""
    source = "scripted_teacher"
    privileged = True

    def __init__(self, session):
        self.s = session
        self.r = session.robots[0].meta["legged"]["command_ranges"]
        self.m = session.scenario.meta
        self.phase2 = 0.0

    def command_values(self) -> np.ndarray:
        x, y, yaw = self.s.base_pose_truth()
        wrap = lambda a: math.atan2(math.sin(a), math.cos(a))
        if x > self.m["wall_x"] + 0.5 * self.m["L"]:
            self.phase2 = 1.0
        if self.phase2:
            e = wrap(self.m["psi_f"] - yaw)
            return np.array([0.0, 0.0, 0.0 if abs(e) < 0.1 else float(np.clip(1.5 * e, -0.5, 0.5))])
        vy = float(np.clip(1.5 * math.cos(yaw) * (self.m["y_c"] - y), self.r["vy"][0], self.r["vy"][1]))
        return np.array([0.5 * self.r["vx"][1], vy, float(np.clip(1.5 * wrap(-yaw), -0.5, 0.5))])

    def act(self) -> NativeCommand:
        return NativeCommand(controller_version=self.s.controller_version(), groups={"base_velocity": self.command_values().tolist()},
                             source="scripted_teacher")


COMMAND_LAYERS = {"h_steps": StepsHeadingTeacher, "h_gap": GapTeacher}


class RLExpertPolicy:
    """A registered learned expert under the task's scripted command layer. The env must have been built with the same
    registry actor (`tracker=<spec>`); reset() checks the loaded actor's sha256 against the registry entry."""

    def __init__(self, entry, sha: str):
        self.entry, self.sha = entry, sha
        public = entry.extra_obs in ("none", "terrain_scan")
        self.label = source_label("learned" if public else "privileged_teacher", f"rl_expert:{sha[:12]}")
        caps = frozenset({"terrain_scan"} if entry.extra_obs == "terrain_scan" else ())
        self.info = PolicyInfo("rl_expert", "learned" if public else "privileged_teacher", sha,
                               Requirements(frozenset({"base_velocity"}), tasks=frozenset(COMMAND_LAYERS), privileged=True,
                                            env_capabilities=caps))
        self.teachers, self.layers = [], []

    def reset(self, spec, task, seeds, *, envs=None):
        from rrp.envs.mujoco.legged_tracker import TrackerMismatch
        name = getattr(task, "name", task)
        if name not in COMMAND_LAYERS:
            raise KeyError(f"rl_expert has no command layer for task {name!r}; has {sorted(COMMAND_LAYERS)}")
        for e in envs:
            tr = e.tracker
            if getattr(tr, "sha256", None) != self.sha:
                raise TrackerMismatch(f"rl_expert {self.entry.spec}: the env runs {getattr(tr, 'version', tr)!r} "
                                      f"(sha {str(getattr(tr, 'sha256', None))[:12]}), not sha {self.sha[:12]}")
            if self.entry.extra_obs == "privileged" and tr.extra_fn is None:
                raise TrackerMismatch(f"rl_expert {self.entry.spec} takes privileged task inputs the env does not supply")
        self.teachers = [COMMAND_LAYERS[name](e) for e in envs]
        self.layers = [f"scripted_teacher:{name}"] * len(self.teachers)

    def act(self, obs):
        return {i: Act(self.teachers[i].act(), info=dict(source_label=self.label, command_layer=self.layers[i]))
                for i in obs}


def make_rl_expert(*, arg: str | None = None, tracker: str | None = None) -> RLExpertPolicy:
    """arg / tracker: "<body>:<version>" of a registry actor (make_policy("rl_expert:<body>:<version>"))."""
    import hashlib
    from rrp.envs.mujoco.legged_tracker import get_entry
    spec = tracker or arg
    if not spec:
        raise ValueError("rl_expert needs a tracker spec '<body>:<version>'")
    e = get_entry(spec)
    sha = hashlib.sha256(e.actor.read_bytes()).hexdigest()
    if e.sha256 is not None and e.sha256 != sha:
        from rrp.envs.mujoco.legged_tracker import TrackerMismatch
        raise TrackerMismatch(f"tracker {spec} sha256 {sha[:12]} does not match its registry pin {e.sha256[:12]}")
    return RLExpertPolicy(e, sha)
