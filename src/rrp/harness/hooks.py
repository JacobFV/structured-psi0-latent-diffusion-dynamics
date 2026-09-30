"""The one hooks module, beside `rollout` (docs/architecture.md sections 6 and 14.1): every rollout hook of the former
per-family eval loops, the bench-task / hold-policy helpers and the two name tables.

Feasibility (pre-episode teacher layout filter -> outcome "infeasible"), session records (sim time, task-event statuses),
post-success settling (dual arm), object displacement, and the ladder's per-tick machinery (object shift,
executed-command log, previous-command feature, frame callback). `HOOKS` names the hook constructors;
`TASK_HOOKS` names the default compositions a `TaskSpec.hooks` entry refers to (the env-id prefix each applies to and a
factory of the task). Probe readouts and the compositions that use them live next to their label functions
(harness.eval.latent_eval: PacketProbeHook, latent_hooks; harness.eval.dual_latent_eval: DualPacketProbeHook,
dual_latent_hooks; harness.data.contact_metrics: MotionRecord). This module imports nothing from harness.eval or
harness.data: both import it (test_layering: the package graph is acyclic).

RP2 additions (the eval loops that stepped sessions themselves): `budget_task` (a task whose judge only spends the tick
budget, the end rules being hooks), `EndWhen` (a predicate ends the episode after a tick, e.g. a teacher reference
finished), `HoldPolicy` / `warm_up` (hold ticks through rollout) and `Recorder` (per-tick probe callback with an
on_end result). Family-specific recorders live next to their metric functions (teacher_quality, dual_teacher_quality,
latent_causal, latent_eval).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from rrp.policies.base import Act, PolicyInfo, Requirements
from rrp.tasks.spec import Judgement, TaskSpec


class Feasibility:
    """on_reset: end the episode as "infeasible" (0 steps, not attempted) when check(env) is False."""

    def __init__(self, check: Callable[[object], bool], reason: str = "teacher_infeasible"):
        self.check, self.reason = check, reason

    @property
    def failure_reasons(self) -> tuple[str, ...]:          # declared to rollout's vocabulary check
        return (self.reason,)

    def on_reset(self, i, env, obs):
        if not self.check(env):
            return Judgement(True, "infeasible", self.reason, bool(env.runtime.succeeded()), False)
        return None


def arm_feasible(env) -> bool:
    """The arm runner's filter: the pick_place scripted teacher finds a feasible grasp/place layout."""
    from rrp.policies.teachers.arm import PickPlaceTeacher
    return bool(PickPlaceTeacher(env).feasibility()["feasible"])


def dual_feasible(task: str) -> Callable[[object], bool]:
    """The dual eval's filter: the task's dual teacher finds the layout feasible (as in teacher validation)."""
    def check(env) -> bool:
        from rrp.policies.teachers.dual import TEACHERS
        return bool(TEACHERS[task](env).feasibility()["feasible"])
    return check


class SessionRecord:
    """on_end: MuJoCo session facts the old loops wrote into every row (sim time, task-event statuses)."""

    def on_end(self, i, env, ep):
        return dict(sim_time=float(env.data.time), events={e: v.status for e, v in env.runtime.instances.items()})


class Settle:
    """on_end (dual arm): after the episode ends, step `n` hold ticks (`warm_up`, a rollout) and re-read privileged
    success, as the dual eval did (success is judged after the objects settle). Timeouts stay timeouts unless the settled
    scene succeeds."""

    def __init__(self, n: int = 5):
        self.n = n

    def on_end(self, i, env, ep):
        if ep.outcome in ("infeasible", "crash"):
            return {}
        warm_up(env, self.n)
        priv, pub = bool(env.privileged_success()), bool(env.runtime.succeeded())
        ep.success_privileged, ep.success_public = priv, pub
        if priv:
            ep.outcome, ep.failure_reason = "success", None
        elif ep.outcome == "success":
            ep.outcome, ep.failure_reason = "failure", "privileged_failure"
        return dict(settle_ticks=self.n)


class Displacement:
    """on_reset/on_end: displacement of every scene object (m) and the list of objects moved > `moved_m`."""

    def __init__(self, moved_m: float = 0.03):
        self.moved_m, self.init = moved_m, {}

    def _pos(self, env):
        return {o.sim_body: env.data.xpos[env.model.body(o.sim_body).id].copy()
                for o in env.scenario.objects if o.kind == "object"}

    def on_reset(self, i, env, obs):
        self.init[i] = self._pos(env)

    def on_end(self, i, env, ep):
        now = self._pos(env)
        disp = {b: float(np.linalg.norm(now[b] - p0)) for b, p0 in self.init.pop(i).items()}
        return dict(final_disp_m=disp, moved=[b for b, d in disp.items() if d > self.moved_m])


class ObjectShift:
    """Intervention (labelled `ladder_disturbance` in the session's intervention log): teleport `body` by (dx, dy) at
    the START of control tick `tick` (tick 0: at reset; later ticks: right after the previous tick's step). An episode
    that ends on the previous tick is not touched: `alive(i)` (default: always) says whether episode i continues."""

    def __init__(self, tick, dx: float, dy: float, body: str = "cube", alive: Callable[[int], bool] | None = None,
                 source: str = "ladder_disturbance"):
        self.tick, self.dx, self.dy, self.body, self.alive, self.source = tick, dx, dy, body, alive, source
        self.n: dict[int, int] = {}

    def _shift(self, env):
        p = env.data.xpos[env.model.body(self.body).id].copy()
        p[0] += self.dx
        p[1] += self.dy
        env.teleport_object(self.body, p, source=self.source)

    def on_reset(self, i, env, obs):
        self.n[i] = 0
        if self.tick == 0:
            self._shift(env)

    def on_step(self, i, env, act, step):
        self.n[i] += 1
        if self.n[i] == self.tick and (self.alive is None or self.alive(i)):
            self._shift(env)


class CommandLog:
    """on_act: log[i] = the EXACT executed command groups per tick (None = hold), for replay."""

    def __init__(self, log: dict):
        self.log = log

    def on_act(self, i, obs, act):
        c = act.command
        self.log.setdefault(i, []).append(None if c is None else {
            g: np.array(v, copy=True) if not np.isscalar(v) else v for g, v in c.groups.items()})


class PrevAction:
    """on_act: feed the command about to be executed to the session's PrevActionFeaturizer (rrp.harness.eval.ladder;
    the training-consistent previous-command input, mode 'own'; the pre-step measured q0 normalizes it). The featurizer
    itself is installed by the env factory, before the policy's reset."""

    def __init__(self):
        self.env = {}

    def on_reset(self, i, env, obs):
        self.env[i] = env

    def on_act(self, i, obs, act):
        env = self.env[i]
        f = env._rrp_featurizer
        if act.command is not None and f.mode == "own":
            f.record(act.command, f.base(env.observe()).q0)


class FrameCallback:
    """on_step: cb(i, env, tick, teacher_phase) after every executed tick (rendering)."""

    def __init__(self, cb: Callable):
        self.cb, self.n = cb, {}

    def on_step(self, i, env, act, step):
        k = self.n.get(i, 0)
        self.n[i] = k + 1
        self.cb(i, env, k, act.info.get("phase"))


class EndWhen:
    """on_step: end the episode after the tick on which pred(i, env) holds (the task judge has not ended it). `outcome`
    is only a placeholder for loops that re-judge in on_end (Settle); default a failure with reason `reason`."""

    def __init__(self, pred: Callable[[int, object], bool], outcome: str = "failure", reason: str = "hook_end"):
        self.pred, self.outcome, self.reason = pred, outcome, reason

    @property
    def failure_reasons(self) -> tuple[str, ...]:          # declared to rollout's vocabulary check
        return (self.reason,)

    def on_step(self, i, env, act, step):
        if self.pred(i, env):
            return Judgement(True, self.outcome, self.reason)
        return None


class Recorder:
    """Per-tick recorder: on_reset(i, env) / on_act(i, env, act) (before the tick executes) / on_step(i, env, act, step)
    (after it) callbacks of a loop that only measures. Any callback may be None; nothing is returned to the rollout."""

    def __init__(self, on_reset=None, on_act=None, on_step=None):
        self._reset, self._act, self._step = on_reset, on_act, on_step
        self.env: dict = {}

    def on_reset(self, i, env, obs):
        self.env[i] = env
        if self._reset is not None:
            self._reset(i, env)

    def on_act(self, i, obs, act):
        if self._act is not None:
            self._act(i, self.env[i], act)

    def on_step(self, i, env, act, step):
        if self._step is not None:
            self._step(i, env, act, step)


def budget_task(name: str, env_id: str, note: str = "") -> TaskSpec:
    """A task whose judge only spends the tick budget: rollout(max_steps=n) ends an episode after exactly n ticks (the
    judge is then asked with the budget spent, "timeout"). The loop's own end rules live in hooks (EndWhen, a policy
    reading its teacher, ...); the loops it serves report privileged outcomes themselves (Settle)."""
    def judge(env, t, max_seconds):
        return Judgement(t >= max_seconds, "timeout", "timeout" if t >= max_seconds else None)
    return TaskSpec(name, {env_id: {}}, float("inf"), judge, note=note or "budget only: end rules live in hooks")


class HoldPolicy:
    """Policy that holds (Act(None): the env keeps its last targets / executes its queued chunk row)."""

    def __init__(self):
        self.info = PolicyInfo("hold", "mock", "hold", Requirements(frozenset(), observations=frozenset()))

    def reset(self, spec, task, seeds, *, envs=None):
        pass

    def act(self, obs):
        return {i: Act(None) for i in obs}


def warm_up(env, ticks: int = 10):
    """Step `env` `ticks` hold ticks (through rollout: same state as calling env.step(None) `ticks` times) and return
    it. rollout only closes the env, which a session survives. A crashed tick raises (as the direct step did)."""
    from rrp.harness.rollout import rollout
    if ticks > 0:
        ep = rollout(lambda seed: env, HoldPolicy(), budget_task("warm_up", env.spec.env_id), [0], batch=1,
                     max_steps=ticks)[0]
        if ep.outcome == "crash":
            raise RuntimeError(ep.metrics.get("note") or ep.failure_reason)
    return env


def arm_scene(seed: int) -> dict:
    """The arm runner's scene rule: pick_place with seed % 3 distractors."""
    return {"n_distractors": seed % 3}


def arm_hooks(check_feasible: bool = True) -> list:
    return ([Feasibility(arm_feasible)] if check_feasible else []) + [SessionRecord()]


def dual_hooks(task: str) -> list:
    return [Feasibility(dual_feasible(task)), Settle(5), SessionRecord()]      # record after settling (as before)


# TaskSpec.hooks name -> (env_id prefix the hook applies to, factory(task) -> [hook]): the per-family conventions of the
# former eval loops (arm feasibility + session record; dual adds settling; MuJoCo session facts for legged).
TASK_HOOKS: dict[str, tuple[str, Callable]] = {"session": ("mujoco/", lambda task: [SessionRecord()]),
                                               "arm": ("mujoco/arm", lambda task: arm_hooks()),
                                               "dual": ("mujoco/dual", lambda task: dual_hooks(task.name))}

HOOKS: dict[str, Callable] = {
    "feasibility": lambda check=arm_feasible, reason="teacher_infeasible": Feasibility(check, reason),
    "session_record": SessionRecord,
    "settle": Settle,
    "displacement": Displacement,
    "object_shift": ObjectShift,
    "command_log": CommandLog,
    "prev_action": PrevAction,
    "frame": FrameCallback,
    "end_when": EndWhen,
    "recorder": Recorder,
}
