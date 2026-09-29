"""Rollout hooks for the special cases of the former per-family eval loops (docs/architecture.md section 6).

Feasibility (pre-episode teacher layout filter -> outcome "infeasible"), session records (sim time, task-event statuses),
post-success settling (dual arm) and object displacement. Probe readouts live next to their label functions
(harness.eval.latent_eval.PacketProbeHook, harness.eval.dual_latent_eval.DualPacketProbeHook).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from rrp.tasks.spec import Judgement


class Feasibility:
    """on_reset: end the episode as "infeasible" (0 steps, not attempted) when check(env) is False."""

    def __init__(self, check: Callable[[object], bool], reason: str = "teacher_infeasible"):
        self.check, self.reason = check, reason

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
    """on_end (dual arm): after the episode ends, step `n` hold ticks and re-read privileged success, as the dual eval
    did (success is judged after the objects settle). Timeouts stay timeouts unless the settled scene succeeds."""

    def __init__(self, n: int = 5):
        self.n = n

    def on_end(self, i, env, ep):
        if ep.outcome in ("infeasible", "crash"):
            return {}
        for _ in range(self.n):
            env.step(None)
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


def arm_scene(seed: int) -> dict:
    """The arm runner's scene rule: pick_place with seed % 3 distractors."""
    return {"n_distractors": seed % 3}


def arm_hooks(check_feasible: bool = True) -> list:
    return ([Feasibility(arm_feasible)] if check_feasible else []) + [SessionRecord()]


def dual_hooks(task: str) -> list:
    return [Feasibility(dual_feasible(task)), SessionRecord(), Settle(5)]
