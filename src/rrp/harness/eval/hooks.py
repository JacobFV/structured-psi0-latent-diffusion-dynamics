"""Rollout hooks for the special cases of the former per-family eval loops (docs/architecture.md section 6).

Feasibility (pre-episode teacher layout filter -> outcome "infeasible"), session records (sim time, task-event statuses),
post-success settling (dual arm), object displacement, and the ladder's per-tick machinery (object shift, motion
recorders, executed-command log, previous-command feature, frame callback). `HOOKS` names the constructors
(docs/architecture.md section 14.1: TaskSpec.hooks refers to these names). Probe readouts live next to their label functions
(harness.eval.latent_eval.PacketProbeHook, harness.eval.dual_latent_eval.DualPacketProbeHook); the per-family
compositions (arm_hooks, dual_hooks, latent_hooks, dual_latent_hooks) are here.
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


class MotionRecord:
    """Per-tick arm motion-quality recorder (envs.mujoco.motion_quality); on_end: metrics["motion"], plus the held-object
    drift keys when contact metrics are on (RRP_CONTACT_METRICS=1, or `contact=True`). A tick counts as a chunk / packet
    boundary when the policy's Act.info says `boundary`."""

    def __init__(self, contact: bool | None = None):
        from rrp.harness.data.contact_metrics import contact_metrics_enabled
        self.contact = contact_metrics_enabled(contact)
        self.rec, self.cf = {}, {}

    def on_reset(self, i, env, obs):
        from rrp.envs.mujoco.motion_quality import ArmMotionRecorder
        self.rec[i] = ArmMotionRecorder(env)
        if self.contact:
            from rrp.harness.data.contact_labels import ContactFrameRecorder
            self.cf[i] = ContactFrameRecorder(env)

    def on_step(self, i, env, act, step):
        self.rec[i].tick(None if act.command is None else act.command.groups, bool(act.info.get("boundary")))
        if self.contact:
            self.cf[i].tick()

    def on_end(self, i, env, ep):
        motion = self.rec.pop(i).summary()
        if self.contact:
            from rrp.harness.data.contact_metrics import arm_contact_motion
            motion.update(arm_contact_motion(self.cf.pop(i).recording()))
        return dict(motion=motion)


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


def arm_scene(seed: int) -> dict:
    """The arm runner's scene rule: pick_place with seed % 3 distractors."""
    return {"n_distractors": seed % 3}


def arm_hooks(check_feasible: bool = True) -> list:
    return ([Feasibility(arm_feasible)] if check_feasible else []) + [SessionRecord()]


def dual_hooks(task: str) -> list:
    return [Feasibility(dual_feasible(task)), Settle(5), SessionRecord()]      # record after settling (as before)


def latent_hooks(policy, probe=None, *, device="cpu", paired: bool = False) -> list:
    """The former evaluate_latent's conventions: arm feasibility, session record, system 0 counters, displacement of
    every object, packet probes (when a probe is given), paired-scene identity."""
    from rrp.harness.eval.latent_eval import PacketProbeHook, PairedMeta, System0Stats
    return (arm_hooks() + [System0Stats(policy), Displacement()] + ([PacketProbeHook(probe, device)] if probe else [])
            + ([PairedMeta()] if paired else []))


def dual_latent_hooks(policy, task: str, probe=None, *, device="cpu", packet_edit: str | None = None) -> list:
    """The former evaluate_dual_latent's conventions: dual teacher feasibility, settle 5 ticks then judge privileged
    success, session record, system 0 counters, per-slot packet probes; packet_edit installs policy.packet_hook."""
    from rrp.harness.eval.dual_latent_eval import PACKET_EDITS, DualPacketProbeHook
    from rrp.harness.eval.latent_eval import System0Stats
    policy.packet_hook = PACKET_EDITS[packet_edit] if packet_edit else None
    return dual_hooks(task) + [System0Stats(policy)] + ([DualPacketProbeHook(probe, device)] if probe else [])


HOOKS: dict[str, Callable] = {
    "feasibility": lambda check=arm_feasible, reason="teacher_infeasible": Feasibility(check, reason),
    "session_record": SessionRecord,
    "settle": Settle,
    "displacement": Displacement,
    "object_shift": ObjectShift,
    "motion": MotionRecord,
    "command_log": CommandLog,
    "prev_action": PrevAction,
    "frame": FrameCallback,
}
