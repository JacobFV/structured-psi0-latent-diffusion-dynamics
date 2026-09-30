"""Humanoid tasks (docs/architecture.md sections 14.1 and 14.4, unit HJ): the judge and the registration of `h_steps`
and `h_gap`. Later units add their tasks here (U2: L0, L3, M1-M3, C1, C2).

The judge maps what the env says (`env.failure_reason()` in its own vocabulary, `env.fell`) and what privileged truth
says (`env.privileged_success()`, the public task graph) to the humanoid failure vocabulary `HUMANOID_REASONS`; what
it returns is reported, never fed back to a policy. Reasons that end an episode at once (`TERMINAL`: the body is down,
in the wall, or has lost / dropped the payload) are read every boundary tick; the others are reported when the
episode ends for another reason (budget, or the public graph completing while privileged truth denies it).
"""
from __future__ import annotations

import math

from rrp.tasks.spec import Judge, Judgement, TaskSpec, register_task

HUMANOID_REASONS = ("fell", "wall_collision", "wrong_heading", "trip", "missed_step", "hold_lost", "dropped", "timeout")
TERMINAL = ("fell", "wall_collision", "hold_lost", "dropped")
HEADING_TOL = 0.3          # rad; the final-heading tolerance of the h_gap privileged evaluator (humanoid_eval)
# env-side codes in another vocabulary -> the humanoid one (identity for the codes already in it)
ENV_CODES = {**{r: r for r in HUMANOID_REASONS}, "missed_foothold": "missed_step", "wrong_foot": "missed_step",
             "dropped_off_table": "dropped"}


def canonical(code: str) -> str:
    """An env failure code in the humanoid vocabulary; an unmapped code is an error (the vocabulary stays closed)."""
    if code not in ENV_CODES:
        raise ValueError(f"env failure code {code!r} is not in the humanoid vocabulary {HUMANOID_REASONS} "
                         f"(aliases {sorted(set(ENV_CODES) - set(HUMANOID_REASONS))})")
    return ENV_CODES[code]


def _heading_error(env) -> float | None:
    """|final heading - psi_f| from privileged truth (h_gap scenes); None when the scene has no target heading."""
    psi = getattr(getattr(env, "scenario", None), "meta", {}).get("psi_f")
    if psi is None:
        return None
    _, _, yaw = env.base_pose_truth()
    return abs(math.atan2(math.sin(psi - yaw), math.cos(psi - yaw)))


def humanoid_judge() -> Judge:
    """Done on a fall or another terminal env failure, the public task graph completing, or the budget. Only the env's
    10 Hz boundary ticks end an episode (as `legged_judge`). Outcome: success iff privileged success; "fell" for a
    fall; "timeout" for an unfinished graph at the budget; otherwise "failure" with the mapped reason."""
    from rrp.envs.base import env_failure_reason

    def judge(env, t: float, max_seconds: float) -> Judgement:
        if not getattr(env, "boundary", True):
            return Judgement(False)
        pub = bool(env.runtime.succeeded())
        code = env_failure_reason(env)
        reason = canonical(code) if code else None
        if reason is None and getattr(env, "fell", False):
            reason = "fell"
        if reason in TERMINAL:
            return Judgement(True, "fell" if reason == "fell" else "failure", reason, pub, False)
        if not pub and t < max_seconds:
            return Judgement(False)
        if env.privileged_success():
            return Judgement(True, "success", None, pub, True)
        if reason is None and not pub:
            reason = "timeout"
        if reason is None:                       # public graph done, truth denies it, the env named nothing
            err = _heading_error(env)
            if err is None or err <= HEADING_TOL:
                raise RuntimeError("humanoid judge: privileged failure after public success and the env reports no "
                                   "failure_reason(); the env must name it")
            reason = "wrong_heading"
        return Judgement(True, "timeout" if reason == "timeout" else "failure", reason, pub, False)

    return judge


def build_mujoco(*, task: str, body: str, seed: int = 0, scene: dict | None = None, **kw):
    """env_id "mujoco/legged" for h_steps / h_gap: the humanoid scene (`scene` = builder kwargs: h_frac | level, contact)
    on the body, driven by a LeggedSession. `kw` go to the session."""
    from rrp.envs.mujoco import humanoid_scenes as hs
    from rrp.envs.mujoco.legged import LeggedSession
    builder = {"h_steps": hs.build_h_steps, "h_gap": hs.build_h_gap}[task]
    return LeggedSession(builder(body, seed, **(scene or {})), seed=seed, **kw)


_BUILD = {"mujoco/legged": "rrp.tasks.humanoid:build_mujoco"}
_ENVS = {"mujoco/legged": {}, "warp/legged": {}}       # warp/legged: the env's own factory (task dispatch in make_warp_env)

register_task(TaskSpec("h_steps", _ENVS, 40.0, humanoid_judge(), graph="h_steps", teacher="teacher:h_steps",
                       hooks=("session",), build=_BUILD, failure_reasons=HUMANOID_REASONS,
                       note="humanoid_judge: fell / trip / missed_step / wrong_heading / timeout"))
register_task(TaskSpec("h_gap", _ENVS, 30.0, humanoid_judge(), graph="h_gap_sidestep", teacher="teacher:h_gap",
                       hooks=("session",), build=_BUILD, failure_reasons=HUMANOID_REASONS,
                       note="humanoid_judge: fell / wall_collision / wrong_heading / timeout"))
