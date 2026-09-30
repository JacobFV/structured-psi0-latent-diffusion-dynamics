"""Humanoid tasks (docs/architecture.md sections 14.1 and 14.4, unit HJ): the judge and the registration of `h_steps`
and `h_gap`; U2 adds L0 `h_walk`, L3 `h_turn`, M1 `h_reach`, M2 `h_squat_pick`, M3 `h_place` (whole-body control: the
`legs` group from a tracker actor plus the `upper` group from the scripted upper-body teacher). U3 adds C1 `h_carry`, C2
`h_loco_pick` and the HELD-OUT `h_steps_carry` (carry over steps) and `h_gap_cart` (push a cart through the gap); the held-out
names are never in a training list (`humanoid_scenes.HELD_OUT_TASKS`, guarded by tests/unit/test_humanoid_carry.py).

The judge maps what the env says (`env.failure_reason()` in its own vocabulary, `env.fell`) and what privileged truth
says (`env.privileged_success()`, the public task graph) to the humanoid failure vocabulary `HUMANOID_REASONS`; what
it returns is reported, never fed back to a policy. Reasons that end an episode at once (`ENDS_AT_ONCE`: the body is down,
in the wall, has lost / dropped the payload, or (h_turn) has drifted off its spot) are read every boundary tick; the others are reported when the
episode ends for another reason (budget, or the public graph completing while privileged truth denies it).
"""
from __future__ import annotations

import math

from rrp.tasks.spec import Judge, Judgement, TaskSpec, register_task

HUMANOID_REASONS = ("fell", "wall_collision", "wrong_heading", "trip", "missed_step", "hold_lost", "dropped", "timeout")
MANIP_REASONS = ("drift", "no_grasp", "not_upright", "place_miss")       # U2: what the manipulation tasks add to the vocabulary
TERMINAL = ("fell", "wall_collision", "hold_lost", "dropped")
ENDS_AT_ONCE = TERMINAL + ("drift",)                                       # read every boundary tick (drift: h_turn left its spot)
HEADING_TOL = 0.3          # rad; the final-heading tolerance of the h_gap privileged evaluator (humanoid_eval)
# env-side codes in another vocabulary -> the humanoid one (identity for the codes already in it)
ENV_CODES = {**{r: r for r in HUMANOID_REASONS + MANIP_REASONS}, "missed_foothold": "missed_step", "wrong_foot": "missed_step",
             "dropped_off_table": "dropped"}


def canonical(code: str) -> str:
    """An env failure code in the humanoid vocabulary; an unmapped code is an error (the vocabulary stays closed)."""
    if code not in ENV_CODES:
        raise ValueError(f"env failure code {code!r} is not in the humanoid vocabulary {HUMANOID_REASONS + MANIP_REASONS} "
                         f"(aliases {sorted(set(ENV_CODES) - set(HUMANOID_REASONS))})")
    return ENV_CODES[code]


def _heading_error(env) -> float | None:
    """|final heading - psi_f| from privileged truth (h_gap scenes); None when the scene has no target heading."""
    psi = getattr(getattr(env, "scenario", None), "meta", {}).get("psi_f")
    if psi is None:
        return None
    _, _, yaw = env.base_pose_truth()
    return abs(math.atan2(math.sin(psi - yaw), math.cos(psi - yaw)))


def humanoid_judge(declared: tuple[str, ...]) -> Judge:
    """`declared` is the task's `failure_reasons`: the judge never emits a reason outside it (an env code the task does not declare is
    an error naming both, not a silently widened vocabulary; X1 makes `harness.rollout` assert the same on every episode).
    Done on a fall or another terminal env failure, the public task graph completing, or the budget. Only the env's
    10 Hz boundary ticks end an episode (as `legged_judge`). Outcome: success iff privileged success; "fell" for a
    fall; "timeout" for an unfinished graph at the budget; otherwise "failure" with the mapped reason."""

    def judge(env, t: float, max_seconds: float) -> Judgement:
        j = _judge(env, t, max_seconds)
        if j.failure_reason is not None and j.failure_reason not in declared:
            raise RuntimeError(f"humanoid judge: failure reason {j.failure_reason!r} is not in the task's declared vocabulary {declared}")
        return j

    def _judge(env, t: float, max_seconds: float) -> Judgement:
        if not getattr(env, "boundary", True):
            return Judgement(False)
        pub = bool(env.runtime.succeeded())
        f = getattr(env, "failure_reason", None)          # the env's own code (rrp.envs.base.env_failure_reason; envs sit above tasks)
        code = f() if callable(f) else None
        reason = canonical(code) if code else None
        if reason is None and getattr(env, "fell", False):
            reason = "fell"
        if reason in ENDS_AT_ONCE:
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


_BUILD = {"mujoco/legged": "rrp.envs.mujoco.humanoid_scenes:make_humanoid_session"}
_ENVS = {"mujoco/legged": {}, "warp/legged": {}}       # warp/legged: the env's own factory (task dispatch in make_warp_env)

register_task(TaskSpec("h_steps", _ENVS, 40.0, humanoid_judge(HUMANOID_REASONS), graph="h_steps", teacher="teacher:h_steps",
                       hooks=("session",), build=_BUILD, failure_reasons=HUMANOID_REASONS, max_steps=400,
                       note="humanoid_judge: fell / trip / missed_step / wrong_heading / timeout"))
register_task(TaskSpec("h_gap", _ENVS, 30.0, humanoid_judge(HUMANOID_REASONS), graph="h_gap_sidestep", teacher="teacher:h_gap",
                       hooks=("session",), build=_BUILD, failure_reasons=HUMANOID_REASONS, max_steps=300,
                       note="humanoid_judge: fell / wall_collision / wrong_heading / timeout"))

# the tasks whose teacher squeezes / pushes with the palms need the two hand roles: `negotiate` reports a legs-only body (S3 berkeley) as
# n/a with this reason (the session declares `arm_roles`; the relational packet binds the hand roles to null, reason `absent_limb`)
ARMS = {"arm_roles": "body has no arm roles"}
ARM_TASKS = ("h_reach", "h_squat_pick", "h_place")           # the U2 tasks that need the hands (h_walk / h_turn are legs-only)

# U2 (whole-body control, 50 Hz steps: max_steps = 50 x seconds). mujoco/legged only: the Warp env has no upper body scenes yet.
_MANIP_ENVS = {"mujoco/legged": {}}
_MANIP = (
    ("h_walk", 30.0, "h_walk", ("fell", "timeout"), "L0: walk to the marker and halt (within 0.25 m, base speed < 0.15 m/s for 1 s)"),
    ("h_turn", 20.0, "h_turn", ("fell", "drift", "wrong_heading", "timeout"),
     "L3: turn in place to face the marker (bearing error < 0.25 rad for 1 s, base within 0.5 m of the start)"),
    ("h_reach", 15.0, "h_reach", ("fell", "timeout"), "M1: a palm within 5 cm of a point in the air for 0.5 s (arm IK)"),
    ("h_squat_pick", 40.0, "h_squat_pick", ("fell", "no_grasp", "dropped", "hold_lost", "not_upright", "timeout"),
     "M2: squat, bimanual palm-squeeze pick of a box from a low crate, stand (pelvis >= 0.9 of default) with the box 5 cm above the crate for 2 s"),
    ("h_place", 50.0, "h_place", ("fell", "no_grasp", "dropped", "hold_lost", "place_miss", "timeout"),
     "M3: lift the box, twist the trunk about the waist and put it down on the crate mark (within 4 cm, hands 15 cm clear for 1 s)"),
)
for _name, _sec, _graph, _why, _note in _MANIP:
    register_task(TaskSpec(_name, _MANIP_ENVS, _sec, humanoid_judge(_why), graph=_graph, teacher=f"teacher:{_name}", hooks=("session",),
                           build=_BUILD, failure_reasons=_why, max_steps=int(50 * _sec), note=f"humanoid_judge; {_note}",
                           needs=ARMS if _name in ARM_TASKS else {}))

# U3: carry / loco-pick (train) and the held-out compositions. Same 50 Hz whole-body env as U2. `dropped`: the payload left the crate /
# hands for the floor; `hold_lost`: it was lifted (or the cart pushed) and the palms have since left it; the cart adds `wall_collision`.
_CARRY = (
    ("h_carry", 45.0, "h_carry", ("fell", "no_grasp", "dropped", "hold_lost", "timeout"),
     "C1: squat pick, stand, carry the box to a goal 1-1.6 leg lengths away behind the start and halt there (box within 0.3 m for 1 s, still held)"),
    ("h_loco_pick", 50.0, "h_loco_pick", ("fell", "no_grasp", "dropped", "hold_lost", "not_upright", "timeout"),
     "C2: walk to a crate 0.8-1.4 m ahead, halt at the stance, squat pick and stand (M2 success condition after a walk)"),
    ("h_steps_carry", 70.0, "h_steps_carry", ("fell", "no_grasp", "dropped", "hold_lost", "timeout"),
     "HELD OUT: C1 with a staircase between the start and the goal (h_steps composed with a carry); needs the arm roles (n/a on a body without them, e.g. S3 berkeley)"),
    ("h_gap_cart", 50.0, "h_gap_cart", ("fell", "wall_collision", "no_grasp", "dropped", "hold_lost", "timeout"),
     "HELD OUT: push a cart by its handle through the h_gap opening to a goal beyond the wall (h_gap composed with a manipulation)"),
)
for _name, _sec, _graph, _why, _note in _CARRY:
    register_task(TaskSpec(_name, _MANIP_ENVS, _sec, humanoid_judge(_why), graph=_graph, teacher=f"teacher:{_name}", hooks=("session",),
                           build=_BUILD, failure_reasons=_why, max_steps=int(50 * _sec), note=f"humanoid_judge; {_note}", needs=ARMS))
