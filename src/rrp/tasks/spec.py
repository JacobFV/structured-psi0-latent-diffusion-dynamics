"""Task registry (docs/architecture.md section 4): which envs a task exists in, how an episode is judged, its scripted
teacher (a policy registry key; teachers are policies, rrp.policies.teachers) and its episode budget.

A task also owns how its env is built and driven: `build` (env_id -> "module:builder", used by `make_env` in place of
the env's task if/elif), `scene` (seed -> scene kwargs; None = the env takes no scene), the default eval `hooks` by
name, an explicit tick budget `max_steps` and the `failure_reasons` vocabulary its judge may emit
(docs/architecture.md section 14.1).

Judges are harness-side evaluators: they may read privileged state through the env's `privileged_truth` capability
(`env.truth()`, `env.privileged_success()`), and what they return is reported, never fed back to a policy. They are
duck-typed on the env so this module stays below rrp.envs. Episode budgets are the ones the existing evaluations use
(arm runner 300 ticks at 20 Hz, dual 800 ticks, legged run_episode 60 s, humanoid step/gap evals 40 s / 30 s).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping

Outcome = Literal["success", "failure", "timeout", "fell", "infeasible", "rejected", "crash"]


@dataclass(frozen=True)
class Judgement:
    done: bool
    outcome: Outcome | None = None          # set when done
    failure_reason: str | None = None       # family-specific code ("dropped_off_table", "fell", "timeout", ...)
    success_public: bool | None = None      # from public estimators / the observation-backed task graph
    success_privileged: bool | None = None  # from privileged truth; reported only


Judge = Callable[[Any, float, float], Judgement]     # (env, elapsed seconds, budget seconds) -> Judgement


@dataclass(frozen=True)
class TaskSpec:
    name: str
    envs: Mapping[str, dict]                 # env_id -> extra env kwargs (make_env(..., **kw)) of the envs the task exists in
    max_seconds: float
    judge: Judge
    graph: str | None = None                 # src/rrp/tasks/graphs/<graph>.json task graph (None: the env supplies success)
    teacher: str | None = None               # POLICIES key of the scripted teacher (None: no demonstrator, explicit)
    gates: Mapping[str, Any] = field(default_factory=dict)
    note: str = ""
    build: Mapping[str, str] = field(default_factory=dict)   # env_id -> "module:builder" (replaces the env factory's task if/elif)
    scene: Callable[[int], dict] | None = None               # seed -> scene kwargs; None = the env takes no scene
    hooks: tuple[str, ...] = ()              # default eval hooks by name (rrp.harness.eval.evaluate.HOOKS)
    max_steps: int | None = None             # explicit control-tick budget (rollout's `max_steps` default)
    failure_reasons: tuple[str, ...] = ()    # the vocabulary the judge may emit (Judgement.failure_reason)
    needs: Mapping[str, str] = field(default_factory=dict)   # env capability the body must offer -> the negotiate reason when it does not


def arm_scene(seed: int) -> dict:
    """The arm pick_place scene rule: seed % 3 distractors."""
    return {"n_distractors": seed % 3}


GRAPH_REASONS = ("timeout", "privileged_failure")
LEGGED_REASONS = ("fell", "timeout", "privileged_failure", "drift_a", "drift_b", "halt")


def graph_judge(*, dropped_below_m: float | None = None, dropped_body: str = "cube") -> Judge:
    """Done when the public task graph completes, the time budget is spent or (arm) the object left the table.
    Outcome: success iff privileged success at the end (the public/privileged agreement is reported separately)."""

    def judge(env, t: float, max_seconds: float) -> Judgement:
        if dropped_below_m is not None:
            z = env.data.xpos[env.model.body(dropped_body).id][2]
            if z < dropped_below_m:
                return Judgement(True, "failure", "dropped_off_table", bool(env.runtime.succeeded()), False)
        pub = bool(env.runtime.succeeded())
        if not pub and t < max_seconds:
            return Judgement(False)
        priv = bool(env.privileged_success())
        if priv:
            return Judgement(True, "success", None, pub, True)
        return Judgement(True, "timeout" if not pub else "failure", "timeout" if not pub else "privileged_failure", pub, False)

    return judge


def legged_judge() -> Judge:
    """mujoco/legged episodes (the former harness.eval.legged_latent_eval.run_episode rules). Episodes end only on the
    env's 10 Hz boundary ticks (`env.boundary`; with the 50 Hz `legs` control most steps are not boundaries): a fall
    (outcome "fell"), the public task graph completing (success iff privileged success), or the budget (success if the
    privileged evaluator already holds, else "timeout" with the waypoint stage: drift_a = waypoint a never reached,
    drift_b = a reached but not b, halt = b reached, halt not completed)."""

    def judge(env, t: float, max_seconds: float) -> Judgement:
        if not getattr(env, "boundary", True):
            return Judgement(False)
        pub = bool(env.runtime.succeeded())
        if env.fell:
            return Judgement(True, "fell", "fell", pub, False)
        if not pub and t < max_seconds:
            return Judgement(False)
        priv = bool(env.privileged_success())
        if priv:
            return Judgement(True, "success", None, pub, True)
        if pub:
            return Judgement(True, "failure", "privileged_failure", True, False)
        ev = {e: v.status for e, v in env.runtime.instances.items()}
        done = lambda e: ev.get(e) in ("succeeded", "completed")
        reason = ("timeout" if "walk_to_a" not in ev else "drift_a" if not done("walk_to_a") else
                  "drift_b" if not done("walk_to_b") else "halt")
        return Judgement(True, "timeout", reason, False, False)

    return judge


TASKS: dict[str, TaskSpec] = {}


def register_task(t: TaskSpec) -> TaskSpec:
    if t.name in TASKS and TASKS[t.name] != t:
        raise ValueError(f"task {t.name} is already registered differently")
    TASKS[t.name] = t
    return t


def get_task(name: str) -> TaskSpec:
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; registered: {sorted(TASKS)}")
    return TASKS[name]


register_task(TaskSpec("pick_place", {"mujoco/arm": {}}, 15.0, graph_judge(dropped_below_m=-0.05), graph="pick_place",
                       teacher="teacher:pick_place", scene=arm_scene, hooks=("arm",),
                       failure_reasons=("dropped_off_table",) + GRAPH_REASONS))
register_task(TaskSpec("reach_pose", {"mujoco/arm": {}}, 15.0, graph_judge(), graph="reach_pose", hooks=("session",),
                       failure_reasons=GRAPH_REASONS))
# dual tasks: (name, graph, teacher). pivot_against_surface is PARKED (D-146 item 5): scene and graph stay, no demonstrator (explicit None)
for _name, _graph, _teacher in [("support_insert", "support_and_insert", "teacher:support_insert"), ("handover", "handover", "teacher:handover"),
                                ("assign_left", "pick_place", "teacher:assign_left"), ("assign_right", "pick_place", "teacher:assign_right"),
                                ("pivot_against_surface", "pivot_against_surface", None)]:
    register_task(TaskSpec(_name, {"mujoco/dual": {}}, 40.0, graph_judge(), graph=_graph, teacher=_teacher,
                           hooks=("dual",), failure_reasons=GRAPH_REASONS))
register_task(TaskSpec("waypoint_contact", {"mujoco/legged": {}}, 60.0, legged_judge(), graph="waypoint_contact",
                       teacher="teacher:waypoint_contact", hooks=("session",), failure_reasons=LEGGED_REASONS,
                       note="legged_judge: fall / drift_a / drift_b / halt (the former run_episode rules)"))
register_task(TaskSpec("foothold_steps", {"mujoco/legged": {}}, 60.0, legged_judge(), graph="foothold_steps",
                       hooks=("session",), failure_reasons=LEGGED_REASONS))
register_task(TaskSpec("locomotion", {"warp/legged": {}}, 20.0, lambda env, t, T: Judgement(t >= T, "timeout" if t >= T else None),
                       failure_reasons=("timeout",),
                       note="tracker training task: command following, reward-driven (capability reward)"))
# ComputerWorld (architecture.md section 5): setups, instructions and judges are env-side (rrp.envs.computerworld reads
# CW scenes); no task graph yet, the env judges success. Budgets at 10 Hz.
for _name, _budget, _why in [
        ("cw/calc_sum", 15.0, ("wrong_result", "no_result", "timeout")),
        ("cw/open_type", 12.0, ("wrong_text", "app_not_open", "incomplete_text", "timeout")),
        ("cw/drag_window", 8.0, ("window_closed", "off_target", "timeout")),
        ("cw/fill_form", 20.0, ("wrong_value:name", "wrong_value:email", "not_submitted", "timeout"))]:
    register_task(TaskSpec(_name, {"computerworld": {}}, _budget, lambda env, t, T: env.cw_judge(t, T),
                           teacher=f"teacher:{_name}", failure_reasons=_why))


# ------------------------------------------------------------------ Ψ₀ SIMPLE benchmark (W10, D-140; research/tracks/psi0.md)
# task -> released SIMPLE-finetuned Ψ₀ run (HF USC-PSI-Lab/psi-model), published successes of 10 at DR level 0|1|2, and
# the step-1 reproduction status on our Isaac 5.1 / aarch64 / path-traced stack (P-010, P-012, P-015, P-016).
SIMPLE_TASKS: dict[str, tuple[str, tuple[int, int, int], str]] = {
    "G1WholebodyTabletopGraspMP-v0": ("g1wholebodytabletopgrasp-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2603181503", (10, 10, 8), "reproduced 10/10"),
    "G1WholebodyBendPickMP-v0": ("g1wholebodybendpick-v0.simple.flow1000.cosine.lr1.0e-04.b256.gpus8.2603151312", (10, 10, 10), "reproduced 10/10"),
    "G1WholebodyHandoverTeleop-v0": ("g1wholebodyhandover-v0.simple.flow1000.cosine.lr1.0e-04.b64.gpus4.2604071507", (7, 7, 10), "reproduced 7/10"),
    "G1WholebodyXMovePickTeleop-v0": ("g1wholebodyxmovepick-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2604022205", (10, 10, 6), "not reproduced 0/10 (P-012)"),
    "G1WholebodyLocomotionPickBetweenTablesTeleop-v0": ("g1wholebodylocomotionpickbetweentablesteleop-v0.simple.flow1000.cosine.lr1.0e-04.b64.gpus4.2604081126", (7, 5, 6), "not reproduced 0/5 (P-015)"),
    "G1WholebodyXMoveBendPickTeleop-v0": ("g1wholebodyxmovebendpickteleop-v0.simple.flow1000.cosine.lr1.0e-04.b112.gpus7.2604100422", (10, 9, 9), "not reproduced 3/6 (P-016)"),
}


def simple_judge(env, t: float, max_seconds: float) -> Judgement:
    """SIMPLE ends its own episodes (TimeLimit over the task's max_episode_steps, stabilization included, as upstream);
    success is SIMPLE's `_success` (simulator truth: success_privileged; SIMPLE has no public success estimator). A
    step that needed a chunk and got none (e.g. a replay that ran out of recorded rows) ends the episode as rejected."""
    if getattr(env, "last_rejected", None):
        tr = env.truth()
        return Judgement(True, "rejected", env.last_rejected, None, bool(tr["success"]))
    tr = env.truth()
    if tr["terminated"] or tr["truncated"] or t >= max_seconds:
        ok = bool(tr["success"])
        return Judgement(True, "success" if ok else "timeout", None if ok else "timeout", None, ok)
    return Judgement(False)


for _task, (_run, _pub, _status) in SIMPLE_TASKS.items():
    register_task(TaskSpec(f"simple/{_task}", {"simple": {}}, 20.0, simple_judge,
                           gates=dict(published_success_of_10=_pub, released_run=_run, step1=_status),
                           failure_reasons=("timeout", "chunk_required"),
                           note="judge reads env.truth(); episode budget = SIMPLE's own TimeLimit (<= 16 s at 50 Hz)"))


def tasks_in(env_id: str) -> tuple[str, ...]:
    """Names of the registered tasks that exist in `env_id` (the one source of "the dual tasks", "the legged tasks")."""
    return tuple(n for n, t in TASKS.items() if env_id in t.envs)


import rrp.tasks.humanoid  # noqa: E402,F401  (registers h_steps, h_gap; it imports this module, so it comes last)
