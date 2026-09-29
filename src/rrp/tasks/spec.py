"""Task registry (docs/architecture.md section 4): which envs a task exists in, how an episode is judged, its scripted
teacher (a policy registry key; teachers are policies, rrp.policies.teachers) and its episode budget.

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
    envs: Mapping[str, dict]                 # env_id -> scene kwargs for the env factory (make_env(..., scene=...))
    max_seconds: float
    judge: Judge
    graph: str | None = None                 # tasks/<graph>.json task graph (None: the env supplies success)
    teacher: str | None = None               # policy registry key of the scripted teacher
    gates: Mapping[str, Any] = field(default_factory=dict)
    note: str = ""


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


for _name, _graph, _teacher in [("pick_place", "pick_place", "teacher:pick_place"), ("reach_pose", "reach_pose", None)]:
    register_task(TaskSpec(_name, {"mujoco/arm": {}}, 15.0,
                           graph_judge(dropped_below_m=-0.05 if _name == "pick_place" else None),
                           graph=_graph, teacher=_teacher))
for _name, _graph in [("support_insert", "support_and_insert"), ("handover", "handover"), ("assign_left", "pick_place"),
                      ("assign_right", "pick_place"), ("pivot_against_surface", "pivot_against_surface"),
                      ("carry_tray_level", "carry_tray_level")]:
    register_task(TaskSpec(_name, {"mujoco/dual": {}}, 40.0, graph_judge(), graph=_graph, teacher=f"teacher:{_name}"))
register_task(TaskSpec("waypoint_contact", {"mujoco/legged": {}}, 60.0, graph_judge(), graph="waypoint_contact",
                       teacher="teacher:waypoint_contact",
                       note="legged fall/drift/halt outcomes: ported from harness.eval.legged_latent_eval.run_episode in S5"))
register_task(TaskSpec("loco_pick", {"mujoco/legged": {}}, 60.0, graph_judge(), graph="loco_pick",
                       note="teacher is a stub (policies.teachers.legged_loco); no working demonstrator"))
register_task(TaskSpec("foothold_steps", {"mujoco/legged": {}}, 60.0, graph_judge(), graph="foothold_steps"))
register_task(TaskSpec("h_steps", {"mujoco/legged": {}, "warp/legged": {}}, 40.0, graph_judge(), graph="h_steps",
                       teacher="teacher:h_steps"))
register_task(TaskSpec("h_gap", {"mujoco/legged": {}, "warp/legged": {}}, 30.0, graph_judge(), graph="h_gap_sidestep",
                       teacher="teacher:h_gap"))
register_task(TaskSpec("locomotion", {"warp/legged": {}}, 20.0, lambda env, t, T: Judgement(t >= T, "timeout" if t >= T else None),
                       note="tracker training task: command following, reward-driven (capability reward)"))
# ComputerWorld (architecture.md section 5): setups, instructions and judges are env-side (rrp.envs.computerworld reads
# CW scenes); no task graph yet, the env judges success. Budgets at 10 Hz.
for _name, _budget in [("cw/calc_sum", 15.0), ("cw/open_type", 12.0), ("cw/drag_window", 8.0), ("cw/fill_form", 20.0)]:
    register_task(TaskSpec(_name, {"computerworld": {}}, _budget, lambda env, t, T: env.cw_judge(t, T),
                           teacher=f"teacher:{_name}"))
