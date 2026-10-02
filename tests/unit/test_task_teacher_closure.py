"""TK (readiness R2, D-146): the humanoid task / teacher registry is closed. Every TASKS entry has a resolvable teacher or an explicit
None, every h_* task has its `max_steps`, one `ALL_MANIP_TEACHERS` table, the retired legacy tasks are gone, a missing body capability is a
`negotiate` reason (not an exception at build) and the judge never emits a failure reason outside the task's declared vocabulary."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from rrp.envs.base import EnvSpec
from rrp.policies.base import POLICIES, make_policy, negotiate
from rrp.tasks.humanoid import ARM_TASKS, ENDS_AT_ONCE, HUMANOID_REASONS, MANIP_REASONS, humanoid_judge
from rrp.tasks.spec import TASKS, get_task

REPO = Path(__file__).resolve().parents[2]
H_TASKS = sorted(t for t in TASKS if t.startswith("h_"))


def test_every_task_has_a_resolvable_teacher_or_an_explicit_none():
    assert TASKS
    for name, t in TASKS.items():
        if t.teacher is None:
            assert name not in {k.split(":", 1)[1] for k in POLICIES if k.startswith("teacher:")}, f"{name}: teacher=None but a key exists"
        else:
            assert t.teacher in POLICIES, f"{name}: teacher {t.teacher!r} is not a POLICIES key"
            assert make_policy(t.teacher).info.requires.tasks >= {name}, name


def test_parked_dual_task_has_no_teacher_and_the_retired_tasks_are_gone():
    assert get_task("pivot_against_surface").teacher is None
    for gone in ("loco_pick", "carry_tray_level"):
        assert gone not in TASKS and f"teacher:{gone}" not in POLICIES
        assert not (REPO / "src" / "rrp" / "tasks" / "graphs" / f"{gone}.json").exists()
        assert (REPO / ".old" / "src" / "rrp" / "tasks" / "graphs" / f"{gone}.json").is_file()
    assert not (REPO / "src" / "rrp" / "policies" / "teachers" / "legged_loco.py").exists()
    from rrp.policies.teachers.dual_smooth import PARKED_TASKS
    assert PARKED_TASKS == ("pivot_against_surface",)           # only the parked scenario; no retired task is named


def test_every_h_task_has_a_step_budget():
    assert len(H_TASKS) >= 10
    assert get_task("h_steps").max_steps == 400 and get_task("h_gap").max_steps == 300
    for n in H_TASKS:
        t = get_task(n)
        hz = 10 if n in ("h_steps", "h_gap") else 50          # the terrain tasks step at the 10 Hz boundary, the wholebody tasks at 50 Hz
        assert t.max_steps and t.max_steps == int(hz * t.max_seconds), n


def test_one_manip_teacher_table():
    from rrp.policies.teachers import humanoid as TH
    assert not hasattr(TH, "MANIP_TEACHERS") and not hasattr(TH, "CARRY_TEACHERS")
    manip = {n for n in H_TASKS if get_task(n).teacher == f"teacher:{n}" and n not in ("h_steps", "h_gap")}
    assert set(TH.ALL_MANIP_TEACHERS) == manip


def _fake_spec(pol, task, *, lacks=()):
    """An env spec that offers everything the policy requires except the capabilities in `lacks`."""
    r = pol.info.requires
    return NS(env_id=next(iter(task.envs)), bodies=[], action_spaces=[NS(group=g, kind=k) for g in r.groups or ("x",) for k in r.action_kinds],
              action_kinds=lambda: set(r.action_kinds), has=lambda c: c not in lacks and c != "batched")


def test_missing_arm_roles_is_a_negotiate_reason_not_an_exception():
    t = get_task("h_carry")
    pol = make_policy("teacher:h_carry")
    assert negotiate(pol.info, _fake_spec(pol, t), t).ok
    c = negotiate(pol.info, _fake_spec(pol, t, lacks=("arm_roles",)), t)
    assert not c.ok and "task 'h_carry' needs arm_roles: body has no arm roles" in c.reasons
    assert [n for n in H_TASKS if get_task(n).needs] == sorted(ARM_TASKS + ("h_carry", "h_loco_pick", "h_steps_carry", "h_gap_cart"))
    for n in ("h_walk", "h_turn", "h_steps", "h_gap"):
        assert not get_task(n).needs, n                  # legs-only bodies stay valid for the locomotion tasks


def test_capability_is_declared_by_the_env_contract():
    assert "arm_roles" in EnvSpec.model_fields["capabilities"].annotation.__args__[0].__args__


def test_judge_never_emits_an_undeclared_reason():
    class Env:
        boundary, fell = True, False
        runtime = NS(succeeded=lambda: False, instances={})
        privileged_success = staticmethod(lambda: False)
        base_pose_truth = staticmethod(lambda: (0.0, 0.0, 0.0))

        def __init__(self, code):
            self._code = code

        def failure_reason(self):
            return self._code

    for n in H_TASKS:
        t = get_task(n)
        j = humanoid_judge(t.failure_reasons)
        assert set(t.failure_reasons) <= set(HUMANOID_REASONS) | set(MANIP_REASONS), n
        assert t.failure_reasons[-1] == "timeout", n
        for code in ENDS_AT_ONCE:
            declared = code in t.failure_reasons
            try:
                r = j(Env(code), 1.0, t.max_seconds)
            except RuntimeError as ex:
                assert not declared and code in str(ex), (n, code, ex)
                continue
            assert r.failure_reason is None or r.failure_reason in t.failure_reasons, (n, code, r.failure_reason)


def test_terrain_task_budget_is_course_derived_and_the_judge_times_out_at_it():
    """D-147 addendum 2026-10-02 item 2: budget = 1.5 x course / commanded speed (h_gap: + turn + hold); the judge uses min(spec, budget)."""
    import math
    from rrp.envs.mujoco.humanoid_scenes import course_budget_s
    assert course_budget_s("h_steps", 1.0, 0.8) == pytest.approx(1.5 * 6.3 / 0.48)
    assert course_budget_s("h_gap_sidestep", 1.0, 0.8, math.pi / 2) == pytest.approx(1.5 * (2.5 / 0.4 + math.pi + 0.5))

    class Env:
        boundary, fell = True, False
        runtime = NS(succeeded=lambda: False, instances={})
        privileged_success = staticmethod(lambda: False)
        scenario = NS(meta={"budget_s": 13.0})
        failure_reason = staticmethod(lambda: None)

    j = humanoid_judge(get_task("h_steps").failure_reasons)
    assert not j(Env(), 12.9, 40.0).done
    r = j(Env(), 13.0, 40.0)
    assert r.done and r.failure_reason == "timeout"
