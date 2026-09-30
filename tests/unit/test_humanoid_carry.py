"""U3 (audit D13 / D9, architecture 14.4): C1 `h_carry`, C2 `h_loco_pick` and the HELD-OUT compositions `h_steps_carry` (carry over steps)
and `h_gap_cart` (push a cart through the gap) on the Menagerie humanoid t1 (`control="wholebody"`).

Registration, graphs, the judge reasons `dropped` / `hold_lost` / `wall_collision`, the held-out guard (the held-out names are in no
training list, recipe or trained-task registry), the arm-role negotiate reason and the teacher wiring run with a stub actor
(no result). The scripted-teacher episodes use the registered `t1:contact_v2` tracker (`artifacts/trackers/`, not in git) and skip
without it. What they show is a MEASURED success of a privileged scripted teacher on a flat-floor tracker that was trained without a
payload or arm motion: see research/tracks/humanoid.md (U3) for the rates over seeds; the tests assert only the cases that were measured
to succeed with a margin (C2 every seed tried, C1 with the goal on the left, the cart)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_humanoid_manip import Fake, _stub_session  # noqa: E402,F401
from tests.conftest import need_weights  # noqa: E402
from test_wholebody import BODY, registry  # noqa: E402,F401  (registry is a fixture)

import rrp.envs.mujoco.humanoid_scenes as HS  # noqa: E402
import rrp.envs.mujoco.legged_tracker as LT  # noqa: E402
from rrp.core.action import NativeCommand  # noqa: E402
from rrp.harness.eval.evaluate import evaluate  # noqa: E402
from rrp.policies.base import POLICIES, make_policy  # noqa: E402
from rrp.policies.teachers import humanoid as TH  # noqa: E402
from rrp.tasks.humanoid import ENDS_AT_ONCE, HUMANOID_REASONS, MANIP_REASONS, humanoid_judge  # noqa: E402
from rrp.tasks.spec import get_task  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TRAINED = ("h_carry", "h_loco_pick")
HELD_OUT = ("h_steps_carry", "h_gap_cart")


# ---------------------------------------------------------------- registration, graphs, vocabulary
@pytest.mark.parametrize("name", TRAINED + HELD_OUT)
def test_task_registered_with_graph_teacher_and_vocabulary(name):
    t = get_task(name)
    assert t.graph == name and t.teacher == f"teacher:{name}" and set(t.envs) == {"mujoco/legged"}
    assert t.build == {"mujoco/legged": "rrp.envs.mujoco.humanoid_scenes:make_humanoid_session"}
    assert t.max_steps == int(50 * t.max_seconds) and t.failure_reasons[0] == "fell" and t.failure_reasons[-1] == "timeout"
    assert set(t.failure_reasons) <= set(HUMANOID_REASONS) | set(MANIP_REASONS)
    g = json.loads((Path(TH.__file__).parents[2] / "tasks" / "graphs" / f"{name}.json").read_text())
    assert g["task_id"] == name and g["success_events"]
    assert POLICIES[f"teacher:{name}"] == "rrp.policies.teachers.humanoid:make_manip_teacher_policy"
    assert type(make_policy(f"teacher:{name}")).__name__ == "ManipTeacherPolicy"
    assert name in TH.ALL_MANIP_TEACHERS


def test_carry_reasons_are_in_the_judged_vocabulary():
    assert {"dropped", "hold_lost"} <= set(get_task("h_carry").failure_reasons)
    assert {"wall_collision", "hold_lost"} <= set(get_task("h_gap_cart").failure_reasons)
    assert {"dropped", "hold_lost", "wall_collision"} <= set(ENDS_AT_ONCE)


@pytest.mark.parametrize("code,t,expect", [
    ("dropped", 3.0, ("failure", "dropped")),
    ("hold_lost", 3.0, ("failure", "hold_lost")),
    ("wall_collision", 3.0, ("failure", "wall_collision")),
])
def test_judge_ends_the_carry_failures_at_once(code, t, expect):
    j = humanoid_judge(HUMANOID_REASONS + MANIP_REASONS)(Fake(pub=False, code=code), t, 45.0)
    assert j.done and (j.outcome, j.failure_reason) == expect and j.success_privileged is False


# ---------------------------------------------------------------- the held-out guard
def test_held_out_tasks_are_in_no_training_list():
    assert set(HS.HELD_OUT_TASKS) == set(HELD_OUT) and set(HS.CARRY_TASKS) == set(TRAINED)
    assert not set(HS.HELD_OUT_TASKS) & set(HS.TRAIN_TASKS)
    assert set(HS.CARRY_TASKS) <= set(HS.TRAIN_TASKS)
    split = json.loads((REPO / "research" / "splits" / "humanoid_v1.json").read_text())
    assert set(HELD_OUT) <= set(split["held_out_tasks"]) and not set(split["held_out_tasks"]) & set(HS.TRAIN_TASKS)


def test_no_recipe_or_config_names_a_held_out_task():
    """No training recipe / config / preset lists a held-out task (an evaluation-only task: the split is frozen, D-138). The only recipe
    that may name one is its own evaluation-only `recipes/humanoid/transfer_<task>.yaml` (HR, D-146 round 2; that it has no collect / pack /
    train / adapt node is asserted in tests/unit/test_humanoid_recipes.py); a YAML comment may mention one."""
    hits = []
    for root in ("recipes", "configs"):
        for p in (REPO / root).rglob("*") if (REPO / root).is_dir() else []:
            if p.is_file() and p.suffix in (".yaml", ".yml", ".json", ".toml", ".py"):
                rel = p.relative_to(REPO).as_posix()
                txt = p.read_text(errors="ignore")
                if p.suffix in (".yaml", ".yml"):
                    txt = "\n".join(ln for ln in txt.splitlines() if not ln.lstrip().startswith("#"))
                hits += [f"{rel}:{n}" for n in HELD_OUT if n in txt and rel not in (f"recipes/humanoid/transfer_{n}.yaml", f"recipes/humanoid/transfer_{n}_legged_none.yaml")]
    assert not hits, hits


# ---------------------------------------------------------------- the arm roles are null on a legs-only body
@pytest.mark.menagerie
@pytest.mark.parametrize("task", TRAINED + HELD_OUT)
def test_legs_only_body_is_na_in_the_matrix_not_an_exception(task):
    """TK: no `AbsentLimb` at build. A legs-only body (berkeley) is an n/a matrix cell answered by `negotiate` from mujoco/legged's static
    `env_spec` (no session is built, none raises): the arm tasks declare `needs={"arm_roles": ...}`, which the body lacks
    (tests/unit/test_task_teacher_closure.py, tests/unit/test_tracker_inputs.py)."""
    pytest.importorskip("mujoco")
    from rrp.harness.eval.evaluate import matrix
    assert not hasattr(HS, "AbsentLimb") and get_task(task).needs == {"arm_roles": "body has no arm roles"}
    (row,) = matrix([f"teacher:{task}"], [("mujoco/legged", "berkeley")], [task])
    assert row["status"] == "n/a" and any("body has no arm roles" in r for r in row["reasons"]), row
    assert not any("env unavailable" in r for r in row["reasons"]), row


@pytest.mark.menagerie
@pytest.mark.parametrize("task", TRAINED + HELD_OUT)
def test_arm_body_session_offers_arm_roles(tmp_path, registry, task):
    s = _stub_session(tmp_path, registry, task)
    assert s.spec.has("arm_roles")


# ---------------------------------------------------------------- scenes and teacher wiring with a stub actor
@pytest.mark.menagerie
@pytest.mark.parametrize("task", TRAINED + HELD_OUT)
def test_teacher_command_carries_both_groups_and_labels_its_legs(tmp_path, registry, task):
    s = _stub_session(tmp_path, registry, task)
    assert s.control == "wholebody" and s.scenario.name == task and s.scenario.meta["scene_version"] == HS.CARRY_SCENE_VERSION
    tch = TH.ALL_MANIP_TEACHERS[task](s)
    assert tch.source == "scripted_teacher" and tch.privileged
    for _ in range(100):                                                    # 2 s
        cmd = tch.act()
        assert isinstance(cmd, NativeCommand) and set(cmd.groups) == {"legs", "upper"} and cmd.source == "scripted_teacher"
        assert np.isfinite(cmd.groups["legs"]).all() and np.isfinite(cmd.groups["upper"]).all()
        s.step(cmd)
    pol = TH.ManipTeacherPolicy(task)
    pol.reset(None, task, [0], envs=[s])
    assert "rl_expert" in TH.ALL_MANIP_TEACHERS[task].legs and pol.labels[0].startswith(f"scripted_teacher:{task}+legs:")
    assert TH.ALL_MANIP_TEACHERS[task].legs.split("+")[0] in pol.labels[0]


@pytest.mark.menagerie
def test_carry_and_loco_pick_scenes_differ_by_the_task_edit(tmp_path, registry):
    """Causal use of the scene: C2 puts the crate `approach_m` beyond the M2 stance (the walk is real), C1 puts the goal behind / beside."""
    c2 = _stub_session(tmp_path, registry, "h_loco_pick").scenario.meta
    m2 = _stub_session(tmp_path, registry, "h_squat_pick").scenario.meta
    a = c2["approach_m"]
    assert HS.LOCO_APPROACH[0] * c2["L"] <= a <= HS.LOCO_APPROACH[1] * c2["L"] and c2["stance_x"] == pytest.approx(a)
    assert c2["box"]["xy"][0] == pytest.approx(m2["box"]["xy"][0] + a)
    c1 = _stub_session(tmp_path, registry, "h_carry").scenario.meta
    gx, gy = c1["goal"]
    assert gx < 0.2 * c1["L"] and np.hypot(gx, gy) >= HS.CARRY_GOAL_DIST[0] * c1["L"] - 1e-9 and c1["goal_tol_m"] == HS.CARRY_TOL
    sc = _stub_session(tmp_path, registry, "h_steps_carry").scenario.meta
    assert sc["goal"][0] < 0 and sc["staircase"] and sc["h_frac"] >= HS.STEPS_CARRY_H[0] - 1e-9


@pytest.mark.menagerie
def test_dropped_and_hold_lost_follow_the_payload_edit(tmp_path, registry):
    """Moving the box in the true state changes the privileged reason: onto the floor -> dropped; away from the palms after a lift ->
    hold_lost (after HOLD_LOST_TICKS boundary ticks); a box beside the palms and never lifted -> no_grasp."""
    s = _stub_session(tmp_path, registry, "h_carry")
    sc = s.scenario.meta
    jq = s.model.joint("box_free").qposadr[0] if any(s.model.joint(i).name == "box_free" for i in range(s.model.njnt)) else None
    if jq is None:
        jq = int(s.model.body_jntadr[s.model.body("box").id])
        jq = int(s.model.jnt_qposadr[jq])
    s.data.qpos[jq:jq + 3] = [0.0, 0.0, 0.5]
    mujoco.mj_forward(s.model, s.data)
    s._lift_max = 0.0
    assert s.failure_reason() == "no_grasp"
    s._lift_max, s._engaged = sc["lift_m"] + 0.01, True
    s.data.qpos[jq:jq + 3] = [0.3, 1.0, 0.7]                                   # in the air, 1 m from both palms
    mujoco.mj_forward(s.model, s.data)
    for k in range(HS.HOLD_LOST_TICKS):
        assert s.failure_reason() != "hold_lost"
        s._track_payload(sc)
    assert s.failure_reason() == "hold_lost"
    s.data.qpos[jq:jq + 3] = [0.3, 0.0, 0.03]                                  # on the floor
    mujoco.mj_forward(s.model, s.data)
    assert s.failure_reason() == "dropped"


@pytest.mark.menagerie
def test_cart_in_the_wall_is_a_wall_collision(tmp_path, registry):
    s = _stub_session(tmp_path, registry, "h_gap_cart")
    sc = s.scenario.meta
    m, d = s.model, s.data
    xj, yj = m.joint("cart_x"), m.joint("cart_y")
    d.qpos[xj.qposadr[0]] = sc["wall_x"] - sc["cart"]["xy"][0]                  # the cart centre on the wall line ...
    d.qpos[yj.qposadr[0]] = sc["y_c"] + 0.5 * sc["gap_width"] + 0.3 - sc["cart"]["xy"][1]    # ... beside the gap: inside the wall block
    mujoco.mj_forward(m, d)
    assert s.failure_reason() != "wall_collision"
    s._track_payload(sc)
    s._track_payload(sc)
    assert s.failure_reason() == "wall_collision"


# ---------------------------------------------------------------- scripted-teacher episodes (real t1:contact_v2, measured)
def _row(task, seed):
    need_weights(Path(LT.TRACKER_DIR) / "t1" / "contact_v2" / "actor.pt")            # meta.json is tracked, actor.pt is not
    pol = make_policy(f"teacher:{task}")
    (ep,) = evaluate(pol, "mujoco/legged", task, BODY, [seed], batch=1, env_kw=dict(tracker="t1:contact_v2", tracker_kind="learned"))
    return ep.row(), pol


@pytest.mark.menagerie
@pytest.mark.parametrize("seed", [0, 3])
def test_loco_pick_teacher_walks_then_picks(seed):
    r, pol = _row("h_loco_pick", seed)
    assert (r["outcome"], r["failure_reason"], r["success_public"], r["success_privileged"]) == ("success", None, True, True), r
    assert r["time"] <= 20.0                                                 # a few seconds of walk plus the M2 pick (measured 15-16 s)
    assert "scripted_teacher" in pol.labels[0] and "h_loco_pick" in pol.labels[0] and "planned_com" in pol.labels[0]


@pytest.mark.menagerie
def test_carry_teacher_carries_the_box_to_a_goal_on_the_left():
    """C1 with a goal on the left of the start (seed 4: measured success; goals on the right fell 0 of 6, research/tracks/humanoid.md)."""
    r, pol = _row("h_carry", 4)
    assert (r["outcome"], r["failure_reason"], r["success_public"], r["success_privileged"]) == ("success", None, True, True), r
    assert "planned_com+rl_expert" in pol.labels[0]


@pytest.mark.menagerie
def test_cart_teacher_pushes_the_cart_through_the_gap():
    r, pol = _row("h_gap_cart", 0)
    assert (r["outcome"], r["failure_reason"], r["success_public"], r["success_privileged"]) == ("success", None, True, True), r
