"""U2 (D-13 / D-9, architecture 14.4): the scripted whole-body teacher and the tasks L0 h_walk, L3 h_turn, M1 h_reach, M2 h_squat_pick,
M3 h_place on the Menagerie humanoid t1 (`control="wholebody"`).

Registration, graphs, vocabulary and the wiring of a teacher to the session run with a stub actor (random tiny MLP, no result). One
short scripted-teacher episode per task uses the registered `t1:contact_v2` tracker (`artifacts/trackers/`, not in git) and skips
without it or without the Menagerie assets. The teachers are PRIVILEGED (`scripted_teacher`): they read the true base pose and joint
state; the judged success is the public task graph, cross-checked against the privileged one."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_wholebody import BODY, _binding, _put, _scene, registry  # noqa: E402,F401  (registry is a fixture)

import rrp.envs.mujoco.legged_tracker as LT  # noqa: E402
from rrp.core.action import NativeCommand  # noqa: E402
from rrp.harness.eval.evaluate import evaluate  # noqa: E402
from rrp.policies.base import POLICIES, make_policy  # noqa: E402
from rrp.policies.teachers import humanoid as TH  # noqa: E402
from rrp.tasks.humanoid import ENDS_AT_ONCE, HUMANOID_REASONS, MANIP_REASONS, humanoid_judge  # noqa: E402
from rrp.tasks.spec import get_task  # noqa: E402

TASKS = ("h_walk", "h_turn", "h_reach", "h_squat_pick", "h_place")
REAL = ("t1", "contact_v2")


# ---------------------------------------------------------------- registration and vocabulary
def test_tasks_registered_with_graph_teacher_and_vocabulary():
    for name in TASKS:
        t = get_task(name)
        assert t.graph == name and t.teacher == f"teacher:{name}" and set(t.envs) == {"mujoco/legged"}
        assert t.build == {"mujoco/legged": "rrp.envs.mujoco.humanoid_scenes:make_humanoid_session"}
        assert t.max_steps == int(50 * t.max_seconds) and t.failure_reasons[0] == "fell" and t.failure_reasons[-1] == "timeout"
        assert set(t.failure_reasons) <= set(HUMANOID_REASONS) | set(MANIP_REASONS)
        assert (Path(TH.__file__).parents[2] / "tasks" / "graphs" / f"{name}.json").is_file()
        assert POLICIES[f"teacher:{name}"] == "rrp.policies.teachers.humanoid:make_manip_teacher_policy"
        assert type(make_policy(f"teacher:{name}")).__name__ == "ManipTeacherPolicy"
    assert set(MANIP_REASONS) == {"drift", "no_grasp", "not_upright", "place_miss"} and "drift" in ENDS_AT_ONCE
    assert set(TH.MANIP_TEACHERS) == set(TASKS)
    with pytest.raises(KeyError, match="no U2/U3 teacher"):
        TH.ManipTeacherPolicy("h_steps")


class Fake:
    def __init__(self, *, pub=False, priv=False, code=None, fell=False):
        self.runtime, self.fell, self.boundary, self._priv, self._code = NS(succeeded=lambda: pub), fell, True, priv, code
        self.scenario = NS(meta={})

    def privileged_success(self):
        return self._priv

    def failure_reason(self):
        return self._code

    def base_pose_truth(self):
        return 0.0, 0.0, 0.0


@pytest.mark.parametrize("code,t,expect", [
    ("drift", 3.0, ("failure", "drift")),                # h_turn: left its spot, ends at once
    ("dropped", 3.0, ("failure", "dropped")),
    ("no_grasp", 40.0, ("failure", "no_grasp")),         # non-terminal codes are read at the end
    ("not_upright", 40.0, ("failure", "not_upright")),
    ("place_miss", 40.0, ("failure", "place_miss")),
])
def test_judge_reads_the_manip_reasons(code, t, expect):
    j = humanoid_judge()(Fake(pub=code != "drift" and t >= 40.0, code=code), t, 40.0)
    assert j.done and (j.outcome, j.failure_reason) == expect and j.success_privileged is False


# ---------------------------------------------------------------- wiring with a stub actor (assets: Menagerie)
def _stub_session(tmp_path, registry, task, seed=0):
    from rrp.envs.mujoco.humanoid_scenes import make_humanoid_session
    _put(tmp_path, BODY, _binding(_scene()), upper=False, seed=5)
    registry()
    return make_humanoid_session(task=task, body=BODY, seed=seed, tracker=f"{BODY}:stub", tracker_kind="learned")


@pytest.mark.menagerie
@pytest.mark.parametrize("task", TASKS)
def test_teacher_command_carries_both_groups_and_labels_its_legs(tmp_path, registry, task):
    s = _stub_session(tmp_path, registry, task)
    assert s.control == "wholebody" and s.scenario.name == task
    tch = TH.MANIP_TEACHERS[task](s)
    assert tch.source == "scripted_teacher" and tch.privileged
    up0 = np.asarray(s.data.qpos[s.binding.held_qadr]).copy()
    for _ in range(100):                                                    # 2 s
        cmd = tch.act()
        assert isinstance(cmd, NativeCommand) and set(cmd.groups) == {"legs", "upper"} and cmd.source == "scripted_teacher"
        assert len(cmd.groups["upper"]) == len(up0) and np.isfinite(cmd.groups["legs"]).all()
        s.step(cmd)
    pol = TH.ManipTeacherPolicy(task)
    pol.reset(None, task, [0], envs=[s])
    assert ("planned_com" in pol.labels[0]) == (TH.MANIP_TEACHERS[task].legs == "planned_com")
    assert TH.MANIP_TEACHERS[task].legs == ("rl_expert" if task in ("h_walk", "h_turn") else "planned_com")


@pytest.mark.menagerie
def test_reach_edit_moves_the_arm_on_the_target_side(tmp_path, registry):
    """Causal use of the scene: the arm the IK drives is the one on the target's side; the other holds its default."""
    moved = {}
    for seed in range(6):
        s = _stub_session(tmp_path, registry, "h_reach", seed)
        tch = TH.MANIP_TEACHERS["h_reach"](s)
        for _ in range(60):
            s.step(tch.act())
        cmd = tch.act().groups["upper"]
        d = np.abs(np.asarray(cmd) - np.asarray(s.binding.q0_held))
        arm = tch.ik.arm[tch.side]
        other = tch.ik.arm["right" if tch.side == "left" else "left"]
        assert d[arm].max() > 0.05 and d[other].max() < 1e-6
        moved.setdefault(tch.side, 0)
        moved[tch.side] += 1
    assert set(moved) == {"left", "right"}                                  # both sides occur across the seeds


# ---------------------------------------------------------------- one scripted-teacher episode per task (real t1:contact_v2)
def _real_tracker():
    pytest.importorskip("torch")
    if REAL not in LT.TRACKERS or not LT.TRACKERS[REAL].actor.exists():    # meta.json is tracked, actor.pt is not
        pytest.skip("needs untracked weights artifacts/trackers/t1/contact_v2/actor.pt (peer store)")


@pytest.mark.menagerie
@pytest.mark.parametrize("task,seed,sec", [("h_walk", 0, 5.0), ("h_turn", 4, 3.5), ("h_reach", 0, 3.0),
                                           ("h_squat_pick", 0, 12.0), ("h_place", 0, 16.5)])
def test_scripted_teacher_succeeds_on_the_task(task, seed, sec):
    _real_tracker()
    pol = make_policy(f"teacher:{task}")
    (ep,) = evaluate(pol, "mujoco/legged", task, BODY, [seed], batch=1, env_kw=dict(tracker="t1:contact_v2", tracker_kind="learned"))
    r = ep.row()
    assert (r["outcome"], r["failure_reason"], r["success_public"], r["success_privileged"]) == ("success", None, True, True), r
    assert r["time"] <= sec + 0.5                                           # a few seconds of sim (squat / place: the whole pick)
    label = pol.labels[0]
    assert "scripted_teacher" in label and task in label
    assert ("planned_com" in label) == (task in ("h_reach", "h_squat_pick", "h_place"))
