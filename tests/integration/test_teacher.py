import pytest
from rrp.envs.mujoco.fixtures import make_pick_place_session
from tests.unit.test_data_train_rollout import fixture_teacher_run, teacher_run   # the teacher as a harness.rollout


def test_real_teacher_trace_has_actions_and_success_evidence():
    result = fixture_teacher_run(seed=4)
    assert len(result.actions) > 0
    assert result.success == result.privileged_evaluator_success
    assert result.success


@pytest.mark.parametrize("gripper", ["parallel", "three_finger"])
def test_teacher_valid_for_both_compatible_modules(gripper):
    ok = 0
    for seed in range(8):
        r = fixture_teacher_run(seed=seed, gripper=gripper)
        ok += int(r.success and r.privileged_evaluator_success)
    assert ok >= 7


def test_infeasible_episode_is_recorded_not_attempted():
    s = make_pick_place_session(seed=0)
    s.teleport_object("target_zone", [1.5, 0.0, 0.0005])
    r = teacher_run(s)
    assert r.failure_reason.startswith("infeasible") and r.steps == 0
    assert "transport" in r.feasibility["unreachable"]
