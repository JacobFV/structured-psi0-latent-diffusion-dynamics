"""D-146 HJ (docs/architecture.md 14.1): the humanoid judge and the registration of h_steps / h_gap. One synthetic env
per failure reason of the vocabulary; the tasks are registered with `build` (no task if/elif in the env factory)."""
import inspect
import math
from types import SimpleNamespace as NS

import pytest

from rrp.envs.base import ENVS, _factory, env_takes_scene
from rrp.tasks.humanoid import HUMANOID_REASONS, TERMINAL, humanoid_judge
from rrp.tasks.spec import TASKS, get_task


class Fake:
    def __init__(self, *, pub=False, priv=False, code=None, fell=False, boundary=True, psi=None, yaw=0.0):
        self.runtime, self.fell, self.boundary, self._priv, self._code, self._yaw = NS(succeeded=lambda: pub), fell, boundary, priv, code, yaw
        self.scenario = NS(meta={} if psi is None else {"psi_f": psi})

    def privileged_success(self):
        return self._priv

    def base_pose_truth(self):
        return 0.0, 0.0, self._yaw

    def failure_reason(self):
        return self._code


JUDGE = humanoid_judge()
T, BUDGET = 40.0, 40.0


def test_registered_with_build_and_vocabulary():
    for name in ("h_steps", "h_gap"):
        t = get_task(name)
        assert t.failure_reasons == HUMANOID_REASONS and set(t.envs) == {"mujoco/legged", "warp/legged"}
        assert t.build == {"mujoco/legged": "rrp.tasks.humanoid:build_mujoco"} and t.teacher == f"teacher:{name}"
        assert t.judge is not TASKS["waypoint_contact"].judge
    assert _factory("mujoco/legged", "h_gap").__name__ == "build_mujoco"
    assert env_takes_scene("mujoco/legged", "h_steps")                      # scene kwargs (h_frac | level) reach the builder
    assert not env_takes_scene("warp/legged", "h_steps")                    # warp keeps its own factory
    assert _factory("warp/legged", "h_steps").__name__ == ENVS["warp/legged"].split(":")[1]
    assert {"trip", "missed_step", "wall_collision", "wrong_heading"} <= set(HUMANOID_REASONS) and set(TERMINAL) <= set(HUMANOID_REASONS)


CASES = [
    (Fake(fell=True), 3.0, ("fell", "fell")),                                          # fell (env flag, no code)
    (Fake(code="fell"), 3.0, ("fell", "fell")),
    (Fake(code="wall_collision"), 3.0, ("failure", "wall_collision")),                 # terminal env codes end at once
    (Fake(code="hold_lost"), 3.0, ("failure", "hold_lost")),
    (Fake(code="dropped_off_table"), 3.0, ("failure", "dropped")),                     # alias -> dropped
    (Fake(pub=True, code="trip"), T, ("failure", "trip")),                             # non-terminal codes at the end
    (Fake(code="missed_foothold"), T, ("failure", "missed_step")),                     # alias -> missed_step
    (Fake(pub=True, psi=1.0, yaw=0.0), 5.0, ("failure", "wrong_heading")),             # truth: heading off by > 0.3 rad
    (Fake(), T, ("timeout", "timeout")),                                               # budget spent, graph unfinished
]


@pytest.mark.parametrize("env,t,expect", CASES)
def test_one_reason_each(env, t, expect):
    j = JUDGE(env, t, BUDGET)
    assert j.done and (j.outcome, j.failure_reason) == expect and j.success_privileged is False
    assert j.failure_reason in HUMANOID_REASONS


def test_every_reason_is_exercised():
    assert {e[1] for _, _, e in CASES} == set(HUMANOID_REASONS)



def test_success_running_and_strictness():
    assert JUDGE(Fake(pub=True, priv=True), 5.0, BUDGET) == JUDGE(Fake(pub=True, priv=True, psi=1.0), 5.0, BUDGET)
    j = JUDGE(Fake(pub=True, priv=True), 5.0, BUDGET)
    assert (j.done, j.outcome, j.failure_reason, j.success_public, j.success_privileged) == (True, "success", None, True, True)
    assert not JUDGE(Fake(), 5.0, BUDGET).done                                          # running
    assert not JUDGE(Fake(fell=True, boundary=False), 5.0, BUDGET).done                 # only boundary ticks end an episode
    with pytest.raises(ValueError, match="not in the humanoid vocabulary"):
        JUDGE(Fake(code="exploded"), 5.0, BUDGET)                                       # the vocabulary stays closed
    with pytest.raises(RuntimeError, match="must name it"):
        JUDGE(Fake(pub=True, psi=0.1, yaw=0.0), 5.0, BUDGET)                            # no code, heading fine: no guessing
