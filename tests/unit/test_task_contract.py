"""D-146 F2 (docs/architecture.md 14.1): the task / eval contract. Every ENVS id builds through `evaluate()` with and
without a scene (the `scene=` crash: the factories of warp / simple take none and used to receive `scene={}`); a scene
for such an env is a negotiation reason; every task's teacher is a resolvable POLICIES key or an explicit None; the
per-family string comparisons of the eval front-ends are task data (`build`, `scene`, `hooks`, `max_steps`)."""
import importlib
import inspect
import sys
from types import SimpleNamespace as NS

import pytest

from rrp.envs.base import (ENVS, ActionSpace, BodyInfo, EnvSpec, SceneUnsupported, env_failure_reason, make_env,
                           scene_reasons)
from rrp.harness.eval.evaluate import evaluate, matrix, task_hooks
from rrp.harness.hooks import TASK_HOOKS
from rrp.harness.rollout import Incompatible, rollout
from rrp.policies.base import POLICIES, Act, PolicyInfo, Requirements, make_policy
from rrp.tasks.spec import TASKS, Judgement, TaskSpec, get_task, tasks_in

THIS = sys.modules[__name__]
BODIES = {"mujoco/arm": ("pick_place", "parm6_pg2"), "mujoco/dual": ("handover", ["parm5l_pg2", "parm6_pg2"]),
          "mujoco/legged": ("waypoint_contact", "go2"), "warp/legged": ("locomotion", "go2"),
          "simple": ("simple/G1WholebodyTabletopGraspMP-v0", "g1_simple"), "computerworld": ("cw/calc_sum", "cw_pointer")}
REAL = ("mujoco/arm", "mujoco/dual")          # the others need assets / GPU / Isaac / the CW wheel: stub factories


class Hold:
    info = PolicyInfo("hold", "mock", "v0", Requirements(frozenset(), observations=frozenset()))

    def reset(self, spec, task, seeds, *, envs=None):
        pass

    def act(self, obs):
        return {i: Act(None) for i in obs}


class StubEnv:
    """Strict constructor like WarpTrackerEnv / SimpleEnv: an unknown keyword (scene=) is a TypeError."""
    made: list = []

    def __init__(self, env_id, task, body, seed, *, level=0):
        self.t, self.level, self.seed = 0.0, level, seed
        self.spec = EnvSpec(env_id=env_id, backend="mujoco", task=task, control_hz=10.0,
                            bodies=[BodyInfo(robot=0, family="x", key=str(body), robot_spec_hash="h")],
                            action_spaces=[ActionSpace(group="g", kind="joint_position", width=1, rate_hz=10.0)],
                            capabilities=[])
        StubEnv.made.append(self)

    def reset(self, seed=None):
        return self.observe()

    def observe(self):
        return NS(sensor_time=self.t)

    def step(self, command):
        self.t += 0.1
        return NS(observation=self.observe(), time=self.t, rejected=None)

    def close(self):
        pass


def _stub_factory(env_id, real_signature):
    """A factory with the REAL factory's parameter list (scene or not) that builds a StubEnv."""
    takes_scene = "scene" in real_signature.parameters
    if takes_scene:
        def factory(*, task, body, seed=0, scene=None, **kw):
            e = StubEnv(env_id, task, body, seed, **kw)
            e.scene = scene
            return e
    else:
        def factory(*, task, body, seed=0, **kw):
            e = StubEnv(env_id, task, body, seed, **kw)
            e.scene = None
            return e
    return factory


def _install(monkeypatch, env_id):
    """Real env ids stay real; the heavy ones become stub factories with the real parameter list."""
    real = inspect.signature(getattr(importlib.import_module(ENVS[env_id].split(":")[0]), ENVS[env_id].split(":")[1]))
    task, body = BODIES[env_id]
    if env_id in REAL:
        return task, body
    name = "stub_" + env_id.replace("/", "_")
    setattr(THIS, name, _stub_factory(env_id, real))
    monkeypatch.setitem(ENVS, env_id, f"{__name__}:{name}")
    monkeypatch.setitem(TASKS, task, TaskSpec(task, {env_id: {}}, 0.3, lambda env, t, T: Judgement(t >= T, "timeout" if t >= T
                                                                                         else None, "timeout" if t >= T else None),
                                             failure_reasons=("timeout",)))
    return task, body


def test_bodies_cover_every_env():
    assert set(BODIES) == set(ENVS)


@pytest.mark.parametrize("env_id", sorted(ENVS))
def test_every_env_builds_through_evaluate_with_and_without_scene(monkeypatch, env_id):
    task, body = _install(monkeypatch, env_id)
    takes_scene = "scene" in inspect.signature(getattr(importlib.import_module(ENVS[env_id].split(":")[0]),
                                                       ENVS[env_id].split(":")[1])).parameters
    StubEnv.made.clear()
    # no scene requested: nothing is passed (this was `scene={}` and a TypeError for warp / simple)
    eps = evaluate(Hold(), env_id, task, body, [3], batch=1, max_steps=2)
    assert [(e.seed, e.steps) for e in eps] == [(3, 2)]
    assert not scene_reasons(env_id, task, None)
    # an explicit scene: accepted by scene-taking factories, a negotiation reason (not a TypeError) otherwise
    if takes_scene:
        eps = evaluate(Hold(), env_id, task, body, [3], batch=1, max_steps=2, scene=lambda sd: {})
        assert len(eps) == 1
        if env_id not in REAL:
            assert StubEnv.made[-1].scene == {}
    else:
        with pytest.raises(Incompatible, match="declares a scene; env .* takes none"):
            evaluate(Hold(), env_id, task, body, [3], batch=1, max_steps=2, scene={"n_distractors": 1})
        with pytest.raises(SceneUnsupported, match="takes none"):
            make_env(env_id, task=task, body=body, seed=0, scene={})
    if env_id not in REAL:                    # --env-kw reaches the factory
        StubEnv.made.clear()
        evaluate(Hold(), env_id, task, body, [3], batch=1, max_steps=1, env_kw={"level": 2})
        assert StubEnv.made[-1].level == 2


def test_task_scene_is_the_default_and_matrix_marks_scene_mismatch(monkeypatch):
    seen = []
    task = TaskSpec("scene_task", {"stub/a": {}, "stub/b": {}}, 0.2,
                    lambda env, t, T: Judgement(t >= T, "timeout" if t >= T else None), scene=lambda sd: {"k": sd})
    monkeypatch.setitem(TASKS, "scene_task", task)
    setattr(THIS, "stub_a", lambda *, task, body, seed=0, scene=None, **kw: seen.append((seed, scene)) or StubEnv(
        "stub/a", task, body, seed))
    setattr(THIS, "stub_b", lambda *, task, body, seed=0, **kw: StubEnv("stub/b", task, body, seed, **kw))
    monkeypatch.setitem(ENVS, "stub/a", f"{__name__}:stub_a")
    monkeypatch.setitem(ENVS, "stub/b", f"{__name__}:stub_b")
    evaluate(Hold(), "stub/a", "scene_task", "b", [5, 6], batch=2, max_steps=1)
    assert seen == [(5, {"k": 5}), (6, {"k": 6})]                       # the task's scene, per seed
    seen.clear()
    evaluate(Hold(), "stub/a", "scene_task", "b", [5], batch=1, max_steps=1, scene={"k": "override"})
    assert seen == [(5, {"k": "override"})]                             # the caller's wins
    rows = matrix([Hold()], [("stub/a", "b"), ("stub/b", "b")], ["scene_task"])
    assert [r["status"] for r in rows] == ["accepted", "n/a"] and "takes none" in rows[1]["reasons"][0]


def test_task_build_replaces_the_factory_task_switch(monkeypatch):
    calls = []
    setattr(THIS, "build_x", lambda *, task, body, seed=0, **kw: calls.append(task) or StubEnv("stub/x", task, body, seed))
    monkeypatch.setitem(ENVS, "stub/x", f"{__name__}:build_never")            # never resolved: the task's builder wins
    monkeypatch.setitem(TASKS, "built", TaskSpec("built", {"stub/x": {}}, 0.2, lambda env, t, T: Judgement(True, "timeout"),
                                                 build={"stub/x": f"{__name__}:build_x"}))
    make_env("stub/x", task="built", body="b")
    assert calls == ["built"]


def test_env_failure_reason_is_optional_and_recorded(monkeypatch):
    class Failing(StubEnv):
        def failure_reason(self):
            return "tipped"
    setattr(THIS, "stub_f", lambda *, task, body, seed=0, **kw: Failing("stub/f", task, body, seed))
    monkeypatch.setitem(ENVS, "stub/f", f"{__name__}:stub_f")
    t = TaskSpec("fail_task", {"stub/f": {}}, 0.2, lambda env, t, T: Judgement(t >= T, "failure", "timeout"))
    eps = rollout(lambda sd: make_env("stub/f", task="fail_task", body="b", seed=sd), Hold(), t, [0], batch=1)
    assert eps[0].metrics["env_failure_reason"] == "tipped"
    assert env_failure_reason(StubEnv("stub/f", "t", "b", 0)) is None


def test_rollout_negotiates_every_env_of_a_group():
    class OnlyBody:
        info = PolicyInfo("only", "mock", "v0", Requirements(frozenset(), observations=frozenset(), bodies=frozenset({"b0"})))

    made = iter(["b0", "b1"])
    t = TaskSpec("neg", {"stub/n": {}}, 0.2, lambda env, t, T: Judgement(True, "timeout"))
    with pytest.raises(Incompatible, match="b1"):                        # the second env's body, not just the first's
        rollout(lambda sd: StubEnv("stub/n", "neg", next(made), sd), OnlyBody(), t, [0, 1], batch=2)


def test_task_max_steps_and_hooks_are_task_data(monkeypatch):
    t = TaskSpec("budget", {"stub/n": {}}, 5.0, lambda env, t, T: Judgement(t >= T, "timeout" if t >= T else None),
                 max_steps=3)
    eps = rollout(lambda sd: StubEnv("stub/n", "budget", "b", sd), Hold(), t, [0], batch=1)
    assert eps[0].steps == 3                                             # no max_steps argument: the task's budget
    assert [type(h).__name__ for h in task_hooks("pick_place", "mujoco/arm")] == ["Feasibility", "SessionRecord"]
    assert [type(h).__name__ for h in task_hooks("handover", "mujoco/dual")] == ["Feasibility", "Settle", "SessionRecord"]
    assert [type(h).__name__ for h in task_hooks("reach_pose", "mujoco/arm")] == ["SessionRecord"]
    assert task_hooks("h_steps", "warp/legged") == [] and task_hooks("cw/calc_sum", "computerworld") == []
    assert get_task("pick_place").scene(4) == {"n_distractors": 1}


def test_every_task_declares_a_resolvable_teacher_or_none():
    for t in TASKS.values():
        assert set(t.hooks) <= set(TASK_HOOKS), t.name
        assert t.failure_reasons or t.name.startswith("stub"), t.name
        if t.teacher is None:
            continue
        assert t.teacher in POLICIES, f"{t.name}: teacher {t.teacher!r} is not a POLICIES key"
        pol = make_policy(t.teacher)                                     # builds no env
        assert pol.info.source == "scripted_teacher" and pol.info.requires.tasks == {t.name}, t.name
        assert pol.info.requires.privileged
    with pytest.raises(KeyError, match="unknown policy"):
        make_policy("teacher:no_such_task")


def test_dual_tasks_have_one_definition():
    assert tasks_in("mujoco/dual") == ("support_insert", "handover", "assign_left", "assign_right",
                                       "pivot_against_surface")
    src = importlib.import_module("rrp.policies.teachers")
    assert not hasattr(src, "DUAL_TASKS") and not hasattr(src, "ARM_TASKS") and not hasattr(src, "LEGGED_TASKS")


def test_no_family_string_comparisons_in_the_eval_front_ends():
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "src" / "rrp"
    pat = re.compile(r"""(env_id|a\.env|a\.task|\btask)\s*(==|!=|in)\s*[("'](mujoco|warp|simple|computerworld|pick_place|h_steps)""")
    for f in ("cli/harness.py", "harness/eval/evaluate.py", "harness/rollout.py", "policies/teachers/__init__.py"):
        bad = [ln for ln in (root / f).read_text().splitlines() if pat.search(ln)]
        assert not bad, (f, bad)
