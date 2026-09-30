"""RP3 (D-146, docs/architecture.md 14.1): the data-collection and training loops that stepped sessions themselves run on
`harness.rollout` + hooks. The outputs of every loop (teacher-episode collection incl. DART / proximity guard /
infeasible layouts, the VLM keyframe replay, the adaptation teacher prefix, the fixture teacher episode) on the smallest
procedural arm scene were recorded from the private loops BEFORE the port (`loop.data.*`, `loop.train.*` in
tests/data/golden.json) and must stay byte-identical (floats rounded to 5 decimals)."""
import numpy as np
import pytest

from tests.unit.test_golden import _pi, golden  # noqa: F401  (golden is the recording fixture)
from tests.unit.test_eval_rollout import rollout_guard  # noqa: F401  (fixture)
from tests.unit.test_ladder_rollout import _digest

pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")

ROBOT = "parm5_pg2"


def _record(rec) -> list:
    """Everything a collected episode holds except wall time and observation ids (a process-global serial)."""
    pub, prv = rec.public, rec.private
    meta = {k: v for k, v in pub["meta"].items() if k != "wall_s"}
    return [meta, [_pi(pi) for pi in pub["inputs"]], pub["actions"], pub["q0"], pub["statuses"], pub["action_space"],
            prv["labels"], prv["phases"], prv["manipulators"], prv["slots"]]


def test_collect_teacher_episodes(golden):
    from rrp.envs.mujoco.fixtures import make_pick_place_session
    from rrp.harness.data.collect import collect_teacher_episode
    S = make_pick_place_session
    rows = [_record(collect_teacher_episode(S(seed=3), max_steps=14, episode_id="a")),
            _record(collect_teacher_episode(S(seed=3), max_steps=600, episode_id="b")),          # to the FSM's end
            _record(collect_teacher_episode(S(seed=2), max_steps=160, exec_noise=0.08, noise_seed=2, teacher_version="v2",
                                            dart_safety="phase", dart_descent_sigma=0.02, episode_id="c")),
            _record(collect_teacher_episode(S(seed=4), max_steps=120, exec_noise=0.08, noise_seed=5,
                                            dart_safety="proximity", episode_id="d"))]
    inf = S(seed=0)
    inf.teleport_object("target_zone", [1.5, 0.0, 0.0005])                                       # unreachable layout
    rows.append(_record(collect_teacher_episode(inf, max_steps=30, episode_id="e")))
    golden("loop.data.collect", _digest(rows))


def test_vlm_replay_render(golden, monkeypatch):
    """replay_render re-simulates a teacher episode and renders keyframes (renderer stubbed: every frame is the sim
    state at that tick, so the digest sees exactly what would have been drawn)."""
    import rrp.policies.nets.backbone as B
    from rrp.harness.data.vlm_features import replay_render

    class Renderer:
        def __init__(self, model, size=256):
            pass

        def cameras(self, want):
            return list(want)

        def render(self, data, cams):
            return [np.concatenate([[data.time], data.qpos[:6], data.xpos[-1]])]

        def close(self):
            pass
    monkeypatch.setattr(B, "Renderer", Renderer)
    ref = np.random.default_rng(0).normal(size=(40, 6)) * 0.01
    out = [replay_render(ROBOT, 3, 1, every=4, max_steps=25, ref_q0=ref),
           replay_render(ROBOT, 4, 0, every=40, max_steps=600)]
    golden("loop.data.vlm_replay", _digest(out))


def test_adaptation_teacher_prefix(golden):
    from rrp.bodies.catalog import workbench_robots
    from rrp.envs.mujoco.scenario import BUILDERS
    from rrp.envs.mujoco.session import Session
    from rrp.harness.train.online_episodes import EpisodeState, PolicyAdapter, event_boundary, teacher_prefix
    robot = workbench_robots()[ROBOT]()
    rows = []
    for seed, n, boundary in ((3, 12, lambda st: st.steps >= 6),            # boundary fires
                              (4, 8, lambda st: False),                     # step allowance ends the prefix
                              (3, 600, lambda st: False),                   # the teacher finishes the task: terminal
                              (3, 600, event_boundary("grasp"))):           # the public grasp event
        s = Session(BUILDERS["pick_place"](robot, seed, n_distractors=seed % 3), seed=seed)
        pol = PolicyAdapter(None, None, "cpu")
        st = EpisodeState(s, seed, n)
        fired = teacher_prefix(pol, st, boundary, n)
        rows.append([fired, st.steps, st.done, st.outcome, st.tag, pol.prev[id(s)], s.data.qpos, float(s.data.time),
                     s.executor.queue is None or len(s.executor.queue)])
    golden("loop.train.teacher_prefix", _digest(rows))


def teacher_run(session, max_steps=600):
    """The scripted pick_place teacher on `session` as one harness.rollout (the loop the retired
    `policies.teachers.arm.run_teacher_episode` was): infeasible layouts are recorded, not attempted; the teacher's end
    ends the episode; one hold tick, then the privileged verdict. Returns a namespace of the trace and the verdicts."""
    from types import SimpleNamespace as NS
    from rrp.harness import rollout as R
    from rrp.harness import hooks as H
    from rrp.policies.teachers import TeacherPolicy
    from rrp.policies.teachers.arm import PickPlaceTeacher
    teacher = PickPlaceTeacher(session)
    res = NS(actions=[], phases=[], rejected_commands=0, feasibility=teacher.feasibility(), steps=0)
    res.failure_reason = None
    if not res.feasibility["feasible"]:
        res.failure_reason = f"infeasible:{','.join(res.feasibility['unreachable'])}"
        return res
    rec = H.Recorder(on_act=lambda i, env, act: (res.actions.append(act.command.groups), res.phases.append(teacher.phase)),
                     on_step=lambda i, env, act, step: setattr(res, "rejected_commands",
                                                               res.rejected_commands + int(bool(step.rejected))))
    pol = TeacherPolicy("pick_place", lambda e: teacher, "test", ("joint_position", "gripper"))
    ep = R.rollout(lambda sd: session, pol, H.budget_task("pick_place", session.spec.env_id), [session.seed], batch=1,
                   max_steps=max_steps, hooks=[rec, H.EndWhen(lambda i, e: bool(teacher.done)), H.Settle(1)])[0]
    res.steps, res.success, res.privileged_evaluator_success = ep.steps, bool(ep.success_public), bool(ep.success_privileged)
    if not res.privileged_evaluator_success:
        res.failure_reason = f"ended_in_phase:{teacher.phase}"
    return res


def fixture_teacher_run(seed=0, gripper="parallel", max_steps=600):
    from rrp.envs.mujoco.fixtures import make_pick_place_session
    return teacher_run(make_pick_place_session(seed=seed, gripper=gripper), max_steps)


def test_fixture_teacher_episode(golden):
    from rrp.envs.mujoco.fixtures import make_pick_place_session
    rows = []
    for r in (fixture_teacher_run(seed=4), fixture_teacher_run(seed=1, gripper="three_finger")):
        rows.append([r.actions, r.phases, r.steps, r.success, r.privileged_evaluator_success, r.failure_reason,
                     r.rejected_commands])
    s = make_pick_place_session(seed=0)
    s.teleport_object("target_zone", [1.5, 0.0, 0.0005])
    r = teacher_run(s)
    rows.append([r.actions, r.steps, r.failure_reason, r.feasibility])
    golden("loop.data.fixture_teacher", _digest(rows))


def test_grpo_teacher_prefix_that_finishes_the_task(golden):
    """A curriculum prefix long enough for the scripted teacher to finish: those episodes end as
    `teacher_prefix_terminal` (shared by the group members of a repeated seed) and never reach the learned policy."""
    from tests.unit.test_golden import _row_h, _tiny_latent
    from rrp.harness.train.latent_grpo import run_episodes
    si, R, _ = _tiny_latent()
    rows = run_episodes(si, R, ROBOT, [3, 3, 4], max_steps=700, prefix_steps=650)
    assert {r["outcome"] for r in rows} == {"teacher_prefix_terminal"}
    golden("loop.grpo.prefix_terminal", _digest([_row_h(r) for r in rows]))


def test_data_and_train_loops_are_rollouts(rollout_guard, monkeypatch):
    """Every ported loop ticks its session inside harness.rollout (rollout_guard fails a tick made outside one)."""
    import rrp.policies.nets.backbone as B
    from rrp.envs.mujoco.fixtures import make_pick_place_session
    from rrp.harness.data.collect import collect_teacher_episode
    from rrp.harness.data.vlm_features import replay_render
    from rrp.harness.train.latent_grpo import run_episodes
    from tests.unit.test_golden import _tiny_latent

    class Renderer:
        def __init__(self, model, size=256):
            pass

        def cameras(self, want):
            return list(want)

        def render(self, data, cams):
            return [np.zeros(3)]

        def close(self):
            pass
    monkeypatch.setattr(B, "Renderer", Renderer)
    calls = rollout_guard["calls"]
    collect_teacher_episode(make_pick_place_session(seed=3), max_steps=4, episode_id="x")
    assert rollout_guard["calls"] == calls + 2          # the episode + its nested settle rollout
    replay_render(ROBOT, 3, 1, every=4, max_steps=5)
    assert rollout_guard["calls"] == calls + 3
    fixture_teacher_run(seed=4, max_steps=3)
    assert rollout_guard["calls"] == calls + 5
    si, R, _ = _tiny_latent()
    run_episodes(si, R, ROBOT, [3, 3, 4], max_steps=6, prefix_steps=3)
    assert rollout_guard["calls"] > calls + 3      # the batched teacher prefix, then the policy episodes
