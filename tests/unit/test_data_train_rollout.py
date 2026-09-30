"""RP3 (D-146, docs/architecture.md 14.1): the data-collection and training loops that stepped sessions themselves run on
`harness.rollout` + hooks. The outputs of every loop (teacher-episode collection incl. DART / proximity guard /
infeasible layouts, the VLM keyframe replay, the adaptation teacher prefix, the fixture teacher episode) on the smallest
procedural arm scene were recorded from the private loops BEFORE the port (`loop.data.*`, `loop.train.*` in
tests/data/golden.json) and must stay byte-identical (floats rounded to 5 decimals)."""
import numpy as np
import pytest

from tests.unit.test_golden import _pi, golden  # noqa: F401  (golden is the recording fixture)
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
    from rrp.harness.train.rollout import EpisodeState, PolicyAdapter, event_boundary, teacher_prefix
    robot = workbench_robots()[ROBOT]()
    rows = []
    for seed, n, boundary in ((3, 12, lambda st: st.steps >= 6),            # boundary fires
                              (4, 8, lambda st: False),                     # step allowance ends the prefix
                              (3, 600, event_boundary("grasp"))):           # the public grasp event
        s = Session(BUILDERS["pick_place"](robot, seed, n_distractors=seed % 3), seed=seed)
        pol = PolicyAdapter(None, None, "cpu")
        st = EpisodeState(s, seed, n)
        fired = teacher_prefix(pol, st, boundary, n)
        rows.append([fired, st.steps, st.done, st.outcome, st.tag, pol.prev[id(s)], s.data.qpos, float(s.data.time),
                     s.executor.queue is None or len(s.executor.queue)])
    golden("loop.train.teacher_prefix", _digest(rows))


def test_fixture_teacher_episode(golden):
    from rrp.envs.mujoco.fixtures import make_pick_place_session
    from rrp.policies.teachers.arm import PickPlaceTeacher, run_fixture_pick_place, run_teacher_episode
    rows = []
    for r in (run_fixture_pick_place(seed=4, max_control_steps=600), run_fixture_pick_place(seed=1, gripper="three_finger")):
        rows.append([r.actions, r.phases, r.steps, r.success, r.privileged_evaluator_success, r.failure_reason,
                     r.rejected_commands])
    s = make_pick_place_session(seed=0)
    s.teleport_object("target_zone", [1.5, 0.0, 0.0005])
    r = run_teacher_episode(s, PickPlaceTeacher(s))
    rows.append([r.actions, r.steps, r.failure_reason, r.feasibility])
    golden("loop.data.fixture_teacher", _digest(rows))
