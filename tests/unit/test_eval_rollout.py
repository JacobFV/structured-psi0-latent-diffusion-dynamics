"""RP2 (D-146, docs/architecture.md 14.1): the arm / dual eval loops that still stepped sessions themselves run on
`harness.rollout` + hooks. The rows of every loop (teacher quality, policy quality, dual audit, dual validate, causal
window + episode protocols, disturbance test, arm and dual videos) on the smallest procedural scenes (random-weight
nets: plumbing only, no number is a result) were recorded from the private loops BEFORE the port (`loop.eval.*` in
tests/data/golden.json) and must stay byte-identical (floats rounded to 5 decimals)."""
import sys
import types

import numpy as np
import pytest

from tests.unit.test_golden import _tiny_bc, _tiny_latent, golden  # noqa: F401  (golden is the recording fixture)
from tests.unit.test_ladder_rollout import _digest

pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")

ROBOT = "parm5_pg2"
DUAL = "parm5l_pg2__parm6_pg2"


def _clean(row, drop=("wall_s",)):
    if isinstance(row, dict):
        return {k: _clean(v, drop) for k, v in row.items() if k not in drop}
    if isinstance(row, (list, tuple)):
        return [_clean(v, drop) for v in row]
    return row


# ------------------------------------------------------------------ arm teacher / policy quality
def test_teacher_quality_episodes(golden):
    from rrp.harness.eval.teacher_quality import run_quality_episode
    rows = [run_quality_episode(ROBOT, sd, "v1", max_steps=14, keep_trace=True) for sd in (3, 4, 50)]
    rows.append(run_quality_episode(ROBOT, 3, "v1", max_steps=10, obj_friction=0.8, obj_mass=1.2))
    golden("loop.eval.quality.teacher", _digest(_clean(rows)))


def test_teacher_quality_full_episode_and_frames(golden):
    from rrp.harness.eval.teacher_quality import run_quality_episode
    seen = []

    class R:
        def update_scene(self, d, camera=None):
            seen.append(camera)

        def render(self):
            return np.zeros((2, 2, 3), np.uint8)
    out = []
    fr = dict(renderer=R(), camera="front", every=7, out=out,
              caption=lambda img, s, teacher, k: (k, teacher.phase, round(float(s.data.time), 5)))
    row = run_quality_episode(ROBOT, 3, "v1", max_steps=600, frames=fr)
    golden("loop.eval.quality.teacher_full", _digest([_clean(row), out, seen]))


def test_policy_quality_episodes(golden):
    from rrp.harness.eval.teacher_quality import run_policy_quality_episode
    pol = _tiny_bc()
    rows = [run_policy_quality_episode(pol, ROBOT, sd, label="tiny", max_steps=14) for sd in (3, 50)]
    rows.append(run_policy_quality_episode(_tiny_bc(), ROBOT, 3, label="tiny", max_steps=9, ckpt="x.pt",
                                           source_labels=True))
    golden("loop.eval.quality.policy", _digest(_clean(rows)))


# ------------------------------------------------------------------ dual audit / validate
def test_dual_audit_episodes(golden):
    from rrp.harness.eval.dual_teacher_quality import run_audit_episode
    rows = [run_audit_episode("support_insert", DUAL, 3, max_steps=12),
            run_audit_episode("support_insert", DUAL, 3, max_steps=14, noise=0.03, burst=(4, 2)),
            run_audit_episode("support_insert", DUAL, 3, max_steps=14, noise=0.03, burst=(1, 1), phase_gate=True,
                              stop_after_success=3, teacher_version="v3", source_labels=True),
            run_audit_episode("support_insert", DUAL, 2, max_steps=12)]
    golden("loop.eval.dual.audit", _digest(_clean(rows)))


def test_dual_validate_episodes(golden):
    """dual_validate lives in the policies layer, which may not import harness.rollout (test_layering): its episode is
    policies.teachers.dual.run_dual_teacher_episode (not an RP2 file); the golden guards it and the eval-layer ports."""
    from rrp.policies.teachers.dual_validate import run_one
    rows = [run_one("support_insert", DUAL, sd, max_steps=12) for sd in (3, 2)]
    rows.append(run_one("support_insert", "no_such_pair", 3, max_steps=5))     # errors are recorded as data
    for r in rows:
        r["error"] = None if "error" not in r else "recorded"
    golden("loop.eval.dual.validate", _digest(_clean(rows)))


# ------------------------------------------------------------------ latent causal protocols
def _causal_stack():
    si, R, P = _tiny_latent()
    return si, R, P


def test_causal_window_protocol(golden):
    from rrp.harness.eval.latent_causal import window_protocol
    si, R, P = _causal_stack()
    rows = window_protocol(si, R, P, ROBOT, [3, 4, 50], decision_ticks=(4, 8), window=3, replan=4,
                           conditions=("control_replay", "rel+x", "rel+y", "rel+xy", "sum_xy", "rand", "cf+x", "zero",
                                       "shuffle", "focus_swap"), edit_steps=3, log=lambda *a, **k: None)
    golden("loop.eval.causal.window", _digest(rows))


def test_causal_episode_protocol(golden):
    from rrp.harness.eval.latent_causal import episode_protocol
    si, R, P = _causal_stack()
    rows = episode_protocol(si, R, P, ROBOT, [3, 4, 50], conditions=("control", "rel+x", "rand", "zero", "shuffle",
                                                                     "cf-x", "focus_swap", "chain_A_then_B"),
                            replan=4, max_steps=10, chain_t1=4, edit_steps=3, log=lambda *a, **k: None)
    golden("loop.eval.causal.episode", _digest(rows))


def test_disturbance_test(golden):
    from rrp.harness.eval.latent_eval import disturbance_test
    si, R, _ = _tiny_latent()
    rows = disturbance_test(si, R, ROBOT, [3], warmup_ticks=9, hold_ticks=3)
    si2, R2, _ = _tiny_latent()
    rows += disturbance_test(si2, R2, ROBOT, [3, 4], warmup_ticks=5, hold_ticks=2, joint_offset=0.05, joint_index=2,
                             prev_action="own")
    golden("loop.eval.disturbance", _digest(rows))


# ------------------------------------------------------------------ latency fixtures
def test_latency_warmup_state(golden):
    """The latency suites time on a fixture stepped 10 hold ticks first; that state is unchanged by the port."""
    from rrp.envs.mujoco.fixtures import make_pick_place_session
    from rrp.harness.eval.hooks import warm_up
    s = warm_up(make_pick_place_session(seed=5, n_distractors=2), 10)
    golden("loop.eval.latency_warmup", _digest([float(s.data.time), s.data.qpos.tolist(), s.data.qvel.tolist()]))


# ------------------------------------------------------------------ videos
def _video_env(monkeypatch, tmp_path):
    frames_seen: dict = {}

    def mimsave(path, frames, fps=None, quality=None):
        frames_seen["name"] = str(path.name)
        frames_seen["frames"] = frames
        frames_seen["fps"] = fps

    fake = types.ModuleType("imageio")
    fake.mimsave = mimsave
    monkeypatch.setitem(sys.modules, "imageio", fake)
    monkeypatch.setenv("MUJOCO_GL", "egl")
    for m in ("rrp.harness.eval.video_arm", "rrp.harness.eval.video_dual"):
        monkeypatch.delitem(sys.modules, m, raising=False)
    import mujoco
    cams = []

    class Rend:
        def __init__(self, model, h, w):
            pass

        def update_scene(self, data, camera=None):
            cams.append(camera)

        def render(self):
            return np.zeros((2, 2, 3), np.uint8)
    monkeypatch.setattr(mujoco, "Renderer", Rend, raising=False)
    return frames_seen, cams


def _video_result(seen, cams, out):
    name = seen["name"][10:]                                   # strip the date
    idx = (out / "INDEX.md").read_text().replace(seen["name"], name)
    idx = "\n".join(l.replace(seen["name"][:10], "DATE") for l in idx.splitlines())
    return [name, seen["fps"], [f for f in seen["frames"]], cams, idx]


def _cap(monkeypatch, mod):
    monkeypatch.setattr(mod, "caption", lambda img, lines: (img.shape, list(lines)))


def _arm_args(tmp_path, **kw):
    a = dict(robot=ROBOT, task="pick_place", seeds="3,4", source="scripted_teacher", checkpoint=None, prefix=4,
             replan=4, out=str(tmp_path), camera="front", width=8, height=8, every=3, fps=5, max_steps=10,
             teacher_prefix_steps=0, patient=0, tag="")
    a.update(kw)
    return types.SimpleNamespace(**a)


@pytest.mark.parametrize("source", ["scripted_teacher", "learned", "learned_latent", "prefix"])
def test_video_arm(golden, monkeypatch, tmp_path, source):
    seen, cams = _video_env(monkeypatch, tmp_path)
    import importlib
    va = importlib.import_module("rrp.harness.eval.video_arm")
    _cap(monkeypatch, va)
    kw = {}
    if source in ("learned", "learned_latent", "prefix"):
        from rrp.policies import bc
        monkeypatch.setattr(bc.LearnedPolicy, "from_checkpoint",
                            classmethod(lambda cls, path, device="cpu", **k: _tiny_bc()))
    if source in ("learned_latent", "prefix"):
        si, R, _ = _tiny_latent()
        from rrp.policies import bundles
        from rrp.policies import latent as lat
        from rrp.policies.nets import checkpoint as ck
        monkeypatch.setattr(lat.LatentPolicy, "from_checkpoint", classmethod(lambda cls, path, device="cpu", **k: si))
        monkeypatch.setattr(bundles, "load_representation", lambda path, dev: (None, None, R, None, None))
        monkeypatch.setattr(ck, "load_checkpoint", lambda path, **k: {"config": {"representation": "rep.pt"}})
    a = _arm_args(tmp_path, source="learned_latent" if source == "prefix" else source,
                  checkpoint=None if source == "scripted_teacher" else "run/policy.pt",
                  teacher_prefix_steps=3 if source == "prefix" else 0, **kw)
    outs = []
    for sd in (3, 4):
        a.seeds = str(sd)
        va.run(a)
        outs.append(_video_result(seen, cams, tmp_path))
        cams.clear()
        (tmp_path / "INDEX.md").unlink()
    golden(f"loop.eval.video_arm.{source}", _digest(outs))


def _dual_args(tmp_path, **kw):
    a = dict(task="support_insert", pair=DUAL, seeds="3", source="scripted_teacher", checkpoint=None, probe=None,
             replan=4, out=str(tmp_path), camera="front", width=8, height=8, every=2, fps=5, max_steps=9)
    a.update(kw)
    return types.SimpleNamespace(**a)


@pytest.mark.parametrize("source", ["scripted_teacher", "learned_latent"])
def test_video_dual(golden, monkeypatch, tmp_path, source):
    seen, cams = _video_env(monkeypatch, tmp_path)
    import importlib
    vd = importlib.import_module("rrp.harness.eval.video_dual")
    _cap(monkeypatch, vd)
    if source == "learned_latent":
        si, R, P = _tiny_latent(dual=True)
        from rrp.policies import bundles
        from rrp.policies import latent as lat
        from rrp.policies.nets import checkpoint as ck
        monkeypatch.setattr(lat.DualLatentPolicy, "from_checkpoint",
                            classmethod(lambda cls, path, device="cpu", **k: si))
        monkeypatch.setattr(bundles, "load_representation", lambda path, dev: (None, None, R, P, None))
        monkeypatch.setattr(ck, "load_checkpoint", lambda path, **k: {"config": {"representation": "rep.pt"}})
    a = _dual_args(tmp_path, source=source, checkpoint=None if source == "scripted_teacher" else "run/policy.pt")
    outs = []
    for sd in (3, 2):                           # 2 = infeasible layout (skipped for the scripted teacher)
        a.seeds = str(sd)
        seen.clear()
        vd.run(a)
        if seen:
            outs.append(_video_result(seen, cams, tmp_path))
            cams.clear()
            (tmp_path / "INDEX.md").unlink()
        else:
            outs.append("skipped")
    golden(f"loop.eval.video_dual.{source}", _digest(outs))


# ------------------------------------------------------------------ every loop is a rollout
@pytest.fixture
def rollout_guard(monkeypatch):
    """Counts harness.rollout calls and fails any session tick made outside one (the loops must not step sessions
    themselves; policy-internal look-aheads are rollouts too)."""
    from rrp.envs.mujoco.session import Session
    from rrp.harness import rollout as R
    state = dict(depth=0, calls=0)
    orig_roll, orig_step = R.rollout, Session.step

    def roll(*a, **k):
        state["depth"] += 1
        state["calls"] += 1
        try:
            return orig_roll(*a, **k)
        finally:
            state["depth"] -= 1

    def step(self, *a, **k):
        assert state["depth"] > 0, "session ticked outside harness.rollout"
        return orig_step(self, *a, **k)
    monkeypatch.setattr(R, "rollout", roll)
    monkeypatch.setattr(Session, "step", step)
    from rrp.envs.mujoco.dual import DualSession
    orig_dual = DualSession.step

    def dual_step(self, *a, **k):
        assert state["depth"] > 0, "dual session ticked outside harness.rollout"
        return orig_dual(self, *a, **k)
    monkeypatch.setattr(DualSession, "step", dual_step)
    return state


def test_loops_step_only_inside_rollout(rollout_guard, monkeypatch, tmp_path):
    from rrp.harness.eval.dual_teacher_quality import run_audit_episode
    from rrp.harness.eval.hooks import warm_up
    from rrp.harness.eval.latent_causal import episode_protocol, window_protocol
    from rrp.harness.eval.latent_eval import disturbance_test
    from rrp.harness.eval.teacher_quality import run_policy_quality_episode, run_quality_episode
    from rrp.envs.mujoco.fixtures import make_pick_place_session
    quiet = dict(log=lambda *a, **k: None)
    run_quality_episode(ROBOT, 3, "v1", max_steps=4)
    run_policy_quality_episode(_tiny_bc(), ROBOT, 3, label="tiny", max_steps=4)
    run_audit_episode("support_insert", DUAL, 3, max_steps=4)
    si, R, P = _tiny_latent()
    window_protocol(si, R, P, ROBOT, [3], decision_ticks=(2,), window=2, replan=2, conditions=("control_replay", "zero"),
                    edit_steps=1, **quiet)
    episode_protocol(si, R, P, ROBOT, [3], conditions=("control", "zero"), replan=2, max_steps=3, edit_steps=1, **quiet)
    si2, R2, _ = _tiny_latent()
    disturbance_test(si2, R2, ROBOT, [3], warmup_ticks=3, hold_ticks=2)
    warm_up(make_pick_place_session(seed=5, n_distractors=0), 2)
    seen, cams = _video_env(monkeypatch, tmp_path)
    import importlib
    va = importlib.import_module("rrp.harness.eval.video_arm")
    _cap(monkeypatch, va)
    va.run(_arm_args(tmp_path, seeds="3", max_steps=3))
    vd = importlib.import_module("rrp.harness.eval.video_dual")
    _cap(monkeypatch, vd)
    vd.run(_dual_args(tmp_path, max_steps=3))
    assert rollout_guard["calls"] >= 12


def test_semantic_oracle_lookahead_is_a_rollout(rollout_guard):
    """latent_semantic_edits' privileged teacher demo (a discarded-snapshot look-ahead) runs as a rollout."""
    from rrp.envs.mujoco.fixtures import make_pick_place_session
    from rrp.harness.eval.latent_semantic_edits import _teacher_demo
    from rrp.policies.teachers.arm import PickPlaceTeacher
    s = make_pick_place_session(seed=5, n_distractors=0)
    teacher = PickPlaceTeacher(s)
    snap, tst = s.snapshot(), teacher.state()
    cmds = _teacher_demo(s, teacher, 3)
    s.restore(snap)
    teacher.load(tst)
    assert len(cmds) == 3 and rollout_guard["calls"] == 1


def test_end_when_and_recorder_hooks_are_registered():
    from rrp.harness.eval import hooks as H
    from rrp.envs.mujoco.fixtures import make_pick_place_session
    from rrp.harness.rollout import rollout
    seen = []
    s = make_pick_place_session(seed=5, n_distractors=0)
    hs = [H.HOOKS["recorder"](on_step=lambda i, e, a, st: seen.append(round(float(e.data.time), 5))),
          H.HOOKS["end_when"](lambda i, e: len(seen) >= 3, outcome="failure", reason="three")]
    ep = rollout(lambda sd: s, H.HoldPolicy(), H.budget_task("pick_place", s.spec.env_id), [5], batch=1,
                 max_steps=50, hooks=hs)[0]
    assert ep.steps == 3 and ep.failure_reason == "three" and len(seen) == 3
