"""U1 (D13, architecture 14.4): `control="wholebody"` = the `legs` group + an `upper` joint_position group (arms, waist, head) at 50 Hz in
MuJoCo and Warp; the tracker actor takes the upper-body joint state; `*_ub` recipes.

Stub actors are random tiny MLPs (`_saved_actor`), nothing trains. Scene tests use the Menagerie humanoid t1 (assets; skipped without them). The Warp env test needs CUDA + mujoco_warp (peer: RRP_TEST_GPU=1) and skips here."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_golden import _saved_actor      # noqa: E402

import rrp.envs.mujoco.legged_core as LC     # noqa: E402
import rrp.envs.mujoco.legged_tracker as LT  # noqa: E402
from rrp.core.action import NativeCommand    # noqa: E402

BODY = "t1"        # a Menagerie humanoid: the procedural phum bodies with arms do not compile a scene (see the lead note)
# held (`upper`) actuator counts of the humanoid family, from the compiled bodies (arms + waist + head, hands where actuated)
UPPER_WIDTH = {"t1": 11, "g1": 17, "h1": 9, "op3": 8, "apollo": 20, "adam_lite": 13, "n1": 11, "talos": 20}


def _binding(sc):
    mr = sc.robots[0]
    return LC.LeggedBinding(sc.model, mr.meta, mr.prefix)


def _put(root: Path, body: str, b, *, upper: bool, seed: int, extra: dict | None = None) -> Path:
    d = root / body / "stub"
    d.mkdir(parents=True, exist_ok=True)
    n_in = b.obs_dim + (b.upper_dim if upper else 0)
    meta = dict(body=body, actuator_limits=b.meta.get("actuator_limits"), contact_model="contact_v2", clock_gate=True)
    meta.update(dict(upper_obs=True) if upper else {})
    meta.update(extra or {})
    _saved_actor(d, "actor", n_in, b.n, meta, seed)
    (d / "meta.json").write_text(json.dumps(dict(meta, obs_dim=n_in)))
    return d


@pytest.fixture
def registry(tmp_path, monkeypatch):
    def install():
        monkeypatch.setattr(LT, "TRACKER_DIR", tmp_path)
        monkeypatch.setattr(LT, "TRACKERS", LT.scan_trackers(tmp_path))
    return install


def _scene(body=BODY):
    from rrp.envs.mujoco.legged import build_waypoint_contact
    return build_waypoint_contact(body, 0, contact="v2")


def _session(tmp_path, registry, *, control="wholebody", upper_obs=False, **kw):
    from rrp.envs.mujoco.legged import LeggedSession
    sc = _scene()
    _put(tmp_path, BODY, _binding(sc), upper=upper_obs, seed=5)
    registry()
    return LeggedSession(sc, tracker_kind="learned", seed=0, tracker=f"{BODY}:stub", control=control, **kw)


def _cmd(s, **groups):
    return NativeCommand(controller_version=s.controller_version(), groups={k: list(map(float, v)) for k, v in groups.items()},
                         source="scripted_teacher")


def _upper_q(s):
    return s.data.qpos[s.binding.held_qadr].copy()


# ---------------------------------------------------------------- group widths per humanoid body
@pytest.mark.menagerie
@pytest.mark.parametrize("key", sorted(UPPER_WIDTH))
def test_group_widths_per_humanoid_body(key):
    """The body's controller declares `legs` (policy joints) and `upper` (held joints); the binding's upper block and the payload bodies
    (the two lateral hands / wrists) follow from it."""
    from rrp.bodies.legged import legged_body
    meta = legged_body(key).meta
    groups = {g["name"]: len(g["actuators"]) for g in meta["controller"]["groups"]}
    L = meta["legged"]
    assert groups == {"legs": len(L["policy_actuators"]), "upper": len(L["held_actuators"])}
    assert groups["upper"] == UPPER_WIDTH[key]
    b = _binding(_scene(key))
    assert len(b.held_act) == UPPER_WIDTH[key] and b.upper_dim == 2 * UPPER_WIDTH[key]
    assert len(b.held_lo) == len(b.held_hi) == len(b.held_qadr) == len(b.held_dadr) == UPPER_WIDTH[key]
    pb = b.payload_bodies()
    assert len(pb) == 2
    import mujoco
    d = mujoco.MjData(b.model)
    b.set_default(d, z=1.0)
    mujoco.mj_kinematics(b.model, d)
    lat = [float(d.xpos[i][1] - d.xpos[b.root_bid][1]) for i in pb]
    assert min(lat) < -0.08 and max(lat) > 0.08


@pytest.mark.menagerie
def test_wholebody_action_spaces(tmp_path, registry):
    s = _session(tmp_path, registry)
    b = s.binding
    sp = {a.group: a for a in s._action_spaces()}
    assert set(sp) == {"legs", "upper"} and sp["legs"].width == b.n and sp["upper"].width == len(b.held_act) > 0
    assert sp["upper"].kind == "joint_position" and sp["upper"].rate_hz == 50.0 and sp["legs"].rate_hz == 50.0
    assert np.allclose(sp["upper"].low, b.held_lo) and np.allclose(sp["upper"].high, b.held_hi)
    assert "wholebody" in s.controller_version()
    legs = _session(tmp_path, registry, control="legs")
    assert [a.group for a in legs._action_spaces()] == ["legs"]


@pytest.mark.menagerie
def test_wholebody_needs_an_upper_group_and_no_scan(tmp_path, registry):
    from rrp.envs.mujoco.legged import LeggedSession, build_waypoint_contact
    with pytest.raises(ValueError, match="terrain_scan"):
        _session(tmp_path, registry, terrain_scan=True)
    with pytest.raises(ValueError, match="control"):
        _session(tmp_path, registry, control="nope")
    with pytest.raises(ValueError, match="upper"):          # a body without held actuators has no upper group
        LeggedSession(build_waypoint_contact("phum_1", 0), tracker_kind="cpg", seed=0, control="wholebody")


def test_procedural_humanoid_group_widths():
    from rrp.bodies.legged import legged_body
    m = legged_body("phum_0").meta
    assert {g["name"]: len(g["actuators"]) for g in m["controller"]["groups"]}["upper"] == len(m["legged"]["held_actuators"]) == 9


# ---------------------------------------------------------------- held upper pose tracks within tolerance for 1 s (CPU fixture)
@pytest.mark.menagerie
def test_upper_pose_tracks_for_one_second(tmp_path, registry):
    s = _session(tmp_path, registry)
    s.reset(0)
    b = s.binding
    off = np.zeros(len(b.held_act))
    off[[k for k, a in enumerate(b.held_act) if "shoulder_pitch" in s.model.actuator(int(a)).name.lower()]] = -0.5   # raise both arms
    tgt = np.clip(b.q0_held + off, b.held_lo, b.held_hi)
    assert np.abs(tgt - b.q0_held).max() > 0.3
    q_start = _upper_q(s)
    for _ in range(50):                                       # 1 s = 50 ticks of 20 ms
        r = s.step(_cmd(s, legs=b.q0, upper=tgt))
        assert r.rejected is None
    err = np.abs(_upper_q(s) - tgt)
    assert err.max() < 0.05, err
    assert np.abs(_upper_q(s) - q_start).max() > 0.3          # it moved: not the old hold at q0_held
    # (a stub-driven t1 on default-stance leg targets topples in 1 s in legs mode too; the check is the upper joints)
    # PD hold: the executed record carries both groups
    assert set(r.command) == {"legs", "upper"}


@pytest.mark.menagerie
def test_upper_target_holds_when_omitted_and_resets(tmp_path, registry):
    s = _session(tmp_path, registry)
    s.reset(0)
    b = s.binding
    tgt = np.clip(b.q0_held + 0.2, b.held_lo, b.held_hi)
    s.step(_cmd(s, upper=tgt))
    assert np.allclose(s.upper_target, tgt)
    s.step(_cmd(s, legs=b.q0))                                # no upper this tick: the last upper target holds
    assert np.allclose(s.upper_target, tgt) and np.allclose(s.data.ctrl[b.held_act], tgt)
    snap = s.snapshot()
    s.step(_cmd(s, upper=b.q0_held))
    s.restore(snap)
    assert np.allclose(s.upper_target, tgt)
    s.reset(0)
    assert np.allclose(s.upper_target, b.q0_held)


@pytest.mark.menagerie
def test_wholebody_validation(tmp_path, registry):
    s = _session(tmp_path, registry)
    s.reset(0)
    b = s.binding
    nh = len(b.held_act)
    for groups, code in (({"arm": [0.0]}, "unknown_group"), ({"upper": [0.0] * (nh + 1)}, "wrong_width"),
                         ({"legs": [0.0] * (b.n - 1)}, "wrong_width"), ({"upper": [float("nan")] * nh}, "nonfinite"), ({}, "unknown_group")):
        r = s.step(NativeCommand(controller_version=s.controller_version(), groups=groups, source="scripted_teacher"))
        assert r.rejected == code, (groups, r.rejected)
    stale = NativeCommand(controller_version="x", groups={"upper": [0.0] * nh}, source="scripted_teacher")
    assert s.step(stale).rejected == "stale_action_chunk"
    s.step(_cmd(s, upper=b.held_hi + 5.0))                    # clipped to the actuator range
    assert np.allclose(s.upper_target, b.held_hi)


@pytest.mark.menagerie
def test_perturb_tick_is_refused_with_wholebody(tmp_path, registry):
    """perturb.install_legged replaces the tick and holds `upper` at q0_held: fail loudly instead of silently ignoring the target."""
    from rrp.envs.mujoco.perturb import PhysicsPerturbation, install_legged
    s = _session(tmp_path, registry)
    s.reset(0)
    install_legged(s, PhysicsPerturbation(), 0)
    with pytest.raises(RuntimeError, match="wholebody"):
        s.step(_cmd(s, legs=s.binding.q0))


# ---------------------------------------------------------------- `control="legs"` unchanged
@pytest.mark.menagerie
def test_legs_mode_unchanged_and_wholebody_legs_only_is_bit_identical(tmp_path, registry):
    a = _session(tmp_path, registry, control="legs")
    w = _session(tmp_path, registry, control="wholebody")
    a.reset(0)
    w.reset(0)
    b = a.binding
    rng = np.random.default_rng(3)
    for k in range(30):
        tg = b.q0 + 0.1 * rng.standard_normal(b.n)
        ra = a.step(_cmd(a, legs=tg))
        rw = w.step(_cmd(w, legs=tg))
        assert np.array_equal(ra.qpos, rw.qpos), k
    assert np.array_equal(a.data.ctrl[b.held_act], b.q0_held)                       # legs mode still holds the default upper pose
    assert np.array_equal(_upper_q(a), _upper_q(w))
    r = a.step(NativeCommand(controller_version=a.controller_version(), groups={"upper": [0.0] * len(b.held_act)}, source="scripted_teacher"))
    assert r.rejected == "unknown_group"                                               # legs mode has no upper group
    assert not hasattr(a.tracker, "upper")


# ---------------------------------------------------------------- the tracker actor takes the upper-body joint state
@pytest.mark.menagerie
def test_upper_obs_block_and_learned_tracker_dims(tmp_path, registry):
    sc = _scene()
    b = _binding(sc)
    assert b.upper_dim == 2 * len(b.held_act)
    import mujoco
    d = mujoco.MjData(b.model)
    b.set_default(d, z=1.0)
    assert np.allclose(b.upper_obs(d), 0)
    d.qpos[b.held_qadr[0]] += 0.25
    d.qvel[b.held_dadr[1]] = 2.0
    u = b.upper_obs(d)
    nh = len(b.held_act)
    assert u.shape == (2 * nh,) and u.dtype == np.float32
    assert u[0] == pytest.approx(0.25) and u[nh + 1] == pytest.approx(0.1)             # (q - q0, qdot * 0.05)
    meta = dict(sc.robots[0].meta, contact_model="contact_v2")
    _put(tmp_path, BODY, b, upper=True, seed=5)
    registry()
    tr = LT.load_tracker(BODY, b, meta, tracker=f"{BODY}:stub")
    assert tr.upper_obs
    d0 = mujoco.MjData(b.model)
    b.set_default(d0, z=1.0)
    a0 = tr.act(d0, np.array([0.3, 0.0, 0.0]))
    tr.reset()
    assert np.array_equal(tr.act(d0, np.array([0.3, 0.0, 0.0])), a0)                   # deterministic
    tr.reset()
    a1 = tr.act(d, np.array([0.3, 0.0, 0.0]))
    assert not np.allclose(a0, a1)                                                       # the actor reads the upper state
    # dims: an actor that declares upper_obs but was built without the block, and the reverse, are refused
    _put(tmp_path, BODY, b, upper=False, seed=6, extra=dict(upper_obs=True))
    registry()
    with pytest.raises(LT.TrackerMismatch, match="dims"):
        LT.load_tracker(BODY, b, meta, tracker=f"{BODY}:stub")
    _put(tmp_path, BODY, b, upper=False, seed=7)
    registry()
    assert not LT.load_tracker(BODY, b, meta, tracker=f"{BODY}:stub").upper_obs
    (tmp_path / BODY / "stub" / "actor.pt").unlink()
    _saved_actor(tmp_path / BODY / "stub", "actor", b.obs_dim + b.upper_dim, b.n, dict(body=BODY, contact_model="contact_v2"), 8)
    with pytest.raises(LT.TrackerMismatch, match="dims"):
        LT.load_tracker(BODY, b, meta, tracker=f"{BODY}:stub")


@pytest.mark.menagerie
def test_upper_obs_with_the_scan_appends_after_it(tmp_path, registry):
    """Layout: proprio | terrain scan | upper block (so a legs-only actor warm-starts by zero-padding the tail)."""
    sc = _scene()
    b = _binding(sc)
    n = b.obs_dim + LC.SCAN_DIM + b.upper_dim
    d = tmp_path / BODY / "stub"
    d.mkdir(parents=True)
    meta = dict(body=BODY, actuator_limits=b.meta.get("actuator_limits"), contact_model="contact_v2", upper_obs=True, extra_obs_dim=LC.SCAN_DIM, terrain_scan=LC.terrain_scan_spec())
    _saved_actor(d, "actor", n, b.n, meta, 4)
    (d / "meta.json").write_text(json.dumps(dict(meta, obs_dim=n)))
    registry()
    tr = LT.load_tracker(BODY, b, dict(sc.robots[0].meta, contact_model="contact_v2"), tracker=f"{BODY}:stub")
    seen = {}
    tr.net = type("N", (), {"__call__": lambda self, x: seen.setdefault("x", x) is None or tr.torch.zeros(1, b.n)})()
    import mujoco
    dd = mujoco.MjData(b.model)
    b.set_default(dd, z=1.0)
    dd.qpos[b.held_qadr[0]] += 0.5
    tr.extra_fn = lambda data: np.full(LC.SCAN_DIM, 0.3, np.float32)
    tr.mean, tr.std = np.zeros(n, np.float32), np.ones(n, np.float32)
    tr.act(dd, np.zeros(3))
    x = seen["x"][0].numpy()
    assert x.shape == (n,)
    assert np.allclose(x[b.obs_dim:b.obs_dim + LC.SCAN_DIM], 0.3) and x[b.obs_dim + LC.SCAN_DIM] == pytest.approx(0.5)


@pytest.mark.menagerie
def test_upper_obs_is_refused_on_the_shared_morph_format(tmp_path):
    from rrp.envs.mujoco.morph_obs import NS, OBS_DIM
    sc = _scene()
    b = _binding(sc)
    p = _saved_actor(tmp_path, "m", OBS_DIM, NS, dict(obs_format="morph_v1", train_bodies=["phum_1"], body="shared", upper_obs=True), 4)
    with pytest.raises(LT.TrackerMismatch, match="upper"):
        LT.LearnedTracker(p, b, BODY)


# ---------------------------------------------------------------- *_ub recipes and trainer flags
UB_BODIES = ("t1", "g1", "h1", "op3", "apollo", "adam_lite")


def test_ub_recipes_and_trainer_flags():
    from rrp.harness.train.tracker_recipes import WARP_RECIPES, recipe_record
    from rrp.harness.train.warp_tracker_ppo import build_args
    ub_keys = {"upper_body", "upper_amp", "upper_speed", "payload_frac"}
    for body in UB_BODIES:
        name = f"{body}_clock_gpu_ub"
        opts, rec = recipe_record(name)
        assert name in WARP_RECIPES and len(rec["sha256"]) == 64
        assert opts["body"] == body and opts["upper_body"] is True and opts["clock_gate"] is True
        assert 0 < opts["upper_amp"] <= 1 and opts["upper_speed"] > 0 and 0 < opts["payload_frac"] <= 0.2
        a = build_args(["--recipe", name, "--out", "x"])
        assert a.upper_body and a.upper_amp == opts["upper_amp"] and a.payload_frac == opts["payload_frac"]
        assert not ub_keys & {k for r, o in WARP_RECIPES.items() if not r.endswith("_ub") and o for k in o}
    a = build_args(["--body", "t1", "--out", "x"])
    assert not a.upper_body                                   # off unless asked: every existing recipe is unchanged
    # warm starts: the legs-only actor gets zero-padded upper columns (the block is the obs tail), from scratch elsewhere
    assert recipe_record("t1_clock_gpu_ub")[0]["init_shared"].endswith("t1_v2ft4/actor.pt")
    assert "init_shared" not in recipe_record("op3_clock_gpu_ub")[0]


# ---------------------------------------------------------------- Warp env (CUDA + mujoco_warp; the peer)
def test_warp_env_upper_body_targets_payload_and_obs_dims():
    pytest.importorskip("mujoco_warp")
    import torch
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from rrp.envs.warp.tracker_env import WarpTrackerEnv
    env = WarpTrackerEnv(BODY, 16, seed=3, upper_body=True, upper_amp=0.5, upper_speed=2.0, payload_frac=0.1)
    b = env.b
    assert env.obs_dim == b.obs_dim + b.upper_dim and env.priv_dim == b.priv_dim + 1 + 1 + len(b.held_act)
    o = env.reset_all()
    assert o.shape == (16, env.obs_dim)
    pl = env.payload.clone()
    assert (pl >= 0).all() and (pl <= 0.1 * env.mass + 1e-6).all() and pl.max() > 0
    added = env.m_mass[:, env.pl_bodies].sum(-1) - env.pl_mass0.sum(-1)      # the payload is in the simulated model
    assert torch.allclose(added, pl, atol=1e-4)
    z = torch.zeros(16, env.nA, device=env.dev)
    for _ in range(60):
        o, priv, r, done, _ = env.step(z)
    tgt = env.ctrl[:, env.held_act]
    assert (tgt - env.q0_held).abs().max() > 0.05                      # the upper targets moved
    assert (tgt >= env.held_lo - 1e-5).all() and (tgt <= env.held_hi + 1e-5).all()
    assert torch.isfinite(o).all() and torch.isfinite(priv).all()
    ub = o[:, -b.upper_dim:]                                            # the block is the tail of the actor obs
    nh = len(b.held_act)
    assert torch.allclose(ub[:, :nh], env.qpos[:, env.held_qadr] - env.q0_held, atol=0.2)
    with pytest.raises(ValueError, match="upper"):
        from rrp.envs.warp.tracker_env import MorphMultiEnv
        MorphMultiEnv([([BODY], 8)], upper_body=True)
