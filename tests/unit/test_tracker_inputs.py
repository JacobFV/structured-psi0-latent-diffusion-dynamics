"""Round-3 (D-146): what a learned tracker declares as inputs decides what the session attaches, in EVERY control mode; `morph_v2`
(legs + upper) loads in LearnedTracker with numpy == Warp observation parity; mujoco/legged answers `negotiate` from a static
`env_spec` (no session, no actor), so a body without arm roles is an n/a matrix cell and never an exception at build.

Stub actors are random tiny MLPs (`_saved_actor`); nothing trains. Scene tests use the Menagerie humanoid t1 (marker `menagerie`)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_golden import _saved_actor      # noqa: E402
from test_wholebody import BODY, _binding, _scene, registry   # noqa: E402,F401  (registry is a fixture)

import rrp.envs.mujoco.legged_core as LC     # noqa: E402
import rrp.envs.mujoco.legged_tracker as LT  # noqa: E402
from rrp.core.action import NativeCommand    # noqa: E402


def _put_actor(root: Path, b, *, scan: bool = False, ring: bool = False, upper: bool = False, seed: int = 5) -> Path:
    """A per-body registry entry `t1:stub` whose declared inputs are [proprio | scan (| ring) | upper] (the order LearnedTracker feeds)."""
    d = root / BODY / "stub"
    d.mkdir(parents=True, exist_ok=True)
    meta = dict(body=BODY, actuator_limits=b.meta.get("actuator_limits"), contact_model="contact_v2", clock_gate=True)
    n = b.obs_dim
    if ring:
        meta.update(extra_obs_dim=LC.EXTRA_DIM_RING, terrain_scan=LC.terrain_scan_spec(), range_ring=LC.range_ring_spec())
        n += LC.EXTRA_DIM_RING
    elif scan:
        meta.update(extra_obs_dim=LC.SCAN_DIM, terrain_scan=LC.terrain_scan_spec())
        n += LC.SCAN_DIM
    if upper:
        meta.update(upper_obs=True)
        n += b.upper_dim
    _saved_actor(d, "actor", n, b.n, meta, seed)
    (d / "meta.json").write_text(json.dumps(dict(meta, obs_dim=n)))
    return d


def _carry(tmp_path, registry, task, **actor):
    from rrp.envs.mujoco.humanoid_scenes import make_humanoid_session
    _put_actor(tmp_path, _binding(_scene()), **actor)
    registry()
    return make_humanoid_session(task=task, body=BODY, seed=0, tracker=f"{BODY}:stub", tracker_kind="learned")


# ---------------------------------------------------------------- (2) sensors follow the tracker's declared inputs, every control mode
@pytest.mark.menagerie
def test_scan_and_upper_actor_gets_the_public_scan_under_wholebody(tmp_path, registry):
    """A steps_ub-style actor (scan + upper_obs) on t1 h_steps_carry: the session publishes the scan, the actor acts (it raised
    `operands could not be broadcast together with shapes (69,) (146,)` with no sensor)."""
    s = _carry(tmp_path, registry, "h_steps_carry", scan=True, upper=True)
    assert s.control == "wholebody" and s.terrain is not None and s.ring is None
    assert "terrain_scan" in s.spec.capabilities and "range_ring" not in s.spec.capabilities
    o = s.reset(0)
    assert any(c.name == "0:terrain_scan" for c in o.declared_sensor_channels)
    bt = s.body_tracker
    assert bt.extra_kind == "terrain_scan" and bt.upper_obs
    tgt = bt.act(s.data, np.zeros(3))
    assert tgt.shape == (s.binding.n,) and np.isfinite(tgt).all()
    assert np.array_equal(bt.extra_fn(s.data), s.terrain.values)                    # the actor's extra block IS the published scan
    cmd = NativeCommand(controller_version=s.controller_version(), groups={"legs": list(map(float, tgt)),
                                                                          "upper": list(map(float, s.binding.q0_held))},
                        source="scripted_teacher")
    s.step(cmd)


@pytest.mark.menagerie
def test_gap_ring_actor_gets_scan_and_ring_under_wholebody_on_h_gap_cart(tmp_path, registry):
    s = _carry(tmp_path, registry, "h_gap_cart", ring=True)
    assert s.terrain is not None and s.ring is not None
    assert {"terrain_scan", "range_ring"} <= set(s.spec.capabilities)
    s.reset(0)
    bt = s.body_tracker
    assert bt.extra_kind == "terrain_scan+range_ring"
    tgt = bt.act(s.data, np.zeros(3))
    assert tgt.shape == (s.binding.n,)
    x = bt.extra_fn(s.data)
    assert x.shape == (LC.EXTRA_DIM_RING,)
    assert np.array_equal(x[:LC.SCAN_DIM], s.terrain.values) and np.array_equal(x[LC.SCAN_DIM:], s.ring.values)   # scan first
    # D-147: both sensors are public, so the rl_expert over a scan + ring actor is `learned` and needs both capabilities
    from rrp.policies.teachers.humanoid import make_rl_expert
    pol = make_rl_expert(arg=f"{BODY}:stub")
    assert pol.label.startswith("learned:rl_expert:") and pol.info.source == "learned"
    assert {"terrain_scan", "range_ring"} <= set(pol.info.requires.env_capabilities)


@pytest.mark.menagerie
@pytest.mark.parametrize("off", ["terrain_scan", "range_ring"])
def test_disabling_a_sensor_the_actor_takes_is_refused_in_every_mode(tmp_path, registry, off):
    from rrp.envs.mujoco.humanoid_scenes import make_humanoid_session
    _put_actor(tmp_path, _binding(_scene()), ring=True)
    registry()
    with pytest.raises(LT.TrackerMismatch, match=off):
        make_humanoid_session(task="h_gap_cart", body=BODY, seed=0, tracker=f"{BODY}:stub", tracker_kind="learned", **{off: False})


@pytest.mark.menagerie
def test_a_plain_actor_under_wholebody_attaches_no_sensor_and_an_explicit_request_still_refuses(tmp_path, registry):
    s = _carry(tmp_path, registry, "h_carry", upper=True)
    assert s.terrain is None and s.ring is None and s.body_tracker.extra_kind == "none"
    assert "terrain_scan" not in s.spec.capabilities
    from rrp.envs.mujoco.humanoid_scenes import make_humanoid_session
    with pytest.raises(ValueError, match="terrain_scan=True"):
        make_humanoid_session(task="h_carry", body=BODY, seed=0, tracker=f"{BODY}:stub", tracker_kind="learned", terrain_scan=True)


# ---------------------------------------------------------------- (4) morph_v2
def _morph_v2_actor(root: Path, *, extra: int, seed: int = 4, train=("t1",)) -> Path:
    from rrp.envs.mujoco.morph_obs import NS, OBS_DIM_V2
    meta = dict(obs_format="morph_v2", upper_obs=True, train_bodies=list(train), shared=True, body="shared", clock_gate=False)
    if extra:
        meta.update(extra_obs_dim=extra)
    if extra == LC.SCAN_DIM:
        meta.update(terrain_scan=LC.terrain_scan_spec())
    return _saved_actor(root, "m2", OBS_DIM_V2 + extra, NS, meta, seed)


@pytest.mark.menagerie
def test_learned_tracker_loads_morph_v2_and_refuses_inconsistent_meta(tmp_path):
    from rrp.envs.mujoco.morph_obs import NS, OBS_DIM, OBS_DIM_V2, UPPER_DIM
    sc = _scene()
    b = _binding(sc)
    assert OBS_DIM_V2 == OBS_DIM + UPPER_DIM
    tr = LT.LearnedTracker(_morph_v2_actor(tmp_path, extra=0), b, BODY)
    assert tr.upper_obs and tr.morph is not None and tr.version.startswith("learned_tracker:shared_morph_v2:")
    assert ":transfer" not in tr.version and tr.extra_kind == "none"
    tgt = tr.act(_default_data(b), np.array([0.3, 0.0, 0.0]))
    assert tgt.shape == (b.n,) and np.isfinite(tgt).all()
    # upper_obs is part of the format: a morph_v2 actor without it, or morph_v1 with it, is a dimension contract violation
    bad = tmp_path / "bad"
    bad.mkdir()
    p = _saved_actor(bad, "m", OBS_DIM_V2, NS, dict(obs_format="morph_v2", train_bodies=["t1"], body="shared"), 4)
    with pytest.raises(LT.TrackerMismatch, match="morph_v2"):
        LT.LearnedTracker(p, b, BODY)
    p = _saved_actor(bad, "m1", OBS_DIM, NS, dict(obs_format="morph_v1", train_bodies=["t1"], body="shared", upper_obs=True), 4)
    with pytest.raises(LT.TrackerMismatch, match="morph_v1"):
        LT.LearnedTracker(p, b, BODY)
    p = _saved_actor(bad, "m3", OBS_DIM_V2 + 3, NS, dict(obs_format="morph_v2", upper_obs=True, train_bodies=["t1"], body="shared"), 4)
    with pytest.raises(LT.TrackerMismatch, match="morph_v2"):
        LT.LearnedTracker(p, b, BODY)


def _default_data(b):
    import mujoco
    d = mujoco.MjData(b.model)
    b.set_default(d, z=1.0)
    mujoco.mj_forward(b.model, d)
    return d


@pytest.mark.menagerie
@pytest.mark.parametrize("extra", [3, LC.SCAN_DIM])      # 3: a privileged extra block (caller-supplied); 77: the public scan
def test_morph_v2_actor_input_equals_the_warp_group_obs(tmp_path, extra, monkeypatch):
    """LearnedTracker's assembled input [morph_v1 | extra | upper block] on an MjData == MorphMultiEnv._group_obs (the trainer's obs)
    for the same state (fake CPU engine; the state is random, the root upright so the IMU gravity agrees)."""
    import mujoco
    import torch
    from test_morph_multi import FakeUpper
    from rrp.envs.warp.tracker_env import MorphMultiEnv
    env = MorphMultiEnv([("t1", 2)], seed=3, obs_noise=0.0, env_cls=FakeUpper, upper_body=True)
    e, g = env.envs[0], env.specs[0]
    blk = torch.linspace(-1.0, 1.0, extra)                # the extra block: a fixed vector on both sides
    e.extra_dim = extra
    e.extra_obs = lambda: blk.expand(e.N, extra).clone()
    e.phase[:] = 0.37
    want = env._group_obs(e, g)[1].numpy()
    d = mujoco.MjData(e.m)
    d.qpos[:] = e.qpos[1].numpy()
    d.qvel[:] = e.qvel[1].numpy()
    mujoco.mj_forward(e.m, d)
    tr = LT.LearnedTracker(_morph_v2_actor(tmp_path, extra=extra), e.b, BODY)
    tr.extra_fn = lambda data: blk.numpy()
    tr.reset(phase=0.37)
    seen = {}
    real = tr.net.forward

    def spy(x):
        seen["x"] = x.numpy()[0].copy()
        return real(x)
    monkeypatch.setattr(tr.net, "forward", spy)
    tr.act(d, e.cmd[1].numpy())                                        # both sides scale the raw command by CMD_SCALE
    got = seen["x"] * tr.std + tr.mean                                  # undo the actor's input normalisation
    clip = np.abs((want - tr.mean) / tr.std) < 5                        # the net clips at +-5 sigma: compare where it did not bite
    assert got.shape == want.shape == (tr.meta["obs_dim"],) and clip.all()
    assert np.allclose(got, want, atol=1e-4), np.flatnonzero(~np.isclose(got, want, atol=1e-4))


# ---------------------------------------------------------------- (7) a static env_spec: negotiate answers before any build
@pytest.fixture
def no_build(monkeypatch):
    """Any session construction fails the test: the static spec must not build one."""
    import rrp.envs.mujoco.legged as L

    def boom(*a, **k):
        raise AssertionError("a session was built for the static env_spec")
    monkeypatch.setattr(L.LeggedSession, "__init__", boom)


@pytest.mark.menagerie
def test_berkeley_h_carry_is_na_through_negotiate_without_building(no_build):
    from rrp.harness.eval.evaluate import matrix
    (row,) = matrix(["teacher:h_carry"], [("mujoco/legged", "berkeley")], ["h_carry"])
    assert row["status"] == "n/a", row
    assert any("body has no arm roles" in r for r in row["reasons"]), row
    assert not any("env unavailable" in r for r in row["reasons"]), row


@pytest.mark.menagerie
@pytest.mark.parametrize("task,caps_in,caps_out", [("h_carry", {"arm_roles"}, set()), ("h_steps", set(), {"arm_roles"}),
                                                   ("h_gap", set(), {"arm_roles", "terrain_scan", "range_ring"})])
def test_static_spec_answers_for_the_arm_body_and_the_terrain_tasks(no_build, task, caps_in, caps_out):
    from rrp.harness.eval.evaluate import env_spec
    sp = env_spec("mujoco/legged", task=task, body=BODY)
    assert caps_in <= set(sp.capabilities) and not (caps_out & set(sp.capabilities))
    assert sp.env_id == "mujoco/legged" and sp.bodies[0].key == BODY
    groups = {a.group for a in sp.action_spaces}
    assert groups == ({"legs", "upper"} if task == "h_carry" else {"base_velocity"})


@pytest.mark.menagerie
@pytest.mark.parametrize("task", ["h_carry", "h_steps_carry", "h_gap_cart", "h_walk", "h_steps", "h_gap"])
def test_static_spec_equals_the_built_sessions_spec(tmp_path, registry, task):
    """Same hash, task name, control rate, action spaces and capabilities as the session the env factory builds (default actor: a stub
    with no extra inputs, the shape of every default-resolution actor)."""
    from rrp.envs.mujoco.humanoid_scenes import make_humanoid_session
    from rrp.harness.eval.evaluate import env_spec
    _put_actor(tmp_path, _binding(_scene()), upper=task not in ("h_steps", "h_gap"))
    registry()
    s = make_humanoid_session(task=task, body=BODY, seed=0, tracker=f"{BODY}:stub", tracker_kind="learned")
    sp = env_spec("mujoco/legged", task=task, body=BODY)
    keep = lambda e: {k: v for k, v in e.model_dump().items() if k != "provenance"}
    assert keep(sp) == keep(s.spec)
    assert sp.provenance["physics"] == s.spec.provenance["physics"]


@pytest.mark.menagerie
def test_a_legs_only_body_gets_a_legs_only_spec_not_an_exception(no_build):
    """wholebody on a body with no held actuators: the session refuses to build, the static spec offers the legs group only."""
    from rrp.harness.eval.evaluate import env_spec
    sp = env_spec("mujoco/legged", task="h_carry", body="berkeley")
    assert {a.group for a in sp.action_spaces} == {"legs"} and not sp.has("arm_roles")


def test_default_resolution_actors_take_no_public_sensor_so_the_static_spec_needs_none():
    """`static_env_spec` assumes the tracker the env resolves by default (`trackers/<body>/[contact_vN/]`) declares no scan / ring input;
    a default slot that did would make the session offer `terrain_scan` where the static spec does not (scan actors are picked by id)."""
    from rrp.bodies.contact import resolve
    slots = {LT.V1, *(f"contact_{resolve(c)}" for c in ("v2",))}
    default = {k: e for k, e in LT.scan_trackers().items() if k[1] in slots}
    assert default, "no committed tracker metas"
    assert {k: e.extra_obs for k, e in default.items() if e.extra_obs != "none"} == {}


@pytest.mark.menagerie
@pytest.mark.parametrize("ring", [False, True])
def test_validation_bench_feeds_the_public_sensors_of_a_scan_actor(tmp_path, ring):
    """D-147 (T1 incident): the D-112 validation bench (bare model, no session) must give a scan / ring input actor its public sensors
    (one wiring with the session, `wire_public_sensors`); before, `validate_tracker` crashed on the obs width (47 vs 124)."""
    from rrp.bodies.legged import legged_body, standalone_model
    from rrp.harness.eval import tracker_validation as TV
    model, _, meta = standalone_model(legged_body(BODY), contact="v2")
    b = LC.LeggedBinding(model, meta)
    d = _put_actor(tmp_path, b, scan=True, ring=ring)
    tr = LT.LearnedTracker(d / "actor.pt", b, BODY)
    terrain, rr = LT.wire_public_sensors(tr, b)
    assert terrain is not None and (rr is not None) == ring
    row = TV.run_episode(model, b, tr, dict(T=0.2, cmd=[0.2, 0.0, 0.0]), seed=3)
    assert row is not None
    assert terrain.values is not None and terrain.values.shape == (LC.SCAN_DIM,)
