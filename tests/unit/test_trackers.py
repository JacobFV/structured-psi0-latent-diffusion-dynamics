"""HT (D8, D10): tracker registry by id, public terrain scan (D-146) in both backends, `rl_expert`, trainer meta.

Stub actors are random tiny MLPs (`_saved_actor`); nothing here trains or needs CUDA. Scene tests use t1 (menagerie marker); the rest needs no assets.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_golden import _saved_actor      # noqa: E402

import rrp.envs.mujoco.legged_core as LC     # noqa: E402
import rrp.envs.mujoco.legged_tracker as LT  # noqa: E402
from tests.conftest import need_assets, need_weights  # noqa: E402

BODY = "t1"
REPO = Path(__file__).resolve().parents[2]


def _binding(sc):
    mr = sc.robots[0]
    return LC.LeggedBinding(sc.model, mr.meta, mr.prefix)


def _put(root: Path, body: str, version: str, b, *, scan: bool, seed: int, extra: dict | None = None, v1: bool = False,
         contact: str = "contact_v2", ring: bool = False) -> Path:
    """A registry entry: actor.pt + meta.json under root/<body>[/<version>]."""
    d = root / body if v1 else root / body / version
    d.mkdir(parents=True, exist_ok=True)
    meta = dict(body=body, actuator_limits=b.meta.get("actuator_limits"), contact_model=contact)
    if scan:
        meta.update(extra_obs_dim=LC.SCAN_DIM, terrain_scan=LC.terrain_scan_spec())
    if ring:
        meta.update(extra_obs_dim=LC.EXTRA_DIM_RING, terrain_scan=LC.terrain_scan_spec(), range_ring=LC.range_ring_spec())
    meta.update(extra or {})
    n_extra = LC.EXTRA_DIM_RING if ring else (LC.SCAN_DIM if scan else 0)
    _saved_actor(d, "actor", b.obs_dim + n_extra, b.n, meta, seed)
    (d / "meta.json").write_text(json.dumps(dict(meta, obs_dim=b.obs_dim + n_extra)))
    return d


@pytest.fixture
def registry(tmp_path, monkeypatch):
    def install():
        monkeypatch.setattr(LT, "TRACKER_DIR", tmp_path)
        monkeypatch.setattr(LT, "TRACKERS", LT.scan_trackers(tmp_path))
    return install


# ---------------------------------------------------------------- registry by id
@pytest.mark.menagerie
def test_registry_two_versions_of_one_body_load_by_id(tmp_path, registry):
    from rrp.envs.mujoco.humanoid_scenes import build_h_steps
    sc = build_h_steps(BODY, 0, h_frac=0.2)
    b = _binding(sc)
    _put(tmp_path, BODY, "contact_v2", b, scan=False, seed=1)
    _put(tmp_path, BODY, "steps_v1", b, scan=True, seed=2)
    _put(tmp_path, BODY, "contact_v2_rejected_x", b, scan=False, seed=3)
    _put(tmp_path, BODY, "", b, scan=False, seed=4, v1=True, contact="contact_v1")
    registry()
    T = LT.TRACKERS
    assert {v for (bd, v) in T if bd == BODY} == {"contact_v2", "steps_v1", "contact_v2_rejected_x", "contact_v1"}
    assert T[(BODY, "steps_v1")].extra_obs == "terrain_scan" and T[(BODY, "contact_v2")].extra_obs == "none"
    assert T[(BODY, "contact_v2_rejected_x")].decision == "rejected" and T[(BODY, "contact_v2")].decision == "accepted"
    assert LT.get_entry(f"{BODY}:steps_v1").spec == f"{BODY}:steps_v1"
    meta = dict(sc.meta, contact_model="contact_v2")
    a = LT.load_tracker(BODY, b, meta, tracker=f"{BODY}:contact_v2")
    c = LT.load_tracker(BODY, b, meta, tracker=f"{BODY}:steps_v1")
    assert a.spec == f"{BODY}:contact_v2" and c.spec == f"{BODY}:steps_v1"
    assert a.sha256 != c.sha256 and a.extra_kind == "none" and c.extra_kind == "terrain_scan"
    with pytest.raises(KeyError, match="unknown tracker"):
        LT.get_entry(f"{BODY}:nope")
    with pytest.raises(ValueError, match="<body>:<version>"):
        LT.parse_spec("phum_3")
    # the default (no id) resolution is unchanged: the contact-model actor of the body
    assert LT.load_tracker(BODY, b, meta).sha256 == a.sha256


@pytest.mark.menagerie
def test_registry_sha_pin_and_contact_model_checked(tmp_path, registry):
    import hashlib
    from rrp.envs.mujoco.humanoid_scenes import build_h_steps
    sc = build_h_steps(BODY, 0, h_frac=0.2)
    b = _binding(sc)
    d = _put(tmp_path, BODY, "pinned", b, scan=False, seed=1)
    sha = hashlib.sha256((d / "actor.pt").read_bytes()).hexdigest()
    m = json.loads((d / "meta.json").read_text())
    (d / "meta.json").write_text(json.dumps(dict(m, sha256=sha)))
    _put(tmp_path, BODY, "badpin", b, scan=False, seed=2, extra=dict(sha256="0" * 64))
    _put(tmp_path, BODY, "v1phys", b, scan=False, seed=3, contact="contact_v1")
    registry()
    meta = dict(sc.meta, contact_model="contact_v2")
    assert LT.get_entry(f"{BODY}:pinned").sha256 == sha
    assert LT.load_tracker(BODY, b, meta, tracker=f"{BODY}:pinned").sha256 == sha
    with pytest.raises(LT.TrackerMismatch, match="registry pin"):
        LT.load_tracker(BODY, b, meta, tracker=f"{BODY}:badpin")
    with pytest.raises(LT.TrackerMismatch, match="scene uses"):
        LT.load_tracker(BODY, b, meta, tracker=f"{BODY}:v1phys")


def test_committed_registry_lists_the_installed_trackers_with_pins():
    """Reads the TRACKED meta.json files of this checkout only: not the shared store, not $RRP_HOME, no actor.pt."""
    T = LT.scan_trackers(REPO / "artifacts" / "trackers")
    assert T[("t1", "contact_v2")].sha256 == "36e9146792743115878c34e0bbf7cc46ccb5419417921358da3658c8377fc591"
    assert T[("anymal_c", "contact_v2")].sha256 == "2a16532bbd07f7abc662bdda10df6bfb70fca2ec11d59f9567142e92a2f22d95"
    assert ("t1", "contact_v1") in T and T[("anymal_c", "contact_v2_rejected_iter2499")].decision == "rejected"
    # D-147 (2026-10-03): the only committed scan-input tracker is h1:steps_ub_v1 (public terrain_scan; installed under the D-147 exception)
    assert {k for k, e in T.items() if e.extra_obs != "none"} == {("h1", "steps_ub_v1")}
    assert T[("h1", "steps_ub_v1")].extra_obs == "terrain_scan" and T[("h1", "steps_ub_v1")].decision == "accepted_d147_exception"


# ---------------------------------------------------------------- the scan: constants and sensor model
def test_scan_layout_and_sensor_model_math():
    assert (LC.SCAN_NX, LC.SCAN_NY, LC.SCAN_DIM) == (11, 7, 77) and LC.SCAN_OFFSETS.shape == (77, 2)
    xs = LC.SCAN_OFFSETS[:, 0].reshape(11, 7)
    ys = LC.SCAN_OFFSETS[:, 1].reshape(11, 7)
    assert np.allclose(xs[:, 0], np.arange(11) * 0.1 - 0.2) and np.allclose(ys[0], np.arange(7) * 0.1 - 0.3)
    spec = LC.terrain_scan_spec()
    assert spec["version"] == "terrain_scan_v1" and spec["shape"] == [11, 7] and spec["latency_ticks"] == 1
    # yaw frame: at yaw = +90 deg the +x scan axis points along world +y
    wx, wy = LC.scan_world_xy(1.0, 2.0, math.pi / 2)
    i = 10 * 7 + 3                                            # x = +0.8, y = 0
    assert wx[i] == pytest.approx(1.0, abs=1e-9) and wy[i] == pytest.approx(2.8)
    # noise N(0, 1 cm), 2 % dropout reading 0.0
    v, ok = LC.scan_sensor_model(np.full((20000, 77), 0.4), np.random.default_rng(0))
    assert (~ok).mean() == pytest.approx(0.02, abs=0.002) and np.all(v[~ok] == 0.0)
    assert v[ok].mean() == pytest.approx(0.4, abs=1e-3) and v[ok].std() == pytest.approx(0.01, rel=0.02)


@pytest.mark.menagerie
def test_warp_and_mujoco_actor_dims_and_layout_agree_without_cuda():
    """The trainer's obs_dim is b.obs_dim + SCAN_DIM (tracker_env) and the MuJoCo adapter appends the same SCAN_DIM cells in
    the same order; the Warp scan / sensor model reproduce the numpy ones on a stub env (CPU torch)."""
    import torch
    from rrp.envs.mujoco.humanoid_scenes import build_h_steps, steps_height_at
    import rrp.envs.warp.tracker_env as TE
    assert (TE.SCAN_DIM, TE.SCAN_SIGMA, TE.SCAN_DROPOUT, TE.SCAN_DROPOUT_VALUE, TE.SCAN_RANGE) == (
        LC.SCAN_DIM, LC.SCAN_SIGMA, LC.SCAN_DROPOUT, LC.SCAN_DROPOUT_VALUE, LC.SCAN_RANGE)
    assert TE.SCAN_OFFSETS is LC.SCAN_OFFSETS
    sc = build_h_steps(BODY, 0, h_frac=0.3)
    b = _binding(sc)
    L, h = float(sc.meta["L"]), float(sc.meta["staircase"]["h"])
    # MuJoCo exact scan (mj_ray) on the stair scene vs the analytic ground the Warp env uses, at a yawed pose in front of the steps
    import mujoco
    data = mujoco.MjData(sc.model)
    mujoco.mj_resetData(sc.model, data)
    x0, y0, yaw = 0.6 * L, 0.05, 0.3
    q = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    data.qpos[b.qa:b.qa + 7] = [x0, y0, b.nominal_height() + 0.02, *q]
    mujoco.mj_forward(sc.model, data)
    ts = LC.TerrainScan(b)
    exact = ts.exact(data)
    wx, wy = LC.scan_world_xy(x0, y0, yaw)
    gz = steps_height_at(wx, L, h)
    want = b.nominal_height() - np.clip(data.qpos[b.qa + 2] - gz, 0, LC.SCAN_RANGE)
    assert exact.shape == (LC.SCAN_DIM,) and np.abs(exact - want).max() < 1e-4 and np.ptp(want) > 0.05
    # the Warp env's scan on the same pose (stub: no simulator), same analytic ground
    env = object.__new__(TE.WarpTrackerEnv)
    env.dev, env.qa = torch.device("cpu"), b.qa
    env.qpos = torch.tensor(data.qpos, dtype=torch.float32)[None].repeat(2, 1)
    env.nominal_h = torch.full((2,), float(b.nominal_height()))
    env._scan_off = torch.as_tensor(LC.SCAN_OFFSETS, dtype=torch.float32)
    env.ground_z = lambda xw, yw: torch.as_tensor(steps_height_at(xw.numpy(), L, h), dtype=torch.float32)
    env.gen = torch.Generator().manual_seed(0)
    ex = env.terrain_exact()
    assert torch.allclose(ex[0], torch.as_tensor(want, dtype=torch.float32), atol=1e-4)
    big = torch.full((4000, 77), 0.4)
    pub = env._scan_model(big)
    drop = pub == LC.SCAN_DROPOUT_VALUE
    assert drop.float().mean().item() == pytest.approx(0.02, abs=0.004)
    assert pub[~drop].std().item() == pytest.approx(0.01, rel=0.05)


# ---------------------------------------------------------------- MuJoCo session with the scan; rl_expert
def _session(tmp_path, registry, *, scan: bool, ring: bool = False, gap: bool = False, **kw):
    from rrp.envs.mujoco.humanoid_scenes import build_h_gap, build_h_steps
    from rrp.envs.mujoco.legged import LeggedSession
    sc = build_h_gap(BODY, 0, level=1.0) if gap else build_h_steps(BODY, 0, h_frac=0.2)
    b = _binding(sc)
    _put(tmp_path, BODY, "stub", b, scan=scan, ring=ring, seed=5, extra=dict(clock_gate=True))
    registry()
    return LeggedSession(sc, tracker_kind="learned", seed=0, tracker=f"{BODY}:stub", **kw), b


@pytest.mark.menagerie
def test_session_terrain_scan_channel_latency_and_actor_input(tmp_path, registry):
    s, b = _session(tmp_path, registry, scan=True)
    assert "terrain_scan" in s.spec.capabilities
    o = s.reset(0)
    ch = next(c for c in o.declared_sensor_channels if c.name == "0:terrain_scan")
    assert ch.kind == "terrain_elevation_grid" and ch.values.shape == (LC.SCAN_DIM,) and ch.mask.shape == (LC.SCAN_DIM,)
    tr = s.tracker
    assert tr.extra_kind == "terrain_scan" and tr.extra_fn(s.data).shape == (LC.SCAN_DIM,)
    # one tick of latency: what the actor consumes at tick k is the scan of the pose at tick k-1 (the published values)
    seen = []
    s.step(None)
    for _ in range(3):
        prev = s.terrain.exact(s.data)             # exact scan of the pose entering the tick
        pub_before = None
        s.step(None)
        seen.append((prev, s.terrain.values.copy()))
    # published values are noisy versions of the *previous* exact scan: |pub - prev exact| ~ sigma (dropout aside)
    for prev, pub in seen:
        ok = pub != LC.SCAN_DROPOUT_VALUE
        assert np.abs(pub[ok] - prev[ok]).max() < 6 * LC.SCAN_SIGMA + 0.02
    # the channel is deterministic given the seed
    s2, _ = _session(tmp_path, registry, scan=True)
    o2 = s2.reset(0)
    ch2 = next(c for c in o2.declared_sensor_channels if c.name == "0:terrain_scan")
    assert np.array_equal(ch.values, ch2.values) and np.array_equal(ch.mask, ch2.mask)


@pytest.mark.menagerie
def test_session_without_scan_actor_has_no_scan_and_forcing_it_off_is_refused(tmp_path, registry):
    s, _ = _session(tmp_path, registry, scan=False)
    o = s.reset(0)
    assert "terrain_scan" not in s.spec.capabilities and all(c.name != "0:terrain_scan" for c in o.declared_sensor_channels)
    s.step(None)
    with pytest.raises(LT.TrackerMismatch, match="terrain"):
        _session(tmp_path, registry, scan=True, terrain_scan=False)


@pytest.mark.menagerie
def test_rl_expert_labels_and_sha_check(tmp_path, registry):
    from rrp.policies.base import POLICIES
    from rrp.policies.teachers.humanoid import make_rl_expert
    assert "rl_expert" in POLICIES and "rl_expert:*" in POLICIES
    s, b = _session(tmp_path, registry, scan=True)
    pol = make_rl_expert(arg=f"{BODY}:stub")
    assert pol.label.startswith("learned:rl_expert:") and pol.label.endswith(s.tracker.sha256[:12])
    assert pol.info.requires.privileged and "terrain_scan" in pol.info.requires.env_capabilities
    o = s.reset(0)
    pol.reset(s.spec, "h_steps", [0], envs=[s])
    a = pol.act({0: o})[0]
    assert a.command.source == "scripted_teacher" and a.info["source_label"] == pol.label
    assert a.info["command_layer"] == "scripted_teacher:h_steps" and len(a.command.groups["base_velocity"]) == 3
    # an env that runs a different actor is refused
    _put(tmp_path, BODY, "other", b, scan=True, seed=9)
    registry()
    other = make_rl_expert(arg=f"{BODY}:other")
    with pytest.raises(LT.TrackerMismatch, match="not sha"):
        other.reset(s.spec, "h_steps", [0], envs=[s])
    with pytest.raises(KeyError, match="command layer"):
        pol.reset(s.spec, "h_nope", [0], envs=[s])
    # privileged extras (e.g. the pre-D-146 steps actor) are labelled privileged_teacher
    _put(tmp_path, BODY, "priv", b, scan=False, seed=6, extra=dict(extra_obs_dim=5))
    registry()
    assert LT.get_entry(f"{BODY}:priv").extra_obs == "privileged"
    assert make_rl_expert(arg=f"{BODY}:priv").label.startswith("privileged_teacher:rl_expert:")


# ---------------------------------------------------------------- HS1: the public range ring, tracker kinds, energy, capabilities
def test_range_ring_layout_and_sensor_model_math():
    assert (LC.RING_N, LC.RING_RANGE, LC.RING_SIGMA, LC.RING_DROPOUT, LC.RING_LATENCY_TICKS) == (16, 3.0, 0.02, 0.02, 1)
    assert LC.EXTRA_DIM_RING == LC.SCAN_DIM + 16 == 93 and LC.RANGE_RING_VERSION == "range_ring_v1"
    assert np.allclose(LC.RING_ANGLES, np.arange(16) * math.pi / 8) and LC.RING_DROPOUT_VALUE == LC.RING_RANGE
    spec = LC.range_ring_spec()
    assert spec["version"] == "range_ring_v1" and spec["n"] == 16 and spec["frame"] == "yaw" and spec["latency_ticks"] == 1
    # noise N(0, 2 cm), clip to [0, range], 2 % dropout reading the range (= no return)
    v, ok = LC.ring_sensor_model(np.full((20000, 16), 1.5), np.random.default_rng(0))
    assert (~ok).mean() == pytest.approx(0.02, abs=0.002) and np.all(v[~ok] == LC.RING_RANGE)
    assert v[ok].mean() == pytest.approx(1.5, abs=1e-3) and v[ok].std() == pytest.approx(0.02, rel=0.02)
    v, ok = LC.ring_sensor_model(np.array([[0.0, LC.RING_RANGE] * 8]), np.random.default_rng(1))
    assert v.min() >= 0.0 and v.max() <= LC.RING_RANGE                      # clipped on both sides


def test_extra_kind_names_the_public_layouts_and_everything_else_privileged(tmp_path):
    scan, ring = LC.terrain_scan_spec(), LC.range_ring_spec()
    assert LT.extra_kind({}) == "none" and LT.extra_kind({"extra_obs_dim": 0}) == "none"
    assert LT.extra_kind(dict(extra_obs_dim=77, terrain_scan=scan)) == "terrain_scan"
    assert LT.extra_kind(dict(extra_obs_dim=93, terrain_scan=scan, range_ring=ring)) == "terrain_scan+range_ring"
    assert LT.PUBLIC_EXTRA["terrain_scan+range_ring"] == ("terrain_scan", "range_ring")      # scan first
    # a width without the matching sensor versions, or a stale version, is never public
    assert LT.extra_kind(dict(extra_obs_dim=93, terrain_scan=scan)) == "privileged"
    assert LT.extra_kind(dict(extra_obs_dim=93, range_ring=ring)) == "privileged"
    assert LT.extra_kind(dict(extra_obs_dim=93, terrain_scan=scan, range_ring=dict(ring, version="range_ring_v0"))) == "privileged"
    assert LT.extra_kind(dict(extra_obs_dim=77, terrain_scan=scan, range_ring=ring)) == "privileged"
    assert LT.extra_kind(dict(extra_obs_dim=8)) == "privileged"
    d = tmp_path / "anymal" / "gap_v1"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps(dict(extra_obs_dim=93, terrain_scan=scan, range_ring=ring)))
    assert LT.scan_trackers(tmp_path)[("anymal", "gap_v1")].extra_obs == "terrain_scan+range_ring"


def test_humanoid_generator_bodies_declare_only_declarable_capabilities():
    """Red before HS1: a phum body with arms declared capability 'manipulate', which the Capability vocabulary rejects."""
    from rrp.bodies.humanoid_gen import sample_params
    from rrp.envs.mujoco.legged import build_waypoint_contact
    seed = next(i for i in range(200) if sample_params(i).arm_dof > 0)
    sc = build_waypoint_contact(f"phum_{seed}", 0)
    assert sc.robots[0].meta["assemblies"][0]["capabilities"] == ["locomote"]


def test_legged_env_factory_has_no_humanoid_branch():
    from rrp.envs.mujoco.legged import make_legged_env
    for task in ("h_steps", "h_gap"):
        with pytest.raises(KeyError, match="no task"):
            make_legged_env(task=task, body=BODY)
    from rrp.envs.base import _factory
    assert _factory("mujoco/legged", "h_gap").__name__ == "make_humanoid_session"


@pytest.mark.menagerie
def test_warp_gap_ring_matches_mujoco_rays_without_cuda():
    """Analytic wall boxes (WarpGapEnv.ring_exact, stub env on CPU torch) == mj_ray (RangeRing.exact) on the h_gap scene at
    several poses; the Warp sensor model reproduces the numpy one; the critic/actor block widths are scan + ring."""
    import mujoco
    import torch
    import rrp.envs.warp.task_env as TK
    from rrp.envs.mujoco.humanoid_scenes import build_h_gap
    assert (TK.RING_N, TK.RING_RANGE, TK.RING_SIGMA, TK.RING_DROPOUT, TK.RING_DROPOUT_VALUE) == (
        LC.RING_N, LC.RING_RANGE, LC.RING_SIGMA, LC.RING_DROPOUT, LC.RING_DROPOUT_VALUE)
    sc = build_h_gap(BODY, 0, level=1.0)
    b = _binding(sc)
    L, gw, yc = float(sc.meta["L"]), float(sc.meta["gap_width"]), float(sc.meta["y_c"])
    data = mujoco.MjData(sc.model)
    ring = LC.RangeRing(b)
    poses = [(0.5 * L, yc, 0.0), (1.2 * L, yc + 0.3 * L, 0.4), (1.5 * L, yc - 0.6 * L, -0.7), (0.8 * L, yc + 2.5 * L, 2.0),
             (1.8 * L, yc, 0.0), (0.2 * L, yc - 0.1, math.pi)]
    env = object.__new__(TK.WarpGapEnv)
    env.dev, env.qa = torch.device("cpu"), b.qa
    env.L = torch.full((len(poses),), L)
    env.gap_y, env.gap_w = torch.full((len(poses),), yc), torch.full((len(poses),), gw)
    env._ring_ang = torch.as_tensor(LC.RING_ANGLES, dtype=torch.float32)
    qp = []
    for (x, y, yaw) in poses:
        mujoco.mj_resetData(sc.model, data)
        data.qpos[b.qa:b.qa + 7] = [x, y, b.nominal_height(), math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        mujoco.mj_forward(sc.model, data)
        qp.append(data.qpos.copy())
    env.qpos = torch.tensor(np.array(qp), dtype=torch.float32)
    got = env.ring_exact().numpy()
    hits = 0
    for i, q in enumerate(qp):
        data.qpos[:] = q
        mujoco.mj_forward(sc.model, data)
        want = ring.exact(data)
        assert np.abs(got[i] - want).max() < 2e-3, (i, poses[i], got[i], want)
        hits += int((want < LC.RING_RANGE).sum())
    assert hits >= 8 and (got == LC.RING_RANGE).any()            # the walls are seen, and open directions read the range
    env.gen = torch.Generator().manual_seed(0)
    env._scan_off = torch.as_tensor(LC.SCAN_OFFSETS, dtype=torch.float32)
    pub = env._scan_model(torch.cat([torch.full((4000, 77), 0.4), torch.full((4000, 16), 1.5)], -1))
    sc_, rg = pub[:, :77], pub[:, 77:]
    assert pub.shape == (4000, LC.EXTRA_DIM_RING) and (sc_ == LC.SCAN_DROPOUT_VALUE).float().mean().item() == pytest.approx(0.02, abs=0.004)
    drop = rg == LC.RING_DROPOUT_VALUE
    assert drop.float().mean().item() == pytest.approx(0.02, abs=0.004) and rg[~drop].std().item() == pytest.approx(0.02, rel=0.05)


@pytest.mark.menagerie
def test_session_range_ring_channel_capability_actor_input_and_refusals(tmp_path, registry):
    s, b = _session(tmp_path, registry, scan=False, ring=True, gap=True)
    assert {"terrain_scan", "range_ring"} <= set(s.spec.capabilities)
    o = s.reset(0)
    ch = next(c for c in o.declared_sensor_channels if c.name == "0:range_ring")
    assert ch.kind == "range_ring_m" and ch.values.shape == (LC.RING_N,) and ch.mask.shape == (LC.RING_N,)
    assert any(c.name == "0:terrain_scan" for c in o.declared_sensor_channels)
    tr = s.tracker
    assert tr.extra_kind == "terrain_scan+range_ring"
    x = tr.extra_fn(s.data)
    assert x.shape == (LC.EXTRA_DIM_RING,)
    assert np.array_equal(x[:LC.SCAN_DIM], s.terrain.values) and np.array_equal(x[LC.SCAN_DIM:], s.ring.values)   # scan first
    # one tick of latency, exactly: a tick publishes the sensor model of what the previous tick measured, then measures the pose
    # entering this tick
    import copy
    s.step(None)
    for _ in range(3):
        before_prev, rng, entering = s.ring.prev.copy(), copy.deepcopy(s.ring.rng), s.ring.exact(s.data)
        s._tracker_tick(s.cmd)
        want, ok = LC.ring_sensor_model(before_prev, rng)
        assert np.array_equal(s.ring.values, want) and np.array_equal(s.ring.valid, ok)
        assert np.array_equal(s.ring.prev, entering)
    # deterministic given the seed; snapshot/restore replays the ring stream exactly
    s2, _ = _session(tmp_path, registry, scan=False, ring=True, gap=True)
    ch2 = next(c for c in s2.reset(0).declared_sensor_channels if c.name == "0:range_ring")
    assert np.array_equal(ch.values, ch2.values) and np.array_equal(ch.mask, ch2.mask)
    snap = s.snapshot()
    s.step(None)
    a = s.ring.values.copy()
    s.restore(snap)
    s.step(None)
    assert np.array_equal(a, s.ring.values)
    with pytest.raises(LT.TrackerMismatch, match="range_ring"):
        _session(tmp_path, registry, scan=False, ring=True, gap=True, range_ring=False)
    # a wrong ring layout in the meta is refused at load
    _put(tmp_path, BODY, "badring", b, scan=False, ring=True, seed=7, extra=dict(range_ring=dict(LC.range_ring_spec(), n=8)))
    registry()
    with pytest.raises(LT.TrackerMismatch, match="range-ring layout"):
        LT.load_tracker(BODY, b, dict(s.robots[0].meta, contact_model="contact_v2"), tracker=f"{BODY}:badring")


@pytest.mark.menagerie
def test_session_ring_is_optional_public_channel_on_other_actors_and_absent_by_default(tmp_path, registry):
    s, _ = _session(tmp_path, registry, scan=True, gap=True)
    o = s.reset(0)
    assert "range_ring" not in s.spec.capabilities and all(c.name != "0:range_ring" for c in o.declared_sensor_channels)
    assert s.tracker.extra_fn(s.data).shape == (LC.SCAN_DIM,)
    s3, _ = _session(tmp_path, registry, scan=True, gap=True, range_ring=True)       # channel only; the actor input is unchanged
    o3 = s3.reset(0)
    assert "range_ring" in s3.spec.capabilities and any(c.name == "0:range_ring" for c in o3.declared_sensor_channels)
    assert s3.tracker.extra_fn(s3.data).shape == (LC.SCAN_DIM,)


@pytest.mark.menagerie
def test_step_result_reports_mechanical_energy_and_none_when_the_tick_is_replaced(tmp_path, registry, monkeypatch):
    import mujoco
    import rrp.envs.mujoco.legged as LG
    from rrp.envs.base import StepResult
    assert StepResult(None, None, 0.0).energy_j is None                 # not measured unless the env reports it
    s, b = _session(tmp_path, registry, scan=True)
    s.reset(0)
    spent = []
    real = mujoco.mj_step

    def spy(m, d, *a, **k):
        real(m, d, *a, **k)
        spent.append(float(np.sum(np.abs(d.actuator_force[b.pol_act] * d.actuator_velocity[b.pol_act]))) * float(m.opt.timestep))
    monkeypatch.setattr(LG.mujoco, "mj_step", spy)
    r1 = s.step(None)
    assert r1.energy_j is not None and r1.energy_j > 0.0
    n = len(spent)
    r2 = s.step(None)
    assert r2.energy_j == pytest.approx(sum(spent[n:]), rel=1e-9) and len(spent) - n == int(round(s.dt * LG.TRACKER_HZ)) * int(
        round(1.0 / (LG.TRACKER_HZ * s.model.opt.timestep)))
    assert r2.energy_j != r1.energy_j                                    # per step, not cumulative
    # perturb.install_legged replaces the tick and owns the substeps: energy is not measured there
    from rrp.envs.mujoco import perturb
    s.reset(0)
    perturb.install_legged(s, perturb.PhysicsPerturbation(), 0)
    assert s.step(None).energy_j is None


# ---------------------------------------------------------------- recipes: one registry, shas preserved
_RECIPE_SHAS = {
    "h1_clock_scratch": "e6fc9cf66864095887f0d228c40117452fcf91581d9b2c9e94567c86b0a6ae65",
    "t1_turn_latency_ft": "d90d599641cdc7c6faf376f570902936991e4cdf22cc8f680c0d1ab85901eaa7",
    "t1_steps_gpu": "e556cf2d0c2fae410dc4d4448a0a7b7c58370d0ec9190a724d6c110cc5402ef0",
    "h1_steps_gpu_v2": "2f6a90f42a85ab9627bd7421001f6dd2bf9c8b3d796d83f0c1b512a5cc210040",
    "t1_clock_gpu_v2ft4": "b49005d954dba0e3fa27176ee9b484a55201618c55ae4bd65c7ec9162685ce0b",
    "shared_morph_v2": "8499ab78ea124e96030efef4357824aff3982baa02835ecdbb1540eb53e82085",
}


def test_recipe_registry_is_single_and_records_are_stable():
    from rrp.harness.train import tracker_recipes as R
    assert not (Path(R.__file__).parent / "humanoid_recipes.py").exists()
    assert not set(R.CPU_RECIPES) & set(R.WARP_RECIPES) and set(R.RECIPES) == set(R.CPU_RECIPES) | set(R.WARP_RECIPES)
    opts, rec = R.recipe_record("t1_steps_gpu")
    assert rec["name"] == "t1_steps_gpu" and rec["module"] == "rrp.harness.train.tracker_recipes" and opts["task"] == "steps"
    assert rec["sha256"] == __import__("hashlib").sha256(json.dumps(R.WARP_RECIPES["t1_steps_gpu"], sort_keys=True).encode()).hexdigest()
    # recorded shas of recipes that already produced actors (pinned when the two registries were merged)
    for name, sha in _RECIPE_SHAS.items():
        assert R.recipe_record(name)[1]["sha256"] == sha, name


# ---------------------------------------------------------------- trainer meta on a CPU fake env
class _FakeEnv:
    """The env surface `warp_tracker_ppo.train` uses, on CPU: obs = noise + a slot that carries the 'scan'."""

    def __init__(self, extra_dim: int, N: int = 16):
        import torch
        self.N, self.nf, self.nA, self.dt = N, 2, 3, 0.02
        self.b = type("B", (), dict(kind="biped", obs_dim=10, priv_dim=4))()
        self.extra_dim = extra_dim
        self.upper_dim = 0
        self.obs_dim, self.priv_dim = 10 + extra_dim, 4 + 1 + extra_dim
        self.meta = dict(contact_model="contact_v2", actuator_limits="sourced_v1")
        self.adaptations = []
        self.cfg0 = type("C", (), dict(options=lambda s: {}))()
        self.g = torch.Generator().manual_seed(0)

    def set_alpha(self, a):
        return dict(alpha=a)

    def pop_stats(self):
        return dict(episodes=2, ret_sum=1.0, len_sum=10, falls=1, tracking=0.0)

    def observe(self):
        import torch
        return torch.randn(self.N, self.obs_dim, generator=self.g)

    def privileged(self, fc):
        import torch
        return torch.randn(self.N, self.priv_dim, generator=self.g)

    def step(self, a):
        import torch
        z = torch.zeros(self.N, dtype=torch.bool)
        return self.observe(), self.privileged(None), torch.randn(self.N, generator=self.g) * 0.1, z, z


@pytest.mark.parametrize("extra", [0, LC.SCAN_DIM])
def test_trainer_runs_two_updates_and_writes_scan_meta(tmp_path, extra):
    import torch
    from rrp.harness.train import warp_tracker_ppo as W
    args = W.build_args(["--body", BODY, "--out", str(tmp_path / "run"), "--iters", "2", "--horizon", "4", "--epochs", "1",
                         "--minibatches", "2", "--hidden", "16,8", "--alpha-schedule", "fixed:1.0", "--ckpt-every", "1"] +
                        (["--terrain-scan"] if extra else []))
    assert args.terrain_scan == bool(extra)
    W.train(args, _FakeEnv(extra), dev=torch.device("cpu"), engine="fake")
    st = torch.load(tmp_path / "run" / "actor.pt", weights_only=False)
    meta = st["meta"]
    assert meta["iter"] == 1 and meta["extra_obs_dim"] == extra and meta["obs_dim"] == 10 + extra
    assert meta["extra_obs"] == ("terrain_scan" if extra else "none")
    assert ("terrain_scan" in meta) == bool(extra)
    if extra:
        assert meta["terrain_scan"] == LC.terrain_scan_spec() and LT.extra_kind(meta) == "terrain_scan"
    assert st["actor"]["0.weight"].shape[1] == 10 + extra
    assert len((tmp_path / "run" / "train_log.jsonl").read_text().splitlines()) == 2


class _FakeGapEnv(_FakeEnv):
    """A WarpGapEnv-shaped actor: extra block = scan + ring (93), declared the way WarpGapEnv declares it."""
    public_extra = ("terrain_scan", "range_ring")

    def __init__(self):
        super().__init__(LC.EXTRA_DIM_RING)


def test_warp_envs_declare_their_public_extra_block():
    import rrp.envs.warp.task_env as TK
    assert TK.WarpGapEnv.public_extra == ("terrain_scan", "range_ring") and TK.WarpStepsEnv.public_extra == ("terrain_scan",)
    assert LC.SCAN_DIM + LC.RING_N == LC.EXTRA_DIM_RING == 93


def test_gap_actor_round_trips_as_public_scan_and_ring(tmp_path):
    """Item 3: a WarpGapEnv actor (scan + ring, 93) is written as public scan+ring, not `privileged`, and install records it."""
    import torch
    from rrp.harness.train import warp_tracker_ppo as W
    from rrp.harness.train.tracker_training import install
    args = W.build_args(["--body", BODY, "--out", str(tmp_path / "run"), "--iters", "1", "--horizon", "4", "--epochs", "1",
                         "--minibatches", "2", "--hidden", "16,8", "--alpha-schedule", "fixed:1.0", "--ckpt-every", "1", "--terrain-scan"])
    W.train(args, _FakeGapEnv(), dev=torch.device("cpu"), engine="fake")
    st = torch.load(tmp_path / "run" / "actor.pt", weights_only=False)
    meta = st["meta"]
    assert meta["extra_obs_dim"] == 93 and meta["extra_obs"] == "terrain_scan+range_ring"
    assert meta["terrain_scan"] == LC.terrain_scan_spec() and meta["range_ring"] == LC.range_ring_spec()
    assert meta["public_extra"] == [dict(sensor="terrain_scan", dim=77, version=LC.TERRAIN_SCAN_VERSION),
                                    dict(sensor="range_ring", dim=16, version=LC.RANGE_RING_VERSION)]
    assert LT.extra_kind(meta) == "terrain_scan+range_ring"
    import hashlib
    run = tmp_path / "run"
    sha = hashlib.sha256((run / "actor.pt").read_bytes()).hexdigest()
    v = tmp_path / "validation.json"
    v.write_text(json.dumps(dict(body=BODY, tracker_sha=sha, w6_gate=dict(verdict="pass", failed=[]))))
    store = install(run, [v], BODY, "ring_v1", label="gap ring", root=tmp_path / "store")
    assert json.loads((store / "meta.json").read_text())["extra_obs"] == "terrain_scan+range_ring"
    assert LT.scan_trackers(tmp_path / "store")[(BODY, "ring_v1")].extra_obs == "terrain_scan+range_ring"


def test_trainer_refuses_an_undeclared_extra_block(tmp_path):
    """No guessing from the width: a 93-wide block with no declaration is not labelled (and not written)."""
    import torch
    from rrp.harness.train import warp_tracker_ppo as W
    args = W.build_args(["--body", BODY, "--out", str(tmp_path / "run"), "--iters", "1", "--horizon", "4", "--hidden", "16,8"])
    with pytest.raises(ValueError, match="public_extra"):
        W.train(args, _FakeEnv(LC.EXTRA_DIM_RING), dev=torch.device("cpu"), engine="fake")
    bad = _FakeGapEnv()
    bad.public_extra = ("terrain_scan",)                          # declared dims (77) != extra_dim (93)
    with pytest.raises(ValueError, match="dims"):
        W.train(args, bad, dev=torch.device("cpu"), engine="fake")
    assert not (tmp_path / "run" / "meta.json").exists()


# ---------------------------------------------------------------- G0: fresh-checkout skips name what is missing; slow is registered
def test_need_weights_and_need_assets_skip_with_the_missing_path_named(tmp_path, monkeypatch):
    import tests.conftest as C
    missing = tmp_path / "t1" / "contact_v2" / "actor.pt"
    with pytest.raises(pytest.skip.Exception, match=str(missing)):
        need_weights(missing)
    missing.parent.mkdir(parents=True)
    missing.write_bytes(b"")
    assert need_weights(missing) == missing                           # present: no skip
    monkeypatch.setattr(C, "MENAGERIE", tmp_path / "no_menagerie")
    with pytest.raises(pytest.skip.Exception, match="Menagerie assets not fetched"):
        need_assets()
    monkeypatch.setattr(C, "MENAGERIE", tmp_path)
    assert need_assets() == tmp_path


def test_slow_marker_is_registered_and_marks_the_long_pointer_tests():
    import tomllib
    cfg = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"]
    assert any(m.startswith("slow:") for m in cfg["markers"]) and cfg["addopts"] == "-ra"     # the gate passes -m "not slow"
    from test_pointer_copy import test_copy_head_types_instr_n_typed_on_unseen_strings_and_characters as copy_head
    from test_pointer_train_smoke import test_ui_factors_train_all_three_trainers_and_supervise_drag_to as ui_trainers
    for t in (copy_head, ui_trainers):
        assert "slow" in {m.name for m in t.pytestmark}
