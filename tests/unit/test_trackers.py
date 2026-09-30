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

BODY = "t1"


def _binding(sc):
    mr = sc.robots[0]
    return LC.LeggedBinding(sc.model, mr.meta, mr.prefix)


def _put(root: Path, body: str, version: str, b, *, scan: bool, seed: int, extra: dict | None = None, v1: bool = False,
         contact: str = "contact_v2") -> Path:
    """A registry entry: actor.pt + meta.json under root/<body>[/<version>]."""
    d = root / body if v1 else root / body / version
    d.mkdir(parents=True, exist_ok=True)
    meta = dict(body=body, actuator_limits=b.meta.get("actuator_limits"), contact_model=contact)
    if scan:
        meta.update(extra_obs_dim=LC.SCAN_DIM, terrain_scan=LC.terrain_scan_spec())
    meta.update(extra or {})
    _saved_actor(d, "actor", b.obs_dim + (LC.SCAN_DIM if scan else 0), b.n, meta, seed)
    (d / "meta.json").write_text(json.dumps(dict(meta, obs_dim=b.obs_dim + (LC.SCAN_DIM if scan else 0))))
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
    T = LT.scan_trackers()
    assert T[("t1", "contact_v2")].sha256 == "36e9146792743115878c34e0bbf7cc46ccb5419417921358da3658c8377fc591"
    assert T[("anymal_c", "contact_v2")].sha256 == "2a16532bbd07f7abc662bdda10df6bfb70fca2ec11d59f9567142e92a2f22d95"
    assert ("t1", "contact_v1") in T and T[("anymal_c", "contact_v2_rejected_iter2499")].decision == "rejected"
    assert all(e.extra_obs == "none" for e in T.values())         # nothing committed takes the scan yet


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
def _session(tmp_path, registry, *, scan: bool, **kw):
    from rrp.envs.mujoco.humanoid_scenes import build_h_steps
    from rrp.envs.mujoco.legged import LeggedSession
    sc = build_h_steps(BODY, 0, h_frac=0.2)
    b = _binding(sc)
    _put(tmp_path, BODY, "stub", b, scan=scan, seed=5, extra=dict(clock_gate=True))
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
