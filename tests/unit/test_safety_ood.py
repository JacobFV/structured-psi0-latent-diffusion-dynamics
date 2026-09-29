"""D-126 #29 packet OOD detector and #30 safety layer: synthetic known answers."""
import numpy as np
import pytest

from rrp.policies.packet_ood import GaussianSubspaceModel, PacketOODModel, PacketOODMonitor
from rrp.policies.safety import SafetyConfig, SafetyLayer, SafetyLimits


# ------------------------------------------------------------------ OOD
def _packets(rng, n, scale=1.0, shift=0.0):
    """In-distribution packets live near a 3-dim subspace of the 4x5x8 packet space."""
    basis = np.random.default_rng(42).normal(0, 1, (3, 4 * 5 * 8))
    c = rng.normal(0, 1, (n, 3)) * scale
    x = c @ basis + rng.normal(0, 0.05, (n, basis.shape[1])) + shift
    return x.reshape(n, 4, 5, 8)


def test_gaussian_score_equals_mahalanobis_when_full_rank():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (400, 3)) @ np.array([[2, 0, 0], [0.5, 1, 0], [0, 0, 0.3]])
    m = GaussianSubspaceModel.fit(X, r=3, shrink=0.0)
    Y = (X - m.mean) / m.scale
    C = np.cov(Y.T)
    d2 = np.einsum("ij,jk,ik->i", Y, np.linalg.inv(C), Y)
    assert np.allclose(m.score(X), d2, rtol=1e-8)


def test_detects_off_manifold_and_scaled_packets():
    rng = np.random.default_rng(1)
    zf, zc = _packets(rng, 400), _packets(rng, 200)
    model = PacketOODModel.fit(zf, zc, latent_space_version="ls-test", r=8)
    ind = model.scores(_packets(rng, 200))
    assert (ind > model.thresholds["enforce"]).mean() < 0.02          # ~q0.999 calibration
    off = model.scores(_packets(rng, 100) + rng.normal(0, 0.5, (100, 4, 5, 8)))   # off the subspace
    big = model.scores(_packets(rng, 100, scale=10.0))                # on the subspace, far along it (c ~ 10 sigma)
    zero = model.score(np.zeros((4, 5, 8)) - 3.0)
    assert (off > model.thresholds["enforce"]).mean() > 0.95
    assert (big > model.thresholds["enforce"]).mean() > 0.8
    assert zero > model.thresholds["enforce"]


def test_token_feature_and_asm_mask():
    rng = np.random.default_rng(2)
    zf, zc = _packets(rng, 200), _packets(rng, 100)
    mask = [True, True, True, False, False]
    m = PacketOODModel.fit(zf, zc, latent_space_version="ls", feature="token", asm_mask=mask, r=4)
    assert m.n_assemblies == 3
    z = zc[0].copy()
    s0 = m.score(z, mask)
    z[:, 4] = 1e3                                                     # masked-out assembly: ignored
    assert m.score(z, mask) == s0


def test_save_load_fingerprint_and_lsv(tmp_path):
    rng = np.random.default_rng(3)
    m = PacketOODModel.fit(_packets(rng, 100), _packets(rng, 50), latent_space_version="ls-a", r=4, body="go2")
    m.save(tmp_path / "ood")
    m2 = PacketOODModel.load(tmp_path / "ood", expect_lsv="ls-a")
    z = _packets(rng, 1)[0]
    assert m2.score(z) == m.score(z) and m2.fingerprint() == m.fingerprint()
    with pytest.raises(ValueError, match="fitted for latent space"):
        PacketOODModel.load(tmp_path / "ood", expect_lsv="ls-b")
    np.savez(tmp_path / "ood.npz", **dict(m.model.arrays(), mean=m.model.mean + 1))     # tamper
    with pytest.raises(ValueError, match="fingerprint"):
        PacketOODModel.load(tmp_path / "ood")


def test_monitor_modes():
    rng = np.random.default_rng(4)
    m = PacketOODModel.fit(_packets(rng, 100), _packets(rng, 50), latent_space_version="ls", r=4)
    bad = _packets(rng, 1, shift=5.0)[0]
    assert PacketOODMonitor(None, "off").check(bad) is None
    r = PacketOODMonitor(m, "monitor").check(bad)
    assert r["over_enforce"] and not r["rejected"]
    mon = PacketOODMonitor(m, "enforce")
    assert mon.check(bad)["rejected"] and mon.summary()["rejected"] == 1
    with pytest.raises(ValueError):
        PacketOODMonitor(None, "enforce")


# ------------------------------------------------------------------ safety
def _limits(n=2):
    return SafetyLimits(lo=np.full(n, -1.0), hi=np.full(n, 1.0), effort=np.full(n, 10.0), qd_max=np.full(n, 5.0),
                        kp=np.full(n, 100.0), kd=np.full(n, 2.0), hold=np.zeros(n), tilt_limit=1.0, source="test")


def test_position_rate_torque_velocity_clamps():
    L = SafetyLayer(_limits(), SafetyConfig(margin=0.1, rate_max=1.0), "enforce")
    q, qd = np.zeros(2), np.zeros(2)
    # position: 2.0 -> 0.9, then torque: |kp (u - q)| <= 10 -> u <= 0.1
    u = L.filter(np.array([2.0, -0.05]), q, qd, 0.0, 0.02)
    assert np.allclose(u, [0.1, -0.05])
    # rate: from 0.1 by at most 1 rad/s * 0.02 s; q moved so torque allows it
    u = L.filter(np.array([0.5, -0.05]), np.array([0.1, 0.0]), qd, 0.02, 0.02)
    assert np.isclose(u[0], 0.12)
    # velocity guard: joint 1 too fast -> hold measured q
    u = L.filter(np.array([0.12, 0.0]), np.array([0.12, 0.3]), np.array([0.0, 6.0]), 0.04, 0.02)
    assert u[1] == 0.3
    s = L.summary()
    assert s["clamp_pos"] == 1 and s["clamp_torque"] >= 1 and s["clamp_rate"] >= 1 and s["vel_guard"] == 1


def test_torque_clamp_is_exact_for_pd():
    L = SafetyLayer(_limits(1), SafetyConfig(margin=0.0, rate_max=1e9), "enforce")
    q, qd = np.array([0.2]), np.array([1.5])
    u = L.filter(np.array([0.9]), q, qd, 0.0, 0.02)
    tau = 100.0 * (u - q) - 2.0 * qd
    assert np.isclose(tau[0], 10.0)


def test_monitor_passes_through_but_counts():
    L = SafetyLayer(_limits(), SafetyConfig(), "monitor")
    raw = np.array([5.0, 0.0])
    assert L.filter(raw, np.zeros(2), np.zeros(2), 0.0, 0.02) is raw
    assert L.stats.clamp_pos == 1 and L.stats.modified == 1


def test_safe_stop_ramps_to_hold_and_fall_hook():
    L = SafetyLayer(_limits(), SafetyConfig(margin=0.0, rate_max=1e9, stop_s=1.0, fall_ticks=3), "enforce")
    lim = L.L
    lim.effort[:] = 1e9
    L.filter(np.array([0.8, -0.8]), np.zeros(2), np.zeros(2), 0.0, 0.02)
    L.safe_stop(0.0, reason="test")
    mid = L.filter(np.array([0.8, -0.8]), np.zeros(2), np.zeros(2), 0.5, 0.02)
    end = L.filter(np.array([0.8, -0.8]), np.zeros(2), np.zeros(2), 1.5, 0.02)
    assert np.allclose(mid, [0.4, -0.4]) and np.allclose(end, [0.0, 0.0])
    # fall detection -> hook (default: safe stop) after fall_ticks consecutive tilted ticks
    calls = []
    F = SafetyLayer(_limits(), SafetyConfig(fall_ticks=3), "enforce", on_fall=lambda layer, t: calls.append(t))
    for k, tilt in enumerate([0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9]):
        F.filter(np.zeros(2), np.zeros(2), np.zeros(2), 0.02 * k, 0.02, tilt=tilt)
    assert F.fallen and calls == [pytest.approx(0.1)]


def test_legged_limits_sourced_for_go2_else_estimate():
    from rrp.bodies.legged import hexapod, standalone_model
    from rrp.policies.safety import legged_limits
    from rrp.envs.mujoco.legged_core import LeggedBinding
    model, _, meta = standalone_model(hexapod())
    L = legged_limits(LeggedBinding(model, meta), "hexapod6")
    assert L.source == "model+vmax_estimate" and (L.lo < L.hi).all() and (L.effort > 0).all()
    assert ((L.hold >= L.lo) & (L.hold <= L.hi)).all()
