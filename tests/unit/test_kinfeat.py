"""D-137 `kinfeat` ablation flag: default off = legacy features; on = base-frame joint axes at home."""
import numpy as np
import pytest

from rrp.features import kinfeat


def _feat(robot_key, yaw=0.0):
    from rrp.bodies.catalog import workbench_robots
    from rrp.envs.native import Session
    from rrp.envs.scenario import build_pick_place
    from rrp.features.featurizer import featurizer_for
    s = Session(build_pick_place(workbench_robots()[robot_key](), 5, base=((0.0, 0.0, 0.0), yaw)), seed=5)
    f = featurizer_for(s)
    return f, f(s.observe())


def test_off_by_default_keeps_local_axes(monkeypatch):
    monkeypatch.delenv(kinfeat.ENV, raising=False)
    f, pi = _feat("parm6_pg2")
    local = np.array([j.axis for j in f.spec.joints if j.name in f.node_joint_names], np.float32)
    by_name = {j.name: j.axis for j in f.spec.joints}
    np.testing.assert_array_equal(f.node_static[:, 2:5], np.array([by_name[n] for n in f.node_joint_names], np.float32))
    assert local.shape[0] == len(f.node_joint_names)


def test_bad_value_rejected(monkeypatch):
    monkeypatch.setenv(kinfeat.ENV, "v9")
    with pytest.raises(ValueError):
        kinfeat.enabled()


def test_on_gives_base_frame_axes_invariant_to_mount_yaw(monkeypatch):
    monkeypatch.setenv(kinfeat.ENV, "v1")
    f0, p0 = _feat("parm6_pg2", 0.0)
    f1, p1 = _feat("parm6_pg2", 0.7)
    np.testing.assert_allclose(f0.node_static, f1.node_static, atol=1e-5)
    ax0 = f0.node_static[:, 2:5]
    assert np.allclose(np.linalg.norm(ax0, axis=1), 1, atol=1e-5)
    # parm6 = z y y z y z: at home the base yaw joint is vertical and pitch axes are horizontal
    assert abs(ax0[0, 2]) > 0.99 and abs(ax0[1, 2]) < 1e-6
    # the dynamic world-axis column is in the base frame -> same for both mounts at the same joint state
    S = f0.static_dim
    col = slice(S + 6, S + 9)
    np.testing.assert_allclose(p0.act_node_feats[:, col], p1.act_node_feats[:, col], atol=1e-4)
    # the load-time table equals the runtime static columns
    np.testing.assert_allclose(kinfeat.table_for_robot("parm6_pg2"), ax0, atol=1e-6)


def test_apply_rows_is_idempotent(monkeypatch):
    tab = kinfeat.table_for_robot("parm5_pg2")
    n = tab.shape[0]
    nodes = np.random.default_rng(0).normal(size=(3, n + 2, 40)).astype(np.float32)
    out = kinfeat.apply_rows(nodes.copy(), np.array([4, 4, 4]), {4: "parm5_pg2"}, np.array([n, n, n]))
    np.testing.assert_array_equal(out[:, :n, 2:5], np.broadcast_to(tab, (3, n, 3)))
    np.testing.assert_array_equal(out[:, :, 5:], nodes[:, :, 5:])
    again = kinfeat.apply_rows(out.copy(), np.array([4, 4, 4]), {4: "parm5_pg2"}, np.array([n, n, n]))
    np.testing.assert_array_equal(again, out)
    with pytest.raises(ValueError):
        kinfeat.apply_rows(nodes.copy(), np.array([4]), {4: "parm5_pg2"}, np.array([n + 1]))


def test_checkpoint_guard(monkeypatch):
    from rrp.models.checkpoint import check_kinfeat
    monkeypatch.delenv(kinfeat.ENV, raising=False)
    check_kinfeat(dict(model={}, versions={}))
    with pytest.raises(ValueError):
        check_kinfeat(dict(model={}, versions={"kinfeat": kinfeat.VERSION}))
    monkeypatch.setenv(kinfeat.ENV, "v1")
    check_kinfeat(dict(model={}, versions={"kinfeat": kinfeat.VERSION}))
    with pytest.raises(ValueError):
        check_kinfeat(dict(model={}, versions={}))
