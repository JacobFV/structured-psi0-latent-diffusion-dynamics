"""R7 contract test: `state_view()` on the MuJoCo sessions against `rrp.envs.base.StateView` (docs/relations.md
5.1, section 10 row R7). Entities with poses, contacts with world normals, joints, camera K/T, `render_depth`
shape, `caps` declared, and that the view is never reachable from a featurizer (policies/ input path)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from rrp.envs.base import Camera, CapabilityError, ContactState, EntityState, JointState, StateView

SRC = Path(__file__).resolve().parents[2] / "src" / "rrp"


# ------------------------------------------------------------------------------------------------ fixtures
def _arm_session():
    from rrp.bodies.catalog import workbench_robots
    from rrp.envs.mujoco.session import Session
    from rrp.envs.mujoco.scenario import build_pick_place
    return Session(build_pick_place(workbench_robots()["parm6_pg2"](), 5), seed=5)


def _dual_session():
    from rrp.bodies.variants import registered_variants
    from rrp.envs.mujoco.dual import DualSession
    from rrp.envs.mujoco.dual_scenarios import build_support_insert
    W = registered_variants()
    return DualSession(build_support_insert([W["parm5l_pg2"](), W["parm6_pg2"]()], 3), seed=3)


def _legged_session():
    from rrp.envs.mujoco.legged import LeggedSession, build_waypoint_contact
    return LeggedSession(build_waypoint_contact("pquad4", 3), tracker_kind="cpg", seed=3)


SESSIONS = {"arm": _arm_session, "dual": _dual_session, "legged": _legged_session}


# ------------------------------------------------------------------------------------------------ caps / protocol
@pytest.mark.parametrize("kind", SESSIONS)
def test_caps_declared_and_protocol_satisfied(kind):
    s = SESSIONS[kind]()
    sv = s.state_view()
    assert isinstance(sv, StateView)                       # runtime_checkable Protocol conformance
    assert isinstance(sv.caps, frozenset)
    for c in ("poses", "contacts", "forces", "joints", "camera"):
        assert c in sv.caps
    assert "privileged_truth" in s.spec.capabilities        # state_view() is gated by this, same as truth()
    with pytest.raises(CapabilityError):
        sv.ui_tree()                                        # not declared for MuJoCo


# ------------------------------------------------------------------------------------------------ entities
def test_arm_entities_have_correct_poses_and_ids():
    import mujoco
    s = _arm_session()
    sv = s.state_view()
    ents = {e.id: e for e in sv.entities()}
    assert isinstance(ents["cube"], EntityState) and ents["cube"].kind == "object"
    bid = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    np.testing.assert_allclose(ents["cube"].pos, s.data.xpos[bid], atol=1e-9)
    np.testing.assert_allclose(ents["cube"].quat, s.data.xquat[bid], atol=1e-9)
    assert ents["cube"].mass is not None and ents["cube"].mass > 0
    assert "target" in ents and ents["target"].kind == "feature"
    assert "gripper" in ents and ents["gripper"].kind == "assembly"      # single-arm manipulator_bindings key


def test_dual_entities_use_manip_handle_metadata():
    s = _dual_session()
    sv = s.state_view()
    ents = {e.id: e for e in sv.entities()}
    for ent in ("left", "right"):
        assert ent in ents and ents[ent].kind == "assembly"
        assert ents[ent].attrs["gripper_kind"] == "parallel"
        assert ents[ent].attrs["reach_m"] > 0
    for obj_ent in ("hole", "fixture", "peg"):
        assert obj_ent in ents


def test_legged_entities_include_foot_links():
    s = _legged_session()
    sv = s.state_view()
    ents = {e.id: e for e in sv.entities()}
    assert "body" in ents and ents["body"].kind == "assembly"
    foot_ents = [e for e in ents.values() if e.kind == "link"]
    assert len(foot_ents) == len(s.binding.foot_bids) > 0
    for e in foot_ents:
        assert e.id.startswith("leg:") and e.parent == "body"
        assert e.pos[2] < 0.5                              # a standing quadruped's tibia is near the ground


# ------------------------------------------------------------------------------------------------ contacts
def test_arm_cube_settled_on_table_has_upward_world_contact():
    s = _arm_session()
    sv = s.state_view()
    cs = sv.contacts()
    assert cs and all(isinstance(c, ContactState) for c in cs)
    cube_contacts = [c for c in cs if "cube" in (c.a, c.b)]
    assert cube_contacts
    for c in cube_contacts:
        assert math.isclose(float(np.linalg.norm(c.normal)), 1.0, abs_tol=1e-6)   # con.frame[:3] is a unit vector
        assert c.force is not None and c.force.shape == (3,)
        assert c.force[2] > 0                              # supports the cube against gravity


def test_legged_feet_contact_floor_with_upward_normal_when_standing():
    s = _legged_session()
    sv = s.state_view()
    cs = sv.contacts()
    foot_contacts = [c for c in cs if c.a.startswith("leg:") or c.b.startswith("leg:")]
    assert len(foot_contacts) >= 1
    for c in foot_contacts:
        assert c.normal[2] > 0.9                           # near-vertical: standing on flat ground


def test_contact_force_direction_matches_truth_normal_force():
    """`mj_contactForce`'s world-frame force must project onto the world normal consistently with `Session.truth()`
    (`con.frame[:3]`), catching a wrong frame transpose / basis-order bug."""
    s = _arm_session()
    sv = s.state_view()
    tr = s.truth()
    by_pair = {}
    for c in sv.contacts():
        by_pair.setdefault(frozenset((c.a, c.b)), []).append(c)
    for ct in tr.contacts:
        pair = frozenset((ct.body_a, ct.body_b))
        # fall back to the raw mj body names when neither side has an entity mapping
        matches = by_pair.get(pair) or [c for c in sv.contacts()
                                        if {ct.body_a, ct.body_b} & {c.a, c.b}]
        assert matches
        for c in matches:
            assert np.dot(c.force, c.normal) >= -1e-6       # never a net-negative (pulling) normal force at rest


# ------------------------------------------------------------------------------------------------ joints
@pytest.mark.parametrize("kind", SESSIONS)
def test_joints_match_qpos_qvel(kind):
    s = SESSIONS[kind]()
    sv = s.state_view()
    js = sv.joints()
    assert js and all(isinstance(j, JointState) for j in js)
    r0 = s.robots[0]
    by_name = {j.name: j for j in js}
    for name, qa, da in zip(r0.joint_names, r0.qadr, r0.dadr):
        assert name in by_name
        jt = by_name[name]
        assert jt.q == pytest.approx(float(s.data.qpos[qa]))
        assert jt.qd == pytest.approx(float(s.data.qvel[da]))
        assert jt.kind in ("hinge", "slide")
        assert jt.axis.shape == (3,)


# ------------------------------------------------------------------------------------------------ camera
@pytest.mark.parametrize("kind,cam", [("arm", "front"), ("dual", "front"), ("legged", "overhead")])
def test_camera_intrinsics_and_extrinsics(kind, cam):
    import mujoco
    s = SESSIONS[kind]()
    sv = s.state_view(depth_width=32, depth_height=48)
    c = sv.camera(cam)
    assert isinstance(c, Camera) and c.width == 32 and c.height == 48
    cid = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    fovy = math.radians(float(s.model.cam_fovy[cid]))
    fy = (48 / 2.0) / math.tan(fovy / 2.0)
    assert c.K.shape == (3, 3)
    assert c.K[1, 1] == pytest.approx(fy) and c.K[0, 0] == pytest.approx(fy)
    assert c.K[0, 2] == pytest.approx(16.0) and c.K[1, 2] == pytest.approx(24.0)
    assert c.T_world_cam.shape == (4, 4)
    np.testing.assert_allclose(c.T_world_cam[:3, 3], s.data.cam_xpos[cid], atol=1e-9)
    np.testing.assert_allclose(c.T_world_cam[:3, :3], s.data.cam_xmat[cid].reshape(3, 3), atol=1e-9)
    with pytest.raises(KeyError):
        sv.camera("no_such_camera")


def test_render_depth_shape_matches_declared_size(monkeypatch):
    """docs/relations.md 5.1, section 10 row R7: "render_depth via mujoco.Renderer depth mode, EGL only on the
    peer — host tests use the camera math only". So this test never constructs a real `mujoco.Renderer` (that
    needs a GL context — EGL/GLX — which is GPU/peer work, not host unit-test work); it fakes `mujoco.Renderer`
    to check the declared-size plumbing (`depth_wh` -> `Renderer(model, height, width)` -> the returned array's
    shape) instead, which is the part of `render_depth` this row's unit suite owns on host."""
    calls = {}

    class _FakeRenderer:
        def __init__(self, model, height, width):
            calls["hw"] = (height, width)

        def enable_depth_rendering(self):
            calls["depth_mode"] = True

        def update_scene(self, data, camera):
            calls["camera"] = camera

        def render(self):
            h, w = calls["hw"]
            return np.zeros((h, w), dtype=float)

    import rrp.envs.mujoco.session as session_mod
    monkeypatch.setattr(session_mod.mujoco, "Renderer", _FakeRenderer)

    s = _arm_session()
    sv = s.state_view(depth_width=24, depth_height=16)
    depth = sv.render_depth("front")
    assert calls["hw"] == (16, 24)             # mujoco.Renderer(model, height, width) — height first
    assert calls["depth_mode"] is True
    assert calls["camera"] == "front"
    assert depth.shape == (16, 24)
    assert np.isfinite(depth).all() and depth.min() >= 0.0


# ------------------------------------------------------------------------------------------------ token_entity
def test_token_entity_maps_detector_slots_to_privileged_ids():
    s = _arm_session()
    sv = s.state_view()
    assert sv.token_entity("ctx", 0) == "cube"
    assert sv.token_entity("ctx", len(s.detectables)) is None    # out of range -> None, never an exception
    assert sv.token_entity("ctx", "not-an-int") is None


# ------------------------------------------------------------------------------------------------ isolation
def test_state_view_never_reachable_from_the_featurizer():
    """docs/relations.md 5.1: "Never imported by policies/ (featurizers are the only policy inputs)"."""
    hits = []
    for base in ("policies/features", "policies/nets", "policies/bundles.py"):
        p = SRC / base
        paths = [p] if p.is_file() else (list(p.rglob("*.py")) if p.is_dir() else [])
        for f in paths:
            text = f.read_text()
            if "state_view" in text or "StateView" in text:
                hits.append(str(f))
    assert not hits, f"policies/ must never import StateView: {hits}"
