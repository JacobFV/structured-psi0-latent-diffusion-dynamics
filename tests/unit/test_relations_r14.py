"""R14 (D-144, docs/relations.md section 10): geometry labels (`pos3d`, `cam_uvd`, `orient`, `contact_normal`) and
scene parts (`table_objects`, `camera_depth`) written once against `rrp.envs.base.StateView` /
`rrp.harness.data.relgen`.

Red/green: every test here failed with `ModuleNotFoundError: rrp.harness.data.relgen.geometry` before this unit's
module existed (confirmed by moving the module aside and rerunning, see `research/tracks/rel-r14.md`); the value
tests pin labels against hand-computed / independently-derived expectations on real arm/dual MuJoCo fixtures and on
a minimal fake `StateView` (for cases -- multi-contact averaging, occlusion, out-of-frame -- a real scene cannot
control precisely).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pytest

from rrp.envs.base import Camera, ContactState, EntityState
from rrp.envs.mujoco.fixtures import make_pick_place_session
from rrp.envs.mujoco.sensors import project_points
from rrp.harness.data.relgen import LABELS, PARTS, SceneDraft, TokenIndex
import rrp.harness.data.relgen.geometry as geometry


# ------------------------------------------------------------------------------------------------ fixtures
def _arm_session(seed=3):
    return make_pick_place_session(seed=seed)


def _dual_session(seed=3):
    from rrp.bodies.variants import registered_variants
    from rrp.envs.mujoco.dual import DualSession
    from rrp.envs.mujoco.dual_scenarios import build_support_insert
    W = registered_variants()
    return DualSession(build_support_insert([W["parm5l_pg2"](), W["parm6_pg2"]()], 3), seed=3)


class _NoDepthRenderView:
    """Wraps a real `StateView`, hiding the "depth_render" cap so `cam_uvd`'s analytic-depth branch runs even on a
    real MuJoCo session without ever constructing a `mujoco.Renderer` / GL context on host (docs/relations.md 10
    row R7: "EGL only on the peer -- host tests use the camera math only")."""

    def __init__(self, inner):
        self._inner = inner

    @property
    def caps(self):
        return self._inner.caps - {"depth_render"}

    def __getattr__(self, name):
        return getattr(self._inner, name)


@dataclass
class _FakeStateView:
    """Minimal `StateView` for cases a real scene cannot control precisely (multi-contact averaging, points behind
    / outside a camera's frame, missing capabilities)."""
    caps: frozenset
    entities_: list = field(default_factory=list)
    contacts_: list = field(default_factory=list)
    camera_: Camera | None = None
    depth_img: np.ndarray | None = None
    time: float = 0.0
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))

    def entities(self):
        return self.entities_

    def contacts(self):
        return self.contacts_

    def joints(self):
        return []

    def camera(self, name):
        if self.camera_ is None:
            raise KeyError(name)
        return self.camera_

    def render_depth(self, name):
        return self.depth_img

    def ui_tree(self):
        return []

    def token_entity(self, token_set, slot):
        return None


# ------------------------------------------------------------------------------------------------ registration
def test_labels_registered_with_correct_arity_and_needs():
    assert LABELS["pos3d"].arity == 1 and LABELS["pos3d"].needs == frozenset({"poses"})
    assert LABELS["orient"].arity == 1 and LABELS["orient"].needs == frozenset({"poses"})
    assert LABELS["contact_normal"].arity == 1 and LABELS["contact_normal"].needs == frozenset({"contacts"})
    assert LABELS["cam_uvd"].arity == 1 and LABELS["cam_uvd"].needs == frozenset({"poses", "camera"})
    for name in ("pos3d", "orient", "contact_normal", "cam_uvd"):
        assert LABELS[name].prov == "gt"


def test_parts_registered_with_declared_activates_and_vary():
    assert PARTS["table_objects"].activates == frozenset({"orient", "above"})
    assert PARTS["table_objects"].vary is not None
    assert PARTS["camera_depth"].activates == frozenset({"depth"})
    assert PARTS["camera_depth"].vary is not None


# ------------------------------------------------------------------------------------------------ pos3d
def test_pos3d_label_matches_entity_positions_on_real_arm_and_dual_fixtures():
    for make in (_arm_session, _dual_session):
        s = make()
        sv = s.state_view()
        ents = {e.id: e for e in sv.entities()}
        ids = list(ents)[:3] + [None]                      # exercise the null-identity slot too
        idx = TokenIndex(sets={"ctx": ids})
        lab = LABELS["pos3d"].fn(sv, idx)
        for i, eid in enumerate(ids[:-1]):
            assert lab.valid[i]
            np.testing.assert_allclose(lab.value[i], ents[eid].pos, atol=1e-9)
        assert not lab.valid[-1] and np.allclose(lab.value[-1], 0.0)


def test_pos3d_label_flattens_multiple_token_sets_in_order():
    fake = _FakeStateView(caps=frozenset({"poses"}),
                          entities_=[EntityState(id="a", kind="object", name="a", pos=np.array([1.0, 2.0, 3.0])),
                                     EntityState(id="b", kind="object", name="b", pos=np.array([4.0, 5.0, 6.0]))])
    idx = TokenIndex(sets={"ctx": ["a", None], "act": ["b"]})
    lab = LABELS["pos3d"].fn(fake, idx)
    assert lab.value.shape == (3, 3)
    np.testing.assert_allclose(lab.value[0], [1.0, 2.0, 3.0])
    assert not lab.valid[1]
    np.testing.assert_allclose(lab.value[2], [4.0, 5.0, 6.0])
    assert list(lab.valid) == [True, False, True]


# ------------------------------------------------------------------------------------------------ orient
def test_orient_label_matches_mujoco_xmat_on_the_cube():
    import mujoco
    s = _arm_session()
    sv = s.state_view()
    bid = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    idx = TokenIndex(sets={"ctx": ["cube"]})
    lab = LABELS["orient"].fn(sv, idx)
    assert lab.valid[0]
    np.testing.assert_allclose(lab.value[0].reshape(3, 3), s.data.xmat[bid].reshape(3, 3), atol=1e-6)


def test_orient_label_invalid_when_entity_has_no_quat():
    fake = _FakeStateView(caps=frozenset({"poses"}),
                          entities_=[EntityState(id="a", kind="feature", name="a", pos=np.zeros(3), quat=None)])
    lab = LABELS["orient"].fn(fake, TokenIndex(sets={"ctx": ["a", "missing"]}))
    assert not lab.valid[0] and not lab.valid[1]


def test_quat_to_mat_identity_and_90deg_z():
    np.testing.assert_allclose(geometry._quat_to_mat(np.array([1.0, 0.0, 0.0, 0.0])), np.eye(3), atol=1e-9)
    # 90 degree rotation about z: wxyz = (cos45, 0, 0, sin45)
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    R = geometry._quat_to_mat(np.array([c, 0.0, 0.0, s]))
    np.testing.assert_allclose(R @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-9)


# ------------------------------------------------------------------------------------------------ contact_normal
def test_contact_normal_averages_and_renormalizes_multiple_contacts():
    n1, n2 = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    fake = _FakeStateView(caps=frozenset({"contacts"}),
                          contacts_=[ContactState(a="x", b="y", pos=np.zeros(3), normal=n1),
                                     ContactState(a="x", b="z", pos=np.zeros(3), normal=n2)])
    idx = TokenIndex(sets={"ctx": ["x", "y", "untouched"]})
    lab = LABELS["contact_normal"].fn(fake, idx)
    expected = (n1 + n2) / np.linalg.norm(n1 + n2)
    np.testing.assert_allclose(lab.value[0], expected, atol=1e-9)          # x is the "a" side of both contacts
    assert lab.valid[0]
    np.testing.assert_allclose(lab.value[1], -n1, atol=1e-9)               # y is the "b" side: outward is -n1
    assert lab.valid[1]
    assert not lab.valid[2]
    assert np.allclose(lab.value[2], 0.0)


def test_contact_normal_on_real_arm_fixture_is_unit_and_only_touching_entities_valid():
    s = _arm_session()
    sv = s.state_view()
    touching = {c.a for c in sv.contacts()} | {c.b for c in sv.contacts()}
    assert "cube" in touching                     # cube rests on the table at reset
    idx = TokenIndex(sets={"ctx": ["cube", "target"]})   # "target" is a non-colliding visual zone: never touches
    lab = LABELS["contact_normal"].fn(sv, idx)
    assert lab.valid[0]
    assert math.isclose(float(np.linalg.norm(lab.value[0])), 1.0, abs_tol=1e-6)
    assert not lab.valid[1]


# ------------------------------------------------------------------------------------------------ cam_uvd
def test_cam_uvd_matches_sensors_project_points_on_the_real_front_camera():
    s = _arm_session()
    sv = s.state_view(depth_width=64, depth_height=64)
    ents = {e.id: e for e in sv.entities()}
    idx = TokenIndex(sets={"ctx": ["cube", "target"]})
    # hide "depth_render" so the label's ANALYTIC branch runs, matching `project_points`'s straight-line depth
    # (the render branch -- a different, occlusion-aware depth -- is covered host-safely with a fake renderer in
    # `test_cam_uvd_uses_render_depth_at_the_projected_pixel_when_depth_render_declared`)
    lab = LABELS["cam_uvd"].fn(_NoDepthRenderView(sv), idx)
    for i, eid in enumerate(("cube", "target")):
        expected = project_points(s.model, s.data, "front", ents[eid].pos[None, :])[0]
        assert lab.valid[i]
        np.testing.assert_allclose(lab.value[i], expected, atol=1e-5)


def test_cam_uvd_uses_render_depth_at_the_projected_pixel_when_depth_render_declared(monkeypatch):
    cam = geometry.front_camera()
    entity_pos = geometry.unproject(cam, 0.1, -0.2, 0.9)
    depth_img = np.full((cam.height, cam.width), 5.0)     # every pixel reports a fake depth of 5.0 m
    fake = _FakeStateView(caps=frozenset({"poses", "camera", "depth_render"}),
                          entities_=[EntityState(id="e", kind="object", name="e", pos=entity_pos)],
                          camera_=cam, depth_img=depth_img)
    lab = LABELS["cam_uvd"].fn(fake, TokenIndex(sets={"ctx": ["e"]}))
    assert lab.valid[0]
    np.testing.assert_allclose(lab.value[0][:2], [0.1, -0.2], atol=1e-6)
    assert lab.value[0][2] == pytest.approx(5.0)          # depth comes from the render, not the analytic 0.9


def test_cam_uvd_point_behind_camera_is_invalid():
    cam = geometry.front_camera()
    behind = cam.T_world_cam[:3, 3] + cam.T_world_cam[:3, :3] @ np.array([0.0, 0.0, 1.0])   # +Z local = behind
    fake = _FakeStateView(caps=frozenset({"poses", "camera"}),
                          entities_=[EntityState(id="e", kind="object", name="e", pos=behind)], camera_=cam)
    lab = LABELS["cam_uvd"].fn(fake, TokenIndex(sets={"ctx": ["e"]}))
    assert not lab.valid[0]


def test_cam_uvd_without_camera_capability_is_all_invalid():
    fake = _FakeStateView(caps=frozenset({"poses"}),
                          entities_=[EntityState(id="e", kind="object", name="e", pos=np.array([1.0, 0.0, 0.0]))])
    lab = LABELS["cam_uvd"].fn(fake, TokenIndex(sets={"ctx": ["e"]}))
    assert not lab.valid[0]


def test_front_camera_matches_real_arm_session_camera():
    s = _arm_session()
    sv = s.state_view(depth_width=64, depth_height=64)
    real = sv.camera("front")
    fake = geometry.front_camera()
    np.testing.assert_allclose(fake.K, real.K, atol=1e-6)
    np.testing.assert_allclose(fake.T_world_cam, real.T_world_cam, atol=1e-6)


def test_project_unproject_round_trip():
    cam = geometry.front_camera()
    rng = np.random.default_rng(0)
    for _ in range(20):
        u, v, d = rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), rng.uniform(0.2, 2.0)
        p = geometry.unproject(cam, u, v, d)
        u2, v2, d2 = geometry.project(cam, p[None, :])[0]
        assert (u, v, d) == pytest.approx((u2, v2, d2), abs=1e-8)


# ------------------------------------------------------------------------------------------------ table_objects
def test_table_objects_build_places_n_objects_in_range_with_provenance():
    draft = SceneDraft(env="mujoco/arm", kwargs={"n_objects": 4, "x_range": (0.2, 0.5), "y_range": (-0.3, 0.3)})
    rng = np.random.default_rng(0)
    PARTS["table_objects"].build(draft, rng)
    assert len(draft.entities) == 4
    for e in draft.entities:
        assert e["kind"] == "object"
        assert 0.2 <= e["pos"][0] <= 0.5 and -0.3 <= e["pos"][1] <= 0.3
        assert -math.pi <= e["yaw"] <= math.pi
    assert draft.active == frozenset({"orient", "above"})
    assert draft.parts == ("table_objects",)
    assert draft.provenance["parts"] == [{"part": "table_objects", "version": "1"}]


def test_table_objects_vary_changes_only_yaw_and_is_seed_deterministic():
    draft = SceneDraft(env="mujoco/arm", kwargs={"n_objects": 3})
    PARTS["table_objects"].build(draft, np.random.default_rng(1))
    before = [dict(e) for e in draft.entities]
    out1 = PARTS["table_objects"].vary(draft, np.random.default_rng(2), "orient", k=2)
    out2 = PARTS["table_objects"].vary(draft, np.random.default_rng(2), "orient", k=2)
    assert len(out1) == 2
    for varied in out1:
        assert len(varied.entities) == len(before)
        for e0, e1 in zip(before, varied.entities):
            np.testing.assert_allclose(e0["pos"], e1["pos"])
            assert e0["id"] == e1["id"]
        assert any(e0["yaw"] != e1["yaw"] for e0, e1 in zip(before, varied.entities))
        assert varied.provenance["vary"] == {"part": "table_objects", "factor": "orient"}
    # original draft is untouched
    for e0, e1 in zip(before, draft.entities):
        assert e0["yaw"] == e1["yaw"]
    # reproducible under a fixed seed
    for a, b in zip(out1, out2):
        assert [e["yaw"] for e in a.entities] == [e["yaw"] for e in b.entities]


def test_table_objects_vary_rejects_unknown_factor():
    draft = SceneDraft(env="mujoco/arm")
    PARTS["table_objects"].build(draft, np.random.default_rng(0))
    with pytest.raises(ValueError):
        PARTS["table_objects"].vary(draft, np.random.default_rng(0), "depth")


# ------------------------------------------------------------------------------------------------ camera_depth
def test_camera_depth_build_places_probe_at_chosen_pixel_and_depth_with_provenance():
    draft = SceneDraft(env="mujoco/arm", kwargs={"pixel_u": 0.3, "pixel_v": -0.4, "depth": 0.8})
    PARTS["camera_depth"].build(draft, np.random.default_rng(0))
    probe = draft.entities[0]
    assert probe["id"] == "depth_probe" and probe["pixel"] == (0.3, -0.4) and probe["depth"] == 0.8
    cam = draft.kwargs["camera"]
    u, v, d = geometry.project(cam, probe["pos"][None, :])[0]
    assert (u, v, d) == pytest.approx((0.3, -0.4, 0.8), abs=1e-6)
    assert draft.active == frozenset({"depth"})
    assert draft.provenance["parts"] == [{"part": "camera_depth", "version": "1"}]


def test_camera_depth_vary_keeps_pixel_changes_depth():
    draft = SceneDraft(env="mujoco/arm", kwargs={"pixel_u": -0.2, "pixel_v": 0.15, "depth": 0.5})
    PARTS["camera_depth"].build(draft, np.random.default_rng(0))
    base_probe = draft.entities[0]
    cam = draft.kwargs["camera"]
    varied = PARTS["camera_depth"].vary(draft, np.random.default_rng(3), "depth", k=3)
    assert len(varied) == 3
    seen_depths = set()
    for nd in varied:
        probe = next(e for e in nd.entities if e["id"] == "depth_probe")
        u, v, d = geometry.project(cam, probe["pos"][None, :])[0]
        assert (u, v) == pytest.approx((-0.2, 0.15), abs=1e-6)     # same pixel
        assert d != pytest.approx(base_probe["depth"], abs=1e-3)   # different depth
        seen_depths.add(round(d, 6))
    assert len(seen_depths) > 1                                    # the k copies are not all the same depth either
    # base draft's own probe is untouched
    np.testing.assert_allclose(draft.entities[0]["pos"], base_probe["pos"])


def test_camera_depth_vary_without_build_raises():
    draft = SceneDraft(env="mujoco/arm")
    with pytest.raises(ValueError):
        PARTS["camera_depth"].vary(draft, np.random.default_rng(0), "depth")


def test_camera_depth_vary_rejects_unknown_factor():
    draft = SceneDraft(env="mujoco/arm")
    PARTS["camera_depth"].build(draft, np.random.default_rng(0))
    with pytest.raises(ValueError):
        PARTS["camera_depth"].vary(draft, np.random.default_rng(0), "orient")
