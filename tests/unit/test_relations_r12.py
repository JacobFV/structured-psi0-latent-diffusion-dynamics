"""R12 (D-144, docs/relations.md section 10): arm/dual token-set fields produced in the collate path from EXISTING
token columns (no dataset rewrite) -- `pos3d` (+`.var`), `cam_uvd`, `orient`, `entity_id`, `assembly_id` -- and the
MuJoCo camera projection helper they use. `feat.base_axes` replaces the `$RRP_KINFEAT` ablation flag at the
featurizer level (see `archived research/tracks/rel-r12.md` for what stays outside this unit's owned files).

Red/green: every `test_*_missing_*` / `test_*_without_*` below fails on the pre-R12 code (no such field / helper
existed) and passes after; the rest pin the VALUES against hand-computed / independently-derived expectations.
"""
import math

import numpy as np
import pytest
import torch

from rrp.envs.mujoco.fixtures import make_pick_place_session
from rrp.policies.features.featurizer import featurizer_for, REL, NODE_ANCHOR_SLICE, STATIC_DIM
from rrp.policies.features.multi import MultiFeaturizer
from rrp.policies.nets.batch import (collate_inputs, ctx_geometry_fields, ctx_id_fields, act_assembly_id,
                                     attach_cam_uvd, relation_token_sets)
from rrp.policies.relations.base import FIELDS
import rrp.policies.relations.catalog  # noqa: F401  (registers FIELDS on import)
from rrp.envs.mujoco.sensors import project_points, camera_frame


def _arm(seed=3):
    return make_pick_place_session(seed=seed)


def _batch(seed=3):
    s = _arm(seed)
    f = featurizer_for(s)
    obs = s.observe()
    pi = f(obs)
    return s, f, pi, obs, collate_inputs([pi])


# ------------------------------------------------------------------ camera projection (envs.mujoco.sensors)
def test_project_points_matches_mujoco_camera_math():
    s = _arm()
    m, d = s.model, s.data
    cpos, cmat, fovy = camera_frame(m, d, "front")
    # a point straight down the principal ray at distance 1.5 projects to the image center with depth 1.5
    center = cpos - 1.5 * cmat[:, 2]              # camera looks along -Z of its own frame
    u, v, depth = project_points(m, d, "front", center)[0]
    assert depth == pytest.approx(1.5, abs=1e-5)
    assert u == pytest.approx(0.0, abs=1e-5) and v == pytest.approx(0.0, abs=1e-5)
    # a point on the edge of the vertical FOV at the same depth projects to v = +-1 (definition of fovy)
    half = 1.5 * math.tan(fovy / 2.0)
    top = center + half * cmat[:, 1]
    _, v_top, depth_top = project_points(m, d, "front", top)[0]
    assert depth_top == pytest.approx(1.5, abs=1e-3)
    assert v_top == pytest.approx(1.0, abs=1e-3)
    # doubling the distance along the ray halves (u, v) at a fixed lateral offset, and doubles depth (pinhole)
    off = center + 0.1 * cmat[:, 0]                # same lateral offset, still at depth 1.5
    far = cpos - 3.0 * cmat[:, 2] + 0.1 * cmat[:, 0]        # same lateral offset, at depth 3.0
    u_near, _, d_near = project_points(m, d, "front", off)[0]
    u_far, _, d_far = project_points(m, d, "front", far)[0]
    assert d_far == pytest.approx(2 * d_near, rel=1e-6)
    assert u_far == pytest.approx(u_near / 2, rel=1e-3)


def test_project_points_behind_camera_is_flagged_not_crashed():
    s = _arm()
    cpos, cmat, _ = camera_frame(s.model, s.data, "front")
    behind = cpos + 1.0 * cmat[:, 2]               # +Z of the camera frame = behind it
    u, v, depth = project_points(s.model, s.data, "front", behind)[0]
    assert depth < 0
    assert u == 0.0 and v == 0.0                  # gated, not garbage (matches attach_cam_uvd's .valid)


def test_project_points_batches():
    s = _arm()
    cpos, cmat, _ = camera_frame(s.model, s.data, "front")
    pts = np.stack([cpos - 1.0 * cmat[:, 2], cpos - 2.0 * cmat[:, 2]])
    out = project_points(s.model, s.data, "front", pts)
    assert out.shape == (2, 3)
    assert out[1, 2] == pytest.approx(2.0, abs=1e-5)


# ------------------------------------------------------------------ pos3d / orient (nets.batch.ctx_geometry_fields)
def test_pos3d_assembly_matches_public_fk():
    s, f, pi, obs, b = _batch()
    fields = ctx_geometry_fields(b)
    off = b.bank_offset["morph"]
    mk = pi.token_kind["morph"]
    asm_idx = np.where(mk == 2)[0]
    assert len(asm_idx) == len(f.spec.assemblies)
    import mujoco
    for a, j in zip(f.spec.assemblies, asm_idx):   # assembly tokens are appended in `f.spec.assemblies` order
        sid = mujoco.mj_name2id(f.model, mujoco.mjtObj.mjOBJ_SITE, a.frame.site)
        expected = f._to_base(f.data.site_xpos[sid])
        got = fields["pos3d"][0, off + j].numpy()
        np.testing.assert_allclose(got, expected, atol=1e-5)
        assert bool(fields["pos3d.valid"][0, off + j])


def test_pos3d_action_node_matches_joint_anchor_token_column():
    s, f, pi, obs, b = _batch()
    fields = ctx_geometry_fields(b)
    off = b.bank_offset["morph"]
    N = len(f.node_joint_names)
    for i in range(N):
        expected = pi.tokens["morph"][i, NODE_ANCHOR_SLICE]
        np.testing.assert_allclose(fields["pos3d"][0, off + i].numpy(), expected, atol=1e-6)
        assert bool(fields["pos3d.valid"][0, off + i])


def test_pos3d_scene_valid_only_when_known_and_var_recovers_cov():
    s, f, pi, obs, b = _batch()
    fields = ctx_geometry_fields(b)
    off = b.bank_offset["scene"]
    sk = pi.token_kind["scene"]
    known = (sk == 0) & (pi.tokens["scene"][:, 7] > 0.5)
    for j in range(len(sk)):
        v = bool(fields["pos3d.valid"][0, off + j])
        assert v == bool(known[j])
    # object_descriptors carry the true position_cov_diag the tracker used to build the token; .var recovers it
    for od in [o for o in obs.object_descriptors if o.position_cov_diag is not None]:
        j = od.slot
        got_var = fields["pos3d.var"][0, off + j].numpy()
        np.testing.assert_allclose(got_var, np.asarray(od.position_cov_diag, np.float32), rtol=0.05, atol=1e-8)


def test_orient_only_valid_on_assembly_tokens_and_is_right_handed():
    s, f, pi, obs, b = _batch()
    fields = ctx_geometry_fields(b)
    off = b.bank_offset["morph"]
    mk = pi.token_kind["morph"]
    for j, k in enumerate(mk):
        v = bool(fields["orient.valid"][0, off + j])
        assert v == (k == 2)
        if v:
            R = fields["orient"][0, off + j].numpy().reshape(3, 3)   # columns x, y, z
            x, y, z = R[:, 0], R[:, 1], R[:, 2]
            np.testing.assert_allclose(np.cross(x, y), z, atol=1e-4)         # right-handed
            np.testing.assert_allclose(np.linalg.norm(R, axis=0), [1, 1, 1], atol=1e-4)  # orthonormal columns


def test_orient_dim_matches_catalog_field_def():
    assert FIELDS["orient"].dim == 9
    assert FIELDS["pos3d"].dim == 3
    assert FIELDS["cam_uvd"].dim == 3
    assert FIELDS["entity_id"].dim == 1
    assert FIELDS["assembly_id"].dim == 1


# ------------------------------------------------------------------ entity_id / assembly_id (nets.batch.ctx_id_fields)
def test_entity_id_defaults_to_self_and_aliases_pointer_targets():
    s, f, pi, obs, b = _batch()
    fields = ctx_id_fields(b)
    C = b.ctx_mask.shape[1]
    ent = fields["entity_id"][0, :, 0].numpy()
    valid = fields["entity_id.valid"][0].numpy()
    assert valid[b.ctx_mask[0].numpy()].all()
    offs = b.bank_offset
    from rrp.policies.features.featurizer import BANKS
    P = np.asarray(pi.pointers).reshape(-1, 4)
    s_flats = np.array([offs[BANKS[sb]] + si for sb, si, _, _ in P])
    d_flats = np.array([offs[BANKS[db]] + di for _, _, db, di in P])
    uniq, counts = np.unique(s_flats, return_counts=True)
    single_sources = set(uniq[counts == 1].tolist())
    fanout_sources = set(uniq[counts > 1].tolist())
    assert single_sources and fanout_sources   # the fixture exercises both (pred_arg fans out; role_points_to doesn't)
    for s_flat, d_flat in zip(s_flats, d_flats):
        if s_flat in single_sources:
            assert ent[s_flat] == d_flat
        else:                                  # a multi-argument predicate token is not any one entity's alias
            assert ent[s_flat] == s_flat
    # a token that is nobody's (single-destination) pointer source keeps its own index (self-identifying)
    untouched = [i for i in range(C) if b.ctx_mask[0, i] and i not in single_sources]
    assert untouched  # the fixture has at least one such token (e.g. an assembly token)
    for i in untouched:
        assert ent[i] == i


def test_assembly_id_self_ids_assembly_tokens_and_reaches_interact_sensors():
    s, f, pi, obs, b = _batch()
    fields = ctx_id_fields(b)
    off = b.bank_offset["morph"]
    mk = pi.token_kind["morph"]
    asm_positions = off + np.where(mk == 2)[0]
    asm = fields["assembly_id"][0, :, 0].numpy()
    for p in asm_positions:
        assert asm[p] == p
        assert bool(fields["assembly_id.valid"][0, p])
    # every interact "sensor" token (ikind 1) mounted on the grasping assembly resolves to a real assembly token
    ioff = b.bank_offset["interact"]
    ik = pi.token_kind["interact"]
    for j in np.where(ik == 1)[0]:
        idx = ioff + j
        if fields["assembly_id.valid"][0, idx]:
            assert int(asm[idx]) in asm_positions.tolist()


def test_act_assembly_id_matches_ctx_assembly_tokens():
    s, f, pi, obs, b = _batch()
    out = act_assembly_id(b)
    ctx = ctx_id_fields(b)
    off = b.bank_offset["morph"]
    mk = pi.token_kind["morph"]
    asm_positions = set((off + np.where(mk == 2)[0]).tolist())
    N = len(f.node_joint_names)
    for i in range(N):
        if out["assembly_id.valid"][0, i]:
            idx = int(out["assembly_id"][0, i, 0])
            assert idx in asm_positions
            assert ctx["assembly_id"][0, idx, 0] == idx   # agrees with the assembly token's own self id


def test_id_fields_missing_before_r12():
    """Red on the pre-R12 collate path: no id-field builder existed at all."""
    import rrp.policies.nets.batch as batch_mod
    assert hasattr(batch_mod, "ctx_id_fields")
    assert hasattr(batch_mod, "act_assembly_id")


# ------------------------------------------------------------------ cam_uvd (attach_cam_uvd)
def test_attach_cam_uvd_matches_direct_projection():
    s, f, pi, obs, b = _batch()
    geo = ctx_geometry_fields(b)
    out = attach_cam_uvd(geo["pos3d"], geo["pos3d.valid"], [(s.model, s.data, "front")])
    off = b.bank_offset["morph"]
    mk = pi.token_kind["morph"]
    j = int(np.where(mk == 2)[0][0])
    pos = geo["pos3d"][0, off + j].numpy()
    expected = project_points(s.model, s.data, "front", pos)[0]
    np.testing.assert_allclose(out["cam_uvd"][0, off + j].numpy(), expected, atol=1e-5)
    assert bool(out["cam_uvd.valid"][0, off + j]) == (expected[2] > 1e-9)


def test_relation_token_sets_shapes_and_masks():
    s, f, pi, obs, b = _batch()
    g = ctx_geometry_fields(b)
    b.extra["ctx_fields"] = attach_cam_uvd(g["pos3d"], g["pos3d.valid"], [(s.model, s.data, "front")])
    sets = relation_token_sets("arm", b)
    assert set(sets) == {"ctx", "act"}
    ctx = sets["ctx"]
    C = b.ctx_mask.shape[1]
    assert torch.equal(ctx.mask, b.ctx_mask)
    for name, dim in (("pos3d", 3), ("cam_uvd", 3), ("orient", 9), ("entity_id", 1), ("assembly_id", 1)):
        assert ctx.field(name).shape == (1, C, dim)
        assert ctx.field(f"{name}.valid").shape == (1, C)
    assert ctx.field("pos3d.var").shape == (1, C, 3)
    act = sets["act"]
    assert act.field("assembly_id").shape == (1, b.node_mask.shape[1], 1)


# ------------------------------------------------------------------ dual / multi featurizer reuses the same layout
def test_ctx_fields_on_dual_featurizer():
    from rrp.bodies.catalog import workbench_robots
    from rrp.envs.mujoco.dual import DualSession
    from rrp.envs.mujoco.dual_scenarios import build_support_insert
    W = workbench_robots()
    d = DualSession(build_support_insert([W["parm5l_pg2"](), W["parm6_pg2"]()], 3), seed=3)
    mf = MultiFeaturizer(d.model, d.scenario.robots)
    pi = mf(d.observe())
    b = collate_inputs([pi])
    fields = ctx_geometry_fields(b)
    assert fields["pos3d.valid"][0].any()
    ids = ctx_id_fields(b)
    assert ids["assembly_id.valid"][0].any()


# ------------------------------------------------------------------ feat.base_axes ($RRP_KINFEAT is gone from
# src/ entirely as of R12c/D-144 addendum: no live code reads os.environ for this any more; the ambient value a
# pipeline stage set via kinfeat.set_base_axes replaces it, and an explicit kwarg always wins over the ambient one)
@pytest.fixture(autouse=True)
def _reset_kinfeat_ambient():
    from rrp.policies.features import kinfeat
    prev = kinfeat.set_base_axes(None)
    yield
    kinfeat.set_base_axes(prev)


def test_base_axes_true_matches_ambient_default_on(monkeypatch):
    from rrp.policies.features import kinfeat
    s = _arm()
    kinfeat.set_base_axes(True)
    ambient = featurizer_for(s)
    kinfeat.set_base_axes(None)
    explicit = featurizer_for(s, base_axes=True)
    np.testing.assert_array_equal(ambient.node_static, explicit.node_static)
    assert explicit.kinfeat is True


def test_base_axes_false_ignores_ambient(monkeypatch):
    from rrp.policies.features import kinfeat
    kinfeat.set_base_axes(True)
    s = _arm()
    off = featurizer_for(s, base_axes=False)
    assert off.kinfeat is False
    kinfeat.set_base_axes(None)
    default = featurizer_for(s)
    np.testing.assert_array_equal(off.node_static, default.node_static)


def test_base_axes_none_is_byte_identical_to_pre_r12(monkeypatch):
    """Explicit control absent (None) must reproduce exactly today's default (no ambient set): goldens unchanged."""
    s = _arm()
    a = featurizer_for(s)
    b = featurizer_for(s, base_axes=None)
    np.testing.assert_array_equal(a.node_static, b.node_static)
    assert a.kinfeat is False and b.kinfeat is False
