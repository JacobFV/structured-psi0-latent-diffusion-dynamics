"""Unit R19 (docs/relations.md section 10, row R19): legged labels + foothold. `leg.foothold` / `leg.com_support`
catalog entries, `harness/data/relgen/body.py` (labels `foot_contacts`, `foothold_next`, `com_support`; part
`terrain_steps`) and the Warp height scan's move out of the deployable `vec` (`envs/warp/task_env.py`
`_layout_dims`, the privileged-layout test)."""
from __future__ import annotations

import numpy as np
import pytest

from rrp.envs.base import ContactState, EntityState
from rrp.harness.data.relgen import LABELS, PARTS, SceneDraft, TokenIndex
from rrp.harness.data.relgen.body import (BODY_ID, FOOT_PREFIX, FOOTHOLD_CELL_KIND, SCAN_AHEAD, SCAN_SIDE,
                                          _NO_SUPPORT_MARGIN, com_support_fn, foot_contacts_fn, foot_ids,
                                          foothold_next_fn, stance_feet, support_polygon_margin, terrain_steps_build,
                                          terrain_steps_vary)
from rrp.policies.relations.base import FactorSpec, PrivilegedInput, assert_deployable, get_factor, resolve
from rrp.policies.relations.ops import OPS


# ------------------------------------------------------------------------------------------------ fixture
class _LeggedView:
    """A minimal `StateView`-shaped fixture (docs 5.1): a quadruped with feet `leg:fl/fr/bl/br`, body assembly
    entity `"body"`, `fl`/`fr`/`bl` in stance (ground contact) and `br` mid-swing (no contact)."""
    caps = frozenset({"poses", "contacts"})
    time = 0.0
    gravity = np.array([0.0, 0.0, -9.81])

    POS = {"leg:fl": (0.2, 0.15, 0.0), "leg:fr": (0.2, -0.15, 0.0), "leg:bl": (-0.2, 0.15, 0.0),
          "leg:br": (-0.2, -0.15, 0.05), "body": (0.0, 0.0, 0.3)}

    def __init__(self, extra_entities=(), stance=("leg:fl", "leg:fr", "leg:bl")):
        self._entities = [EntityState(id=i, kind="link" if i.startswith(FOOT_PREFIX) else "body", name=i,
                                      pos=np.array(p)) for i, p in self.POS.items()]
        self._entities += list(extra_entities)
        self._contacts = [ContactState(a=f, b="ground", pos=np.array(self.POS[f]), normal=np.array([0.0, 0.0, 1.0]))
                          for f in stance]

    def entities(self):
        return list(self._entities)

    def contacts(self):
        return list(self._contacts)

    def joints(self):
        raise NotImplementedError

    def camera(self, name):
        raise NotImplementedError

    def ui_tree(self):
        raise NotImplementedError

    def token_entity(self, token_set, slot):
        return None


IDS = ["leg:fl", "leg:fr", "leg:bl", "leg:br", "body", None]


# ------------------------------------------------------------------------------------------------ stance_feet / foot_ids
def test_stance_feet_reads_both_contact_endpoints():
    view = _LeggedView()
    assert stance_feet(view) == {"leg:fl", "leg:fr", "leg:bl"}


def test_foot_ids_filters_and_preserves_order():
    assert foot_ids(["leg:fl", "body", "leg:br", None, "leg:fr"]) == ["leg:fl", "leg:br", "leg:fr"]


# ------------------------------------------------------------------------------------------------ foot_contacts_fn
def test_foot_contacts_marks_stance_feet_and_masks_non_feet():
    lab = foot_contacts_fn(_LeggedView(), TokenIndex(sets={"ctx": IDS}))
    np.testing.assert_array_equal(lab.valid, [True, True, True, True, False, False])
    np.testing.assert_array_equal(lab.value[:4, 0], [1.0, 1.0, 1.0, 0.0])
    assert lab.value[4, 0] == 0.0 and lab.value[5, 0] == 0.0     # body / null slot: invalid, value untouched (0)


def test_foot_contacts_needs_ctx_token_set():
    with pytest.raises(KeyError):
        foot_contacts_fn(_LeggedView(), TokenIndex(sets={"other": IDS}))


# ------------------------------------------------------------------------------------------------ foothold_next_fn
def test_foothold_next_all_invalid_with_no_candidate_cells():
    lab = foothold_next_fn(_LeggedView(), TokenIndex(sets={"ctx": IDS}))
    assert lab.valid.sum() == 0
    assert lab.value.sum() == 0.0


def test_foothold_next_swing_foot_pairs_with_nearest_cell_and_stance_feet_never_true():
    cells = [
        EntityState(id="cell_near", kind=FOOTHOLD_CELL_KIND, name="cell_near", pos=np.array([-0.19, -0.14, 0.0])),
        EntityState(id="cell_far", kind=FOOTHOLD_CELL_KIND, name="cell_far", pos=np.array([1.0, 1.0, 0.0])),
    ]
    ids = IDS + ["cell_near", "cell_far"]
    lab = foothold_next_fn(_LeggedView(extra_entities=cells), TokenIndex(sets={"ctx": ids}))
    i_br, i_near, i_far = ids.index("leg:br"), ids.index("cell_near"), ids.index("cell_far")
    # every foot x cell pair is a valid (scored) slot ...
    for i_foot in (ids.index("leg:fl"), ids.index("leg:fr"), ids.index("leg:bl"), i_br):
        assert lab.valid[i_foot, i_near] and lab.valid[i_foot, i_far]
    # ... but only the swinging foot (leg:br) has a true pair, and it is the NEAREST cell
    assert lab.value[i_br, i_near, 0] == 1.0 and lab.value[i_br, i_far, 0] == 0.0
    for i_foot in (ids.index("leg:fl"), ids.index("leg:fr"), ids.index("leg:bl")):
        assert lab.value[i_foot].sum() == 0.0                       # planted feet: never a true "next foothold"
    # non-foot / non-cell rows and columns stay invalid
    assert not lab.valid[ids.index("body"), i_near]
    assert not lab.valid[i_br, ids.index("body")]


# ------------------------------------------------------------------------------------------------ support_polygon_margin
def test_support_polygon_margin_no_feet_is_negative_infinity():
    assert support_polygon_margin([0.0, 0.0], []) == float("-inf")


def test_support_polygon_margin_one_foot_is_negated_distance():
    assert support_polygon_margin([1.0, 0.0], [[0.0, 0.0]]) == pytest.approx(-1.0)


def test_support_polygon_margin_inside_triangle_is_positive():
    tri = [[-1.0, -1.0], [1.0, -1.0], [0.0, 1.0]]
    assert support_polygon_margin([0.0, -0.5], tri) > 0.0


def test_support_polygon_margin_outside_triangle_is_negative():
    tri = [[-1.0, -1.0], [1.0, -1.0], [0.0, 1.0]]
    assert support_polygon_margin([5.0, 5.0], tri) < 0.0


# ------------------------------------------------------------------------------------------------ com_support_fn
def test_com_support_written_only_on_body_slot():
    lab = com_support_fn(_LeggedView(), TokenIndex(sets={"ctx": IDS}))
    np.testing.assert_array_equal(lab.valid, [False, False, False, False, True, False])
    expected = support_polygon_margin([0.0, 0.0], [_LeggedView.POS[f][:2] for f in ("leg:fl", "leg:fr", "leg:bl")])
    assert lab.value[4, 0] == pytest.approx(expected)


def test_com_support_no_stance_feet_uses_fixed_negative_margin():
    lab = com_support_fn(_LeggedView(stance=()), TokenIndex(sets={"ctx": IDS}))
    assert lab.value[4, 0] == pytest.approx(_NO_SUPPORT_MARGIN)


def test_com_support_no_body_entity_is_all_invalid():
    view = _LeggedView()
    view._entities = [e for e in view._entities if e.id != BODY_ID]
    lab = com_support_fn(view, TokenIndex(sets={"ctx": IDS}))
    assert lab.valid.sum() == 0


# ------------------------------------------------------------------------------------------------ terrain_steps part
def test_terrain_steps_build_declares_grid_and_activates():
    draft = SceneDraft(env="mujoco/legged")
    terrain_steps_build(draft, np.random.default_rng(0))
    n_expected = len(SCAN_AHEAD) * len(SCAN_SIDE)
    assert len(draft.entities) == n_expected
    assert all(e["kind"] == FOOTHOLD_CELL_KIND for e in draft.entities)
    assert draft.active == {"terrain"}
    assert draft.parts == ("terrain_steps",)
    assert "terrain_steps" in draft.provenance and draft.provenance["terrain_steps"]["n"] == n_expected
    heights = draft.kwargs["terrain_steps"]["heights"]
    assert len(heights) == len(SCAN_AHEAD)
    assert all(0.0 <= h <= draft.kwargs["terrain_steps"]["h_max"] for h in heights)


def test_terrain_steps_vary_height_changes_only_z():
    draft = SceneDraft(env="mujoco/legged")
    terrain_steps_build(draft, np.random.default_rng(1))
    before = {e["id"]: tuple(e["pos"]) for e in draft.entities}
    [varied] = terrain_steps_vary(draft, np.random.default_rng(2), "height")
    after = {e["id"]: tuple(e["pos"]) for e in varied.entities}
    assert set(before) == set(after)
    for cid in before:
        bx, by, bz = before[cid]
        ax, ay, az = after[cid]
        assert (ax, ay) == (bx, by)                                  # xy layout / cell count / ids untouched
    assert any(after[cid][2] != before[cid][2] for cid in before)    # heights actually resampled
    assert varied.kwargs["terrain_steps"]["heights"] != draft.kwargs["terrain_steps"]["heights"]


def test_terrain_steps_vary_rejects_unknown_factor():
    draft = SceneDraft(env="mujoco/legged")
    terrain_steps_build(draft, np.random.default_rng(0))
    with pytest.raises(ValueError):
        terrain_steps_vary(draft, np.random.default_rng(0), "not_a_factor")


def test_terrain_steps_vary_requires_build_first():
    with pytest.raises(ValueError):
        terrain_steps_vary(SceneDraft(env="mujoco/legged"), np.random.default_rng(0), "height")


def test_labels_and_part_registered():
    assert {"foot_contacts", "foothold_next", "com_support"} <= set(LABELS)
    assert "terrain_steps" in PARTS
    assert PARTS["terrain_steps"].activates == frozenset({"terrain", "foothold", "com_support"})
    assert LABELS["foothold_next"].needs == frozenset({"poses", "contacts"})
    assert LABELS["com_support"].needs == frozenset({"poses", "contacts"})


# ------------------------------------------------------------------------------------------------ catalog.py: leg.*
def test_catalog_leg_foothold_is_bilinear_pair_over_terrain_steps():
    d = get_factor("leg.foothold")
    assert d.op == "bilinear" and d.form == "aug" and d.label == "foothold_next" and d.gen == ("terrain_steps",)
    assert d.sources[0] == "probe" and "gt" in d.sources
    assert d.readout is not None and d.readout.address == "pair" and d.readout.label == "foothold_next"
    assert d.form in OPS[d.op].forms


def test_catalog_leg_com_support_is_an_asm_readout():
    d = get_factor("leg.com_support")
    assert d.op == "inert" and d.form == "readout" and d.label == "com_support"
    assert d.readout is not None and d.readout.address == "asm" and d.readout.out == 1
    assert d.form in OPS[d.op].forms


def test_catalog_leg_factors_resolve_and_default_deploy_safe():
    specs = resolve(["leg.foothold", "leg.com_support"])
    assert {s.name for s in specs} == {"leg.foothold", "leg.com_support"}
    assert_deployable(specs)                                          # default sources: probe / given, both deployable


def test_catalog_leg_foothold_gt_source_is_blocked_in_deploy_mode():
    specs = resolve([FactorSpec(name="leg.foothold", source="gt")])
    with pytest.raises(PrivilegedInput):
        assert_deployable(specs)


def test_catalog_legged_r19_preset():
    specs = resolve(["preset:legged-r19"])
    assert {s.name for s in specs} == {"leg.foothold", "leg.com_support"}


# ------------------------------------------------------------------------------------------------ privileged-layout test (Warp)
# WarpTrackerEnv itself always needs CUDA (`self.dev` is hard-coded), so this exercises the pure-arithmetic split
# (`_layout_dims`) and a source-level check that `observe()` is no longer overridden, matching R8's "Warp on a
# 2-env CPU batch if available, else skip-marked" -- here there is nothing CUDA-shaped to skip: `_layout_dims` and
# class-attribute inspection need no simulator at all.
def test_layout_dims_widens_priv_not_obs():
    from rrp.envs.warp.task_env import _layout_dims
    obs_dim, priv_dim = _layout_dims(base_obs_dim=48, base_priv_dim=12, extra_dim=34)
    assert obs_dim == 48                                              # UNCHANGED by the extra (privileged) block
    assert priv_dim == 12 + 34                                        # extra block widens priv_dim instead


def test_warp_steps_and_gap_env_no_longer_override_observe():
    from rrp.envs.warp.task_env import WarpGapEnv, WarpStepsEnv
    for cls in (WarpStepsEnv, WarpGapEnv):
        assert "observe" not in cls.__dict__, f"{cls.__name__} must inherit WarpTrackerEnv.observe unchanged"
        assert "privileged" in cls.__dict__, f"{cls.__name__} must widen privileged() with its extra_obs()"


def test_warp_steps_extra_obs_width_matches_scan_grid():
    from rrp.envs.mujoco.humanoid_scenes import SCAN_X, SCAN_Y
    assert SCAN_X.size * SCAN_Y.size + 1 == 34                        # 11 x 3 height-scan cells + h_frac
