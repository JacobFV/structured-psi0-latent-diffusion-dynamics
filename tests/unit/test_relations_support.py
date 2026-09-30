"""Unit R17 (docs/relations.md section 10, row R17): support / force-flow factors and their data generation --
`ix.support` / `ix.force_flow` catalog entries, `relgen.support.support_matrix` / `support_closure` (labels
`support_pairs` / `support_closure`) and the `stack` scene part. Math tests against a hand-built stack fixture
(support pairs + upstream/downstream closure), `stack.vary` changing only the order, registry resolution and the
deploy guard (docs 7)."""
from __future__ import annotations

import numpy as np
import pytest

from rrp.envs.base import EntityState, ContactState
from rrp.harness.data.relgen import LABELS, PARTS, SceneDraft, TokenIndex
from rrp.harness.data.relgen.support import (SUPPORT_ANGLE_DEG, support_closure, support_closure_fn, support_matrix,
                                             support_pairs_fn)
from rrp.policies.relations.base import FactorSpec, PrivilegedInput, assert_deployable, get_factor, resolve
from rrp.policies.relations.ops import OPS


# ------------------------------------------------------------------------------------------------ fixture
class _StackView:
    """A minimal `StateView`-shaped fixture (docs 5.1): a 4-high chain `block_a -> block_b -> block_c -> block_e`
    (the last contact reports its `ContactState` with `a`/`b` reversed, on purpose, to exercise the "normal points
    down" branch of `support_matrix`), a side lean `block_d` against `block_b` that must NOT read as support, and a
    stray contact against `block_a` positioned below its own origin that must be rejected even though its normal
    passes the angle test."""
    caps = frozenset({"poses", "contacts"})
    time = 0.0
    gravity = np.array([0.0, 0.0, -9.81])

    def __init__(self):
        pos = {"block_a": (0.0, 0.0, 0.05), "block_b": (0.0, 0.0, 0.15), "block_c": (0.0, 0.0, 0.25),
              "block_d": (0.1, 0.0, 0.15), "block_e": (0.0, 0.0, 0.35), "block_f": (0.0, 0.0, -0.60)}
        self._entities = [EntityState(id=i, kind="object", name=i, pos=np.array(p)) for i, p in pos.items()]
        deg20 = np.deg2rad(20.0)
        self._contacts = [
            ContactState(a="block_a", b="block_b", pos=np.array([0.0, 0.0, 0.10]), normal=np.array([0.0, 0.0, 1.0])),
            ContactState(a="block_b", b="block_c", pos=np.array([0.0, 0.0, 0.20]),
                        normal=np.array([np.sin(deg20), 0.0, np.cos(deg20)])),          # ~20 deg off vertical: still support
            ContactState(a="block_d", b="block_b", pos=np.array([0.1, 0.0, 0.15]), normal=np.array([1.0, 0.0, 0.05])),
            ContactState(a="block_e", b="block_c", pos=np.array([0.0, 0.0, 0.30]), normal=np.array([0.0, 0.0, -1.0])),
            ContactState(a="block_a", b="block_f", pos=np.array([0.0, 0.0, -0.5]), normal=np.array([0.0, 0.0, 1.0])),
        ]

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


IDS = ["block_a", "block_b", "block_c", "block_d", "block_e", None]     # last slot: a masked / null token


# ------------------------------------------------------------------------------------------------ support_matrix
def test_support_matrix_direct_pairs_and_angle_threshold():
    m = support_matrix(_StackView())
    assert m["block_a"] == {"block_b": True}
    assert m["block_b"] == {"block_c": True}                            # ~20 deg contact still within 30 deg
    assert m["block_c"] == {"block_e": True}                            # reversed a/b endpoints, normal points down
    assert "block_d" not in m and all("block_d" not in row for row in m.values())   # side lean: not support
    assert all("block_f" not in row for row in m.values())              # stray contact below block_a: rejected


def test_support_matrix_rejects_contacts_past_the_angle_threshold():
    view = _StackView()
    view._contacts = [ContactState(a="x", b="y", pos=np.zeros(3), normal=np.array([1.0, 0.0, 0.0]))]  # pure sideways
    assert support_matrix(view) == {}


def test_support_matrix_ignores_degenerate_zero_normals():
    view = _StackView()
    view._contacts = [ContactState(a="x", b="y", pos=np.zeros(3), normal=np.zeros(3))]
    assert support_matrix(view) == {}


# ------------------------------------------------------------------------------------------------ closure (pure math)
def test_support_closure_upstream_downstream_correct():
    direct = {"block_a": {"block_b": True}, "block_b": {"block_c": True}, "block_c": {"block_e": True}}
    closure = support_closure(direct)
    assert closure["block_a"] == {"block_b": True, "block_c": True, "block_e": True}     # a's full downstream
    assert closure["block_b"] == {"block_c": True, "block_e": True}
    assert closure["block_c"] == {"block_e": True}
    assert "block_e" not in closure                                                      # e supports nothing
    upstream_of_e = {a for a, bs in closure.items() if "block_e" in bs}
    assert upstream_of_e == {"block_a", "block_b", "block_c"}


def test_support_closure_of_the_stack_fixture():
    closure = support_closure(support_matrix(_StackView()))
    assert closure["block_a"] == {"block_b": True, "block_c": True, "block_e": True}
    assert closure["block_b"] == {"block_c": True, "block_e": True}
    assert closure["block_c"] == {"block_e": True}
    assert "block_d" not in closure and "block_e" not in closure


def test_support_closure_terminates_on_a_cycle():
    direct = {"a": {"b": True}, "b": {"a": True}}
    closure = support_closure(direct)                    # must not hang; a 2-cycle reaches both from both
    assert closure["a"] == {"a": True, "b": True} and closure["b"] == {"a": True, "b": True}


def test_support_closure_disjoint_components_stay_disjoint():
    direct = {"a": {"b": True}, "x": {"y": True}}
    closure = support_closure(direct)
    assert closure["a"] == {"b": True} and closure["x"] == {"y": True}


# ------------------------------------------------------------------------------------------------ LabelDef end to end
def test_support_pairs_label_registered_and_correct_on_the_fixture():
    d = LABELS["support_pairs"]
    assert d.arity == 2 and d.needs == frozenset({"contacts", "poses"}) and d.fn is support_pairs_fn
    idx = TokenIndex(sets={"ctx": IDS})
    label = support_pairs_fn(_StackView(), idx)
    assert label.value.shape == (6, 6, 1) and label.valid.shape == (6, 6) and label.prov == "gt"
    ia, ib, ic, id_, ie = (IDS.index(n) for n in ("block_a", "block_b", "block_c", "block_d", "block_e"))
    assert label.value[ia, ib, 0] == 1.0 and label.value[ib, ic, 0] == 1.0 and label.value[ic, ie, 0] == 1.0
    assert label.value[ia, ic, 0] == 0.0                        # direct pairs only, no closure here
    assert label.value[id_, ib, 0] == 0.0 and label.value[ib, id_, 0] == 0.0
    assert not label.valid[:, 5].any() and not label.valid[5, :].any()      # the null slot is never valid
    assert not label.valid[ia, ia]                                          # no self-support


def test_support_closure_label_registered_and_matches_the_pure_closure():
    d = LABELS["support_closure"]
    assert d.arity == 2 and d.needs == frozenset({"contacts", "poses"}) and d.fn is support_closure_fn
    idx = TokenIndex(sets={"ctx": IDS})
    label = support_closure_fn(_StackView(), idx)
    ia, ic, ie = IDS.index("block_a"), IDS.index("block_c"), IDS.index("block_e")
    assert label.value[ia, ic, 0] == 1.0 and label.value[ia, ie, 0] == 1.0    # transitive, unlike support_pairs


# ------------------------------------------------------------------------------------------------ part `stack`
def test_stack_part_registered():
    p = PARTS["stack"]
    assert p.activates == frozenset({"support", "force_flow"})
    assert p.envs == ("mujoco/arm", "mujoco/dual") and p.vary is not None


def _built_draft(n=3, seed=0) -> SceneDraft:
    d = SceneDraft(env="mujoco/arm", kwargs={"stack_n": n})
    PARTS["stack"].build(d, np.random.default_rng(seed))
    return d


def test_stack_build_produces_n_objects_and_activates_its_dynamics():
    d = _built_draft(n=3)
    order = d.kwargs["stack"]["order"]
    assert len(order) == 3 == len(set(order)) == len(d.entities)
    assert all(e["kind"] == "object" and e["id"] in order for e in d.entities)
    assert d.active == frozenset({"support", "force_flow"}) and d.parts == ("stack",)
    assert set(d.kwargs["stack"]["objects"]) == set(order)


def test_stack_vary_changes_only_the_order():
    base = _built_draft(n=4, seed=1)
    variants = PARTS["stack"].vary(base, np.random.default_rng(2), "order")
    assert variants                                       # at least one permutation for n=4
    base_order = base.kwargs["stack"]["order"]
    for v in variants:
        vo = v.kwargs["stack"]["order"]
        assert sorted(vo) == sorted(base_order) and vo != base_order          # same objects, different order
        assert v.kwargs["stack"]["objects"] == base.kwargs["stack"]["objects"]     # every physical property unchanged
        assert {k: val for k, val in v.kwargs.items() if k != "stack"} == \
               {k: val for k, val in base.kwargs.items() if k != "stack"}
        assert v.entities == base.entities and v.events == base.events
        assert v.env == base.env and v.active == base.active
    orders = {tuple(v.kwargs["stack"]["order"]) for v in variants}
    assert len(orders) == len(variants)                    # no duplicate permutations


def test_stack_vary_is_seed_deterministic():
    b1, b2 = _built_draft(n=4, seed=1), _built_draft(n=4, seed=1)
    v1 = PARTS["stack"].vary(b1, np.random.default_rng(7), "order")
    v2 = PARTS["stack"].vary(b2, np.random.default_rng(7), "order")
    assert [v.kwargs["stack"]["order"] for v in v1] == [v.kwargs["stack"]["order"] for v in v2]


def test_stack_vary_noop_below_two_objects():
    d = _built_draft(n=1)
    assert PARTS["stack"].vary(d, np.random.default_rng(0), "order") == [d]


def test_stack_build_does_not_collide_ids_across_two_parts_in_one_draft():
    d = SceneDraft(env="mujoco/arm", kwargs={"stack_n": 2})
    PARTS["stack"].build(d, np.random.default_rng(0))
    PARTS["stack"].build(d, np.random.default_rng(1))
    ids = [e["id"] for e in d.entities]
    assert len(ids) == len(set(ids)) == 4 and d.parts == ("stack", "stack")


# ------------------------------------------------------------------------------------------------ catalog entries
def test_factors_registered_with_the_documented_shape():
    sup, flow = get_factor("ix.support"), get_factor("ix.force_flow")
    assert sup.op == "bilinear" and sup.form == "aug" and sup.label == "support_pairs" and sup.gen == ("stack",)
    assert sup.sources[0] == "probe" and sup.readout is not None and sup.readout.address == "pair"
    assert flow.op == "flow" and flow.form == "bias" and flow.label == "support_closure" and flow.gen == ("stack",)
    assert flow.sources[0] == "probe" and flow.algebra.transitive is True
    assert flow.form in OPS[flow.op].forms and sup.form in OPS[sup.op].forms


def test_factors_resolve_and_default_deploy_safe():
    specs = resolve(["ix.support", "ix.force_flow"])
    assert {s.name for s in specs} == {"ix.support", "ix.force_flow"}
    assert_deployable(specs)                                            # default source "probe": deployable


def test_factor_gt_source_is_blocked_in_deploy_mode():
    # bilinear ops don't accept `control="gt"` (BilinearOp.controls excludes it); `source="gt"` (control stays "on")
    # is the way to ask for the privileged source, mirroring "geo.pos3d source=gt" in docs/relations.md section 7.
    specs = resolve([FactorSpec(name="ix.support", source="gt")])
    with pytest.raises(PrivilegedInput):
        assert_deployable(specs)
    specs2 = resolve([FactorSpec(name="ix.force_flow", control="gt")])   # `flow` is a graph op: control="gt" IS allowed
    with pytest.raises(PrivilegedInput):
        assert_deployable(specs2)


def test_support_angle_threshold_is_30_degrees():
    assert SUPPORT_ANGLE_DEG == pytest.approx(30.0)
