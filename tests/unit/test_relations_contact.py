"""Unit R16 (docs/relations.md section 10): contact / grasp / handover. `catalog.py` §contact (`ix.contact`,
`ix.held_by`, `ix.handover`, all `hidden` / `bilinear` / `aug`) and `harness/data/relgen/contact.py` (labels
`contact_pairs`, `held_pairs`, `handover_pairs`; part `grasp_target`).

Label tests use a minimal fake `StateView` (a plain object satisfying the `runtime_checkable` Protocol by having the
right methods/attributes, docs/relations.md 5.1) with hand-picked entities and contact records, so every expected
label value below is hand-computed from the module docstring's rule, not derived from physics timing. A real-MuJoCo
smoke test at the end checks the labels also run against a live `Session.state_view()` without crashing.

Red/green: every test here fails on pre-R16 `main` (`ix.*` were unregistered globs / `contact.py` did not exist) and
passes after.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import rrp.policies.relations.catalog  # noqa: F401  (registers FIELDS/FACTORS on import)
from rrp.envs.base import ContactState, EntityState
from rrp.harness.data.relgen import LABELS, PARTS, SceneDraft, TokenIndex
from rrp.harness.data.relgen.contact import contact_pairs, grasp_target_build, grasp_target_vary, handover_pairs, \
    held_pairs
from rrp.policies.relations.base import FactorError, PrivilegedInput, assert_deployable, get_factor, resolve
from rrp.policies.relations.ops import OPS


# ------------------------------------------------------------------------------------------------ fake StateView
class _FakeStateView:
    """The `StateView` Protocol (docs/relations.md 5.1) needs only `caps`/`time`/`gravity` and the methods the
    labels under test call (`entities`, `contacts`); the rest are unused by this module and omitted."""
    caps = frozenset({"poses", "contacts"})
    time = 0.0
    gravity = np.array([0.0, 0.0, -9.81])

    def __init__(self, entities: list[EntityState], contacts: list[ContactState]):
        self._entities, self._contacts = entities, contacts

    def entities(self):
        return self._entities

    def contacts(self):
        return self._contacts


def _ent(id_, kind):
    return EntityState(id=id_, kind=kind, name=id_, pos=np.zeros(3))


def _contact(a, b):
    return ContactState(a=a, b=b, pos=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]))


def _grasp_scene(n_finger_contacts: int) -> tuple[_FakeStateView, TokenIndex]:
    """One gripper assembly + one object, touching at `n_finger_contacts` separate points (2 distinct finger bodies
    of the SAME assembly both resolve to the one "gripper" entity id, docs/relations.md 5.1's `_body_entity_map`, so
    `n_finger_contacts` separate `ContactState`s is exactly how `>= 2` distinct fingers shows up here)."""
    sv = _FakeStateView([_ent("gripper", "assembly"), _ent("cube", "object")],
                        [_contact("gripper", "cube") for _ in range(n_finger_contacts)])
    idx = TokenIndex(sets={"ctx": ["gripper", "cube", None]})   # trailing None: a null-identity / non-entity slot
    return sv, idx


def _dual_handover_scene() -> tuple[_FakeStateView, TokenIndex]:
    """Two manipulator assemblies each touching the same object once (one contact each -- below the `held` threshold
    on either side alone, but a handover in progress: both hands on the object at once)."""
    sv = _FakeStateView([_ent("left", "assembly"), _ent("right", "assembly"), _ent("peg", "object")],
                        [_contact("left", "peg"), _contact("peg", "right")])   # order (a, b) must not matter
    idx = TokenIndex(sets={"ctx": ["left", "right", "peg"]})
    return sv, idx


# ------------------------------------------------------------------------------------------------ registration
def test_labels_and_part_registered():
    for name, fn in (("contact_pairs", contact_pairs), ("held_pairs", held_pairs), ("handover_pairs", handover_pairs)):
        assert name in LABELS
        d = LABELS[name]
        assert d.fn is fn and d.arity == 2 and d.needs == frozenset({"poses", "contacts"})
    assert "grasp_target" in PARTS
    p = PARTS["grasp_target"]
    assert p.activates == frozenset({"contact"}) and p.build is grasp_target_build and p.vary is grasp_target_vary


def test_ix_factors_registered_in_own_catalog_section():
    for name, label in (("ix.contact", "contact_pairs"), ("ix.held_by", "held_pairs"), ("ix.handover", "handover_pairs")):
        d = get_factor(name)
        assert d.field == "hidden" and d.op == "bilinear" and d.form == "aug"
        assert d.label == label and d.sources[0] == "probe" and "gt" in d.sources
        assert d.readout is not None and d.readout.address == "pair" and d.readout.label == label
        assert d.p.get("rank") == 8


# ------------------------------------------------------------------------------------------------ contact_pairs
def test_contact_pairs_true_only_for_touching_entities():
    sv, idx = _grasp_scene(n_finger_contacts=1)
    lab = contact_pairs(sv, idx)
    assert lab.value.shape == (3, 3, 1) and lab.valid.shape == (3, 3)
    # slot 2 is the null identity: never valid, regardless of row/column
    assert not lab.valid[2].any() and not lab.valid[:, 2].any()
    assert lab.valid[0, 1] and lab.valid[1, 0]
    assert lab.value[0, 1, 0] == 1.0 and lab.value[1, 0, 0] == 1.0     # symmetric
    assert lab.value[0, 0, 0] == 0.0 and lab.value[1, 1, 0] == 0.0     # no self-contact


def test_contact_pairs_false_when_no_contact_recorded():
    sv, idx = _grasp_scene(n_finger_contacts=0)
    lab = contact_pairs(sv, idx)
    assert lab.value[0, 1, 0] == 0.0 and lab.value[1, 0, 0] == 0.0
    assert lab.valid[0, 1]                                             # still a valid (known) pair, just untouching


# ------------------------------------------------------------------------------------------------ held_pairs
def test_held_pairs_needs_at_least_two_contact_records():
    sv1, idx = _grasp_scene(n_finger_contacts=1)
    assert held_pairs(sv1, idx).value[0, 1, 0] == 0.0                  # one finger touching: not yet held
    sv2, _ = _grasp_scene(n_finger_contacts=2)
    lab2 = held_pairs(sv2, idx)
    assert lab2.value[0, 1, 0] == 1.0 and lab2.value[1, 0, 0] == 1.0
    sv3, _ = _grasp_scene(n_finger_contacts=3)
    assert held_pairs(sv3, idx).value[0, 1, 0] == 1.0                  # more than the threshold also counts


def test_held_pairs_excludes_manipulator_manipulator_and_object_object():
    sv, idx = _dual_handover_scene()
    lab = held_pairs(sv, idx)
    assert lab.value[0, 1, 0] == 0.0 and lab.value[1, 0, 0] == 0.0     # left <-> right: both assemblies
    assert lab.value[0, 2, 0] == 0.0                                   # only 1 contact each: below the threshold


# ------------------------------------------------------------------------------------------------ handover_pairs
def test_handover_pairs_true_for_two_hands_on_one_object():
    sv, idx = _dual_handover_scene()
    lab = handover_pairs(sv, idx)
    assert lab.value[0, 1, 0] == 1.0 and lab.value[1, 0, 0] == 1.0     # left, right: co-contact on `peg`
    assert lab.value[0, 2, 0] == 0.0 and lab.value[1, 2, 0] == 0.0     # hand <-> object is not a handover pair


def test_handover_pairs_false_for_a_single_gripper_grasp():
    sv, idx = _grasp_scene(n_finger_contacts=2)
    lab = handover_pairs(sv, idx)
    assert not lab.value.any()                                        # only one manipulator assembly present


def test_contact_labels_need_a_ctx_token_index():
    sv, _ = _grasp_scene(n_finger_contacts=1)
    with pytest.raises(KeyError):
        contact_pairs(sv, TokenIndex(sets={"knots": ["gripper", "cube"]}))


# ------------------------------------------------------------------------------------------------ deploy guard
def test_ix_factors_default_to_probe_and_gt_is_privileged_only():
    on = resolve(["ix.contact"])
    assert_deployable(on)                                              # probe (learned) is deployable
    gt = resolve([{"name": "ix.contact", "source": "gt"}])
    with pytest.raises(PrivilegedInput):
        assert_deployable(gt)


def test_ix_factor_rejects_a_source_it_does_not_declare():
    with pytest.raises(FactorError):
        resolve([{"name": "ix.held_by", "source": "given"}])           # no public "given" contact field exists


# ------------------------------------------------------------------------------------------------ part `grasp_target`
def test_grasp_target_build_adds_one_graspable_object():
    draft = SceneDraft(env="mujoco/arm")
    rng = np.random.default_rng(0)
    grasp_target_build(draft, rng)
    assert len(draft.entities) == 1 and draft.entities[0]["role"] == "graspable"
    assert len(draft.events) == 1 and draft.events[0]["entity"] == draft.entities[0]["id"]


def test_grasp_target_vary_needs_a_built_draft():
    with pytest.raises(ValueError):
        grasp_target_vary(SceneDraft(env="mujoco/arm"), np.random.default_rng(0), "reach")


def test_grasp_target_vary_reach_changes_only_the_objects_position():
    draft = SceneDraft(env="mujoco/arm")
    grasp_target_build(draft, np.random.default_rng(0))
    near, far = grasp_target_vary(draft, np.random.default_rng(0), "reach")
    assert near.entities[0]["id"] == far.entities[0]["id"] == draft.entities[0]["id"]
    assert near.entities[0]["pos"] != far.entities[0]["pos"]
    assert near.events == far.events == draft.events
    assert near.env == far.env == draft.env


def test_grasp_target_vary_rejects_unknown_factor():
    draft = SceneDraft(env="mujoco/arm")
    grasp_target_build(draft, np.random.default_rng(0))
    with pytest.raises(ValueError):
        grasp_target_vary(draft, np.random.default_rng(0), "color")


# ------------------------------------------------------------------------------------------------ bilinear readout
# training (acceptance: "bilinear readout trains on a synthetic batch, loss decreases, host-cheap")
def test_ix_held_by_bilinear_readout_trains_on_a_synthetic_batch():
    """The `bilinear` op's own kernel IS the pair probe (docs/relations.md 3.2: p_ij = sigmoid(<Ux_i,Vx_j> + c)); a
    factor's readout is this same score, so training it is training `U`/`V`/`g`/`c` from `OPS["bilinear"].build`
    directly (no net, no simulation -- CPU, T=6, ~a hundred Adam steps: host-cheap). The synthetic target is itself a
    bilinear function of fixed random token hiddens (`sign(x_i . x_j)`), so it is realizable by this exact op family
    and the loss must decrease; this is not a claim about real contact data, only that the registered `ix.*` readout
    mechanism trains."""
    torch.manual_seed(0)
    d, s = get_factor("ix.held_by"), resolve(["ix.held_by"])[0]
    bop = OPS[d.op]
    dim, heads, T = 16, 1, 6
    mod = bop.build(d, s, heads, dim)
    x = torch.randn(1, T, dim)                                         # fixed synthetic token hiddens
    y = (torch.einsum("bqd,bkd->bqk", x, x) > 0).float()                # bilinear-realizable target

    def _loss():
        phi_q, phi_k = bop.features(d, s, mod, None, "ctx>ctx", x, x)
        logits = torch.einsum("bhqr,bhkr->bhqk", phi_q, phi_k)[:, 0] + mod.c
        return F.binary_cross_entropy_with_logits(logits, y)

    opt = torch.optim.Adam(mod.parameters(), lr=0.05)
    with torch.no_grad():
        initial = float(_loss())
    for _ in range(200):
        opt.zero_grad()
        loss = _loss()
        loss.backward()
        opt.step()
    with torch.no_grad():
        final = float(_loss())
    assert final < initial - 0.1, f"bilinear readout did not train: {initial=} {final=}"
    assert final < 0.2, f"bilinear readout failed to fit a realizable synthetic target: {final=}"


# ------------------------------------------------------------------------------------------------ live StateView smoke
def test_contact_pairs_runs_on_a_real_mujoco_state_view():
    """Not a physics/timing assertion (that would be flaky): just that the label runs against a REAL
    `Session.state_view()` (docs/relations.md 5.1) and returns arrays shaped by that scene's own entity count."""
    from rrp.envs.mujoco.fixtures import make_pick_place_session
    s = make_pick_place_session(seed=3)
    sv = s.state_view()
    ents = [e.id for e in sv.entities()]
    idx = TokenIndex(sets={"ctx": ents})
    lab = contact_pairs(sv, idx)
    n = len(ents)
    assert lab.value.shape == (n, n, 1) and lab.valid.shape == (n, n)
    assert lab.valid.all()                                              # every slot here is a real entity
