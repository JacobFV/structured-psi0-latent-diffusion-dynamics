"""Unit R18 (D-144, docs/relations.md section 10): task / temporal / epistemic factors.

Covers this unit's acceptance row: `task.next_contact` candidates resolve on the arm (pick_place) and dual
fixtures; `reveal` / `surprise` (R9) apply to its targets; the `task` gate changes the bias between two task
(summary) vectors. Also covers `time.same_track` (registration + operator math) and the `next_contact` label
against a real MuJoCo `StateView` and a synthetic contact scene.

Red/green: none of `task.next_contact`, `time.same_track`, `next_contact`, `candidate_interaction_edges` existed
before this unit; every test below fails on pre-R18 `main` (ImportError / KeyError) and passes after.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from rrp.harness.data.relgen import LABELS, TRANSFORMS
from rrp.harness.data.relgen.task import manipulator_ids, next_contact_sample, touching_entities
import rrp.harness.data.relgen.transforms  # noqa: F401  (registers TRANSFORMS on import)
from rrp.policies.nets.batch import CAND_REL_VOCAB, candidate_interaction_edges, collate_inputs
from rrp.policies.relations.base import (Algebra, FactorDef, FactorError, PrivilegedInput, RelCtx, TokenSet, compat_hash,
                                         register_factor, resolve)
from rrp.policies.relations.ops import FactorSite

pytest.importorskip("mujoco")

# `time.same_track` is planned (round 2: no net family fills `track_id`), so its operator math runs on this twin.
register_factor(FactorDef("test.r18_same_track", "1", field="track_id", op="same", form="aug",
                          algebra=Algebra(arity=2, direction="symmetric"), sources=("given",)))


# ------------------------------------------------------------------------------------------------ fixtures
def _arm_session(seed=3):
    from rrp.envs.mujoco.fixtures import make_pick_place_session
    return make_pick_place_session(seed=seed)


def _dual_session(seed=3):
    from rrp.bodies.catalog import workbench_robots
    from rrp.envs.mujoco.dual import DualSession
    from rrp.envs.mujoco.dual_scenarios import build_support_insert
    W = workbench_robots()
    return DualSession(build_support_insert([W["parm5l_pg2"](), W["parm6_pg2"]()], seed), seed=seed)


def _arm_batch(seed=3):
    from rrp.policies.features.featurizer import featurizer_for
    s = _arm_session(seed)
    pi = featurizer_for(s)(s.observe())
    return s, pi, collate_inputs([pi])


def _dual_batch(seed=3):
    from rrp.policies.features.multi import MultiFeaturizer
    s = _dual_session(seed)
    pi = MultiFeaturizer(s.model, s.scenario.robots)(s.observe())
    return s, pi, collate_inputs([pi])


# ------------------------------------------------------------------------------------------------ registry
def test_factors_registered_and_resolve():
    specs = resolve(["task.next_contact", "test.r18_same_track"])
    assert [s.name for s in specs] == ["task.next_contact", "test.r18_same_track"]
    assert compat_hash(specs) != compat_hash(resolve(["task.next_contact"]))
    with pytest.raises(FactorError):
        resolve(["time.same_track"])                                         # planned (round 2): no net carries track_id
    resolve([{"name": "task.next_contact", "gate": "task"}])                 # "task" is an allowed gate: no raise
    with pytest.raises(FactorError):
        resolve([{"name": "task.next_contact", "gate": "goal"}])             # "goal" is not in FactorDef.gates
    with pytest.raises(FactorError):
        # `bilinear`'s controls are on/off/zero/rewired only (never gt/estimated: it reads hiddens, not a
        # source-resolved field) -- true for every factor built on `op="bilinear"`, `task.next_contact` included.
        resolve([{"name": "task.next_contact", "control": "gt"}])
    with pytest.raises(FactorError):
        resolve([{"name": "test.r18_same_track", "control": "gt"}])              # "given"-only: no gt source either


def test_next_contact_readout_is_soft_ce_pair():
    from rrp.policies.relations.base import get_factor
    d = get_factor("task.next_contact")
    assert d.readout.loss == "soft_ce" and d.readout.address == "pair" and d.readout.reads == "hidden"
    assert d.gates == ("task",) and d.op == "bilinear" and d.form == "aug"


def test_next_contact_source_gt_is_reachable_only_via_explicit_source_override():
    """`control="gt"` is never valid on a `bilinear` factor (checked above); `sources=("probe", "gt")` is still
    meaningful through an explicit `FactorSpec(source="gt")` override (matches sibling `ix.*`, R16), which the
    deploy guard still treats as privileged."""
    from rrp.policies.relations.base import assert_deployable, effective_source
    [gt_spec] = resolve([{"name": "task.next_contact", "source": "gt"}])
    assert effective_source(gt_spec) == "gt"
    with pytest.raises(PrivilegedInput):                                  # deploy guard: gt-sourced never deploys
        assert_deployable([gt_spec])
    [probe_spec] = resolve(["task.next_contact"])
    assert effective_source(probe_spec) == "probe"
    assert_deployable([probe_spec])                                       # must NOT raise


# ------------------------------------------------------------------------------------------------ gate "task"
def _randomize(module, seed=7):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=g))


def test_gate_task_changes_bias_between_two_task_texts():
    """The `task` gate (docs 3.4) scales a factor's `aug` contribution by sigmoid(u . c + b); with randomized gate
    parameters, two different task summary vectors ("two task texts") must produce different q/k augmentations and
    therefore different attention logits -- this is the causal mechanism `docs/relations.md` describes as "the same
    scene shows several candidate next-contact edges; after the task description the bias focuses on the required
    sequence"."""
    B, T, dim, heads = 2, 5, 16, 2
    mask = torch.ones(B, T, dtype=torch.bool)
    rc = RelCtx(sets={"q": TokenSet("q", mask), "k": TokenSet("k", mask)})
    site = FactorSite(heads, dim, "q>k", resolve([{"name": "task.next_contact", "gate": "task"}]), ("hidden",))
    _randomize(site)
    x = torch.randn(B, T, dim)

    task_a = torch.randn(B, dim)
    rc.task = task_a
    qa_a, ka_a = site.augment(rc, x, x)
    logits_a = qa_a @ ka_a.transpose(-1, -2)

    task_b = torch.randn(B, dim)
    rc.task = task_b
    qa_b, ka_b = site.augment(rc, x, x)
    logits_b = qa_b @ ka_b.transpose(-1, -2)

    assert ka_a.equal(ka_b)                                    # the gate scales the QUERY side only (docs 3.4)
    assert not torch.allclose(qa_a, qa_b)
    assert not torch.allclose(logits_a, logits_b)

    # same task text twice -> byte-identical bias (determinism, no hidden RNG use)
    rc.task = task_a
    qa_a2, ka_a2 = site.augment(rc, x, x)
    assert torch.equal(qa_a, qa_a2) and torch.equal(ka_a, ka_a2)


def test_zero_init_gate_is_a_no_op():
    """Un-randomized (freshly built) site: every learned coefficient is zero-init (docs 3.2), so the gate multiplier
    is exactly 1 regardless of the task text and enabling the factor changes nothing at step 0."""
    B, T, dim, heads = 1, 4, 8, 1
    mask = torch.ones(B, T, dtype=torch.bool)
    rc = RelCtx(sets={"q": TokenSet("q", mask), "k": TokenSet("k", mask)})
    site = FactorSite(heads, dim, "q>k", resolve([{"name": "task.next_contact", "gate": "task"}]), ("hidden",))
    x = torch.randn(B, T, dim)
    rc.task = torch.randn(B, dim)
    qa, ka = site.augment(rc, x, x)
    assert torch.equal(qa, torch.zeros_like(qa))               # g (bilinear) zero-init -> q features are exactly 0


# ------------------------------------------------------------------------------------------------ time.same_track
def test_same_track_is_exact_equality_aug():
    B, Tq, Tk, dim, heads = 1, 4, 5, 8, 2
    track_q = torch.tensor([[0, 1, 1, 2]])
    track_k = torch.tensor([[1, 1, 0, 2, 3]])
    q = TokenSet("q", torch.ones(B, Tq, dtype=torch.bool), fields={"track_id": track_q[..., None].float()})
    k = TokenSet("k", torch.ones(B, Tk, dtype=torch.bool), fields={"track_id": track_k[..., None].float()})
    rc = RelCtx(sets={"q": q, "k": k})
    # n_ids <= code_dim: SameOp's fixed random codes are then ORTHONORMAL, i.e. EXACT equality (docs 3.2's `same`
    # op); the registry default (n_ids=64 > code_dim=16) is only approximate, so this test overrides it.
    site = FactorSite(heads, dim, "q>k", resolve([{"name": "test.r18_same_track", "params": {"n_ids": 8}}]), ("track_id",))
    with torch.no_grad():
        site.f["test__r18_same_track"].g.fill_(1.0)
    x = torch.randn(B, Tq, dim)
    xk = torch.randn(B, Tk, dim)
    qa, ka = site.augment(rc, x, xk)
    got = (qa @ ka.transpose(-1, -2))[0, 0]                    # [Tq, Tk]
    expected = (track_q[0][:, None] == track_k[0][None, :]).float()
    assert torch.allclose(got, expected, atol=1e-4)


# ------------------------------------------------------------------------------------------------ candidate edges (batch.py)
@pytest.mark.parametrize("which", ["arm", "dual"])
def test_candidate_interaction_edges_shape_and_provenance(which):
    _, pi, b = _arm_batch() if which == "arm" else _dual_batch()
    es = candidate_interaction_edges([pi], b)
    assert es.vocab == CAND_REL_VOCAB
    assert es.prov == "public"
    C = b.ctx_mask.shape[1]
    assert tuple(es.data.shape) == (1, C, C, 3)
    assert es.data.min() >= 0.0 and es.data.max() <= 1.0 + 1e-6


def test_candidate_graspable_rows_are_uniform_over_scene_entities():
    _, pi, b = _arm_batch()
    es = candidate_interaction_edges([pi], b)
    scene_off = b.bank_offset["scene"]
    scene_kind = b.bank_kind["scene"][0].numpy()
    scene_mask = b.bank_mask["scene"][0].numpy()
    n_entities = int((scene_mask & (scene_kind == 0)).sum())
    assert n_entities >= 1                                     # pick_place always has at least the cube
    graspable = es.data[0, :, :, 0]
    sensor_rows = (graspable.sum(-1) > 0).nonzero(as_tuple=True)[0]
    assert len(sensor_rows) >= 1                               # at least one touch/grip sensor token found it
    for r in sensor_rows.tolist():
        row = graspable[r]
        nz = row[row > 0]
        assert len(nz) == n_entities
        np.testing.assert_allclose(nz.numpy(), np.full(n_entities, 1.0 / n_entities), atol=1e-6)
        assert row.sum().item() == pytest.approx(1.0, abs=1e-6)   # a proper per-row prior distribution


def test_candidate_support_destination_exclude_self():
    from rrp.envs.mujoco.fixtures import make_pick_place_session
    from rrp.policies.features.featurizer import featurizer_for
    s = make_pick_place_session(seed=5, n_distractors=2)       # >=2 scene entities so support/destination are non-trivial
    pi2 = featurizer_for(s)(s.observe())
    b2 = collate_inputs([pi2])
    es = candidate_interaction_edges([pi2], b2)
    scene_off = b2.bank_offset["scene"]
    scene_kind = b2.bank_kind["scene"][0].numpy()
    scene_mask = b2.bank_mask["scene"][0].numpy()
    entity_idx = scene_off + np.where(scene_mask & (scene_kind == 0))[0]
    assert len(entity_idx) >= 2
    for ch in (1, 2):                                          # support, destination
        for q in entity_idx.tolist():
            assert es.data[0, q, q, ch].item() == 0.0           # never its own support / destination
            row = es.data[0, q, :, ch]
            nz = (row > 0).nonzero(as_tuple=True)[0].numpy()
            assert set(nz.tolist()) == set(entity_idx.tolist()) - {q}


def test_candidate_edges_empty_batch_degrades_to_all_zero():
    """No sensor tokens / a single scene entity: the corresponding rows/channels stay all-zero, never crash."""
    C = 3
    from rrp.policies.nets.batch import Batch
    b = Batch(bank_tokens={}, bank_mask={"scene": torch.zeros(1, 1, dtype=torch.bool),
                                        "interact": torch.zeros(1, 1, dtype=torch.bool)},
             bank_kind={"scene": torch.zeros(1, 1, dtype=torch.long),
                        "interact": torch.zeros(1, 1, dtype=torch.long)},
             bank_text={}, bank_offset={"scene": 0, "interact": 1}, ctx_mask=torch.ones(1, C, dtype=torch.bool),
             node_feats=torch.zeros(1, 1, 1), node_mask=torch.zeros(1, 1, dtype=torch.bool),
             ctx_rel=torch.zeros(1, C, C, 1, dtype=torch.bool), act_rel=torch.zeros(1, 1, C, 1, dtype=torch.bool),
             node_rel=torch.zeros(1, 1, 1, 1, dtype=torch.bool), pointers=-torch.ones(1, 1, 2, dtype=torch.long),
             extra={})
    es = candidate_interaction_edges([None], b)
    assert es.data.abs().sum().item() == 0.0


# ------------------------------------------------------------------------------------------------ next_contact label (StateView)
class _FakeView:
    """Minimal `rrp.envs.base.StateView` stand-in (no simulator): a synthetic contact scene isolating the label's
    logic (manipulator vs. non-manipulator contact partners) from any MuJoCo geometry."""
    caps = frozenset({"contacts"})
    time = 0.0
    gravity = np.array([0.0, 0.0, -9.81])

    def __init__(self, entities, contacts):
        self._entities, self._contacts = entities, contacts

    def entities(self):
        return self._entities

    def contacts(self):
        return self._contacts

    def joints(self):
        return []

    def camera(self, name):
        raise NotImplementedError

    def ui_tree(self):
        return []

    def token_entity(self, token_set, slot):
        return None


def _entity(id_, kind):
    from rrp.envs.base import EntityState
    return EntityState(id=id_, kind=kind, name=id_, pos=np.zeros(3))


def _contact(a, b):
    from rrp.envs.base import ContactState
    return ContactState(a=a, b=b, pos=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]))


def test_touching_entities_ignores_manipulator_manipulator_and_nonmanip_contacts():
    view = _FakeView([_entity("gripper", "assembly"), _entity("cube", "object"), _entity("table", "object"),
                      _entity("distractor", "object")],
                     [_contact("gripper", "cube"), _contact("cube", "table"), _contact("gripper", "gripper")])
    assert manipulator_ids(view) == {"gripper"}
    assert touching_entities(view) == {"cube"}                # table<->table/cube contact never touches a manipulator


def test_next_contact_label_marks_only_the_touched_candidate():
    from rrp.harness.data.relgen import TokenIndex
    view = _FakeView([_entity("gripper", "assembly"), _entity("cube", "object"), _entity("target", "feature")],
                     [_contact("gripper", "cube")])
    index = TokenIndex(sets={"ctx": ["gripper", "cube", "target", None]})
    label = LABELS["next_contact"].fn(view, index)
    np.testing.assert_array_equal(label.value[:, 0], [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(label.valid, [True, True, True, False])
    assert label.prov == "gt"


def test_next_contact_label_all_zero_when_nothing_touching():
    from rrp.harness.data.relgen import TokenIndex
    view = _FakeView([_entity("gripper", "assembly"), _entity("cube", "object")], [])
    index = TokenIndex(sets={"ctx": ["gripper", "cube"]})
    label = LABELS["next_contact"].fn(view, index)
    assert label.value.sum() == 0.0


def test_next_contact_label_runs_on_real_mujoco_state_view_arm_and_dual():
    """Structural sanity against the REAL StateView contract (docs 5.1), not a mock: runs to completion, correct
    shape, and `caps` (only "contacts" is `needs`) actually covers what `label_runs_in` requires."""
    from rrp.harness.data.relgen import TokenIndex, label_runs_in
    for make in (_arm_session, _dual_session):
        s = make()
        view = s.state_view()
        assert label_runs_in("next_contact", view.caps)
        ids = [e.id for e in view.entities()][:3] + [None]
        index = TokenIndex(sets={"ctx": ids})
        label = LABELS["next_contact"].fn(view, index)
        assert label.value.shape == (len(ids), 1)
        assert label.valid.tolist() == [True, True, True, False]


# ------------------------------------------------------------------------------------------------ reveal / surprise applied to next_contact targets (acceptance)
def test_reveal_applied_to_next_contact_targets_collapses_on_the_touched_candidate():
    view = _FakeView([_entity("gripper", "assembly"), _entity("cube", "object"), _entity("distractor", "object")],
                     [_contact("gripper", "cube")])
    candidates = ["cube", "distractor"]
    sample = next_contact_sample(candidates, view)
    assert sample["inputs"]["evidence"] == [{"t": 0, "excludes": ["distractor"]}]
    [out] = TRANSFORMS["reveal"].fn(sample, np.random.default_rng(0), {"schedule": [0]})
    q0 = out["labels"]["reveal"]["value"][0]
    np.testing.assert_allclose(q0, [1.0, 0.0])                 # collapsed on the touched candidate at t=0


def test_reveal_applied_to_next_contact_targets_uniform_when_nothing_touching_yet():
    view = _FakeView([_entity("gripper", "assembly"), _entity("cube", "object"), _entity("distractor", "object")], [])
    candidates = ["cube", "distractor"]
    sample = next_contact_sample(candidates, view)
    assert sample["inputs"]["evidence"] == []                  # nothing excluded yet: no evidence event
    [out] = TRANSFORMS["reveal"].fn(sample, np.random.default_rng(0), {"schedule": [0]})
    np.testing.assert_allclose(out["labels"]["reveal"]["value"][0], [0.5, 0.5])


def test_surprise_applied_to_next_contact_targets_can_revise_after_collapse():
    """A multi-step episode: `evidence` grows over time (the caller's job per `next_contact_sample`'s docstring);
    once collapsed on "cube" (t=1), `surprise` (R9, unmodified) flips the target back open one step later
    (`after=1`) with the supplied rate=1.0 (deterministic trigger)."""
    view = _FakeView([_entity("gripper", "assembly"), _entity("cube", "object"), _entity("box", "object"),
                      _entity("bin", "object")],
                     [_contact("gripper", "cube")])
    candidates = ["cube", "box", "bin"]
    evidence = [{"t": 0, "excludes": ["box"]}, {"t": 1, "excludes": ["bin"]}]   # collapses onto "cube" at t=1
    sample = next_contact_sample(candidates, view, evidence=evidence)
    [out] = TRANSFORMS["surprise"].fn(sample, np.random.default_rng(1), {"schedule": [0, 1, 2, 3],
                                                                          "rate": 1.0, "after": 1})
    q = out["labels"]["surprise"]["value"]
    np.testing.assert_allclose(q[1], [1.0, 0.0, 0.0])          # collapsed on "cube" at t=1
    assert out["labels"]["surprise"]["switch_at"] == 2          # collapse_t(1) + after(1)
    np.testing.assert_allclose(q[2], [0.0, 0.5, 0.5])           # "cube" now known WRONG: reopened to box/bin
    np.testing.assert_allclose(q[3], [0.0, 0.5, 0.5])           # stays reopened (no further evidence)
