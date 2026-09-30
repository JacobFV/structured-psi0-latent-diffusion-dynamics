"""rel-geo (D-144 addendum; follow-up to R13/R17/R18's own "for the lead" notes -- archived research/tracks/rel-r13.md,
rel-r17.md, rel-r18.md; see research/decisions.md D-144 addendum): wire the geometry/interaction factors so they
actually reach attention.

  (1) `nets/flow.py` `CTX_CARRIES` now names the R12 geometry fields (`cam_uvd`, `pos3d`, `orient`, `normal`) so
      `geo.*` (unit R13) applies at the real `ctx>ctx` site -- previously a structural no-op inside `FlowPolicy`
      (R13's own lead question). Covered here: `geo.depth3d` reaches the REAL `ContextEncoder`/`FlowPolicy` (no
      per-process attribute patch, unlike R13's own peer smoke) and changes ctx attention logits once its zero-init
      coefficients are nudged; the default "arm" preset's `FactorSite.specs` at `ctx>ctx` -- and `tests/data/
      golden.json` -- are byte-identical to before this change.
  (2) `nets/batch.py` `support_edges`: `ix.force_flow`'s "edges:support-v1" `EdgeSet`, built from `ix.support`'s
      privileged label (`relgen.support.support_matrix`) in the collate path (R17's own lead question; the
      PROBE/estimate half needs a `relations/ops.py` hook this unit does not add -- see the decisions log).
  (3) Presets: `ix` now includes `ix.support` / `ix.force_flow`; new preset `task`
      (`task.next_contact`, `time.same_track`). A `route.assembly_reads` preset for R5's Ψ₀ usage was also asked
      for, but R5 landed its own `s0-psi0` (with a real `params.reads` table) while this unit was in flight --
      nothing left to add there; see the test/decisions-log note for the record.

Never touches `relations/base.py` / `relations/ops.py` (imported read-only).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from rrp.envs.base import ContactState, EntityState
from rrp.harness.data.relgen.support import support_matrix
from rrp.policies.nets.batch import Batch, SUPPORT_REL_VOCAB, collate_inputs, support_edges
from rrp.policies.nets.flow import CTX_CARRIES, ContextEncoder, FlowPolicy, PolicyConfig
import rrp.policies.relations.catalog  # noqa: F401  (registers FACTORS / FIELDS / PRESETS on import)
from rrp.policies.relations.base import FACTORS, PRESETS, resolve
from rrp.policies.relations.ops import FactorSite

pytest.importorskip("mujoco")


def _arm_batch(seed=3, n_distractors=1):
    from rrp.envs.mujoco.fixtures import make_pick_place_session
    from rrp.policies.features.featurizer import featurizer_for
    s = make_pick_place_session(seed=seed, n_distractors=n_distractors)
    f = featurizer_for(s)
    pi = f(s.observe())
    b = collate_inputs([pi])
    return s, pi, b


# ================================================================== (1) CTX_CARRIES
def test_ctx_carries_names_the_r12_geometry_fields():
    for name in ("cam_uvd", "pos3d", "orient", "normal", "edges:arm-rel-v1", "hidden"):
        assert name in CTX_CARRIES


def test_geo_factors_now_apply_at_ctx_ctx_with_the_real_carries_tuple():
    specs = resolve(["preset:geo"])
    site = FactorSite(heads=2, dim=8, site="ctx>ctx", specs=specs, carries=CTX_CARRIES)
    applied = {s.name for s in site.specs}
    # every geo.* factor field (pos3d, cam_uvd, orient, normal) is now named in CTX_CARRIES
    assert applied == {"geo.pos3d", "geo.depth3d", "geo.orient", "geo.normal_align", "geo.above"}


def test_default_arm_preset_factor_site_unchanged_by_the_wider_carries_tuple():
    """The exact acceptance ask: enabling geo.* is additive, never touches what the DEFAULT ("arm") preset applies
    at ctx>ctx -- same `FactorSite.specs` names/order as the old, narrower `CTX_CARRIES` would have produced (edge.*
    factors key on "edges:arm-rel-v1" only; msg.incidence is form="message", filtered out before the carries check
    at all, on EITHER tuple -- neither can newly match the added field names)."""
    specs = resolve(["preset:arm"])
    old_carries = ("edges:arm-rel-v1", "hidden")
    site_new = FactorSite(heads=2, dim=8, site="ctx>ctx", specs=specs, carries=CTX_CARRIES)
    site_old = FactorSite(heads=2, dim=8, site="ctx>ctx", specs=specs, carries=old_carries)
    assert [s.name for s in site_new.specs] == [s.name for s in site_old.specs]
    assert len(site_new.specs) == 17                                 # the 17 edge.* factors; msg.incidence never
                                                                      # becomes a FactorSite spec on any site (form
                                                                      # "message", handled separately by ContextEncoder)


def test_geo_depth3d_reaches_the_real_flowpolicy_ctx_ctx_site_without_any_runtime_patch():
    """R13's own peer smoke had to patch `flow_mod.CTX_CARRIES` in-process because "cam_uvd" was not yet in the
    tuple (archived research/tracks/rel-r13.md "peer smoke"). With this unit's change, a real `FlowPolicy` built with
    `factors=["preset:arm", "geo.depth3d"]` picks it up out of the box -- no patch, no `FactorSite` built by hand."""
    cfg = PolicyConfig(width=16, heads=2, ctx_layers=1, blocks=1, horizon=2,
                       factors=["preset:arm", "geo.depth3d"])
    torch.manual_seed(0)
    model = FlowPolicy(cfg)
    site = model.context.layers[0]["bias"]
    assert "geo.depth3d" in {s.name for s in site.specs}


def test_geo_depth3d_changes_ctx_attention_logits_in_the_real_context_encoder():
    """End-to-end: run the real `ContextEncoder.forward` once (populates `rc.estimates["ctx","cam_uvd"]` via the
    existing `FieldReadouts` hook, unedited, at layer 0) so `rc` / `mask` come from the real model, then compare the
    SAME layer-0 `ctx>ctx` attention weights, at a fixed hidden state and a fixed (manually supplied, non-degenerate)
    `cam_uvd` probe estimate, before vs. after nudging `geo.depth3d`'s zero-init PaPE coefficients off zero (docs
    3.2: a freshly-registered aug factor is a no-op at step 0, so the comparison must nudge it to see any effect --
    same technique tests/unit/test_relations_geo.py already uses on a hand-built FactorSite; this test instead uses
    the REAL model's own FactorSite instance, reached only because CTX_CARRIES now names `cam_uvd`). The estimate is
    supplied directly (rather than read off `FieldReadouts`' own zero-init head, which predicts an identical mu=0
    for every token at a freshly-constructed model regardless of `h` -- true of ANY freshly-registered probe field,
    not specific to this factor -- and so would make BOTH q/k PaPE features collapse to the same all-zero point
    yielding a genuinely zero dot product no amount of nudging PaPE's own coefficients alone can move, an
    initialization artifact orthogonal to the CTX_CARRIES wiring under test here)."""
    s, pi, b = _arm_batch()
    cfg = PolicyConfig(width=16, heads=2, ctx_layers=1, blocks=1, horizon=2,
                       factors=["preset:arm", "geo.depth3d"])
    torch.manual_seed(0)
    model = FlowPolicy(cfg).eval()
    ce: ContextEncoder = model.context
    site = ce.layers[0]["bias"]
    names = [sp.name for sp in site.specs]
    assert "geo.depth3d" in names and len(names) == 18               # 17 edge.* (still apply via "edges:arm-rel-v1")
                                                                      # + geo.depth3d (now applies via "cam_uvd")

    with torch.no_grad():
        _, mask, rc = ce.forward(b)                                  # real rc/mask; real FieldReadouts wiring
    C = mask.shape[1]
    torch.manual_seed(1)
    mu = torch.randn(1, C, 3)                                        # a non-degenerate cam_uvd estimate: real
    var = torch.rand(1, C, 3) + 0.1                                  # `site_field(source="probe")` just reads
    rc.estimates[("ctx", "cam_uvd")] = (mu, var)                     # rc.estimates, however it got populated
    xn = torch.randn(1, C, cfg.D)
    att = ce.layers[0]["att"]

    def attn_weights():
        bias = site.bias(rc)
        qa, ka = site.augment(rc, xn, xn)
        _, w = att(xn, key_mask=mask, bias=bias, q_aug=qa, k_aug=ka, need_weights=True)
        return w

    w0 = attn_weights()
    with torch.no_grad():
        site.f["geo__depth3d"].g.fill_(1.0)
        site.f["geo__depth3d"].b.bias.fill_(0.3)
    w1 = attn_weights()
    assert not torch.allclose(w0, w1)
    assert torch.isfinite(w1).all()


def test_golden_suite_is_byte_identical_after_the_ctx_carries_change():
    """Direct evidence for the acceptance ask ("golden.json unchanged"): run the existing golden digests (recorded
    on pre-refactor code, `tests/unit/test_golden.py`) against this checkout WITHOUT `RRP_GOLDEN_RECORD` -- any
    drift fails loudly inside that module's own `_check`. This just confirms the mechanism this test file's other
    assertions already established structurally (default-preset FactorSite output is untouched) also holds for the
    real recorded digests; the full command + count are in this unit's track note (archived research/tracks/rel-geo.md /
    this run's transcript), not duplicated as a second golden mechanism here."""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/unit/test_golden.py", "-q"],
                       cwd=__file__.rsplit("/tests/", 1)[0], capture_output=True, text=True,
                       env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""})
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]


# ================================================================== (2) support_edges (ix.force_flow's edges:support-v1)
class _View:
    """Minimal `StateView` stand-in (docs 5.1), a chain `a -> b -> c` plus a non-supporting side object `d`; unlike
    `test_relations_support.py`'s `_StackView`, `token_entity` is IMPLEMENTED (slot -> this view's own entity ids),
    matching what `MujocoStateView.token_entity` does for real scene-bank slots (R7)."""
    caps = frozenset({"poses", "contacts"})
    time = 0.0
    gravity = np.array([0.0, 0.0, -9.81])

    def __init__(self, order=("a", "b", "c", "d")):
        pos = {"a": (0.0, 0.0, 0.05), "b": (0.0, 0.0, 0.15), "c": (0.0, 0.0, 0.25), "d": (0.5, 0.0, 0.05)}
        self._order = list(order)
        self._entities = [EntityState(id=i, kind="object", name=i, pos=np.array(pos[i])) for i in order]
        self._contacts = [ContactState(a="a", b="b", pos=np.array([0.0, 0.0, 0.10]), normal=np.array([0.0, 0.0, 1.0])),
                          ContactState(a="b", b="c", pos=np.array([0.0, 0.0, 0.20]), normal=np.array([0.0, 0.0, 1.0]))]

    def entities(self):
        return list(self._entities)

    def contacts(self):
        return list(self._contacts)

    def joints(self):
        return []

    def camera(self, name):
        raise NotImplementedError

    def ui_tree(self):
        return []

    def token_entity(self, token_set, slot):
        return self._order[slot] if 0 <= slot < len(self._order) else None


def _scene_batch(n_scene: int, n_ctx: int | None = None):
    C = n_ctx if n_ctx is not None else n_scene
    scene_mask = torch.zeros(1, C, dtype=torch.bool)
    scene_mask[0, :n_scene] = True
    scene_kind = torch.zeros(1, C, dtype=torch.long)                 # 0 = entity everywhere valid
    return Batch(bank_tokens={}, bank_mask={"scene": scene_mask}, bank_kind={"scene": scene_kind}, bank_text={},
                bank_offset={"scene": 0}, ctx_mask=torch.ones(1, C, dtype=torch.bool),
                node_feats=torch.zeros(1, 1, 1), node_mask=torch.zeros(1, 1, dtype=torch.bool),
                ctx_rel=torch.zeros(1, C, C, 1, dtype=torch.bool), act_rel=torch.zeros(1, 1, C, 1, dtype=torch.bool),
                node_rel=torch.zeros(1, 1, 1, 1, dtype=torch.bool), pointers=-torch.ones(1, 1, 2, dtype=torch.long),
                extra={})


def test_support_edges_direct_chain_matches_support_matrix():
    b = _scene_batch(n_scene=4)                                       # slots 0..3 -> a, b, c, d (this batch's order)
    view = _View()
    es = support_edges([None], b, graphs=[support_matrix(view)], views=[view])
    assert es.vocab == SUPPORT_REL_VOCAB == ("support",)
    d = es.data[0, :, :, 0]
    assert d[0, 1].item() == 1.0                                      # a supports b
    assert d[1, 2].item() == 1.0                                      # b supports c
    assert d[0, 2].item() == 0.0                                      # no direct a->c edge (that's ix.force_flow's
                                                                       # closure to compute, not this function's job)
    assert d[3].sum().item() == 0.0 and d[:, 3].sum().item() == 0.0   # d supports / is supported by nothing
    assert d.sum().item() == 2.0                                      # exactly the two direct edges, nothing else


def test_support_edges_channel_matches_ix_force_flow_edge_param():
    assert FACTORS["ix.force_flow"].field == "edges:support-v1"
    assert FACTORS["ix.force_flow"].p["edge"] in SUPPORT_REL_VOCAB


def test_support_edges_prov_is_privileged():
    b = _scene_batch(n_scene=4)
    view = _View()
    es = support_edges([None], b, graphs=[support_matrix(view)], views=[view])
    assert es.prov == "privileged"


def test_support_edges_no_view_or_too_few_entities_is_all_zero_never_errors():
    b = _scene_batch(n_scene=4)
    view = _View()
    assert support_edges([None], b, graphs=[support_matrix(view)], views=[None]).data.abs().sum().item() == 0.0
    assert support_edges([None], b, graphs=[None], views=[view]).data.abs().sum().item() == 0.0
    assert support_edges([None], b, graphs=[], views=[]).data.abs().sum().item() == 0.0   # shorter than inputs
    b1 = _scene_batch(n_scene=1)
    view1 = _View(order=("a",))
    assert support_edges([None], b1, graphs=[support_matrix(view1)],
                         views=[view1]).data.abs().sum().item() == 0.0


def test_support_edges_unresolved_token_entity_ids_are_skipped_not_errors():
    """A scene slot whose `token_entity` returns an id the graph never mentions (or `None`) simply contributes
    nothing for that slot -- matches `candidate_interaction_edges`'s "never crashes" precedent."""
    class _SparseView(_View):
        def token_entity(self, token_set, slot):
            return None if slot == 1 else super().token_entity(token_set, slot)   # slot 1 ("b") unresolved
    b = _scene_batch(n_scene=4)
    view = _SparseView()
    es = support_edges([None], b, graphs=[support_matrix(view)], views=[view])
    assert es.data.sum().item() == 0.0                                # both real edges touch "b": neither resolves


def test_support_edges_on_a_real_arm_fixture_shape_and_range_smoke():
    """Real `Session.state_view()` (R7) + real featurizer/collate path: no crash, right shape, values in {0, 1}."""
    s, pi, b = _arm_batch(n_distractors=2)
    view = s.state_view()
    es = support_edges([pi], b, graphs=[support_matrix(view)], views=[view])
    C = b.ctx_mask.shape[1]
    assert es.data.shape == (1, C, C, 1)
    vals = torch.unique(es.data)
    assert set(vals.tolist()) <= {0.0, 1.0}


# ================================================================== (3) presets
def test_ix_preset_includes_support_and_force_flow():
    assert set(PRESETS["ix"]) == {"ix.contact", "ix.held_by", "ix.handover", "ix.support", "ix.force_flow"}
    specs = resolve(["preset:ix"])
    assert {s.name for s in specs} == set(PRESETS["ix"])


def test_task_preset_is_next_contact_only():
    """`time.same_track` is planned (round 2: no family carries a `track_id` field), so the preset omits it."""
    assert PRESETS["task"] == ("task.next_contact",)
    specs = resolve(["preset:task"])
    assert [s.name for s in specs] == ["task.next_contact"]


def test_route_assembly_reads_preset_already_resolved_by_r5():
    """The third preset this unit's brief asked for -- a `route.assembly_reads` preset for R5's Ψ₀ usage -- turned
    out to already exist: R5 (`db8b603a`) landed on `main` with `register_preset("s0-psi0", [{"name":
    "route.assembly_reads", "params": {"reads": ...}}])` (a real `[M,M]` table from `bodies.g1_simple.reads_table()`,
    not a bare name) while this unit was in flight. Nothing to add; this just pins that it stayed resolved and
    resolves correctly, so a future rebase conflict here is obvious."""
    assert "s0-psi0" in PRESETS
    specs = resolve(["preset:s0-psi0"])
    assert [s.name for s in specs] == ["route.assembly_reads"]
    assert specs[0].p.get("reads") is not None                       # R5's real table, not a bare fallback
    assert FACTORS["route.assembly_reads"].field == "assembly_id"
    assert FACTORS["route.assembly_reads"].form == "mask"
