"""R13 (D-144, docs/relations.md section 10): geometry factors -- `catalog.py` section `geo` (`geo.pos3d`,
`geo.depth3d`, `geo.orient`, `geo.normal_align`, `geo.above`): sqdiff+diff (PaPE) on `pos3d` / `cam_uvd`, rel_rot on
`orient`, align on `normal`, order along the world-up axis on `pos3d`. Field readouts (`source="probe"`,
`params.readout_layer`) are wired through the foundation hook, `rrp.policies.relations.ops.FieldReadouts`
(`rrp.policies.nets.flow.ContextEncoder` already calls `.observe(layer, "ctx", h, rc)` every layer -- units never
touch `relations/base.py` / `relations/ops.py`; this file only checks the entries this unit owns resolve and wire
against that existing hook).

Red/green: `test_deploy_guard_*` is the row's "guard red/green with source=gt" acceptance -- RED asserts the guard
actually blocks a privileged source at deploy, GREEN asserts a normal spec is unaffected.
"""
import math

import pytest
import torch

from rrp.envs.mujoco.fixtures import make_pick_place_session
from rrp.policies.features.featurizer import featurizer_for
from rrp.policies.features.multi import MultiFeaturizer
from rrp.policies.nets.batch import collate_inputs, relation_token_sets
import rrp.policies.relations.catalog  # noqa: F401  (registers FACTORS / FIELDS on import)
from rrp.policies.relations.base import FACTORS, FIELDS, PrivilegedInput, RelCtx, TokenSet, assert_deployable, resolve
from rrp.policies.relations.catalog import ARM_REL_VOCAB
from rrp.policies.relations.ops import FactorSite, FieldReadouts, site_field

GEO_NAMES = ("geo.pos3d", "geo.depth3d", "geo.orient", "geo.normal_align", "geo.above")


def _arm_batch(seed=3):
    s = make_pick_place_session(seed=seed)
    f = featurizer_for(s)
    pi = f(s.observe())
    b = collate_inputs([pi])
    return s, pi, b


# ------------------------------------------------------------------ catalog entries match the brief
def test_geo_entries_registered_field_op_form():
    assert set(FACTORS) >= set(GEO_NAMES)
    assert (FACTORS["geo.pos3d"].field, FACTORS["geo.pos3d"].op, FACTORS["geo.pos3d"].form) == ("pos3d", "sqdiff+diff", "aug")
    assert (FACTORS["geo.depth3d"].field, FACTORS["geo.depth3d"].op, FACTORS["geo.depth3d"].form) == ("cam_uvd", "sqdiff+diff", "aug")
    assert (FACTORS["geo.orient"].field, FACTORS["geo.orient"].op, FACTORS["geo.orient"].form) == ("orient", "rel_rot", "aug")
    assert (FACTORS["geo.normal_align"].field, FACTORS["geo.normal_align"].op, FACTORS["geo.normal_align"].form) == ("normal", "align", "aug")
    assert (FACTORS["geo.above"].field, FACTORS["geo.above"].op, FACTORS["geo.above"].form) == ("pos3d", "order", "bias")
    for n in GEO_NAMES:
        assert FACTORS[n].status == "implemented"
    for n in ("pos3d", "cam_uvd", "orient", "normal"):
        assert n in FIELDS                                        # unit R12 (+ foundation) owns these FieldDefs


def test_geo_entries_declare_readout_layer_and_probe_source_where_expected():
    for n in ("geo.pos3d", "geo.depth3d", "geo.orient", "geo.normal_align"):
        d = FACTORS[n]
        assert "probe" in d.sources
        assert d.readout is not None and d.readout.reads == "tokens"
        assert int(d.p.get("readout_layer", -1)) == 0
    assert "probe" not in FACTORS["geo.above"].sources                   # a bias/order relation, no readout of its own


# ------------------------------------------------------------------ resolves on arm / dual (row acceptance)
def test_resolve_on_arm():
    specs = resolve(["preset:arm", "preset:geo"])
    names = [s.name for s in specs]
    assert names[:len(ARM_REL_VOCAB) + 1] == [f"edge.{n}" for n in ARM_REL_VOCAB] + ["msg.incidence"]
    assert set(GEO_NAMES) <= set(names)
    for s in specs:
        if s.name in GEO_NAMES:
            assert s.control == "on"                                    # resolves clean, no override needed


def test_resolve_on_dual():
    """Dual featurizer reuses the same `arm-rel-v1` vocabulary / `ctx` token-set family (docs/relations.md section 2,
    unit R12's `test_ctx_fields_on_dual_featurizer`); the geo entries resolve against it identically."""
    from rrp.bodies.catalog import workbench_robots
    from rrp.envs.mujoco.dual import DualSession
    from rrp.envs.mujoco.dual_scenarios import build_support_insert
    W = workbench_robots()
    d = DualSession(build_support_insert([W["parm5l_pg2"](), W["parm6_pg2"]()], 3), seed=3)
    mf = MultiFeaturizer(d.model, d.scenario.robots)
    pi = mf(d.observe())
    b = collate_inputs([pi])
    sets = relation_token_sets([pi], b)
    ctx = sets["ctx"]
    specs = resolve(["preset:geo"])
    site = FactorSite(heads=2, dim=8, site="ctx>ctx", specs=specs, carries=("pos3d", "orient", "hidden"))
    applied = {s.name for s in site.specs}
    assert {"geo.pos3d", "geo.orient", "geo.above"} <= applied            # fields R12 populates on the dual ctx set
    rc = RelCtx(sets={"ctx": ctx}, edges={}, generator=torch.Generator())
    h = torch.randn(1, ctx.mask.shape[1], 8)
    bias = site.bias(rc)
    qa, ka = site.augment(rc, h, h)
    assert bias is not None and torch.isfinite(bias).all()
    assert qa is not None and torch.isfinite(qa).all() and torch.isfinite(ka).all()


# ------------------------------------------------------------------ deploy guard, red / green (row acceptance)
def test_deploy_guard_red_blocks_gt_control():
    specs = resolve([{"name": "geo.pos3d", "control": "gt"}])
    with pytest.raises(PrivilegedInput):
        assert_deployable(specs)


def test_deploy_guard_red_blocks_gt_source():
    specs = resolve([{"name": "geo.depth3d", "source": "gt"}])
    with pytest.raises(PrivilegedInput):
        assert_deployable(specs)


def test_deploy_guard_green_default_and_off_gt_pass():
    assert_deployable(resolve(["preset:geo"]))                                     # defaults: never privileged
    assert_deployable(resolve([{"name": "geo.orient", "control": "off", "source": "gt"}]))  # off is never a deploy risk


def test_deploy_guard_rejects_gt_where_sources_disallow_it():
    with pytest.raises(Exception):
        resolve([{"name": "geo.above", "source": "probe"}])                        # geo.above has no "probe" source


# ------------------------------------------------------------------ field readouts (source=probe, readout_layer)
# wired through the foundation hook (`ops.FieldReadouts` / `ops.site_field`; `nets.flow.ContextEncoder` already
# calls `FieldReadouts.observe` once per layer -- this is the mechanism it plugs into).
def test_field_readout_populates_estimate_only_at_its_layer():
    specs = resolve(["geo.depth3d"])                                      # default source = probe, readout_layer 0
    fr = FieldReadouts(dim=8, specs=specs)
    assert [n for n, *_ in fr.items] == ["geo.depth3d"]
    rc = RelCtx(sets={"ctx": TokenSet("ctx", torch.ones(1, 3, dtype=torch.bool))}, generator=torch.Generator())
    h = torch.randn(1, 3, 8)
    fr.observe(layer=1, set_name="ctx", h=h, rc=rc)                       # wrong layer: nothing written
    assert ("ctx", "cam_uvd") not in rc.estimates
    fr.observe(layer=0, set_name="ctx", h=h, rc=rc)                       # its own layer: written
    mu, var = rc.estimates[("ctx", "cam_uvd")]
    assert mu.shape == (1, 3, FIELDS["cam_uvd"].dim) and var.shape == mu.shape
    assert torch.isfinite(mu).all() and (var > 0).all()


def test_readout_layer_param_moves_the_hook():
    specs = resolve([{"name": "geo.orient", "source": "probe", "params": {"readout_layer": 2}}])
    fr = FieldReadouts(dim=8, specs=specs)
    rc = RelCtx(sets={"ctx": TokenSet("ctx", torch.ones(1, 2, dtype=torch.bool))}, generator=torch.Generator())
    h = torch.randn(1, 2, 8)
    fr.observe(layer=0, set_name="ctx", h=h, rc=rc)
    assert ("ctx", "orient") not in rc.estimates
    fr.observe(layer=2, set_name="ctx", h=h, rc=rc)
    assert ("ctx", "orient") in rc.estimates


def test_site_field_probe_source_reads_the_readout_estimate_not_the_raw_field():
    specs = resolve(["geo.depth3d"])
    ts = TokenSet("ctx", torch.ones(1, 4, dtype=torch.bool), fields={})    # NO cam_uvd given: must come from the probe
    rc = RelCtx(sets={"ctx": ts}, generator=torch.Generator())
    fr = FieldReadouts(dim=8, specs=specs)
    fr.observe(0, "ctx", torch.randn(1, 4, 8), rc)
    v, var = site_field(rc, "ctx", "cam_uvd", specs[0])
    mu, evar = rc.estimates[("ctx", "cam_uvd")]
    assert torch.equal(v, mu) and torch.equal(var, evar)
    with pytest.raises(Exception):                                         # source=probe with NO prior observe(): errors, not silent zeros
        site_field(rc, "ctx", "orient", resolve(["geo.orient"])[0])


def test_factor_site_augments_from_the_probe_estimate_end_to_end():
    """`FactorSite` at a site that carries the probed field picks the factor up and produces finite q/k features
    sourced entirely from the readout (no raw `cam_uvd` field exists on the token set at all). At zero-init `phi_q`
    is exactly 0 (docs 3.2: a-coefficient and b zero-init -> M = bt = 0 identically, so `Mr` does not yet depend on
    `rq`); nudging the op's zero-init coefficient off zero -- exactly what one optimizer step does -- opens the real
    gradient path loss -> qa -> rq (the probe estimate) -> `FieldReadouts` parameters, which this test exercises."""
    specs = resolve(["geo.depth3d"])
    B, T, D, H = 2, 5, 8, 2
    mask = torch.ones(B, T, dtype=torch.bool)
    ts = TokenSet("ctx", mask, fields={})
    rc = RelCtx(sets={"ctx": ts}, generator=torch.Generator())
    fr = FieldReadouts(D, specs)
    site = FactorSite(heads=H, dim=D, site="ctx>ctx", specs=specs, carries=("cam_uvd",))
    assert [s.name for s in site.specs] == ["geo.depth3d"]
    h = torch.randn(B, T, D, requires_grad=True)
    fr.observe(0, "ctx", h, rc)
    qa0, ka0 = site.augment(rc, h, h)
    assert qa0.shape == (B, H, T, 12) and ka0.shape == (B, H, T, 12)       # PaPE: p^2 + p = 9 + 3 = 12
    assert torch.isfinite(qa0).all() and torch.isfinite(ka0).all()
    assert torch.allclose(qa0, torch.zeros_like(qa0))                     # zero-init equivalence (docs 3.2)
    with torch.no_grad():                                                 # simulate a few optimizer steps off zero-init
        site.f["geo__depth3d"].g.fill_(1.0)
        site.f["geo__depth3d"].b.bias.fill_(0.3)
    qa, ka = site.augment(rc, h, h)
    assert not torch.allclose(qa, torch.zeros_like(qa))
    loss = (qa @ ka.transpose(-1, -2)).sum()
    loss.backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in fr.parameters())


# ------------------------------------------------------------------ zero-init equivalence (docs 3.2): a freshly
# built site contributes nothing until trained, regardless of source.
def test_geo_aug_factors_are_zero_init():
    specs = resolve(["preset:geo"])
    B, T, D, H = 1, 4, 8, 2
    mask = torch.ones(B, T, dtype=torch.bool)
    fields = {"pos3d": torch.randn(B, T, 3), "orient": torch.eye(3).reshape(-1)[None, None].expand(B, T, 9).clone(),
             "normal": torch.nn.functional.normalize(torch.randn(B, T, 3), dim=-1)}
    ts = TokenSet("ctx", mask, fields=fields)
    rc = RelCtx(sets={"ctx": ts}, generator=torch.Generator())
    given = resolve([{"name": "geo.pos3d", "source": "given"}, {"name": "geo.orient", "source": "given"},
                     {"name": "geo.normal_align", "source": "given"}])
    site = FactorSite(heads=H, dim=D, site="ctx>ctx", specs=given, carries=("pos3d", "orient", "normal", "hidden"))
    h = torch.randn(B, T, D)
    qa, ka = site.augment(rc, h, h)
    assert torch.allclose(qa, torch.zeros_like(qa))                        # g / w zero-init -> phi_q == 0 at step 0


# ------------------------------------------------------------------ correctness on a real fixture (unit R12 fields)
def test_geo_above_sign_matches_hand_computed_height_order():
    s, pi, b = _arm_batch()
    sets = relation_token_sets([pi], b, cameras=[(s.model, s.data, "front")])
    ctx = sets["ctx"]
    specs = resolve([{"name": "geo.above", "source": "given"}])
    site = FactorSite(heads=1, dim=4, site="ctx>ctx", specs=specs, carries=("pos3d",))
    rc = RelCtx(sets={"ctx": ctx}, generator=torch.Generator())
    from rrp.policies.relations.ops import OPS
    d = FACTORS["geo.above"]
    raw = OPS[d.op].value(d, specs[0], None, rc, "ctx>ctx")                # [1, C, C] in {-1, 0, +1}
    pos = ctx.field("pos3d")[0]
    v = ctx.field("pos3d.valid")[0]
    ii = torch.where(v)[0][:6]
    for i in ii:
        for j in ii:
            expected = 0.0
            dz = float(pos[j, 2] - pos[i, 2])
            if abs(dz) > 0.01:
                expected = math.copysign(1.0, dz)
            assert float(raw[0, i, j]) == pytest.approx(expected)


def test_geo_factors_resolve_and_run_on_the_arm_fixture_ctx_set():
    s, pi, b = _arm_batch()
    sets = relation_token_sets([pi], b, cameras=[(s.model, s.data, "front")])
    ctx = sets["ctx"]
    specs = resolve([{"name": "geo.pos3d", "source": "given"}, {"name": "geo.orient", "source": "given"},
                     {"name": "geo.above", "source": "given"}, "geo.depth3d"])   # depth3d stays source=probe (default)
    D, H = 8, 2
    site = FactorSite(heads=H, dim=D, site="ctx>ctx", specs=specs,
                      carries=("pos3d", "orient", "cam_uvd", "hidden"))
    assert {s.name for s in site.specs} == {"geo.pos3d", "geo.orient", "geo.above", "geo.depth3d"}
    rc = RelCtx(sets={"ctx": ctx}, generator=torch.Generator())
    fr = FieldReadouts(D, specs)
    h = torch.randn(1, ctx.mask.shape[1], D)
    fr.observe(0, "ctx", h, rc)                                            # populates cam_uvd for geo.depth3d
    bias = site.bias(rc)
    qa, ka = site.augment(rc, h, h)
    assert torch.isfinite(bias).all() and torch.isfinite(qa).all() and torch.isfinite(ka).all()
