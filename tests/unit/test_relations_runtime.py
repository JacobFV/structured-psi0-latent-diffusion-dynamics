"""F1 (docs/relations.md section 11): the relation runtime contract. A factor a net family claims runs in that net
(fields, pair estimates, labels, supervision); one it cannot run is refused by `resolve(family=...)` with a clear
error; labels never reach a deployable forward; checkpoints carry and check the factor structure hash; the coverage
JSON is generated from the catalog."""
import json

import pytest
import torch

from rrp.envs.mujoco.fixtures import make_pick_place_session
from rrp.harness.data import relgen
from rrp.policies.features.featurizer import featurizer_for
from rrp.policies.nets.batch import attach_cam_uvd, collate_inputs, ctx_geometry_fields, relation_token_sets
from rrp.policies.nets.checkpoint import load_checkpoint, save_checkpoint
from rrp.policies.nets.flow import FlowPolicy, PolicyConfig
from rrp.policies.relations import catalog
from rrp.policies.relations.base import (FACTORS, FAMILIES, EdgeSet, FactorDef, FactorError, PrivilegedInput, RelCtx,
                                         TokenSet, estimates_loss, register_factor, require_factors, resolve,
                                         stamp_versions)
from rrp.policies.relations.ops import FactorSite

torch.manual_seed(0)
relgen.load_families()


@pytest.fixture(scope="module")
def fixture():
    s = make_pick_place_session(seed=3)
    pi = featurizer_for(s)(s.observe())
    return s, pi


def _batch(fixture, cam=False):
    s, pi = fixture
    b = collate_inputs([pi, pi])
    if cam:
        g = ctx_geometry_fields(b)
        b.extra["ctx_fields"] = attach_cam_uvd(g["pos3d"], g["pos3d.valid"], [(s.model, s.data, "front")] * 2)
    return b


def _policy(factors, **kw):
    return FlowPolicy(PolicyConfig(width=16, heads=2, ctx_layers=1, blocks=1, horizon=2,
                                   factors=factors, **kw))


# ------------------------------------------------------------------ every claimed factor runs (no KeyError)
@pytest.fixture(scope="module")
def coverage():
    return relgen.coverage()


def test_every_factor_the_arm_family_claims_runs(fixture, coverage):
    claimed = {n: r["resolved_as"]["arm"] for n, r in coverage["factors"].items()
               if "arm" in r["resolved_as"] and not n.startswith("test.")}  # other tests register throwaway factors globally
    assert {"geo.pos3d", "ix.support", "ix.force_flow", "task.next_contact", "geo.depth3d"} <= set(claimed)
    b = _batch(fixture, cam=True)
    for name, fl in claimed.items():
        net = _policy(fl)
        cache = net.prepare(b)
        assert cache.rc is not None, name


def test_unsupported_factor_is_refused_clearly():
    with pytest.raises(FactorError, match="foothold_next"):
        resolve(["leg.foothold"], family="arm")                       # label the arm collate cannot attach
    with pytest.raises(FactorError, match="not filled"):
        resolve([{"name": "geo.normal_align", "source": "given"}], family="arm")   # `normal` is never filled
    with pytest.raises(FactorError, match="implements no readout"):
        resolve(["probe.legged.goal"], family="arm")
    with pytest.raises(FactorError, match="unknown net family"):
        resolve(["ix.support"], family="nope")
    resolve([{"name": "geo.normal_align", "source": "probe"}], family="arm")  # the probe source is runnable


def test_env_and_scene_part_checks():
    with pytest.raises(FactorError, match="needs StateView caps"):
        resolve(["ix.contact"], family="arm", env_caps={"poses"}, training=True)
    with pytest.raises(FactorError, match="nothing of"):
        resolve([{"name": "geo.pos3d", "source": "probe", "mix": 0.5}], family="arm", env_caps={"poses"},
                env="computerworld", training=True)
    resolve([{"name": "geo.pos3d", "source": "probe", "mix": 0.5}], family="arm", env_caps={"poses"},
            env="mujoco/arm", training=True)


def test_unknown_relgen_names_raise():
    for fn in (relgen.label_def, relgen.part_def, relgen.transform_def):
        with pytest.raises(ValueError, match="unknown relgen"):
            fn("no-such-name")
    with pytest.raises(FactorError, match="probe\\.arm\\.\\*|implements no"):
        resolve(["probe.psi0.lift"], family="arm")


def test_pointer_carries_come_from_the_family():
    from rrp.policies import pointer
    from rrp.policies.nets import flow
    assert pointer.UI_CARRIES == FAMILIES["pointer"].sites["ctx>ctx"]
    assert flow.CTX_CARRIES == FAMILIES["arm"].sites["ctx>ctx"] and flow.ACT_CARRIES == FAMILIES["arm"].sites["act>ctx"]


# ------------------------------------------------------------------ provenance: labels are privileged
def test_labels_refused_in_deploy_and_validated(fixture):
    b = _batch(fixture)
    C = b.ctx_mask.shape[1]
    lab = {"ctx": {"pos3d": torch.zeros(2, C, 3), "pos3d.valid": b.ctx_mask}}
    with pytest.raises(PrivilegedInput):
        relation_token_sets("arm", b, lab, deploy=True)
    with pytest.raises(FactorError, match="not one family"):
        relation_token_sets("arm", b, {"ctx": {"bogus": torch.zeros(2, C, 1)}})
    ts = relation_token_sets("arm", b, lab)["ctx"]
    assert ts.label("pos3d").shape == (2, C, 3)
    net = _policy([{"name": "geo.pos3d", "source": "probe"}]).set_deploy(True)
    b.extra["relation_labels"] = lab
    with pytest.raises(PrivilegedInput):
        net.prepare(b)
    with pytest.raises(PrivilegedInput):
        rc = RelCtx(sets={"ctx": TokenSet("ctx", b.ctx_mask)}, deploy=True)
        estimates_loss(rc, [])


# ------------------------------------------------------------------ the estimates learn (50 CPU steps)
def _support_label(b):
    pos = ctx_geometry_fields(b)
    z, v = pos["pos3d"][..., 2], pos["pos3d.valid"]
    y = ((z[:, :, None] - z[:, None, :]) > 0.05) & v[:, :, None] & v[:, None, :]
    return y.float(), v[:, :, None] & v[:, None, :]


def test_probe_field_and_pair_estimates_learn(fixture):
    b = _batch(fixture)
    g = ctx_geometry_fields(b)
    y, ok = _support_label(b)
    assert y.sum() > 0
    b.extra["relation_labels"] = {"ctx": {"pos3d": g["pos3d"], "pos3d.valid": g["pos3d.valid"],
                                          "support_pairs": y[..., None], "support_pairs.valid": ok}}
    net = _policy([{"name": "geo.pos3d", "source": "probe"}, "ix.support"])
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    N = b.node_mask.shape[1]
    tgt = torch.zeros(2, 2, N, 1)
    valid = b.node_mask[:, None, :].expand(2, 2, N)
    first, last = [], []
    for step in range(50):
        _, logs = net.loss(b, tgt, valid)
        cache = net.prepare(b)
        el, elogs, metrics = estimates_loss(cache.rc, net.factor_specs())
        opt.zero_grad()
        el.backward()
        opt.step()
        (first if step < 5 else last if step >= 45 else []).append(elogs)
    for k in ("probe_geo.pos3d", "probe_ix.support"):
        assert k in logs, k
        f, l = sum(e[k] for e in first) / 5, sum(e[k] for e in last) / 5
        assert l < f - 0.05, (k, f, l)
    hit, n = metrics["ix.support_acc"]
    assert n > 0 and 0 <= hit <= n


def test_missing_labels_mean_no_term(fixture):
    net = _policy([{"name": "geo.pos3d", "source": "probe"}, "ix.support"])
    cache = net.prepare(_batch(fixture))
    loss, logs, metrics = estimates_loss(cache.rc, net.factor_specs())
    assert float(loss) == 0.0 and logs == {} and metrics["ix.support_acc"][1] == 0


# ------------------------------------------------------------------ checkpoints carry the structure hash
def test_stamp_and_require_factors_round_trip(tmp_path, fixture):
    net = _policy([{"name": "geo.pos3d", "source": "probe"}])
    p = tmp_path / "m.pt"
    meta = save_checkpoint(p, model=net, step=1, versions={"x": "1"}, config={})
    specs = net.factor_specs()
    assert meta["versions"]["factors"] == stamp_versions({}, specs)["factors"]
    load_checkpoint(p, specs=specs)
    other = resolve(["ix.support"], family="arm")
    with pytest.raises(FactorError, match="factor structure"):
        load_checkpoint(p, specs=other)
    load_checkpoint(p, specs=other, allow_factor_mismatch=True)
    with pytest.raises(FactorError):
        require_factors({}, specs)                                     # a checkpoint without a hash is refused


# ------------------------------------------------------------------ coverage JSON is generated from the catalog
def test_coverage_json_matches_catalog(tmp_path):
    out = tmp_path / "coverage.json"
    assert relgen.coverage_main(["coverage", "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    impl = {n for n, d in FACTORS.items() if d.status == "implemented"}
    assert set(doc["factors"]) == impl and set(doc["families"]) == set(FAMILIES)
    row = doc["factors"]["ix.contact"]
    assert row["families"]["arm"] is True and row["envs"]["mujoco/arm"]["label_runnable"] is True
    assert row["envs"]["computerworld"]["label_runnable"] is False       # no `contacts` cap there
    assert doc["factors"]["ix.handover"]["families"]["arm"] != True and doc["factors"]["ix.handover"]["families"]["dual"] is True
    assert doc["factors"]["task.next_contact"]["envs"]["mujoco/arm"]["parts"] == {"reveal": True, "surprise": True}
    assert doc["factors"]["probe.arm.visible"]["label"] == "visible" and \
        doc["factors"]["probe.arm.visible"]["envs"]["mujoco/arm"]["label_runnable"] is None   # family-level label
    assert doc["factors"]["ix.contact"]["training"]["arm"]["computerworld"] != True


# ------------------------------------------------------------------ review additions (architect pass)
def test_every_family_resolves_its_own_presets():
    assert set(FAMILIES) == {"arm", "dual", "legged", "humanoid", "psi0", "pointer"}       # docs section 11
    own = {"arm": ("arm", "s0-arm", "probes:arm-packet-v1"), "dual": ("arm", "s0-arm"),
           "legged": ("legged-s0", "probes:legged-v1"), "humanoid": ("legged-s0", "probes:legged-v1"),
           "psi0": ("psi0-dims", "s0-psi0", "probes:psi0-v1"), "pointer": ("ui", "probes:pointer-v1")}
    for fam, presets in own.items():
        for p in presets:
            resolve([f"preset:{p}"], family=fam)


def test_token_sets_are_built_on_the_batch_device(fixture):
    """The field builders run inside the forward (not at collate time), so every tensor they create must live on
    the batch's device. The `meta` device stands in for a GPU: mixing it with a CPU tensor raises."""
    b = _batch(fixture).to("meta")
    sets = relation_token_sets("arm", b)
    for ts in sets.values():
        for v in [ts.mask, ts.kind, *ts.fields.values()]:
            assert v is None or v.device.type == "meta"
    assert {"pos3d", "orient", "entity_id", "assembly_id"} <= set(sets["ctx"].fields)


def test_image_pad_pads_fields_and_pair_labels(fixture):
    b = _batch(fixture)
    C = b.ctx_mask.shape[1]
    y, ok = _support_label(b)
    lab = {"ctx": {"pos3d": torch.zeros(2, C, 3), "pos3d.valid": b.ctx_mask,
                   "support_pairs": y[..., None], "support_pairs.valid": ok}}
    ts = relation_token_sets("arm", b, lab, pad_ctx=3)["ctx"]
    assert ts.mask.shape == (2, C + 3) and not ts.mask[:, C:].any() and ts.kind.shape == (2, C + 3)
    assert ts.fields["pos3d"].shape == (2, C + 3, 3) and ts.fields["pos3d.valid"].shape == (2, C + 3)
    assert ts.labels["pos3d"].shape == (2, C + 3, 3)
    assert ts.labels["support_pairs"].shape == (2, C + 3, C + 3, 1)
    assert ts.labels["support_pairs.valid"].shape == (2, C + 3, C + 3) and not ts.labels["support_pairs.valid"][:, C:].any()


def test_coverage_env_caps_match_the_envs():
    """`relgen._env_caps` repeats the caps the non-MuJoCo envs set in heavy constructors: keep them equal."""
    from pathlib import Path
    import rrp.envs as envs
    root = Path(envs.__file__).parent
    caps = relgen._env_caps()
    for env, rel in (("warp/legged", "warp/tracker_env.py"), ("simple", "simple/__init__.py"),
                     ("computerworld", "computerworld.py")):
        lit = "frozenset({" + ", ".join(f'"{c}"' for c in (("poses", "contacts") if env != "computerworld"
                                                          else ("poses", "ui_tree"))) + "})"
        assert lit in (root / rel).read_text(), (env, lit)
        assert caps[env] == frozenset(eval(lit))


# ------------------------------------------------------------------ RC (round 2, D-146): registry closure
# 1. field operators honour `<field>.valid`: an invalid token contributes zero on both sides (only on the side that
#    reads a field: a unary / world-frame align query reads none), whatever garbage its field holds.
register_factor(FactorDef("test.rc_pape", "1", field="pos3d", op="sqdiff+diff", form="aug", params=(("p", 3),)))
register_factor(FactorDef("test.rc_pape_q", "1", field="pos3d", op="sqdiff+diff", form="aug",
                          params=(("p", 3), ("frame", "query"), ("orient", "orient"))))
register_factor(FactorDef("test.rc_diff", "1", field="pos3d", op="diff", form="aug", params=(("p", 3),)))
register_factor(FactorDef("test.rc_rot", "1", field="orient", op="rel_rot", form="aug"))
register_factor(FactorDef("test.rc_align", "1", field="normal", op="align", form="aug", params=(("frame", "world"),)))
register_factor(FactorDef("test.rc_align_q", "1", field="normal", op="align", form="aug",
                          params=(("frame", "query"), ("orient", "orient"))))
register_factor(FactorDef("test.rc_sim", "1", field="pos3d", op="sim", form="aug"))
register_factor(FactorDef("test.rc_unary", "1", field="pos3d", op="unary", form="aug", params=(("p", 3),)))
register_factor(FactorDef("test.rc_same", "1", field="entity_id", op="same", form="aug", params=(("n_ids", 8),)))
register_factor(FactorDef("test.rc_same_b", "1", field="entity_id", op="same", form="bias"))
register_factor(FactorDef("test.rc_order", "1", field="pos3d", op="order", form="bias"))

_FIELDS = ("pos3d", "orient", "normal", "entity_id")


def _half_invalid_sets(B=2, T=6, garbage=False):
    g = torch.Generator().manual_seed(0)
    sets = {}
    for n, off in (("q", 0), ("k", 1)):
        valid = (torch.arange(T) % 2 == off)[None].expand(B, T).clone()             # half of the tokens are invalid
        rot = torch.linalg.qr(torch.randn(B, T, 3, 3, generator=g))[0].flatten(-2)
        f = {"pos3d": torch.randn(B, T, 3, generator=g), "orient": rot,
             "normal": torch.nn.functional.normalize(torch.randn(B, T, 3, generator=g), dim=-1),
             "entity_id": torch.randint(0, 8, (B, T, 1), generator=g)}
        if garbage:                                                                 # invalid tokens hold other values
            gg = torch.Generator().manual_seed(9)
            f = {k: torch.where(valid.reshape(B, T, *[1] * (v.dim() - 2)), v,
                                (torch.randn(v.shape, generator=gg) * 50).to(v.dtype).abs() + 1 if v.is_floating_point()
                                else (v + 3) % 8) for k, v in f.items()}
        f.update({k + ".valid": valid for k in _FIELDS})
        sets[n] = TokenSet(n, torch.ones(B, T, dtype=torch.bool), fields=f)
    return RelCtx(sets=sets), sets["q"].fields["pos3d.valid"], sets["k"].fields["pos3d.valid"]


def _pair_score(name, garbage):
    rc, vq, vk = _half_invalid_sets(garbage=garbage)
    site = FactorSite(2, 16, "q>k", resolve([name]), _FIELDS)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for p in site.parameters():
            p.copy_(torch.randn(p.shape, generator=g))
    x = torch.randn(2, 6, 16, generator=torch.Generator().manual_seed(5))
    if FACTORS[name].form == "bias":
        return site.bias(rc), vq, vk
    qa, ka = site.augment(rc, x, x)
    return qa @ ka.transpose(-1, -2), vq, vk


# (factor, does an invalid QUERY token contribute zero, does an invalid KEY token)
_VALID_CASES = [("test.rc_pape", True, True), ("test.rc_pape_q", True, True), ("test.rc_diff", True, True),
                ("test.rc_rot", True, True), ("test.rc_align", False, True), ("test.rc_align_q", True, True),
                ("test.rc_sim", True, True), ("test.rc_unary", False, True), ("test.rc_same", True, True),
                ("test.rc_same_b", True, True), ("test.rc_order", True, True)]


@pytest.mark.parametrize("name,q_zero,k_zero", _VALID_CASES)
def test_field_operators_honour_the_valid_mask(name, q_zero, k_zero):
    s, vq, vk = _pair_score(name, garbage=False)
    assert s.abs().sum() > 0                                                       # the half-valid set is not all zero
    if q_zero:
        assert s.masked_select(~vq[:, None, :, None].expand_as(s)).abs().max() == 0
    if k_zero:
        assert s.masked_select(~vk[:, None, None, :].expand_as(s)).abs().max() == 0
    # an invalid token's field value never reaches the output: scrambling it changes nothing
    s2, _, _ = _pair_score(name, garbage=True)
    assert torch.allclose(s, s2, atol=1e-5)


def test_valid_mask_is_a_noop_when_every_token_is_valid():
    rc, vq, vk = _half_invalid_sets()
    for ts in rc.sets.values():
        ts.fields["pos3d.valid"] = torch.ones_like(ts.fields["pos3d.valid"])
    site = FactorSite(2, 16, "q>k", resolve(["test.rc_order"]), _FIELDS)
    site.f["test__rc_order"].w.data.fill_(1.0)
    full = site.bias(rc)
    for ts in rc.sets.values():
        del ts.fields["pos3d.valid"]                                               # no mask declared: all valid
    assert torch.equal(full, site.bias(rc))


# 2. catalog truth
def test_catalog_truth_round_two():
    assert "kin.mirror" not in FACTORS
    assert FACTORS["time.same_track"].status == "planned"
    with pytest.raises(FactorError):
        resolve(["time.same_track"])                                              # planned factors do not resolve
    for fam in ("arm", "dual"):
        assert [s.name for s in resolve(["id.same_body"], family=fam)] == ["id.same_body"]
    assert "entity_id" in FAMILIES["arm"].sites["ctx>ctx"] and "entity_id" in FAMILIES["dual"].sites["ctx>ctx"]


def test_ui_factors_name_the_computerworld_scene_parts(coverage):
    for n in ("ui.label_for", "ui.focus_next", "ui.same_window", "ui.above", "ui.drag_to"):
        assert {"cw_viewport", "cw_depth"} <= set(FACTORS[n].gen), n
        assert coverage["factors"][n]["envs"]["computerworld"]["parts"].get("cw_viewport") is True, n
        assert coverage["factors"][n]["envs"]["computerworld"]["parts"].get("cw_depth") is True, n
    assert FACTORS["ui.label_for"].gen[:2] == ("reveal", "surprise")


def test_every_implemented_factor_is_covered_in_some_family(coverage):
    uncovered = [n for n, r in coverage["factors"].items() if not n.startswith("test.") and not any(v is True for v in r["families"].values())]
    assert uncovered == []


# 3. the `@gt` EdgeSet `ix.force_flow` reads
def test_support_pairs_label_attaches_the_privileged_edgeset(fixture):
    b = _batch(fixture)
    g = ctx_geometry_fields(b)
    y, ok = _support_label(b)
    lab = {"ctx": {"support_pairs": y[..., None], "support_pairs.valid": ok}}
    edges = {}
    relation_token_sets("arm", b, lab, edges=edges, pad_ctx=2)
    es = edges["ctx>ctx#support-v1@gt"]
    C = b.ctx_mask.shape[1]
    assert es.prov == "privileged" and es.vocab == ("support",) and es.data.shape == (2, C + 2, C + 2, 1)
    assert torch.equal(es.data[:, :C, :C, 0], y * ok) and es.data[:, C:].abs().sum() == 0
    none = {}
    relation_token_sets("arm", b, None, edges=none)
    relation_token_sets("arm", b, {"ctx": {"pos3d": g["pos3d"], "pos3d.valid": g["pos3d.valid"]}}, edges=none)
    assert none == {}                                                              # no support label, no edge
    with pytest.raises(PrivilegedInput):
        relation_token_sets("arm", b, lab, deploy=True, edges={})


def test_force_flow_reads_the_gt_edges_a_net_builds(fixture):
    b = _batch(fixture)
    y, ok = _support_label(b)
    b.extra["relation_labels"] = {"ctx": {"support_pairs": y[..., None], "support_pairs.valid": ok}}
    net = _policy(["ix.support", {"name": "ix.force_flow", "source": "gt"}])
    cache = net.prepare(b)
    es = cache.rc.edges["ctx>ctx#support-v1@gt"]
    assert es.prov == "privileged" and es.data.sum() > 0
    with pytest.raises(PrivilegedInput):                                           # a gt source is never deployable
        net.set_deploy(True)


# 4. the dual family
def test_policy_config_family_round_trips_for_dual():
    cfg = PolicyConfig.from_dict({"width": 16, "heads": 2, "family": "dual", "factors": ["id.same_body"]})
    assert cfg.family == "dual" and [s.name for s in cfg.specs()] == ["id.same_body"]
    assert _policy(["id.same_body"], family="dual").cfg.family == "dual"
