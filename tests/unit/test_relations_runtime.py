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
from rrp.policies.relations.base import (FACTORS, FAMILIES, FactorError, PrivilegedInput, RelCtx, TokenSet,
                                         estimates_loss, require_factors, resolve, stamp_versions)

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
    return FlowPolicy(PolicyConfig(width=16, heads=2, ctx_layers=1, blocks=1, horizon=2, aux=False,
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
