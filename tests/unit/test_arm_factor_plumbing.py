"""Readiness A1: per-site factor lists on the arm encoder / realizer, the resolved hash in the bundle checkpoint,
shared factors across compared methods, `SemanticReadout` / `aux` removed (D-146 item 4), one SFT budget table."""
import inspect
from pathlib import Path

import pytest
import torch

from rrp.policies.bundles import assert_shared_factors, bundle_factor_specs, load_representation
from rrp.policies.nets.checkpoint import load_checkpoint, save_checkpoint
from rrp.policies.nets.flow import FlowPolicy, PolicyConfig
from rrp.policies.nets.probes import ReadoutProbe
from rrp.policies.nets.semantic_latent import LatentConfig, TargetEncoder
from rrp.policies.relations.base import FactorError, compat_hash, require_factors
from rrp.policies.system0 import LatentRealizer, make_realizer

# structure hashes ignore controls, so a different list (not an `off`) is what moves the hash
NO_ROUTE = []
ENC = ["edge.same_node"]


def tiny_cfg(**kw):
    return LatentConfig(width=32, heads=2, ctx_layers=1, enc_layers=1, knots=2, dz=4, horizon=4, realizer_layers=1, **kw)


# ---------------------------------------------------------------- SemanticReadout / aux removed
def test_semantic_readout_and_aux_are_gone():
    import rrp.policies.nets.flow as flow
    assert not hasattr(flow, "SemanticReadout")
    assert "aux" not in PolicyConfig.__dataclass_fields__
    with pytest.raises(TypeError):
        PolicyConfig(aux=True)
    m = FlowPolicy(PolicyConfig(width=32, heads=2, ctx_layers=1, blocks=1, horizon=4))
    assert not hasattr(m, "readout") and not any(k.startswith("readout.") for k in m.state_dict())
    assert list(inspect.signature(FlowPolicy.loss).parameters)[:4] == ["self", "batch", "target", "valid"]
    assert "labels" not in inspect.signature(FlowPolicy.loss).parameters
    assert "aux_weight" not in inspect.signature(FlowPolicy.loss).parameters


def test_recorded_aux_config_and_readout_weights_still_load():
    """On-disk data (D-144 b): the kept BC experts recorded `aux: true` and carry `readout.*` tensors; both are dropped."""
    pc = PolicyConfig.from_dict(dict(width=32, heads=2, ctx_layers=1, blocks=1, horizon=4, aux=True))
    assert "aux" not in pc.to_dict() if hasattr(pc, "to_dict") else True
    m = FlowPolicy(pc)
    sd = dict(m.state_dict(), **{"readout.held.weight": torch.zeros(3, 32), "readout.q.bias": torch.zeros(32)})
    FlowPolicy(pc).load_state_dict(sd, strict=True)


def test_baseline_source_config_carries_no_aux():
    from rrp.harness.train.baseline_campaign import source_config
    cfg = source_config("baseline_direct_action", 0, Path("/tmp/x"))
    assert "aux" not in cfg["policy"] and "aux_weight" not in cfg


# ---------------------------------------------------------------- per-site factor lists
def test_latent_version_unchanged_at_default_and_moved_by_site_lists():
    base = tiny_cfg().version()
    assert tiny_cfg(encoder_factors=None, realizer_factors=None).version() == base
    assert tiny_cfg(realizer_factors=NO_ROUTE).version() != base
    assert tiny_cfg(encoder_factors=ENC).version() != base


def test_encoder_and_realizer_take_their_own_factor_lists():
    assert compat_hash(TargetEncoder(tiny_cfg(encoder_factors=ENC)).factor_specs()) != \
        compat_hash(TargetEncoder(tiny_cfg()).factor_specs())
    r0, r1 = make_realizer(4, 1), make_realizer(4, 1, factors=NO_ROUTE)
    assert compat_hash(r0.factor_specs()) != compat_hash(r1.factor_specs())
    assert [s.name for s in r0.factor_specs()] == ["route.own_assembly"]
    assert r1.factor_specs() == ()
    assert isinstance(LatentRealizer(4, width=32, layers=1).factor_specs(), tuple)


def _save(tmp_path, cfg):
    from rrp.harness.train.latent_train import _bundle
    E, R = TargetEncoder(cfg), make_realizer(cfg.dz, cfg.realizer_layers, factors=cfg.realizer_factors)
    P = ReadoutProbe(cfg.dz, cfg.knots, specs=["preset:probes:arm-packet-v1"])
    b = _bundle(E, R, P)
    lv = cfg.version()
    save_checkpoint(tmp_path / "rep.pt", model=b, optimizer=None, step=1, versions=dict(latent=lv),
                    config=dict(latent={f: getattr(cfg, f) for f in ("width", "heads", "ctx_layers", "enc_layers", "knots",
                                                                     "dz", "horizon", "realizer_layers", "realizer_factors",
                                                                     "encoder_factors")}, probe={}),
                    extra=dict(result=dict(latent_space_version=lv)))
    return b


def test_bundle_checkpoint_stores_the_resolved_hash_and_load_verifies_it(tmp_path):
    cfg = tiny_cfg(realizer_factors=NO_ROUTE)
    b = _save(tmp_path, cfg)
    st = load_checkpoint(tmp_path / "rep.pt")
    assert st["versions"]["factors"].split("+")[0] == compat_hash(b.factor_specs())
    cfg2, E, R, P, res = load_representation(tmp_path / "rep.pt", "cpu")           # verified round trip
    assert compat_hash(bundle_factor_specs(E, R, P)) == compat_hash(b.factor_specs())
    assert R.factor_specs() == ()
    with pytest.raises(FactorError):                                               # different realizer list, same weights
        require_factors(st["versions"], bundle_factor_specs(E, make_realizer(4, 1), P))


def test_load_representation_refuses_a_config_whose_factors_moved(tmp_path):
    _save(tmp_path, tiny_cfg(realizer_factors=NO_ROUTE))
    st = torch.load(tmp_path / "rep.pt", map_location="cpu", weights_only=False)
    st["config"]["latent"]["realizer_factors"] = None          # config edited after the stamp
    torch.save(st, tmp_path / "rep2.pt")
    with pytest.raises(Exception):
        load_representation(tmp_path / "rep2.pt", "cpu")


# ---------------------------------------------------------------- shared factors, one SFT table
def test_compared_methods_must_share_a_factor_set():
    a = PolicyConfig(width=32, heads=2, ctx_layers=1, blocks=1, horizon=4).specs()
    b = PolicyConfig(width=32, heads=2, ctx_layers=1, blocks=1, horizon=4,
                     factors=["preset:arm", {"name": "msg.incidence", "control": "off"}]).specs()
    assert assert_shared_factors({"m1": a, "m2": a}) == compat_hash(a)
    with pytest.raises(FactorError, match="m2"):   # an `off` control is a different arm even at the same structure hash
        assert_shared_factors({"m1": a, "m2": b})


def test_baseline_arms_share_policy_factors():
    from rrp.harness.train.baseline_campaign import assert_shared_policy_factors
    assert assert_shared_policy_factors(Path("/tmp/x")).startswith("fx-")


def test_one_sft_budget_table():
    from rrp.harness.pipelines import arm
    from rrp.harness.train import baseline_campaign as bc, latent_campaign
    assert arm.SFT_STEPS is bc.SFT_STEPS and arm.SFT_LR is bc.SFT_LR
    assert "150, 20: 300" not in inspect.getsource(latent_campaign)
