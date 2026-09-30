"""HP (readiness round 2): the legged policy is built from its checkpoint's own factor stamp, refuses `upper=True` unless the
checkpoint declares `upper_trained`, and is fed the public terrain scan at deploy when its factors read it. Tiny random
weights (plumbing only: no number from them is a result)."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rrp.policies.relations.base import FactorError, compat_hash
from rrp.policies import legged as LG

from ._legged_tiny import DZ, tiny_bundle

CPU = torch.device("cpu")
REL = ["preset:probes:legged-v1", "preset:legged-r19"]       # probes + leg.foothold + leg.com_support (terrain-reading)
NONE = ["preset:probes:legged-v1"]                           # stamped, no relational factor


def _ctl(flow, **kw):
    return LG.LatentLeggedController(flow, CPU, nfe=2, seed=0, **kw)


# ------------------------------------------------------------------ upper=True only on an upper_trained checkpoint
def test_upper_is_refused_on_a_legs_only_checkpoint(tmp_path):
    _, flow = tiny_bundle(tmp_path)                                   # bare legacy legs-only checkpoint
    with pytest.raises(ValueError, match=r"upper=True.*upper_trained"):
        _ctl(flow, upper=True)
    assert _ctl(flow).upper is False                                   # the legs-only route still loads


def test_upper_is_refused_on_a_stamped_legs_only_checkpoint(tmp_path):
    _, flow = tiny_bundle(tmp_path, factors=REL, upper_trained=False)
    with pytest.raises(ValueError, match="upper_trained"):
        _ctl(flow, upper=True)


def test_upper_is_accepted_on_an_upper_trained_checkpoint(tmp_path):
    _, flow = tiny_bundle(tmp_path, factors=REL, upper_trained=True)
    pol = LG.LeggedLatentPolicy(_ctl(flow, upper=True))
    assert pol.ctl.upper and pol.info.requires.groups == {"legs", "upper"} and pol.controls == ("wholebody",)


def test_refit_realizer_without_the_declaration_refuses_upper_even_on_an_upper_trained_rep(tmp_path):
    """The realizer that acts is the one whose declaration counts: a refit R (legs-only DAgger labels) is not upper-trained."""
    rep, _ = tiny_bundle(tmp_path, factors=REL, upper_trained=True)
    from rrp.harness.train.legged_latent_train import _save
    from rrp.policies.nets.legged_latent import LeggedRealizer, legged_specs
    specs = legged_specs(REL)
    st = torch.load(rep, weights_only=False)
    rz = tmp_path / "rz" / "realizer.pt"
    rz.parent.mkdir()
    _save(rz, R=st["R"], cfg=st["cfg"], step=1, representation=str(rep), specs=specs,
          result=dict(upper_trained=False, action_groups=["legs"]))
    with pytest.raises(ValueError, match="upper_trained"):
        LG.LatentLeggedController(None, CPU, rep=str(rep), realizer=str(rz), upper=True)
    assert LG.LatentLeggedController(None, CPU, rep=str(rep), realizer=str(rz)).upper is False


# ------------------------------------------------------------------ nets come from the stamp; a mismatch is refused
def test_nets_are_built_from_the_checkpoint_factor_stamp(tmp_path):
    rep, flow = tiny_bundle(tmp_path, factors=REL)
    ctl = _ctl(flow, rep=None)
    from rrp.policies.nets.legged_latent import legged_specs, relational_specs
    assert compat_hash(ctl.specs) == compat_hash(legged_specs(REL)) and relational_specs(ctl.specs)
    assert ctl.F.specs is not None and compat_hash(ctl.F.specs) == compat_hash(ctl.specs)
    assert sum(p.numel() for p in ctl.E.parameters()) > sum(                 # the relational rows exist (legged-none has none)
        p.numel() for p in _ctl(tiny_bundle(tmp_path / "n", factors=NONE)[1]).E.parameters())


def test_stamp_mismatch_is_refused(tmp_path):
    rep, flow = tiny_bundle(tmp_path, factors=REL)
    st = torch.load(rep, weights_only=False)
    st["factors"] = st["factors"][:-1]                                        # the list no longer matches the saved stamp
    bad = tmp_path / "bad_rep.pt"
    torch.save(st, bad)
    with pytest.raises(Exception, match="factor|fingerprint|stamp|fx-"):
        LG.load_legged_rep(bad, CPU)


def test_flow_on_another_factor_structure_is_refused(tmp_path):
    rep, _ = tiny_bundle(tmp_path / "a", factors=REL)
    _, other_flow = tiny_bundle(tmp_path / "b", factors=NONE)
    _, _, _, _, _, specs = LG.load_legged_rep(rep, CPU)
    with pytest.raises(FactorError, match="flow factor structure"):
        LG.load_legged_flow(other_flow, CPU, specs, DZ)
    with pytest.raises(FactorError, match="flow factor structure"):            # an unstamped flow on a relational rep
        LG.load_legged_flow(tiny_bundle(tmp_path / "c")[1], CPU, specs, DZ)


# ------------------------------------------------------------------ the public scan is required and fed
def test_terrain_need_follows_the_factors_and_is_a_requirement(tmp_path):
    _, f_rel = tiny_bundle(tmp_path / "r", factors=REL)
    _, f_none = tiny_bundle(tmp_path / "n", factors=NONE)
    _, f_legacy = tiny_bundle(tmp_path / "l")
    p_rel, p_none, p_leg = (LG.LeggedLatentPolicy(_ctl(f)) for f in (f_rel, f_none, f_legacy))
    assert p_rel.ctl.needs_terrain and p_rel.info.requires.env_capabilities == {"terrain_scan"}
    assert not p_none.ctl.needs_terrain and not p_none.info.requires.env_capabilities
    assert not p_leg.ctl.needs_terrain and not p_leg.info.requires.env_capabilities


def _stub_policy(flow):
    pol = LG.LeggedLatentPolicy(_ctl(flow))
    calls = []
    ad = SimpleNamespace(stats=dict(packets=0), upper_tgt=None, packet=None, terrain=None,
                         act=lambda data, cmd: calls.append(1) or np.zeros(12))
    pol.env = SimpleNamespace(data=None, controller_version=lambda: "c1")
    pol.ad = ad
    return pol, ad, calls


def test_missing_scan_channel_is_a_hard_error_not_a_swallowed_packet_rejection(tmp_path):
    _, flow = tiny_bundle(tmp_path, factors=REL)
    pol, ad, calls = _stub_policy(flow)
    with pytest.raises(RuntimeError, match="0:terrain_scan"):
        pol.act({0: SimpleNamespace(declared_sensor_channels=[])})
    assert not calls


def test_scan_channel_reaches_the_deployed_batch(tmp_path, monkeypatch):
    _, flow = tiny_bundle(tmp_path, factors=REL)
    pol, ad, calls = _stub_policy(flow)
    vals, mask = np.linspace(0, 0.2, 77).astype(np.float32), np.ones(77, bool)
    mask[3] = False
    ch = SimpleNamespace(name="0:terrain_scan", values=vals, mask=mask)
    pol.act({0: SimpleNamespace(declared_sensor_channels=[ch])})
    assert calls and np.array_equal(ad.terrain[0], vals) and np.array_equal(ad.terrain[1], mask)

    monkeypatch.setattr(LG, "local_state", lambda s, m: (np.zeros(3, np.float32), np.zeros(3, np.float32),
                                                         np.zeros(6, np.float32), np.zeros(2, np.float32)))
    real = LG.System0Adapter(SimpleNamespace(dev=CPU, static={}, needs_terrain=True, lsv="x"),
                             SimpleNamespace(), SimpleNamespace(gait_period=0.5))
    with pytest.raises(RuntimeError, match="terrain"):                         # no scan observed yet: never a silent zero scan
        real.dyn_batch()
    real.terrain = ad.terrain
    b = real.dyn_batch()
    assert b["terrain"].shape == (1, 77) and b["terrain_valid"].shape == (1, 77) and b["terrain_valid"].dtype == torch.bool
    assert not bool(b["terrain_valid"][0, 3]) and float(b["terrain"][0, 76]) == pytest.approx(0.2)
    real.c.needs_terrain = False                                               # a scan-free checkpoint's batch is unchanged
    assert "terrain" not in real.dyn_batch()


def test_legs_only_policy_without_terrain_factors_ignores_the_channel(tmp_path):
    _, flow = tiny_bundle(tmp_path, factors=NONE)
    pol, ad, calls = _stub_policy(flow)
    pol.act({0: SimpleNamespace(declared_sensor_channels=[])})                # no scan needed, none required
    assert calls and ad.terrain is None
