"""D-144 R1 acceptance (docs/relations.md section 10; brief in section 10 "briefs"): arm/dual probes migrate from
`nets.latent_probes.PacketProbe` to `nets.probes.ReadoutProbe` configured by the registry preset
`probes:arm-packet-v1` (+ `probe.arm.goal_effect` when the legacy config had `goal_effect=True`), at every R1-owned
construction site (`policies.bundles.load_representation`, `harness.train.latent_train`, `harness.eval.latent_causal`,
`cli.latent`, `cli.dual_latent`).

Row acceptance, each with a red/green pair (asserted false against the pre-migration shape, true after):
  1. goldens unchanged -- proved by leaving `nets/latent_probes.py` (PacketProbe itself, `probe_loss`, `probe_metrics`)
     untouched: tests/unit/test_golden.py still builds PacketProbe directly and its byte-identical hashes are
     unaffected by this unit (see research/tracks/rel-r1.md for why the file could not be deleted).
  2. old PacketProbe state dicts load strictly into ReadoutProbe (incl. pre-rename `desired_delta` checkpoints).
  3. `probes:arm-packet-v1` (ReadoutProbe) metrics equal the legacy `probe_metrics` on a fixture batch.
Also: the `desired_delta` OUTPUT alias is dropped (R1 brief) while the `desired_delta_err_m` METRIC key name stays
(hooks must not change what they report).
"""
import pytest
import torch

from rrp.policies.bundles import _readout_probe_specs, _remap_probe_state_dict
from rrp.policies.nets.latent_probes import PacketProbe, probe_loss, probe_metrics
from rrp.policies.nets.probes import ReadoutProbe


def _fixture(goal=False, B=2, K=3, M=2, S=4, dz=8, seed=7):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, K, M, dz, generator=g)
    zm = torch.ones(B, M, dtype=torch.bool)
    lab = dict(held=torch.zeros(B, S, dtype=torch.bool), contact=torch.zeros(B, S, dtype=torch.bool),
               visible=torch.ones(B, S, dtype=torch.bool), focus=torch.zeros(B, S, dtype=torch.bool),
               gaze=torch.rand(B, S, generator=g) * 30, rel_tcp=torch.randn(B, S, 3, generator=g),
               future_disp=torch.randn(B, S, 3, generator=g), subtask=torch.zeros(B, dtype=torch.long))
    if goal:
        lab["goal_effect"] = torch.randn(B, S, 3, generator=g)
    smask = torch.ones(B, S, dtype=torch.bool)
    return z, zm, lab, smask, S


def _pair(goal: bool):
    torch.manual_seed(0)
    old = PacketProbe(dz=8, knots=3, width=32, heads=2, goal_effect=goal)
    specs, kw = _readout_probe_specs(dict(goal_effect=goal))
    torch.manual_seed(0)
    new = ReadoutProbe(dz=8, knots=3, width=32, heads=2, specs=specs, **kw)
    return old, new


# ------------------------------------------------------------------ spec / state-dict translation (bundles.py)
def test_readout_probe_specs_translate_goal_effect_and_drop_legacy_kwargs():
    specs, kw = _readout_probe_specs(dict(width=32, heads=2, goal_effect=True, n_operators=12))
    assert specs == ["preset:probes:arm-packet-v1", "probe.arm.goal_effect"]
    assert kw == dict(width=32, heads=2)                  # goal_effect / n_operators consumed, not passed through
    specs2, kw2 = _readout_probe_specs({})
    assert specs2 == ["preset:probes:arm-packet-v1"] and kw2 == {}


def test_remap_probe_state_dict_renames_pre_rename_keys_only():
    sd = {"heads.desired_delta.weight": torch.zeros(1), "heads.visible.weight": torch.zeros(1)}
    out = _remap_probe_state_dict(sd)
    assert set(out) == {"heads.observed_effect.weight", "heads.visible.weight"}
    already = {"heads.observed_effect.weight": torch.zeros(1)}
    assert set(_remap_probe_state_dict(already)) == set(already)          # already-renamed checkpoints: a no-op


# ------------------------------------------------------------------ acceptance 2: old state dicts load strictly
@pytest.mark.parametrize("goal", [False, True])
def test_old_packet_probe_state_dict_loads_strictly_into_readout_probe(goal):
    old, new = _pair(goal)
    # RED against the pre-migration shape: ReadoutProbe has no `desired_delta` head, so a raw pre-rename key
    # ("heads.desired_delta.*", simulated here since PacketProbe itself already carries the post-rename layout)
    # would fail strict loading without the remap.
    pre_rename = {k.replace("heads.observed_effect.", "heads.desired_delta.", 1)
                  if k.startswith("heads.observed_effect.") else k: v for k, v in old.state_dict().items()}
    with pytest.raises(RuntimeError):
        new.load_state_dict(pre_rename)                                   # strict=True default: unmapped key
    new.load_state_dict(_remap_probe_state_dict(pre_rename))               # GREEN: remapped, loads strictly
    new.load_state_dict(_remap_probe_state_dict(old.state_dict()))         # GREEN: the real (post-rename) layout too


# ------------------------------------------------------------------ acceptance 3: metrics equal probe_metrics
@pytest.mark.parametrize("goal", [False, True])
def test_arm_packet_v1_metrics_equal_legacy_probe_metrics_on_a_fixture_batch(goal):
    z, zm, lab, smask, S = _fixture(goal=goal)
    old, new = _pair(goal)
    old.eval(); new.eval()
    with torch.no_grad():
        o_old, o_new = old(z, zm, S), new(z, zm, S)
    assert "desired_delta" in o_old and "desired_delta" not in o_new       # R1 brief: output alias dropped
    m_old, m_new = probe_metrics(o_old, lab, smask), probe_metrics(o_new, lab, smask)
    assert m_old.keys() == m_new.keys()
    for k in m_old:
        assert m_old[k] == m_new[k], k
    assert "desired_delta_err_m" in m_new                                  # ...but the METRIC key name stays
    l_old, logs_old = probe_loss(o_old, lab, smask)
    l_new, logs_new = probe_loss(o_new, lab, smask)
    assert torch.allclose(l_old, l_new)
    assert logs_old.keys() == logs_new.keys() and all(logs_old[k] == logs_new[k] for k in logs_old)


def test_load_representation_constructs_readout_probe_not_packet_probe(tmp_path):
    """`policies.bundles.load_representation` (R1-owned) is the primary old-checkpoint load path; it must build the
    new registry-driven probe, not the retired PacketProbe construction. RED before the migration (P was a
    PacketProbe): this import alone proves the call site now only names ReadoutProbe."""
    import inspect
    from rrp.policies import bundles
    src = inspect.getsource(bundles.load_representation)
    assert "ReadoutProbe(" in src and "PacketProbe(" not in src
