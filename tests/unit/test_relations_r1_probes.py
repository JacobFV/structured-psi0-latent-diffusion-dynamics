"""D-144 R1 acceptance (docs/relations.md section 10; brief in section 10 "briefs") + its deferred-scope follow-up
(research/tracks/rel-r1c.md): arm/dual probes migrate from `nets.latent_probes.PacketProbe` to
`nets.probes.ReadoutProbe` configured by the registry preset `probes:arm-packet-v1` (+ `probe.arm.goal_effect` when
the legacy config had `goal_effect=True`), at every R1-owned construction site (`policies.bundles.load_representation`,
`harness.train.latent_train`, `harness.eval.latent_causal`, `cli.latent`, `cli.dual_latent`); `nets/latent_probes.py`
(PacketProbe, `probe_loss`, `probe_metrics`) is deleted for real and every caller now imports the ported
`readout_loss` / `readout_metrics` from `harness.eval.latent_eval`.

Row acceptance:
  1. goldens unchanged (verified separately by tests/unit/test_golden.py, whose `_tiny_latent` fixture now builds
     `ReadoutProbe(specs=["preset:probes:arm-packet-v1"])` with the same seed: same module names / parameter
     creation order as the retired `PacketProbe` -> byte-identical state dict / outputs -> unchanged golden hashes).
  2. old PacketProbe state dicts load strictly into ReadoutProbe (incl. pre-rename `desired_delta` checkpoints) --
     tested below without needing the retired class (a ReadoutProbe already carries the identical layout, so its
     OWN state dict, renamed back to the pre-rename key spelling, stands in for "an old checkpoint").
  3. `readout_loss` / `readout_metrics` (harness.eval.latent_eval), applied to a ReadoutProbe's output, still report
     the exact legacy key set (arm hooks must not change what they report -- deferred-scope instructions). This was
     proved once, at merge time, against the real PacketProbe (now retired); ported code that never touched a byte
     of the math keeps the invariant, checked here by the key set / alias assertions instead of a live comparison.
Also: the `desired_delta` OUTPUT alias is dropped (R1 brief) while the `desired_delta_err_m` METRIC key name stays.
"""
import pytest
import torch

from rrp.policies.bundles import _readout_probe_specs, _remap_probe_state_dict
from rrp.harness.eval.latent_eval import readout_loss, readout_metrics
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


def _probe(goal: bool):
    specs, kw = _readout_probe_specs(dict(goal_effect=goal))
    torch.manual_seed(0)
    return ReadoutProbe(dz=8, knots=3, width=32, heads=2, specs=specs, **kw)


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
    new = _probe(goal)
    # A pre-rename PacketProbe checkpoint had a `heads.desired_delta.*` key where ReadoutProbe now has
    # `heads.observed_effect.*` (identical layout otherwise, docs/relations.md 4); simulate one from a ReadoutProbe's
    # own state dict (PacketProbe itself no longer exists to build one from -- decision D-144 addendum (b)).
    pre_rename = {k.replace("heads.observed_effect.", "heads.desired_delta.", 1)
                  if k.startswith("heads.observed_effect.") else k: v for k, v in new.state_dict().items()}
    with pytest.raises(RuntimeError):
        new.load_state_dict(pre_rename)                                   # strict=True default: unmapped key
    new.load_state_dict(_remap_probe_state_dict(pre_rename))               # GREEN: remapped, loads strictly
    new.load_state_dict(_remap_probe_state_dict(new.state_dict()))         # GREEN: the real (post-rename) layout too


# ------------------------------------------------------------------ acceptance 3: readout_loss/readout_metrics keys
@pytest.mark.parametrize("goal", [False, True])
def test_readout_loss_and_metrics_report_the_legacy_arm_key_set(goal):
    """readout_loss / readout_metrics (ported from the deleted probe_loss / probe_metrics, unchanged) applied to
    ReadoutProbe's own output must still report exactly the pre-migration key set -- "metric key names reported by
    hooks must not change" (deferred-scope instructions), and the loss keys stay `probe_<query>`."""
    z, zm, lab, smask, S = _fixture(goal=goal)
    new = _probe(goal).eval()
    with torch.no_grad():
        o_new = new(z, zm, S)
    assert "desired_delta" not in o_new                                    # R1 brief: output alias dropped
    m_new = readout_metrics(o_new, lab, smask)
    expected = {"visible", "visible_pos", "focused_on", "focused_on_pos", "held_by", "held_by_pos",
                "acting_on", "acting_on_pos", "rel_pos_err_m", "observed_effect_err_m", "desired_delta_err_m",
                "subtask"}
    if goal:
        expected |= {"goal_effect_err_m", "goal_effect_err_patient_m", "goal_effect_zero_baseline_patient_m"}
    assert set(m_new) == expected
    assert m_new["desired_delta_err_m"] == m_new["observed_effect_err_m"]  # METRIC key name / value stays
    l_new, logs_new = readout_loss(o_new, lab, smask)
    expected_logs = {f"probe_{q}" for q in ("visible", "focused_on", "held_by", "acting_on", "looking_at",
                                            "rel_pos", "observed_effect", "subtask") + (("goal_effect",) if goal else ())}
    assert set(logs_new) == expected_logs
    assert torch.isfinite(l_new)


def test_load_representation_constructs_readout_probe_not_packet_probe(tmp_path):
    """`policies.bundles.load_representation` (R1-owned) is the primary old-checkpoint load path; it must build the
    new registry-driven probe, not the retired PacketProbe construction. RED before the migration (P was a
    PacketProbe): this import alone proves the call site now only names ReadoutProbe."""
    import inspect
    from rrp.policies import bundles
    src = inspect.getsource(bundles.load_representation)
    assert "ReadoutProbe(" in src and "PacketProbe(" not in src


def test_latent_probes_module_is_gone():
    """docs/relations.md 8.2 delete list; D-144 addendum (b): no live code path reads the old name."""
    with pytest.raises(ModuleNotFoundError):
        import rrp.policies.nets.latent_probes  # noqa: F401
