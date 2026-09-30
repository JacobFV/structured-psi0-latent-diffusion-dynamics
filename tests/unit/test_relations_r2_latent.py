"""D-144 unit R2 (docs/relations.md 10): arm latent config + flags -> factors. Red before this unit's changes
landed: `LatentConfig` had no `factors` field, `nets.binding_aug` existed and `nets.latent_batch` had no `rebind`,
`core.runconfig._check_variant` did not understand `params.latent.factors`.

Acceptance (the table row):
  1. `LatentConfig.version()` identical for every config under configs/ that constructs one.
  2. `_check_variant` equivalent (old-style AND new `factors:`-style configs validate the same recipes).
  3. `cf_swap("binding")` batches equal `binding_aug` batches on a fixture.
  4. no `semantic_weight` / `probe_lv_min` / `binding_cf` as LatentConfig's primary (dataclass-field / RunConfig-flag)
     surface -- see the documented exception in nets/semantic_latent.py (`LatentConfig.version()`'s hash back-compat,
     the one on-disk legacy-key mapping table, `LATENT_LEGACY_KEYS`) and research/tracks/rel-r2.md /
     research/tracks/rel-r2c.md. `fit_probes_on_frozen`'s parameter was `binding_cf` in R2 (out-of-scope call sites
     at the time); the R2c addendum (D-144, decision (b)) renamed it and both call sites to `cf_mix` (R2c owns
     cli/latent.py and harness/pipelines/arm.py too) -- see test_no_binding_cf_as_live_parameter_name below.
"""
from __future__ import annotations

import glob
import json

import numpy as np
import pytest
import torch

from rrp.core.runconfig import RunConfig
from rrp.harness.data.relgen import TRANSFORMS
import rrp.harness.data.relgen.transforms  # noqa: F401  (registers "cf_swap" into TRANSFORMS)
from rrp.policies.nets.batch import Batch
from rrp.policies.nets.latent_batch import rebind
from rrp.policies.nets.semantic_latent import LatentConfig


def _rebind_via_cf_swap(batch: Batch, b: int, src: int, dst: int):
    """Test-only bridge (NOT in `nets/latent_batch.py`: `policies` may not import `harness`, see
    tests/unit/test_layering.py, and this file lives outside src/). Performs the same slot swap as
    `nets.latent_batch.rebind` on ONE example (batch row `b`), but by calling R9's actual registered `cf_swap`
    transform on that row converted to the generic relgen `Sample` shape (docs/relations.md 5.2). Returns
    `(ctx_rel, act_rel)` numpy arrays for comparison against `rebind`'s tensor output; does not touch `pointers`
    (outside cf_swap's `{tokens, edges, labels}` contract -- see nets/latent_batch.py's module docstring)."""
    cf_swap = TRANSFORMS["cf_swap"].fn
    o = batch.bank_offset["scene"]
    C = batch.ctx_mask.shape[1]
    ctx_rel = batch.ctx_rel[b].cpu().numpy()          # [C,C,R] bool
    act_rel = batch.act_rel[b].cpu().numpy()          # [N,C,R] bool
    i, j = o + src, o + dst
    sample = {
        "inputs": {
            "tokens": {"ctx": {"fields": {"slot": np.arange(C)}}},
            "edges": {
                "ctx_rel": {"sets": ("ctx", "ctx"), "data": ctx_rel},
                "act_rel": {"sets": ("act", "ctx"), "data": act_rel},
            },
        },
        "labels": {},
    }
    out = cf_swap(sample, np.random.default_rng(0), {"field": "slot", "set": "ctx", "pair": (i, j)})[0]
    return out["inputs"]["edges"]["ctx_rel"]["data"], out["inputs"]["edges"]["act_rel"]["data"]

# ------------------------------------------------------------------ 1. version() identity
# Frozen with the PRE-R2 code (git show origin/main:.../semantic_latent.py) over every configs/**/*.json whose
# "latent" block is valid LatentConfig kwargs (computed once, before this unit's changes; see research/tracks/
# rel-r2.md for how). Every one of these files round-trips through `RunConfig.load_legacy` (test_runconfig.py), so
# this also proves the R2 codemod of configs/latent/*.json (semantic_weight/binding_cf/slot_handles -> factors/
# cf_mix) changed nothing a consumer can observe.
_FROZEN_VERSIONS = {
    "configs/latent/rep-binding_latent_nosem_v2.json": "ls-b77960b3f5f3",
    "configs/latent/rep-binding_latent_sem_v2.json": "ls-0a308bea97e5",
    "configs/latent/rep-binding_paired_nosem_v3.json": "ls-554a8c9ad891",
    "configs/latent/rep-binding_paired_nosem_v4.json": "ls-af96778ac916",
    "configs/latent/rep-binding_paired_sem_v3.json": "ls-1e56aefa44d1",
    "configs/latent/rep-binding_paired_sem_v4.json": "ls-6d2448d800d3",
    "configs/latent/rep-dualarm_latent_nosem_v1.json": "ls-3e761d4ece22",
    "configs/latent/rep-dualarm_latent_sem_v1.json": "ls-8665417ce5bb",
    "configs/latent/rep-latent_nosem_v1.json": "ls-dc6ca4a55d80",
    "configs/latent/rep-latent_sem_v1.json": "ls-8db814f941f5",
    "configs/ladder/rep-latent_sem_b1fix_anchor.json": "ls-80e5f25be2f0",
    "configs/ladder/armsemfix/rep-latent_semfix_b1fix_anchor.json": "ls-b3975ca84351",
    "configs/ladder/armnosem/rep-latent_nosem_b1fix_anchor.json": "ls-041f79337259",
    "configs/ladder/armseed2/sfjf2/rep-ladder_latent_semfix_b1fix_anchor_s2.json": "ls-6f6f9b3eb46b",
    "configs/ladder/armseed2/sejf2/rep-ladder_latent_sem_b1fix_anchor_s2.json": "ls-e01d65a468e9",
    "configs/ladder/armseed2/nsjf2/rep-ladder_latent_nosem_b1fix_anchor_s2.json": "ls-22623fe2b543",
    # legged configs whose "latent" block happens to use only LatentConfig-legacy-compatible keys (no legged-only
    # key like qd_dropout): not really this unit's family, but since `LatentConfig(**lat)` DOES construct from them,
    # they are in scope for "identical for every config under configs/" too.
    "configs/legged_latent/rep_nosem_v1.json": "ls-f78e3bffac2c",
    "configs/legged_latent/rep_sem_v1.json": "ls-a59976298c69",
}


@pytest.mark.parametrize("path", sorted(_FROZEN_VERSIONS))
def test_version_identical_to_pre_r2(path):
    d = json.load(open(path))
    assert LatentConfig(**d["latent"]).version() == _FROZEN_VERSIONS[path]


def test_frozen_set_covers_every_latent_config():
    """No configs/**/*.json with a LatentConfig-shaped "latent" block was missed (added after the freeze, say)."""
    seen = set()
    for p in sorted(glob.glob("configs/**/*.json", recursive=True)):
        d = json.load(open(p))
        lat = d.get("latent")
        if not isinstance(lat, dict):
            continue
        try:
            LatentConfig(**lat)
        except TypeError:
            continue     # not arm/dual (e.g. legged's `qd_dropout`): a different, unrelated "latent" block
        seen.add(p)
    assert seen == set(_FROZEN_VERSIONS)


def test_new_and_legacy_construction_agree():
    """factors=[...] (new) and the flat legacy keys (old) build the SAME effective config when they say the same
    thing, and reject being mixed (PolicyConfig.from_dict's rule)."""
    old = LatentConfig(name="x", semantic_weight=0.7, probe_lv_min=-4.0, slot_handles=True, binding_cf=0.3,
                       binding_contrast=0.1)
    new = LatentConfig(name="x", factors=[{"name": "probe.arm.visible", "weight": 0.7, "params": {"lv_min": -4.0}},
                                         {"name": "probe.arm.looking_at", "weight": 0.7, "params": {"lv_min": -4.0}},
                                         {"name": "probe.arm.focused_on", "weight": 0.7, "params": {"lv_min": -4.0}},
                                         {"name": "probe.arm.held_by", "weight": 0.7, "params": {"lv_min": -4.0}},
                                         {"name": "probe.arm.acting_on", "weight": 0.7, "params": {"lv_min": -4.0}},
                                         {"name": "probe.arm.rel_pos", "weight": 0.7, "params": {"lv_min": -4.0}},
                                         {"name": "probe.arm.observed_effect", "weight": 0.7,
                                          "params": {"lv_min": -4.0}},
                                         {"name": "probe.arm.subtask", "weight": 0.7, "params": {"lv_min": -4.0}},
                                         "id.slot_handle"],
                       cf_mix=0.3, cf_contrast=0.1)
    assert old.version() == new.version()
    assert (old.weight, old.lv_min, old.slot_handles) == (new.weight, new.lv_min, new.slot_handles) == (0.7, -4.0, True)
    with pytest.raises(ValueError, match="mixes"):
        LatentConfig(name="x", semantic_weight=1.0, factors=[])


# ------------------------------------------------------------------ 2. _check_variant equivalent
# D-144 sweep-flags: arm/dual `train_rep` no longer accepts the OLD-style (flat `latent.semantic_weight` +
# `flags.probe_lv_min`) shape at all -- `probe_lv_min` was retired from `FLAG_SPEC[("arm"|"dual", "train_rep")]`
# (core/runconfig.py), so a RunConfig that tries to set it is rejected before `_check_variant` even runs. `_rc`'s
# default params is `factors:`-shaped accordingly; `_rc_legged` exercises the old-style branch, which stays live
# ONLY for `legged` (its own dags -- out of this row's file list -- still render the flat shape).
def _rc(**kw):
    base = dict(schema_version="runconfig-1", family="arm", stage="train_rep", variant="sem", seed=1,
               lineage="l", tag=None,
               flags=dict(zero_prev_action=True, realizer_anchor=True, realizer_drop_qd=None, probe_lv_min=None,
                          qd_dropout=None, contact_version="contact_v1"),
               params={"latent": {"factors": [{"name": "probe.arm.visible", "weight": 1.0}]}})
    base.update(kw)
    return RunConfig.model_validate(base)


def _rc_legged(**kw):
    base = dict(schema_version="runconfig-1", family="legged", stage="train_rep", variant="sem", seed=1,
               lineage="l", tag=None,
               flags=dict(zero_prev_action=None, realizer_anchor=None, realizer_drop_qd=None, probe_lv_min=-8.0,
                          qd_dropout=0.0, contact_version="contact_v1"),
               params={"latent": {"semantic_weight": 1.0}})
    base.update(kw)
    return RunConfig.model_validate(base)


def test_check_variant_old_style_is_legged_only():
    _rc_legged(variant="sem")
    with pytest.raises(Exception, match="does not match"):
        _rc_legged(variant="nosem")
    with pytest.raises(Exception, match="must be explicit"):
        _rc_legged(params={"latent": {}})
    with pytest.raises(Exception, match="does not apply"):    # arm: the flag no longer applies at all
        _rc(flags=dict(zero_prev_action=True, realizer_anchor=True, realizer_drop_qd=None, probe_lv_min=-8.0,
                       qd_dropout=None, contact_version="contact_v1"))


def test_check_variant_factors_style():
    factors_sem = [{"name": "probe.arm.visible", "weight": 1.0}]
    factors_nosem = [{"name": "probe.arm.visible", "weight": 0.0}]
    factors_semfix = [{"name": "probe.arm.visible", "weight": 1.0, "params": {"lv_min": -4.0}}]
    _rc(variant="sem", params={"latent": {"factors": factors_sem}})
    with pytest.raises(Exception, match="does not match"):
        _rc(variant="nosem", params={"latent": {"factors": factors_sem}})
    _rc(variant="nosem", params={"latent": {"factors": factors_nosem}})
    _rc(variant="semfix", params={"latent": {"factors": factors_semfix}})


# ------------------------------------------------------------------ 3. cf_swap("binding") == binding_aug (rebind)
def _fixture_batch(B=3, C=6, N=2, R=4, S=4, seed=0) -> Batch:
    """Minimal synthetic Batch: one "scene" bank of S tokens at ctx offset 0 (padding out to C), N action nodes, R
    relation channels, random boolean ctx_rel/act_rel and NO pointer rows (see the module docstring in
    nets/latent_batch.py: `Batch.pointers` is outside cf_swap's {tokens, edges, labels} contract by construction, so
    the equivalence fixture keeps it empty and the structural swap -- the part cf_swap actually covers -- is proven
    directly)."""
    g = torch.Generator().manual_seed(seed)
    ctx_rel = torch.rand(B, C, C, R, generator=g) > 0.6
    act_rel = torch.rand(B, N, C, R, generator=g) > 0.6
    return Batch(
        bank_tokens={"scene": torch.zeros(B, S, 3)}, bank_mask={"scene": torch.ones(B, S, dtype=torch.bool)},
        bank_kind={"scene": torch.zeros(B, S, dtype=torch.long)}, bank_text={"scene": torch.zeros(B, S, 8)},
        bank_offset={"scene": 0}, ctx_mask=torch.ones(B, C, dtype=torch.bool),
        node_feats=torch.zeros(B, N, 3), node_mask=torch.ones(B, N, dtype=torch.bool),
        ctx_rel=ctx_rel, act_rel=act_rel, node_rel=torch.zeros(B, N, N, R, dtype=torch.bool),
        pointers=torch.zeros(B, 0, 2, dtype=torch.long), extra={})


def test_rebind_matches_cf_swap_reference():
    batch = _fixture_batch()
    src, dst = torch.tensor([0, 1, 2]), torch.tensor([2, 3, 0])
    out = rebind(batch, src, dst)
    for b in range(batch.B):
        ref_ctx, ref_act = _rebind_via_cf_swap(batch, b, int(src[b]), int(dst[b]))
        assert (out.ctx_rel[b].numpy() == ref_ctx).all(), f"ctx_rel mismatch at row {b}"
        assert (out.act_rel[b].numpy() == ref_act).all(), f"act_rel mismatch at row {b}"


def test_rebind_is_involution_on_ctx_rel():
    """Swapping src<->dst twice is the identity (sanity check on the fixture / the permutation itself)."""
    batch = _fixture_batch(seed=1)
    src, dst = torch.tensor([0, 1, 0]), torch.tensor([2, 3, 1])
    once = rebind(batch, src, dst)
    twice = rebind(once, src, dst)
    assert torch.equal(twice.ctx_rel, batch.ctx_rel) and torch.equal(twice.act_rel, batch.act_rel)


def test_binding_aug_deleted():
    with pytest.raises(ModuleNotFoundError):
        import rrp.policies.nets.binding_aug  # noqa: F401


# ------------------------------------------------------------------ 4. retired flat keys (documented exceptions)
def test_no_legacy_flat_keys_as_new_primary_surface():
    """The 3 retired flat keys are not LatentConfig's construction surface any more (they still work as a legacy
    shim -- see test_new_and_legacy_construction_agree -- and `LatentConfig.version()` still spells them out
    internally, a documented, unavoidable exception: byte-identical hashes over the SAME JSON object keys, see
    nets/semantic_latent.py's `version()` docstring)."""
    from dataclasses import fields as dc_fields
    names = {f.name for f in dc_fields(LatentConfig)}
    for k in ("semantic_weight", "probe_lv_min", "binding_cf"):
        assert k not in names, f"{k} is still a LatentConfig dataclass field"
        assert not hasattr(LatentConfig, k), f"{k} is still a LatentConfig attribute/property"
    # `factors`, `cf_mix`, `cf_contrast` are the whole new field surface (docs/relations.md 10 R2).
    assert names == {"width", "heads", "ctx_layers", "enc_layers", "knots", "knot_times", "dz", "horizon",
                     "control_dt", "beta_kl", "realizer_layers", "max_phase_ticks", "name", "factors", "cf_mix",
                     "cf_contrast"}


def test_no_binding_cf_as_live_parameter_name():
    """D-144 addendum (decision (b)) + sweep-flags: `fit_probes_on_frozen`'s own parameter is `cf_mix`, not
    `binding_cf` (R2c renamed both external call sites, cli/latent.py and harness/pipelines/arm.py, that R2 could
    not). The sweep-flags row went further and deleted the two remaining legacy surfaces decision (b) had allowed
    to stay (the deprecated `--binding-cf` CLI alias and the `options.get("cf_mix", options.get("binding_cf", ...))`
    pipeline-options fallback): `binding_cf` is not a live read anywhere in src/ any more."""
    import inspect
    from rrp.harness.train.latent_train import fit_probes_on_frozen
    params = inspect.signature(fit_probes_on_frozen).parameters
    assert "cf_mix" in params and "binding_cf" not in params
    import argparse
    from rrp.cli.latent import register_probe_cmd
    p = argparse.ArgumentParser().add_subparsers()
    register_probe_cmd(p)
    fit_probes_parser = p.choices["fit-probes"]
    opt_strings = {s for a in fit_probes_parser._actions for s in a.option_strings}
    assert "--cf-mix" in opt_strings and "--binding-cf" not in opt_strings
    # bytecode co_names/co_consts, not source text: a code comment mentioning the retired name for history is fine
    # (e.g. "options.binding_cf" in a docstring), an actual identifier/string LITERAL the running code touches is not.
    from rrp.harness.pipelines.arm import probes as arm_probes_stage
    code = arm_probes_stage.__code__
    assert "binding_cf" not in code.co_names and "binding_cf" not in code.co_consts
