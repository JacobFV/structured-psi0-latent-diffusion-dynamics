"""D-144 sweep-flags follow-up (2026-09-30): legged's own frozen-version test, extending
`tests/unit/test_relations_r2_latent.py`'s arm acceptance criterion ("LatentConfig.version() identical for every
config under configs/") to the legged family, whose `"latent"` blocks are a plain dict (no `LatentConfig`
instance) converted through `harness/train/legged_latent_train.py::_legged_probe_factors` /
`_legged_probe_weight_lv` instead.

Red before this row's changes landed: every `configs/{legged_latent,legged_fixsem,t1_diag}/rep_*.json` file used a
flat `latent.semantic_weight` / `latent.probe_lv_min` pair, and `dags/legged_v2_*.yaml` (+ friends) rendered the
same flat shape via `flags.probe_lv_min` + `params.latent.semantic_weight`; `core/runconfig.py`'s
`FLAG_SPEC[("legged", "train_rep")]` still mapped `probe_lv_min`. Green after: every one of those config files (all
but the two noted below) renders `params.latent.factors` (a `probe.legged.*` FactorSpec per query,
`nets/semantic_latent.py::legacy_latent_factors`), `FLAG_SPEC[("legged", "train_rep")]` no longer maps
`probe_lv_min`, and `_legged_version` below -- the legged analogue of `LatentConfig.version()`, computed from the
SAME derived `(weight, lv_min)` pair `_legged_probe_weight_lv(_legged_probe_factors(lat))` returns either way --
is unchanged for every file (verified against the pre-codemod flat dicts at `git show HEAD:<path>` before this row
started, not just re-asserted here: see `research/tracks/sweep-flags.md`'s follow-up note for the exact command).

`configs/legged_latent/rep_{nosem,sem}_v1.json` are deliberately NOT codemodded to `factors:` (unlike the other 27
rep files): their `"latent"` block has no legged-only key (no `qd_dropout`), so it is ALSO valid
`LatentConfig(**lat)` kwargs and is separately pinned by `test_relations_r2_latent.py`'s OWN frozen table under
ARM's `probe.arm.*` reading. Adding `factors: [{"name": "probe.legged.*", ...}]` there would silently feed that
unrelated test a `probe.legged.*` list its `LatentConfig.weight`/`.lv_min` properties read as an EMPTY `probe.arm.*`
set (defaults: weight 1.0, lv_min -8.0) -- i.e. `LatentConfig(**lat).version()` would change for `rep_nosem_v1.json`
(real weight 0.0 -> silently-read default 1.0) despite the file's actual meaning under legged's OWN
`_legged_probe_factors` reading staying nosem. These two stay flat and are covered here (their `_legged_version` is
unaffected either way, since `_legged_probe_factors` reads the SAME `LATENT_LEGACY_KEYS`-named flat keys through
`nets/semantic_latent.py::legacy_latent_factors` for a config that has no `"factors"` key at all).
"""
from __future__ import annotations

import glob
import hashlib
import json

import pytest

from rrp.harness.train.legged_latent_train import _legged_probe_factors, _legged_probe_weight_lv

_LEGGED_REP_GLOBS = ("configs/legged_latent/rep_*.json", "configs/legged_fixsem/rep_*.json",
                    "configs/t1_diag/rep_*.json")


def _legged_version(lat: dict) -> str:
    """The legged analogue of `nets/semantic_latent.py::LatentConfig.version()`: same back-compat shape (hash of
    the effective scalar fields, `probe_lv_min` omitted at the -8.0 default), but over legged's own derived
    `(weight, lv_min)` -- works identically whether `lat` is still flat (`semantic_weight`/`probe_lv_min`) or
    already `factors:`-shaped, since both go through the same `_legged_probe_factors`/`_legged_probe_weight_lv`
    conversion this row's code uses at training time."""
    w, lv = _legged_probe_weight_lv(_legged_probe_factors(lat))
    d = {"dz": lat.get("dz"), "width": lat.get("width"), "beta_kl": lat.get("beta_kl"),
        "qd_dropout": lat.get("qd_dropout", 0.0), "semantic_weight": w, "probe_lv_min": lv}
    if d["probe_lv_min"] == -8.0:
        d.pop("probe_lv_min")
    return "legged-ls-" + hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


# Frozen BEFORE this row's dag/config codemod (verified against `git show HEAD:<path>` at the row's starting
# commit, not just re-asserted against the current, already-codemodded files -- see the module docstring).
_FROZEN_LEGGED_VERSIONS = {
    "configs/legged_fixsem/rep_fixsem_go2_s1.json": "legged-ls-deb830a83784",
    "configs/legged_fixsem/rep_fixsem_go2_s2.json": "legged-ls-deb830a83784",
    "configs/legged_fixsem/rep_fixsem_hexapod6_s1.json": "legged-ls-deb830a83784",
    "configs/legged_fixsem/rep_fixsem_hexapod6_s2.json": "legged-ls-deb830a83784",
    "configs/legged_fixsem/rep_nosem_go2_s1.json": "legged-ls-9187d4dc7c04",
    "configs/legged_fixsem/rep_nosem_go2_s2.json": "legged-ls-9187d4dc7c04",
    "configs/legged_fixsem/rep_nosem_hexapod6_s1.json": "legged-ls-9187d4dc7c04",
    "configs/legged_fixsem/rep_nosem_hexapod6_s2.json": "legged-ls-9187d4dc7c04",
    "configs/legged_fixsem/rep_sem_go2_lv4.json": "legged-ls-deb830a83784",
    "configs/legged_fixsem/rep_sem_hexapod6_lv4.json": "legged-ls-deb830a83784",
    "configs/legged_latent/rep_nosem_g1_v2.json": "legged-ls-9187d4dc7c04",
    "configs/legged_latent/rep_nosem_go2_v2.json": "legged-ls-9187d4dc7c04",
    "configs/legged_latent/rep_nosem_hexapod6_v2.json": "legged-ls-9187d4dc7c04",
    "configs/legged_latent/rep_nosem_t1_v2.json": "legged-ls-9187d4dc7c04",
    "configs/legged_latent/rep_nosem_t1_v2s1.json": "legged-ls-9187d4dc7c04",
    "configs/legged_latent/rep_nosem_t1_v2s2.json": "legged-ls-9187d4dc7c04",
    "configs/legged_latent/rep_nosem_t1_v2s3.json": "legged-ls-9187d4dc7c04",
    "configs/legged_latent/rep_nosem_v1.json": "legged-ls-8b17f5e643dd",
    "configs/legged_latent/rep_sem_g1_v2.json": "legged-ls-9da6c14c5633",
    "configs/legged_latent/rep_sem_go2_v2.json": "legged-ls-9da6c14c5633",
    "configs/legged_latent/rep_sem_hexapod6_v2.json": "legged-ls-9da6c14c5633",
    "configs/legged_latent/rep_sem_t1_v2.json": "legged-ls-9da6c14c5633",
    "configs/legged_latent/rep_sem_t1_v2s1.json": "legged-ls-9da6c14c5633",
    "configs/legged_latent/rep_sem_t1_v2s2.json": "legged-ls-9da6c14c5633",
    "configs/legged_latent/rep_sem_t1_v2s3.json": "legged-ls-9da6c14c5633",
    "configs/legged_latent/rep_sem_v1.json": "legged-ls-dd7ed22b9277",
    "configs/t1_diag/rep_sem_lv4.json": "legged-ls-deb830a83784",
    "configs/t1_diag/rep_sem_lv4_s1.json": "legged-ls-deb830a83784",
    "configs/t1_diag/rep_sem_lv4_s3.json": "legged-ls-deb830a83784",
}


@pytest.mark.parametrize("path", sorted(_FROZEN_LEGGED_VERSIONS))
def test_legged_version_identical_to_pre_codemod(path):
    d = json.load(open(path))
    assert _legged_version(d["latent"]) == _FROZEN_LEGGED_VERSIONS[path]


def test_frozen_set_covers_every_legged_rep_config():
    """No `configs/{legged_latent,legged_fixsem,t1_diag}/rep_*.json` was missed (added after the freeze, say)."""
    seen = set()
    for pat in _LEGGED_REP_GLOBS:
        for p in sorted(glob.glob(pat)):
            seen.add(p)
    assert seen == set(_FROZEN_LEGGED_VERSIONS)


def test_legged_configs_now_factors_shaped_except_the_two_arm_compatible_ones():
    """Every codemodded file is `factors:`-shaped; the two deliberately-left-flat files (module docstring) are
    exactly `rep_{nosem,sem}_v1.json` and nothing else."""
    still_flat = []
    for pat in _LEGGED_REP_GLOBS:
        for p in sorted(glob.glob(pat)):
            lat = json.load(open(p))["latent"]
            if "factors" not in lat:
                still_flat.append(p)
    assert still_flat == ["configs/legged_latent/rep_nosem_v1.json", "configs/legged_latent/rep_sem_v1.json"]
