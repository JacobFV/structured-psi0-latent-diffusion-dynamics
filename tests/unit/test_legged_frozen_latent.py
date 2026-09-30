"""Frozen legged latent version, anchored to the RENDERED `recipes/templates/legged_lineage.yaml` `rep` nodes (D-145 P1).

The legged `"latent"` block is a plain dict (no `LatentConfig`), converted by
`harness/train/legged_latent_train.py::_legged_probe_factors` / `_legged_probe_weight_lv`. `_legged_version` below is the
legged analogue of `LatentConfig.version()`: a hash of the effective `(weight, lv_min)` pair plus the scalar fields.
The frozen values were computed before the D-144 codemod from the pre-codemod flat `configs/legged_*/rep_*.json`
dicts (semfix -> weight 1.0 / lv_min -4.0, nosem -> weight 0.0), and are unchanged by rendering the recipe. The legacy
flat `latent` shape (`semantic_weight` / `probe_lv_min` with no `factors` key) is still read by the same conversion,
so a checkpoint config saved in the old shape keeps its version.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rrp.harness.dag import load_dag, plan_dag
from rrp.harness.train.legged_latent_train import _legged_probe_factors, _legged_probe_weight_lv

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "recipes/templates/legged_lineage.yaml"
SHA = "af3f06f4" + "0" * 56


def _legged_version(lat: dict) -> str:
    w, lv = _legged_probe_weight_lv(_legged_probe_factors(lat))
    d = {"dz": lat.get("dz"), "width": lat.get("width"), "beta_kl": lat.get("beta_kl"),
         "qd_dropout": lat.get("qd_dropout", 0.0), "semantic_weight": w, "probe_lv_min": lv}
    if d["probe_lv_min"] == -8.0:
        d.pop("probe_lv_min")
    return "legged-ls-" + hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


# Frozen from the pre-codemod flat configs (`configs/legged_fixsem/rep_{fixsem,nosem}_*`, `rep_sem_*_lv4`).
FROZEN = {"semfix": "legged-ls-deb830a83784", "nosem": "legged-ls-9187d4dc7c04"}
# Flat legacy shape (pre-factors): the two `configs/legged_latent/rep_{sem,nosem}_v1.json` latent blocks.
FROZEN_FLAT = {
    "legged-ls-dd7ed22b9277": {"dz": 32, "width": 192, "semantic_weight": 1.0, "beta_kl": 0.001},
    "legged-ls-8b17f5e643dd": {"dz": 32, "width": 192, "semantic_weight": 0.0, "beta_kl": 0.001},
}


def _rep_latents(tmp_path):
    child = tmp_path / "legged_v2_x.yaml"
    child.write_text(f"extends: {TEMPLATE}\nname: legged_v2_x\nvars:\n  body: go2\n  tracker_sha256: {SHA}\n")
    plan = plan_dag(load_dag(child), source="t")
    out = [(n.point["variant"], n.rc.to_native()["latent"]) for n in plan.nodes.values() if n.rc.stage == "train_rep"]
    assert len(out) == 6                                  # 2 variants x 3 seeds
    return out


def test_rendered_rep_latent_matches_frozen_version(tmp_path):
    for variant, lat in _rep_latents(tmp_path):
        assert _legged_version(lat) == FROZEN[variant], variant


@pytest.mark.parametrize("version,lat", sorted(FROZEN_FLAT.items()))
def test_legacy_flat_latent_reads_to_the_same_version(version, lat):
    assert _legged_version(lat) == version
