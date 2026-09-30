"""Recipes (D-145 P1): every recipe under `recipes/` plans dry, and the kept recipes render byte-identically to the
`recipe.*` digests recorded in tests/data/golden.json BEFORE the `dags/` -> `recipes/` move (same nodes, run ids, hashes,
deps, placement, resources and defaults). `arm_collect`/`arm_bc` are the two templates split out of the armdiv data/bc
instances (new, so no pre-move golden; their instances' goldens cover them)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rrp.harness.dag import load_dag, plan_dag, resolve_recipe

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = json.loads((ROOT / "tests/data/golden.json").read_text())
GO2_SHA = "af3f06f4e029e9f92fafc50e6174ffdb171c5512fbab4cd6fac18e6ce23cf18b"
RECIPES = sorted(p for p in (ROOT / "recipes").rglob("*.yaml"))
TIME_CAP_S = 21600


def digest(plan) -> str:
    rows = []
    for nid in plan.order:
        n = plan.nodes[nid]
        rows.append(dict(id=nid, rc=n.rc.model_dump(mode="json"), hash=n.rc.config_hash(), deps=sorted(n.deps),
                         placement=n.placement, retries=n.retries, resources=n.resources.__dict__, label=n.label,
                         run_id=n.rc.run_id))
    blob = json.dumps(dict(nodes=rows, defaults=plan.defaults, track=plan.track, caveat=plan.caveat),
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def _plan(path):
    return plan_dag(load_dag(path), source="t")


def test_recipe_tree_is_present():
    assert len(RECIPES) >= 21


@pytest.mark.parametrize("path", RECIPES, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_recipe_plans(path):
    spec = load_dag(path)
    assert _plan(path).order or not spec["nodes"]       # an overlay fragment (D-126: `extends: [base, fragment]`, `nodes: {}`) plans as an empty DAG


@pytest.mark.parametrize("path", [p for p in RECIPES if p.parent.name == "armdiv"], ids=lambda p: p.stem)
def test_armdiv_recipes_respect_the_time_cap(path):
    assert all((n.resources.max_seconds or 0) <= TIME_CAP_S for n in _plan(path).nodes.values())


@pytest.mark.parametrize("path", [p for p in RECIPES if f"recipe.{p.stem}" in GOLDEN],
                         ids=lambda p: str(p.relative_to(ROOT)))
def test_recipe_digest_matches_pre_move_golden(path):
    assert digest(_plan(path)) == GOLDEN[f"recipe.{path.stem}"]


def test_every_pre_move_golden_has_a_recipe():
    stems = {p.stem for p in RECIPES}
    missing = [k for k in GOLDEN if k.startswith("recipe.") and k != "recipe.legged_lineage.go2"
               and k.split(".", 1)[1] not in stems]
    assert not missing


def test_legged_lineage_go2_child_matches_golden(tmp_path):
    child = tmp_path / "legged_v2_x.yaml"
    child.write_text(f"extends: {ROOT / 'recipes/templates/legged_lineage.yaml'}\nname: legged_v2_x\n"
                     f"label_prefix: 'l8z_{{tg}}{{seed}}'\nglobal_label_prefix: 'l8z'\n"
                     f"vars:\n  body: go2\n  tracker_sha256: {GO2_SHA}\n")
    assert digest(_plan(child)) == GOLDEN["recipe.legged_lineage.go2"]


@pytest.mark.parametrize("name", ["arm_lineage_v8div", "arm_lineage_v8div_smoke"])
def test_v8div_grasp_contact_v21_in_every_simulated_stage(name):
    plan = _plan(ROOT / "recipes/armdiv" / f"{name}.yaml")
    sim = [n for n in plan.nodes.values() if n.rc.stage in ("dagger_collect", "eval_r2", "heldout")]
    assert sim
    assert all((n.rc.options or {}).get("grasp_contact") == "v2.1" for n in sim)


def test_resolve_recipe_by_name_and_path():
    assert resolve_recipe("armdiv/arm_lineage_v8div", ROOT) == ROOT / "recipes/armdiv/arm_lineage_v8div.yaml"
    assert resolve_recipe("templates/arm_lineage.yaml", ROOT).is_file()
    assert resolve_recipe(str(ROOT / "recipes/templates/arm_grpo.yaml")).is_file()
    with pytest.raises(Exception, match="recipe not found"):
        resolve_recipe("nope/none", ROOT)
