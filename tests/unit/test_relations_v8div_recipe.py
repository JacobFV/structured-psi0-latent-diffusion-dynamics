"""T9 v8div (D-147 re-plan): the factor-set lineages are the armdiv v8div semfix lineage with ONLY the factor set varied.
Every planned node equals the control node of `arm_lineage_v8div` (same stage, data, labeller, steps, seeds, eval sets) once
lineage-local paths are mapped to the control lineage; the only other differences are the factor list, the curriculum and
relgen input, the pins of the reused control outputs, and names. Flow-independent nodes are not planned (reused)."""
from __future__ import annotations

import pytest

from rrp.harness.dag import load_dag, plan_dag

CTL = "runs/armdiv/arm8div-semfix/"
ALLOWED = {"lineage", "track", "params.name", "params.policy.name", "inputs.relgen", "options.pin_sha256.representation"}
ALLOWED_PREFIX = ("params.policy.factors", "params.curriculum")
REUSED = {"stageA", "bc1", "rzbcdag1", "rzbcdag1long", "bc2", "bc3", "rzbcdag2"}


def _flat(d, p=""):
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            out.update(_flat(v, f"{p}.{k}" if p else k))
        return out
    return {p: d}


def _norm(v, fset):
    if isinstance(v, str):
        return v.replace(f"runs/relations/relations8-{fset}/", CTL)
    if isinstance(v, list):
        return [_norm(x, fset) for x in v]
    return v


@pytest.fixture(scope="module")
def control():
    from rrp.harness.pipelines.base import _load_families
    _load_families()
    return plan_dag(load_dag("recipes/armdiv/arm_lineage_v8div.yaml"), source="t")


@pytest.mark.parametrize("recipe,sets", [("relations_v8div", ["geo", "ix", "task"]), ("relations_v8div_all", ["all"])])
def test_factor_lineage_differs_from_control_only_in_the_factor_set(control, recipe, sets):
    plan = plan_dag(load_dag(f"recipes/relations/{recipe}.yaml"), source="t")
    assert {n.point["fset"] for n in plan.nodes.values()} == set(sets)
    assert not REUSED & {n.name for n in plan.nodes.values()}
    for nid, n in plan.nodes.items():
        fset, seed = n.point["fset"], n.point["seed"]
        if n.name == "relgen":
            assert n.rc.stage == "relations_data" and n.rc.options["grasp_contact"] == "v2.1"
            continue
        c = control.nodes[f"{n.name}@semfix.s{seed}"]
        a, b = _flat(n.rc.model_dump(mode="json")), _flat(c.rc.model_dump(mode="json"))
        diff = {k for k in set(a) | set(b) if _norm(a.get(k), fset) != b.get(k)}
        extra = {k for k in diff if k not in ALLOWED and not k.startswith(ALLOWED_PREFIX)}
        assert not extra, (nid, sorted(extra))
        if n.rc.stage in ("train_flow", "flow_ft"):           # the architecture is on every flow node (init_from loads strictly)
            assert n.rc.params["policy"]["factors"][0] == "preset:arm" and len(n.rc.params["policy"]["factors"]) > 1
        # every reference to the control lineage points at a flow-independent (reused) node's output
        for v in a.values():
            for s in (v if isinstance(v, list) else [v]):
                if isinstance(s, str) and s.startswith(CTL):
                    assert s.split("/")[3].split("_s")[0] in ("train_rep", "refit-bcdag1_long", "refit-bcdag2",
                                                              "dagger_collect-bc1", "dagger_collect-bc2", "dagger_collect-bc3"), s


def test_curriculum_on_the_pack_trained_flows_only_and_factor_lists_build():
    from rrp.policies.relations.base import effective_source, resolve
    plan = plan_dag(load_dag("recipes/relations/relations_v8div.yaml"), source="t")
    for nid, n in plan.nodes.items():
        if n.rc.stage not in ("train_flow", "flow_ft"):
            continue
        has = "curriculum" in n.rc.params
        assert has == (n.name in ("F0", "Fft")), nid
        assert ("relgen" in n.rc.inputs) == has, nid
        specs = {s.name: s for s in resolve(n.rc.params["policy"]["factors"], default="arm", family="arm")}
        for f in n.rc.params.get("curriculum", {}).get("factors", []):
            assert f in specs and effective_source(specs[f]) == "probe", (nid, f)
