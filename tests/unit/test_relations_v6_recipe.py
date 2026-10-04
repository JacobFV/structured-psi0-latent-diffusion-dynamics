"""T9 on the v6 semfix lineage (D-147 re-plan 2): (1) the same-code control `none` renders the RECORDED v6 suffix configs
(tests/data/armv6_semfix_suffix_runconfigs.json, from the v6 run manifests) up to identity, pins and the legacy policy flags
bias_mode="true" / structured=True, which map to the default factor list preset:arm; (2) every factor-set node equals the
`none` node except the factor list, curriculum, relgen input and names."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rrp.harness.dag import load_dag, plan_dag

ROOT = Path(__file__).resolve().parents[2]
REC = json.loads((ROOT / "tests/data/armv6_semfix_suffix_runconfigs.json").read_text())["runs"]
IDENT = {"lineage", "track", "params.name", "params.policy.name"}


def _flat(d, p=""):
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            out.update(_flat(v, f"{p}.{k}" if p else k))
        return out
    return {p: d}


def _norm(v, src, dst):
    if isinstance(v, str):
        return v.replace(src, dst)
    if isinstance(v, list):
        return [_norm(x, src, dst) for x in v]
    return v


@pytest.fixture(scope="module")
def plan():
    from rrp.harness.pipelines.base import _load_families
    _load_families()
    return plan_dag(load_dag(ROOT / "recipes/relations/relations_v6.yaml"), source="t")


def test_none_control_renders_the_recorded_v6_suffix(plan):
    from rrp.policies.nets.flow import PolicyConfig
    nodes = [n for n in plan.nodes.values() if n.point["fset"] == "none"]
    assert len(nodes) == 28 and {n.rc.run_id.split("/")[-1] for n in nodes} == set(REC)
    for n in nodes:
        a = _flat(n.rc.model_dump(mode="json"))
        b = _flat(REC[n.rc.run_id.split("/")[-1]]["runconfig"])
        diff = {k for k in set(a) | set(b)
                if _norm(a.get(k), "runs/relations/relations6-none/", "runs/armv6/arm6-semfix/") != b.get(k)}
        diff -= IDENT | {k for k in diff if k.startswith("options.pin_sha256.")}
        legacy = {"params.policy.bias_mode", "params.policy.structured"}
        assert diff <= legacy, (n.id, sorted(diff - legacy))
        if diff:                                   # the legacy flags resolve to the default factor list (= none's)
            pol = REC[n.rc.run_id.split("/")[-1]]["runconfig"]["params"]["policy"]
            assert PolicyConfig.from_dict(pol).factors is None and "factors" not in n.rc.params["policy"]


def test_factor_sets_differ_from_none_only_in_the_factor_set(plan):
    for nid, n in plan.nodes.items():
        fset = n.point["fset"]
        if fset == "none" or n.name == "relgen":
            continue
        c = plan.nodes[f"{n.name}@semfix.s{n.point['seed']}.none"]
        a, b = _flat(n.rc.model_dump(mode="json")), _flat(c.rc.model_dump(mode="json"))
        diff = {k for k in set(a) | set(b)
                if _norm(a.get(k), f"relations6-{fset}/", "relations6-none/") != b.get(k)}
        extra = {k for k in diff - IDENT - {"inputs.relgen"} if not k.startswith(("params.policy.factors", "params.curriculum"))}
        assert not extra, (nid, sorted(extra))
        flow = n.rc.stage in ("train_flow", "flow_ft")
        assert flow == ("params.policy.factors" in a or any(k.startswith("params.policy.factors") for k in a))
        assert ("curriculum" in n.rc.params) == (n.name in ("F0", "Fft"))


def test_host_instance_is_placement_only(plan):
    host = plan_dag(load_dag(ROOT / "recipes/relations/relations_v6_host.yaml"), source="t")
    assert set(host.nodes) == set(plan.nodes)
    for nid, n in plan.nodes.items():
        h = host.nodes[nid]
        assert (h.rc.config_hash(), h.rc.run_id, h.deps, h.resources.__dict__) == (n.rc.config_hash(), n.rc.run_id, n.deps, n.resources.__dict__)
        assert (n.placement, h.placement) == ("peer", "host")
