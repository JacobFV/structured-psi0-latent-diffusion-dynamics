"""`extends:` recipe overlays (D-102, D-145): cycles are refused; instances render on top of their template."""
from pathlib import Path

import pytest

from rrp.harness.dag import DagError, load_dag, plan_dag

ROOT = Path(__file__).resolve().parents[2]


def test_extends_cycle_refused(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\nname: a\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\nname: b\n")
    with pytest.raises(DagError):
        load_dag(tmp_path / "a.yaml")


def test_instance_is_the_template_plus_its_deltas():
    """The armdiv v7div instance (a flattened v2/v6/v7div chain) keeps the template's recipe: the same nodes except the
    four dropped diagnostics and the added new-arm eval, and the same flags and seeds per node."""
    tpl = plan_dag(load_dag(ROOT / "recipes/templates/arm_lineage.yaml"), source="t")
    inst = plan_dag(load_dag(ROOT / "recipes/armdiv/arm_lineage_v7div.yaml"), source="t")
    names = lambda p: {n.name for n in p.nodes.values()}
    assert names(inst) == names(tpl) - {"semedits", "prog20k", "proggdag1", "orcbc"} | {"newarms"}
    for nid, n in inst.nodes.items():
        if n.name == "newarms":
            continue
        t = tpl.nodes[nid].rc
        assert t.flags == n.rc.flags and t.seed == n.rc.seed and t.stage == n.rc.stage, nid
