"""pointer / psi0 pipeline families (D-145 P4c): every node of every recipe renders, its stage runs against a stubbed
subprocess runner, and the argv it builds only uses options the wrapped CLI defines (no torch, no sim, nothing launched)."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from rrp.core.runconfig import RunIndex
from rrp.harness.dag import load_dag, plan_dag
from rrp.harness.pipelines import base as B

ROOT = Path(__file__).resolve().parents[2]
RECIPES = sorted(p for d in ("pointer", "psi0") for p in (ROOT / "recipes" / d).glob("*.yaml"))
SOURCES = dict(pointer=ROOT / "src/rrp/harness/train/pointer/__init__.py", psi0=ROOT / "src/rrp/policies/psi0/train.py")


def _options(path: Path) -> set[str]:
    return set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', path.read_text()))


@pytest.mark.parametrize("path", RECIPES, ids=lambda p: p.stem)
def test_stages_build_valid_commands(path, tmp_path, monkeypatch):
    plan = plan_dag(load_dag(path), source="t")
    (tmp_path / "research/splits").mkdir(parents=True)
    shutil.copy(ROOT / "research/splits/cworld_pointer_v1.json", tmp_path / "research/splits/cworld_pointer_v1.json")
    for n in plan.nodes.values():                # a recipe's declared split that is not in the repo yet (C2's v2): a v1-shaped stand-in
        want = n.rc.options.get("split")
        if want and not (ROOT / want).exists():
            v1 = json.loads((ROOT / "research/splits/cworld_pointer_v1.json").read_text())
            (tmp_path / want).write_text(json.dumps(dict(v1, split_id=Path(want).stem)))
    calls: list[list[str]] = []

    def fake_run(self, argv, **kw):
        calls.append(list(argv))
        if argv[:5] == ["-m", "rrp.cli", "train", "psi0", "heldout"]:      # what the wrapped CLI writes (architecture 14.5)
            f = Path(argv[argv.index("--out") + 1])
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(dict(gate=dict(gap=0.2, margin=0.05))))
    monkeypatch.setattr(B.StageContext, "run", fake_run)
    B._load_families()
    seen = set()
    for nid, node in plan.nodes.items():
        rc = node.rc
        ctx = B.StageContext(rc=rc, index=RunIndex(), root=tmp_path)
        for v in ctx.rc.input_paths(ctx.index).values():          # fake upstream products
            p = tmp_path / v
            (p if not Path(v).suffix else p.parent).mkdir(parents=True, exist_ok=True)
            if Path(v).suffix == ".json":
                p.write_text(json.dumps(dict(val_eps=[1, 2, 3], train_eps=[0], episodes=4, gate=dict(gap=0.2, margin=0.05))))
            elif Path(v).suffix:
                p.write_text("x")
            else:
                (p / "t.npz").write_text("x")
                if p.name == "feat":
                    (p / "meta.json").write_text(json.dumps(dict(episodes=4)))
        calls.clear()
        B._REGISTRY[(rc.family, rc.stage)].fn(ctx)
        assert calls, nid
        seen.add((rc.family, rc.stage))
        for argv in calls:
            if argv[:3] == ["-m", "rrp.cli", "eval"] or argv[2:3] == ["eval"]:
                continue
            known = _options(SOURCES[rc.family]) | ({"--vlm", "--repo-id", "--run-dir", "--out", "--stride", "--batch", "--shard"}
                                                    if "psi0-features" in argv else set())
            if "psi0-labels" in argv:
                known = _options(ROOT / "src/rrp/cli/data.py")
            bad = [a for a in argv if a.startswith("--") and a not in known]
            assert not bad, (nid, bad)
    assert seen


def test_pointer_variant_must_match_semantic_weight():
    plan = plan_dag(load_dag(ROOT / "recipes/pointer/pointer_v1.yaml"), source="t")
    rc = plan.nodes["rep@semfix.s1"].rc
    assert rc.params["w_sem"] == 1.0 and plan.nodes["rep@nosem.s1"].rc.params["w_sem"] == 0.0
    with pytest.raises(Exception, match="does not match"):
        type(rc).model_validate(dict(rc.model_dump(mode="json"), variant="nosem"))
