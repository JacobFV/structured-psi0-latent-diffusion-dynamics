"""One family/stage loading point (D-146 round 3, unit stage-load): recipe planning, `rrp stage run`, `rrp stage child` and a
grandchild's inherited run context all see every family and stage (registered on import of harness/pipelines/*) without
the caller importing anything first. Each check runs in a FRESH interpreter: an in-process test would pass on import
order. Host-light: no stage executes (`stage run --dry-run` validates the config and resolves the stage function)."""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
# (recipe, family, stage): one node of each family the round-2 check saw fail, plus the stages registered on import.
CASES = [
    ("humanoid/transfer_h_carry", "humanoid", "adapt_refit"),
    ("humanoid/transfer_h_carry", "humanoid", "eval_transfer"),
    ("humanoid/transfer_h_carry", "humanoid", "sealed_eval"),
    ("relations/relations_geo_smoke", "arm", "relations_data"),
    ("psi0/psi0_tabletop_step2", "psi0", "labels"),
    ("psi0/psi0_tabletop_step2", "psi0", "gate"),
    ("armdiv/arm_lineage_v8div_smoke", "arm", "eval_r2"),
]


def _env(**extra) -> dict:
    e = {k: v for k, v in os.environ.items() if k != "RRP_RUN_CONTEXT"}
    e.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT)]), **extra)
    return e


def _fresh(code: str, *args, env=None, cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, timeout=300,
                          env=env or _env(), cwd=cwd or ROOT)


PLAN = """
import base64, json, sys
from rrp.harness.dag import load_dag, plan_dag, resolve_recipe
plan = plan_dag(load_dag(resolve_recipe(sys.argv[1], sys.argv[2])), source="t")      # nothing imported beforehand
out = {}
for n in plan.nodes.values():
    out.setdefault(n.rc.family + "/" + n.rc.stage, base64.b64encode(json.dumps(n.rc.model_dump(mode="json")).encode()).decode())
print(json.dumps(out))
"""


@pytest.fixture(scope="module")
def configs():
    cache: dict[str, dict] = {}

    def get(recipe: str) -> dict:
        if recipe not in cache:
            p = _fresh(PLAN, recipe, str(ROOT))
            assert p.returncode == 0, f"plan_dag of {recipe} in a fresh process:\n{p.stderr[-1500:]}"
            cache[recipe] = json.loads(p.stdout.strip().splitlines()[-1])
        return cache[recipe]
    return get


@pytest.mark.parametrize("recipe,family,stage", CASES, ids=lambda v: str(v).split("/")[-1])
def test_stage_run_validates_a_node_in_a_fresh_process(configs, recipe, family, stage):
    b64 = configs(recipe)[f"{family}/{stage}"]
    p = subprocess.run([sys.executable, "-m", "rrp.cli", "stage", "run", "--config-b64", b64, "--dry-run"],
                       capture_output=True, text=True, timeout=300, env=_env(), cwd=ROOT)
    assert p.returncode == 0, p.stderr[-1500:]
    got = json.loads(p.stdout.strip().splitlines()[-1])
    assert got["dry_run"] is True and (got["family"], got["stage"]) == (family, stage)


def test_dry_run_still_refuses_an_unknown_stage(configs):
    cfg = json.loads(base64.b64decode(configs("humanoid/transfer_h_carry")["humanoid/adapt_refit"]))
    cfg["stage"] = "no_such_stage"
    b64 = base64.b64encode(json.dumps(cfg).encode()).decode()
    p = subprocess.run([sys.executable, "-m", "rrp.cli", "stage", "run", "--config-b64", b64, "--dry-run"],
                       capture_output=True, text=True, timeout=300, env=_env(), cwd=ROOT)
    assert p.returncode != 0 and "no_such_stage" in p.stderr


def test_inherited_context_and_child_load_families_themselves(configs, tmp_path):
    """A grandchild (`rrp.cli.main` under $RRP_RUN_CONTEXT) parses the stage's RunConfig with no loading of its own."""
    cfg = base64.b64decode(configs("humanoid/transfer_h_carry")["humanoid/adapt_refit"]).decode()
    ctx = tmp_path / "run_context.json"
    ctx.write_text(cfg)
    p = _fresh("from rrp.harness.pipelines.base import inherited_run_config as f; r = f(); print(r.family, r.stage)",
               env=_env(RRP_RUN_CONTEXT=str(ctx)))
    assert p.returncode == 0, p.stderr[-1500:]
    assert p.stdout.split() == ["humanoid", "adapt_refit"]
    c = _fresh("import sys; from rrp.harness.pipelines.base import child_main; "
               "sys.exit(child_main(['--context', sys.argv[1], '--', '-c', 'pass']))", str(ctx))
    # `-c pass` is not a module target: what matters is that the context parsed (no 'unknown family')
    assert "unknown family" not in c.stderr and "unknown stage" not in c.stderr, c.stderr[-1500:]


def test_stage_env_absolutizes_a_relative_pythonpath(tmp_path, monkeypatch):
    """A relative inherited PYTHONPATH entry means another directory once the child changes cwd (`Pipeline.run` does)."""
    from rrp.harness.pipelines import base as B
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(B.StageContext, "context_file", lambda self: tmp_path / "run_context.json")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(["rel", ".", "/abs/dir", ""]))
    e = B.StageContext(rc=None, index=None, root=tmp_path / "repo").env()
    parts = e["PYTHONPATH"].split(os.pathsep)
    assert parts[0] == str(tmp_path / "repo" / "src")
    assert str(tmp_path / "rel") in parts and str(tmp_path) in parts and "/abs/dir" in parts
    assert all(os.path.isabs(q) for q in parts), parts
    assert len(parts) == len(set(parts))


def test_stage_env_without_an_inherited_pythonpath(tmp_path, monkeypatch):
    from rrp.harness.pipelines import base as B
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setattr(B.StageContext, "context_file", lambda self: tmp_path / "run_context.json")
    assert B.StageContext(rc=None, index=None, root=tmp_path).env()["PYTHONPATH"] == str(tmp_path / "src")
