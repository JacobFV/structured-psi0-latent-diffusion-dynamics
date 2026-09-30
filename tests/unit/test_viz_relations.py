"""Room documents for the one-repo harness and relation factors (rrp.viz.export.relations)."""
import json
import os
import time
from types import SimpleNamespace

from rrp.viz.export.relations import _catalog, build_factors, build_matrix


def _cfg(repo):
    return SimpleNamespace(repo=repo, main_checkout=None, rrp_data=None, now=time.time(), out=repo / "out", live=False)


def test_matrix_reads_recorded_rows_newest_wins_and_registries(tmp_path):
    (tmp_path / "src/rrp/envs").mkdir(parents=True)
    (tmp_path / "src/rrp/policies").mkdir(parents=True)
    (tmp_path / "src/rrp/envs/base.py").write_text('ENVS: dict[str, str] = {"mujoco/arm": "m:f", "simple": "s:f"}\n')
    (tmp_path / "src/rrp/policies/base.py").write_text('POLICIES = {"bc": "b:f"}\n')
    old, new = tmp_path / "artifacts/runs/a/matrix.jsonl", tmp_path / "artifacts/runs/b/matrix.jsonl"
    for p, status in ((old, "n/a"), (new, "accepted")):
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"policy": "bc", "env_id": "mujoco/arm", "body": "panda", "task": "pick_place", "status": status, "reasons": []}) + "\n")
    os.utime(old, (1, 1))
    d = build_matrix(_cfg(tmp_path))
    assert d["schema"] == "rrp-viz/matrix/v1"
    assert [r["status"] for r in d["rows"]] == ["accepted"] and d["rows"][0]["source_file"].endswith("b/matrix.jsonl")
    assert d["envs"] == {"mujoco/arm": "m:f", "simple": "s:f"} and d["policies"] == {"bc": "b:f"}


def test_catalog_status_tokens(tmp_path):
    p = tmp_path / "cat.md"
    p.write_text("## A. geometry\n\n| family | candidates | decomposition | status | label | envs |\n|---|---|---|---|---|---|\n"
                 "| depth | depth | cam × aug | W1 `geo.depth3d` | cam_uvd | arm |\n| shape | scale | extent | P | extent | arm |\n")
    rows = _catalog(p)
    assert [(r["family"], r["status"], r["section"]) for r in rows] == [("depth", "W1", "A. geometry"), ("shape", "P", "A. geometry")]


def test_factors_doc_from_the_registry(tmp_path):
    d = build_factors(_cfg(tmp_path))
    assert d["schema"] == "rrp-viz/factors/v1" and d["n_factors"] > 0
    assert {"name", "field", "op", "form", "status", "field_prov"} <= set(d["factors"][0])
    assert d["runs"] == [] and d["schedules"] == []
