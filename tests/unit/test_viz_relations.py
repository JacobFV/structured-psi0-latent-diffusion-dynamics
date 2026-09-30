"""Room documents for the one-repo harness and relation factors (rrp.viz.export.relations)."""
import json
import os
import time
from types import SimpleNamespace

import pytest

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


def test_schedule_competence_by_depth_when_present(tmp_path):
    """R21: competence by composition depth reduces the latest `ScheduleState` line; a factor with no `signals` yet
    (today's R11 placeholder scheduler) stays `competence: None`, never fabricated; `has_competence` flips once a
    line carries real signals (R11 landed)."""
    run_dir = tmp_path / "artifacts/runs/rel_curriculum_demo"
    run_dir.mkdir(parents=True)
    sched = run_dir / "schedule.jsonl"
    sched.write_text(json.dumps({"step": 0, "level": {"geo.depth3d": 1, "ix.contact": 1},
                                 "share": {"geo.depth3d": 0.1}, "full_world": 0.1, "signals": {}}) + "\n")
    d = build_factors(_cfg(tmp_path))
    assert len(d["schedules"]) == 1
    s0 = d["schedules"][0]
    assert s0["step"] == 0 and s0["has_competence"] is False
    cbd = {r["factor"]: r for r in s0["competence_by_depth"]}
    assert cbd["geo.depth3d"] == {"factor": "geo.depth3d", "depth": 1, "share": 0.1, "competence": None,
                                  "plateau": None, "interference": None, "attributed_failures": None}
    sched.write_text(sched.read_text() + json.dumps({"step": 500, "level": {"geo.depth3d": 2, "ix.contact": 1},
                                                      "share": {"geo.depth3d": 0.2}, "full_world": 0.12,
                                                      "signals": {"geo.depth3d": {"competence": 0.61, "plateau": 0.02}}}) + "\n")
    d2 = build_factors(_cfg(tmp_path))
    s1 = d2["schedules"][0]
    assert s1["step"] == 500 and s1["has_competence"] is True
    dep = next(r for r in s1["competence_by_depth"] if r["factor"] == "geo.depth3d")
    assert dep["depth"] == 2 and dep["competence"] == pytest.approx(0.61) and dep["plateau"] == pytest.approx(0.02)
