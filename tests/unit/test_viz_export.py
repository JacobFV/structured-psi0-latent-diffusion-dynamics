"""Room data exporter (rrp.viz.export, D-131) on small fixture trees: schema keys, provenance on every row, the D-entry
parser, result/edit/training extraction, and no peer access without --live."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest

from rrp.viz import api
from rrp.viz.export import DOCS, run
from rrp.viz.export.common import Config, parse_entries, parse_tables, source_label, wilson
from rrp.viz.export.scan import edit_label, extract_results, parse_trainlog

DECISIONS = """# decision log

## D-001 2026-09-21 repository location
Repo at `~/work/relational-robot-policy`. See W2.

## D-105 2026-09-27 W8 anymal_c on contact v2: context halt replicates
Raw: `artifacts/runs/legged_edits/go2/r2_sem_s1/effects.json`, compare in `research/tracks/legged8/legged8_compare_go2.json`.
Refers to D-001 and psi1z P-006; W8, W10.
- bullet

## notes (not a decision)
text
"""

STATUS = """# project status
Updated: 2026-09-27 21:30 PDT (records agent).

## current state (2026-09-27 21:30)
| id | workstream | state | status (decisions) | where |
|---|---|---|---|---|
| W1 | physics | running | contact v2 (D-105) | research/tracks/contact.md |

## evidence summary (through D-105)
- **Legged halt replicates.** go2 R2 29/30 vs BC 30/30 (D-105).
- **Arm re-eval (INTERIM, 3/5 cells).** semfix 45 → 62/90 (D-001).
- **Still not tested:** held-out bodies.

## blockers and known limits
- Legged: contact v1 skated (D-001).
"""

ROADMAP = """# roadmap
## A. Arm
| # | question | depends on | cost | experiment | code |
|---|---|---|---|---|---|
| 1 | Does it survive grasp_v2? | – | ~free | done → D-105 | ✅ |
| 2 | Is v6dart good? | 1 | ~3 h | running (launched) | ✅ |
"""


def _fixture(root: Path) -> Config:
    repo = root / "repo"
    (repo / "research/tracks/legged8").mkdir(parents=True)
    (repo / "docs").mkdir()
    (repo / "research/reports").mkdir(parents=True)
    (repo / "research/decisions.md").write_text(DECISIONS)
    (repo / "STATUS.md").write_text(STATUS)
    (repo / "docs/experiments_roadmap.md").write_text(ROADMAP)
    (repo / "docs/intentions_backlog.md").write_text("# backlog\n## Physics\n| item | source | status | evidence | size | "
                                                     "blocks |\n|---|---|---|---|---|---|\n| MJX | D-105 | prototype | x "
                                                     "| L | – |\n")
    (repo / "docs/strategy.md").write_text("# strategy\n## 2. Workstreams\n| id | workstream | owner | depends on | "
                                           "status |\n|---|---|---|---|---|\n| W1 | Physics | contact agent | — | "
                                           "running (D-105) |\n### W1 Physics realism (running)\nbody\n")
    (repo / "docs/related_repos.md").write_text("# related\n## decision crosswalk\n| rrp | psi1z | topic |\n|---|---|---|\n"
                                                "| D-105 | P-006 | admission |\n")
    (repo / "research/reports/evidence_matrix.md").write_text("# evidence\n")
    (repo / "research/tracks/legged8.md").write_text("# legged8\nDAG legged_v2_go2 ETA 2026-09-28 06:00 (stated).\n")
    # results
    run_dir = repo / "artifacts/runs/robust/legged_anymal_c/bc/anymal_c"
    run_dir.mkdir(parents=True)
    (run_dir / "friction=0.4.summary.json").write_text(json.dumps(
        {"n": 20, "success": 3, "rate": 0.15, "wilson95": [0.05, 0.36], "falls": 14, "stages": {"fell": 14},
         "sources": ["bc:train_bc_s0/policy.pt"], "route": {"name": "bc", "kind": "bc"}, "robot": "anymal_c",
         "condition": {"factor": "friction", "level": 0.4, "key": "friction=0.4"}}))
    (repo / "research/tracks/legged8/legged8_compare_go2.json").write_text(json.dumps(
        {"body": "go2", "contact_version": "contact_v2", "variants": {"semfix": {"0": {
            "r2": {"r2_final": {"success": 29, "n": 30, "fell": 1, "source": "learned:train_flow_s0/policy.pt"}},
            "effects": {"ctx halt Δforward m": [-0.34, -0.44, -0.25], "z random 8 toward m": [0.01, -0.01, 0.02]}}}},
         "references": {"teacher": "30/30", "bc": "28/30 (2 fell)"}}))
    arm = repo / "artifacts/runs/armexpert_gc2eval"
    arm.mkdir(parents=True)
    (arm / "compare_gc2_final.json").write_text(json.dumps({"frozen sem": {"panda_pg2": {"grasp_v1": [36, 90],
                                                                                         "grasp_v2": [31, 90]}}}))
    ed = repo / "artifacts/runs/legged_edits/go2/r2_sem_s1"
    ed.mkdir(parents=True)
    (ed / "effects.json").write_text(json.dumps({"source": "learned:x.pt", "conditions": {
        "halt": {"n": 20, "forward": [-1.2, -1.4, -1.0]}, "rand_norm_8": {"n": 20, "forward": [0.01, -0.02, 0.03]},
        "none": {"n": 20, "forward": [0.0, 0.0, 0.0]}}}))
    tl = repo / "artifacts/runs/ladder_flow_x"
    tl.mkdir(parents=True)
    (tl / "train_log.jsonl").write_text("\n".join(json.dumps({"step": i * 10, "loss": 1.0 / (i + 1), "gn": 0.5,
                                                              "lr": 1e-3, "t": i}) for i in range(5000)))
    tr = repo / "artifacts/trackers/go2/contact_v2"
    tr.mkdir(parents=True)
    (tr / "train_log_every10.jsonl").write_text("\n".join(json.dumps({"iter": i, "reward_per_step": 1.0, "alpha": 0.1 * i,
                                                                      "gate": {"action": "hold", "fall_rate": 0.1}})
                                                          for i in range(5)))
    dag = repo / "artifacts/runs/legged8/_dags/legged_v2_go2"
    dag.mkdir(parents=True)
    (dag / "ledger.json").write_text(json.dumps({"schema": "dag-ledger-1", "created": 1.0, "nodes": {
        "collect": {"state": "completed", "stage": "collect", "out": "artifacts/runs/legged8/go2/collect",
                    "attempts": [{"lease_id": "L1", "started": 1.0, "finished": 2.0, "rc": 0}]},
        "eval": {"state": "running", "stage": "eval", "out": "artifacts/runs/robust/legged_anymal_c",
                 "attempts": [{"lease_id": "L2", "started": 3.0}]}}}))
    (repo / "artifacts/video").mkdir(parents=True)
    (repo / "artifacts/video/INDEX.md").write_text(
        "- `2026-09-21_scripted_teacher_panda_pg2_pick_place_s3000001_success.mp4` — source=scripted_teacher ckpt=- "
        "robot=panda_pg2 task=pick_place seed=3000001 outcome=success (privileged evaluator)\n")
    data = root / "rrp-data"
    (data / "viz/replays").mkdir(parents=True)
    (data / "viz/replays/index.json").write_text(json.dumps({"schema": "rrp-viz/replays-index/v1", "replays": [
        {"id": "r1", "family": "legged", "file": "legged/r1.json.gz", "n_frames": 3, "fps": 30}]}))
    (data / "viz/replays/legged").mkdir()
    (data / "viz/replays/legged/r1.json.gz").write_bytes(b"x")
    return Config(repo=repo, out=root / "out", wt_root=None, main_checkout=None, rrp_data=data, psi1z=None,
                  peer="nonexistent-peer-for-tests")


@pytest.fixture()
def exported(tmp_path, monkeypatch):
    calls = []
    real = subprocess.run

    def guard(args, *a, **k):
        calls.append(list(args))
        if args and args[0] in ("ssh", "rsync"):
            raise AssertionError(f"peer access without --live: {args}")
        return real(args, *a, **k)

    monkeypatch.setattr(subprocess, "run", guard)
    cfg = _fixture(tmp_path)
    m = run(cfg)
    return cfg, m, calls


def _doc(cfg, name):
    return json.loads((cfg.out / f"{name}.json").read_text())


def test_every_document_has_the_envelope(exported):
    cfg, m, _ = exported
    for name in DOCS:
        assert m["documents"][name]["error"] is None, (name, m["documents"][name])
        d = _doc(cfg, name)
        assert d["schema"] == f"rrp-viz/{name}/v1"
        for k in ("generated_at", "git_sha", "sources"):
            assert k in d
        assert isinstance(d["sources"], list)


def test_no_peer_access_without_live(exported):
    cfg, _, calls = exported
    assert not [c for c in calls if c and c[0] in ("ssh", "rsync")]
    live = _doc(cfg, "live")
    assert live["stale"] is True and live["peer_read_at"] is None and live["leases"] == []


def test_results_rows_carry_provenance_and_labels(exported):
    cfg, _, _ = exported
    rows = _doc(cfg, "results")["rows"]
    assert rows
    for r in rows:
        for k in ("source_file", "location", "decision", "source_label", "interim", "caveat", "k", "n", "rate",
                  "ci_lo", "ci_hi", "family", "body", "metric"):
            assert k in r, k
        assert r["source_file"] and not r["source_file"].startswith("/")
    rb = next(r for r in rows if r["source_file"].endswith("friction=0.4.summary.json"))
    assert (rb["k"], rb["n"], rb["ci_lo"], rb["ci_hi"], rb["ci_method"]) == (3, 20, 0.05, 0.36, "recorded")
    assert rb["body"] == "anymal_c" and rb["family"] == "legged" and rb["source_label"] == "bc:train_bc_s0/policy.pt"
    assert rb["condition"] == "friction=0.4"
    # a run dir named in the robust report's DAG node that is still running -> interim
    assert rb["interim"] and "legged_v2_go2" in rb["interim_reason"]
    g = [r for r in rows if r["source_file"].endswith("legged8_compare_go2.json")]
    fin = next(r for r in g if r["metric"] == "success")
    assert (fin["k"], fin["n"], fin["variant"], fin["seed"], fin["contact_version"]) == (29, 30, "semfix", 0, "contact_v2")
    assert fin["decision"] == "D-105" and fin["source_label"] == "learned:train_flow_s0/policy.pt"
    assert "inexact resume" in (fin["caveat"] or "")
    bc = next(r for r in g if r["metric"] == "bc")
    assert (bc["k"], bc["n"], bc["fell"], bc["ci_method"]) == (28, 30, 2, "wilson95_computed")
    arm = [r for r in rows if "armexpert_gc2eval" in r["source_file"]]
    assert {(r["grasp_version"], r["k"], r["n"]) for r in arm} == {("grasp_v1", 36, 90), ("grasp_v2", 31, 90)}
    v1 = next(r for r in arm if r["grasp_version"] == "grasp_v1")
    assert "grasp_v1" in v1["caveat"] and v1["family"] == "arm"


def test_edits(exported):
    cfg, _, _ = exported
    rows = _doc(cfg, "edits")["rows"]
    halt = next(r for r in rows if r["edit"] == "halt")
    assert (halt["effect"], halt["ci"], halt["n_pairs"], halt["role"], halt["body"], halt["variant"], halt["seed"]) == \
        (-1.2, [-1.4, -1.0], 20, "edit", "go2", "sem", 1)
    assert next(r for r in rows if r["edit"] == "rand_norm_8")["control"] is True
    assert not [r for r in rows if r["edit"] == "none"]          # the zero reference row is dropped
    ctx = next(r for r in rows if r["edit"] == "ctx_halt")
    assert ctx["metric"] == "Δforward m" and ctx["effect"] == -0.34 and ctx["source_file"].startswith("research/")


def test_training_downsampled_with_alpha_and_gate(exported):
    cfg, _, _ = exported
    t = _doc(cfg, "training")
    flow = next(r for r in t["runs"] if r["run"].endswith("ladder_flow_x"))
    assert flow["kind"] == "flow" and flow["n_records"] == 5000 and flow["n_points"] <= 2000
    assert flow["grad_norm_key"] == "gn" and flow["clip_scale_key"] is None
    s = json.loads((cfg.out / flow["series_file"]).read_text())
    assert len(s["step"]) <= 2000 and s["step"][0] == 0 and s["step"][-1] == 49990 and s["grad_norm"][0] == 0.5
    trk = next(r for r in t["runs"] if "trackers/go2" in r["run"])
    st = json.loads((cfg.out / trk["series_file"]).read_text())
    assert trk["kind"] == "tracker" and st["alpha"][-1] == pytest.approx(0.4) and st["gate_state"][0] == "hold"


def test_knowledge_overview_dags_videos_replays(exported):
    cfg, _, _ = exported
    k = _doc(cfg, "knowledge")
    assert [d["id"] for d in k["decisions"]] == ["D-001", "D-105"]
    assert [r["status"] for r in k["roadmap"]] == ["done", "running"]
    assert k["workstreams"][0]["id"] == "W1" and "body" in k["workstreams"][0]["detail_markdown"]
    o = _doc(cfg, "overview")
    assert [c["status"] for c in o["claims"]] == ["established", "interim", "open"]
    assert any(n["k"] == 29 and n["n"] == 30 for n in o["key_numbers"])
    assert o["latest_decisions"][0]["id"] == "D-105" and len(o["open"]) == 1
    d = _doc(cfg, "dags")["dags"][0]
    assert d["dag"] == "legged_v2_go2" and d["counts"] == {"completed": 1, "running": 1} and not d["complete"]
    assert d["eta_statements"] and "ETA" in d["eta_statements"][0]["text"]
    v = _doc(cfg, "videos")["videos"][0]
    assert v["label"] == "scripted_teacher" and v["robot"] == "panda_pg2" and v["exists_locally"] is False
    r = _doc(cfg, "replays")["replays"][0]
    assert r["id"] == "r1" and r["exists_locally"] and api.replay_file(cfg, "r1").name == "r1.json.gz"
    assert api.replay_file(cfg, "../etc") is None


def test_decision_parser_sample():
    es = parse_entries(DECISIONS, "research/decisions.md")
    assert [e["id"] for e in es] == ["D-001", "D-105"]
    e = es[1]
    assert e["date"] == "2026-09-27" and e["title"].startswith("W8 anymal_c on contact v2")
    assert e["refs_d"] == ["D-001"] and e["refs_p"] == ["P-006"] and e["workstreams"] == ["W8", "W10"]
    assert "artifacts/runs/legged_edits/go2/r2_sem_s1/effects.json" in e["paths"]
    assert e["body"].endswith("- bullet") and "notes (not a decision)" not in e["body"]
    assert e["line"] == 6


def test_small_parsers():
    t = parse_tables("## h\n| a | b |\n|---|---|\n| 1 | x \\| y |\n", "f.md")[0]
    assert t["heading"] == "h" and t["header"] == ["a", "b"] and t["rows"] == [["1", "x | y"]] and t["line"] == 2
    assert edit_label("ctx mirror ACTIVE toward m") == ("ctx_mirror_active", "toward m")
    assert edit_label("z turn +0.6 Δyaw") == ("z_turn_+0.6", "Δyaw")
    assert source_label("teacher") == "scripted_teacher" and source_label("bc:x.pt") == "bc:x.pt"
    assert wilson(0, 20) == [0.0, 0.1611]
    r = extract_results({"meta": {"robot": "panda_pg2", "source": "learned:a"}, "summary": {"ctl": {"n": 8, "k": 2}}})
    assert r["rows"][0]["ctx"]["body"] == "panda_pg2" and r["rows"][0]["metric"] == "ctl"
    p = parse_trainlog("\n".join(json.dumps({"iter": i, "x": i}) for i in range(4001)))
    assert len(p["step"]) <= 2000 and p["step"][-1] == 4000


def test_api_doc_allowlist(exported):
    cfg, _, _ = exported
    assert api.doc(cfg, "STATUS.md")["markdown"].startswith("# project status")
    assert api.doc(cfg, "docs/strategy.md") is not None
    for bad in ("../x.md", "/etc/passwd", "src/rrp/x.md", "docs/../../x.md", "STATUS.txt", "psi1z/secret.md"):
        assert api.doc(cfg, bad) is None, bad


def test_only_and_cache_reuse(exported):
    cfg, _, _ = exported
    t0 = time.time()
    m = run(cfg, ["results"])
    assert m["last_run"]["only"] == ["results"] and time.time() - t0 < 5
    assert (cfg.out / "_cache/files.pkl").exists() and (cfg.out / "_cache/found.pkl").exists()


def test_cached_peer_read_is_stale_and_peer_only_summaries_enter_results(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda args, *a, **k: (_ for _ in ()).throw(AssertionError(args))
                        if args and args[0] in ("ssh", "rsync") else subprocess.CompletedProcess(args, 1, "", ""))
    cfg = _fixture(tmp_path)
    cfg.cache_dir.mkdir(parents=True)
    (cfg.cache_dir / "peer_read.json").write_text(json.dumps({"ok": True, "t": time.time() - 120, "data": {
        "broker": {"admission_stopped": False}, "watchdog": [{"t": 1.0, "level": "ok", "reasons": []}],
        "leases": {"L2": {"state": "active", "created": time.time() - 200, "request": {"label": "x", "memory_bytes": 1},
                          "cgroup": {"memory.current": "5", "memory.peak": "7", "memory.events": {"high": "0"}}}},
        "recent_summaries": [{"path": "artifacts/runs/new_eval/panda_pg2/x.summary.json", "mtime": 1.0,
                              "doc": {"n": 10, "success": 4, "robot": "panda_pg2", "route": "bc"}}]}}))
    run(cfg, ["live", "results"])
    live = _doc(cfg, "live")
    assert live["stale"] is True and live["age_s"] >= 119 and live["refreshed_now"] is False
    lease = live["leases"][0]
    assert lease["workstream"] == "legged8" and lease["dag"] == "legged_v2_go2" and lease["measured"]["memory_peak"] == 7
    r = next(r for r in _doc(cfg, "results")["rows"] if r.get("peer_only"))
    assert (r["k"], r["n"], r["body"], r["location"]) == (4, 10, "panda_pg2", "peer:nonexistent-peer-for-tests")
