"""The docs say what the code does (D-146 round 2, unit X2).

One progress-state vocabulary (`rrp.core.runs.RunState`) in STATUS.md and every track note; `verified` / `completed`
carry recorded evidence; cited commits exist; the per-family live-site table of docs/relations.md equals the
`FAMILIES` registry; and the contract statements X1 / HS1 / RG changed are not contradicted by the live docs."""
import re
import subprocess
from pathlib import Path
from typing import get_args

import pytest

from rrp.core.runs import RunState

ROOT = Path(__file__).resolve().parents[2]
STATES = set(get_args(RunState)) - {"failed", "cancelled"}       # the AGENTS.md vocabulary (RunState also has run-level ends)
EVIDENCE_STATES = {"verified", "completed"}
LIVE_DOCS = ["README.md", "STATUS.md", *sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "docs").glob("*.md")),
             *sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "research/tracks").glob("*.md")),
             "research/relations_catalog.md"]


def _text(rel: str) -> str:
    return (ROOT / rel).read_text()


def _table(text: str, header_start: str) -> list[list[str]]:
    """Rows (cells) of the markdown table whose header row starts with `header_start`."""
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith(header_start):
            rows = []
            for row in lines[i + 2:]:
                if not row.startswith("|"):
                    break
                rows.append([c.strip() for c in row.strip().strip("|").split("|")])
            return rows
    raise AssertionError(f"no table starting {header_start!r}")


def _status_tables() -> dict[str, list[list[str]]]:
    t = _text("STATUS.md")
    return {"tracks": _table(t, "| track | workstream | state"), "ledger": _table(t, "| unit | scope | state"),
            "runs": _table(t, "| run | what | state")}


def test_vocabulary_is_the_agents_contract():
    assert STATES == {"planned", "implementing", "test_failed", "verified", "running", "completed", "failed_hypothesis",
                      "blocked_external", "budget_exhausted"}


def test_status_states_are_in_the_vocabulary_and_evidenced():
    tabs = _status_tables()
    for name, col in (("tracks", 2), ("ledger", 2), ("runs", 2)):
        assert tabs[name], name
        for row in tabs[name]:
            assert row[col] in STATES, f"STATUS.md {name} row {row[0]!r}: state {row[col]!r} is not in the vocabulary"
    for row in tabs["ledger"] + tabs["runs"]:
        if row[2] in EVIDENCE_STATES:
            assert re.search(r"`[^`]+`", row[-1]), f"{row[0]}: a {row[2]} row needs a backticked command / artifact / commit"


def test_cited_commits_exist():
    shas = sorted({m for row in _status_tables()["ledger"] for m in re.findall(r"`(?:main )?([0-9a-f]{8})`", row[-1])})
    assert shas
    try:
        shallow = subprocess.run(["git", "rev-parse", "--is-shallow-repository"], cwd=ROOT, capture_output=True,
                                 text=True, check=True).stdout.strip() == "true"
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("needs a git checkout (no git metadata here)")
    if shallow:
        pytest.skip("needs full git history (shallow clone)")
    for sha in shas:
        r = subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=ROOT, capture_output=True)
        assert r.returncode == 0, f"STATUS.md cites commit {sha}, which is not in this repository"


def test_track_notes_carry_one_state_line():
    tracks = {row[0]: row[2] for row in _status_tables()["tracks"]}
    for track, state in tracks.items():
        m = re.search(r"^State: \*\*([a-z_]+)\*\*", _text(f"research/tracks/{track}.md"), re.M)
        assert m, f"research/tracks/{track}.md has no 'State: **<state>**' line"
        assert m.group(1) in STATES, f"{track}: {m.group(1)!r} is not in the vocabulary"
        assert m.group(1) == state, f"{track}: note says {m.group(1)}, STATUS.md says {state}"


def test_live_site_table_equals_the_registry():
    pytest.importorskip("torch", reason="needs torch (the families register when policies.nets.batch imports)")
    import rrp.policies.nets.batch  # noqa: F401  (fills FAMILIES)
    from rrp.policies.relations.base import FAMILIES
    rows = _table(_text("docs/relations.md"), "| family | token sets | live sites")
    doc = {r[0]: r for r in rows}
    assert set(doc) == set(FAMILIES), f"docs/relations.md section 11 lists {sorted(doc)}; the registry has {sorted(FAMILIES)}"
    for fam, ft in FAMILIES.items():
        sets = set(re.findall(r"`([a-z]+)`", doc[fam][1]))
        sites = set(re.findall(r"`([a-z]+>[a-z]+)`", doc[fam][2]))
        vocabs = {c.split(":", 1)[1] for carries in ft.sites.values() for c in carries if c.startswith("edges:")}
        assert sets == set(ft.sets), f"{fam}: token sets {sorted(sets)} != {sorted(ft.sets)}"
        assert sites == set(ft.sites), f"{fam}: live sites {sorted(sites)} != {sorted(ft.sites)}"
        assert set(re.findall(r"`([a-z0-9-]+-rel-v1)`", doc[fam][3])) == vocabs, f"{fam}: edge vocabulary"


def test_no_live_doc_names_the_moved_hooks_module():
    for rel in LIVE_DOCS:
        bad = [i for i, ln in enumerate(_text(rel).splitlines(), 1) if re.search(r"eval[./]hooks", ln)]
        assert not bad, f"{rel}:{bad} names harness/eval/hooks (moved to rrp.harness.hooks by X1)"
    assert "harness.hooks" in _text("docs/architecture.md")


def test_architecture_14_3_states_the_yaw_frame_and_the_ring():
    t = _text("docs/architecture.md")
    sec = t[t.index("14.3"):t.index("14.4")]
    assert "yaw frame" in sec and "`range_ring`" in sec


def test_relgen_shards_live_under_the_run_directory():
    rel = _text("docs/relations.md")
    assert "<run>/relgen" in rel
    assert not re.search(r"(?<![`/])artifacts/relgen(?!`: `schema)", rel.replace("no top-level `artifacts/relgen`", ""))


def test_dual_rows_of_the_roadmap_are_marked_parked():
    rows = [ln for ln in _text("docs/experiments_roadmap.md").splitlines() if ln.startswith("|") and re.search(r"\bdual\b", ln, re.I)]
    assert rows
    assert all("parked" in ln.lower() for ln in rows), [ln[:60] for ln in rows if "parked" not in ln.lower()]
