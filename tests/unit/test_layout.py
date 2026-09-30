"""Repository schema (D-145): every tracked path matches schema.toml; legacy lives only under .old/ and is never read
by live code. docs/architecture.md section 13. The purge units (P1-P9) bring the tree into the schema; until the last
one lands these tests are xfail (strict=False) -- unit P9 removes the marker."""
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCHEMA = tomllib.loads((REPO / "schema.toml").read_text())
PURGE_PENDING = pytest.mark.xfail(reason="D-145 purge in progress (docs/architecture.md 13.6)", strict=False)


def _tracked() -> list[str]:
    try:
        out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [p for p in out.split("\n") if p and (REPO / p).exists()]


def _rx(glob: str) -> re.Pattern:
    out, i = "", 0
    while i < len(glob):
        if glob.startswith("**", i):
            out, i = out + ".*", i + 2
        elif glob[i] == "*":
            out, i = out + "[^/]*", i + 1
        else:
            out, i = out + re.escape(glob[i]), i + 1
    return re.compile(out + r"\Z")


ALLOW = [_rx(g) for a in SCHEMA["area"].values() for g in a["allow"]] + [_rx(f) for f in SCHEMA["top"]["files"]]


def test_schema_manifest_is_consistent():
    top = SCHEMA["top"]
    areas = {g.split("/")[0] for a in SCHEMA["area"].values() for g in a["allow"]}
    assert areas == set(top["dirs"]), (areas, top["dirs"])
    assert all(t["state"] in ("open", "paused") for t in SCHEMA["tracks"].values())


@PURGE_PENDING
def test_every_tracked_path_is_in_the_schema():
    bad = [p for p in _tracked() if not any(r.match(p) for r in ALLOW)]
    tops = sorted({p.split("/")[0] for p in bad})
    assert not bad, f"{len(bad)} tracked paths outside schema.toml (top-level: {tops}); first: {bad[:15]}"


@PURGE_PENDING
def test_live_code_never_references_legacy_areas():
    f = SCHEMA["forbid"]
    pats = [re.compile(p) for p in f["patterns"]]
    hits = []
    for p in _tracked():
        if p in f["exempt"] or not any(p == t or p.startswith(t + "/") for t in f["live_trees"]):
            continue
        if not p.endswith((".py", ".sh", ".yaml", ".json", ".toml")):
            continue
        text = (REPO / p).read_text(errors="ignore")
        hits += [f"{p}: {m.group(0)}" for r in pats for m in [r.search(text)] if m]
    assert not hits, f"{len(hits)} live files reference legacy areas: {hits[:15]}"


@PURGE_PENDING
def test_recipes_are_templates_plus_thin_instances():
    a = SCHEMA["area"]["recipes"]
    bad = []
    for p in _tracked():
        parts = p.split("/")
        if parts[0] != "recipes" or len(parts) != 3 or parts[1] in ("templates", "presets"):
            continue
        if parts[1] not in SCHEMA["tracks"]:
            bad.append(f"{p}: track {parts[1]!r} not in schema.toml [tracks]")
            continue
        text = (REPO / p).read_text()
        keys = set(re.findall(r"^([A-Za-z_]+):", text, re.M))
        missing = [k for k in a["header"] if k not in keys]
        if missing:
            bad.append(f"{p}: missing header keys {missing}")
        if "extends" not in keys:
            bad.append(f"{p}: an instance must extend a template (extends: ../templates/<x>.yaml)")
        if text.count("\n") > a["instance_max_lines"]:
            bad.append(f"{p}: {text.count(chr(10))} lines > {a['instance_max_lines']} (move the pipeline into a template)")
    assert any(p.startswith("recipes/templates/") for p in _tracked()), "no recipes/templates"
    assert not bad, bad[:15]


@PURGE_PENDING
def test_track_notes_are_open_tracks_only():
    names = {Path(p).stem for p in _tracked() if re.match(r"research/tracks/[^/]+\.md\Z", p)} - {"BRIEF"}
    assert names <= set(SCHEMA["tracks"]), f"notes of closed tracks (move to .old/research/tracks): {sorted(names - set(SCHEMA['tracks']))}"
    assert set(SCHEMA["tracks"]) <= names, f"tracks without a note: {sorted(set(SCHEMA['tracks']) - names)}"


def test_artifacts_are_a_frozen_or_track_named_evidence_store():
    a = SCHEMA["area"]["artifacts"]
    ok_names = set(a["frozen_runs"]) | set(SCHEMA["tracks"])
    bad = []
    for p in _tracked():
        parts = p.split("/")
        if parts[0] != "artifacts":
            continue
        if parts[1] == "runs":
            if parts[2] not in ok_names:
                bad.append(f"{p}: run dir {parts[2]!r} is neither a track nor a frozen pre-schema run")
            elif "." in parts[-1] and parts[-1].rsplit(".", 1)[1] not in a["run_extensions"] and parts[2] not in a["frozen_runs"]:
                bad.append(f"{p}: extension not allowed for tracked results")
        if (REPO / p).stat().st_size > a["max_bytes"] and p not in a["oversize_frozen"]:
            bad.append(f"{p}: larger than {a['max_bytes']} bytes")
    assert not bad, bad[:15]


@PURGE_PENDING
def test_old_groups_are_indexed():
    groups = sorted({p.split("/")[1] for p in _tracked() if p.startswith(".old/") and p.count("/") >= 2})
    assert (REPO / ".old" / "README.md").exists()
    index = (REPO / ".old" / "README.md").read_text()
    missing = [g for g in groups if not (REPO / ".old" / g / "README.md").exists() or f".old/{g}/" not in index]
    assert groups and not missing, f".old groups without a README / index line: {missing}"
