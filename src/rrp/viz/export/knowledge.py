"""Decisions: parsed D-entries, the decision index that attaches D-ids to result files (by explicit mention of the run
directory in the decision text), and the overview document (latest decisions)."""
from __future__ import annotations

import re
from pathlib import Path

from .common import Config, bodies_in, envelope, parse_entries

_TOKEN = re.compile(r"[A-Za-z0-9_.\-=+/]+")
_STOP = {"artifacts", "runs", "research", "tracks", "eval", "train", "diag", "val", "summary", "json", "jsonl", "md",
         "policy", "final", "compare", "results", "logs", "data", "none", "shard", "shard0", "part0", "seed", "docs",
         "trackers", "contact_v2", "video", "gates", "arm", "legged", "dual", "teacher", "oracle", "generated",
         "learned", "semfix", "fixsem", "nosem", "flow", "snap", "r2_final", "r2_snap", "edits", "effects"}


def _read(p: Path) -> str | None:
    try:
        return p.read_text()
    except OSError:
        return None


_GEN = re.compile(r"^(shard|part|seed|s|ep|eval|train|r)[_\-]?\d+[a-z]?$")


class DecisionIndex:
    """token -> D-ids whose text mentions it (tokens split on '/'), and the same for research/tracks/*.md notes."""

    def __init__(self, decisions: list[dict], track_notes: dict[str, str]):
        self.tok: dict[str, set[str]] = {}
        self.track: dict[str, set[str]] = {}
        for d in decisions:
            self._add(self.tok, d["id"], d["title"] + "\n" + d["body"])
        for name, text in track_notes.items():
            self._add(self.track, name, text)
        self._memo: dict[str, dict] = {}
        self.generic: set[str] = set()

    def set_generic_stems(self, rels) -> None:
        """File stems shared by more than 3 files (semantic_summary_generated, none.summary, …) never identify a run."""
        cnt: dict[str, int] = {}
        for r in rels:
            st = self._stem(r)
            cnt[st] = cnt.get(st, 0) + 1
        self.generic = {k for k, v in cnt.items() if v > 3}
        self._memo.clear()

    @staticmethod
    def _stem(rel: str) -> str:
        stem = rel.rsplit("/", 1)[-1]
        for suf in (".summary.json", ".json", ".jsonl", ".md"):
            if stem.endswith(suf):
                return stem[: -len(suf)].lower()
        return stem.lower()

    @staticmethod
    def _add(index, key, text):
        for t in _TOKEN.findall(text):
            t = t.strip(".,;:()`'\"").lower()
            if not any(ch in t for ch in "/_-.="):
                continue                              # plain words never identify a run (identifier-like tokens only)
            for piece in {t, *t.split("/")}:
                piece = piece.strip(".,;:()`'\"")
                if len(piece) >= 4:
                    index.setdefault(piece, set()).add(key)
                    for suf in (".summary.json", ".json", ".jsonl", ".md", ".pt"):
                        if piece.endswith(suf):
                            index.setdefault(piece[: -len(suf)], set()).add(key)

    @staticmethod
    def candidates(rel: str) -> list[str]:
        parts = rel.split("/")
        if parts[:2] in (["artifacts", "runs"], ["artifacts", "trackers"], ["research", "tracks"]):
            parts = parts[2:] if parts[1] != "trackers" else ["trackers"] + parts[2:]
        stem = parts[-1]
        for suf in (".summary.json", ".json", ".jsonl", ".md"):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                break
        cands = [rel.lower(), "/".join(parts).lower(), stem.lower()] + [p.lower() for p in reversed(parts[:-1])]
        return [c for c in cands if len(c) >= 4 and c not in _STOP and not c.isdigit() and not _GEN.match(c)
                and bodies_in(c) != [c]]

    def lookup(self, rel: str) -> dict:
        if rel in self._memo:
            return self._memo[rel]
        out = {"decision": None, "decisions": [], "decision_match": None, "track_notes": []}
        for c in self.candidates(rel):
            if c in self.generic:
                continue
            ids = self.tok.get(c)
            if ids:
                ds = sorted(ids)
                out.update(decision=ds[-1], decisions=ds[-8:], decision_match=c)
                break
        for c in self.candidates(rel):
            tr = self.track.get(c)
            if tr:
                out["track_notes"] = sorted(tr)[:6]
                break
        self._memo[rel] = out
        return out


def load_decisions(cfg: Config) -> list[dict]:
    t = _read(cfg.repo / "research/decisions.md") or ""
    return parse_entries(t, "research/decisions.md", "D")


def load_track_notes(cfg: Config) -> dict[str, str]:
    out = {}
    base = cfg.repo / "research/tracks"
    if base.is_dir():
        for p in sorted(base.glob("*.md")):
            out[f"research/tracks/{p.name}"] = _read(p) or ""
    return out


def build_overview(cfg: Config, decisions: list[dict]) -> dict:
    """The latest 15 decisions (Board list and ticker)."""
    latest = sorted(decisions, key=lambda d: int(d["id"][2:]))[-15:][::-1]
    return envelope("overview", cfg, ["research/decisions.md"],
                    latest_decisions=[{k: d[k] for k in ("id", "date", "title", "source_file", "line", "refs_d", "workstreams")}
                                      for d in latest])
