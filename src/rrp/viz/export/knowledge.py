"""Knowledge and overview documents: parsed D-entries, roadmap, backlog, strategy workstreams, STATUS, docs list; the
decision index that attaches D-ids to result files (by explicit mention of the run directory in the decision text)."""
from __future__ import annotations

import re
from pathlib import Path

from .common import (Config, bodies_in, bullets, d_refs, envelope, parse_entries, parse_sections, parse_tables)

DOC_FILES = ("research/decisions.md", "research/reports/evidence_matrix.md", "docs/strategy.md",
             "docs/experiments_roadmap.md", "docs/intentions_backlog.md", "STATUS.md", "research/tracks/psi0.md")
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


def _roadmap(cfg: Config) -> list[dict]:
    t = _read(cfg.repo / "docs/experiments_roadmap.md") or ""
    rows = []
    for tab in parse_tables(t, "docs/experiments_roadmap.md"):
        h = [c.lower() for c in tab["header"]]
        if "#" not in h and "question" not in h:
            continue
        for r in tab["rows"]:
            d = dict(zip(h, r))
            st = d.get("experiment", "")
            status = ("done" if st.lower().startswith("done") else "running" if "running" in st.lower()
                      else "planned" if "planned" in st.lower() else "conditional" if "conditional" in st.lower()
                      else st.split(" ")[0].lower() if st else None)
            rows.append({"n": d.get("#"), "section": tab["heading"], "question": d.get("question"),
                         "depends_on": d.get("depends on"), "cost": d.get("cost"), "status_text": st, "status": status,
                         "code": d.get("code"), "decisions": d_refs(" ".join(r)), "source_file": tab["source_file"],
                         "line": tab["line"]})
    return rows


def _backlog(cfg: Config) -> list[dict]:
    t = _read(cfg.repo / "docs/intentions_backlog.md") or ""
    rows = []
    for tab in parse_tables(t, "docs/intentions_backlog.md"):
        h = [c.lower() for c in tab["header"]]
        if "item" not in h:
            continue
        for r in tab["rows"]:
            d = dict(zip(h, r))
            rows.append({"section": tab["heading"], "item": d.get("item"), "source": d.get("source"),
                         "status": d.get("status"), "evidence": d.get("evidence"), "size": d.get("size"),
                         "blocks": d.get("blocks"), "decisions": d_refs(" ".join(r)),
                         "source_file": tab["source_file"], "line": tab["line"]})
    return rows


def _workstreams(cfg: Config) -> list[dict]:
    t = _read(cfg.repo / "docs/strategy.md") or ""
    secs = {s["title"].split(" ")[0]: s for s in parse_sections(t) if re.match(r"W\d+\b", s["title"])}
    rows = []
    for tab in parse_tables(t, "docs/strategy.md"):
        h = [c.lower() for c in tab["header"]]
        if h[:2] != ["id", "workstream"]:
            continue
        for r in tab["rows"]:
            d = dict(zip(h, r))
            s = secs.get(d.get("id"))
            rows.append({"id": d.get("id"), "workstream": d.get("workstream"), "owner": d.get("owner"),
                         "depends_on": d.get("depends on"), "status": d.get("status"), "decisions": d_refs(" ".join(r)),
                         "detail_markdown": s["body"] if s else None, "source_file": "docs/strategy.md",
                         "line": tab["line"]})
    return rows


def _status_table(cfg: Config) -> list[dict]:
    t = _read(cfg.repo / "STATUS.md") or ""
    for tab in parse_tables(t, "STATUS.md"):
        h = [c.lower() for c in tab["header"]]
        if h[:3] == ["id", "workstream", "state"]:
            return [dict(zip(h, r), decisions=d_refs(" ".join(r)), source_file="STATUS.md", line=tab["line"])
                    for r in tab["rows"]]
    return []


def docs_list(cfg: Config) -> list[dict]:
    out = []
    for base in ("docs", "research"):
        b = cfg.repo / base
        if b.is_dir():
            for p in sorted(b.rglob("*.md")):
                r = str(p.relative_to(cfg.repo))
                first = ""
                try:
                    with p.open() as fh:
                        first = fh.readline().strip().lstrip("# ")[:160]
                except OSError:
                    pass
                out.append({"path": r, "title": first, "bytes": p.stat().st_size, "mtime": p.stat().st_mtime})
    for name in ("STATUS.md", "README.md", "AGENTS.md"):
        p = cfg.repo / name
        if p.exists():
            out.append({"path": name, "title": name, "bytes": p.stat().st_size, "mtime": p.stat().st_mtime})
    return out


def build_knowledge(cfg: Config, decisions: list[dict]) -> dict:
    status_md = _read(cfg.repo / "STATUS.md")
    return envelope("knowledge", cfg, [f for f in DOC_FILES if (cfg.repo / f).exists()],
                    decisions=decisions, n_decisions=len(decisions),
                    roadmap=_roadmap(cfg), backlog=_backlog(cfg), workstreams=_workstreams(cfg),
                    status_table=_status_table(cfg), status_markdown=status_md,
                    evidence_matrix_markdown=_read(cfg.repo / "research/reports/evidence_matrix.md"),
                    docs=docs_list(cfg))


_KN = re.compile(r"(\d+)\s*/\s*(\d+)")


def build_overview(cfg: Config, decisions: list[dict], results_meta: dict | None = None) -> dict:
    t = _read(cfg.repo / "STATUS.md") or ""
    secs = parse_sections(t)
    ev = next((s for s in secs if s["title"].startswith("evidence summary")), None)
    lim = next((s for s in secs if s["title"].startswith("blockers and known limits")), None)
    cur = next((s for s in secs if s["title"].startswith("current state")), None)
    established, key_numbers = [], []
    if ev:
        for i, b in enumerate(bullets(ev["body"])):
            m = re.match(r"\*\*(.+?)\*\*\s*(.*)", b)
            title, text = (m.group(1), m.group(2)) if m else (b[:80], b)
            interim = bool(re.search(r"\binterim\b|\bINTERIM\b|not final", b))
            refs = d_refs(b)
            kind = ("open" if title.lower().startswith("still not tested") else
                    "interim" if interim else "established")
            established.append({"title": title.rstrip("."), "text": text, "decisions": refs, "status": kind,
                                "interim": interim, "source_file": "STATUS.md", "section": ev["title"],
                                "line": ev["line"]})
            for km in _KN.finditer(b):
                a, n = int(km.group(1)), int(km.group(2))
                if n == 0 or a > n:
                    continue
                s0, s1 = max(0, km.start() - 90), min(len(b), km.end() + 40)
                key_numbers.append({"k": a, "n": n, "text": km.group(0), "context": b[s0:s1].replace("**", ""),
                                    "claim": title.rstrip("."), "decisions": refs, "interim": interim,
                                    "source_file": "STATUS.md", "line": ev["line"]})
    caveats = [{"text": b, "decisions": d_refs(b), "source_file": "STATUS.md", "line": lim["line"]}
               for b in bullets(lim["body"])] if lim else []
    open_items = [r for r in _roadmap(cfg) if r["status"] not in ("done",)]
    latest = sorted(decisions, key=lambda d: int(d["id"][2:]))[-15:][::-1]
    return envelope("overview", cfg, ["STATUS.md", "docs/experiments_roadmap.md", "research/decisions.md"],
                    status_updated=(re.search(r"Updated:\s*([^.(]+)", t).group(1).strip()
                                    if re.search(r"Updated:\s*([^.(]+)", t) else None),
                    current_state_markdown=cur["body"] if cur else None,
                    claims=established, open=open_items, caveats=caveats, key_numbers=key_numbers,
                    latest_decisions=[{k: d[k] for k in ("id", "date", "title", "source_file", "line", "refs_d",
                                                         "workstreams")} for d in latest],
                    workstreams=_status_table(cfg), results_summary=results_meta or {})
