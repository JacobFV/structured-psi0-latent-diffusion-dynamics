"""/api/dags: every run-dag ledger (host worktrees + peer copies read by --live), with node states and any ETA that
track notes state explicitly (never computed)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .common import Config, envelope, iso, sha1_bytes

_ETA = re.compile(r"\bETA\b")


def _ledger_name(path: str):
    m = re.search(r"artifacts/runs/([^/]+)/_dags/([^/]+)/ledger\.json$", path)
    return (m.group(1), m.group(2)) if m else (None, None)


def _nodes(doc: dict) -> list[dict]:
    out = []
    for nid, n in (doc.get("nodes") or {}).items():
        att = n.get("attempts") or []
        last = att[-1] if att else {}
        gate = ((n.get("metrics") or {}).get("gate") or {})
        out.append({"id": nid, "stage": n.get("stage"), "state": n.get("state"), "placement": n.get("placement"),
                    "lease": last.get("lease_id"), "unit": last.get("unit"), "log": last.get("log"),
                    "started": last.get("started"), "ended": last.get("finished"), "rc": last.get("rc"),
                    "attempts": len(att), "run_id": n.get("run_id"), "out": n.get("out"),
                    "resources": n.get("resources"), "gate_verdict": gate.get("verdict"),
                    "caveat": n.get("caveat") or n.get("note") or (f"gate {gate.get('verdict')}: {gate.get('failed')}"
                                                                   if gate.get("verdict") not in (None, "pass") else None),
                    "was_reset": "reset_from" in n, "updated": n.get("updated")})
    return out


def local_ledgers(cfg: Config) -> list[tuple[str, Path]]:
    """(location, ledger path) for the repo, the main checkout and every worktree (a cheap glob, no full scan)."""
    out = [("repo", cfg.repo)]
    if cfg.main_checkout and cfg.main_checkout.exists() and cfg.main_checkout.resolve() != cfg.repo.resolve():
        out.append(("main", cfg.main_checkout))
    if cfg.wt_root and cfg.wt_root.is_dir():
        out += [(f"wt:{p.name}", p) for p in sorted(cfg.wt_root.iterdir())
                if p.is_dir() and p.resolve() != cfg.repo.resolve()]
    res = []
    for loc, root in out:
        for f in sorted(root.glob("artifacts/runs/*/_dags/*/ledger.json")):
            res.append((loc, root, f))
    return res


def collect(cfg: Config, peer: dict | None) -> list[dict]:
    items, by_hash = [], {}
    for loc, root, f in local_ledgers(cfg):
        try:
            b = f.read_bytes()
            doc = json.loads(b)
            mt = f.stat().st_mtime
        except Exception:
            continue
        h = sha1_bytes(b)
        if h in by_hash:
            by_hash[h]["copies"] += 1
            by_hash[h]["mtime"] = max(by_hash[h]["mtime"], mt)
            if len(by_hash[h]["alt_paths"]) < 4:
                by_hash[h]["alt_paths"].append(f"{loc}:{f.relative_to(root)}")
            continue
        rel_ = str(f.relative_to(root))
        track, name = _ledger_name(rel_)
        it = {"track": track, "dag": name, "location": loc, "source_file": rel_, "copies": 1, "alt_paths": [],
              "mtime": mt, "doc": doc}
        by_hash[h] = it
        items.append(it)
    for path, e in ((peer or {}).get("ledgers") or {}).items():
        doc = e.get("doc")
        track, name = _ledger_name(path)
        items.append({"track": track, "dag": name, "location": f"peer:{cfg.peer}", "source_file": path,
                      "copies": 1, "alt_paths": [], "mtime": e.get("mtime"), "doc": doc})
    return items


def lease_workstreams(items: list[dict]) -> dict:
    out = {}
    for it in items:
        for nid, n in (it["doc"].get("nodes") or {}).items():
            for a in n.get("attempts") or []:
                if a.get("lease_id"):
                    out[a["lease_id"]] = {"track": it["track"], "dag": it["dag"], "node": nid}
    return out


def dag_outputs(items: list[dict]) -> list[tuple[str, str, str]]:
    """(node out dir, dag, state) for nodes of DAGs that are not complete: results there are interim."""
    out = []
    for it in items:
        nodes = it["doc"].get("nodes") or {}
        if all(n.get("state") == "completed" for n in nodes.values()):
            continue
        for n in nodes.values():
            if n.get("out") and n.get("state") != "completed":
                out.append((n["out"], f"{it['track']}/{it['dag']}", n.get("state")))
    return out


def build_dags(cfg: Config, items: list[dict], notes: dict[str, str]) -> dict:
    dags = []
    for it in sorted(items, key=lambda x: -(x["mtime"] or 0)):
        nodes = _nodes(it["doc"])
        counts: dict[str, int] = {}
        for n in nodes:
            counts[str(n["state"])] = counts.get(str(n["state"]), 0) + 1
        eta = []
        for fn, text in notes.items():
            if it["dag"] and (it["dag"] in text) or fn == f"research/tracks/{it['track']}.md":
                for i, ln in enumerate(text.splitlines()):
                    if _ETA.search(ln) and (it["dag"] in ln or fn == f"research/tracks/{it['track']}.md"):
                        eta.append({"text": ln.strip()[:300], "source_file": fn, "line": i + 1})
        dags.append({"track": it["track"], "dag": it["dag"], "location": it["location"],
                     "source_file": it["source_file"], "copies": it["copies"], "alt_paths": it["alt_paths"], "mtime": it["mtime"],
                     "updated": iso(it["mtime"]), "created": it["doc"].get("created"),
                     "schema_ledger": it["doc"].get("schema"), "counts": counts, "n_nodes": len(nodes),
                     "complete": bool(nodes) and all(n["state"] == "completed" for n in nodes),
                     "eta_statements": eta[-5:], "nodes": nodes})
    return envelope("dags", cfg, [d["source_file"] for d in dags], n_dags=len(dags),
                    notes=["ETA appears only where track notes state one (eta_statements, verbatim); none is computed.",
                           "Host ledgers come from every local worktree (identical copies deduplicated); peer ledgers "
                           "from the last --live read."],
                    dags=dags)
