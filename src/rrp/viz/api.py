"""Helper for the room's Vite API plugin (viz/CONTRACT.md "plugin interface"). Read-only; stdlib only.

The plugin shells out (cwd = repo root, PYTHONPATH=src, like IBM-2's progress plugin) and caches the result:
  python -m rrp.cli viz api get <name> [--max-age 15] [--live]   -> prints the absolute path of viz/data/<name>.json after
                                                                re-exporting it if older than --max-age seconds
  python -m rrp.cli viz api replay <id>                           -> prints the absolute path of a replay file (exit 2)
  python -m rrp.cli viz api media <name>                          -> prints the absolute path of a video (exit 2)
  python -m rrp.cli viz api training <id>                         -> prints the absolute path of one training series
  python -m rrp.cli viz api routes                                -> prints the route table as JSON
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from rrp.viz.export import DOCS, run
from rrp.viz.export.common import Config, read_json

ROUTES = {**{f"/api/{n}": f"{n}.json" for n in DOCS},
          "/api/replay/<id>": "api replay <id>", "/api/training/<id>": "training/<id>.json",
          "/media/<name>": "api media <name>"}
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.\-]{1,200}$")
LIVE_MAX_AGE_S = 10
DEFAULT_MAX_AGE_S = 15


def _cfg(repo: Path | str = ".", out: Path | str | None = None, live: bool = False) -> Config:
    repo = Path(repo).resolve()
    return Config.from_env(repo, Path(out) if out else repo / "viz/data", live=live)


def ensure(cfg: Config, name: str, max_age_s: float | None = None) -> Path:
    """Path of viz/data/<name>.json, re-exported first when older than max_age_s (live: 10 s, others: 15 s)."""
    if name not in DOCS:
        raise KeyError(name)
    max_age = max_age_s if max_age_s is not None else (LIVE_MAX_AGE_S if name == "live" else DEFAULT_MAX_AGE_S)
    p = cfg.out / f"{name}.json"
    if not p.exists() or time.time() - p.stat().st_mtime > max_age:
        run(cfg, [name])
    return p


def replay_file(cfg: Config, rid: str) -> Path | None:
    if not _SAFE_ID.match(rid or ""):
        return None
    root = (cfg.rrp_data / "viz/replays") if cfg.rrp_data else None
    idx = read_json(root / "index.json") if root else None
    rows = (idx.get("replays") if isinstance(idx, dict) else idx) or []
    for r in rows:
        if r.get("id") == rid and r.get("file"):
            p = (root / r["file"]).resolve()
            try:
                p.relative_to(root.resolve())
            except ValueError:
                return None
            return p if p.is_file() else None
    return None


def media_file(cfg: Config, name: str) -> Path | None:
    if not _SAFE_ID.match(name or "") or not name.endswith((".mp4", ".webm", ".gif", ".png", ".jpg")):
        return None
    for d in [cfg.repo / "artifacts/video"] + ([cfg.main_checkout / "artifacts/video"] if cfg.main_checkout else []):
        p = d / name
        if p.is_file():
            return p.resolve()
    return None


def training_file(cfg: Config, rid: str) -> Path | None:
    if not _SAFE_ID.match(rid or ""):
        return None
    p = cfg.out / "training" / f"{rid}.json"
    return p if p.is_file() else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rrp viz api", description="room API helper (read-only)")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("get")
    g.add_argument("name", choices=DOCS)
    g.add_argument("--max-age", type=float, default=None)
    g.add_argument("--live", action="store_true")
    for c in ("replay", "media", "training"):
        sub.add_parser(c).add_argument("arg")
    sub.add_parser("routes")
    a = ap.parse_args(argv)
    cfg = _cfg(a.repo, a.out, live=getattr(a, "live", False))
    if a.cmd == "routes":
        print(json.dumps(ROUTES, indent=1))
        return 0
    if a.cmd == "get":
        print(ensure(cfg, a.name, a.max_age))
        return 0
    fn = {"replay": replay_file, "media": media_file, "training": training_file}[a.cmd]
    p = fn(cfg, a.arg)
    if p is None:
        return 2
    print(p)
    return 0
