"""/api/videos (artifacts/video/INDEX.md, labels as written) and /api/replays (the replay recorder's index.json)."""
from __future__ import annotations

import re
from pathlib import Path

from .common import Config, bodies_in, d_refs, envelope, read_json

_LINE = re.compile(r"^- `([^`]+)`\s*[—-]+\s*(.*)$")
_KV = re.compile(r"\b(source|ckpt|robot|robots|task|seed|outcome)=(\S+)")


def video_label(name: str, desc: str) -> str:
    """The unmistakable source class of a clip, from its own description (INDEX.md)."""
    s = (name + " " + desc).lower()
    kinds = []
    if "oracle" in s:
        kinds.append("oracle")
    if "scripted_teacher" in s or "teacher" in name.lower():
        kinds.append("scripted_teacher")
    if "learned" in s or "bc:" in s or "_bc_" in s:
        kinds.append("learned")
    if "privileged" in s and not kinds:
        kinds.append("privileged")
    if "triptych" in s or "side-by-side" in s or "side by side" in s:
        kinds.append("comparison")
    return "+".join(dict.fromkeys(kinds)) or "unlabelled"


def build_videos(cfg: Config) -> dict:
    idx = cfg.repo / "artifacts/video/INDEX.md"
    rows = []
    dirs = [cfg.repo / "artifacts/video"] + ([cfg.main_checkout / "artifacts/video"] if cfg.main_checkout else [])
    try:
        lines = idx.read_text().splitlines()
    except OSError:
        lines = []
    for i, ln in enumerate(lines):
        m = _LINE.match(ln.strip())
        if not m:
            continue
        name, desc = m.group(1), m.group(2)
        kv = dict(_KV.findall(desc))
        where = next((d for d in dirs if (d / name).exists()), None)
        date = name[:10] if re.match(r"\d{4}-\d{2}-\d{2}", name) else None
        rows.append({"name": name, "description": desc, "label": video_label(name, desc),
                     "source": kv.get("source"), "ckpt": kv.get("ckpt"), "robot": kv.get("robot") or kv.get("robots"),
                     "bodies": bodies_in(name + " " + desc)[:4], "task": kv.get("task"), "seed": kv.get("seed"),
                     "outcome": kv.get("outcome") or (re.search(r"_(success|failure[\w-]*|followed)\.mp4$", name) or
                                                      [None, None])[1],
                     "date": date, "decisions": d_refs(desc), "exists_locally": where is not None,
                     "bytes": (where / name).stat().st_size if where else None,
                     "url": f"/media/{name}" if where else None, "source_file": "artifacts/video/INDEX.md",
                     "line": i + 1})
    return envelope("videos", cfg, ["artifacts/video/INDEX.md"], n=len(rows),
                    n_missing_files=sum(not r["exists_locally"] for r in rows),
                    notes=["label is derived from the clip's own INDEX.md description; the description is shown verbatim."],
                    videos=rows)


def replays_dir(cfg: Config) -> Path | None:
    return cfg.rrp_data / "viz/replays" if cfg.rrp_data else None


def build_replays(cfg: Config) -> dict:
    d = replays_dir(cfg)
    idx = read_json(d / "index.json") if d else None
    rows = []
    if isinstance(idx, dict):
        rows = idx.get("replays") or []
    elif isinstance(idx, list):
        rows = idx
    out = []
    for r in rows:
        f = r.get("file")
        exists = bool(d and f and (d / f).exists())
        out.append({**r, "exists_locally": exists, "url": f"/api/replay/{r.get('id')}" if exists else None,
                    "source_file": f"~/work/rrp-data/viz/replays/{f}" if f else None})
    return envelope("replays", cfg, [str(d / "index.json")] if idx is not None else [], n=len(out),
                    index_generated_at=idx.get("generated_at") if isinstance(idx, dict) else None,
                    missing=[] if idx is not None else ["no replay index yet (~/work/rrp-data/viz/replays/index.json)"],
                    replays=out)
