"""Room data exporter (viz/CONTRACT.md, D-131): `python -m rrp.cli viz export [--live] --out viz/data [--only name]`.

Writes one JSON document per API path (overview, live, dags, results, edits, training, robustness, physics, psi0,
knowledge, replays, videos) plus training/<id>.json series and _manifest.json (timings, sizes, row counts, errors).
Host-light: stdlib only, one process, caches every source by (path, mtime, size) under <out>/_cache. The peer is read
only with --live, by ONE bounded ssh call; without it the cached peer read is re-served and marked stale.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

from .common import Config, FileCache, envelope, read_json, write_json

DOCS = ("overview", "live", "dags", "results", "edits", "training", "robustness", "physics", "psi0", "knowledge",
        "replays", "videos", "radar")
_SCAN_DOCS = {"results", "edits", "training", "robustness", "physics"}


def _rows(doc: dict) -> int | None:
    for k in ("rows", "runs", "dags", "decisions", "videos", "replays", "leases", "claims", "levels", "trackers"):
        v = doc.get(k)
        if isinstance(v, list):
            return len(v)
    return None


def run(cfg: Config, only: list[str] | None = None, sync_psi1z: bool = False) -> dict:
    """Serialized by a flock on <out>/_cache/lock (the plugin may start a live and a full export at once)."""
    import fcntl
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.cache_dir / "lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        return _run(cfg, only, sync_psi1z)


def _run(cfg: Config, only: list[str] | None = None, sync_psi1z: bool = False) -> dict:
    from . import dags as dags_m
    from . import knowledge, live, media, psi0, results
    t_all = time.time()
    want = set(only or DOCS)
    unknown = want - set(DOCS)
    if unknown:
        raise SystemExit(f"unknown document(s): {sorted(unknown)}; choose from {DOCS}")
    cfg.out.mkdir(parents=True, exist_ok=True)
    manifest = read_json(cfg.out / "_manifest.json", {}) or {}
    docs_m = manifest.get("documents", {})
    timings: dict[str, float] = {}

    def emit(name, doc):
        t = time.time()
        n = write_json(cfg.out / f"{name}.json", doc)
        docs_m[name] = {"generated_at": doc.get("generated_at"), "bytes": n, "rows": _rows(doc),
                        "seconds": round(timings.get(name, 0) + time.time() - t, 3), "error": None,
                        "file": f"{name}.json"}

    def timed(name, fn):
        t = time.time()
        try:
            out = fn()
        except Exception as e:  # noqa: BLE001 — one failing source must not take down the other documents
            docs_m[name] = {"generated_at": None, "error": f"{type(e).__name__}: {e}",
                            "traceback": traceback.format_exc()[-2000:]}
            timings[name] = time.time() - t
            return None
        timings[name] = time.time() - t
        return out

    if sync_psi1z:
        manifest["psi1z_sync"] = psi0.sync_psi1z(cfg)
    t = time.time()
    decisions = knowledge.load_decisions(cfg)
    notes = knowledge.load_track_notes(cfg)
    dec = knowledge.DecisionIndex(decisions, notes)
    timings["_decisions"] = time.time() - t

    state = None
    if want & {"live", "dags"} or want & _SCAN_DOCS:
        t = time.time()
        state = live.peer_state(cfg)
        timings["_peer"] = time.time() - t
    peer_data = (state or {}).get("prev", {}) or {}
    peer_data = peer_data.get("data") if peer_data.get("ok") else None
    items = dags_m.collect(cfg, peer_data) if state is not None else []

    if "live" in want:
        d = timed("live", lambda: live.build_live(cfg, state, dags_m.lease_workstreams(items)))
        if d:
            emit("live", d)
    if "dags" in want:
        d = timed("dags", lambda: dags_m.build_dags(cfg, items, notes))
        if d:
            emit("dags", d)

    results_meta = (manifest.get("results_meta") or {})
    if want & _SCAN_DOCS:
        from .scan import discover_cached
        t = time.time()
        cache = FileCache(cfg.cache_dir / "files.pkl")
        timings["_cache_load"] = time.time() - t
        t = time.time()
        found, reused = discover_cached(cfg, cache, force=want >= _SCAN_DOCS)
        timings["_discover"] = time.time() - t
        manifest["discovery_reused"] = reused
        dec.set_generic_stems(f.rel for k in ("json", "md") for f in found[k])
        manifest["discovered"] = {k: len(v) for k, v in found.items()}
        dag_outs = dags_m.dag_outputs(items)
        res_doc = None
        if want & {"results", "robustness"}:
            r = timed("results", lambda: results.build_results(cfg, cache, found, dec, dag_outs, peer_data))
            if r:
                res_doc, results_meta = r
                manifest["results_meta"] = results_meta
                if "results" in want:
                    emit("results", res_doc)
        if "edits" in want:
            d = timed("edits", lambda: results.build_edits(cfg, cache, found, dec))
            if d:
                emit("edits", d)
        if "robustness" in want and res_doc is not None:
            d = timed("robustness", lambda: results.build_robustness(cfg, cache, found, dec, res_doc["rows"]))
            if d:
                emit("robustness", d)
        if "physics" in want:
            d = timed("physics", lambda: results.build_physics(cfg, cache, found, dec))
            if d:
                emit("physics", d)
        if "training" in want:
            def _tr():
                clip = []
                for f in found["json"]:
                    if "clipscale" in f.rel:
                        from .results import content
                        from .scan import provenance
                        o = content(cfg, cache, f).get("obj")
                        if o is not None:
                            clip.append({"doc": o, **provenance(f, dec)})
                return results.build_training(cfg, cache, found, dec, clip)
            r = timed("training", _tr)
            if r:
                doc, series = r
                emit("training", doc)
                sd = cfg.out / "training"
                sd.mkdir(exist_ok=True)
                keep = set()
                for rid, s in series.items():
                    p = sd / f"{rid}.json"
                    keep.add(p.name)
                    old = read_json(p) if p.exists() else None
                    body = envelope("training-series", cfg, [s["source_file"]], **s)
                    if old is None or {k: v for k, v in old.items() if k not in ("generated_at", "generated_at_unix",
                                                                                 "git_sha")} != \
                            {k: v for k, v in body.items() if k not in ("generated_at", "generated_at_unix", "git_sha")}:
                        write_json(p, body)
                for p in sd.glob("*.json"):
                    if p.name not in keep:
                        p.unlink()
        cache.save(prune_kinds=("x", "md", "tl") if want >= _SCAN_DOCS else ())
    if "overview" in want:
        d = timed("overview", lambda: knowledge.build_overview(cfg, decisions, results_meta))
        if d:
            emit("overview", d)
    if "knowledge" in want:
        d = timed("knowledge", lambda: knowledge.build_knowledge(cfg, decisions))
        if d:
            emit("knowledge", d)
    if "psi0" in want:
        d = timed("psi0", lambda: psi0.build_psi0(cfg, decisions))
        if d:
            emit("psi0", d)
    if "videos" in want:
        d = timed("videos", lambda: media.build_videos(cfg))
        if d:
            emit("videos", d)
    if "replays" in want:
        d = timed("replays", lambda: media.build_replays(cfg))
        if d:
            emit("replays", d)
    if "radar" in want:  # after results/edits/robustness: it reads their exported files
        from . import radar
        d = timed("radar", lambda: radar.build_radar(cfg))
        if d:
            emit("radar", d)
    manifest.update(schema="rrp-viz/manifest/v1", documents=docs_m, last_run={
        "only": sorted(want) if only else None, "live": cfg.live, "seconds": round(time.time() - t_all, 3),
        "timings": {k: round(v, 3) for k, v in timings.items()}, "finished_at": time.time()})
    write_json(cfg.out / "_manifest.json", manifest, indent=1)
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rrp viz export", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="viz/data", help="output directory (gitignored)")
    ap.add_argument("--repo", default=".", help="repository checkout to read (default: cwd)")
    ap.add_argument("--live", action="store_true", help="read the peer (one bounded ssh call); otherwise never touch it")
    ap.add_argument("--only", action="append", help=f"document name (repeatable or comma-separated): {', '.join(DOCS)}")
    ap.add_argument("--sync-psi1z", action="store_true",
                    help="rsync small psi1z summary files from the peer into ~/work/rrp-data/viz/psi1z (≤ 50 MB)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    only = [x for o in (a.only or []) for x in o.split(",") if x] or None
    cfg = Config.from_env(Path(a.repo), Path(a.out), live=a.live)
    m = run(cfg, only, sync_psi1z=a.sync_psi1z)
    errs = {k: v["error"] for k, v in m["documents"].items() if v.get("error") and (not only or k in only)}
    if not a.quiet:
        lr = m["last_run"]
        print(f"rrp.viz.export: {lr['seconds']} s -> {cfg.out}  " +
              " ".join(f"{k}={v.get('rows')}" for k, v in m["documents"].items() if not only or k in only))
        for k, v in errs.items():
            print(f"  ERROR {k}: {v}", file=sys.stderr)
    return 1 if errs else 0
