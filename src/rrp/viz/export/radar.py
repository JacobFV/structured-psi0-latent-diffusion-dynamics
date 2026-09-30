"""Route radar (`rrp-viz/radar/v1`) built from the declared axes in `viz/radar_axes.json` (after IBM-2's radar builder).

Every value is read from the exporter's own normalized documents (results, edits, robustness in the output directory);
the config only declares selectors, floors and reference series/constants. Normalization (per axis):
    r = (x - floor) / (ref - floor)   higher is better
    r = (floor - x) / (floor - ref)   lower is better
clamped to the declared range for drawing (`drawn`); the unclamped `r` is kept. A missing input stays missing (a gap) with
the reason. Spread: the range over groups (e.g. training seeds) when a sum spans several, else the recorded interval.
Stdlib only; reads a few JSON files (host-light).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .common import Config, envelope, read_json

SCHEMA_IN = "rrp-viz/radar-axes/v1"


def _wilson(k: float, n: float):
    from rrp.harness.eval.statistics import wilson
    return list(wilson(k, n)) if n > 0 else None


def _get(row, path):
    cur = row
    for p in path:
        if isinstance(cur, dict):
            cur = cur.get(p)
        elif isinstance(cur, list) and isinstance(p, int) and p < len(cur):
            cur = cur[p]
        else:
            return None
    return cur


def _s(v) -> str:
    return "" if v is None else str(v)


def _match(row: dict, sel: dict) -> bool:
    for field, rx in (sel.get("where") or {}).items():
        if not re.search(rx, _s(_get(row, field.split(".")))):
            return False
    kp = sel.get("key_path")
    if kp is not None:
        got = row.get("key_path") or []
        if len(got) != len(kp) or not all(re.search(rx, _s(g)) for rx, g in zip(kp, got)):
            return False
    return True


def _dedupe(rows: list[dict]) -> list[dict]:
    """Identical copies of one row (the same file found in several checkouts) count once."""
    seen, out = set(), []
    for r in rows:
        key = json.dumps({k: r.get(k) for k in ("key_path", "k", "n", "effect", "rate", "metric", "variant", "seed", "body",
                                                  "route", "robot", "factor", "level", "edit")}, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _evidence(rows: list[dict]) -> dict:
    files = sorted({_s(r.get("source_file")) for r in rows if r.get("source_file")})
    return {"evidence": files, "sha1": sorted({_s(r.get("sha1")) for r in rows if r.get("sha1")}),
            "decisions": sorted({_s(r.get("decision")) for r in rows if r.get("decision")}), "n_rows": len(rows),
            "source_labels": sorted({_s(r.get("source_label")) for r in rows if r.get("source_label")})[:4]}


def resolve_value(sel: dict, docs: dict) -> dict:
    """One series value on one axis: {value, spread?, evidence…} or {missing: reason}."""
    if "missing" in sel:
        return {"missing": sel["missing"]}
    if "file" in sel:  # a committed evidence file read directly (repo-relative; docs["_repo"] is the checkout)
        repo = docs.get("_repo")
        path = (Path(repo) / sel["file"]) if repo else None
        if not path or not path.is_file():
            return {"missing": f"{sel['file']} not found"}
        raw = path.read_bytes()
        obj = json.loads(raw)
        v = _get(obj, sel["value"])
        if not isinstance(v, (int, float)):
            return {"missing": "value not recorded in the evidence file"}
        out = {"value": float(v), "evidence": [sel["file"]], "sha1": [hashlib.sha1(raw).hexdigest()[:12]], "n_rows": 1, "decisions": []}
        if sel.get("n"):
            out["n_samples"] = _get(obj, sel["n"])
        return out
    doc = docs.get(sel["doc"])
    if not isinstance(doc, dict):
        return {"missing": f"document {sel['doc']} not exported"}
    rows = [r for r in (doc.get(sel.get("list", "rows")) or []) if isinstance(r, dict) and _match(r, sel)]
    rows = _dedupe(rows)
    if not rows:
        return {"missing": "no matching row in the curated evidence"}
    agg = sel.get("agg", "sum_kn")
    out: dict = {}
    if agg == "sum_kn":
        ks = [(r.get("k"), r.get("n")) for r in rows if isinstance(r.get("k"), (int, float)) and isinstance(r.get("n"), (int, float))]
        k, n = sum(a for a, _ in ks), sum(b for _, b in ks)
        if not n:
            return {"missing": "matching rows carry no k/n"}
        out.update(value=k / n, k=k, n=n)
        gi = sel.get("group")
        if gi is not None:
            groups: dict[str, list] = {}
            for r in rows:
                groups.setdefault(_s((r.get("key_path") or [None] * (gi + 1))[gi]), []).append(r)
            if len(groups) > 1:
                rates = [sum(r["k"] for r in g) / sum(r["n"] for r in g) for g in groups.values() if sum(r["n"] for r in g)]
                out["spread"] = {"x": [min(rates), max(rates)], "meaning": f"range over {len(groups)} groups ({', '.join(sorted(groups))})"}
        if "spread" not in out:
            w = _wilson(k, n)
            if w:
                out["spread"] = {"x": w, "meaning": "Wilson 95% interval"}
    elif agg == "effect":
        r = rows[0]
        v = r.get("effect")
        if not isinstance(v, (int, float)):
            return {"missing": "row has no effect"}
        sign = -1.0 if sel.get("negate") else 1.0
        out["value"] = sign * v
        ci = r.get("ci")
        if isinstance(ci, list) and len(ci) == 2 and all(isinstance(x, (int, float)) for x in ci):
            lo, hi = sorted([sign * ci[0], sign * ci[1]])
            out["spread"] = {"x": [lo, hi], "meaning": "recorded 95% interval"}
        out["n_pairs"] = r.get("n_pairs")
    elif agg == "path":
        r = rows[0]
        v = _get(r, sel["value"])
        if not isinstance(v, (int, float)):
            return {"missing": "value not recorded in the matching row"}
        out["value"] = float(v)
        if sel.get("ci"):
            ci = _get(r, sel["ci"])
            if isinstance(ci, list) and len(ci) == 2:
                out["spread"] = {"x": [float(ci[0]), float(ci[1])], "meaning": "recorded 95% interval"}
        if sel.get("k"):
            out["k"], out["n"] = _get(r, sel["k"]), _get(r, sel["n"])
    else:
        return {"missing": f"unknown aggregation {agg}"}
    out.update(_evidence(rows))
    return out


def _norm(x, floor, ref, direction):
    if x is None or ref is None or floor is None or ref == floor:
        return None
    return (x - floor) / (ref - floor) if direction == "max" else (floor - x) / (floor - ref)


def build_radar_from(config: dict, docs: dict) -> dict:
    lo, hi = config["normalization"]["clamp"]
    series_ids = [s["id"] for s in config["series"]]
    axes_out = []
    for ax in config["axes"]:
        vals = {sid: resolve_value(ax["series"].get(sid, {"missing": "not declared for this axis"}), docs) for sid in series_ids}
        ref_decl = ax["reference"]
        if "series" in ref_decl:
            ref = vals.get(ref_decl["series"], {}).get("value")
            ref_meaning = f"{ref_decl['series']} on this axis"
        else:
            ref, ref_meaning = float(ref_decl["value"]), ref_decl.get("meaning", "declared constant")
        fdecl = ax["floor"]
        if "reference_multiple" in fdecl:
            if ref is None:
                floor = None
            else:
                floor = float(fdecl["reference_multiple"]) * ref
        else:
            floor = float(fdecl["value"])
        for sid, v in vals.items():
            if "value" not in v:
                continue
            r = _norm(v["value"], floor, ref, ax["direction"])
            if r is None:
                v["missing_r"] = "reference value missing or equal to the floor"
                continue
            v["r"] = round(r, 4)
            v["drawn"] = round(min(hi, max(lo, r)), 4)
            if "spread" in v:
                a, b = (_norm(x, floor, ref, ax["direction"]) for x in v["spread"]["x"])
                ra = sorted([a, b])
                v["spread"]["r"] = [round(x, 4) for x in ra]
                v["spread"]["drawn"] = [round(min(hi, max(lo, x)), 4) for x in ra]
            v["value"] = round(v["value"], 4)
        axes_out.append({k: ax[k] for k in ("id", "label", "metric", "direction", "protocol", "decision")} |
                        {"floor": fdecl | {"value": None if floor is None else round(floor, 4)}} |
                        {"reference": ref_decl | {"resolved_value": ref, "meaning": ref_meaning}, "series": vals})
    return {"normalization": config["normalization"], "series": config["series"], "axes": axes_out,
            "n_axes": len(axes_out), "n_values": sum(1 for a in axes_out for v in a["series"].values() if "drawn" in v)}


def build_radar(cfg: Config) -> dict:
    cfg_path = Path(cfg.repo) / "viz/radar_axes.json"
    config = read_json(cfg_path)
    if not isinstance(config, dict) or config.get("schema") != SCHEMA_IN:
        return envelope("radar", cfg, [], config="viz/radar_axes.json", axes=[], series=[], n_axes=0, n_values=0,
                        missing=[f"{cfg_path} missing or not {SCHEMA_IN}"])
    docs = {name: read_json(cfg.out / f"{name}.json") for name in ("results", "edits", "robustness")}
    docs["_repo"] = str(cfg.repo)
    body = build_radar_from(config, docs)
    sources = [str(cfg_path)] + sorted({f for a in body["axes"] for v in a["series"].values() for f in v.get("evidence", [])})
    return envelope("radar", cfg, sources, config="viz/radar_axes.json", **body)
