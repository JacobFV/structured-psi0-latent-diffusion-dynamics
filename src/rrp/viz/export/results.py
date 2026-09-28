"""results / edits / robustness / physics / training documents from the discovered files (scan.py)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .common import (Config, FileCache, bodies_in, sha1_bytes, envelope, family_of, parse_tables, rnd, source_label, wilson)
from .scan import Found, extract_edits, extract_results, parse_trainlog, provenance

SMALL_DOC = 64_000
_SEED = re.compile(r"(?:^|[_/\-])(?:s|seed|trainseed)(\d{1,2})(?=[_/.\-]|$)")
_VARIANTS = ("frozen_sem", "semfix", "fixsem", "nosem", "sem")
_VAR_RE = re.compile(r"(?:^|[_/\-])(frozen_sem|semfix|fixsem|nosem|sem)(?=[_/.\-\d]|$)")
_TASKS = ("pick_place_paired", "pick_place", "support_insert", "handover", "assign_left", "assign_right", "locomotion")
_ROUTE_TOK = [("generated", "generated"), ("oracle", "oracle"), ("teacher", "teacher"), ("learned_bc", "bc"),
              ("learned", "learned"), ("r2", "r2"), ("r1", "r1"), ("bc", "bc")]
_GRASP_P = re.compile(r"(?:^|[/_\-])(grasp_v\d(?:\.\d)?)(?=$|[/_\-])")
_CONTACT_P = re.compile(r"(?:^|[/_\-])(contact_v\d)(?=$|[/_\-])")
EDIT_PATH = re.compile(r"effect|edit|compare|summary_contact|variant|permutation|semantic", re.I)
PHYS_PATH = re.compile(r"contact_v2|contact/|trackers/|gate|grasp|gc2eval|w12|fallrate|validation|eligib|slope|clipscale",
                       re.I)

# Caveats that belong to a run lineage (recorded in decisions / track notes), matched on path + key path.
CAVEAT_RULES = [
    (re.compile(r"legged8-go2-semfix|legged8_compare_go2.*semfix"),
     "inexact resume in lineage: semfix flow s0 from step 1500, semfix Stage A s1/s2 from step 5500 (pre-CUDA-RNG-fix "
     "checkpoints)", "D-113"),
    (re.compile(r"(^|/)(partial|interim)|_partial|compare_partial"), "partial/interim file", None),
]
INTERIM_RULES = [
    (re.compile(r"partial|interim"), "file marked partial/interim", None),
    (re.compile(r"armexpert_gc2eval/(?!compare_gc2_final)"), "grasp_v2 re-evaluation of old arm routes (interim)", "D-121"),
    (re.compile(r"legged8[-_/].*t1|t1sl|legged_v2_t1"), "t1 sourced-limit lineage is interim (W8 track notes)", "D-113"),
]
T1_LABEL = ("t1 dataset fails D-112 slip gate (86.2% < 95%), tracker w8d fails lab forward 0.72 (D-113 exception); "
            "learned t1 routes mostly fall (D-114)")


def _load(f: Found):
    try:
        return json.loads(f.abs.read_text())
    except Exception:
        return None


def content(cfg: Config, cache: FileCache, f: Found) -> dict:
    """Cached content products of one json file: results rows, edit rows, and the object itself when small and
    physics/robustness-relevant."""
    p = cache.product("x", f.sha)
    if p is not None:
        return p
    obj = _load(f)
    if obj is None:
        p = {"error": "unreadable json"}
    else:
        p = {"results": extract_results(obj)}
        if EDIT_PATH.search(f.rel):
            p["edits"] = extract_edits(obj)
        if f.size <= SMALL_DOC and (PHYS_PATH.search(f.rel) or "robust" in f.rel):
            p["obj"] = obj
        if f.rel.endswith("robustness_report.json") or re.search(r"robust/variant_.*\.json$", f.rel):
            p["obj"] = obj
    cache.put("x", f.sha, p)
    return p


def _dag_interim(dag_outs: list[tuple[str, str, str]], rel: str):
    for out, dag, state in dag_outs:
        if rel.startswith(out.rstrip("/") + "/") or rel == out:
            return f"run-dag {dag}: node state {state}"
    return None


def _variant_seed(rel: str, kp: list[str]):
    variant = seed = None
    if "variants" in kp:
        i = kp.index("variants")
        if i + 1 < len(kp):
            variant = kp[i + 1]
        if i + 2 < len(kp) and kp[i + 2].isdigit():
            seed = int(kp[i + 2])
    s = rel.lower()
    if variant is None:
        m = _VAR_RE.search(s)
        variant = m.group(1) if m else None
        if variant is None:
            for k in kp:
                if k in _VARIANTS:
                    variant = k
                    break
    if seed is None:
        m = _SEED.search(s)
        seed = int(m.group(1)) if m else None
    return variant, seed


_KP_ROUTES = {"teacher": "teacher", "scripted_teacher": "teacher", "oracle": "oracle", "generated": "generated",
              "bc": "bc", "r2": "r2", "r1": "r1", "r2_final": "r2", "r2_snap": "r2", "r2_snap_s4000": "r2"}


def _route(rel: str, ctx: dict, kp: list[str] | None = None) -> tuple[str | None, str | None]:
    r = ctx.get("route")
    if isinstance(r, str):
        return r, "json"
    for k in reversed(kp or []):
        if k.lower() in _KP_ROUTES:
            return _KP_ROUTES[k.lower()], "key_path"
    stem = rel.rsplit("/", 1)[-1].lower()
    for tok, name in _ROUTE_TOK:
        if stem.startswith(tok + "_") or stem.startswith(tok + "."):
            return name, "path"
    return None, None


def _label(ctx: dict, route: str | None) -> tuple[str | None, str]:
    s = ctx.get("source")
    if s is not None:
        return source_label(s), "recorded"
    pl = ctx.get("policy_label")
    if route in ("teacher",):
        return "scripted_teacher", "route"
    if route == "oracle":
        return "oracle", "route"
    if route in ("learned", "bc") and pl:
        return f"learned:{pl}", "route+policy_label"
    if route == "generated":
        return "learned:generated", "route"
    return None, "missing"


def norm_rows(cfg: Config, f: Found, prod: dict, dec, dag_outs) -> list[dict]:
    rows = []
    prov = provenance(f, dec)
    base_interim = _dag_interim(dag_outs, f.rel)
    for i, r in enumerate(prod.get("results", {}).get("rows", [])):
        ctx, kp = r["ctx"], r["key_path"]
        kps = "/".join(kp)
        body = ctx.get("body") if isinstance(ctx.get("body"), str) else None
        body_from = "json" if body else None
        if body is None:
            for src, name in ((kps, "key_path"), (f.rel, "path")):
                b = bodies_in(src)
                if b:
                    body, body_from = b[0], name
                    break
        fam = family_of(body, f.rel + "/" + kps)
        short = None
        if body is None:
            from .common import ARM_SHORT
            m = ARM_SHORT.search(kps) or ARM_SHORT.search(f.rel)
            short = m.group(1) if m else None
        route, route_from = _route(f.rel, ctx, kp)
        variant, seed = _variant_seed(f.rel, kp)
        lab, lab_from = _label(ctx, route)
        task = ctx.get("task") if isinstance(ctx.get("task"), str) else next((t for t in _TASKS if t in f.rel), None)
        grasp, contact, act = ctx.get("grasp"), ctx.get("contact"), ctx.get("actuator")
        version_from = {}
        if r["metric"] in ("grasp_v1", "grasp_v2", "grasp_v2.1"):
            grasp, version_from["grasp"] = r["metric"], "key_path"
        elif grasp is not None:
            version_from["grasp"] = "json"
        else:
            m = _GRASP_P.search(f.rel + "/" + kps)
            if m:
                grasp, version_from["grasp"] = m.group(1), "path"
        if contact is not None:
            version_from["contact"] = "json"
        else:
            m = _CONTACT_P.search(f.rel + "/" + kps)
            if m:
                contact, version_from["contact"] = m.group(1), "path"
        caveats = []
        both = f.rel + "|" + kps
        for rx, text, d in CAVEAT_RULES:
            if rx.search(both):
                caveats.append(text + (f" ({d})" if d else ""))
        if body == "t1" and fam == "legged" and (contact == "contact_v2" or "legged8" in f.rel or "t1sl" in f.rel):
            caveats.append(T1_LABEL)
        if fam == "arm":
            g = grasp if isinstance(grasp, str) else (grasp[0] if isinstance(grasp, list) and len(grasp) == 1 else grasp)
            if g is None:
                caveats.append("grasp contact version not recorded in this file (runs before D-110 used grasp_v1: "
                                "grasps held by "
                               "interpenetration, D-108)")
            elif g == "grasp_v1":
                caveats.append("grasp_v1 physics (interpenetration, D-108); superseded by grasp_v2/v2.1 (D-110, D-118)")
        if fam == "legged":
            c = contact if isinstance(contact, str) else (contact[0] if isinstance(contact, list) and len(contact) == 1
                                                          else contact)
            if c is None:
                caveats.append("contact version not recorded in this file (legged runs before D-101 used "
                                "contact_v1: skating "
                               "trackers, D-093)")
            elif c == "contact_v1":
                caveats.append("contact_v1 physics (skating trackers, D-093)")
        interim_reason = base_interim
        if interim_reason is None:
            for rx, text, d in INTERIM_RULES:
                if rx.search(both):
                    interim_reason = text + (f" ({d})" if d else "")
                    break
        rows.append({
            "id": f"{f.sha[:10]}:{i}", "family": fam, "task": task, "body": body, "body_from": body_from,
            "body_hint": short, "route": route, "route_from": route_from, "variant": variant, "seed": seed, "eval_seeds": ctx.get("seeds"),
            "condition": ctx.get("condition") or ctx.get("edit"), "grasp_version": grasp, "contact_version": contact,
            "actuator_version": act, "version_from": version_from or None, "metric": r["metric"], "k": r["k"], "n": r["n"], "rate": r["rate"],
            "ci_lo": r["ci"][0] if r.get("ci") else None, "ci_hi": r["ci"][1] if r.get("ci") else None,
            "ci_method": r.get("ci_method"), "fell": r.get("fell"), "public_success": r.get("public_success"),
            "stages": r.get("stages"), "text": r.get("text"), "key_path": kp,
            "source_label": lab, "source_label_from": lab_from,
            "interim": interim_reason is not None, "interim_reason": interim_reason,
            "caveat": "; ".join(caveats) if caveats else None, **prov})
    return rows


def peer_only_files(cfg: Config, found: dict, peer: dict | None) -> list[tuple[Found, dict]]:
    """Small summaries from the last --live read (modified in the last 24 h on the peer) whose path exists in NO local copy."""
    local = {f.rel for f in found["json"]}
    local |= {a.split(":", 2)[-1] for f in found["json"] for a in getattr(f, "alt", [])}
    out = []
    for e in (peer or {}).get("recent_summaries") or []:
        if e.get("path") in local or not isinstance(e.get("doc"), (dict, list)):
            continue
        b = json.dumps(e["doc"], sort_keys=True).encode()
        f = Found("json", Path(e["path"]), e["path"], f"peer:{cfg.peer}", e.get("mtime") or 0, len(b), sha1_bytes(b))
        out.append((f, e["doc"]))
    return out


def build_results(cfg: Config, cache: FileCache, found: dict, dec, dag_outs,
                  peer: dict | None = None) -> tuple[dict, dict]:
    rows, truncated, errors = [], [], []
    for f in found["json"]:
        prod = content(cfg, cache, f)
        if "error" in prod:
            errors.append({"source_file": f.rel, "location": f.loc, "error": prod["error"]})
            continue
        if prod["results"].get("truncated"):
            truncated.append(f.rel)
        rows.extend(norm_rows(cfg, f, prod, dec, dag_outs))
    peer_only = peer_only_files(cfg, found, peer)
    for f, obj in peer_only:
        prod = {"results": extract_results(obj)}
        for r in norm_rows(cfg, f, prod, dec, dag_outs):
            r["peer_only"] = True
            r["caveat"] = "; ".join(x for x in (r["caveat"], "peer-only copy (not yet on the host), from the last --live read") if x)
            rows.append(r)
    tables = []
    for f in found["md"]:
        p = cache.product("md", f.sha)
        if p is None:
            try:
                p = parse_tables(f.abs.read_text(), f.rel)
            except OSError:
                p = []
            cache.put("md", f.sha, p)
        prov = provenance(f, dec)
        for t in p:
            if not t["rows"]:
                continue
            tables.append({**t, **prov, "line": t["line"]})
    fams: dict[str, int] = {}
    for r in rows:
        fams[str(r["family"])] = fams.get(str(r["family"]), 0) + 1
    meta = {"n_rows": len(rows), "n_files": len(found["json"]), "by_family": fams,
            "n_interim": sum(r["interim"] for r in rows), "n_with_caveat": sum(bool(r["caveat"]) for r in rows),
            "n_tables": len(tables), "n_peer_only_files": len(peer_only)}
    doc = envelope("results", cfg, sorted({r["source_file"] for r in rows}),
                   **meta, truncated_files=truncated, unreadable=errors,
                   columns=["id", "family", "task", "body", "route", "variant", "seed", "grasp_version",
                            "contact_version", "actuator_version", "metric", "k", "n", "rate", "ci_lo", "ci_hi",
                            "source_label", "source_file", "decision", "interim", "caveat"],
                   notes=["Rows are extracted from every result-shaped node ({n, success}, {k, n}, [k, n], 'k/n') of every "
                          "summary/compare/gate JSON; key_path locates the node inside source_file.",
                          "body/route/variant/seed come from the JSON when recorded, else from the key path or file path "
                          "(body_from / source_label_from say which); missing stays null.",
                          "decision = latest D-entry whose text mentions this file or its run directory "
                          "(decision_match is the token); decisions lists all matches.",
                          "ci_method 'wilson95_computed' = Wilson 95% interval computed from k/n here; 'recorded' = "
                          "copied from the file."],
                   rows=rows, tables=tables)
    return doc, meta


def build_edits(cfg: Config, cache: FileCache, found: dict, dec) -> dict:
    from .scan import edit_role
    rows = []
    for f in found["json"]:
        prod = content(cfg, cache, f)
        ed = prod.get("edits")
        if not ed:
            continue
        prov = provenance(f, dec)
        for i, r in enumerate(ed["rows"]):
            kp = r["key_path"]
            kps = "/".join(kp)
            b = bodies_in(kps) or bodies_in(f.rel)
            variant, seed = _variant_seed(f.rel, kp)
            role = edit_role(r["edit"])
            if role == "reference" and r.get("kind") == "effect" and r.get("effect") == 0:
                continue
            rows.append({"id": f"{f.sha[:10]}:e{i}", "body": b[0] if b else None, "variant": variant, "seed": seed,
                         "edit": r["edit"], "role": role, "control": role == "control", "metric": r.get("metric"),
                         "effect": r.get("effect"), "ci": r.get("ci"), "n_pairs": r.get("n_pairs"),
                         "p_one_sided": r.get("p_one_sided"), "p_two_sided": r.get("p_two_sided"),
                         "n_perm": r.get("n_perm"), "direction": r.get("direction"),
                         "every_seed_ordered": r.get("every_seed_ordered"), "kind": r["kind"], "key_path": kp,
                         "context_or_packet": ("context" if kps.lower().startswith(("ctx", "effects/ctx")) or "/ctx" in
                                               kps.lower() or "r2ctx" in f.rel else
                                               "packet" if "effects/z " in kps.lower() or "/z " in kps.lower() else None),
                         **prov})
    return envelope("edits", cfg, sorted({r["source_file"] for r in rows}), n_rows=len(rows),
                    notes=["effect = mean paired difference (edited − unedited) with its recorded 95% interval [lo, hi].",
                           "role: 'control' for inactive/random/irrelevant edits, 'difference' for recorded "
                           "edit-minus-control contrasts, 'edit' otherwise. Permutation rows carry the recorded exact "
                           "p-values (n_perm = number of label permutations)."],
                    rows=rows)


def build_robustness(cfg: Config, cache: FileCache, found: dict, dec, results_rows: list[dict]) -> dict:
    tables, comparisons, variant_diffs, levels = [], [], [], []
    for f in found["json"]:
        prod = content(cfg, cache, f)
        obj = prod.get("obj")
        prov = provenance(f, dec)
        if f.rel.endswith("robustness_report.json") and isinstance(obj, dict):
            for route, per in (obj.get("table") or {}).items():
                for robot, t in (per or {}).items():
                    nom = t.get("nominal") or {}
                    for fac, fd in (t.get("factors") or {}).items():
                        for lv in fd.get("levels", []):
                            levels.append({"report": f.rel, "route": route, "robot": robot, "factor": fac,
                                           "level": lv.get("level"), "key": lv.get("key"), "k": lv.get("success"),
                                           "n": lv.get("n"), "rate": rnd(lv.get("rate")),
                                           "ci": [rnd(x) for x in lv.get("wilson95", [])] or None,
                                           "falls": lv.get("falls"), "drop": rnd(lv.get("drop")),
                                           "public_success": (lv.get("task") or {}).get("public_success"),
                                           "motion": {k: rnd(v) for k, v in (lv.get("motion") or {}).items()},
                                           "break_point": fd.get("break_point"), **prov})
                    tables.append({"route": route, "robot": robot, "nominal": {
                        "k": nom.get("success"), "n": nom.get("n"), "rate": rnd(nom.get("rate")),
                        "ci": [rnd(x) for x in nom.get("wilson95", [])] or None, "falls": nom.get("falls"),
                        "public_success": nom.get("public_success"),
                        "motion": {k: rnd(v) for k, v in (nom.get("motion") or {}).items()},
                        "sources": nom.get("sources")},
                        "break_points": {fac: fd.get("break_point") for fac, fd in (t.get("factors") or {}).items()
                                         if "break_point" in fd},
                        "pooled_perturbed": t.get("pooled_perturbed"), **prov})
            for c in obj.get("comparisons") or []:
                comparisons.append({**{k: rnd(v) if not isinstance(v, list) else [rnd(x) for x in v]
                                       for k, v in c.items()}, **prov})
        elif re.search(r"robust/variant_.*\.json$", f.rel) and isinstance(obj, dict):
            variant_diffs.append({"robot": obj.get("robot"), "a": obj.get("a"), "b": obj.get("b"),
                                  "training_seeds": obj.get("training_seeds"), "metrics": obj.get("metrics"), **prov})
    per_level = [r for r in results_rows if "/robust/" in "/" + r["source_file"] and r.get("condition")]
    return envelope("robustness", cfg, sorted({t["source_file"] for t in tables + comparisons + variant_diffs}),
                    notes=["Break-points are shown only where a report records them; none are derived here."],
                    reports=tables, levels=levels, comparisons=comparisons, variant_diffs=variant_diffs,
                    n_per_level_summary_rows=len(per_level),
                    per_level_rows=[{k: r[k] for k in ("family", "body", "route", "variant", "seed", "condition", "k",
                                                       "n", "rate", "ci_lo", "ci_hi", "source_label", "source_file",
                                                       "decision", "interim", "caveat")} for r in per_level])


def build_physics(cfg: Config, cache: FileCache, found: dict, dec) -> dict:
    trackers, gates, dataset_gates, docs = [], [], [], []
    for f in found["json"]:
        prod = content(cfg, cache, f)
        obj = prod.get("obj")
        if not isinstance(obj, dict):
            continue
        prov = provenance(f, dec)
        if "tracker_version" in obj and isinstance(obj.get("gate"), dict) and isinstance(obj.get("summary"), dict):
            g = obj["gate"]
            trackers.append({"body": obj.get("body"), "tracker_version": obj.get("tracker_version"),
                             "tracker_sha": obj.get("tracker_sha"), "tracker_source": obj.get("tracker_source"),
                             "synthetic": obj.get("synthetic"), "passed": g.get("passed"),
                             "gate": {k: rnd(v) for k, v in g.items() if not isinstance(v, dict)},
                             "contact_gate": {k: rnd(v) for k, v in (g.get("contact_gate") or {}).items()},
                             "modes": {m: {k: rnd(v) for k, v in s.items() if not isinstance(v, (list, dict))}
                                       for m, s in obj["summary"].items() if isinstance(s, dict)}, **prov})
        elif "verdict" in obj and isinstance(obj.get("criteria"), list):
            gates.append({"gate": obj.get("gate"), "version": obj.get("version"), "subject": obj.get("subject"),
                          "verdict": obj.get("verdict"),
                          "criteria": [{k: c.get(k) for k in ("name", "status", "value", "threshold", "note")}
                                       for c in obj["criteria"] if isinstance(c, dict)], **prov})
        elif isinstance(obj.get("gate"), dict) and "passed" in obj["gate"]:
            dataset_gates.append({"body": obj.get("body"), "gate": obj["gate"], "by_sigma": obj.get("by_sigma"),
                                  "tracker_sha256": obj.get("tracker_sha256"),
                                  "actuator_limits": obj.get("actuator_limits"), **prov})
        elif not f.rel.endswith(".summary.json"):
            docs.append({"title": f.rel.rsplit("/", 1)[-1], "doc": obj, **prov})
    gate_tables = []
    for f in found["md"]:
        if "gates" in f.rel or "contact" in f.rel:
            p = cache.product("md", f.sha) or []
            gate_tables += [{**t, **provenance(f, dec)} for t in p if t["rows"]]
    return envelope("physics", cfg, sorted({x["source_file"] for x in trackers + gates + dataset_gates + docs}),
                    notes=["Tracker validation = learned tracker vs its gate; gates = D-112/D-114 gate reports "
                           "(report-only where the report says so); documents = other small physics JSON files verbatim."],
                    trackers=trackers, gates=gates, dataset_gates=dataset_gates, gate_tables=gate_tables,
                    documents=docs[:400], n_documents=len(docs))


_KIND_RULES = [("trackers/", "tracker"), ("/contact_", "tracker"), ("psi1z", "psi1z"), ("grpo", "grpo"), ("adapt_expo", "grpo"),
               ("adapt_grpo", "grpo"), ("stagea", "stageA"), ("train_rep", "stageA"), ("_rep_", "stageA"),
               ("latent_sem", "stageA"), ("latent_nosem", "stageA"), ("rep_", "stageA"), ("codec", "bc"),
               ("flow", "flow"), ("_rz_", "refit"), ("rz_", "refit"), ("refit", "refit"), ("system0", "refit"),
               ("bc", "bc"), ("baseline", "bc"), ("dagger", "bc")]


def train_kind(rel: str) -> str | None:
    s = rel.lower()
    for tok, k in _KIND_RULES:
        if tok in s:
            return k
    return None


def build_training(cfg: Config, cache: FileCache, found: dict, dec, clipscale: list[dict]) -> tuple[dict, dict]:
    index, series = [], {}
    for f in found["trainlog"]:
        p = cache.product("tl", f.sha)
        if p is None:
            try:
                p = parse_trainlog(f.abs.read_text(errors="replace"))
            except OSError:
                p = {"n": 0}
            cache.put("tl", f.sha, p)
        if not p.get("n"):
            continue
        run = f.rel.rsplit("/", 1)[0]
        rid = re.sub(r"[^A-Za-z0-9_.\-]+", "__", run)[-150:] + "__" + f.sha[:8]
        s = p["series"]
        gn = "grad_norm" if "grad_norm" in s else "gn" if "gn" in s else None
        cs = next((k for k in ("clip_scale", "update_scale", "clip_coef") if k in s), None)
        kind = train_kind(f.rel) or ("tracker" if "reward_per_step" in s else None)
        index.append({"id": rid, "run": run, "file": f.rel.rsplit("/", 1)[-1], "kind": kind,
                      "n_records": p["n"], "n_points": len(p["step"]), "stride": p["stride"], "step_key": p["step_key"],
                      "last_step": p["step"][-1] if p["step"] else None, "series_keys": sorted(s),
                      "grad_norm_key": gn, "clip_scale_key": cs, "has_alpha": "alpha" in s,
                      "has_gate_state": p.get("gate_state") is not None, "last": p.get("last"),
                      "body": (bodies_in(f.rel) or [None])[0], "series_file": f"training/{rid}.json",
                      **provenance(f, dec)})
        series[rid] = {"id": rid, "run": run, "step": p["step"], "losses": {k: v for k, v in s.items()
                                                                            if k not in ("lr", "alpha", gn, cs)},
                       "grad_norm": s.get(gn) if gn else None, "clip_scale": s.get(cs) if cs else None,
                       "lr": s.get("lr"), "alpha": s.get("alpha"), "gate_state": p.get("gate_state"),
                       "source_file": f.rel, "location": f.loc}
    doc = envelope("training", cfg, sorted({i["source_file"] for i in index}), n_runs=len(index),
                   notes=["Series are downsampled by a fixed stride to <= 2000 points (last point kept); the full "
                          "series for one run is at /api/training/<id> (static: training/<id>.json).",
                          "kind is inferred from the run path (stageA / flow / refit / bc / tracker / grpo / psi1z); "
                          "null when the path does not say.",
                          "clip_scale is the logged update scale where the trainer logs it (D-085); alpha is the "
                          "tracker reward-schedule α (train_log_every10.jsonl)."],
                   runs=index, grad_health=clipscale)
    return doc, series
