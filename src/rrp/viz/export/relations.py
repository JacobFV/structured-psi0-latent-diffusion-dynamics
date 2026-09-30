"""Room documents for the one-repo harness and the relation-factor system (D-140, D-144).

- `matrix` (rrp-viz/matrix/v1): recorded `rrp matrix` outputs (harness.eval.evaluate.matrix, JSONL written with --out),
  newest row per policy × env × body × task, with the env and policy registries read from source (no imports of the
  heavy modules). Nothing is negotiated here: the room shows what a matrix run recorded.
- `factors` (rrp-viz/factors/v1): the relation-factor registry (rrp.policies.relations: stdlib + numpy), presets, field
  provenance, the candidate catalog (research/relations_catalog.md status W1/W2/P/X/M), runs whose recorded versions name a
  factor set, and curriculum schedules (`<run>/schedule.jsonl`, `<run>/steer.jsonl`) when any run has written them.
Host-light: file reads only (bounded walks), no simulation, no torch.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict
from pathlib import Path

from .common import Config, envelope

MAX_BYTES = 4_000_000


def _registry(path: Path, name: str) -> dict:
    """A module-level `NAME: dict[...] = {...}` literal, read with ast (the module is not imported)."""
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return {}
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else (node.targets[0] if isinstance(node, ast.Assign) else None)
        if isinstance(target, ast.Name) and target.id == name and node.value is not None:
            try:
                return ast.literal_eval(node.value)
            except ValueError:
                return {}
    return {}


def _roots(cfg: Config) -> list[tuple[str, Path]]:
    out = [("repo", Path(cfg.repo))]
    main = getattr(cfg, "main_checkout", None)
    if main and Path(main).resolve() != Path(cfg.repo).resolve():
        out.append(("main", Path(main)))
    return out


def _walk(base: Path, pattern: re.Pattern, depth: int = 5):
    if not base.is_dir():
        return
    stack = [(base, 0)]
    while stack:
        d, k = stack.pop()
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for e in entries:
            if e.is_dir() and k < depth and not e.name.startswith("."):
                stack.append((e, k + 1))
            elif e.is_file() and pattern.search(e.name):
                yield e


def build_matrix(cfg: Config) -> dict:
    repo = Path(cfg.repo)
    envs = _registry(repo / "src/rrp/envs/base.py", "ENVS")
    policies = _registry(repo / "src/rrp/policies/base.py", "POLICIES")
    rows: dict[tuple, dict] = {}
    files = []
    for loc, root in _roots(cfg):
        for f in _walk(root / "artifacts/runs", re.compile(r"^matrix.*\.jsonl$"), depth=4):
            if f.stat().st_size > MAX_BYTES:
                continue
            rel = str(f.relative_to(root))
            mtime = f.stat().st_mtime
            files.append({"file": rel, "location": loc, "mtime": mtime})
            for line in f.read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (r.get("policy"), r.get("env_id"), r.get("body"), r.get("task"))
                if key in rows and rows[key]["_mtime"] >= mtime:
                    continue
                rows[key] = dict(r, source_file=rel, location=loc, _mtime=mtime)
    out = []
    for r in rows.values():
        r.pop("_mtime", None)
        out.append(r)
    out.sort(key=lambda r: (str(r.get("env_id")), str(r.get("task")), str(r.get("policy"))))
    return envelope("matrix", cfg, [f["file"] for f in files], envs=envs, policies=policies, rows=out, files=files,
                    notes=["rows are recorded `rrp matrix --out` outputs (newest per policy × env × body × task); "
                           "an env or policy with no recorded row was not evaluated, not declined"])


def _catalog(path: Path) -> list[dict]:
    """research/relations_catalog.md tables: section, family, candidates, decomposition, status (W1/W2/P/X/M), label, envs."""
    rows, section = [], ""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return rows
    for ln in lines:
        if ln.startswith("## "):
            section = ln[3:].strip()
        if not ln.startswith("|") or set(ln.replace("|", "").strip()) <= set("-: "):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if cells and cells[0] in ("family", "candidate", "relation"):
            continue
        if len(cells) >= 5:
            status = cells[3]
            tok = re.match(r"\s*(W1|W2|P|X|M)\b", status)
            rows.append({"section": section, "family": cells[0], "candidates": cells[1], "decomposition": cells[2],
                         "status": tok.group(1) if tok else status[:12], "status_text": status,
                         "label": cells[4] if len(cells) > 4 else "", "envs": cells[5] if len(cells) > 5 else ""})
    return rows


def _factor_runs(cfg: Config) -> list[dict]:
    """Runs whose recorded versions / provenance name a factor set (versions["factors"]): the top-level `versions` of a
    run / summary file, or `provenance.versions` of a `pipeline_manifest.json` (unit F3; its `factors` list is the
    factor provenance, `pins` the versions a node is adopted under)."""
    out = []
    pat = re.compile(r"^((run|provenance|config|meta|summary)[\w.-]*|pipeline_manifest)\.json$")
    for loc, root in _roots(cfg):
        for f in _walk(root / "artifacts/runs", pat, depth=3):
            if f.stat().st_size > 512_000:
                continue
            try:
                d = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(d, dict):
                continue
            v = d.get("versions") or (d.get("provenance") or {}).get("versions")
            fac = v.get("factors") if isinstance(v, dict) else None
            if fac:
                row = {"run": str(f.parent.relative_to(root)), "file": str(f.relative_to(root)), "location": loc,
                       "factors": fac, "factor_set": d.get("factors") or d.get("factor_specs")}
                if f.name == "pipeline_manifest.json":
                    row.update(stage=(d.get("runconfig") or {}).get("stage"), pins=d.get("pins"))
                out.append(row)
    return out


def _competence_by_depth(records: list[dict]) -> tuple[list[dict], bool]:
    """R21: the latest `ScheduleState` (`relgen/curriculum.py`) reduced to one row per factor — its current
    composition depth (`level`), share and, once R11 fills in the scheduler's real signals (today's foundation
    placeholder always leaves `signals` empty), competence / plateau / interference. Never fabricated when absent:
    a factor with no signals yet gets `competence: None`, not a guessed number. Returns (rows, any_competence)."""
    if not records:
        return [], False
    last = records[-1] if isinstance(records[-1], dict) else {}
    level, share, signals = last.get("level") or {}, last.get("share") or {}, last.get("signals") or {}
    rows = []
    for f in sorted(level):
        sig = signals.get(f) or {}
        rows.append({"factor": f, "depth": level.get(f), "share": share.get(f), "competence": sig.get("competence"),
                     "plateau": sig.get("plateau"), "interference": sig.get("interference"),
                     "attributed_failures": sig.get("attributed_failures")})
    return rows, any(bool(signals.get(f)) for f in level)


def _schedules(cfg: Config) -> list[dict]:
    out = []
    roots = [(loc, root / "artifacts/runs") for loc, root in _roots(cfg)]
    if getattr(cfg, "rrp_data", None):
        roots.append(("rrp-data", Path(cfg.rrp_data)))
    for loc, base in roots:
        for f in _walk(base, re.compile(r"^schedule\.jsonl$"), depth=5):
            if f.stat().st_size > MAX_BYTES:
                continue
            recs = []
            for line in f.read_text().splitlines()[-3000:]:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            steer = []
            sf = f.parent / "steer.jsonl"
            if sf.is_file() and sf.stat().st_size < MAX_BYTES:
                for line in sf.read_text().splitlines():
                    try:
                        steer.append(json.loads(line))
                    except json.JSONDecodeError:
                        steer.append({"raw": line})
            cbd, has_comp = _competence_by_depth(recs)
            last_step = recs[-1].get("step") if recs and isinstance(recs[-1], dict) else None
            out.append({"run": str(f.parent), "location": loc, "file": str(f), "records": recs, "steer": steer,
                        "step": last_step, "competence_by_depth": cbd, "has_competence": has_comp})
    return out


def build_factors(cfg: Config) -> dict:
    from rrp.policies.relations import base as rel
    rel._ensure_catalog()
    fields = {n: asdict(f) for n, f in sorted(rel.FIELDS.items())}
    factors = []
    for n, d in sorted(rel.FACTORS.items()):
        fdef = rel.FIELDS.get(d.field)
        factors.append({"name": n, "version": d.version, "field": d.field, "op": d.op, "form": d.form,
                        "sources": list(d.sources), "status": d.status, "gates": list(d.gates), "label": d.label,
                        "family": n.split(".")[0], "field_prov": fdef.prov if fdef else ("edges" if d.field.startswith("edges") else None),
                        "readout": bool(d.readout), "doc": d.doc,
                        "algebra": asdict(d.algebra) if hasattr(d.algebra, "__dataclass_fields__") else str(d.algebra)})
    presets = {n: [i if isinstance(i, str) else rel.spec(i).to_dict() for i in items] for n, items in sorted(rel.PRESETS.items())}
    catalog = _catalog(Path(cfg.repo) / "research/relations_catalog.md")
    runs = _factor_runs(cfg)
    schedules = _schedules(cfg)
    return envelope("factors", cfg, ["src/rrp/policies/relations/catalog.py", "research/relations_catalog.md"],
                    factors=factors, fields=fields, presets=presets, catalog=catalog, runs=runs, schedules=schedules,
                    n_factors=len(factors), n_catalog=len(catalog),
                    notes=["registry = factors declared in code (status implemented / planned); catalog = candidate relations "
                           "from research/relations_catalog.md (W1 first wave, W2 next, P planned, X out of scope, M meta)",
                           "schedules appear once a run writes <run>/schedule.jsonl (docs/relations.md 5.5)"])
