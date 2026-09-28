"""File discovery over the repo, the agent worktrees and ~/work/rrp-data, with content-hash deduplication, plus the
content-level extractors (results rows, causal-edit rows) whose products are cached by sha1.

A file that exists in several places (git + N worktrees) is reported ONCE, from the most authoritative location
(repo > main checkout > worktrees > rrp-data), with the number of copies and a few alternative paths.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .common import Config, FileCache, rnd, wilson

SKIP_DIRS = {".git", "node_modules", "__pycache__", "datasets", "packed", "checkpoints", "episodes", "raw", ".cache",
             ".venv", "venv", "wandb", "frames"}
SKIP_JSON = {"config.json", "manifest.json", "meta.json", "policy.json", "representation.json", "rz_last.json",
             "policy_last.json", "pipeline_manifest.json", "probe_cfg.json", "dataset_manifest.json", "provenance.json",
             "run_config.json", "codec.json"}
SKIP_JSON_RE = re.compile(r"^(policy_b\d+|s\d+-\d+|snap_s\d+|shard\d+.*|episode_\d+|ep\d+)\.json$")
MAX_JSON = 2_000_000
TRAIN_LOGS = {"train_log.jsonl", "train_log_every10.jsonl"}
SUBDIRS = ("artifacts/runs", "artifacts/trackers", "research/tracks")
LOC_RANK = {"repo": 0, "main": 1, "wt": 2, "rrp-data": 3, "peer": 4}


@dataclass
class Found:
    kind: str          # json | trainlog | ledger | md
    abs: Path
    rel: str           # path relative to its checkout root (artifacts/runs/…)
    loc: str           # repo | main | wt:<name> | rrp-data:<dir>
    mtime: float
    size: int
    sha: str | None = None

    @property
    def rank(self):
        return (LOC_RANK[self.loc.split(":")[0]], self.loc, self.rel)


def roots(cfg: Config) -> list[tuple[str, Path]]:
    out = [("repo", cfg.repo)]
    seen = {cfg.repo.resolve()}
    if cfg.main_checkout and cfg.main_checkout.exists() and cfg.main_checkout.resolve() not in seen:
        out.append(("main", cfg.main_checkout))
        seen.add(cfg.main_checkout.resolve())
    if cfg.wt_root and cfg.wt_root.is_dir():
        for p in sorted(cfg.wt_root.iterdir()):
            if p.is_dir() and (p / "artifacts").exists() and p.resolve() not in seen:
                out.append((f"wt:{p.name}", p))
                seen.add(p.resolve())
    if cfg.rrp_data and cfg.rrp_data.is_dir():
        for sub in ("git-untracked-2026-09-26", "frozen"):
            if (cfg.rrp_data / sub).is_dir():
                out.append((f"rrp-data:{sub}", cfg.rrp_data / sub))
    return out


def _classify(name: str, dirpath: str, size: int) -> str | None:
    if name in TRAIN_LOGS:
        return "trainlog"
    if name.endswith(".md"):
        return "md"
    if name.endswith(".json"):
        if name == "ledger.json" and "/_dags/" in dirpath + "/":
            return "ledger"
        if size > MAX_JSON or name in SKIP_JSON or SKIP_JSON_RE.match(name) or name.endswith(".mem.json"):
            return None
        return "json"
    return None


def discover(cfg: Config, cache: FileCache) -> dict[str, list[Found]]:
    """All candidate files, grouped by kind, deduplicated by content (canonical copy first, `copies` attached)."""
    raw: list[Found] = []
    for loc, root in roots(cfg):
        subdirs = SUBDIRS if not loc.startswith("rrp-data") else ("artifacts/runs", "artifacts/trackers", ".")
        done = set()
        for sd in subdirs:
            base = (root / sd) if sd != "." else root
            if not base.is_dir():
                continue
            rb = base.resolve()
            if rb in done:
                continue
            done.add(rb)
            for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
                for fn in filenames:
                    p = os.path.join(dirpath, fn)
                    try:
                        st = os.stat(p)
                    except OSError:
                        continue
                    kind = _classify(fn, dirpath, st.st_size)
                    if kind is None:
                        continue
                    if kind == "md" and "/research/tracks" not in dirpath and "/artifacts/runs" not in dirpath:
                        continue
                    ap = Path(p)
                    f = Found(kind, ap, os.path.relpath(p, root), loc, st.st_mtime, st.st_size)
                    f.sha = cache.digest(ap, st)
                    if f.sha:
                        raw.append(f)
    groups: dict[tuple[str, str], list[Found]] = {}
    for f in raw:
        groups.setdefault((f.kind, f.sha), []).append(f)
    out: dict[str, list[Found]] = {"json": [], "trainlog": [], "ledger": [], "md": []}
    for (kind, _), fs in groups.items():
        fs.sort(key=lambda f: f.rank)
        head = fs[0]
        head.copies = len(fs)                       # type: ignore[attr-defined]
        head.alt = [f"{f.loc}:{f.rel}" for f in fs[1:4]]  # type: ignore[attr-defined]
        head.newest_mtime = max(f.mtime for f in fs)      # type: ignore[attr-defined]
        out[kind].append(head)
    for v in out.values():
        v.sort(key=lambda f: (f.rel, f.loc))
    return out


def provenance(f: Found, decision_index=None) -> dict:
    d = {"source_file": f.rel, "location": f.loc, "in_git": f.loc == "repo", "sha1": f.sha[:12] if f.sha else None,
         "copies": getattr(f, "copies", 1)}
    alt = getattr(f, "alt", None)
    if alt:
        d["alt_paths"] = alt
    if decision_index is not None:
        d.update(decision_index.lookup(f.rel))
    return d


# ================================================================ content extraction (cached per sha1)
CTX_KEYS = {"robot": "body", "body": "body", "route": "route", "source": "source", "sources": "source",
            "source_label": "source", "edit": "edit", "task": "task", "variant": "variant", "policy_label": "policy_label",
            "grasp_contact_version": "grasp", "grasp_contact_versions": "grasp", "contact_version": "contact",
            "contact_versions": "contact", "actuator_limits": "actuator", "teacher_version": "teacher_version",
            "tracker_version": "tracker_version", "label": "label", "condition": "condition", "seeds": "seeds"}
MAX_ROWS = 600
_KN_STR = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\b(.*)$")
_FELL = re.compile(r"(\d+)\s+fell")


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _ctx_value(key, v):
    if key == "route" and isinstance(v, dict):
        return v.get("name") or v.get("kind")
    if key == "condition" and isinstance(v, dict):
        return v.get("key") or json.dumps(v, sort_keys=True)[:80]
    if key == "seeds":
        if isinstance(v, list) and v and all(_is_int(s) for s in v):
            return f"{v[0]}..{v[-1]}" if len(v) != 3 or v[2] > 3 else f"{v[0]}..{v[1]}"
        return v if isinstance(v, str) else None
    if isinstance(v, (str, int, float)) or v is None:
        return v
    if isinstance(v, list) and all(isinstance(x, (str, int, float)) for x in v):
        return v if len(v) != 1 else v[0]
    return None


def extract_results(obj) -> dict:
    rows: list[dict] = []
    truncated = [False]

    def emit(kp, ctx, k, n, metric, extra):
        if len(rows) >= MAX_ROWS:
            truncated[0] = True
            return
        r = {"key_path": [str(x) for x in kp], "metric": metric, "k": k, "n": n,
             "rate": round(k / n, 4) if n else None, "ctx": {a: b for a, b in ctx.items() if b is not None}}
        r.update(extra)
        rows.append(r)

    def walk(o, kp, ctx):
        if isinstance(o, dict):
            c = dict(ctx)
            for key, tgt in CTX_KEYS.items():
                if key in o:
                    v = _ctx_value(key, o[key])
                    if v is not None and not (tgt == "source" and "source" in c and key == "sources"):
                        c[tgt] = v
            n = o.get("n")
            if _is_int(n) and n > 0:
                for kk, metric in (("success", "success"), ("k", None), ("followed", "followed"),
                                   ("privileged_success", "privileged_success")):
                    k = o.get(kk)
                    if _is_int(k) and 0 <= k <= n:
                        ci = o.get("wilson95") or o.get(f"{kk}_wilson95") or o.get("ci")
                        ok_ci = isinstance(ci, list) and len(ci) == 2 and all(isinstance(x, (int, float)) for x in ci)
                        extra = {"ci": [rnd(ci[0]), rnd(ci[1])] if ok_ci else wilson(k, n),
                                 "ci_method": "recorded" if ok_ci else "wilson95_computed"}
                        for fk in ("fell", "falls"):
                            if _is_int(o.get(fk)):
                                extra["fell"] = o[fk]
                        st = o.get("failed_stage") or o.get("stages")
                        if isinstance(st, dict):
                            extra["stages"] = {a: b for a, b in st.items() if _is_int(b)}
                        if isinstance(o.get("public_success"), int):
                            extra["public_success"] = o["public_success"]
                        elif isinstance(o.get("task"), dict) and _is_int(o["task"].get("public_success")):
                            extra["public_success"] = o["task"]["public_success"]
                        mname = metric or next((str(x) for x in reversed(kp) if not str(x).isdigit()), "k")
                        emit(kp, c, k, n, mname, extra)
                        break
            for key, v in o.items():
                if key in ("seeds", "success_by_seed", "rows", "episodes", "shards", "checkpoints", "perturbation"):
                    continue
                if isinstance(v, list) and len(v) in (2, 3) and _is_int(v[0]) and _is_int(v[1]) and v[1] > 0 \
                        and 0 <= v[0] <= v[1] and (len(v) == 2 or isinstance(v[2], dict)) \
                        and key not in ("seeds", "wilson95", "ci", "steps", "range"):
                    extra = {"ci": wilson(v[0], v[1]), "ci_method": "wilson95_computed"}
                    if len(v) == 3:
                        extra["stages"] = {a: b for a, b in v[2].items() if _is_int(b)}
                    emit(kp + [key], c, v[0], v[1], str(key), extra)
                elif isinstance(v, str):
                    m = _KN_STR.match(v)
                    if m and int(m.group(2)) > 0 and int(m.group(1)) <= int(m.group(2)):
                        k_, n_ = int(m.group(1)), int(m.group(2))
                        extra = {"ci": wilson(k_, n_), "ci_method": "wilson95_computed", "text": v}
                        fm = _FELL.search(m.group(3))
                        if fm:
                            extra["fell"] = int(fm.group(1))
                        emit(kp + [key], c, k_, n_, str(key), extra)
                else:
                    walk(v, kp + [key], c)
        elif isinstance(o, list):
            for i, v in enumerate(o[:200]):
                if isinstance(v, (dict, list)):
                    walk(v, kp + [i], ctx)

    root_ctx: dict = {}
    if isinstance(obj, dict):
        for mk in ("meta", "metadata", "run"):
            m = obj.get(mk)
            if isinstance(m, dict):
                for key, tgt in CTX_KEYS.items():
                    if key in m:
                        v = _ctx_value(key, m[key])
                        if v is not None:
                            root_ctx[tgt] = v
    walk(obj, [], root_ctx)
    return {"rows": rows, "truncated": truncated[0]}


def _triple(v):
    return (isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                                         for x in v) and v[1] < v[2] and v[1] - 1e-9 <= v[0] <= v[2] + 1e-9)


_EDIT_SPLIT = re.compile(r"\s(?=Δ|toward\b|Δ)")


def edit_label(label: str) -> tuple[str, str | None]:
    """'ctx halt Δforward m' -> ('ctx_halt', 'Δforward m'); 'z turn +0.6 Δyaw' -> ('z_turn_+0.6', 'Δyaw')."""
    parts = _EDIT_SPLIT.split(label, maxsplit=1)
    ed = parts[0].strip()
    metric = parts[1].strip() if len(parts) > 1 else None
    if "," in (metric or ""):
        metric, note = metric.split(",", 1)
        ed = ed + " (" + note.strip() + ")"
    ed = ed.replace("−", "-").replace("ACTIVE-INACTIVE", "active_minus_inactive")
    return re.sub(r"\s+", "_", ed).lower(), metric


def edit_role(edit: str) -> str:
    e = edit.lower()
    if "minus" in e or "-inactive" in e or "diff" in e:
        return "difference"
    if any(t in e for t in ("inactive", "random", "rand_norm", "irrelevant", "control", "orthogonal", "noise_replay")):
        return "control"
    if e in ("none", "unedited", "baseline"):
        return "reference"
    return "edit"


def extract_edits(obj) -> dict:
    """Effect triples [mean, lo, hi] and permutation tests anywhere in the document."""
    rows: list[dict] = []

    def walk(o, kp, n_ctx):
        if len(rows) >= MAX_ROWS or not isinstance(o, dict):
            if isinstance(o, list):
                for i, v in enumerate(o[:50]):
                    walk(v, kp + [i], n_ctx)
            return
        n_here = o.get("n") if _is_int(o.get("n")) else n_ctx
        perm = o.get("permutation")
        if isinstance(perm, dict) and ("p_one_sided" in perm or "p" in perm):
            ed, met = edit_label(str(kp[-1])) if kp else ("?", None)
            rows.append({"key_path": [str(x) for x in kp], "edit": ed, "metric": (met or "") + " diff (a−b)",
                         "effect": perm.get("diff"), "ci": (o.get("pooled_diff_fixsem_minus_nosem") or [None] * 3)[1:],
                         "p_one_sided": perm.get("p_one_sided", perm.get("p")), "p_two_sided": perm.get("p_two_sided"),
                         "n_perm": perm.get("n_perm"), "direction": perm.get("direction"),
                         "every_seed_ordered": perm.get("every_seed_ordered"), "kind": "permutation",
                         "seeds_a": perm.get("seeds_fixsem"), "seeds_b": perm.get("seeds_nosem")})
        for key, v in o.items():
            if _triple(v):
                parent = str(kp[-1]) if kp else ""
                if parent in ("effects",) or " " in str(key) or "Δ" in str(key):
                    ed, metric = edit_label(str(key))
                else:
                    ed, metric = (parent or "?"), str(key)
                if parent == "conditions" or (len(kp) >= 2 and str(kp[-2]) == "conditions"):
                    ed = parent
                rows.append({"key_path": [str(x) for x in kp + [key]], "edit": ed, "metric": metric,
                             "effect": rnd(v[0]), "ci": [rnd(v[1]), rnd(v[2])], "n_pairs": n_here, "kind": "effect"})
            elif isinstance(v, (dict, list)):
                walk(v, kp + [key], n_here)

    walk(obj, [], None)
    return {"rows": rows}


_STEP_KEYS = ("step", "iter", "update", "updates", "global_step", "epoch_step")
_SKIP_SERIES = {"t", "wall_s", "rollout_s", "iter_s", "update_s", "rollout_wall_s", "update_wall_s", "samples",
                "time", "elapsed"}
MAX_POINTS = 2000


def parse_trainlog(text: str) -> dict:
    recs = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            recs.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    if not recs:
        return {"n": 0}
    step_key = next((k for k in _STEP_KEYS if any(k in r for r in recs[:50])), None)
    keys: dict[str, int] = {}
    for r in recs:
        for k, v in r.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k != step_key:
                keys[k] = keys.get(k, 0) + 1
        g = r.get("gate")
        if isinstance(g, dict):
            for k, v in g.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    keys[f"gate.{k}"] = keys.get(f"gate.{k}", 0) + 1
    n = len(recs)
    stride = max(1, -(-n // MAX_POINTS))
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    idx = idx[:MAX_POINTS] if len(idx) <= MAX_POINTS else idx[:MAX_POINTS - 1] + [n - 1]

    def get(r, k):
        if k.startswith("gate."):
            g = r.get("gate")
            return g.get(k[5:]) if isinstance(g, dict) else None
        return r.get(k)

    steps = [recs[i].get(step_key, i) if step_key else i for i in idx]
    series = {k: [rnd(get(recs[i], k), 6) for i in idx] for k in sorted(keys) if k not in _SKIP_SERIES}
    gate_state = None
    if any(isinstance(r.get("gate"), dict) and "action" in r["gate"] for r in recs):
        gate_state = [(recs[i].get("gate") or {}).get("action") if isinstance(recs[i].get("gate"), dict) else None
                      for i in idx]
    wall = next((recs[-1].get(k) for k in ("wall_s", "t") if isinstance(recs[-1].get(k), (int, float))), None)
    last = {k: rnd(v, 6) for k, v in recs[-1].items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return {"n": n, "stride": stride, "step_key": step_key, "step": steps, "series": series, "gate_state": gate_state,
            "last": last, "wall_s_last": rnd(wall, 1)}
