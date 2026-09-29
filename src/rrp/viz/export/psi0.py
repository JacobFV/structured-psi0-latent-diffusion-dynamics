"""/api/psi0: the Ψ₀ line (W10): the P-decisions (appendix P of research/decisions.md, folded from psi1z by D-140), the
D ↔ P links from their headings, the track note research/tracks/psi0.md, and the small result files copied from the
peer's ~/work/ext/runs/psi1z/cl (historical dir name) into ~/work/rrp-data/viz/psi1z (summary files only, `--sync-psi1z`)."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from .common import Config, envelope, parse_entries, parse_tables, wilson

SYNC_INCLUDE = ("episodes.jsonl", "eval_stats*", "summary.json", "result.json")
SYNC_LIMIT_BYTES = 50 * 1024 * 1024
INFRA_FAIL = re.compile(r"(cpufail|nvrtcfail|uidfail|portcollision|_fail\b)")


def local_dir(cfg: Config) -> Path | None:
    return cfg.rrp_data / "viz/psi1z" if cfg.rrp_data else None


def sync_psi1z(cfg: Config, timeout: float = 60) -> dict:
    """rsync ONLY small summary files from the peer's ~/work/ext/runs/psi1z/cl/*/ (≤ 5 MB each, ≤ 50 MB total)."""
    dst = local_dir(cfg)
    if dst is None:
        return {"ok": False, "error": "no rrp-data dir"}
    dst.mkdir(parents=True, exist_ok=True)
    args = ["rsync", "-a", "--max-size=5M", "--prune-empty-dirs", "--include=*/"]
    args += [f"--include={p}" for p in SYNC_INCLUDE] + ["--exclude=*", f"{cfg.peer}:work/ext/runs/psi1z/cl/", f"{dst}/"]
    try:
        cp = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "rsync timeout"}
    total = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
    return {"ok": cp.returncode == 0, "rc": cp.returncode, "bytes": total, "over_limit": total > SYNC_LIMIT_BYTES,
            "stderr": cp.stderr[-300:]}


def _read(p: Path) -> str | None:
    try:
        return p.read_text()
    except OSError:
        return None


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _label(src: str | None) -> str | None:
    if not src:
        return None
    if src.startswith("Ψ₀:") or src.lower().startswith("psi0:"):
        return "third_party:psi0_released:" + src.split(":", 1)[1].split("/")[-1]
    return src if ":" in src else f"recorded:{src}"


def run_rows(cfg: Config) -> list[dict]:
    base = local_dir(cfg)
    if base is None or not base.is_dir():
        return []
    rows = []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        res = _load(d / "result.json") or _load(d / "part0/result.json") or {}
        summ = _load(d / "summary.json") or _load(d / "part0/simple_eval/summary.json")
        eps = []
        for ep in sorted(d.rglob("episodes.jsonl")):
            for ln in (_read(ep) or "").splitlines():
                try:
                    eps.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
        stats = next(iter(sorted(d.rglob("eval_stats*"))), None)
        stat_eps, task_from_stats = [], None
        if stats:
            for ln in (_read(stats) or "").splitlines():
                m = re.match(r"episode_(\d+):\s*(True|False)", ln.strip())
                if m:
                    stat_eps.append(m.group(2) == "True")
                m2 = re.match(r"run:\s*simple/(\S+)", ln.strip())
                if m2:
                    task_from_stats = m2.group(1)
        if summ and isinstance(summ.get("n"), int):
            k, n, how = summ.get("successes", summ.get("success")), summ["n"], "summary.json"
        elif eps:
            k, n, how = sum(bool(e.get("success")) for e in eps), len(eps), "episodes.jsonl"
        elif stat_eps:
            k, n, how = sum(stat_eps), len(stat_eps), "eval_stats.txt"
        else:
            k = n = how = None
        files = sorted(str(p.relative_to(base)) for p in d.rglob("*") if p.is_file())
        rows.append({"run": d.name, "task": res.get("task") or task_from_stats, "level": res.get("level"),
                     "source": res.get("source"), "source_label": _label(res.get("source")), "k": k, "n": n,
                     "rate": round(k / n, 4) if n else None,
                     "ci": (summ or {}).get("wilson95") or (wilson(k, n) if n else None), "count_from": how,
                     "rtc": res.get("rtc"), "episodes": [{"episode": e.get("episode"), "success": e.get("success"),
                                                          "steps": e.get("steps"), "max_reward": e.get("max_reward")}
                                                         for e in eps][:100],
                     "interim": how not in ("summary.json", None),
                     "interim_reason": (None if how in ("summary.json", None) else
                                        f"no summary.json: k/n counted from {how} (run in progress, partial or "
                                        "split across directories)"),
                     "infra_failure_run": bool(INFRA_FAIL.search(d.name)),
                     "caveat": ("run name records an infrastructure failure" if INFRA_FAIL.search(d.name) else None),
                     "source_files": [f"~/work/rrp-data/viz/psi1z/{d.name}/{f.split('/', 1)[1] if '/' in f else f}"
                                      for f in files][:12],
                     "copied_from": f"{cfg.peer}:~/work/ext/runs/psi1z/cl/{d.name}/"})
    return rows


def build_psi0(cfg: Config, rrp_decisions: list[dict]) -> dict:
    dec = cfg.repo / "research/decisions.md"
    pdec = parse_entries(_read(dec) or "", "research/decisions.md", "P")
    crosswalk = [{"rrp": e["refs_d"], "psi1z": [e["id"]], "topic": e["title"], "source_file": e["source_file"],
                  "line": e["line"]} for e in pdec if e["refs_d"]]
    note = cfg.repo / "research/tracks/psi0.md"
    notes_md = _read(note)
    notes_tables = parse_tables(notes_md, "research/tracks/psi0.md") if notes_md else []
    srcs = [dec] + ([note] if notes_md else [])
    w10 = [{"id": d["id"], "date": d["date"], "title": d["title"], "refs_p": d["refs_p"], "source_file": d["source_file"],
            "line": d["line"]} for d in rrp_decisions
           if d["refs_p"] or "W10" in d["workstreams"] or "Ψ₀" in d["title"] or "psi" in d["title"].lower()]
    runs = run_rows(cfg)
    return envelope("psi0", cfg, [str(s) for s in srcs] +
                    ([str(local_dir(cfg))] if runs else []),
                    notes=["Released-checkpoint runs are upstream Ψ₀ (third-party), not our model (D-120).",
                           "Result files are local copies of small peer files (rsync of summary.json / result.json / "
                           "episodes.jsonl / eval_stats only); count_from says which file gave k/n."],
                    p_decisions=pdec, crosswalk=crosswalk, p_to_d_table=[], rrp_w10_decisions=w10,
                    notes_tables=notes_tables, notes_markdown=notes_md, readme_markdown=None, runs=runs,
                    n_runs=len(runs), missing=[] if runs else ["no local psi1z result copies (run --sync-psi1z)"])
