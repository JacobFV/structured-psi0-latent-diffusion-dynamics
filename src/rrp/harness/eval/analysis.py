"""Deterministic analysis from raw per-episode rows (no hand-entered numbers).

Reads artifacts/runs/primary/<method>/seed<k>/eval/*.jsonl and sft/*/result.json, emits:
research/reports/primary_tables.json, figures under artifacts/figures/, and a markdown table.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .statistics import wilson
from .adaptation import first_sustained_crossing


def load_rows(root: Path) -> list[dict]:
    rows = []
    for f in root.glob("*/seed*/eval/*.jsonl"):
        method, seed = f.parts[-4], int(f.parts[-3][4:])
        tag = f.stem
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r.update(_method=method, _seed=seed, _tag=tag)
                rows.append(r)
    return rows


def transitions_for(root: Path, method: str, seed: int, target: str, budget: int) -> int | None:
    p = root / method / f"seed{seed}" / "sft" / f"{target}_b{budget}" / "result.json"
    return json.loads(p.read_text())["demo_control_transitions"] if p.exists() else (0 if budget == 0 else None)


def analyze(root: Path, out_dir: Path, fig_dir: Path, targets: list[str], budgets: list[int],
            threshold: float = 0.8) -> dict:
    rows = load_rows(root)
    cells = defaultdict(list)
    for r in rows:
        cells[(r["_method"], r["_seed"], r["_tag"], r["robot"])].append(r)
    table = []
    for (m, sd, tag, robot), rs in sorted(cells.items()):
        att = [r for r in rs if r["outcome"] != "infeasible"]
        k = sum(r["privileged_success"] for r in att)
        lo, hi = wilson(k, len(att))
        table.append(dict(method=m, seed=sd, tag=tag, robot=robot, attempted=len(att), successes=k,
                          rate=k / len(att) if att else None, wilson95=[lo, hi],
                          infeasible=len(rs) - len(att),
                          outcomes={o: sum(r["outcome"] == o for r in rs) for o in sorted({r["outcome"] for r in rs})},
                          public_private_disagreements=sum(r["privileged_success"] != r["public_success"] for r in att)))
    curves = {}
    methods = sorted({t["method"] for t in table})
    for m in methods:
        for tgt in targets:
            per_seed = {}
            for sd in sorted({t["seed"] for t in table if t["method"] == m}):
                xs, ys = [], []
                for b in [0] + budgets:
                    c = [t for t in table if t["method"] == m and t["seed"] == sd and t["tag"] == f"{tgt}_b{b}"]
                    if c and c[0]["rate"] is not None:
                        xs.append(transitions_for(root, m, sd, tgt, b))
                        ys.append(c[0]["rate"])
                if ys:
                    cr = first_sustained_crossing(xs, ys, threshold)
                    auc = None
                    if len(xs) > 1 and None not in xs:
                        lx = np.log1p(np.array(xs, float))
                        auc = float(np.trapezoid(ys, lx) / (lx[-1] - lx[0])) if lx[-1] > lx[0] else None
                    per_seed[sd] = dict(transitions=xs, success=ys, crossing=cr.__dict__, auc_log1p=auc)
            curves[f"{m}/{tgt}"] = per_seed
    out = dict(threshold=threshold, cells=table, curves=curves, n_rows=len(rows))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "primary_tables.json").write_text(json.dumps(out, indent=1, default=str))
    _plots(curves, fig_dir, targets, methods)
    _markdown(table, curves, out_dir / "primary_tables.md")
    return out


def _plots(curves, fig_dir: Path, targets, methods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {"relational_structured": "#2563eb", "matched_shared_unstructured": "#9ca3af"}
    for tgt in targets:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for m in methods:
            for sd, c in curves.get(f"{m}/{tgt}", {}).items():
                xs = [x if x is not None else np.nan for x in c["transitions"]]
                ax.plot(np.log1p(xs), c["success"], "-o", color=colors.get(m, None), alpha=0.8, ms=3,
                        label=f"{m} s{sd}")
        ax.axhline(0.8, ls="--", lw=0.8, color="k")
        ax.set_xlabel("log(1 + new-body demonstration control transitions)")
        ax.set_ylabel("task success (privileged evaluator)")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"adaptation: {tgt}")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(fig_dir / f"adaptation_{tgt}.png", dpi=150)
        plt.close(fig)


def _markdown(table, curves, path: Path):
    lines = ["| method | seed | condition | robot | successes/attempted | rate | Wilson 95% | outcomes |", "|---|---|---|---|---|---|---|---|"]
    for t in table:
        w = t["wilson95"]
        lines.append(f"| {t['method']} | {t['seed']} | {t['tag']} | {t['robot']} | {t['successes']}/{t['attempted']} | "
                     f"{(t['rate'] if t['rate'] is not None else float('nan')):.2f} | "
                     f"[{(w[0] or 0):.2f}, {(w[1] or 0):.2f}] | {t['outcomes']} |")
    lines.append("")
    lines.append("## sustained 80% crossing (control transitions; censored if unreached)")
    for k, per in curves.items():
        for sd, c in per.items():
            lines.append(f"- {k} seed {sd}: {c['crossing']}  AUC(log1p)={c['auc_log1p']}")
    path.write_text("\n".join(lines))
