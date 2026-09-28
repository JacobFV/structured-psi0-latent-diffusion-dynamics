"""ARM V6 TARGET BODIES: tabulate the sealed-protocol results (configs/eval/latent_slice1.json; 100 scenes from
2,000,000 per cell, infeasible excluded and counted) of the latent route (v6 semfix / nosem x seeds 1, 2: zero-shot,
flow SFT and system-0 refit at budgets 5 / 20 / 100) and the direct-action BC baseline (v6 BC experts 1701 / 1702:
zero-shot and BC SFT at the same budgets), plus source competence on the held-out source bodies. JSON only.
Usage: armtgt_compare.py --root <pulled artifacts/runs/armtgt> --out OUT.json"""
import argparse
import json
from pathlib import Path

TARGETS = ("xarm7_pg2", "xarm7_tf3", "panda_tf3")
SOURCES = ("parm5s_tf3", "parm5l_pg2")
BUDGETS = (5, 20, 100)


def cell(p: Path):
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return dict(k=d["success"], n=d["n"], rate=d.get("rate"), wilson95=d.get("wilson95"), failed_stage=d.get("failed_stage"),
                seeds=d.get("seeds"), sealed_run=d.get("sealed_run"), kind=d.get("kind"), path=str(p))


def latent(root: Path, variant: str, seed: int) -> dict:
    out = {}
    for t in TARGETS:
        d = root / f"armtgt6-{variant}-{t}"
        row = {"zero_shot": cell(d / f"target_eval-zs_s{seed}" / "generated_b0.summary.json")}
        for b in BUDGETS:
            row[f"flow_sft_b{b}"] = cell(d / f"target_eval-flow_sft_b{b}_s{seed}" / f"generated_flow_sft_b{b}.summary.json")
            row[f"system0_refit_b{b}"] = cell(d / f"target_eval-system0_refit_b{b}_s{seed}" / f"generated_system0_refit_b{b}.summary.json")
        out[t] = row
    d = root / f"armtgt6-{variant}-xarm7_pg2"
    out["source"] = {s: cell(d / f"target_eval-src_{s}_s{seed}" / "generated_source.summary.json") for s in SOURCES}
    return out


def bc(root: Path, seed: int) -> dict:
    out = {}
    for t in TARGETS:
        d = root / f"armtgt6-bc{seed}-{t}"
        row = {"zero_shot": cell(d / f"target_eval-zs_s{seed}" / "learned_b0.summary.json")}
        for b in BUDGETS:
            row[f"bc_sft_b{b}"] = cell(d / f"target_eval-bc_sft_b{b}_s{seed}" / f"learned_bc_sft_b{b}.summary.json")
        out[t] = row
    d = root / f"armtgt6-bc{seed}-xarm7_pg2"
    out["source"] = {s: cell(d / f"target_eval-src_{s}_s{seed}" / "learned_source.summary.json") for s in SOURCES}
    return out


def fmt(c):
    return "-" if not c else f"{c['k']}/{c['n']}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    res = dict(latent={f"{v}_s{s}": latent(root, v, s) for v in ("semfix", "nosem") for s in (1, 2)},
               bc={f"bc{s}": bc(root, s) for s in (1701, 1702)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for grp in ("latent", "bc"):
        for lab, x in res[grp].items():
            for t in TARGETS:
                print(f"{grp:6s} {lab:10s} {t:10s}", "  ".join(f"{k}={fmt(v)}" for k, v in x[t].items()))
            print(f"{grp:6s} {lab:10s} source    ", "  ".join(f"{k}={fmt(v)}" for k, v in x["source"].items()))
