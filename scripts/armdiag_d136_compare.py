"""D-136 table: pre-registered joint_adapt (split, gen_frac 0.5; BC-matched updates) and the secondary paired
(D-135 flow SFT b + refit b, 2x updates) sealed cells vs D-135 (BC SFT, system-0 refit, flow SFT; targets_v6.json).
Usage: armdiag_d136_compare.py --root <pulled artifacts/runs/armdiag> --d135 research/tracks/ladder/armv6/targets_v6.json --out OUT.json"""
import argparse
import json
from pathlib import Path

from rrp.evaluation.statistics import newcombe_diff, wilson

TARGETS = ("xarm7_pg2", "xarm7_tf3")
BUDGETS = (5, 20, 100)


def cell(p: Path):
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    assert d.get("sealed_run") is True and d.get("scene_kind") == "target", p
    return dict(k=d["success"], n=d["n"], failed_stage=d.get("failed_stage"), kind=d.get("kind"), path=str(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--d135", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, old = Path(a.root), json.loads(Path(a.d135).read_text())
    cells, pooled = {}, {}
    for v in ("semfix", "nosem"):
        for t in TARGETS:
            for b in BUDGETS:
                row = {}
                for s in (1, 2):
                    d = root / f"armja136-{v}-{t}"
                    row[f"joint_adapt_s{s}"] = cell(d / f"target_eval-joint_adapt_b{b}_s{s}" / f"generated_joint_adapt_b{b}.summary.json")
                    row[f"paired_unmatched_s{s}"] = cell(d / f"target_eval-paired_unmatched_b{b}_s{s}" / f"generated_paired_unmatched_b{b}.summary.json")
                    lo = old["latent"][f"{v}_s{s}"][t]
                    row[f"refit_s{s}"] = lo[f"system0_refit_b{b}"]
                    row[f"flow_sft_s{s}"] = lo[f"flow_sft_b{b}"]
                    row[f"bc_sft_s{s}"] = old["bc"][f"bc{1700 + s}"][t][f"bc_sft_b{b}"]
                cells[f"{v}/{t}/b{b}"] = row
                P = {}
                for m in ("joint_adapt", "paired_unmatched", "refit", "flow_sft", "bc_sft"):
                    cs = [row[f"{m}_s{s}"] for s in (1, 2)]
                    if all(cs):
                        k, n = sum(c["k"] for c in cs), sum(c["n"] for c in cs)
                        P[m] = dict(k=k, n=n, wilson95=wilson(k, n))
                if "joint_adapt" in P:
                    j = P["joint_adapt"]
                    for m in ("bc_sft", "refit", "paired_unmatched"):
                        if m in P:
                            P[f"joint_minus_{m}"] = dict(diff=j["k"] / j["n"] - P[m]["k"] / P[m]["n"],
                                                         newcombe95=newcombe_diff(j["k"], j["n"], P[m]["k"], P[m]["n"]))
                    lo_, hi_ = P.get("joint_minus_bc_sft", {}).get("newcombe95", (None, None))
                    if hi_ is not None:
                        P["prereg_reading"] = ("worse than BC" if hi_ < 0 else "as well as BC (CI includes 0 or positive)")
                pooled[f"{v}/{t}/b{b}"] = P
    Path(a.out).write_text(json.dumps(dict(label="D-136 (method added AFTER D-135; same sealed scenes)", pooled=pooled,
                                           cells=cells), indent=1))
    for key, P in pooled.items():
        print(key, {m: f"{P[m]['k']}/{P[m]['n']}" for m in ("joint_adapt", "paired_unmatched", "bc_sft", "refit", "flow_sft") if m in P},
              P.get("joint_minus_bc_sft"), P.get("prereg_reading"))


if __name__ == "__main__":
    main()
