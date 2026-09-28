"""ARM V6 LINEAGES: tabulate the v6 set (semfix / nosem x seeds 1, 2; teacher v2 + grasp_v2.1 data, bcv6 labeller,
evaluated under grasp_v2.1) against the v1 set as recorded (grasp_v1, D-091/D-095) and the v1 checkpoints re-evaluated
under grasp_v2 (D-127, artifacts/runs/armexpert_gc2eval/compare_gc2_final.json). JSON only; no simulation.
Usage: armv6_compare.py --v6 <pulled armv6 runs dir> --gc2 <compare_gc2_final.json> --out OUT.json"""
import argparse
import json
from pathlib import Path

from armnosem_compare import edit_key, newcombe, rate, rows_of, se   # scripts/ is on sys.path when run from scripts/
from rrp.evaluation.statistics import wilson

BODIES = {"panda_pg2": (3000000, 3000100, 3000200), "parm6_tf3": (3000000, 3000100, 3000200),
          "parm5s_tf3": (3000000,), "parm5l_pg2": (3000000,)}
GC2_KEYS = {("semfix", 1): "semfix, seed 1", ("semfix", 2): "semfix, seed 2",
            ("nosem", 1): "nosem, seed 1", ("nosem", 2): "nosem, seed 2"}


def summ(p: Path):
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return dict(k=d["success"], n=d["n"], failed_stage=d["failed_stage"], seeds=d["seeds"], path=str(p))


def v6_lineage(root: Path, variant: str, seed: int) -> dict:
    lin = root / f"arm6-{variant}"
    out = {}
    for body, sets in BODIES.items():
        ev = "heldout-final" if body.startswith("parm5") else "eval_r2-final"
        per = {s: summ(lin / f"{ev}_s{seed}" / body / f"generated_zero_final_s{s}.summary.json") for s in sets}
        k = sum(x["k"] for x in per.values() if x); n = sum(x["n"] for x in per.values() if x)
        out[body] = dict(per_set=per, k=k, n=n, approach=sum(x["failed_stage"].get("approach", 0) for x in per.values() if x))
    out["all"] = dict(k=sum(out[b]["k"] for b in BODIES), n=sum(out[b]["n"] for b in BODIES))
    out["progression"] = {b: dict(
        flow20k_gendag1=summ(lin / f"eval_r2-prog20k_s{seed}" / b / "generated_zero_prog20k_s3000000.summary.json"),
        flowgdag1_gendag3=summ(lin / f"eval_r2-proggdag1_s{seed}" / b / "generated_zero_proggdag1_s3000000.summary.json"),
        r1_stateless=summ(lin / f"eval_r1-orcbc_s{seed}" / b / "oracle_zero_gendag3noqd_orcbc.summary.json"))
        for b in ("panda_pg2", "parm6_tf3")}
    edits = {}
    for body in ("parm6", "panda"):
        paths = sorted(str(p) for p in (lin / f"edits-gen_s{seed}" / body).glob("shard*/semantic_rows_generated.jsonl"))
        rows = rows_of(paths)
        edits[body] = dict(key=edit_key(se.summarize_semantic(rows), rows), n_rows=len(rows), shards=len(paths))
    out["edits"] = edits
    return out


def gc2_row(gc2: dict, key: str) -> dict:
    d = gc2[key]
    return {b: dict(grasp_v1=d[b]["grasp_v1"], grasp_v2=d[b]["grasp_v2"]) for b in BODIES}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--v6", required=True)
    ap.add_argument("--gc2", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gc2 = json.loads(Path(a.gc2).read_text())
    res = dict(v6={}, v1=gc2 and {}, diffs={})
    for (v, s), key in GC2_KEYS.items():
        res["v6"][f"{v}_s{s}"] = v6_lineage(Path(a.v6), v, s)
        res["v1"][f"{v}_s{s}"] = gc2_row(gc2, key)
    # pooled over seeds per variant; differences v6 - v1(grasp_v2) and semfix - nosem within v6
    for v in ("semfix", "nosem"):
        k6 = sum(res["v6"][f"{v}_s{s}"]["all"]["k"] for s in (1, 2)); n6 = sum(res["v6"][f"{v}_s{s}"]["all"]["n"] for s in (1, 2))
        k1 = sum(res["v1"][f"{v}_s{s}"][b]["grasp_v2"][0] for s in (1, 2) for b in BODIES)
        n1 = sum(res["v1"][f"{v}_s{s}"][b]["grasp_v2"][1] for s in (1, 2) for b in BODIES)
        k0 = sum(res["v1"][f"{v}_s{s}"][b]["grasp_v1"][0] for s in (1, 2) for b in BODIES)
        res["diffs"][v] = dict(v6=rate(k6, n6), v1_grasp_v2=rate(k1, n1), v1_grasp_v1=rate(k0, n1),
                               v6_minus_v1_grasp_v2=dict(diff=k6 / n6 - k1 / n1, ci95=newcombe(k6, n6, k1, n1)))
    s6 = res["diffs"]["semfix"]["v6"]; n6 = res["diffs"]["nosem"]["v6"]
    res["diffs"]["v6_semfix_minus_nosem"] = dict(diff=s6["rate"] - n6["rate"], ci95=newcombe(s6["k"], s6["n"], n6["k"], n6["n"]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    for lab, d in res["v6"].items():
        print(lab, {b: f"{d[b]['k']}/{d[b]['n']}" for b in BODIES}, "all", f"{d['all']['k']}/{d['all']['n']}",
              "| approach fails", {b: d[b]["approach"] for b in BODIES})
    print(json.dumps(res["diffs"], default=str))
