"""armdiv (D-137) admission screen: scripted teacher v2 ONLY (privileged demonstrator; no policy is involved).

  PY scripts/armdiv_screen.py --set train|targets --out-dir artifacts/runs/armdiv/screen/<set> [--workers 6]
      [--limit-keys N] [--seeds 0-19]
1. build every candidate key once (generation/attachment failures are recorded as `build_failed`, not screened);
2. run rrp.evaluation.teacher_quality (teacher v2, grasp contact from $RRP_GRASP_CONTACT, which must be v2.1) on the
   buildable keys;
3. write admission.json: per key feasible/success counts and the pre-declared rule
   admitted = feasible_rate >= 0.75 and success / feasible >= 0.9.
Seed sets: train candidates use source-train seeds 0-19; target candidates use target-demo seeds 1000000-1000019
(research/splits/primary_v1.json seed ranges; no sealed evaluation scene is touched).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

FEAS_MIN, SUCC_MIN = 0.75, 0.9


def candidate_keys(which: str) -> list[str]:
    from rrp.bodies.armdiv import candidates
    c = candidates()
    gs = c["grippers"]
    if which == "train":
        men = [k for k in c["menagerie_v2"] if k not in ("gen3", "iiwa14")]      # rizon4 is also a target fallback
        return [f"pa2s{s}_{g}" for s in c["procedural_train_seeds"] for g in gs] + [f"{k}_{g}" for k in men for g in gs]
    if which == "targets":
        return ([f"{k}_{g}" for k in ("gen3", "iiwa14", "rizon4") for g in gs]
                + [f"pa2s{s}_{g}" for s in c["procedural_sealed_seeds"] for g in gs])
    raise SystemExit(f"unknown set {which}")


def admission(rows: list[dict], n_seeds: int, build_failed: dict) -> dict:
    per = {}
    for r in rows:
        e = per.setdefault(r["robot"], dict(episodes=0, feasible=0, success=0))
        e["episodes"] += 1
        e["feasible"] += int(bool(r.get("feasible")))
        e["success"] += int(bool(r.get("success")))
    for k, e in per.items():
        fr = e["feasible"] / max(e["episodes"], 1)
        sr = e["success"] / max(e["feasible"], 1)
        e.update(feasible_rate=round(fr, 3), success_rate=round(sr, 3),
                 admitted=bool(e["episodes"] == n_seeds and fr >= FEAS_MIN and sr >= SUCC_MIN))
    for k, why in build_failed.items():
        per[k] = dict(episodes=0, feasible=0, success=0, admitted=False, build_failed=why)
    return per


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["train", "targets"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit-keys", type=int, default=0)
    a = ap.parse_args(argv)
    if os.environ.get("RRP_GRASP_CONTACT") != "v2.1":
        raise SystemExit("set RRP_GRASP_CONTACT=v2.1 (the v6 recipe)")
    seeds = a.seeds or ("0-19" if a.set == "train" else "1000000-1000019")
    from rrp.bodies.catalog import workbench_robots
    from rrp.evaluation import teacher_quality as tq
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    keys = candidate_keys(a.set)
    if a.limit_keys:
        keys = keys[:a.limit_keys]
    w = workbench_robots()
    ok, failed = [], {}
    t0 = time.time()
    for k in keys:
        try:
            w[k]()
            ok.append(k)
        except Exception as e:          # generation/attachment failure = not admissible (recorded)
            failed[k] = f"{type(e).__name__}: {str(e)[:200]}"
    print(f"[armdiv_screen] {len(ok)}/{len(keys)} keys build ({time.time() - t0:.0f}s)", flush=True)
    rows_path = out / "teacher_v2.jsonl"
    if ok:
        tq.main(["--bodies", ",".join(ok), "--seeds", seeds, "--versions", "v2", "--workers", str(a.workers),
                 "--chunk", "20", "--out", str(rows_path)])
    rows = [json.loads(l) for l in rows_path.read_text().splitlines()] if rows_path.exists() else []
    n_seeds = len(tq._parse_seeds(seeds))
    adm = admission(rows, n_seeds, failed)
    summary = dict(set=a.set, seeds=seeds, rule=dict(feasible_rate_min=FEAS_MIN, success_given_feasible_min=SUCC_MIN),
                   teacher="scripted_teacher v2 (privileged)", grasp_contact=os.environ["RRP_GRASP_CONTACT"],
                   n_keys=len(keys), n_built=len(ok), n_admitted=sum(e["admitted"] for e in adm.values()), keys=adm)
    (out / "admission.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"[armdiv_screen] admitted {summary['n_admitted']}/{len(keys)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
