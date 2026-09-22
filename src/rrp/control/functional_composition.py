"""Runtime-level functional-composition demonstration for support_insert (scripted_teacher).

Conditions per seed (same scene, same seed):
  receipt      : teacher places the peg with the locate#k.hole_frame RECEIPT (normal run)
  shifted      : the locate estimator output is offset by DELTA before it becomes a receipt
                 (an intervention on the upstream event's OUTPUT, nothing else changes)
  fixed_order  : same supplied stage order and runtime gating, but the hole frame is a fixed
                 nominal constant (sequencing without data flow from locate)
  stale_receipt: the locate receipt is invalidated right after it is produced -> align must be
                 rejected with a provenance reason instead of using "the latest frame"
Measured: commanded align/insert TCP goals, receipt provenance used by the teacher, runtime
statuses/rejection reasons, public vs privileged success.
Usage: python -m rrp.control.functional_composition --pair parm5_pg2__parm5_pg2 --seeds 0:10 --out ...
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

DELTA = np.array([0.02, -0.015, 0.0])
NOMINAL_HOLE = np.array([0.43, -0.045, 0.045])   # mean of the randomized layout (fixture + hole offset)


def run(pair: str, seed: int, condition: str, max_steps: int = 900) -> dict:
    from rrp.control.dual_validate import make_session
    from rrp.control.dual_teachers import SupportInsertTeacher, run_dual_teacher_episode
    s = make_session("support_insert", pair, seed)
    base_provider = s._output_provider
    if condition == "shifted":
        def provider(event_id, output_name, output_type, obs):
            v = base_provider(event_id, output_name, output_type, obs)
            if v is not None and output_type == "frame_estimate":
                v = dict(v, pos=(np.array(v["pos"]) + DELTA).tolist(), intervention="locate_output_offset")
            return v
        s._output_provider = provider
        s.runtime.output_provider = provider
    teacher = SupportInsertTeacher(s)
    if condition == "fixed_order":
        teacher.frame_override = (NOMINAL_HOLE.copy(), np.array([0.0, 0.0, 1.0]), dict(event="fixed_nominal"))
    goals = {"r_align": [], "r_insert": []}
    stale_done = [False]

    def on_step(k, cmd, out):
        ph = teacher.phase["right"]
        if ph in goals and teacher.arms["right"].goal is not None:
            goals[ph].append(teacher.arms["right"].goal.tolist())
        if condition == "stale_receipt" and not stale_done[0]:
            r = s.runtime.receipts.latest("locate", 0, "hole_frame")
            if r is not None and r.valid:
                s.runtime.receipts.invalidate("locate", 0, "hole_frame", "estimate_stale_intervention")
                s.runtime.runtime_version += 1
                stale_done[0] = True

    res = run_dual_teacher_episode(s, teacher, max_control_steps=max_steps, on_step=on_step)
    rec = s.runtime.receipts.latest("locate", 0, "hole_frame")
    return dict(pair=pair, seed=seed, condition=condition, source="scripted_teacher",
                public_success=res.success, privileged_success=res.privileged_evaluator_success,
                failure_reason=res.failure_reason, statuses=res.statuses,
                locate_receipt=None if rec is None else dict(pos=rec.value["pos"], version=rec.version,
                                                             valid=rec.valid, invalid_reason=rec.invalid_reason),
                hole_true=s.scenario.meta["privileged_layout"]["hole_world"],
                hole_frame_used=teacher.hole_frame_used,
                align_goal_last=goals["r_align"][-1] if goals["r_align"] else None,
                insert_goal_last=goals["r_insert"][-1] if goals["r_insert"] else None,
                align_rejections=[r["reason"] for r in s.runtime.rejections.get("align", [])][-5:],
                n_align_rejections=len(s.runtime.rejections.get("align", [])),
                truth=res.truth)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="parm5_pg2__parm5_pg2")
    ap.add_argument("--seeds", default="0:10")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    lo, hi = map(int, a.seeds.split(":"))
    rows = []
    t0 = time.time()
    part = Path(a.out + ".rows.jsonl")      # incremental rows: an interrupted run resumes
    part.parent.mkdir(parents=True, exist_ok=True)
    if part.exists():
        rows = [json.loads(l) for l in part.read_text().splitlines() if l.strip()]
    done = {(r["seed"], r["condition"]) for r in rows}
    for seed in range(lo, hi):
        for cond in ("receipt", "shifted", "fixed_order", "stale_receipt"):
            if (seed, cond) in done:
                continue
            rows.append(run(a.pair, seed, cond))
            with part.open("a") as fh:
                fh.write(json.dumps(rows[-1], default=str) + "\n")
            print(json.dumps({k: rows[-1][k] for k in ("seed", "condition", "public_success", "privileged_success",
                                                       "failure_reason")}), flush=True)
    # paired analysis: command shift vs receipt shift
    by = {(r["seed"], r["condition"]): r for r in rows}
    pairs = []
    for seed in range(lo, hi):
        a0, a1 = by.get((seed, "receipt")), by.get((seed, "shifted"))
        if a0 and a1 and a0["align_goal_last"] and a1["align_goal_last"]:
            dg = np.array(a1["align_goal_last"]) - np.array(a0["align_goal_last"])
            dr = np.array(a1["locate_receipt"]["pos"]) - np.array(a0["locate_receipt"]["pos"])
            pairs.append(dict(seed=seed, d_align_goal=dg.tolist(), d_receipt=dr.tolist(),
                              residual=float(np.linalg.norm(dg - dr))))
    summary = {c: dict(n=sum(1 for r in rows if r["condition"] == c),
                       public_success=sum(r["public_success"] for r in rows if r["condition"] == c),
                       privileged_success=sum(r["privileged_success"] for r in rows if r["condition"] == c))
               for c in ("receipt", "shifted", "fixed_order", "stale_receipt")}
    out = dict(pair=a.pair, delta=DELTA.tolist(), nominal_hole=NOMINAL_HOLE.tolist(), summary=summary,
               paired_command_shift=pairs, rows=rows, wall_s=time.time() - t0, source="scripted_teacher")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
