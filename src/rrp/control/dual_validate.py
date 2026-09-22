"""Teacher validation for dual-arm tasks over arm pairs (scripted_teacher, privileged).

Pair keys: "<left_robot>__<right_robot>" (two separately mounted catalogue robots) or a
dual-arm body key such as "aloha". Records every episode (success, failure, infeasible) with
its failure reason, public-vs-privileged agreement and, for support_insert, true insertion
geometry. Usage:
  python -m rrp.control.dual_validate --task support_insert --pairs parm5_pg2__parm5_pg2 \
      --seeds 0:30 --workers 4 --out artifacts/assets/dual_teacher_validation/support_insert.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_CACHE: dict = {}


def dual_bodies() -> dict:
    out = {}
    try:
        from rrp.morphology.aloha import aloha_body, ALOHA_DIR
        if ALOHA_DIR.exists():
            out["aloha"] = aloha_body
    except ImportError:
        pass
    return out


def make_pair(key: str) -> list:
    """Robots for a pair key (cached per worker; bounded to one pair)."""
    if key in _CACHE:
        return _CACHE[key]
    _CACHE.clear()
    bodies = dual_bodies()
    if key in bodies:
        robots = [bodies[key]()]
    else:
        from rrp.morphology.catalog import workbench_robots
        W = workbench_robots()
        lk, rk = key.split("__")
        robots = [W[lk](), W[rk]()]
    _CACHE[key] = robots
    return robots


def make_session(task: str, pair: str, seed: int):
    from rrp.sim.dual import DualSession
    from rrp.sim.dual_scenarios import DUAL_BUILDERS
    return DualSession(DUAL_BUILDERS[task](make_pair(pair), seed), seed=seed)


def run_one(task: str, pair: str, seed: int, max_steps: int = 1200) -> dict:
    from rrp.control.dual_teachers import TEACHERS, run_dual_teacher_episode
    t0 = time.time()
    try:
        s = make_session(task, pair, seed)
        teacher = TEACHERS[task](s)
        res = run_dual_teacher_episode(s, teacher, max_control_steps=max_steps)
    except Exception as e:  # noqa: BLE001 - errors are recorded as data
        return dict(task=task, pair=pair, seed=seed, status="error", error=repr(e)[:400], wall_s=time.time() - t0)
    status = ("infeasible" if res.failure_reason and res.failure_reason.startswith("infeasible")
              else "success" if res.privileged_evaluator_success else "failure")
    return dict(task=task, pair=pair, seed=seed, status=status, source="scripted_teacher", privileged_teacher=True,
                public_runtime_success=res.success, privileged_success=res.privileged_evaluator_success,
                agree=res.success == res.privileged_evaluator_success, failure_reason=res.failure_reason,
                steps=res.steps, sim_s=res.steps * s.dt, statuses=res.statuses, truth=res.truth,
                feasibility=res.feasibility, rejected_commands=res.rejected_commands,
                retries={e: v[1] for e, v in res.statuses.items() if v[1] > 0},
                hole_frame_used=getattr(teacher, "hole_frame_used", None), wall_s=time.time() - t0)


def _job(args):
    return run_one(*args)


def summarize(rows: list[dict]) -> dict:
    out = {}
    for r in rows:
        k = (r["task"], r["pair"])
        d = out.setdefault(k, dict(n=0, success=0, failure=0, infeasible=0, error=0, agree=0, public_success=0,
                                   retried=0, reasons={}))
        d["n"] += 1
        d[r["status"]] += 1
        d["agree"] += int(r.get("agree", False))
        d["public_success"] += int(r.get("public_runtime_success", False))
        d["retried"] += int(bool(r.get("retries")))
        if r["status"] != "success":
            reason = (r.get("failure_reason") or r.get("error") or "")[:80]
            d["reasons"][reason] = d["reasons"].get(reason, 0) + 1
    res = {}
    for (task, pair), d in out.items():
        feas = d["n"] - d["infeasible"] - d["error"]
        d["success_rate_all"] = d["success"] / max(d["n"], 1)
        d["success_rate_feasible"] = d["success"] / max(feas, 1)
        res[f"{task}|{pair}"] = d
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--pairs", required=True, help="comma-separated pair keys")
    ap.add_argument("--seeds", default="0:30")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    lo, hi = map(int, a.seeds.split(":"))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if out.exists():   # resume: keep finished rows (interrupted runs are common under the watchdog)
        for line in out.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    done = {(r["task"], r["pair"], r["seed"]) for r in rows}
    jobs = [(a.task, p, s, a.max_steps) for p in a.pairs.split(",") for s in range(lo, hi)
            if (a.task, p, s) not in done]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, max_tasks_per_child=100) as ex, out.open("a") as fh:
        futs = [ex.submit(_job, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            rows.append(r)
            fh.write(json.dumps(r, default=str) + "\n")
            fh.flush()
            if i % 20 == 0:
                print(f"[dual_validate] {i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    want = {p for p in a.pairs.split(",")}
    summ = summarize([r for r in rows if r["pair"] in want and lo <= r["seed"] < hi])
    out.with_suffix(".summary.json").write_text(json.dumps(summ, indent=1, sort_keys=True))
    for k, d in sorted(summ.items()):
        print(k, {x: d[x] for x in ("n", "success", "failure", "infeasible", "error", "agree")}, d["reasons"])


if __name__ == "__main__":
    main()
