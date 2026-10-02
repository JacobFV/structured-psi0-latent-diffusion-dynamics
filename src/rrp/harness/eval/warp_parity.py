"""Warp-vs-CPU parity and throughput of the batched evaluation backend (protocol pre-registered in research/tracks/compute.md,
section "parity protocol"; tolerances are the constants below and are not tuned after measuring).

  rrp suite warp-parity --fixture F1 --device cuda:0 --out artifacts/runs/compute/warpeval/F1.json
  rrp suite warp-parity --fixture F1 --bench 8,32,64 --device cuda:0          # episodes/s, CPU vs Warp

Single-step parity: a CPU rollout (the reference trajectory) in which, at every control step, a copy of the CPU state is advanced by the Warp
hybrid with the CPU's own ctrl schedule and compared with the CPU's own result (`ShadowStepper`), so errors never accumulate.
Closed loop: the same policy and seeds on `eval_backend=cpu` and `warp`: outcome / failure_reason must be identical (HARD GATE), trajectories
are compared with the declared divergence tolerances. Sources: every row is `scripted_teacher` (teacher:<task>) on the named tracker; nothing learned.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from rrp.envs.mujoco.session import Integrate, cpu_integrate
from rrp.envs.warp import batch_sim as bs

ARM_SINGLE = dict(p99=1e-3, max=1e-2)           # rad / m per control step
LEGGED_SINGLE = dict(p99=2e-3, max=2e-2)
ARM_LOOP = dict(arm_qpos_median=0.05, arm_qpos_p95=0.15, object_median=0.02, object_p95=0.05)
LEGGED_LOOP = dict(base_xy_median=0.30, base_xy_p95=1.0, end_time_s=0.5)

FIXTURES = {
    "F1": dict(family="arm", env_id="mujoco/arm", task="pick_place", body="panda_pg2", policy="teacher:pick_place", seeds=list(range(12)),
               env_kw={}, note="arm_scene(seed): seed % 3 distractors"),
    "F2": dict(family="legged", env_id="mujoco/legged", task="waypoint_contact", body="hexapod6", policy="teacher:waypoint_contact",
               seeds=list(range(8)), env_kw=dict(tracker_kind="cpg"), note="hexapod6, cpg tracker, base_velocity commands of the scripted teacher"),
    "F3w": dict(family="legged", env_id="mujoco/legged", task="h_walk", body="g1", policy="teacher:h_walk", seeds=list(range(8)),
                env_kw=dict(tracker="g1:ub_v1"), note="peer only (untracked tracker weights)"),
    "F3t": dict(family="legged", env_id="mujoco/legged", task="h_turn", body="g1", policy="teacher:h_turn", seeds=list(range(8)),
                env_kw=dict(tracker="g1:ub_v1"), note="peer only (untracked tracker weights)"),
}


def stats(x) -> dict:
    x = np.asarray(x, float)
    if x.size == 0:
        return dict(n=0)
    return dict(n=int(x.size), mean=float(x.mean()), median=float(np.median(x)), p95=float(np.percentile(x, 95)),
                p99=float(np.percentile(x, 99)), max=float(x.max()))


class ShadowStepper(bs.BatchStepper):
    """CPU-driven stepper that, before every CPU integration, advances a COPY of the state with the Warp hybrid and records the
    max |dqpos| between the two results (the single-step parity metric). The returned trajectory is the CPU one."""
    errors: list = []

    def step(self, commands: dict) -> dict:
        out = {}
        for i, c in commands.items():
            env, g = self.envs[i], self.envs[i]._step_gen(c)
            try:
                req = next(g)
                while True:
                    self._compare(i, env, req)
                    cpu_integrate(env.model, env.data, req)
                    req = g.send(None)
            except StopIteration as e:
                out[i] = e.value
            except Exception as ex:  # noqa: BLE001
                out[i] = ex
        return out

    def _compare(self, i, env, req):
        shadow = mujoco.MjData(env.model)
        mujoco.mj_copyData(shadow, env.model, env.data)
        r2 = Integrate(req.n, np.array(req.ctrl, copy=True), req.energy_act)
        self.pool.integrate({i: r2}, {i: shadow})
        ref = mujoco.MjData(env.model)
        mujoco.mj_copyData(ref, env.model, env.data)
        cpu_integrate(env.model, ref, Integrate(req.n, np.array(req.ctrl, copy=True), req.energy_act))
        ShadowStepper.errors.append(float(np.max(np.abs(shadow.qpos - ref.qpos))))


class Trace:
    """Hook: per-episode qpos trace and the final quantities the closed-loop metrics compare."""

    def __init__(self, family):
        self.family, self.q = family, {}

    def on_step(self, i, env, a, step):
        self.q.setdefault(id(env), []).append(np.array(env.data.qpos, copy=True))

    def on_end(self, i, env, ep):
        q = np.array(self.q.pop(id(env), []))
        m = env.model
        out = dict(q=q, t_end=float(env.data.time))
        if self.family == "arm":
            names = list(env.robots[0].arm_joints)
            out["arm_idx"] = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])
            out["object"] = np.array(env.data.xpos[m.body("cube").id])
        else:
            out["base_xy"] = np.array(env.data.qpos[:2])
        return dict(_trace=out)


def _run(fx, backend, device, batch, hooks_extra=(), seeds=None):
    from rrp.harness.eval.evaluate import evaluate, task_hooks
    tr = Trace(fx["family"])
    t = time.time()
    eps = evaluate(fx["policy"], fx["env_id"], fx["task"], fx["body"], seeds if seeds is not None else fx["seeds"], batch=batch,
                   hooks=[*task_hooks(fx["task"], fx["env_id"]), tr, *hooks_extra], env_kw=fx["env_kw"] or None,
                   eval_backend=backend, device=device)
    return eps, time.time() - t


def single_step(fx, device) -> dict:
    ShadowStepper.errors = []
    orig = bs.BatchStepper
    bs.BatchStepper = lambda envs, dev=None: ShadowStepper(envs, dev)     # rollout imports BatchStepper at call time
    try:
        _run(fx, "warp", device, batch=len(fx["seeds"]))
    finally:
        bs.BatchStepper = orig
    tol = ARM_SINGLE if fx["family"] == "arm" else LEGGED_SINGLE
    st = stats(ShadowStepper.errors)
    return dict(stats=st, tolerance=tol, ok=bool(st["n"] and st["p99"] <= tol["p99"] and st["max"] <= tol["max"]))


def closed_loop(fx, device) -> dict:
    cpu, tc = _run(fx, "cpu", None, batch=len(fx["seeds"]))
    wp, tw = _run(fx, "warp", device, batch=len(fx["seeds"]))
    rows, same = [], True
    for a, b in zip(cpu, wp):
        ok = (a.outcome, a.failure_reason) == (b.outcome, b.failure_reason)
        same &= ok
        ta, tb = a.metrics["_trace"], b.metrics["_trace"]
        n = min(len(ta["q"]), len(tb["q"]))
        row = dict(seed=a.seed, cpu=[a.outcome, a.failure_reason], warp=[b.outcome, b.failure_reason], same=ok, steps=[a.steps, b.steps],
                   end_time_diff=abs(ta["t_end"] - tb["t_end"]),
                   fallback=b.provenance.get("eval_backend", {}).get("fallback"))
        if fx["family"] == "arm":
            row["arm_qpos_max"] = float(np.max(np.abs(ta["q"][:n, ta["arm_idx"]] - tb["q"][:n, ta["arm_idx"]]))) if n else None
            row["object_err"] = float(np.linalg.norm(ta["object"] - tb["object"]))
        else:
            row["base_xy_err"] = float(np.linalg.norm(ta["base_xy"] - tb["base_xy"]))
        rows.append(row)
    out = dict(outcomes_identical=same, n=len(rows), wall_s=dict(cpu=tc, warp=tw), rows=rows)
    if fx["family"] == "arm":
        out["arm_qpos_max"] = stats([r["arm_qpos_max"] for r in rows if r["arm_qpos_max"] is not None])
        out["object_err"] = stats([r["object_err"] for r in rows])
        s1, s2, tol = out["arm_qpos_max"], out["object_err"], ARM_LOOP
        out["within_tolerance"] = bool(s1["median"] <= tol["arm_qpos_median"] and s1["p95"] <= tol["arm_qpos_p95"] and
                                       s2["median"] <= tol["object_median"] and s2["p95"] <= tol["object_p95"])
    else:
        out["base_xy_err"] = stats([r["base_xy_err"] for r in rows])
        out["end_time_diff"] = stats([r["end_time_diff"] for r in rows])
        s1, tol = out["base_xy_err"], LEGGED_LOOP
        out["within_tolerance"] = bool(s1["median"] <= tol["base_xy_median"] and s1["p95"] <= tol["base_xy_p95"] and
                                       out["end_time_diff"]["max"] <= tol["end_time_s"])
    return out


def bench(fx, device, ns) -> list[dict]:
    """episodes/s of the same N episodes on cpu (in-process, one core: the rollout's own loop) and warp (physics batched)."""
    res = []
    for n in ns:
        seeds = [fx["seeds"][k % len(fx["seeds"])] + 1000 * (k // len(fx["seeds"])) for k in range(n)]
        _run(fx, "warp", device, batch=min(n, 4), seeds=seeds[:min(n, 4)])               # module load / kernel compile / graph capture
        cpu, tc = _run(fx, "cpu", None, batch=n, seeds=seeds)
        wp, tw = _run(fx, "warp", device, batch=n, seeds=seeds)
        res.append(dict(n=n, cpu_s=tc, warp_s=tw, cpu_eps_per_s=n / tc, warp_eps_per_s=n / tw, speedup=tc / tw,
                        same_outcomes=sum((a.outcome, a.failure_reason) == (b.outcome, b.failure_reason) for a, b in zip(cpu, wp)),
                        fallback=wp[0].provenance.get("eval_backend", {}).get("fallback")))
        print(json.dumps(res[-1]), flush=True)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rrp suite warp-parity")
    ap.add_argument("--fixture", required=True, choices=sorted(FIXTURES))
    ap.add_argument("--device", default=None, help="warp device (default cuda:0 when available else cpu)")
    ap.add_argument("--seeds", type=int, default=None, help="use the first K seeds of the fixture (smoke runs; the protocol uses all)")
    ap.add_argument("--bench", default=None, help="comma list of N: measure episodes/s cpu vs warp instead of parity")
    ap.add_argument("--skip-single", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    fx = dict(FIXTURES[a.fixture])
    if a.seeds:
        fx["seeds"] = fx["seeds"][:a.seeds]
    if a.bench:
        out = dict(fixture=a.fixture, bench=bench(fx, a.device, [int(x) for x in a.bench.split(",")]))
    else:
        out = dict(fixture=a.fixture, spec={k: v for k, v in fx.items()}, device=a.device)
        if not a.skip_single:
            out["single_step"] = single_step(fx, a.device)
            print("single_step", json.dumps(out["single_step"]), flush=True)
        out["closed_loop"] = closed_loop(fx, a.device)
        cl = out["closed_loop"]
        print("closed_loop outcomes_identical", cl["outcomes_identical"], "within_tolerance", cl["within_tolerance"], flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    return 0

