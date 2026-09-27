"""Arm scripted-teacher quality report (W7): success / failure stage, motion smoothness, grasp physics.

Runs a pick_place teacher episode (any registered teacher version, `rrp.teachers.arm_smooth.ARM_TEACHERS`) and records
per control tick the commanded and measured arm joints, the commanded/measured TCP, the phase, IK residuals and the
object-gripper contacts. All of this is PRIVILEGED diagnostics (simulator truth); nothing feeds a policy.

Metrics (per episode; control rate 1/dt, finite differences):
- success (privileged evaluator), failure phase / reason, time to done, feasibility (the v1 feasibility check, which
  defines the evaluation seed sets, is used for every version so the seed sets stay identical).
- joint-space commanded and measured velocity / acceleration / jerk peaks and RMS (rad/s, rad/s^2, rad/s^3);
  Cartesian TCP measured (and commanded via FK of the command) acceleration / jerk peaks (m/s^2, m/s^3).
- velocity discontinuity: max |dq_cmd[k+1] - dq_cmd[k]| (rad/s) at phase switches (window +-1 tick) and anywhere.
- joint-limit margin: min over time/joints of distance to the nearest limit as a fraction of the range.
- gripper timing: close-command time, first held tick (truth), close->held latency, whether lift started before
  the object was held, drops (held lost before the open command).
- contact: max penetration depth object<->robot geoms (m, from MuJoCo contact dist), slip of the object in the TCP
  frame while held during lift/transport/lower (m, max deviation from its value at the first held lift tick).
- IK: max IK position residual per phase; ticks with residual > 1 cm.

CLI: python -m rrp.evaluation.teacher_quality --bodies panda_pg2,... --seeds 0-49 --versions v1,v2 --out X.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

SWITCH_WINDOW = 1


def _parse_seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _robot_bodies(session, r) -> set[str]:
    return {l.name for l in r.spec.links}


def _hand_bodies(session, r) -> set[str]:
    asm = next(a for a in r.spec.assemblies if a.kind in ("gripper", "hand"))
    return {l.name for l in r.spec.links if l.address in asm.members}


class _IKRecorder:
    """Wraps robot.ik.solve to record the residual of every call made by the teacher (no behaviour change)."""

    def __init__(self, ik):
        self.ik = ik
        self.orig = ik.solve
        self.last = None

    def __call__(self, *a, **k):
        q, e = self.orig(*a, **k)
        self.last = float(e)
        return q, e

    def install(self):
        self.ik.solve = self

    def remove(self):
        self.ik.solve = self.orig


def run_quality_episode(robot_key: str, seed: int, version: str = "v1", *, max_steps: int = 600,
                        robot=None, frames: dict | None = None, keep_trace: bool = False) -> dict:
    """One teacher episode with full diagnostics. `frames` = dict(renderer=..., camera=..., every=..., out=[...],
    caption=callable) to also collect rendered frames."""
    from rrp.bodies.catalog import workbench_robots
    from rrp.envs.native import Session
    from rrp.envs.scenario import BUILDERS
    from rrp.teachers.arm import PickPlaceTeacher
    from rrp.teachers.arm_smooth import make_arm_teacher, teacher_source

    robot = robot or workbench_robots()[robot_key]()
    s = Session(BUILDERS["pick_place"](robot, seed, n_distractors=seed % 3), seed=seed)
    m, d = s.model, s.data
    row = dict(robot=robot_key, seed=seed, version=version, source=teacher_source(version), privileged=True,
               n_distractors=seed % 3)
    # the v1 feasibility check defines the seed set for every version (identical evaluation seeds)
    feas = PickPlaceTeacher(s).feasibility()
    row["feasible"] = bool(feas["feasible"])
    row["feasibility"] = feas
    if not feas["feasible"]:
        row.update(outcome="infeasible", success=False)
        return row
    teacher = make_arm_teacher(s, version)
    r = teacher.r
    rec = _IKRecorder(r.ik)
    rec.install()
    cube_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, teacher.obj)
    robot_bodies, hand_bodies = _robot_bodies(s, r), _hand_bodies(s, r)
    body_name = [m.body(i).name for i in range(m.nbody)]
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, teacher.tcp_site)
    arm_q = r.ik.qadr
    lo, hi = r.ik.lo, r.ik.hi
    T = dict(phase=[], q_cmd=[], grip=[], q=[], tcp=[], tcp_R=[], obj=[], held=[], pen=[], pen_hand=[], ik=[],
             tcp_cmd_fk=[], n_hand_contacts=[])
    t0 = time.time()
    steps = 0
    error = None
    try:
        for k in range(max_steps):
            rec.last = None
            cmd = teacher.act()
            s.step(cmd)
            steps = k + 1
            q_cmd = np.asarray(cmd.groups["arm"], float)
            T["phase"].append(teacher.phase)
            T["q_cmd"].append(q_cmd)
            T["grip"].append(float(cmd.groups["gripper"][0]))
            T["q"].append(d.qpos[arm_q].copy())
            T["tcp"].append(d.site_xpos[sid].copy())
            T["tcp_R"].append(d.site_xmat[sid].reshape(3, 3).copy())
            T["obj"].append(d.xpos[cube_bid].copy())
            T["ik"].append(rec.last if rec.last is not None else np.nan)
            pen = pen_h = 0.0
            touching = set()
            for c in range(d.ncon):
                con = d.contact[c]
                b1, b2 = body_name[m.geom_bodyid[con.geom1]], body_name[m.geom_bodyid[con.geom2]]
                if teacher.obj not in (b1, b2):
                    continue
                other = b2 if b1 == teacher.obj else b1
                if other in robot_bodies:
                    pen = max(pen, -float(con.dist))
                    if other in hand_bodies:
                        pen_h = max(pen_h, -float(con.dist))
                        touching.add(other)
            T["pen"].append(pen)
            T["pen_hand"].append(pen_h)
            T["n_hand_contacts"].append(len(touching))
            T["held"].append(len(touching) >= 2)
            p_fk, _ = r.ik.fk(d.qpos.copy(), q_cmd)
            T["tcp_cmd_fk"].append(p_fk)
            if frames is not None and k % frames.get("every", 1) == 0:
                frames["renderer"].update_scene(d, camera=frames.get("camera", "front"))
                frames["out"].append(frames["caption"](frames["renderer"].render().copy(), s, teacher, k))
            if teacher.done:
                break
        s.step(None)
    except Exception as e:  # noqa: BLE001 - a crash is a recorded failure, never hidden
        error = repr(e)[:300]
    finally:
        rec.remove()
    ok = bool(s.privileged_success()) if error is None else False
    row.update(success=ok, outcome="success" if ok else "failure", steps=steps, time_s=round(steps * s.dt, 3),
               end_phase=teacher.phase, error=error, wall_s=round(time.time() - t0, 2), dt=s.dt,
               teacher_state=getattr(teacher, "diag", None))
    row.update(_metrics(T, s.dt, lo, hi, teacher))
    row["failure_stage"] = None if ok else _failure_stage(row, T)
    if keep_trace:
        row["_trace"] = T
    return row


def _fd(x, dt, n):
    for _ in range(n):
        x = np.diff(x, axis=0) / dt
    return x


def _peak(x):
    return float(np.max(np.abs(x))) if x.size else 0.0


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0


def run_policy_quality_episode(policy, robot_key: str, seed: int, *, label: str, max_steps: int = 300, robot=None) -> dict:
    """The same motion metrics for a LEARNED chunk policy (rrp.controllers.policy_runner.LearnedPolicy), rolled out
    like the ladder `learned` route (execute_prefix rows per chunk, public observations only). Phases are the chunk
    index, so `vel_jump_switch_max` = the largest joint-velocity step at chunk boundaries."""
    from rrp.bodies.catalog import workbench_robots
    from rrp.envs.native import Session
    from rrp.envs.scenario import BUILDERS
    from rrp.teachers.arm import PickPlaceTeacher
    robot = robot or workbench_robots()[robot_key]()
    s = Session(BUILDERS["pick_place"](robot, seed, n_distractors=seed % 3), seed=seed)
    m, d = s.model, s.data
    row = dict(robot=robot_key, seed=seed, version="policy", source=label, privileged=False, n_distractors=seed % 3)
    feas = PickPlaceTeacher(s).feasibility()          # same seed-set definition as every other route
    row["feasible"] = bool(feas["feasible"])
    if not feas["feasible"]:
        row.update(outcome="infeasible", success=False)
        return row
    r = s.robots[0]
    tcp_site = r.tcp_sites[next(a.id for a in r.spec.assemblies if a.kind in ("gripper", "hand"))]
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, tcp_site)
    cube_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
    robot_bodies, hand_bodies = _robot_bodies(s, r), _hand_bodies(s, r)
    body_name = [m.body(i).name for i in range(m.nbody)]
    T = dict(phase=[], q_cmd=[], grip=[], q=[], tcp=[], tcp_R=[], obj=[], held=[], pen=[], pen_hand=[], ik=[],
             tcp_cmd_fk=[], n_hand_contacts=[])
    nchunk, steps, error, t0 = 0, 0, None, time.time()
    q_last = r.controller.current_targets(d)["arm"].copy()
    try:
        for k in range(max_steps):
            if not s.executor.queue:
                s.submit_chunk(policy.chunks([s])[0], execute_prefix=policy.execute_prefix)
                nchunk += 1
            out = s.step(None)
            steps = k + 1
            cmd = out.command or {}
            q_cmd = np.asarray(cmd.get("arm", q_last), float)
            q_last = q_cmd
            T["phase"].append(f"chunk{nchunk}")
            T["q_cmd"].append(q_cmd)
            T["grip"].append(float(cmd["gripper"][0]) if "gripper" in cmd else np.nan)
            T["q"].append(d.qpos[r.ik.qadr].copy())
            T["tcp"].append(d.site_xpos[sid].copy())
            T["tcp_R"].append(d.site_xmat[sid].reshape(3, 3).copy())
            T["obj"].append(d.xpos[cube_bid].copy())
            T["ik"].append(np.nan)
            pen = pen_h = 0.0
            touching = set()
            for c in range(d.ncon):
                con = d.contact[c]
                b1, b2 = body_name[m.geom_bodyid[con.geom1]], body_name[m.geom_bodyid[con.geom2]]
                if "cube" not in (b1, b2):
                    continue
                other = b2 if b1 == "cube" else b1
                if other in robot_bodies:
                    pen = max(pen, -float(con.dist))
                    if other in hand_bodies:
                        pen_h = max(pen_h, -float(con.dist))
                        touching.add(other)
            T["pen"].append(pen)
            T["pen_hand"].append(pen_h)
            T["n_hand_contacts"].append(len(touching))
            T["held"].append(len(touching) >= 2)
            T["tcp_cmd_fk"].append(r.ik.fk(d.qpos.copy(), q_cmd)[0])
            if s.runtime.succeeded():
                break
        s.step(None)
    except Exception as e:  # noqa: BLE001 - recorded failure
        error = repr(e)[:300]
    ok = bool(s.privileged_success()) if error is None else False
    row.update(success=ok, outcome="success" if ok else "failure", steps=steps, time_s=round(steps * s.dt, 3),
               error=error, wall_s=round(time.time() - t0, 2), dt=s.dt, n_chunks=nchunk)
    met = _metrics(T, s.dt, r.ik.lo, r.ik.hi, None)
    met["chunk_vel_jump_max"] = met.pop("vel_jump_switch_max", None)
    met["chunk_tcp_vel_jump_max"] = met.pop("tcp_vel_jump_switch_max", None)
    g = np.array(T["grip"], float)
    if g.size > 3 and np.isfinite(g).all():
        met["grip_cmd_step_max"] = float(np.max(np.abs(np.diff(g))))
    row.update(met)
    row["failure_stage"] = None if ok else ("crash" if error else "timeout_or_drop")
    return row


def _policy_job(args):
    ck, label, rk, seeds, max_steps, device = args
    from rrp.bodies.catalog import workbench_robots
    from rrp.controllers.policy_runner import LearnedPolicy
    import torch
    torch.set_num_threads(1)
    pol = LearnedPolicy.from_checkpoint(ck, device=device, nfe=8, execute_prefix=8, seed=0)
    robot = workbench_robots()[rk]()
    return [run_policy_quality_episode(pol, rk, sd, label=label, max_steps=max_steps, robot=robot) for sd in seeds]


def _metrics(T, dt, lo, hi, teacher) -> dict:
    out = {}
    if len(T["q"]) < 5:
        return out
    qc, q = np.array(T["q_cmd"]), np.array(T["q"])
    tcp, tcpc = np.array(T["tcp"]), np.array(T["tcp_cmd_fk"])
    for tag, x in (("cmd", qc), ("meas", q)):
        v, a, j = _fd(x, dt, 1), _fd(x, dt, 2), _fd(x, dt, 3)
        out[f"joint_{tag}_vel_peak"] = _peak(v)
        out[f"joint_{tag}_acc_peak"] = _peak(a)
        out[f"joint_{tag}_jerk_peak"] = _peak(j)
        out[f"joint_{tag}_jerk_rms"] = _rms(j)
        out[f"joint_{tag}_acc_rms"] = _rms(a)
    for tag, x in (("cmd", tcpc), ("meas", tcp)):
        v = _fd(x, dt, 1)
        a = _fd(x, dt, 2)
        j = _fd(x, dt, 3)
        out[f"tcp_{tag}_speed_peak"] = float(np.max(np.linalg.norm(v, axis=1)))
        out[f"tcp_{tag}_acc_peak"] = float(np.max(np.linalg.norm(a, axis=1)))
        out[f"tcp_{tag}_jerk_peak"] = float(np.max(np.linalg.norm(j, axis=1)))
        out[f"tcp_{tag}_jerk_rms"] = float(np.sqrt(np.mean(np.sum(j ** 2, axis=1))))
    # velocity discontinuities: dv[k] = v[k+1]-v[k] with v[k] = (q[k+1]-q[k])/dt, i.e. around tick k+1
    v = _fd(qc, dt, 1)
    dv = np.max(np.abs(np.diff(v, axis=0)), axis=1)              # rad/s per tick, index k -> tick k+1
    ph = T["phase"]
    sw = [k for k in range(1, len(ph)) if ph[k] != ph[k - 1]]
    at = []
    for k in sw:
        for kk in range(k - 1 - SWITCH_WINDOW, k + SWITCH_WINDOW):
            if 0 <= kk < len(dv):
                at.append(dv[kk])
    out["vel_jump_switch_max"] = float(max(at)) if at else 0.0
    out["vel_jump_any_max"] = float(dv.max()) if dv.size else 0.0
    vt = _fd(tcpc, dt, 1)
    dvt = np.linalg.norm(np.diff(vt, axis=0), axis=1)
    att = [dvt[kk] for k in sw for kk in range(k - 1 - SWITCH_WINDOW, k + SWITCH_WINDOW) if 0 <= kk < len(dvt)]
    out["tcp_vel_jump_switch_max"] = float(max(att)) if att else 0.0
    out["n_phase_switches"] = len(sw)
    rng_ = np.maximum(hi - lo, 1e-6)
    marg = np.minimum(q - lo, hi - q) / rng_
    out["joint_limit_margin_min"] = float(marg.min())
    out["joint_limit_margin_joint"] = int(np.unravel_index(np.argmin(marg), marg.shape)[1])
    # gripper timing
    held = np.array(T["held"])
    grip = np.array(T["grip"])
    closing = [k for k in range(len(ph)) if ph[k] in ("close", "grasp_close", "grasp_settle")]
    kc = closing[0] if closing else None
    out["t_close_cmd"] = kc * dt if kc is not None else None
    kh = next((k for k in range(kc or 0, len(held)) if held[k]), None) if kc is not None else None
    out["t_held"] = kh * dt if kh is not None else None
    out["close_to_held_s"] = (kh - kc) * dt if (kh is not None and kc is not None) else None
    klift = next((k for k in range(len(ph)) if ph[k] == "lift"), None)
    out["lift_before_held"] = bool(klift is not None and (kh is None or klift < kh))
    kopen = next((k for k in range(len(ph)) if ph[k] == "open"), None)
    drops = 0
    if kh is not None:
        end = kopen if kopen is not None else len(held)
        for k in range(kh + 1, end):
            if held[k - 1] and not held[k]:
                drops += 1
    out["held_lost_before_open"] = drops
    carry = [k for k in range(len(ph)) if ph[k] in ("lift", "transport", "lower") and held[k]]
    out["held_frac_carry"] = (len(carry) / max(1, sum(p in ("lift", "transport", "lower") for p in ph)))
    if carry:
        rel = np.array([T["tcp_R"][k].T @ (T["obj"][k] - T["tcp"][k]) for k in carry])
        out["slip_max_m"] = float(np.max(np.linalg.norm(rel - rel[0], axis=1)))
    else:
        out["slip_max_m"] = None
    out["pen_max_m"] = float(np.max(T["pen"]))
    out["pen_hand_max_m"] = float(np.max(T["pen_hand"]))
    grasp_ticks = [k for k in range(len(ph)) if ph[k] not in ("pregrasp", "descend", "approach")]
    out["pen_hand_max_grasp_m"] = float(max(T["pen_hand"][k] for k in grasp_ticks)) if grasp_ticks else 0.0
    ik = np.array(T["ik"], float)
    per = {}
    for p in dict.fromkeys(ph):
        vals = ik[[k for k in range(len(ph)) if ph[k] == p]]
        vals = vals[np.isfinite(vals)]
        per[p] = float(vals.max()) if vals.size else None
    out["ik_err_max_by_phase"] = per
    out["ik_err_gt1cm_ticks"] = int(np.sum(np.nan_to_num(ik) > 0.01))
    out["phase_durations_s"] = {p: sum(1 for x in ph if x == p) * dt for p in dict.fromkeys(ph)}
    # tracking error (measured TCP vs FK of the command)
    out["track_err_max_m"] = float(np.max(np.linalg.norm(tcp - tcpc, axis=1)))
    out["obj_final_z"] = float(T["obj"][-1][2])
    return out


def _failure_stage(row, T) -> str:
    ph = T["phase"]
    end = ph[-1] if ph else "none"
    if row.get("error"):
        return f"crash:{end}"
    ev = [e["what"].split(" ")[0] for e in ((row.get("teacher_state") or {}).get("events") or [])]
    if "give_up" in ev:
        return "gave_up:" + ",".join(e for e in ev if e != "give_up")
    if row.get("t_held") is None:
        return f"grasp_never_held:{end}"
    if row.get("held_lost_before_open", 0) > 0:
        return f"dropped:{end}"
    if end not in ("retreat",):
        return f"timeout_in:{end}"
    return "placed_outside_zone"


def summarize(rows: list[dict]) -> dict:
    """Per (robot, version): counts, failure stages and medians/max of the smoothness metrics."""
    import collections
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["robot"], r["version"])].append(r)
    keys = ["joint_cmd_jerk_peak", "joint_cmd_jerk_rms", "joint_cmd_acc_peak", "joint_meas_jerk_peak",
            "joint_meas_jerk_rms", "joint_meas_acc_peak", "tcp_meas_jerk_peak", "tcp_meas_jerk_rms", "tcp_meas_acc_peak",
            "tcp_cmd_jerk_peak", "vel_jump_switch_max", "vel_jump_any_max", "tcp_vel_jump_switch_max",
            "joint_limit_margin_min", "time_s", "close_to_held_s", "pen_hand_max_grasp_m", "slip_max_m",
            "track_err_max_m"]
    out = {}
    for (rb, v), rs in sorted(g.items()):
        feas = [r for r in rs if r.get("feasible")]
        succ = [r for r in feas if r.get("success")]
        e = dict(n=len(rs), feasible=len(feas), success=len(succ),
                 failure_stages=dict(collections.Counter(r["failure_stage"] for r in feas if not r["success"])),
                 lift_before_held=sum(bool(r.get("lift_before_held")) for r in feas),
                 dropped=sum(1 for r in feas if r.get("held_lost_before_open", 0) > 0))
        for k in keys:
            vals = np.array([r[k] for r in feas if r.get(k) is not None], float)
            if vals.size:
                e[k] = dict(median=float(np.median(vals)), p90=float(np.percentile(vals, 90)), max=float(vals.max()),
                            min=float(vals.min()))
        out[f"{rb}|{v}"] = e
    return out


def _job(args):
    rk, seeds, version, max_steps = args
    from rrp.bodies.catalog import workbench_robots
    robot = workbench_robots()[rk]()
    rows = []
    for sd in seeds:
        rows.append(run_quality_episode(rk, sd, version, max_steps=max_steps, robot=robot))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", required=True)
    ap.add_argument("--seeds", required=True, help="e.g. 0-49,3000000-3000029")
    ap.add_argument("--versions", default="v1")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", action="append", default=[], help="label=checkpoint (learned chunk policy mode)")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    seeds = _parse_seeds(a.seeds)
    if a.policy:
        return _main_policy(a, seeds)
    jobs = [(rk, seeds[i:i + a.chunk], v, a.max_steps) for v in a.versions.split(",") for rk in a.bodies.split(",")
            for i in range(0, len(seeds), a.chunk)]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    if a.workers <= 1:
        results = map(_job, jobs)
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        ex = ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn"))
        results = ex.map(_job, jobs)
    with open(out, "w") as f:
        for i, rs in enumerate(results):
            for r in rs:
                f.write(json.dumps(r, default=float) + "\n")
                rows.append(r)
            f.flush()
            print(f"[teacher_quality] {i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    summ = summarize(rows)
    Path(str(out).replace(".jsonl", ".summary.json")).write_text(json.dumps(summ, indent=1))
    for k, e in summ.items():
        print(k, f"{e['success']}/{e['feasible']}", e["failure_stages"],
              "jerk_cmd_peak_med=%.1f" % e.get("joint_cmd_jerk_peak", {}).get("median", np.nan),
              "vjump_sw_med=%.2f" % e.get("vel_jump_switch_max", {}).get("median", np.nan), flush=True)


def _main_policy(a, seeds):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    jobs = []
    for spec in a.policy:
        label, ck = spec.split("=", 1)
        for rk in a.bodies.split(","):
            for i in range(0, len(seeds), a.chunk):
                jobs.append((ck, label, rk, seeds[i:i + a.chunk], min(a.max_steps, 300), a.device))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ex = ProcessPoolExecutor(max_workers=max(1, a.workers), mp_context=mp.get_context("spawn"))
    rows = []
    with open(out, "w") as f:
        for i, rs in enumerate(ex.map(_policy_job, jobs)):
            for r in rs:
                f.write(json.dumps(r, default=float) + "\n")
                rows.append(r)
            f.flush()
            print(f"[policy_quality] {i + 1}/{len(jobs)}", flush=True)
    ex.shutdown(wait=True)
    for r in rows:
        r["version"] = r["source"]
    summ = summarize_policy(rows)
    Path(str(out).replace(".jsonl", ".summary.json")).write_text(json.dumps(summ, indent=1))
    for k, e in summ.items():
        print(k, json.dumps(e)[:400], flush=True)


def summarize_policy(rows):
    import collections
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["robot"], r["source"])].append(r)
    keys = ["joint_cmd_jerk_peak", "joint_cmd_jerk_rms", "joint_cmd_acc_peak", "joint_meas_jerk_peak", "joint_meas_jerk_rms",
            "tcp_meas_jerk_peak", "tcp_meas_jerk_rms", "chunk_vel_jump_max", "vel_jump_any_max", "grip_cmd_step_max",
            "time_s", "pen_hand_max_m"]
    out = {}
    for (rb, src), rs in sorted(g.items()):
        fe = [r for r in rs if r.get("feasible")]
        e = dict(n=len(rs), feasible=len(fe), success=sum(r["success"] for r in fe))
        for k in keys:
            vals = np.array([r[k] for r in fe if r.get(k) is not None], float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                e[k] = dict(median=float(np.median(vals)), p90=float(np.percentile(vals, 90)), max=float(vals.max()))
        out[f"{rb}|{src}"] = e
    return out


if __name__ == "__main__":
    main()
