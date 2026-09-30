"""Arm scripted-teacher quality report (W7): success / failure stage, motion smoothness, grasp physics.

Runs a pick_place teacher episode (any registered teacher version, `rrp.policies.teachers.arm_smooth.ARM_TEACHERS`) and records
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

CLI: python -m rrp.cli suite teacher-quality --bodies panda_pg2,... --seeds 0-49 --versions v1,v2 --out X.jsonl
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


def _scale_object(s, obj_friction: float, obj_mass: float) -> dict:
    """W6-style object perturbation: x friction of the task cube AND the finger pads, x mass of the cube."""
    m = s.model
    if obj_friction == 1.0 and obj_mass == 1.0:
        return {}
    from rrp.bodies.grasp_contact import PAD_RE
    cube = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
    gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == cube or PAD_RE.search(m.geom(g).name or "")]
    for g in gs:
        m.geom_friction[g] *= obj_friction
    if obj_mass != 1.0:
        m.body_mass[cube] *= obj_mass
        m.body_inertia[cube] *= obj_mass
        mujoco.mj_setConst(m, s.data)
    return dict(obj_friction_scale=obj_friction, obj_mass_scale=obj_mass, cube_mass_kg=float(m.body_mass[cube]))


def _touching(m, d, body_name, obj, robot_bodies, hand_bodies):
    """(max robot<->object penetration, max hand<->object penetration, hand bodies touching the object) this tick."""
    pen = pen_h = 0.0
    touching = set()
    for c in range(d.ncon):
        con = d.contact[c]
        b1, b2 = body_name[m.geom_bodyid[con.geom1]], body_name[m.geom_bodyid[con.geom2]]
        if obj not in (b1, b2):
            continue
        other = b2 if b1 == obj else b1
        if other in robot_bodies:
            pen = max(pen, -float(con.dist))
            if other in hand_bodies:
                pen_h = max(pen_h, -float(con.dist))
                touching.add(other)
    return pen, pen_h, len(touching)


class QualityTrace:
    """Rollout hook: the per-tick PRIVILEGED diagnostics of a quality episode (commanded / measured arm joints, TCP, object,
    contacts, IK residual), read-only. `policy` is a TeacherPolicy (the tick's command is the teacher's, the phase its
    FSM phase, IK residuals from a wrapped `robot.ik.solve`); None = a chunk policy (the command is the executed row of
    the env's chunk executor, the phase is the chunk index). `frames` (teacher mode) = dict(renderer, camera, every, out,
    caption). After the episode `result[i]` holds the trace `T` and what `_metrics` needs."""

    def __init__(self, policy=None, *, frames: dict | None = None):
        self.policy, self.frames = policy, frames
        self.st: dict = {}
        self.result: dict = {}

    def on_reset(self, i, env, obs):
        m = env.model
        if self.policy is not None:
            teacher = self.policy.teachers[i]
            r, obj, site = teacher.r, teacher.obj, teacher.tcp_site
            rec = _IKRecorder(r.ik)
            rec.install()
        else:
            teacher, rec, obj = None, None, "cube"
            r = env.robots[0]
            site = r.tcp_sites[next(a.id for a in r.spec.assemblies if a.kind in ("gripper", "hand"))]
        self.st[i] = dict(
            teacher=teacher, rec=rec, r=r, obj=obj, sid=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site),
            cube=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, obj), robot_bodies=_robot_bodies(env, r),
            hand_bodies=_hand_bodies(env, r), body_name=[m.body(b).name for b in range(m.nbody)], k=0, nchunk=0,
            q_last=None if self.policy is not None else r.controller.current_targets(env.data)["arm"].copy(),
            T=dict(phase=[], q_cmd=[], grip=[], q=[], tcp=[], tcp_R=[], obj=[], held=[], pen=[], pen_hand=[], ik=[],
                   tcp_cmd_fk=[], n_hand_contacts=[]))

    def on_step(self, i, env, act, step):
        st = self.st[i]
        d, T, r, teacher = env.data, st["T"], st["r"], st["teacher"]
        if teacher is not None:
            q_cmd = np.asarray(act.command.groups["arm"], float)
            grip = float(act.command.groups["gripper"][0])
            T["phase"].append(teacher.phase)
            T["ik"].append(st["rec"].last if st["rec"].last is not None else np.nan)
            st["rec"].last = None
        else:
            st["nchunk"] += act.chunk is not None
            cmd = step.command or {}
            q_cmd = np.asarray(cmd.get("arm", st["q_last"]), float)
            st["q_last"] = q_cmd
            grip = float(cmd["gripper"][0]) if "gripper" in cmd else np.nan
            T["phase"].append(f"chunk{st['nchunk']}")
            T["ik"].append(np.nan)
        T["q_cmd"].append(q_cmd)
        T["grip"].append(grip)
        T["q"].append(d.qpos[r.ik.qadr].copy())
        T["tcp"].append(d.site_xpos[st["sid"]].copy())
        T["tcp_R"].append(d.site_xmat[st["sid"]].reshape(3, 3).copy())
        T["obj"].append(d.xpos[st["cube"]].copy())
        pen, pen_h, n = _touching(env.model, d, st["body_name"], st["obj"], st["robot_bodies"], st["hand_bodies"])
        T["pen"].append(pen)
        T["pen_hand"].append(pen_h)
        T["n_hand_contacts"].append(n)
        T["held"].append(n >= 2)
        T["tcp_cmd_fk"].append(r.ik.fk(d.qpos.copy(), q_cmd)[0])
        fr = self.frames
        if fr is not None and teacher is not None and st["k"] % fr.get("every", 1) == 0:
            cam = fr.get("camera", "front")
            fr["renderer"].update_scene(d, camera=cam(env.model, d) if callable(cam) else cam)
            fr["out"].append(fr["caption"](fr["renderer"].render().copy(), env, teacher, st["k"]))
        st["k"] += 1

    def on_end(self, i, env, ep):
        st = self.st.pop(i)
        if st["rec"] is not None:
            st["rec"].remove()
        self.result[i] = dict(T=st["T"], teacher=st["teacher"], lo=st["r"].ik.lo, hi=st["r"].ik.hi, nchunk=st["nchunk"])
        return {}


def _quality_rollout(env, policy, seed, max_steps, trace, done, hooks=()):
    """One episode of `env` under `policy` (rollout, batch 1): ticks until max_steps or done(env); then one hold tick
    and the privileged verdict (Settle) - the episode row is built from the trace by the caller."""
    from rrp.harness import rollout as R
    from rrp.harness.eval import hooks as H
    return R.rollout(lambda sd: env, policy, H.budget_task("pick_place", env.spec.env_id), [seed], batch=1,
                     max_steps=max_steps, hooks=[H.EndWhen(lambda i, e: done(i, e)), H.Settle(1), trace, *hooks])[0]


def run_quality_episode(robot_key: str, seed: int, version: str = "v1", *, max_steps: int = 600,
                        robot=None, frames: dict | None = None, keep_trace: bool = False,
                        obj_friction: float = 1.0, obj_mass: float = 1.0, hooks=()) -> dict:
    """One teacher episode with full diagnostics. `frames` = dict(renderer=..., camera=..., every=..., out=[...],
    caption=callable) to also collect rendered frames. The episode is a `harness.rollout` (scripted teacher policy,
    QualityTrace hook); a crash is a recorded failure (`error` = the rollout's crash note), never hidden. `hooks`: extra
    rollout hooks (read-only recorders such as the replay recorder, rrp.viz.record) after the episode's own."""
    from rrp.bodies.catalog import workbench_robots
    from rrp.envs.mujoco.session import Session
    from rrp.envs.mujoco.scenario import BUILDERS
    from rrp.policies.teachers import make_arm_teacher_policy
    from rrp.policies.teachers.arm import PickPlaceTeacher
    from rrp.policies.teachers.arm_smooth import teacher_source

    robot = robot or workbench_robots()[robot_key]()
    s = Session(BUILDERS["pick_place"](robot, seed, n_distractors=seed % 3), seed=seed)
    from rrp.bodies.grasp_contact import model_grasp_version
    row = dict(robot=robot_key, seed=seed, version=version, source=teacher_source(version), privileged=True,
               n_distractors=seed % 3, grasp_contact=model_grasp_version(s.model) or "grasp_v1")
    # the v1 feasibility check defines the seed set for every version (identical evaluation seeds)
    feas = PickPlaceTeacher(s).feasibility()
    row["feasible"] = bool(feas["feasible"])
    row["feasibility"] = feas
    if not feas["feasible"]:
        row.update(outcome="infeasible", success=False)
        return row
    row["perturbation"] = _scale_object(s, obj_friction, obj_mass)
    pol = make_arm_teacher_policy(arg="pick_place", version=version)
    trace = QualityTrace(pol, frames=frames)
    t0 = time.time()
    ep = _quality_rollout(s, pol, seed, max_steps, trace, lambda i, e: pol.teachers[i].done, hooks)
    res = trace.result[0]
    T, teacher = res["T"], res["teacher"]
    error = ep.metrics.get("note", "crash") if ep.outcome == "crash" else None
    ok = bool(ep.success_privileged) if error is None else False
    row.update(success=ok, outcome="success" if ok else "failure", steps=ep.steps, time_s=round(ep.steps * s.dt, 3),
               end_phase=teacher.phase, error=error, wall_s=round(time.time() - t0, 2), dt=s.dt,
               teacher_state=getattr(teacher, "diag", None))
    row.update(_metrics(T, s.dt, res["lo"], res["hi"], teacher))
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


def run_policy_quality_episode(policy, robot_key: str, seed: int, *, label: str, max_steps: int = 300, robot=None,
                               ckpt: str | None = None, source_labels: bool | None = None) -> dict:
    """The same motion metrics for a LEARNED chunk policy (rrp.policies.bc.LearnedPolicy), rolled out like the ladder
    `learned` route (execute_prefix rows per chunk, public observations only; BCPolicy on harness.rollout). Phases are
    the chunk index, so `vel_jump_switch_max` = the largest joint-velocity step at chunk boundaries."""
    from rrp.bodies.catalog import workbench_robots
    from rrp.envs.mujoco.session import Session
    from rrp.envs.mujoco.scenario import BUILDERS
    from rrp.policies.bc import BCPolicy
    from rrp.policies.teachers.arm import PickPlaceTeacher
    robot = robot or workbench_robots()[robot_key]()
    s = Session(BUILDERS["pick_place"](robot, seed, n_distractors=seed % 3), seed=seed)
    row = dict(robot=robot_key, seed=seed, version="policy", source=label, privileged=False, n_distractors=seed % 3)
    if ckpt is not None:        # D-126 sl-1 (default off): the free `label` stays; the canonical label names the checkpoint
        from rrp.core.provenance import Source, parse_legacy_source, stamp_source_label
        try:
            kind = "bc" if parse_legacy_source(label).kind is Source.BC else "learned"
        except ValueError:
            kind = "learned"
        stamp_source_label(row, kind, str(ckpt), enabled=source_labels)
    feas = PickPlaceTeacher(s).feasibility()          # same seed-set definition as every other route
    row["feasible"] = bool(feas["feasible"])
    if not feas["feasible"]:
        row.update(outcome="infeasible", success=False)
        return row
    trace = QualityTrace(None)
    t0 = time.time()
    ep = _quality_rollout(s, BCPolicy(policy, name=label), seed, max_steps, trace,
                          lambda i, e: bool(e.runtime.succeeded()))
    res = trace.result[0]
    T, r = res["T"], s.robots[0]
    error = ep.metrics.get("note", "crash") if ep.outcome == "crash" else None
    ok = bool(ep.success_privileged) if error is None else False
    row.update(success=ok, outcome="success" if ok else "failure", steps=ep.steps, time_s=round(ep.steps * s.dt, 3),
               error=error, wall_s=round(time.time() - t0, 2), dt=s.dt, n_chunks=res["nchunk"])
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
    from rrp.policies.bc import LearnedPolicy
    import torch
    torch.set_num_threads(1)
    pol = LearnedPolicy.from_checkpoint(ck, device=device, nfe=8, execute_prefix=8, seed=0)
    robot = workbench_robots()[rk]()
    return [run_policy_quality_episode(pol, rk, sd, label=label, max_steps=max_steps, robot=robot, ckpt=ck) for sd in seeds]


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
    # the D-108 gate: penetration while the object is HELD during the lift / transport (carry) phases
    lc = [k for k in range(len(ph)) if ph[k] in ("lift", "transport")]
    lch = [k for k in lc if held[k]]
    out["pen_hand_max_carry_m"] = float(max((T["pen_hand"][k] for k in lch), default=0.0))
    out["pen_hand_med_carry_m"] = float(np.median([T["pen_hand"][k] for k in lch])) if lch else 0.0
    out["held_frac_lift_transport"] = float(np.mean([held[k] for k in lc])) if lc else None
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
        pz = r.get("perturbation") or {}
        tag = r["version"] + (f"|of{pz.get('obj_friction_scale', 1.0):g}|om{pz.get('obj_mass_scale', 1.0):g}" if pz else "")
        g[(r["robot"], tag)].append(r)
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
    rk, seeds, version, max_steps = args[:4]
    of, om = (list(args[4:6]) + [1.0, 1.0])[:2] if len(args) > 4 else (1.0, 1.0)
    from rrp.bodies.catalog import workbench_robots
    robot = workbench_robots()[rk]()
    rows = []
    for sd in seeds:
        rows.append(run_quality_episode(rk, sd, version, max_steps=max_steps, robot=robot, obj_friction=of, obj_mass=om))
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
    ap.add_argument("--obj-friction", default="1.0", help="comma list: x friction of the cube and finger pads")
    ap.add_argument("--obj-mass", default="1.0", help="comma list: x cube mass")
    ap.add_argument("--resume", action="store_true", help="keep rows of complete chunks already in --out")
    a = ap.parse_args(argv)
    seeds = _parse_seeds(a.seeds)
    if a.policy:
        return _main_policy(a, seeds)
    jobs = [(rk, seeds[i:i + a.chunk], v, a.max_steps, of, om) for v in a.versions.split(",") for rk in a.bodies.split(",")
            for of in [float(x) for x in a.obj_friction.split(",")] for om in [float(x) for x in a.obj_mass.split(",")]
            for i in range(0, len(seeds), a.chunk)]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if a.resume and out.exists():       # keep complete chunks of an interrupted run (rows are written per chunk)
        for l in out.read_text().splitlines():
            try:
                rows.append(json.loads(l))
            except ValueError:
                pass

        def key(rk, sd, v, of, om):
            return (rk, sd, v, float(of), float(om))
        have = {key(r["robot"], r["seed"], r["version"], (r.get("perturbation") or {}).get("obj_friction_scale", 1.0),
                    (r.get("perturbation") or {}).get("obj_mass_scale", 1.0)) for r in rows}
        jobs = [j for j in jobs if not all(key(j[0], sd, j[2], j[4], j[5]) in have for sd in j[1])]
        out.write_text("".join(json.dumps(r, default=float) + "\n" for r in rows))
        print(f"[teacher_quality] resume: {len(rows)} rows kept, {len(jobs)} chunks to run", flush=True)
    t0 = time.time()
    if a.workers <= 1:
        results = map(_job, jobs)
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        ex = ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn"))
        results = ex.map(_job, jobs)
    with open(out, "a" if a.resume else "w") as f:
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
