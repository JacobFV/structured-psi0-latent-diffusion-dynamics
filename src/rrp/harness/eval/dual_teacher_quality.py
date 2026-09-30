"""Dual-arm scripted-teacher audit (W12 phase B): W6/D-112 motion gates + W12 contact-frame metrics per episode.

Runs the dual teachers (rrp.policies.teachers.dual, label scripted_teacher, PRIVILEGED inputs) under the CURRENT physics
($RRP_GRASP_CONTACT, recorded per row) and records per control tick, read-only:
  commanded and measured arm joints per manipulator, the teacher phase per arm, robot<->object penetration, robot<->robot
  contacts, and the privileged contact frames (rrp.harness.data.contact_labels.ContactFrameRecorder).
Per episode and manipulator (the arm-teacher gate definitions of D-112/D-114, rrp.harness.eval.gates):
  jerk RMS / peak of measured and commanded joints; phase-switch velocity step (commanded joint velocity step at the
  arm's teacher phase switches, +-1 tick; gate <= 0.5 rad/s); joint-limit margin (gate >= 0.02); penetration max over
  task objects (gate <= 3 mm), ticks above 3 mm, arm-arm contact ticks.
Plus the W12 keys (rrp.harness.data.contact_metrics.dual_contact_motion): held-object drift vs gripper and vs the
supporting hand, support-anchor slip, contact-sequence order error vs the task spec, re-anchoring latency.
Optional DART (burst noise on executed arm commands, as rrp.harness.data.collect_dual) to audit noisy-episode penetration.

CLI (peer CPU, under a lease):
  python -m rrp.cli suite dual-teacher-quality --task support_insert --pairs A__B,C__D --seeds 0:8 \
      [--noise 0.04 --burst 20,4] --out artifacts/runs/w12_dualaudit/support_insert.jsonl --workers 2

Also here (moved from policies.teachers.dual_validate, readiness R2 DP): the teacher VALIDATION over arm pairs
(`run_one` / `summarize_validation` / `validate_main`; every episode recorded with its failure reason, public-vs-privileged
agreement and, for support_insert, the true insertion geometry) and `run_teacher_episode`, the one rollout of a scripted
dual teacher (the former `run_dual_teacher_episode`, which policies.teachers.functional_composition also runs on):
  python -m rrp.cli suite dual-validate --task support_insert --pairs parm5_pg2__parm5_pg2 --seeds 0:30 --workers 4 \
      --out artifacts/assets/dual_teacher_validation/support_insert.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mujoco
import numpy as np

from rrp.core.provenance import stamp_source_label

AUDIT_VERSION = "rrp.evaluation.dual_teacher_quality/v1"
PEN_GATE_M = 0.003
PHASE_STEP_GATE = 0.5
MARGIN_GATE = 0.02


class _AuditHook:
    """Rollout hook of the audit: DART burst noise on the executed arm commands (`on_act`), the per-tick read-only
    DualQualityRecorder (before/after each env tick) and the stop rules (teacher done / `stop_after_success` ticks after
    the public runtime first succeeded) as an on_step verdict."""

    def __init__(self, s, teacher, task, *, noise, burst, phase_gate, stop_after_success, seed):
        from rrp.harness.data.dual_quality import DualQualityRecorder
        self.s, self.teacher, self.task = s, teacher, task
        self.noise, self.burst, self.phase_gate, self.stop = noise, burst, phase_gate, stop_after_success
        m = s.model
        self.lo, self.hi = {}, {}
        for e, h in s.handles.items():
            jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in h.arm_joints]
            self.lo[e], self.hi[e] = m.jnt_range[jid, 0].copy(), m.jnt_range[jid, 1].copy()
        self.qrec = DualQualityRecorder(s, teacher)
        self.rng = np.random.default_rng([seed, 7])
        self.nz: dict = {}
        self.k, self.succ_at = 0, None

    def on_act(self, i, obs, act):
        from rrp.policies.base import Act
        from rrp.policies.teachers.dual_smooth import dart_phase_allowed
        s, k, clean = self.s, self.k, act.command
        cmds = clean
        if self.noise > 0 and k % self.burst[0] < self.burst[1]:
            noisy = {}
            for ri, c in clean.items():
                g2 = dict(c.groups)
                for e, h in s.handles.items():
                    if h.robot != ri or h.arm_group not in g2:
                        continue
                    if k % 5 == 0 or e not in self.nz:
                        self.nz[e] = self.rng.normal(0, self.noise, len(g2[h.arm_group]))
                    if self.phase_gate and not dart_phase_allowed(self.task, self.teacher.phase.get(e, "")):
                        continue
                    g2[h.arm_group] = np.clip(np.asarray(g2[h.arm_group], float) + self.nz[e], self.lo[e],
                                              self.hi[e]).tolist()
                noisy[ri] = c.model_copy(update={"groups": g2})
            cmds = noisy
        self.qrec.before_step(clean)
        return Act(cmds) if cmds is not clean else None

    def on_step(self, i, env, act, step):
        from rrp.tasks.spec import Judgement
        self.qrec.after_step()
        self.k += 1
        if self.teacher.done:
            return Judgement(True, "success", "teacher_done")
        if self.stop is not None:
            if self.succ_at is None and env.runtime.succeeded():
                self.succ_at = self.k
            if self.succ_at is not None and self.k - self.succ_at >= self.stop:
                return Judgement(True, "success", "stop_after_success")
        return None


def run_audit_episode(task: str, pair: str, seed: int, *, max_steps: int = 1200, noise: float = 0.0,
                      burst: tuple = (1, 1), stop_after_success: int | None = 10, teacher_version: str | None = None,
                      teacher_options: dict | None = None, phase_gate: bool = False,
                      source_labels: bool | None = None) -> dict:
    """One audited teacher episode. teacher_version None/'v2' = the default teacher; 'v3' = rrp.policies.teachers.dual_smooth.
    phase_gate: DART noise only in free-space phases (as collect_dual noise_phase_gate)."""
    from rrp.harness.data.dual_quality import DualQualityRecorder
    from rrp.bodies.grasp_contact import model_grasp_version
    from rrp.policies.teachers.dual_smooth import dart_phase_allowed, make_dual_teacher
    from rrp.policies.teachers.dual_validate import make_session
    t0 = time.time()
    s = make_session(task, pair, seed)
    teacher = make_dual_teacher(task, s, teacher_version, teacher_options)
    m, d = s.model, s.data
    f = teacher.feasibility()
    row = dict(version=AUDIT_VERSION, task=task, pair=pair, seed=seed, source="scripted_teacher", privileged_teacher=True,
               noise=noise, burst=list(burst), grasp_contact_version=model_grasp_version(m) or "grasp_v1",
               feasible=bool(f["feasible"]))
    stamp_source_label(row, "scripted_teacher", f"dual_{task}", enabled=source_labels)   # D-126 sl-1 (default off)
    if teacher_version not in (None, "v2"):
        row.update(teacher_version=teacher.teacher_version, teacher_options=dict(vars(teacher.o)))
    if phase_gate:
        row["dart_variant"] = "phase_gated_v1"
    if not f["feasible"]:
        row.update(status="infeasible", unreachable=f["unreachable"], wall_s=time.time() - t0)
        return row
    from rrp.harness import rollout as R
    from rrp.harness.eval import hooks as H
    from rrp.policies.teachers import TeacherPolicy
    audit = _AuditHook(s, teacher, task, noise=noise, burst=burst, phase_gate=phase_gate,
                       stop_after_success=stop_after_success, seed=seed)
    pol = TeacherPolicy(task, lambda e: teacher, f"dual:{teacher_version or 'default'}", ("joint_position", "gripper"))
    ep = R.rollout(lambda sd: s, pol, H.budget_task(task, s.spec.env_id), [seed], batch=1, max_steps=max_steps,
                   hooks=[audit, H.Settle(5)])[0]
    if ep.outcome == "crash":                      # a crashed episode is an error row of the caller (_job), never hidden
        raise RuntimeError(ep.metrics.get("note") or ep.failure_reason)
    qrec, steps = audit.qrec, ep.steps
    ok = bool(ep.success_privileged)
    q = qrec.summary(task)
    per, pen = q["per_arm"], q["penetration_max_m"]
    gate = dict(penetration=pen <= PEN_GATE_M,
                phase_switch=all((p["phase_switch_vel_step_max"] or 0) <= PHASE_STEP_GATE for p in per.values()),
                joint_margin=all((p["joint_limit_margin_min"] is not None and p["joint_limit_margin_min"] >= MARGIN_GATE)
                                 for p in per.values()))
    row.update(status="success" if ok else "failure", public_runtime_success=bool(ep.success_public),
               failure_phase=None if ok else teacher.phase_label, steps=steps, dt=q["dt"], per_arm=per,
               penetration_max_m=pen, penetration_ticks_over_3mm=q["penetration_ticks_over_3mm"],
               arm_arm_contact_ticks=q["arm_arm_contact_ticks"], gate=gate, contact=q["contact"],
               phase_switch_ticks=q["phase_switch_ticks"],
               statuses={e: v.status for e, v in s.runtime.instances.items()}, wall_s=time.time() - t0)
    if hasattr(teacher, "limits"):
        row["teacher_limits"] = dict(teacher.limits)
    return row


def _job(a):
    try:
        return run_audit_episode(*a[:3], **a[3])
    except Exception as e:  # noqa: BLE001 - errors are data
        return dict(task=a[0], pair=a[1], seed=a[2], status="error", error=repr(e)[:400], **{k: a[3][k] for k in ("noise",)})


def summarize(rows: list[dict]) -> dict:
    out = {}
    for key in sorted({(r["task"], r.get("noise", 0.0)) for r in rows}):
        rs = [r for r in rows if (r["task"], r.get("noise", 0.0)) == key]
        done = [r for r in rs if r.get("status") in ("success", "failure")]
        arm = [p for r in done for p in r["per_arm"].values()]
        med = lambda xs: float(np.median([x for x in xs if x is not None])) if any(x is not None for x in xs) else None
        mx = lambda xs: float(np.max([x for x in xs if x is not None])) if any(x is not None for x in xs) else None
        frac = lambda xs: float(np.mean(xs)) if xs else None
        c = [r["contact"] for r in done]
        # commanded-jerk ratio vs the arm v2 teacher on the same body (D-112 arm gate: <= 2x); left = first body of the pair
        from rrp.harness.eval.gates import ARM_TEACHER_V2_REFERENCE
        ref = ARM_TEACHER_V2_REFERENCE["bodies"]
        ratios = []
        for r in done:
            bodies = dict(zip(("left", "right"), r["pair"].split("__"))) if "__" in r["pair"] else {}
            for e, p in r["per_arm"].items():
                if bodies.get(e) in ref and p.get("cmd_jerk_rms") is not None:
                    ratios.append(p["cmd_jerk_rms"] / ref[bodies[e]])
        out[f"{key[0]}|noise={key[1]}"] = dict(
            n=len(rs), success=sum(r["status"] == "success" for r in rs), failure=sum(r["status"] == "failure" for r in rs),
            infeasible=sum(r["status"] == "infeasible" for r in rs), error=sum(r["status"] == "error" for r in rs),
            gate_penetration_pass=frac([r["gate"]["penetration"] for r in done]),
            gate_phase_switch_pass=frac([r["gate"]["phase_switch"] for r in done]),
            gate_joint_margin_pass=frac([r["gate"]["joint_margin"] for r in done]),
            penetration_max_m_median=med([r["penetration_max_m"] for r in done]),
            penetration_max_m_max=mx([r["penetration_max_m"] for r in done]),
            arm_arm_contact_episodes=sum(r["arm_arm_contact_ticks"] > 0 for r in done),
            phase_switch_step_median=med([p["phase_switch_vel_step_max"] for p in arm]),
            phase_switch_step_max=mx([p["phase_switch_vel_step_max"] for p in arm]),
            vel_step_any_median=med([p["vel_step_any_max"] for p in arm]),
            cmd_jerk_rms_median=med([p["cmd_jerk_rms"] for p in arm]),
            joint_jerk_rms_median=med([p["joint_jerk_rms"] for p in arm]),
            cmd_jerk_ratio_vs_arm_v2_median=med(ratios), cmd_jerk_ratio_vs_arm_v2_max=mx(ratios),
            gate_jerk_2x_pass=frac([x <= 2.0 for x in ratios]),
            joint_margin_min_median=med([p["joint_limit_margin_min"] for p in arm]),
            held_rot_drift_grip_max_rad_median=med([x.get("cf_held_rot_drift_grip_max_rad") for x in c]),
            held_rot_drift_grip_max_rad_max=mx([x.get("cf_held_rot_drift_grip_max_rad") for x in c]),
            held_pos_drift_grip_max_m_median=med([x.get("cf_held_pos_drift_grip_max_m") for x in c]),
            held_rot_drift_support_max_rad_median=med([x.get("cf_held_rot_drift_support_max_rad") for x in c]),
            support_anchor_slip_max_m_median=med([x.get("cf_support_anchor_slip_max_m") for x in c]),
            support_anchor_slip_max_m_max=mx([x.get("cf_support_anchor_slip_max_m") for x in c]),
            contact_order_error_mean=frac([x["cf_contact_order_error"] for x in c if x.get("cf_contact_order_error") is not None]),
            contact_missing_episodes=sum((x.get("cf_contact_missing") or 0) > 0 for x in c),
            reanchor_settle_mean_s_median=med([x.get("cf_reanchor_settle_mean_s") for x in c]),
            anchor_receipt_latency_mean_s_median=med([x.get("cf_anchor_receipt_latency_mean_s") for x in c]),
            failure_phases=sorted({r.get("failure_phase") for r in rs if r["status"] == "failure"}),
            grasp_contact_versions=sorted({r.get("grasp_contact_version") for r in rs if r.get("grasp_contact_version")}))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--seeds", default="0:8")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--burst", default="1,1")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--teacher-version", default=None, help="v2 (default) | v3 (rrp.policies.teachers.dual_smooth)")
    ap.add_argument("--teacher-options", default=None, help="JSON dict of V3Options (ablations)")
    ap.add_argument("--phase-gate", action="store_true", help="DART only in free-space phases (D-121 style)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    lo_, hi_ = map(int, a.seeds.split(":"))
    burst = tuple(int(x) for x in a.burst.split(","))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()] if out.exists() else []
    done = {(r["task"], r["pair"], r["seed"], r.get("noise", 0.0)) for r in rows}
    kw = dict(max_steps=a.max_steps, noise=a.noise, burst=burst, teacher_version=a.teacher_version,
              teacher_options=json.loads(a.teacher_options) if a.teacher_options else None, phase_gate=a.phase_gate)
    jobs = [(a.task, p, sd, kw)
            for p in a.pairs.split(",") for sd in range(lo_, hi_) if (a.task, p, sd, a.noise) not in done]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex, out.open("a") as fh:
        for i, fu in enumerate(as_completed([ex.submit(_job, j) for j in jobs])):
            r = fu.result()
            rows.append(r)
            fh.write(json.dumps(r, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)) + "\n")
            fh.flush()
            print(f"[dual_audit] {i + 1}/{len(jobs)} {r['task']} {r['pair']} s{r['seed']} {r['status']} "
                  f"{time.time() - t0:.0f}s", flush=True)
    summ = summarize(rows)
    out.with_suffix(".summary.json").write_text(json.dumps(summ, indent=1, sort_keys=True))
    print(json.dumps(summ, indent=1, sort_keys=True))


class _TeacherEpisode:
    """Rollout hook of one scripted dual-teacher episode: on_reset ends an infeasible layout ("infeasible", before its
    first tick) by the teacher's own feasibility check; on_step calls `callback(k, cmd, step)` (intervention / logging) and
    ends the episode when the teacher is done; on_end (after Settle) records the runtime instance statuses, the
    transitions, the teacher's phase log and, for support_insert, the true insertion geometry."""

    def __init__(self, teacher, callback=None, check_feasibility=True):
        self.teacher, self.callback, self.check = teacher, callback, check_feasibility
        self.feasibility, self.k = None, 0

    def on_reset(self, i, env, obs):
        from rrp.tasks.spec import Judgement
        if not self.check:
            return None
        self.feasibility = f = self.teacher.feasibility()
        if not f["feasible"]:
            return Judgement(True, "infeasible", f"infeasible:{','.join(f['unreachable'])}", False, False)
        return None

    def on_step(self, i, env, act, step):
        from rrp.tasks.spec import Judgement
        if self.callback is not None:
            self.callback(self.k, act.command, step)
        self.k += 1
        return Judgement(True, "success", "teacher_done") if self.teacher.done else None

    def on_end(self, i, env, ep):
        out = dict(feasibility=self.feasibility)
        if ep.outcome == "infeasible":
            return out
        out.update(statuses={e: (v.status, v.attempt, v.reason) for e, v in env.runtime.instances.items()},
                   transitions=list(env.runtime.transitions), phase_log=list(self.teacher.log),
                   phase_label=self.teacher.phase_label, truth={})
        if hasattr(env, "insertion_truth") and env.scenario.name == "support_insert":
            out["truth"] = env.insertion_truth()
        return out


def run_teacher_episode(task: str, session, teacher, version: str, seed: int, *, max_steps: int = 1200, callback=None,
                        check_feasibility: bool = True):
    """One scripted dual-teacher episode on `session` through harness.rollout (the former
    policies.teachers.dual.run_dual_teacher_episode): teacher.act() every control tick, `max_steps` ticks at most, then 5
    settle ticks and the privileged verdict (hooks.Settle). Returns the rollout Episode; a crash raises."""
    from rrp.harness import rollout as R
    from rrp.harness.eval import hooks as H
    from rrp.policies.teachers import TeacherPolicy
    pol = TeacherPolicy(task, lambda e: teacher, f"dual:{version}", ("joint_position", "gripper"))
    ep = R.rollout(lambda sd: session, pol, H.budget_task(task, session.spec.env_id), [seed], batch=1,
                   max_steps=max_steps, hooks=[H.Settle(5), _TeacherEpisode(teacher, callback, check_feasibility)])[0]
    if ep.outcome == "crash":                      # a crashed episode is an error row of the caller, never hidden
        raise RuntimeError(ep.metrics.get("note") or ep.failure_reason)
    return ep


def teacher_failure_reason(ep) -> str | None:
    """The teacher-episode failure vocabulary of the validation rows: `infeasible:<unreachable>` (not attempted),
    `ended_in_phase:<phase>` (attempted, the privileged verdict failed), None on success."""
    if ep.outcome == "infeasible":
        return ep.failure_reason
    return None if ep.success_privileged else f"ended_in_phase:{ep.metrics['phase_label']}"


def run_one(task: str, pair: str, seed: int, max_steps: int = 1200) -> dict:
    """One validation episode of the default (v2) dual teacher: a row that records success, failure, infeasibility or
    an error (as data), with the public-vs-privileged agreement and the insertion geometry."""
    from rrp.policies.teachers.dual import TEACHERS
    from rrp.policies.teachers.dual_validate import make_session
    t0 = time.time()
    try:
        s = make_session(task, pair, seed)
        teacher = TEACHERS[task](s)
        ep = run_teacher_episode(task, s, teacher, "v2", seed, max_steps=max_steps)
    except Exception as e:  # noqa: BLE001 - errors are recorded as data
        return dict(task=task, pair=pair, seed=seed, status="error", error=repr(e)[:400], wall_s=time.time() - t0)
    m = ep.metrics
    priv = bool(ep.success_privileged)
    status = "infeasible" if ep.outcome == "infeasible" else "success" if priv else "failure"
    statuses = m.get("statuses", {})
    return dict(task=task, pair=pair, seed=seed, status=status, source="scripted_teacher", privileged_teacher=True,
                public_runtime_success=bool(ep.success_public), privileged_success=priv,
                agree=bool(ep.success_public) == priv, failure_reason=teacher_failure_reason(ep),
                steps=ep.steps, sim_s=ep.steps * s.dt, statuses=statuses, truth=m.get("truth", {}),
                feasibility=m["feasibility"], rejected_commands=m["command_rejections"],
                retries={e: v[1] for e, v in statuses.items() if v[1] > 0},
                hole_frame_used=getattr(teacher, "hole_frame_used", None), wall_s=time.time() - t0)


def _validate_job(args):
    return run_one(*args)


def summarize_validation(rows: list[dict]) -> dict:
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


def validate_main(argv=None):
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
    with ProcessPoolExecutor(max_workers=a.workers) as ex, out.open("a") as fh:
        futs = [ex.submit(_validate_job, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            rows.append(r)
            fh.write(json.dumps(r, default=str) + "\n")
            fh.flush()
            if i % 20 == 0:
                print(f"[dual_validate] {i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    want = {p for p in a.pairs.split(",")}
    summ = summarize_validation([r for r in rows if r["pair"] in want and lo <= r["seed"] < hi])
    out.with_suffix(".summary.json").write_text(json.dumps(summ, indent=1, sort_keys=True))
    for k, d in sorted(summ.items()):
        print(k, {x: d[x] for x in ("n", "success", "failure", "infeasible", "error", "agree")}, d["reasons"])
