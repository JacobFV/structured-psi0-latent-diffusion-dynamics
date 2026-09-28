"""Per-episode dual-arm motion/contact quality recorder (W12 / D-126 #18). PRIVILEGED, read-only; never a policy input.

`DualQualityRecorder(session, teacher)`:
  before_step(clean_cmds)  -- records the CLEAN (label) arm commands and the teacher phase switches of each arm
  after_step()             -- records measured arm joints, robot<->object penetration, arm<->arm contact, contact frames
  summary(task)            -- dict(family="dual", per_arm {jerk, phase-switch velocity step, joint margin, ...},
                              penetration_max_m, penetration_ticks_over_3mm, arm_arm_contact_ticks, contact {cf_* keys})
Used by rrp.evaluation.dual_teacher_quality (audit) and by rrp.data.collect_dual (`record_quality: true`, stored as the
episode meta's `motion`, which rrp.evaluation.gates.check_dual_dataset gates).
"""
from __future__ import annotations

import mujoco
import numpy as np

from rrp.data.contact_labels import ContactFrameRecorder
from rrp.data.contact_metrics import dual_contact_motion
from rrp.envs.motion_quality import chunk_boundary_steps, jerk_stats, joint_limit_margin

DUAL_MQ_VERSION = "rrp.data.dual_quality/v1"
PEN_TICK_M = 0.003


class DualQualityRecorder:
    def __init__(self, session, teacher=None):
        s = self.s = session
        self.teacher = teacher
        m = s.model
        self.ents = list(s.handles)
        self.qadr, self.lo, self.hi = {}, {}, {}
        for e, h in s.handles.items():
            jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in h.arm_joints]
            self.qadr[e] = np.array([m.jnt_qposadr[j] for j in jid])
            self.lo[e], self.hi[e] = m.jnt_range[jid, 0].copy(), m.jnt_range[jid, 1].copy()
        self.body_robot = {}
        for ri, r in enumerate(s.robots):
            names = {l.name for l in r.spec.links}
            for b in range(m.nbody):
                if m.body(b).name in names:
                    self.body_robot[b] = ri
        self.obj_bodies = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o.sim_body) for o in s.detectables} - {-1}
        self.cf = ContactFrameRecorder(s)
        self.q = {e: [] for e in self.ents}
        self.cmd = {e: [] for e in self.ents}
        self.last = {e: None for e in self.ents}
        self.switches = {e: [] for e in self.ents}
        self.prev_phase = dict(teacher.phase) if teacher is not None and hasattr(teacher, "phase") else None
        self.k = 0
        self.pen, self.pen_ticks, self.arm_arm = 0.0, 0, 0

    def before_step(self, cmds: dict | None):
        d = self.s.data
        if self.prev_phase is not None:
            for e in self.ents:
                if self.teacher.phase.get(e) != self.prev_phase.get(e):
                    self.switches[e].append(self.k)
            self.prev_phase = dict(self.teacher.phase)
        for e, h in self.s.handles.items():
            g = (cmds or {}).get(h.robot)
            g = g.groups if g is not None else {}
            if h.arm_group in g:
                self.last[e] = np.asarray(g[h.arm_group], float).copy()
            self.cmd[e].append(self.last[e] if self.last[e] is not None else d.qpos[self.qadr[e]].copy())
        self.cf.tick()

    def after_step(self):
        m, d = self.s.model, self.s.data
        self.k += 1
        for e in self.ents:
            self.q[e].append(d.qpos[self.qadr[e]].copy())
        pt, arms = 0.0, False
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            r1, r2 = self.body_robot.get(b1), self.body_robot.get(b2)
            if (r1 is not None and b2 in self.obj_bodies) or (r2 is not None and b1 in self.obj_bodies):
                pt = max(pt, -float(c.dist))
            if r1 is not None and r2 is not None and r1 != r2:
                arms = True
        self.pen = max(self.pen, pt)
        self.pen_ticks += int(pt > PEN_TICK_M)
        self.arm_arm += int(arms)

    def summary(self, task: str | None = None) -> dict:
        dt = float(self.s.dt)
        per = {}
        for e in self.ents:
            Q, C = np.array(self.q[e]), np.array(self.cmd[e])
            js, cj = jerk_stats(Q, dt), jerk_stats(C, dt)
            cb = chunk_boundary_steps(C, dt, self.switches[e])
            per[e] = dict(joint_jerk_rms=js["joint_jerk_rms"], joint_jerk_peak=js["joint_jerk_peak"],
                          cmd_jerk_rms=cj["joint_jerk_rms"], cmd_jerk_peak=cj["joint_jerk_peak"],
                          phase_switch_vel_step_max=cb["chunk_vel_step_max"], vel_step_any_max=cb["vel_step_any_max"],
                          n_phase_switches=len(self.switches[e]),
                          joint_limit_margin_min=joint_limit_margin(Q, self.lo[e], self.hi[e]) if len(Q) else None)
        rec = self.cf.recording()
        cfm = dual_contact_motion(rec, task=task, receipt_log=self.s.runtime.receipts.log) if rec.T >= 3 else {}
        from rrp.physics.grasp_contact import model_grasp_version
        return dict(version=DUAL_MQ_VERSION, family="dual", ticks=self.k, dt=dt, per_arm=per,
                    penetration_max_m=self.pen, penetration_ticks_over_3mm=self.pen_ticks,
                    arm_arm_contact_ticks=self.arm_arm, contact=cfm, phase_switch_ticks=self.switches,
                    grasp_contact_version=model_grasp_version(self.s.model) or "grasp_v1")
