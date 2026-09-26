"""Closed-loop evaluation of the legged latent-packet policy (system i flow -> LatentActionChunk -> system 0).

Controller sources (labelled in every output row):
  learned:<flow ckpt>      system i samples z from PUBLIC context every 0.4 s; system 0 realizes it every 20 ms
  scripted_teacher         privileged WaypointTeacher -> frozen body tracker (reference only)
Packet interventions (causal edits of the RECEIVED packet, everything else unchanged):
  none | mirror_goal (packet generated from the context with lateral waypoint estimates mirrored)
  | halt (packet generated from the context with the task view set to `halt`) | probe_yaw:+X / probe_yaw:-X
  (gradient edit of z so the frozen probe reads a desired yaw displacement X rad over the horizon)
  | freeze (the first packet is held for the whole episode, validity extended — declared diagnostic)
  | zero (z = 0).  Edits start at t_edit and are applied to every later packet.

The system-0 adapter replaces the body tracker inside LeggedSession (the native joint-target layer is unchanged);
held upper-body actuators remain servoed to the default pose. Truth (base pose, contacts) is logged per tick
for evaluation only.

usage: python -m rrp.evaluation.legged_latent_eval --flow artifacts/runs/X/policy.pt --bodies go2 --seeds 10000-10019
       --out artifacts/runs/X/eval_dev.jsonl [--edit mirror_goal --t-edit 1.0] [--video-dir artifacts/video --video-n 2]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from rrp.contracts.action import NativeCommand
from rrp.contracts.latent_action import LatentActionChunk, AssemblyHandle, check_packet
from rrp.control.legged_latent import (LeggedMorph, public_context, local_state, active_event, TICK_DT,
                                       TICKS_PER_PACKET, KNOT_TIMES, MAX_N, MAX_M, EVENTS)
from rrp.learning.legged_latent_train import load_rep, _dev
from rrp.model.legged_latent import LeggedFlow, LeggedProbe
from rrp.sim.legged import LeggedSession, build_waypoint_contact

REALIZER_COMPAT = "legged-rz-osc-v1"


def static_batch(morph: LeggedMorph, dev):
    N, M = morph.N, morph.M
    ns = np.zeros((MAX_N, morph.node_static.shape[1]), np.float32); ns[:N] = morph.node_static
    na = np.zeros(MAX_N, np.int64); na[:N] = morph.node_asm
    asm = np.zeros((MAX_M, morph.asm_static.shape[1]), np.float32); asm[:M] = morph.asm_static
    nm = np.zeros(MAX_N, bool); nm[:N] = True
    am = np.zeros(MAX_M, bool); am[:M] = True
    leg = np.zeros(MAX_M, bool); leg[:morph.nf] = True
    t = lambda x: torch.from_numpy(np.asarray(x))[None].to(dev)
    return dict(node_static=t(ns), node_asm=t(na), asm_static=t(asm), node_mask=t(nm), asm_mask=t(am),
                asm_is_leg=t(leg), body_asm=torch.tensor([morph.nf], device=dev))


class System0Adapter:
    """System 0 inside LeggedSession's native tick. Also hosts the system-i replan call (every 20 ticks)."""
    source = "learned"

    def __init__(self, ctl, session, morph):
        self.c, self.s, self.m = ctl, session, morph
        self.version = f"latent_system0:{ctl.lsv}"
        self.armed = False
        self.packet = None
        self.ticks = 0
        self.log = []
        self.stats = dict(ticks=0, packets=0, rejected=0, fallback=0)

    def reset(self, phase=0.0):
        self.ticks = 0
        self.packet = None
        self.armed = False

    def state(self):
        return dict(ticks=self.ticks)

    def load(self, st):
        self.ticks = st["ticks"]

    def osc(self):
        return (self.ticks * TICK_DT / self.m.gait_period) % 1.0

    def dyn_batch(self):
        q, qd, imu, touch = local_state(self.s, self.m)
        dev = self.c.dev
        qq = np.zeros(MAX_N, np.float32); qq[:len(q)] = q
        dd = np.zeros(MAX_N, np.float32); dd[:len(qd)] = qd
        tt = np.zeros(MAX_M, np.float32); tt[:len(touch)] = touch
        b = dict(self.c.static)
        t = lambda x: torch.from_numpy(np.asarray(x, np.float32))[None].to(dev)
        b.update(q=t(qq), qd=t(dd), imu=t(imu), asm_touch=t(tt), osc=torch.tensor([self.osc()], device=dev))
        return b

    def act(self, data, cmd):
        b = self.s.binding
        now = float(data.time)
        if self.armed and self.ticks % TICKS_PER_PACKET == 0 and not (self.c.edit == "freeze" and self.packet
                                                                       and now >= self.c.t_edit):
            try:
                p = self.c.generate(self, now)
                check_packet(p, latent_space_version=self.c.lsv, realizer_compat_version=REALIZER_COMPAT,
                             robot_spec_hash=self.m.spec_hash, now=now)
                self.packet = p
                self.stats["packets"] += 1
            except Exception as e:                      # rejected -> keep fallback semantics below
                self.stats["rejected"] += 1
                self.log.append(dict(t=now, event="packet_rejected", err=str(e)[:200]))
        tgt = None
        if self.packet is not None and (now <= self.packet.valid_until or self.c.edit == "freeze"):
            bb = self.dyn_batch()
            if self.c.zero_qd:
                bb["qd"] = torch.zeros_like(bb["qd"])
            z = torch.from_numpy(np.asarray(self.packet.z, np.float32))[None].to(self.c.dev)
            zp = torch.zeros(1, z.shape[1], MAX_M, z.shape[3], device=self.c.dev)
            zp[:, :, :z.shape[2]] = z
            ph = torch.tensor([now - self.packet.valid_from], device=self.c.dev)
            if self.c.edit == "freeze":
                ph = ph.clamp(max=KNOT_TIMES[-1])
            with torch.no_grad():
                a = self.c.R(zp, bb, ph)[0, :b.n].cpu().numpy().astype(np.float64)
            tgt = b.targets(np.clip(a, -5, 5))
            self.stats["ticks"] += 1
        else:
            self.stats["fallback"] += 1
            tgt = np.clip(b.q0, b.lo, b.hi)          # declared fallback: hold the default stance
        fc, _ = b.contacts(data)
        self.c.trace.append(dict(t=now, pose=self.s.base_pose_truth().tolist(), contact=fc.astype(int).tolist()))
        self.ticks += 1
        return tgt


class BCController:
    """POSITIVE CONTROL: plain behaviour-cloning flow policy (no packet); replans a 40-tick native-target chunk
    every `replan` ticks from the same public inputs (context + local state) and executes it open-loop between."""

    def __init__(self, ckpt: Path, dev, nfe=8, replan=5, seed=0):
        from rrp.learning.legged_bc import load_bc
        self.model, st = load_bc(ckpt, dev)
        self.dev, self.nfe, self.replan = dev, nfe, replan
        self.gen = torch.Generator(device=dev).manual_seed(seed)
        self.policy_version = f"learned:{Path(ckpt).parent.name}/{Path(ckpt).name}"
        self.edit, self.t_edit, self.trace, self.packets = "none", None, [], []

    def bind(self, session, morph):
        self.static = static_batch(morph, self.dev)
        self.morph = morph
        self.trace, self.packets = [], []


class BCAdapter:
    source = "learned"

    def __init__(self, ctl, session, morph):
        self.c, self.s, self.m = ctl, session, morph
        self.version = f"bc:{ctl.policy_version}"
        self.armed, self.ticks, self.chunk, self.k = False, 0, None, 0
        self.log = []
        self.stats = dict(ticks=0, packets=0, rejected=0, fallback=0)

    def reset(self, phase=0.0):
        self.ticks, self.chunk, self.armed = 0, None, False

    def state(self):
        return dict(ticks=self.ticks)

    def load(self, st):
        self.ticks = st["ticks"]

    osc = System0Adapter.osc
    dyn_batch = System0Adapter.dyn_batch

    def act(self, data, cmd):
        b = self.s.binding
        now = float(data.time)
        if self.armed and (self.chunk is None or self.k >= self.c.replan):
            bb = self.dyn_batch()
            bb["ctx"] = torch.from_numpy(public_context(self.s, self.osc()))[None].to(self.c.dev)
            self.chunk = self.c.model.sample(bb, nfe=self.c.nfe, generator=self.c.gen)[0, :b.n].cpu().numpy()
            self.k = 0
            self.stats["packets"] += 1
        if self.chunk is not None:
            tgt = b.targets(np.clip(self.chunk[:, self.k].astype(np.float64), -5, 5))
            self.k += 1
            self.stats["ticks"] += 1
        else:
            self.stats["fallback"] += 1
            tgt = np.clip(b.q0, b.lo, b.hi)
        fc, _ = b.contacts(data)
        self.c.trace.append(dict(t=now, pose=self.s.base_pose_truth().tolist(), contact=fc.astype(int).tolist()))
        self.ticks += 1
        return tgt


def failure_stage(row):
    """success | fell | stall (base moved < 0.3 m over the episode) | drift_a (never reached waypoint a) |
    drift_b (reached a, not b) | halt (reached b, halt not completed)."""
    if row["success"]:
        return "success"
    if row["fell"]:
        return "fell"
    tr = row.get("trace") or []
    path = sum(math.hypot(b["pose"][0] - a["pose"][0], b["pose"][1] - a["pose"][1]) for a, b in zip(tr, tr[1:]))
    ev = row["events"]
    if ev.get("walk_to_a") not in ("succeeded", "completed"):
        return "stall" if path < 0.3 else "drift_a"
    if ev.get("walk_to_b") not in ("succeeded", "completed"):
        return "drift_b"
    return "halt"


class LatentLeggedController:
    def __init__(self, flow_ckpt: Path | None, dev, nfe=8, edit="none", t_edit=1.0, seed=0, posthoc_probe=None,
                 rep=None, realizer=None, zero_qd=False):
        st = torch.load(str(flow_ckpt), map_location=dev, weights_only=False) if flow_ckpt else None
        self.cfg = st["cfg"] if st else dict(representation=str(rep))
        rcfg, self.E, self.R, self.P, rres = load_rep(Path(rep or self.cfg["representation"]), dev)
        if realizer:                                   # refit system 0 (same encoder / latent space)
            rs = torch.load(str(realizer), map_location=dev, weights_only=False)
            self.R.load_state_dict(rs["R"])
        self.zero_qd = zero_qd
        if posthoc_probe:                              # measurement probe for latent_nosem (frozen, detached z)
            pp = torch.load(posthoc_probe, map_location=dev, weights_only=False)
            self.P = LeggedProbe(dz=rcfg["latent"]["dz"]).to(dev)
            self.P.load_state_dict(pp["state"]); self.P.eval()
        self.F = None
        if st:
            self.F = LeggedFlow(dz=rcfg["latent"]["dz"], D=self.cfg.get("width", 256), layers=self.cfg.get("layers", 4)).to(dev)
            self.F.load_state_dict(st["flow"]); self.F.eval()
        self.lsv = rres["latent_space_version"]
        self.policy_version = (f"learned:{Path(flow_ckpt).parent.name}/{Path(flow_ckpt).name}" if flow_ckpt else
                               f"rep:{Path(rep).parent.name}") + (f"+rz:{Path(realizer).parent.name}/{Path(realizer).name}"
                                                                    if realizer else "")
        self.bc = None                                 # stateless BC expert (oracle packets = E(BC chunk))
        self.dev, self.nfe, self.edit, self.t_edit = dev, nfe, edit, t_edit
        self.gen = torch.Generator(device=dev).manual_seed(seed)
        self.packets, self.trace = [], []

    def bind(self, session, morph):
        self.static = static_batch(morph, self.dev)
        self.morph = morph
        self.packets, self.trace = [], []

    def _ctx(self, ad, mode=None):
        c = public_context(ad.s, ad.osc())
        if mode == "mirror_goal":
            c = c.copy()
            c[9] = -c[9]; c[13] = -c[13]                 # lateral body-frame waypoint estimates (a, b)
        if mode == "halt":
            c = c.copy()
            c[16:20] = np.eye(4)[EVENTS.index("halt")]
        return torch.from_numpy(c)[None].to(self.dev)

    def _sample(self, b):
        return self.F.sample(b, nfe=self.nfe, generator=self.gen)

    def generate(self, ad, now):
        b = ad.dyn_batch()
        edit = self.edit if now >= self.t_edit else "none"
        b["ctx"] = self._ctx(ad, edit if edit in ("mirror_goal", "halt") else None)
        if self.bc is not None:
            with torch.no_grad():
                z, _ = self.E(b, self.bc.sample(b, nfe=self.nfe, generator=self.gen)[..., :40])
        elif getattr(self, "oracle", None) is not None:
            with torch.no_grad():
                z, _ = self.E(b, self.oracle.demo(ad))
        else:
            z = self._sample(b)
        if edit.startswith("probe_yaw"):
            z = self.probe_edit(z, b, float(edit.split(":")[1]))
        if edit == "zero":
            z = torch.zeros_like(z)
        with torch.no_grad():
            b0 = dict(b); b0["ctx"] = self._ctx(ad)
            pout = self.P(z, b["asm_mask"], b["body_asm"])
        M = self.morph.M
        zz = z[0, :, :M].cpu().numpy().astype(np.float32)
        p = LatentActionChunk(latent_space_version=self.lsv, realizer_compat_version=REALIZER_COMPAT, z=zz,
                              knot_times=list(KNOT_TIMES),
                              assemblies=[AssemblyHandle(handle=h, robot_index=0) for h in self.morph.handles],
                              assembly_mask=[True] * M, observation_id=f"lg{ad.ticks}",
                              graph_version=int(ad.s.runtime.graph_version), runtime_version=int(ad.s.runtime.runtime_version),
                              robot_spec_hash=self.morph.spec_hash, generated_at=time.time(), valid_from=now,
                              valid_until=now + 0.6, source="target_encoder_oracle" if (getattr(self, "oracle", None) is not None
                                                                                  or self.bc is not None) else "learned",
                              policy_version=self.policy_version,
                              sampling=dict(nfe=self.nfe, sampler="euler_rectified_flow", edit=edit))
        self.packets.append(dict(t=now, ev=active_event(ad.s.runtime), edit=edit,
                                 probe=dict(contact=(pout["contact"][0, :, :M] > 0).int().tolist(),
                                            goal=pout["goal"][0, :2].tolist(), disp=pout["disp"][0, :3].tolist(),
                                            subtask=int(pout["subtask"][0].argmax()),
                                            fall=float(torch.sigmoid(pout["fall"][0, 0]))),
                                 pose=ad.s.base_pose_truth().tolist(),
                                 z_norm=float(np.linalg.norm(zz))))
        return p

    def probe_edit(self, z, b, yaw, steps=60, lr=0.05):
        """Move z so the frozen probe reads displacement yaw = `yaw` rad (x/y displacement readout kept at its
        current value); small L2 anchor to the sampled packet."""
        z0 = z.detach()
        with torch.no_grad():
            d0 = self.P(z0, b["asm_mask"], b["body_asm"])["disp"][:, :3].clone()
        tgt = d0.clone(); tgt[:, 2] = yaw
        zz = z0.clone().requires_grad_(True)
        opt = torch.optim.Adam([zz], lr=lr)
        am = b["asm_mask"][:, None, :, None].float()
        for _ in range(steps):
            out = self.P(zz * am, b["asm_mask"], b["body_asm"])
            loss = ((out["disp"][:, :3] - tgt) ** 2).sum() + 0.01 * ((zz - z0) ** 2 * am).sum() / am.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        return (zz * am).detach()


class OracleShadow:
    """DIAGNOSTIC (privileged, source=target_encoder_oracle): at each replan, roll the privileged waypoint teacher +
    frozen body tracker forward 0.8 s in a SHADOW copy of the physics state, encode those demonstrated targets with E
    and send the resulting packet. Isolates system 0 (realization) from system i (generation)."""

    def __init__(self, ctl, session, inner):
        import mujoco
        from rrp.control.legged_tracker import load_tracker
        self.c, self.s = ctl, session
        self.inner = load_tracker(session.body_key, session.binding, session.robots[0].meta, session.tracker_kind)
        self.mj = mujoco
        self.shadow = mujoco.MjData(session.model)

    def demo(self, ad):
        from rrp.control.legged_core import yaw_of
        mj, s, b = self.mj, self.s, self.s.binding
        mj.mj_copyData(self.shadow, s.model, s.data)
        d = self.shadow
        tr = self.inner
        if hasattr(tr, "last_a"):
            tr.last_a = np.clip((d.ctrl[b.pol_act] - b.q0) / b.action_scale, -5, 5)
        tr.phase = ad.osc()
        ev = active_event(s.runtime)
        L = s.robots[0].meta["legged"]["command_ranges"]
        wps = {o.task_entity: o.sim_body for o in s.scenario.objects}
        acts = []
        n = max(1, int(round(1.0 / (50.0 * s.model.opt.timestep))))
        for k in range(40):
            if k % 5 == 0:
                if ev >= 2:
                    cmd = np.zeros(3)
                else:
                    q = d.qpos
                    x, y, yaw = q[b.qa], q[b.qa + 1], yaw_of(q[b.qa + 3:b.qa + 7])
                    bid = mj.mj_name2id(s.model, mj.mjtObj.mjOBJ_BODY, wps["waypoint_a" if ev == 0 else "waypoint_b"])
                    tx, ty = d.xpos[bid][:2]
                    err = (math.atan2(ty - y, tx - x) - yaw + math.pi) % (2 * math.pi) - math.pi
                    dist = math.hypot(tx - x, ty - y)
                    vmax, wmax = 0.6 * L["vx"][1], 0.8 * L["wz"][1]
                    wz = float(np.clip(1.5 * err, -wmax, wmax))
                    vx = 0.0 if abs(err) > 1.0 else vmax * max(0.0, math.cos(err)) ** 2 * min(1.0, dist / 0.6 + 0.3)
                    cmd = np.array([vx, 0.0, wz])
            tgt = tr.act(d, cmd)
            acts.append((tgt - b.q0) / b.action_scale)
            d.ctrl[b.pol_act] = tgt
            if len(b.held_act):
                d.ctrl[b.held_act] = b.q0_held
            for _ in range(n):
                mj.mj_step(s.model, d)
        a = np.zeros((MAX_N, 40), np.float32)
        a[:b.n] = np.array(acts, np.float32).T
        return torch.from_numpy(a)[None].to(self.c.dev)


def run_episode(ctl, body, seed, max_s=60.0, video=None, oracle=False, scenario=None):
    sc = scenario if scenario is not None else build_waypoint_contact(body, seed)
    s = LeggedSession(sc, tracker_kind="cpg" if body not in ("go2", "t1", "g1", "anymal_c", "h1") else "auto", seed=seed)
    morph = LeggedMorph(s.model, s.binding, sc.robots[0].robot_spec.spec_hash)
    is_bc = isinstance(ctl, BCController)
    ad = (BCAdapter if is_bc else System0Adapter)(ctl, s, morph) if ctl is not None else None
    teacher = None
    if ctl is not None:
        ctl.bind(s, morph)
        if not is_bc:
            ctl.oracle = OracleShadow(ctl, s, s.tracker) if oracle else None
        s.tracker = ad
    s.reset()
    if ctl is None:
        from rrp.control.legged_teachers import WaypointTeacher
        teacher = WaypointTeacher(s)
    else:
        ad.armed = True
    frames = []
    rend = None
    if video is not None:
        import mujoco
        rend = mujoco.Renderer(s.model, 368, 480)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = s.binding.root_bid
        cam.distance, cam.elevation, cam.azimuth = 3.0 * max(0.5, s.binding.nominal_height() / 0.35) ** 0.5, -25, 135
    zero = NativeCommand(controller_version=s.controller_version(), groups={"base_velocity": [0.0, 0.0, 0.0]},
                         source="learned")
    t0 = time.time()
    steps = 0
    while s.data.time < max_s:
        cmd = teacher.act() if teacher else zero
        s.step(cmd)
        steps += 1
        if rend is not None and steps % 2 == 0:
            rend.update_scene(s.data, camera=cam)
            st = " ".join(f"{e}:{v.status}" for e, v in s.runtime.instances.items())
            frames.append((rend.render().copy(), f"t={s.data.time:.1f}s {st}"))
        if s.fell or s.runtime.succeeded() or (teacher is not None and teacher.done):
            break
    ok = bool(s.privileged_success() and not s.fell)
    src = "scripted_teacher" if ctl is None else (
        f"privileged_oracle_packet:{ctl.lsv}" if oracle else (
            f"oracle_diagnostic:E(bc_chunk)|{ctl.policy_version}|bc={ctl.bc_version}" if getattr(ctl, "bc", None) is not None
            else ctl.policy_version))
    row = dict(body=body, seed=seed, source=src,
               edit=(ctl.edit if ctl else "none"), t_edit=(ctl.t_edit if ctl else None), success=ok, fell=bool(s.fell),
               public_success=bool(s.runtime.succeeded()), sim_time=float(s.data.time), wall_s=time.time() - t0,
               events={e: v.status for e, v in s.runtime.instances.items()},
               final_pose=s.base_pose_truth().tolist(), waypoints=sc.meta["waypoints"])
    if ctl is not None:
        row.update(stats=ad.stats, packets=ctl.packets, packet_log=ad.log[:20])
        row["trace"] = ctl.trace[::5]
        if getattr(ctl, "zero_qd", False):
            row["zero_qd"] = True
    row["failure_stage"] = failure_stage(row)
    return row, frames


def packet_probe_accuracy(rows):
    """Closed-loop packet probes vs truth (generated packets, privileged truth used only for scoring)."""
    agg = dict(subtask=[0, 0], contact=[0, 0], goal_err=[0.0, 0], disp_yaw_err=[0.0, 0], disp_xy_err=[0.0, 0])
    for r in rows:
        tr = r.get("trace") or []
        if not tr:
            continue
        ts = np.array([x["t"] for x in tr])
        for p in r["packets"]:
            if p["edit"] != "none":
                continue
            agg["subtask"][0] += int(p["probe"]["subtask"] == min(p["ev"], 3)); agg["subtask"][1] += 1
            x, y, yaw = p["pose"]
            wp = r["waypoints"]["a" if p["ev"] == 0 else "b"]
            c, s_ = math.cos(yaw), math.sin(yaw)
            gx, gy = (c * (wp[0] - x) + s_ * (wp[1] - y)) / 2, (-s_ * (wp[0] - x) + c * (wp[1] - y)) / 2
            if p["ev"] < 3:
                agg["goal_err"][0] += 2 * math.hypot(p["probe"]["goal"][0] - gx, p["probe"]["goal"][1] - gy)
                agg["goal_err"][1] += 1
            k2 = int(np.searchsorted(ts, p["t"] + 0.8))
            if k2 < len(tr):
                x2, y2, yaw2 = tr[k2]["pose"]
                ex, ey = (c * (x2 - x) + s_ * (y2 - y)), (-s_ * (x2 - x) + c * (y2 - y))
                dyaw = (yaw2 - yaw + math.pi) % (2 * math.pi) - math.pi
                agg["disp_yaw_err"][0] += abs(p["probe"]["disp"][2] - dyaw); agg["disp_yaw_err"][1] += 1
                agg["disp_xy_err"][0] += math.hypot(p["probe"]["disp"][0] * 0.5 - ex, p["probe"]["disp"][1] * 0.5 - ey)
                agg["disp_xy_err"][1] += 1
            for kk, kt in enumerate(KNOT_TIMES):
                k3 = int(np.searchsorted(ts, p["t"] + kt))
                if k3 < len(tr):
                    for m, c_ in enumerate(tr[k3]["contact"]):
                        agg["contact"][0] += int(p["probe"]["contact"][kk][m] == c_); agg["contact"][1] += 1
    return {k: (v[0] / v[1] if v[1] else None) for k, v in agg.items()}


def _caption(img, lines):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 14 * len(lines) + 6], fill=(0, 0, 0))
    for i, t in enumerate(lines):
        d.text((6, 3 + 14 * i), t, fill=(255, 255, 255))
    return np.asarray(im)


def save_video(frames, row, video_dir: Path, label: str):
    import imageio
    video_dir.mkdir(parents=True, exist_ok=True)
    tag = "success" if row["success"] else ("fell" if row["fell"] else "failure")
    src = row["source"].replace(":", "-").replace("/", "_")
    ed = "" if row["edit"] == "none" else f"_edit-{row['edit'].replace(':', '')}"
    name = f"{dt.date.today()}_legged_{src}{ed}_{row['body']}_waypoint_contact_s{row['seed']}_{tag}.mp4"
    imgs = [_caption(f, [f"{label} | {row['body']} | waypoint_contact | seed {row['seed']}" + (
        f" | EDIT {row['edit']}@{row['t_edit']}s" if row['edit'] != 'none' else ""), st]) for f, st in frames]
    imageio.mimsave(video_dir / name, imgs, fps=25, quality=6)
    with open(video_dir / "INDEX.md", "a") as f:
        f.write(f"- `{name}` — source={row['source']} robot={row['body']} task=waypoint_contact seed={row['seed']} "
                f"edit={row['edit']} outcome={tag} (privileged evaluator)\n")
    return name


def _seeds(spec):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", default=None, help="flow policy.pt; omit for the scripted_teacher reference")
    ap.add_argument("--bodies", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--edit", default="none")
    ap.add_argument("--t-edit", type=float, default=1.0)
    ap.add_argument("--nfe", type=int, default=8)
    ap.add_argument("--max-s", type=float, default=60.0)
    ap.add_argument("--posthoc-probe", default=None)
    ap.add_argument("--oracle", action="store_true", help="DIAGNOSTIC: E-encoded shadow teacher rollouts as packets")
    ap.add_argument("--bc", default=None, help="BC policy.pt: alone = BC route (positive control); with --rep or "
                    "--flow and --oracle-bc = stateless oracle packets E(BC chunk) (DIAGNOSTIC)")
    ap.add_argument("--oracle-bc", action="store_true")
    ap.add_argument("--rep", default=None, help="representation.pt (oracle routes without a flow)")
    ap.add_argument("--realizer", default=None, help="refit system-0 state (R) to use instead of the rep's R")
    ap.add_argument("--zero-qd", action="store_true", help="DIAGNOSTIC: system 0 sees qd = 0")
    ap.add_argument("--replan", type=int, default=5, help="BC replan period (ticks)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--video-n", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu (default; eval runs in CPU leases) or cuda")
    a = ap.parse_args(argv)
    dev = _dev() if a.device == "cuda" else torch.device("cpu")
    torch.set_num_threads(2)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out, "a") as f:
        for body in a.bodies.split(","):
            nv = 0
            for sd in _seeds(a.seeds):
                if a.bc and not a.oracle_bc:
                    ctl = BCController(Path(a.bc), dev, nfe=a.nfe, replan=a.replan, seed=sd)
                elif a.flow or a.rep:
                    ctl = LatentLeggedController(Path(a.flow) if a.flow else None, dev, nfe=a.nfe, edit=a.edit,
                                                 t_edit=a.t_edit, seed=sd, posthoc_probe=a.posthoc_probe, rep=a.rep,
                                                 realizer=a.realizer, zero_qd=a.zero_qd)
                    if a.oracle_bc:
                        from rrp.learning.legged_bc import load_bc
                        ctl.bc, _ = load_bc(a.bc, dev)
                        ctl.bc_version = f"{Path(a.bc).parent.name}/{Path(a.bc).name}"
                else:
                    ctl = None
                want = a.video_dir is not None and nv < a.video_n
                row, frames = run_episode(ctl, body, sd, a.max_s, video=True if want else None, oracle=a.oracle)
                if want:
                    lab = ("SCRIPTED TEACHER (privileged)" if ctl is None else (
                        "PRIVILEGED ORACLE packets (E on shadow teacher) + LEARNED sys-0" if a.oracle
                        else f"LEARNED BC (no packet; positive control) {row['source']}" if isinstance(ctl, BCController)
                        else f"ORACLE DIAGNOSTIC packets E(BC chunk) + LEARNED sys-0" if a.oracle_bc
                        else f"LEARNED latent sys-i+sys-0 {row['source']}"))
                    row["video"] = save_video(frames, row, Path(a.video_dir), lab)
                    nv += 1
                rows.append(row)
                f.write(json.dumps(row) + "\n"); f.flush()
                print(json.dumps({k: row[k] for k in ("body", "seed", "source", "edit", "success", "fell", "sim_time",
                                                      "failure_stage")}),
                      flush=True)
    summ = {}
    for body in a.bodies.split(","):
        rs = [r for r in rows if r["body"] == body]
        stages = {}
        for r in rs:
            stages[r["failure_stage"]] = stages.get(r["failure_stage"], 0) + 1
        summ[body] = dict(n=len(rs), success=sum(r["success"] for r in rs), fell=sum(r["fell"] for r in rs),
                          stages=stages, seeds=a.seeds,
                          mean_sim_time=float(np.mean([r["sim_time"] for r in rs])),
                          packet_probes=packet_probe_accuracy(rs) if (a.flow or a.rep) else None)
    out.with_suffix(".summary.json").write_text(json.dumps(dict(source=rows[0]["source"], edit=a.edit, per_body=summ),
                                                           indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
