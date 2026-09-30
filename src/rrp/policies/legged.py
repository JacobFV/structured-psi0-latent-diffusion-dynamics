"""Legged policies (D-140 S4): the latent packet stack (system i = LeggedFlow, system 0 = LeggedRealizer), the BC
positive control, and the privileged oracle packet source, as `rrp.policies.base.Policy`s on the `legs` action space
of `mujoco/legged` (control="legs"). Moved from rrp.harness.eval.legged_latent_eval, where they lived in the
session's tracker slot; `System0Adapter` / `BCAdapter` keep the exact per-tick logic (packet every TICKS_PER_PACKET
ticks, fallback hold, OOD / safety hooks) and are now driven by the policy instead of the session.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch

from rrp.core.latent_action import AssemblyHandle, LatentActionChunk, check_packet
from rrp.policies.bundles import load_rep
from rrp.policies.features.legged import (EVENTS, KNOT_TIMES, MAX_M, MAX_N, TICK_DT, TICKS_PER_PACKET, LeggedMorph,
                                          active_event, local_state, public_context)
from rrp.policies.nets.legged_latent import LeggedFlow, legged_probe, legged_probe_read, remap_legged_probe_state

REALIZER_COMPAT = "legged-rz-osc-v1"     # base of the system-0 compatibility ID (osc-v1 phase input)


def legged_bundle_versions(base_lsv: str, encoder_state: dict, realizer_state: dict) -> tuple[str, str]:
    """Fingerprinted compatibility IDs (as the arm bundle_versions, D-038): the latent space is defined by the
    encoder weights, system-0 compatibility additionally by the realizer weights, so a refit realizer
    (--realizer) gets a new ID and packets for the original are rejected instead of silently reinterpreted."""
    from rrp.core.provenance import weights_digest
    import re
    lsv = base_lsv if re.search(r"-w[0-9a-f]{12}$", base_lsv) else f"{base_lsv}-w{weights_digest(encoder_state)}"
    return lsv, f"{REALIZER_COMPAT}-{lsv}-r{weights_digest(realizer_state)}"


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
        self.ood = None                    # D-126 #29: rrp.controllers.packet_ood.PacketOODMonitor (None = off)
        self.fallback_mode = "hold_default"
        self.safety = None                 # the SafetyLayer, when the OOD fallback is the safe stop

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
        lt = getattr(self.c, "local_transform", None)       # D-126 #28 audit hook (None = unchanged)
        if lt is not None:
            q, qd, imu, touch = lt(q.copy(), qd.copy(), imu.copy(), touch.copy(), self)
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
                check_packet(p, latent_space_version=self.c.lsv, realizer_compat_version=self.c.rcv,
                             robot_spec_hash=self.m.spec_hash, now=now)
                if self.ood is not None:
                    self._ood_check(p, now)
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
            if self.fallback_mode == "hold_measured":
                tgt = np.clip(data.qpos[b.pol_qadr], b.lo, b.hi)
            else:
                tgt = np.clip(b.q0, b.lo, b.hi)          # declared fallback: hold the default stance
        fc, _ = b.contacts(data)
        self.c.trace.append(dict(t=now, pose=self.s.base_pose_truth().tolist(), contact=fc.astype(int).tolist()))
        if getattr(self.c, "recorder", None) is not None and self.armed:
            self.c.recorder(self, fc)
        self.ticks += 1
        return tgt


    def _ood_check(self, p, now):
        """D-126 #29: score the accepted-by-compatibility packet; in enforce mode an OOD packet is rejected like an
        incompatible one (counted, logged, previous packet kept until it expires, then the declared fallback)."""
        rec = self.ood.check(p.z, t=now)
        if rec is not None and rec["rejected"]:
            if self.fallback_mode == "safe_stop" and self.safety is not None:
                self.safety.safe_stop(now, reason="packet_ood")
            raise ValueError(f"packet_ood: score {rec['score']:.1f} > {self.ood.model.thresholds['enforce']:.1f}")


class BCController:
    """POSITIVE CONTROL: plain behaviour-cloning flow policy (no packet); replans a 40-tick native-target chunk
    every `replan` ticks from the same public inputs (context + local state) and executes it open-loop between."""

    def __init__(self, ckpt: Path, dev, nfe=8, replan=5, seed=0):
        from rrp.policies.nets.legged_bc import load_bc
        self.model, st = load_bc(ckpt, dev)
        self.dev, self.nfe, self.replan = dev, nfe, replan
        self.gen = torch.Generator(device=dev).manual_seed(seed)
        from rrp.core.provenance import source_label
        self.policy_version = source_label("bc", f"{Path(ckpt).parent.name}/{Path(ckpt).name}")   # was "learned:" (W3)
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
            cx = public_context(self.s, self.osc())
            if getattr(self.c, "ctx_transform", None) is not None:     # D-126 #28 audit hook (same as the latent route)
                cx = np.asarray(self.c.ctx_transform(cx.copy(), self), np.float32)
            bb["ctx"] = torch.from_numpy(cx)[None].to(self.c.dev)
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
        if getattr(self.c, "recorder", None) is not None and self.armed:
            self.c.recorder(self, fc)
        self.ticks += 1
        return tgt


class LatentLeggedController:
    def __init__(self, flow_ckpt: Path | None, dev, nfe=8, edit="none", t_edit=1.0, seed=0, posthoc_probe=None,
                 rep=None, realizer=None, zero_qd=False):
        st = torch.load(str(flow_ckpt), map_location=dev, weights_only=False) if flow_ckpt else None
        self.cfg = st["cfg"] if st else dict(representation=str(rep))
        rcfg, self.E, self.R, self.P, rres = load_rep(Path(rep or self.cfg["representation"]), dev)
        from rrp.policies.bundles import checkpoint_provenance
        self.checkpoint_provenance = dict(representation=checkpoint_provenance(
            torch.load(str(rep or self.cfg["representation"]), map_location="cpu", weights_only=False),
            rep or self.cfg["representation"]).model_dump(mode="json", include={"legacy", "weights", "notes"}))
        if st:
            self.checkpoint_provenance["flow"] = checkpoint_provenance(st, flow_ckpt).model_dump(
                mode="json", include={"legacy", "weights", "notes"})
        if realizer:                                   # refit system 0 (same encoder / latent space)
            rs = torch.load(str(realizer), map_location=dev, weights_only=False)
            self.checkpoint_provenance["realizer"] = checkpoint_provenance(rs, realizer).model_dump(
                mode="json", include={"legacy", "weights", "notes"})
            self.R.load_state_dict(rs["R"])
        self.zero_qd = zero_qd
        if posthoc_probe:                              # measurement probe for latent_nosem (frozen, detached z)
            pp = torch.load(posthoc_probe, map_location=dev, weights_only=False)
            self.P = legged_probe(dz=rcfg["latent"]["dz"]).to(dev)
            self.P.load_state_dict(remap_legged_probe_state(pp["state"])); self.P.eval()
        self.F = None
        if st:
            self.F = LeggedFlow(dz=rcfg["latent"]["dz"], D=self.cfg.get("width", 256), layers=self.cfg.get("layers", 4)).to(dev)
            self.F.load_state_dict(st["flow"]); self.F.eval()
        # fingerprinted IDs computed from the weights actually loaded (legacy checkpoints included)
        self.lsv, self.rcv = legged_bundle_versions(rres["latent_space_version"], self.E.state_dict(),
                                                    self.R.state_dict())
        self.policy_version = (f"learned:{Path(flow_ckpt).parent.name}/{Path(flow_ckpt).name}" if flow_ckpt else
                               f"rep:{Path(rep).parent.name}") + (f"+rz:{Path(realizer).parent.name}/{Path(realizer).name}"
                                                                    if realizer else "")
        self.bc = None                                 # stateless BC expert (oracle packets = E(BC chunk))
        self.dev, self.nfe, self.edit, self.t_edit = dev, nfe, edit, t_edit
        self.gen = torch.Generator(device=dev).manual_seed(seed)
        self.packets, self.trace = [], []
        self.ctx_transform = None          # D-126 #28 audit: f(ctx np[22], adapter) -> ctx (None = unchanged)
        self.record_z = None               # D-126 #29: list of (t, edit, z) when packets are recorded

    def bind(self, session, morph):
        self.static = static_batch(morph, self.dev)
        self.morph = morph
        self.packets, self.trace = [], []

    def _ctx(self, ad, mode=None):
        c = public_context(ad.s, ad.osc())
        if mode == "mirror_goal":
            c = c.copy()
            c[9] = -c[9]; c[13] = -c[13]                 # lateral body-frame waypoint estimates (a, b)
        if mode in ("mirror_active", "mirror_inactive"):   # task-context edit of ONE waypoint estimate
            c = c.copy()
            ev = active_event(ad.s.runtime)
            act = 9 if ev == 0 else 13
            idx = act if mode == "mirror_active" else (13 if act == 9 else 9)
            c[idx] = -c[idx]
        if mode == "halt":
            c = c.copy()
            c[16:20] = np.eye(4)[EVENTS.index("halt")]
        ct = getattr(self, "ctx_transform", None)
        if ct is not None:
            c = np.asarray(ct(c.copy(), ad), np.float32)
        return torch.from_numpy(c)[None].to(self.dev)

    def _sample(self, b):
        return self.F.sample(b, nfe=self.nfe, generator=self.gen)

    def generate(self, ad, now):
        b = ad.dyn_batch()
        edit = self.edit if now >= self.t_edit else "none"
        b["ctx"] = self._ctx(ad, edit if edit in ("mirror_goal", "halt", "mirror_active", "mirror_inactive") else None)
        if self.bc is not None:
            with torch.no_grad():
                z, _ = self.E(b, self.bc.sample(b, nfe=self.nfe, generator=self.gen)[..., :40])
        elif getattr(self, "oracle", None) is not None:
            with torch.no_grad():
                z, _ = self.E(b, self.oracle.demo(ad))
        else:
            z = self._sample(b)
        z_pre = z
        if edit.startswith("probe_yaw"):
            z = self.probe_edit(z, b, float(edit.split(":")[1]))
        elif edit in ("probe_goal_mirror", "probe_halt"):
            z = self.readout_edit(z, b, edit)
        elif edit.startswith("contact:"):              # contact:<leg>:<0 swing|1 stance> at every knot
            _, leg, v = edit.split(":")
            z = self.contact_edit(z, b, int(leg), float(v))
        elif edit.startswith("rand_norm:"):            # irrelevant-edit control: random direction, fixed |dz|
            gg = torch.Generator(device=z.device).manual_seed(int(now * 1000) + 7)
            am = b["asm_mask"][:, None, :, None].float()
            d = torch.randn(z.shape, generator=gg, device=z.device) * am
            z = z + d / d.norm() * float(edit.split(":")[1])
        if edit == "zero":
            z = torch.zeros_like(z)
        with torch.no_grad():
            b0 = dict(b); b0["ctx"] = self._ctx(ad)
            pout = legged_probe_read(self.P, z, b["asm_mask"], b["body_asm"])
        M = self.morph.M
        zz = z[0, :, :M].cpu().numpy().astype(np.float32)
        if getattr(self, "record_z", None) is not None:
            self.record_z.append((now, edit, zz.copy()))
        p = LatentActionChunk(latent_space_version=self.lsv, realizer_compat_version=self.rcv, z=zz,
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
                                 z_norm=float(np.linalg.norm(zz)),
                                 dz_norm=float((z - z_pre).norm()) if edit != "none" else 0.0))
        return p

    def readout_edit(self, z, b, kind, steps=60, lr=0.05):
        """probe_goal_mirror: z moved so the probe reads the goal mirrored laterally (gx, -gy), rest anchored.
        probe_halt: z moved so the probe reads subtask=halt and zero base displacement."""
        z0 = z.detach()
        with torch.no_grad():
            o0 = legged_probe_read(self.P, z0, b["asm_mask"], b["body_asm"])
            g0, d0 = o0["goal"][:, :2].clone(), o0["disp"][:, :3].clone()
        zz = z0.clone().requires_grad_(True)
        opt = torch.optim.Adam([zz], lr=lr)
        am = b["asm_mask"][:, None, :, None].float()
        for _ in range(steps):
            o = legged_probe_read(self.P, zz * am, b["asm_mask"], b["body_asm"])
            if kind == "probe_goal_mirror":
                loss = ((o["goal"][:, :2] - g0 * torch.tensor([1.0, -1.0], device=z.device)) ** 2).sum() * 4
            else:
                loss = torch.nn.functional.cross_entropy(o["subtask"], torch.full((z.shape[0],), 2, device=z.device)) + \
                    ((o["disp"][:, :3]) ** 2).sum()
            loss = loss + 0.01 * ((zz - z0) ** 2 * am).sum() / am.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        return (zz * am).detach()

    def contact_edit(self, z, b, leg, v, steps=60, lr=0.05):
        """Move z so the frozen probe reads leg `leg` in contact state v at every knot (other entries anchored)."""
        z0 = z.detach()
        with torch.no_grad():
            c0 = legged_probe_read(self.P, z0, b["asm_mask"], b["body_asm"])["contact"].clone()
        zz = z0.clone().requires_grad_(True)
        opt = torch.optim.Adam([zz], lr=lr)
        am = b["asm_mask"][:, None, :, None].float()
        tgt = torch.full_like(c0[:, :, leg], v)
        for _ in range(steps):
            c = legged_probe_read(self.P, zz * am, b["asm_mask"], b["body_asm"])["contact"]
            other = torch.ones_like(c); other[:, :, leg] = 0
            loss = torch.nn.functional.binary_cross_entropy_with_logits(c[:, :, leg], tgt, reduction="sum") + \
                ((c - c0) ** 2 * other).sum() * 0.1 + 0.01 * ((zz - z0) ** 2 * am).sum() / am.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        return (zz * am).detach()

    def probe_edit(self, z, b, yaw, steps=60, lr=0.05):
        """Move z so the frozen probe reads displacement yaw = `yaw` rad (x/y displacement readout kept at its
        current value); small L2 anchor to the sampled packet."""
        z0 = z.detach()
        with torch.no_grad():
            d0 = legged_probe_read(self.P, z0, b["asm_mask"], b["body_asm"])["disp"][:, :3].clone()
        tgt = d0.clone(); tgt[:, 2] = yaw
        zz = z0.clone().requires_grad_(True)
        opt = torch.optim.Adam([zz], lr=lr)
        am = b["asm_mask"][:, None, :, None].float()
        for _ in range(steps):
            out = legged_probe_read(self.P, zz * am, b["asm_mask"], b["body_asm"])
            loss = ((out["disp"][:, :3] - tgt) ** 2).sum() + 0.01 * ((zz - z0) ** 2 * am).sum() / am.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        return (zz * am).detach()


class OracleShadow:
    """DIAGNOSTIC (privileged, source=target_encoder_oracle): at each replan, roll the privileged waypoint teacher +
    frozen body tracker forward 0.8 s in a SHADOW copy of the physics state, encode those demonstrated targets with E
    and send the resulting packet. Isolates system 0 (realization) from system i (generation)."""

    def __init__(self, ctl, session, inner):
        import mujoco
        from rrp.envs.mujoco.legged_tracker import load_tracker
        self.c, self.s = ctl, session
        self.inner = load_tracker(session.body_key, session.binding, session.robots[0].meta, session.tracker_kind)
        self.mj = mujoco
        self.shadow = mujoco.MjData(session.model)

    def demo(self, ad):
        from rrp.envs.mujoco.legged_core import yaw_of
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


# ------------------------------------------------------------------ Policy adapters (rrp.policies.base) on the legs space
class _LeggedPolicy:
    """Drives a System0Adapter / BCAdapter per tick on a `mujoco/legged` env built with control="legs". One episode
    at a time (the controllers keep per-episode traces and one generator, as the sequential legged evaluation did).
    The adapter's tick counter starts after the env's reset settle, so gait phase and replan ticks are unchanged."""

    adapter_cls = None

    def __init__(self, ctl, info):
        self.ctl, self.info = ctl, info
        self.env = self.ad = None

    def _bind(self, s, morph):
        self.ctl.bind(s, morph)

    def reset(self, spec, task, seeds, *, envs=None):
        if len(envs) != 1:
            raise ValueError("legged policies run one episode at a time (rollout batch=1)")
        s = envs[0]
        if getattr(s, "control", None) != "legs":
            raise ValueError("legged policies need a mujoco/legged env built with control='legs'")
        morph = LeggedMorph(s.model, s.binding, s.scenario.robots[0].robot_spec.spec_hash)
        self._bind(s, morph)
        self.env, self.ad = s, self.adapter_cls(self.ctl, s, morph)
        self.ad.ticks = s.settle_ticks
        self.ad.armed = True

    def act(self, obs):
        from rrp.core.action import NativeCommand
        from rrp.policies.base import Act
        s, ad = self.env, self.ad
        n0 = ad.stats["packets"]
        tgt = ad.act(s.data, None)
        new = ad.stats["packets"] > n0
        cmd = NativeCommand(controller_version=s.controller_version(), groups={"legs": np.asarray(tgt, float).tolist()},
                            source=self.info.source)
        return {i: Act(cmd, packet=getattr(ad, "packet", None) if new else None) for i in obs}


class LeggedLatentPolicy(_LeggedPolicy):
    """System i (LeggedFlow; or oracle packets E(privileged teacher look-ahead) with oracle=True, or E(BC chunk) when
    ctl.bc is set) + system 0 (LeggedRealizer) on the legs space. Edits: ctl.edit / ctl.t_edit as before."""
    adapter_cls = System0Adapter

    def __init__(self, ctl: LatentLeggedController, *, oracle: bool = False, name: str = "legged_latent",
                 variant: str | None = None):
        from rrp.envs.base import LEGGED_FAMILIES
        from rrp.policies.base import PolicyInfo, Requirements
        self.oracle = oracle
        src = "oracle" if (oracle or ctl.bc is not None) else "learned"
        super().__init__(ctl, PolicyInfo(name, src, f"{ctl.policy_version}|{ctl.lsv}|{ctl.rcv}", Requirements(
            frozenset({"joint_position"}), groups=frozenset({"legs"}), observations=frozenset({"proprio", "task_graph"}),
            body_families=LEGGED_FAMILIES, privileged=oracle), variant))

    def _bind(self, s, morph):
        self.ctl.bind(s, morph)
        self.ctl.oracle = OracleShadow(self.ctl, s, None) if self.oracle else None


class LeggedBCPolicy(_LeggedPolicy):
    """BC positive control (no packet): a native-target chunk every ctl.replan ticks, open loop in between."""
    adapter_cls = BCAdapter

    def __init__(self, ctl: BCController, *, name: str = "legged_bc"):
        from rrp.envs.base import LEGGED_FAMILIES
        from rrp.policies.base import PolicyInfo, Requirements
        super().__init__(ctl, PolicyInfo(name, "bc", ctl.policy_version, Requirements(
            frozenset({"joint_position"}), groups=frozenset({"legs"}), observations=frozenset({"proprio", "task_graph"}),
            body_families=LEGGED_FAMILIES)))


def make_legged_latent(*, flow: str | None = None, rep: str | None = None, realizer: str | None = None,
                       device: str = "cpu", oracle: bool = False, **kw) -> LeggedLatentPolicy:
    return LeggedLatentPolicy(LatentLeggedController(Path(flow) if flow else None, torch.device(device), rep=rep,
                                                     realizer=realizer, **kw), oracle=oracle)


def make_legged_bc(*, checkpoint: str, device: str = "cpu", **kw) -> LeggedBCPolicy:
    return LeggedBCPolicy(BCController(Path(checkpoint), torch.device(device), **kw))
