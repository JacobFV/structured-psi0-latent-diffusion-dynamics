"""Closed-loop evaluation of the corrected path (R38): system i emits LatentActionChunk every replan period;
system 0 realizes the received packet every control tick; packet-only probes are scored on the ACTUAL
noise-started packets against privileged/public labels at packet time (sampled semantics, test 5).
Also: disturbance test with the packet held fixed (test 7)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import mujoco
import numpy as np
import torch

from rrp.contracts.errors import ControllerRejection, StaleActionError
from rrp.control.latent_realizer import LatentSystem0
from rrp.data.collect import featurizer_for, privileged_labels
from rrp.learning.packed import active_operator, _focus
from rrp.model.latent_probes import probe_metrics


@dataclass
class LatentEpisode:
    robot: str
    seed: int
    method: str
    outcome: str
    privileged_success: bool
    public_success: bool
    steps: int
    system_i_calls: int
    system0_ticks: int
    packet_rejections: int
    fallback_holds: int
    sim_time: float
    wall_s: float
    events: dict
    probe_counts: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


def packet_labels(session, pi, M=1, S=None):
    feat = session._rrp_featurizer
    lab = privileged_labels(session, feat)
    S = S or lab["slot_pos"].shape[0]
    t = lambda x: torch.as_tensor(np.asarray(x))[None]
    return dict(held=t(lab["slot_held"][:S, 0]), contact=t(lab["slot_contact"][:S, 0]), visible=t(lab["slot_visible"][:S]),
                focus=t(_focus(pi, S)), gaze=t(lab["slot_gaze_angle"][:S]).float(),
                rel_tcp=t(lab["slot_rel_tcp"][:S, 0]).float(), future_disp=torch.zeros(1, S, 3),
                subtask=torch.tensor([active_operator(pi)]), goal_effect=_goal(pi)[:, :S])


def _goal(pi):
    from rrp.model.batch import collate_inputs
    from rrp.model.binding_aug import goal_effect_from_batch
    return goal_effect_from_batch(collate_inputs([pi]))


def paired_scene_fn(key: int):
    """Episode key = 10 * scene_seed + patient; n_objects = 2 + scene_seed % 2 (as in data generation)."""
    from rrp.sim.scenario import build_pick_place_paired
    sd, p = divmod(int(key), 10)

    def fn(robot, _key):
        sc = build_pick_place_paired(robot, sd, patient=p, n_objects=2 + sd % 2)
        return sc, dict(scene_seed=sd, patient=p, patient_color=sc.meta["cube_color"], sim_seed=sd,
                        patient_slot=sc.meta["patient_slot"], physical_names=None)
    return fn


def paired_keys(seed_start: int, n_scenes: int) -> list[int]:
    return [10 * sd + p for sd in range(seed_start, seed_start + n_scenes) for p in range(2 + sd % 2)]


def evaluate_latent(policy, realizer, probe, robot_key: str, seeds: list[int], *, method: str, replan_ticks: int = 8,
                    max_steps: int = 300, batch: int = 16, out_path: Path | None = None, task: str = "pick_place",
                    device="cpu", scene_fn=None) -> list[LatentEpisode]:
    """scene_fn(robot, seed) -> (Scenario, extra dict) overrides the default builder (e.g. paired binding scenes;
    `seed` is then an opaque episode key). Episodes record displacement of every non-assigned object."""
    from rrp.morphology.catalog import workbench_robots
    from rrp.sim.scenario import BUILDERS
    from rrp.sim.native import Session
    from rrp.control.teachers import PickPlaceTeacher
    robot = workbench_robots()[robot_key]()
    results = []
    for i in range(0, len(seeds), batch):
        group = seeds[i:i + batch]
        S, meta, s0 = [], [], []
        ex = []
        for sd in group:
            if scene_fn is not None:
                scen, e_ = scene_fn(robot, sd)
                s = Session(scen, seed=e_.get("sim_seed", sd))
            else:
                s, e_ = Session(BUILDERS[task](robot, sd, n_distractors=sd % 3), seed=sd), {}
            e_ = dict(e_, init={o.sim_body: s.data.xpos[s.model.body(o.sim_body).id].tolist()
                                for o in s.scenario.objects if o.kind == "object"})
            ex.append(e_)
            feas = PickPlaceTeacher(s).feasibility()["feasible"]
            f = policy.featurizer(s)
            S.append(s)
            s0.append(LatentSystem0(realizer, f, latent_space_version=policy.lsv, realizer_compat_version=policy.rcv,
                                    device=device))
            meta.append(dict(t0=time.time(), done=not feas, outcome=None if feas else "infeasible", steps=0,
                             calls=0, probes={}))
        for step in range(max_steps):
            act = [k for k, m in enumerate(meta) if not m["done"]]
            if not act:
                break
            need = [k for k in act if step % replan_ticks == 0 or s0[k].packet is None]
            if need:
                pk = policy.packets([S[k] for k in need])
                for k, p in zip(need, pk):
                    meta[k]["calls"] += 1
                    try:
                        s0[k].receive(p, now=float(S[k].data.time), graph_version=S[k].runtime.graph_version)
                    except (ControllerRejection, StaleActionError):
                        pass
                    if probe is not None:          # diagnostic only: never feeds back into control
                        with torch.no_grad():
                            pi = S[k]._rrp_featurizer(S[k].observe())
                            lab = packet_labels(S[k], pi)
                            z = torch.from_numpy(p.z)[None].to(device)
                            am = torch.tensor([p.assembly_mask], device=device)
                            Sn = lab["held"].shape[1]
                            out = probe(z, am, Sn)
                            lab = {kk: vv.to(device) for kk, vv in lab.items()}
                            for q, (x, n) in probe_metrics(out, lab, torch.ones(1, Sn, dtype=torch.bool, device=device)).items():
                                a_, b_ = meta[k]["probes"].get(q, (0, 0))
                                meta[k]["probes"][q] = (a_ + x, b_ + n)
            for k in act:
                s = S[k]
                cmd = s0[k].tick(s, s.controller_version())
                r = s.step(cmd)
                meta[k]["steps"] += 1
                if s.data.xpos[s.model.body("cube").id][2] < -0.05:
                    meta[k].update(done=True, outcome="failure")
                elif s.runtime.succeeded():
                    meta[k]["done"] = True
        for k, s in enumerate(S):
            m = meta[k]
            priv = bool(s.privileged_success()) if m["outcome"] != "infeasible" else False
            if m["outcome"] is None:
                m["outcome"] = "success" if priv else ("timeout" if m["steps"] >= max_steps else "failure")
            e_ = ex[k]
            disp = {b: float(np.linalg.norm(s.data.xpos[s.model.body(b).id] - np.array(p0))) for b, p0 in e_["init"].items()}
            e_ = dict(e_, final_disp_m=disp, moved=[b for b, d in disp.items() if d > 0.03])
            e_.pop("init")
            results.append(LatentEpisode(robot_key, group[k], method, m["outcome"], priv, bool(s.runtime.succeeded()),
                                         m["steps"], m["calls"], s0[k].stats.ticks, s0[k].stats.rejected,
                                         s0[k].stats.fallback_holds, float(s.data.time), time.time() - m["t0"],
                                         {e: v.status for e, v in s.runtime.instances.items()}, m["probes"], e_))
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a") as fh:
            for r in results:
                fh.write(json.dumps(asdict(r)) + "\n")
    return results


def _tcp(s):
    r = s.robots[0]
    site = r.tcp_sites[next(a.id for a in r.spec.assemblies if a.kind in ("gripper", "hand"))]
    return s.data.site_xpos[mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_SITE, site)].copy()


def disturbance_test(policy, realizer, robot_key: str, seeds: list[int], *, warmup_ticks=30, hold_ticks=8,
                     joint_offset=0.12, joint_index=1, device="cpu", session_hook=None) -> list[dict]:
    """Packet held FIXED (no system-i call). Compare TCP deviation from the undisturbed rollout for:
    A) system 0 closed loop (state-dependent realization), B) open-loop replay of the nominal commands expressed as
    deltas from the current state (no feedback), C) replay of the nominal ABSOLUTE targets (native servo
    stabilization only).
    session_hook(s): optional, called on each new session before anything else (e.g. the ladder installs its
    input featurizer there)."""
    from rrp.morphology.catalog import workbench_robots
    from rrp.sim.scenario import BUILDERS
    from rrp.sim.native import Session
    from rrp.contracts.action import NativeCommand
    robot = workbench_robots()[robot_key]()
    rows = []
    for sd in seeds:
        s = Session(BUILDERS["pick_place"](robot, sd, n_distractors=0), seed=sd)
        if session_hook is not None:
            session_hook(s)
        f = policy.featurizer(s)
        s0 = LatentSystem0(realizer, f, latent_space_version=policy.lsv, realizer_compat_version=policy.rcv, device=device)
        for k in range(warmup_ticks):             # normal operation to mid-approach
            if k % 8 == 0:
                s0.receive(policy.packets([s])[0], now=float(s.data.time), graph_version=s.runtime.graph_version)
            s.step(s0.tick(s, s.controller_version()))
        calls_before = policy.calls
        p = policy.packets([s])[0]
        s0.receive(p, now=float(s.data.time), graph_version=s.runtime.graph_version)
        snap = s.snapshot()
        arm_q = lambda: s.data.qpos[s.robots[0].qadr[:len(s.robots[0].arm_joints)]].copy()
        # nominal
        nominal_cmds, tcp_nom, q_nom = [], [], []
        for _ in range(hold_ticks):
            c = s0.tick(s, s.controller_version())
            nominal_cmds.append(c)
            q_nom.append(arm_q())
            s.step(c)
            tcp_nom.append(_tcp(s))

        def perturb():
            s.restore(snap)
            s0.receive(p, now=float(s.data.time), graph_version=s.runtime.graph_version)
            adr = s.robots[0].qadr[joint_index]
            s.data.qpos[adr] += joint_offset
            s.data.qvel[:] = 0
            mujoco.mj_forward(s.model, s.data)
        # A: closed-loop system 0
        perturb()
        tcp_a = []
        for _ in range(hold_ticks):
            s.step(s0.tick(s, s.controller_version()))
            tcp_a.append(_tcp(s))
        # B: open-loop deltas (predetermined trajectory relative to state at disturbance)
        perturb()
        tcp_b = []
        q_start = arm_q()
        for h, c in enumerate(nominal_cmds):
            g = dict(c.groups)
            g["arm"] = (np.array(c.groups["arm"]) - q_nom[0] + q_start).tolist()
            s.step(NativeCommand(controller_version=c.controller_version, groups=g, source="debug"))
            tcp_b.append(_tcp(s))
        # C: absolute nominal targets (servo stabilization)
        perturb()
        tcp_c = []
        for c in nominal_cmds:
            s.step(c)
            tcp_c.append(_tcp(s))
        dev_ = lambda tr: float(np.linalg.norm(tr[-1] - tcp_nom[-1]))
        rows.append(dict(robot=robot_key, seed=sd, system_i_calls_during_hold=policy.calls - calls_before - 1,
                         final_dev_closed_loop_m=dev_(tcp_a), final_dev_open_loop_delta_m=dev_(tcp_b),
                         final_dev_servo_absolute_m=dev_(tcp_c), hold_ticks=hold_ticks, joint_offset=joint_offset,
                         feedback_dt_s=s.dt))
    return rows
