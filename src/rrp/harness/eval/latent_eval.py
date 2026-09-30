"""Arm latent evaluation conventions for harness.rollout (the loop itself is rollout + policies.latent.LatentStackPolicy):
packet-only probes scored on the ACTUAL noise-started packets against privileged/public labels at packet time (sampled
semantics, test 5), system 0 counters, paired binding scenes; plus the disturbance test with the packet held fixed
(test 7)."""
from __future__ import annotations

import mujoco
import numpy as np
import torch

from rrp.policies.system0 import LatentSystem0
from rrp.harness.data.collect import privileged_labels
from rrp.harness.data.packed import _focus
from rrp.policies.features.derived import active_operator
from rrp.policies.nets.latent_probes import probe_metrics


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
    from rrp.policies.nets.batch import collate_inputs
    from rrp.policies.nets.latent_batch import goal_effect_from_batch
    return goal_effect_from_batch(collate_inputs([pi]))


def paired_env(robot_key: str):
    """make_env for paired binding scenes: episode key = 10 * scene_seed + patient; n_objects = 2 + scene_seed % 2
    (as in data generation); the session is seeded with scene_seed."""
    from rrp.bodies.catalog import workbench_robots
    from rrp.envs.mujoco.scenario import build_pick_place_paired
    from rrp.envs.mujoco.session import Session

    def make(key: int):
        sd, p = divmod(int(key), 10)
        sc = build_pick_place_paired(workbench_robots()[robot_key](), sd, patient=p, n_objects=2 + sd % 2)
        sc.meta.setdefault("body_key", robot_key)
        return Session(sc, seed=sd)
    return make


def paired_keys(seed_start: int, n_scenes: int) -> list[int]:
    return [10 * sd + p for sd in range(seed_start, seed_start + n_scenes) for p in range(2 + sd % 2)]


class PairedMeta:
    """on_end: the paired scene's identity (scene seed, patient, its colour and slot) for binding summaries."""

    def on_end(self, i, env, ep):
        m = env.scenario.meta
        return dict(scene_seed=m["pair_seed"], patient=m["patient"], patient_color=m["cube_color"],
                    patient_slot=m["patient_slot"])


class System0Stats:
    """on_end: the episode's system 0 counters (LatentStackPolicy.s0[i]): realized ticks, rejected packets, holds."""

    def __init__(self, policy):
        self.policy = policy

    def on_end(self, i, env, ep):
        st = self.policy.s0[i].stats
        return dict(system0_ticks=st.ticks, packet_rejections=st.rejected, fallback_holds=st.fallback_holds)


def acc_probe_counts(d: dict, res: dict) -> None:
    for q, (x, n) in res.items():
        a_, b_ = d.get(q, (0, 0))
        d[q] = (a_ + x, b_ + n)


class PacketProbeHook:
    """on_act: packet-only probes scored on each packet system i ACTUALLY emitted (Act.packet, after any edit), against
    labels read from the env at packet time (privileged labels: diagnostics only, never fed back into control).
    on_end: probe_counts {query: (correct or error sum, n)}."""

    def __init__(self, probe, device="cpu", labels=None):
        self.probe, self.device, self.labels = probe, device, labels or packet_labels
        self.envs, self.counts = {}, {}

    def on_reset(self, i, env, obs):
        self.envs[i], self.counts[i] = env, {}

    def on_act(self, i, obs, act):
        if act.packet is not None:
            self._score(i, act.packet)
        return act

    @torch.no_grad()
    def _score(self, i, p):
        env, dev = self.envs[i], self.device
        pi = env._rrp_featurizer(env.observe())
        lab = self.labels(env, pi)
        z = torch.from_numpy(p.z)[None].to(dev)
        am = torch.tensor([p.assembly_mask], device=dev)
        Sn = lab["held"].shape[1]
        lab = {k: v.to(dev) for k, v in lab.items()}
        acc_probe_counts(self.counts[i], probe_metrics(self.probe(z, am, Sn), lab,
                                                       torch.ones(1, Sn, dtype=torch.bool, device=dev)))

    def on_end(self, i, env, ep):
        return dict(probe_counts=self.counts.pop(i))


def probe_rates(episodes, key: str = "probe_counts") -> dict:
    tot: dict = {}
    for e in episodes:
        acc_probe_counts(tot, e.metrics.get(key, {}))
    return {q: (x / n if n else None) for q, (x, n) in tot.items()}


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
    from rrp.bodies.catalog import workbench_robots
    from rrp.envs.mujoco.scenario import BUILDERS
    from rrp.envs.mujoco.session import Session
    from rrp.core.action import NativeCommand
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
