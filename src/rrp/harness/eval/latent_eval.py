"""Arm latent evaluation conventions for harness.rollout (the loop itself is rollout + policies.latent.LatentStackPolicy):
packet-only probes scored on the ACTUAL noise-started packets against privileged/public labels at packet time (sampled
semantics, test 5), system 0 counters, paired binding scenes; plus the disturbance test with the packet held fixed
(test 7).

`readout_loss` / `readout_metrics` (below): D-144 R1 follow-up (archived research/tracks/rel-r1c.md) ported here, unchanged,
from the deleted `nets.latent_probes.probe_loss` / `probe_metrics` -- the arm/dual `nets.probes.ReadoutProbe` output
dict has the identical shape/keys `PacketProbe` had (foundation equivalence, docs/relations.md 10 R1), so this math
still applies verbatim. NOT the same function as the generic `nets.probes.readout_loss` / `readout_metrics`
(spec-driven, one (sum, n) pair per query, `<query>_acc` / `<query>_mae` key scheme): those cannot reproduce arm's
established key names (`visible`, `rel_pos_err_m`, `desired_delta_err_m`, ...), the positives-balanced `*_pos`
views, the goal-effect patient/zero-baseline breakdown or the dual `held_m`/`contact_m` per-slot (`@m`) addressing
that every existing hook, dashboard and threshold already depends on -- "metric key names reported by hooks must
not change" (deferred-scope instructions) rules out the generic keys for this family. Every caller that used to
import `probe_loss`/`probe_metrics` from `nets.latent_probes` now imports `readout_loss`/`readout_metrics` from
here instead."""
from __future__ import annotations

import mujoco
import numpy as np
import torch
import torch.nn.functional as F

from rrp.core.compute import f32
from rrp.harness import hooks as H
from rrp.policies.system0 import LatentSystem0
from rrp.harness.data.collect import privileged_labels
from rrp.harness.data.packed import _focus
from rrp.policies.features.derived import active_operator
from rrp.policies.nets.probes import gaussian_nll  # identical formula (d=3 either way); shared with the foundation

ENTITY_QUERIES = ("visible", "looking_at", "focused_on")


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


# ------------------------------------------------------------------ readout_loss / readout_metrics (arm / dual)
# Ported unchanged from the deleted `nets.latent_probes.probe_loss` / `probe_metrics` (+ `_multi` variants);
# see this file's module docstring for why these stay a dedicated arm/dual implementation rather than the generic
# `nets.probes.readout_loss` / `readout_metrics`.
def _goal_terms(out, lab, smask):
    """Goal-effect loss/metric only when both the probe head and the label exist."""
    return "goal_effect" in out and "goal_effect" in lab


@f32
def readout_loss(out: dict, lab: dict, smask: torch.Tensor, m0: int = 0, lv_min: float = -8.0) -> tuple[torch.Tensor, dict]:
    """lab: held/contact/visible/focus [B,S] (manipulator 0 for held/contact/rel), rel_tcp/future_disp [B,S,3],
    gaze [B,S], subtask [B]. Positions scaled to decimeters for conditioning.
    Multi-assembly labels (held_m/contact_m [B,S,M], rel_tcp_m [B,S,M,3], subtask_m [B,M], packet slot order)
    switch the manipulator-indexed queries to ALL packet slots."""
    if "held_m" in lab:
        return _readout_loss_multi(out, lab, smask, lv_min=lv_min)
    m = smask.float()
    den = m.sum().clamp(min=1)
    bce = lambda logit, y: (F.binary_cross_entropy_with_logits(logit.squeeze(-1), y.float(), reduction="none")
                            * m).sum() / den
    L = dict(
        visible=bce(out["visible"], lab["visible"]),
        focused_on=bce(out["focused_on"], lab["focus"]),
        held_by=bce(out["held_by"][:, :, m0], lab["held"]),
        acting_on=bce(out["acting_on"][:, :, m0], lab["contact"]),
        looking_at=((out["looking_at"].squeeze(-1) - lab["gaze"] / 30).pow(2) * m).sum() / den,
        rel_pos=gaussian_nll(out["rel_pos"][:, :, m0], lab["rel_tcp"] * 10, smask, lv_min),
        observed_effect=gaussian_nll(out["observed_effect"], lab["future_disp"] * 10, smask, lv_min),
        subtask=F.cross_entropy(out["subtask"][:, m0], lab["subtask"].long()),
    )
    if _goal_terms(out, lab, smask):
        L["goal_effect"] = gaussian_nll(out["goal_effect"], lab["goal_effect"] * 10, smask, lv_min)
    total = sum(L.values())
    return total, {f"probe_{k}": float(v.detach()) for k, v in L.items()}


@torch.no_grad()
def readout_metrics(out: dict, lab: dict, smask: torch.Tensor, m0: int = 0) -> dict:
    """Accuracy / error metrics (raw sums for aggregation)."""
    if "held_m" in lab:
        return _readout_metrics_multi(out, lab, smask)
    m = smask.bool()
    res = {}
    for q, key in (("visible", "visible"), ("focused_on", "focus"), ("held_by", "held"), ("acting_on", "contact")):
        logit = out[q][..., 0] if q in ENTITY_QUERIES else out[q][:, :, m0, 0]
        pred = logit > 0
        y = lab[key].bool()
        res[q] = (int(((pred == y) & m).sum()), int(m.sum()))
        pos = y & m
        res[q + "_pos"] = (int(((pred == y) & pos).sum()), int(pos.sum()))   # balanced view on rare positives
    err = (out["rel_pos"][:, :, m0, :3] / 10 - lab["rel_tcp"]).norm(dim=-1)
    res["rel_pos_err_m"] = (float((err * m).sum()), int(m.sum()))
    derr = (out["observed_effect"][..., :3] / 10 - lab["future_disp"]).norm(dim=-1)
    res["observed_effect_err_m"] = (float((derr * m).sum()), int(m.sum()))
    res["desired_delta_err_m"] = res["observed_effect_err_m"]          # deprecated alias (same quantity)
    res["subtask"] = (int((out["subtask"][:, m0].argmax(-1) == lab["subtask"].long()).sum()), int(len(lab["subtask"])))
    if _goal_terms(out, lab, smask):
        res.update(goal_metrics(out, lab, m))
    return res


@torch.no_grad()
def goal_metrics(out, lab, m) -> dict:
    """goal_effect error on all slots and on goal-bearing slots (bound patient), plus a zero-prediction baseline."""
    g = lab["goal_effect"]
    err = (out["goal_effect"][..., :3] / 10 - g).norm(dim=-1)
    gp = (g.norm(dim=-1) > 1e-6) & m
    return dict(goal_effect_err_m=(float((err * m).sum()), int(m.sum())),
                goal_effect_err_patient_m=(float((err * gp).sum()), int(gp.sum())),
                goal_effect_zero_baseline_patient_m=(float((g.norm(dim=-1) * gp).sum()), int(gp.sum())))


def _readout_loss_multi(out: dict, lab: dict, smask: torch.Tensor, lv_min: float = -8.0) -> tuple[torch.Tensor, dict]:
    """Every packet slot m answers its own held_by/acting_on/rel_pos/subtask queries (role-addressed)."""
    m = smask.float()
    den = m.sum().clamp(min=1)
    M = lab["held_m"].shape[-1]
    mm = m[:, :, None].expand(-1, -1, M)
    denm = mm.sum().clamp(min=1)
    bce = lambda logit, y, w, d: (F.binary_cross_entropy_with_logits(logit, y.float(), reduction="none") * w).sum() / d
    L = dict(
        visible=bce(out["visible"].squeeze(-1), lab["visible"], m, den),
        focused_on=bce(out["focused_on"].squeeze(-1), lab["focus"], m, den),
        held_by=bce(out["held_by"][:, :, :M, 0], lab["held_m"], mm, denm),
        acting_on=bce(out["acting_on"][:, :, :M, 0], lab["contact_m"], mm, denm),
        looking_at=((out["looking_at"].squeeze(-1) - lab["gaze"] / 30).pow(2) * m).sum() / den,
        rel_pos=gaussian_nll(out["rel_pos"][:, :, :M], lab["rel_tcp_m"] * 10, mm.bool(), lv_min),
        observed_effect=gaussian_nll(out["observed_effect"], lab["future_disp"] * 10, smask, lv_min),
        subtask=F.cross_entropy(out["subtask"][:, :M].reshape(-1, out["subtask"].shape[-1]),
                                lab["subtask_m"].reshape(-1).long()),
    )
    if _goal_terms(out, lab, smask):
        L["goal_effect"] = gaussian_nll(out["goal_effect"], lab["goal_effect"] * 10, smask, lv_min)
    total = sum(L.values())
    return total, {f"probe_{k}": float(v.detach()) for k, v in L.items()}


@torch.no_grad()
def _readout_metrics_multi(out: dict, lab: dict, smask: torch.Tensor) -> dict:
    """Per packet slot m: held_by / acting_on accuracy (all and on positives), rel_pos error, subtask accuracy;
    plus the slot-independent queries. Keys are suffixed @m (m = packet slot, i.e. role order)."""
    m = smask.bool()
    res = {}
    for q, key in (("visible", "visible"), ("focused_on", "focus")):
        pred = out[q][..., 0] > 0
        y = lab[key].bool()
        res[q] = (int(((pred == y) & m).sum()), int(m.sum()))
        pos = y & m
        res[q + "_pos"] = (int(((pred == y) & pos).sum()), int(pos.sum()))
    derr = (out["observed_effect"][..., :3] / 10 - lab["future_disp"]).norm(dim=-1)
    res["observed_effect_err_m"] = (float((derr * m).sum()), int(m.sum()))
    res["desired_delta_err_m"] = res["observed_effect_err_m"]          # deprecated alias
    if _goal_terms(out, lab, smask):
        res.update(goal_metrics(out, lab, m))
    M = lab["held_m"].shape[-1]
    for a in range(M):
        for q, key in (("held_by", "held_m"), ("acting_on", "contact_m")):
            pred = out[q][:, :, a, 0] > 0
            y = lab[key][:, :, a].bool()
            res[f"{q}@{a}"] = (int(((pred == y) & m).sum()), int(m.sum()))
            pos = y & m
            res[f"{q}_pos@{a}"] = (int(((pred == y) & pos).sum()), int(pos.sum()))
            neg = ~y & m
            res[f"{q}_neg@{a}"] = (int(((pred == y) & neg).sum()), int(neg.sum()))
        err = (out["rel_pos"][:, :, a, :3] / 10 - lab["rel_tcp_m"][:, :, a]).norm(dim=-1)
        res[f"rel_pos_err_m@{a}"] = (float((err * m).sum()), int(m.sum()))
        res[f"subtask@{a}"] = (int((out["subtask"][:, a].argmax(-1) == lab["subtask_m"][:, a].long()).sum()),
                               int(lab["subtask_m"].shape[0]))
    # pooled over slots (comparable with single-assembly keys)
    for k in ("held_by", "held_by_pos", "acting_on", "acting_on_pos", "rel_pos_err_m", "subtask"):
        xs = [res[f"{k}@{a}"] for a in range(M)]
        res[k] = (sum(x for x, _ in xs), sum(n for _, n in xs))
    return res


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
        acc_probe_counts(self.counts[i], readout_metrics(self.probe(z, am, Sn), lab,
                                                         torch.ones(1, Sn, dtype=torch.bool, device=dev)))

    def on_end(self, i, env, ep):
        return dict(probe_counts=self.counts.pop(i))


def latent_hooks(policy, probe=None, *, device="cpu", paired: bool = False) -> list:
    """The former evaluate_latent's conventions: arm feasibility, session record, system 0 counters, displacement of
    every object, packet probes (when a probe is given), paired-scene identity."""
    return (H.arm_hooks() + [System0Stats(policy), H.Displacement()] + ([PacketProbeHook(probe, device)] if probe else [])
            + ([PairedMeta()] if paired else []))


def probe_rates(episodes, key: str = "probe_counts") -> dict:
    tot: dict = {}
    for e in episodes:
        acc_probe_counts(tot, e.metrics.get(key, {}))
    return {q: (x / n if n else None) for q, (x, n) in tot.items()}


def _tcp(s):
    r = s.robots[0]
    site = r.tcp_sites[next(a.id for a in r.spec.assemblies if a.kind in ("gripper", "hand"))]
    return s.data.site_xpos[mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_SITE, site)].copy()


class _CmdPolicy:
    """Policy whose tick-h command is `fn(h)` (Act.command; None = hold): a closed-loop system 0 with a HELD packet, or a
    replay of recorded commands."""

    def __init__(self, info, fn):
        self.info, self.fn, self.h = info, fn, 0

    def reset(self, spec, task, seeds, *, envs=None):
        self.h = 0

    def act(self, obs):
        from rrp.policies.base import Act
        c = self.fn(self.h)
        self.h += 1
        return {i: Act(c) for i in obs}


def _roll_ticks(s, info, fn, ticks, hooks=()):
    """`ticks` control ticks of `s` through harness.rollout under the command policy `fn` (budget-only task); a crashed
    tick raises."""
    from rrp.harness import rollout as R
    if ticks <= 0:
        return
    ep = R.rollout(lambda sd: s, _CmdPolicy(info, fn), H.budget_task("pick_place", s.spec.env_id), [0], batch=1,
                   max_steps=ticks, hooks=list(hooks))[0]
    if ep.outcome == "crash":
        raise RuntimeError(ep.metrics.get("note") or ep.failure_reason)


def disturbance_test(policy, realizer, robot_key: str, seeds: list[int], *, warmup_ticks=30, hold_ticks=8,
                     joint_offset=0.12, joint_index=1, device="cpu", prev_action: str | None = None) -> list[dict]:
    """Packet held FIXED (no system-i call). Compare TCP deviation from the undisturbed rollout for:
    A) system 0 closed loop (state-dependent realization), B) open-loop replay of the nominal commands expressed as
    deltas from the current state (no feedback), C) replay of the nominal ABSOLUTE targets (native servo
    stabilization only). Every stretch of ticks is a `harness.rollout` (warm-up: the latent stack's schedule; the
    held-packet stretches: a command policy + Recorder hooks).
    prev_action: optional PrevActionFeaturizer mode (the ladder's `--prev-action`); installed on each new session before
    the policy sees it, and fed by `hooks.PrevAction` on every stretch of ticks (snapshot/restore carries its state)."""
    from rrp.bodies.catalog import workbench_robots
    from rrp.envs.mujoco.scenario import BUILDERS
    from rrp.envs.mujoco.session import Session
    from rrp.core.action import NativeCommand
    from rrp.harness import rollout as R
    from rrp.policies.latent import LatentStackPolicy
    robot = workbench_robots()[robot_key]()
    rows = []
    for sd in seeds:
        s = Session(BUILDERS["pick_place"](robot, sd, n_distractors=0), seed=sd)
        pa = [H.PrevAction()] if prev_action is not None else []
        if prev_action is not None:
            from rrp.harness.eval.ladder import PrevActionFeaturizer
            from rrp.policies.features.featurizer import cached_featurizer
            s._rrp_featurizer = PrevActionFeaturizer(cached_featurizer(s), prev_action)
        f = policy.featurizer(s)
        s0 = LatentSystem0(realizer, f, latent_space_version=policy.lsv, realizer_compat_version=policy.rcv, device=device)
        stack = LatentStackPolicy(policy, realizer, replan_ticks=8, device=device, name="latent_disturbance",
                                  make_s0=lambda e: s0)
        if warmup_ticks > 0:                      # normal operation to mid-approach
            ep = R.rollout(lambda sd_: s, stack, H.budget_task("pick_place", s.spec.env_id), [sd], batch=1,
                           max_steps=warmup_ticks, hooks=pa)[0]
            if ep.outcome == "crash":
                raise RuntimeError(ep.metrics.get("note") or ep.failure_reason)
        info = stack.info
        calls_before = policy.calls
        p = policy.packets([s])[0]
        s0.receive(p, now=float(s.data.time), graph_version=s.runtime.graph_version)
        snap = s.snapshot()
        prev_snap = None if prev_action is None or s._rrp_featurizer.prev is None else s._rrp_featurizer.prev.copy()
        arm_q = lambda: s.data.qpos[s.robots[0].qadr[:len(s.robots[0].arm_joints)]].copy()
        held = lambda h: s0.tick(s, s.controller_version())
        # nominal
        nominal_cmds, tcp_nom, q_nom = [], [], []

        def rec(tcp, cmds=None, qs=None):
            def on_act(i, e, a):
                if cmds is not None:
                    cmds.append(a.command)
                if qs is not None:
                    qs.append(arm_q())
            return H.Recorder(on_act=on_act, on_step=lambda i, e, a, st: tcp.append(_tcp(e)))
        _roll_ticks(s, info, held, hold_ticks, [*pa, rec(tcp_nom, nominal_cmds, q_nom)])

        def perturb():
            s.restore(snap)
            if prev_action is not None:
                s._rrp_featurizer.prev = None if prev_snap is None else prev_snap.copy()
            s0.receive(p, now=float(s.data.time), graph_version=s.runtime.graph_version)
            adr = s.robots[0].qadr[joint_index]
            s.data.qpos[adr] += joint_offset
            s.data.qvel[:] = 0
            mujoco.mj_forward(s.model, s.data)
        # A: closed-loop system 0
        perturb()
        tcp_a = []
        _roll_ticks(s, info, held, hold_ticks, [*pa, rec(tcp_a)])
        # B: open-loop deltas (predetermined trajectory relative to state at disturbance)
        perturb()
        tcp_b = []
        q_start = arm_q()

        def delta_cmd(h):
            c = nominal_cmds[h]
            g = dict(c.groups)
            g["arm"] = (np.array(c.groups["arm"]) - q_nom[0] + q_start).tolist()
            return NativeCommand(controller_version=c.controller_version, groups=g, source="debug")
        _roll_ticks(s, info, delta_cmd, len(nominal_cmds), [*pa, rec(tcp_b)])
        # C: absolute nominal targets (servo stabilization)
        perturb()
        tcp_c = []
        _roll_ticks(s, info, lambda h: nominal_cmds[h], len(nominal_cmds), [*pa, rec(tcp_c)])
        dev_ = lambda tr: float(np.linalg.norm(tr[-1] - tcp_nom[-1]))
        rows.append(dict(robot=robot_key, seed=sd, system_i_calls_during_hold=policy.calls - calls_before - 1,
                         final_dev_closed_loop_m=dev_(tcp_a), final_dev_open_loop_delta_m=dev_(tcp_b),
                         final_dev_servo_absolute_m=dev_(tcp_c), hold_ticks=hold_ticks, joint_offset=joint_offset,
                         feedback_dt_s=s.dt))
    return rows
