"""Semantic interventions on the transmitted packet with irrelevant-edit controls (correction 2026-09-25, item 4).

Packet dependence is not semantic control. Here every edit is a VALID semantic change of the task context, and the
packet is regenerated for it by a frozen packet source; system 0 (frozen) then acts in the REAL, unchanged scene.
We measure whether behaviour follows the edit (which object is grasped, where it is placed, success of the new goal)
and whether irrelevant edits of matched size leave behaviour alone.

Packet sources (labelled in every packet and row):
  oracle     : source="target_encoder_oracle" (ORACLE DIAGNOSTIC, not deployable). z = E(public context, supplied
               task, morphology, demonstrated behaviour), where the demonstration is the privileged scripted teacher's
               next H=16 commands, rolled out in a snapshot of the current state and discarded. A shadow teacher per
               condition is advanced on the real state every tick (its commands are never executed), so its phase
               follows the actual rollout (DAgger-style expert).
  teacher    : reference rung, source="scripted_teacher": the privileged expert's native commands for the edited
               task (shows each edit is physically achievable; no packet involved).
  generated  : source="learned" — system i's own packet (frozen flow) for the edited context; same flow-noise key as
               the control.

Conditions (edits of the context the packet is generated for; the physical scene is never changed):
  control                 task as given: task object ("cube") -> target zone.
  rebind_obj              the task is rebound to distractor0: in the public belief the bound object slot now holds
                          distractor0's track (and vice versa); the oracle's teacher demonstrates distractor0.
                          Prediction: distractor0 is grasped and placed in the zone; the cube stays.
  goal_shift              requested physical effect changed: the target zone belief is displaced by G (default
                          12 cm, toward the workspace centre line); the oracle's teacher places at the displaced goal.
                          Prediction: the cube ends near the displaced goal, not in the real zone.
  irrelevant_distractor   CONTROL: distractor0's belief is displaced by 10 cm (not bound to any role). Prediction: no
                          change in which object is grasped or where it is placed.
  orthogonal_matched      CONTROL: z_control + r with ||r|| = ||z_goal_shift - z_control|| (per packet) and r
                          orthogonal to the probe gradients of all semantic readouts (focus, rel_pos, held, acting_on,
                          subtask, visibility). Prediction: no change.
Physical outcomes are measured with privileged simulator truth (measurement only).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from rrp.control.teachers import PickPlaceTeacher
from rrp.evaluation.latent_causal import _body_pos, _key, _recv, _slot, _tcp, deliver

CONDITIONS = ("control", "rebind_obj", "goal_shift", "irrelevant_distractor", "orthogonal_matched")
ZONE_R = 0.05


class ShiftedGoalTeacher(PickPlaceTeacher):
    """Scripted teacher whose requested placement is the zone displaced by `zone_offset` (privileged demo)."""

    def __init__(self, session, zone_offset, **kw):
        self.zone_offset = np.asarray(zone_offset, float)
        super().__init__(session, **kw)

    def _body(self, name):
        p, q = super()._body(name)
        return (p + self.zone_offset if name == self.zone else p), q


# ------------------------------------------------------------------ belief-level context edits (public observation)
def _swap_tracks(s, a, b):
    ta, tb = s.tracker.tracks[_slot(s, a)], s.tracker.tracks[_slot(s, b)]
    for k in ("mean", "var", "visible", "last_seen"):
        va, vb = getattr(ta, k), getattr(tb, k)
        setattr(ta, k, vb)
        setattr(tb, k, va)


def _shift_track(s, body, d):
    tr = s.tracker.tracks[_slot(s, body)]
    if tr.mean is not None:
        tr.mean = (np.asarray(tr.mean) + np.asarray(d)).tolist()


def goal_offset(s, g=0.12):
    z = _body_pos(s, "target_zone")
    return np.array([0.0, -np.sign(z[1] or 1.0) * g, 0.0])


def context_edit(cond, s, goal_off):
    if cond == "rebind_obj":
        _swap_tracks(s, "cube", "distractor0")
    elif cond == "goal_shift":
        _shift_track(s, "target_zone", goal_off)
    elif cond == "irrelevant_distractor":
        _shift_track(s, "distractor0", np.array([0.0, 0.10, 0.0]))


def make_teacher(cond, s, goal_off):
    if cond == "rebind_obj":
        return PickPlaceTeacher(s, obj="distractor0")
    if cond == "goal_shift":
        return ShiftedGoalTeacher(s, goal_off)
    return PickPlaceTeacher(s)


# ------------------------------------------------------------------ packet sources
def build_packet(f, s, o, z, *, lsv, rcv, knot_times, source, name, sampling, validity=0.8):
    from rrp.contracts.latent_action import LatentActionChunk, AssemblyHandle, EntityHandle
    M = z.shape[1]
    gasms = [a for a in f.spec.assemblies if a.kind in ("gripper", "hand")][:M]
    now = float(s.data.time)
    return LatentActionChunk(
        latent_space_version=lsv, realizer_compat_version=rcv, z=np.ascontiguousarray(z, np.float32),
        knot_times=list(knot_times),
        assemblies=[AssemblyHandle(handle=f"asm:{f.spec.spec_hash}:{a.frame.link}", robot_index=0) for a in gasms],
        assembly_mask=[True] * M, entity_registry=[EntityHandle(handle=f"ent:{d.slot}") for d in o.object_descriptors],
        observation_id=o.observation_id, graph_version=s.runtime.graph_version,
        runtime_version=s.runtime.runtime_version, robot_spec_hash=f.spec.spec_hash, generated_at=time.time(),
        valid_from=now, valid_until=now + validity, source=source, policy_version=name, sampling=sampling)


class OracleSource:
    label = "oracle"

    def __init__(self, E, lcfg, res, rep_path, dev):
        self.E, self.lcfg, self.dev = E, lcfg, dev
        self.lsv, self.rcv = res["latent_space_version"], res["realizer_compat_version"]
        self.name = f"oracle:E({rep_path})"

    def featurizer(self, s):
        from rrp.data.collect import featurizer_for
        f = getattr(s, "_rrp_featurizer", None)
        if f is None:
            f = s._rrp_featurizer = featurizer_for(s)
        return f

    @torch.no_grad()
    def packet(self, s, cond, teacher, goal_off, key=None):
        from rrp.model.batch import collate_inputs
        from rrp.model.semantic_latent import assembly_tokens
        f = self.featurizer(s)
        snap, tst = s.snapshot(), teacher.state()
        try:
            context_edit(cond, s, goal_off)
            o = s.observe()
            pi = f(o)
            cmds = []
            for _ in range(self.lcfg.horizon):          # privileged teacher demo in a discarded snapshot
                c = teacher.act()
                cmds.append(c.groups)
                s.step(c)
        finally:
            s.restore(snap)
            teacher.load(tst)
        a = f.aspace.normalize(cmds, pi.q0)
        b = collate_inputs([pi]).to(self.dev)
        af, am, ai = assembly_tokens(b)
        at = torch.as_tensor(np.asarray(a, np.float32), device=self.dev)[None]
        N = b.node_feats.shape[1]
        at = at[..., :N]
        v = torch.ones_like(at, dtype=torch.bool) & b.node_mask[:, None, :]
        mu, _ = self.E(b, at, v, af, am, ai)
        M = int(am[0].sum())
        z = mu[0, :, :M].float().cpu().numpy()
        return build_packet(f, s, o, z, lsv=self.lsv, rcv=self.rcv, knot_times=self.lcfg.knot_times,
                            source="target_encoder_oracle", name=self.name,
                            sampling=dict(route="oracle_diagnostic", condition=cond, demo="scripted_teacher"))


class TeacherSource:
    """Reference rung (not a packet route): the scripted privileged teacher's native commands for the edited task."""
    label = "teacher"
    lsv = rcv = "n/a"

    def featurizer(self, s):
        from rrp.data.collect import featurizer_for
        return featurizer_for(s)


class GeneratedSource:
    label = "generated"

    def __init__(self, policy):
        self.pol = policy
        self.lsv, self.rcv = policy.lsv, policy.rcv

    def featurizer(self, s):
        return self.pol.featurizer(s)

    def packet(self, s, cond, teacher, goal_off, key=None):
        snap = s.snapshot()
        try:
            context_edit(cond, s, goal_off)
            p = self.pol.packets([s], noise_keys=[key])[0]
        finally:
            s.restore(snap)
        return p.model_copy(update=dict(sampling=dict(p.sampling, route="generated", condition=cond)))


# ------------------------------------------------------------------ irrelevant direction (probe-orthogonal)
def probe_orthogonal(P, z, n_ent, key, norm):
    """Random direction orthogonal to the probe gradients of every semantic readout, scaled to `norm`."""
    zt = torch.tensor(z[None], dtype=torch.float32, requires_grad=True)
    dev = next(P.parameters()).device
    zt_d = zt.to(dev)
    out = P(zt_d, torch.ones(1, z.shape[1], dtype=torch.bool, device=dev), n_ent)
    rows = []
    feats = [out["focused_on"][0, :, 0], out["visible"][0, :, 0], out["held_by"][0, :, 0, 0],
             out["acting_on"][0, :, 0, 0], out["rel_pos"][0, :, 0, :3].reshape(-1), out["subtask"][0, 0],
             out["desired_delta"][0, :, :3].reshape(-1)]
    y = torch.cat(feats)
    for i in range(len(y)):
        g, = torch.autograd.grad(y[i], zt, retain_graph=True)
        rows.append(g.reshape(-1).numpy())
    J = np.stack(rows)
    Q, _ = np.linalg.qr(J.T)                            # orthonormal basis of the probe-relevant span
    r = np.random.default_rng(key).standard_normal(z.size)
    r = r - Q @ (Q.T @ r)
    r = r / max(np.linalg.norm(r), 1e-12) * norm
    return r.reshape(z.shape).astype(np.float32), dict(probe_span_dim=int(np.linalg.matrix_rank(J)),
                                                      z_dim=int(z.size))


# ------------------------------------------------------------------ episodes
def run_condition(src, R, P, robot, robot_key, seed, cond, *, max_steps=300, replan=8, dev="cpu", g=0.12):
    from rrp.sim.scenario import BUILDERS
    from rrp.sim.native import Session
    from rrp.control.latent_realizer import LatentSystem0
    s = Session(BUILDERS["pick_place"](robot, seed, n_distractors=max(1, seed % 3)), seed=seed)
    goal_off = goal_offset(s, g)
    ctrl_t = PickPlaceTeacher(s)
    if not ctrl_t.feasibility()["feasible"]:
        return dict(robot=robot_key, seed=seed, condition=cond, skipped="infeasible_control")
    teachers = {}
    if src.label in ("oracle", "teacher"):
        need = {"orthogonal_matched": ("control", "goal_shift")}.get(cond, (cond,))
        for c in need:
            t = make_teacher(c, s, goal_off)
            if not t.feasibility()["feasible"]:
                return dict(robot=robot_key, seed=seed, condition=cond, skipped=f"infeasible_{c}")
            teachers[c] = t
    f = src.featurizer(s)
    s0 = LatentSystem0(R, f, latent_space_version=src.lsv, realizer_compat_version=src.rcv, device=dev)
    z0 = {b: _body_pos(s, b) for b in ("cube", "distractor0", "target_zone")}
    goal_new = z0["target_zone"] + goal_off
    lift = {"cube": 0.0, "distractor0": 0.0}
    tcp_tr, info, calls = [], [], 0
    for step in range(max_steps):
        if src.label == "teacher":                      # reference rung: the expert's native commands, no packet
            if cond == "orthogonal_matched":
                return dict(robot=robot_key, seed=seed, condition=cond, skipped="n/a_for_teacher")
            s.step(teachers[cond].act())
            for b in lift:
                lift[b] = max(lift[b], float(_body_pos(s, b)[2] - z0[b][2]))
            tcp_tr.append(_tcp(s))
            if teachers[cond].done:
                break
            continue
        if step % replan == 0 or s0.packet is None:
            key = _key(seed, calls)
            calls += 1
            if cond == "orthogonal_matched":
                pc = src.packet(s, "control", teachers.get("control"), goal_off, key)
                pg = src.packet(s, "goal_shift", teachers.get("goal_shift"), goal_off, key)
                nrm = float(np.linalg.norm(pg.z - pc.z))
                r, inf = probe_orthogonal(P, pc.z, len(s.detectables), key, nrm)
                p = deliver(pc, s, pc.z + r, "orthogonal_matched")
                info.append(dict(norm=nrm, rel_norm=nrm / float(np.linalg.norm(pc.z)), **inf))
            else:
                p = src.packet(s, cond, teachers.get(cond), goal_off, key)
            _recv(s0, p, s)
        s.step(s0.tick(s, s.controller_version()))
        for t in teachers.values():                     # shadow experts follow the REAL state (commands discarded)
            t.act()
        for b in lift:
            lift[b] = max(lift[b], float(_body_pos(s, b)[2] - z0[b][2]))
        tcp_tr.append(_tcp(s))
        if s.runtime.succeeded() or _body_pos(s, "cube")[2] < -0.05:
            break
    held = {b: any(b in v for v in s.truth().held_by.values()) for b in lift}
    fin = {b: _body_pos(s, b) for b in lift}
    xy = lambda a, b: float(np.linalg.norm((a - b)[:2]))
    placed = lambda b, goal: bool(xy(fin[b], goal) < ZONE_R and fin[b][2] < 0.06 and not held[b])
    tcp_tr = np.array(tcp_tr)
    row = dict(robot=robot_key, seed=seed, condition=cond, route=src.label, steps=step + 1, system_i_calls=calls,
               packets_rejected=s0.stats.rejected, privileged_success=bool(s.privileged_success()),
               lifted={b: v > 0.03 for b, v in lift.items()}, max_lift_m=lift,
               held_at_end=held,
               final_xy_to_zone={b: xy(fin[b], z0["target_zone"]) for b in lift},
               final_xy_to_shifted_goal={b: xy(fin[b], goal_new) for b in lift},
               placed_in_zone={b: placed(b, z0["target_zone"]) for b in lift},
               placed_at_shifted_goal={b: placed(b, goal_new) for b in lift},
               min_tcp_dist={b: float(np.linalg.norm(tcp_tr - z0[b], axis=1).min()) for b in lift},
               goal_offset=goal_off.tolist(), tcp=tcp_tr[::4].round(4).tolist(), edit_info=info[:3])
    row["followed"] = followed(row)
    return row


def followed(r):
    c = r["condition"]
    if c == "rebind_obj":
        return bool(r["lifted"]["distractor0"] and not r["lifted"]["cube"])
    if c == "goal_shift":
        return bool(r["placed_at_shifted_goal"]["cube"])
    return bool(r["placed_in_zone"]["cube"])        # control and irrelevant edits: original task behaviour


def semantic_suite(src, R, P, robot_key, seeds, conditions=CONDITIONS, *, max_steps=300, dev="cpu", log=print,
                   out_path: Path | None = None):
    from rrp.morphology.catalog import workbench_robots
    robot = workbench_robots()[robot_key]()
    rows = []
    for sd in seeds:
        for c in conditions:
            t0 = time.time()
            r = run_condition(src, R, P, robot, robot_key, sd, c, max_steps=max_steps, dev=dev)
            r["wall_s"] = round(time.time() - t0, 1)
            rows.append(r)
            if out_path:
                with open(out_path, "a") as fh:
                    fh.write(json.dumps(r) + "\n")
            log(f"[semantic] {src.label} {robot_key} seed {sd} {c}: "
                + (r.get("skipped") or f"followed={r['followed']} lifted={r['lifted']} "
                                        f"placed_zone={r['placed_in_zone']} placed_shift={r['placed_at_shifted_goal']}"),
                flush=True)
    return rows


def summarize_semantic(rows):
    from rrp.evaluation.statistics import wilson
    from rrp.evaluation.latent_causal import boot_ci
    ok = [r for r in rows if "skipped" not in r]
    ctrl = {(r["robot"], r["seed"]): r for r in ok if r["condition"] == "control"}
    out = {}
    for c in sorted({r["condition"] for r in ok}):
        rc = [r for r in ok if r["condition"] == c]
        n = len(rc)
        k = sum(r["followed"] for r in rc)
        e = dict(n=n, followed=k, followed_wilson95=wilson(k, n),
                 cube_lifted=sum(r["lifted"]["cube"] for r in rc),
                 distractor0_lifted=sum(r["lifted"]["distractor0"] for r in rc),
                 cube_in_zone=sum(r["placed_in_zone"]["cube"] for r in rc),
                 distractor0_in_zone=sum(r["placed_in_zone"]["distractor0"] for r in rc),
                 cube_at_shifted_goal=sum(r["placed_at_shifted_goal"]["cube"] for r in rc),
                 privileged_success=sum(r["privileged_success"] for r in rc),
                 skipped=sum(1 for r in rows if r["condition"] == c and "skipped" in r))
        e["cube_final_xy_to_zone_m"] = boot_ci([r["final_xy_to_zone"]["cube"] for r in rc])
        e["cube_final_xy_to_shifted_goal_m"] = boot_ci([r["final_xy_to_shifted_goal"]["cube"] for r in rc])
        # paired with control: did the same physical outcome occur?
        pair = [(r, ctrl[(r["robot"], r["seed"])]) for r in rc if (r["robot"], r["seed"]) in ctrl]
        if pair and c != "control":
            e["paired_n"] = len(pair)
            e["same_lifted_object_as_control"] = sum(r["lifted"] == q["lifted"] for r, q in pair)
            e["same_cube_in_zone_as_control"] = sum(r["placed_in_zone"]["cube"] == q["placed_in_zone"]["cube"]
                                                    for r, q in pair)
            e["tcp_mean_deviation_from_control_m"] = boot_ci([float(np.linalg.norm(
                np.array(r["tcp"])[:min(len(r["tcp"]), len(q["tcp"]))] -
                np.array(q["tcp"])[:min(len(r["tcp"]), len(q["tcp"]))], axis=1).mean()) for r, q in pair])
        out[c] = e
    return out
