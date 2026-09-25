"""Causal edits and composition of the RECEIVED latent packet (R38 tests 8-9; acceptance list in
research/reports/latent_slice1_progress.md "Pending").

System i and system 0 are frozen. The only thing changed is the packet that system 0 receives. Probes are used for
one purpose: to DEFINE an edit direction in z (a delta that moves the probe's readout by a declared amount while
anchoring its other readouts). The evidence is the behaviour of system 0 in closed-loop physics, compared with the
unedited packet from the identical state (same seed, same snapshot, same flow noise).

Two protocols:
  window  : along an unedited closed-loop rollout, at declared decision ticks, snapshot the state; for every
            condition restore the snapshot, hand system 0 the (edited) packet and run one replan period (8 ticks,
            no system-i call). Exactly paired: the control is the same packet re-run from the same snapshot.
  episode : whole closed-loop episodes where EVERY received packet is transformed (system i keeps replanning from
            the actual state). Flow noise is keyed by (seed, system-i call), so control and edited runs share noise.

Conditions (packet transforms):
  rel+x / rel-x / rel+y : probe-guided delta so that probe rel_pos(task object, manipulator 0) moves by +-X along
                          world x / y. Prediction: the TCP (and a carried object) shifts along the same direction.
  rel+xy                : one delta optimized for (+X, +X, 0)                                       [composition]
  sum_xy                : delta(rel+x) + delta(rel+y), added without re-optimization               [composition]
  rand                  : random direction with the norm of delta(rel+x) (specificity control)
  cf+x / cf-x           : packet transplanted from system i's own output for a counterfactual state in which the
                          task object (and its tracker belief) is displaced by +-X along x: an on-manifold edit of
                          the same semantic quantity. Prediction as rel+x / rel-x. Only while the object is not held.
  zero                  : z = 0 (negative control: a system 0 that ignores z would not change)
  shuffle               : packet generated for ANOTHER seed (same body) at the same tick (negative control)
  focus_swap            : packet transplanted from a counterfactual state where the task object and distractor0
                          exchange places (physically and in the public tracker belief). Prediction: the TCP moves
                          toward the distractor. Only while the task object is not held and a distractor exists.
  chain_A_then_B        : (episode) unedited packets until tick T1, then focus_swap packets: two objects in
                          sequence under the latent path                                             [composition]
Source label for every edited packet: source="debug", sampling.intervention=<condition>; the policy is
learned:<checkpoint>. Evaluation uses privileged simulator truth (TCP site, object poses) for MEASUREMENT only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from rrp.control.latent_realizer import LatentSystem0
from rrp.contracts.errors import ControllerRejection, StaleActionError

WINDOW_CONDS = ("control_replay", "rel+x", "rel-x", "rel+y", "rel+xy", "sum_xy", "rand", "cf+x", "cf-x", "zero",
                "shuffle", "focus_swap")
EPISODE_CONDS = ("control", "rel+x", "rel-x", "rand", "cf+x", "cf-x", "zero", "shuffle", "focus_swap",
                 "chain_A_then_B")
DIRS = {"rel+x": (1.0, 0.0, 0.0), "rel-x": (-1.0, 0.0, 0.0), "rel+y": (0.0, 1.0, 0.0), "rel+xy": (1.0, 1.0, 0.0)}
CF_DIRS = {"cf+x": (1.0, 0.0, 0.0), "cf-x": (-1.0, 0.0, 0.0)}


# ------------------------------------------------------------------ helpers
def load_probe(probe_path, rep_P, dev):
    """Measurement probe used to define edit directions (post-hoc probe = same procedure for sem and nosem)."""
    if not probe_path:
        return rep_P
    from rrp.model.latent_probes import PacketProbe
    st = torch.load(probe_path, map_location=dev, weights_only=False)
    P = PacketProbe(**st["cfg"]).to(dev).eval()
    P.load_state_dict(st["state"])
    for p in P.parameters():
        p.requires_grad_(False)
    return P


def _body_pos(s, name):
    return s.data.xpos[s.model.body(name).id].copy()


def _tcp(s):
    from rrp.evaluation.latent_eval import _tcp as t
    return t(s)


def _held(s, body="cube"):
    return any(body in v for v in s.truth().held_by.values())


def _slot(s, body):
    return next(i for i, o in enumerate(s.detectables) if o.sim_body == body)


def deliver(p, s, z=None, tag=None):
    """Declared transform of a packet before hand-over: restamp validity to now (same interval length); replace z
    for edits. Edited packets are labelled source=debug + sampling.intervention."""
    now = float(s.data.time)
    upd = dict(valid_from=now, valid_until=now + (p.valid_until - p.valid_from), graph_version=s.runtime.graph_version,
               runtime_version=s.runtime.runtime_version)
    if z is not None:
        upd["z"] = np.ascontiguousarray(np.asarray(z, np.float32))
    if tag:
        upd.update(source="debug", sampling=dict(p.sampling, intervention=tag))
    return p.model_copy(update=upd)


def _anchor(o, o0, emask, tmask):
    """Keep every other probe readout of the packet where it was (per packet, [B])."""
    em, tm = emask.float(), tmask.float()
    den = em.sum(-1).clamp(min=1)
    L = 0
    for q in ("visible", "focused_on", "looking_at", "desired_delta"):
        L = L + (((o[q] - o0[q]) ** 2).sum(-1) * em).sum(-1) / den
    for q in ("held_by", "acting_on"):
        L = L + (((o[q][:, :, 0, 0] - o0[q][:, :, 0, 0]) ** 2) * em).sum(-1) / den
    L = L + ((((o["rel_pos"][:, :, 0, :3] - o0["rel_pos"][:, :, 0, :3]) ** 2).sum(-1)) * tm).sum(-1) / den
    L = L + ((o["subtask"][:, 0] - o0["subtask"][:, 0]) ** 2).sum(-1)
    return L


def probe_edits(P, zs, ents, n_ents, deltas_m, *, steps=80, lr=0.05, anchor=1.0, l2=1e-3, dev="cpu"):
    """Batched probe-guided edits. zs: list of [K,M,D]; ents: task-object slot; deltas_m [B,3]: requested change of
    the probe's rel_pos(ent, manipulator 0) readout in metres. Returns (edited z list, info rows)."""
    B = len(zs)
    z0 = torch.tensor(np.stack(zs), dtype=torch.float32, device=dev)
    am = torch.ones(B, z0.shape[2], dtype=torch.bool, device=dev)
    S = int(max(n_ents))
    emask = torch.arange(S, device=dev)[None] < torch.tensor(n_ents, device=dev)[:, None]
    ent = torch.tensor(ents, device=dev)
    ar = torch.arange(B, device=dev)
    tmask = emask.clone()
    tmask[ar, ent] = False
    dm = torch.tensor(np.asarray(deltas_m, np.float32), device=dev)
    with torch.enable_grad():
        with torch.no_grad():
            o0 = P(z0, am, S)
        goal = o0["rel_pos"][ar, ent, 0, :3] + dm * 10          # probe works in decimetres
        d = torch.zeros_like(z0, requires_grad=True)
        opt = torch.optim.Adam([d], lr=lr)
        for _ in range(steps):
            o = P(z0 + d, am, S)
            Lt = ((o["rel_pos"][ar, ent, 0, :3] - goal) ** 2).sum(-1)
            L = (Lt + anchor * _anchor(o, o0, emask, tmask) + l2 * (d ** 2).flatten(1).mean(-1)).sum()
            opt.zero_grad()
            L.backward()
            opt.step()
    with torch.no_grad():
        o = P(z0 + d, am, S)
        ach = ((o["rel_pos"][ar, ent, 0, :3] - o0["rel_pos"][ar, ent, 0, :3]) / 10).cpu().numpy()
        anc = _anchor(o, o0, emask, tmask).cpu().numpy()
        dn = d.flatten(1).norm(dim=-1).cpu().numpy()
        zn = z0.flatten(1).norm(dim=-1).cpu().numpy()
    ze = (z0 + d).detach().cpu().numpy()
    dd = d.detach().cpu().numpy()
    info = [dict(probe_shift_m=ach[b].tolist(), anchor_residual=float(anc[b]), delta_norm=float(dn[b]),
                 delta_norm_ratio=float(dn[b] / max(zn[b], 1e-9))) for b in range(B)]
    return [ze[b] for b in range(B)], [dd[b] for b in range(B)], info


def _rand_like(delta, key):
    g = np.random.default_rng(key)
    r = g.standard_normal(delta.shape).astype(np.float32)
    return r / np.linalg.norm(r) * np.linalg.norm(delta)


def swap_state(s, a="cube", b="distractor0"):
    """Counterfactual: objects a and b exchange places, physically and in the public tracker belief."""
    pa, pb = _body_pos(s, a), _body_pos(s, b)
    s.teleport_object(a, [pb[0], pb[1], pa[2]], source="causal_counterfactual")
    s.teleport_object(b, [pa[0], pa[1], pb[2]], source="causal_counterfactual")
    ta, tb = s.tracker.tracks[_slot(s, a)], s.tracker.tracks[_slot(s, b)]
    for k in ("mean", "var", "visible", "last_seen"):
        va, vb = getattr(ta, k), getattr(tb, k)
        setattr(ta, k, vb)
        setattr(tb, k, va)


def shift_state(s, d, body="cube"):
    """Counterfactual: the task object is displaced by d (world), physically and in the public tracker belief."""
    p = _body_pos(s, body)
    s.teleport_object(body, (p + np.asarray(d)).tolist(), source="causal_counterfactual")
    tr = s.tracker.tracks[_slot(s, body)]
    if tr.mean is not None:
        tr.mean = (np.asarray(tr.mean) + np.asarray(d)).tolist()


def counterfactual_packets(policy, sessions, keys, mutate=swap_state):
    """System-i packets generated in a counterfactual of each session (then the real state is restored). The
    packet is system i's OWN (on-manifold) packet for that context; the same flow noise key is used."""
    snaps = [s.snapshot() for s in sessions]
    for s in sessions:
        mutate(s)
    try:
        return policy.packets(sessions, noise_keys=keys)
    finally:
        for s, sn in zip(sessions, snaps):
            s.restore(sn)


def _key(seed, call):
    return int(seed) * 1000 + int(call)


def _session(robot, seed, task="pick_place"):
    from rrp.sim.scenario import BUILDERS
    from rrp.sim.native import Session
    from rrp.control.teachers import PickPlaceTeacher
    s = Session(BUILDERS[task](robot, seed, n_distractors=seed % 3), seed=seed)
    return s, PickPlaceTeacher(s).feasibility()["feasible"]


def _s0(policy, R, s, dev):
    return LatentSystem0(R, policy.featurizer(s), latent_space_version=policy.lsv, realizer_compat_version=policy.rcv,
                         device=dev)


def _recv(s0, p, s):
    try:
        s0.receive(p, now=float(s.data.time), graph_version=s.runtime.graph_version)
        return True
    except (ControllerRejection, StaleActionError):
        return False


# ------------------------------------------------------------------ window protocol (exactly paired)
def window_protocol(policy, R, P, robot_key, seeds, *, decision_ticks=(8, 32, 56, 80, 104, 128), window=8,
                    replan=8, delta_m=0.05, conditions=WINDOW_CONDS, edit_steps=80, dev="cpu", log=print):
    from rrp.morphology.catalog import workbench_robots
    robot = workbench_robots()[robot_key]()
    rows = []
    last_ctrl = {}                     # decision tick -> control packet of the most recent OTHER seed (shuffle)
    horizon = max(decision_ticks) + 1
    for sd in seeds:
        s, feas = _session(robot, sd)
        if not feas:
            rows.append(dict(robot=robot_key, seed=sd, skipped="infeasible"))
            continue
        s0 = _s0(policy, R, s, dev)
        calls = 0
        mine = {}
        for step in range(horizon):
            if step % replan == 0 or s0.packet is None:
                p = policy.packets([s], noise_keys=[_key(sd, calls)])[0]
                calls += 1
                _recv(s0, p, s)
                if step in decision_ticks:
                    mine[step] = p
                    rows.extend(_window_conditions(policy, R, P, s, p, robot_key, sd, step, window, delta_m,
                                                   conditions, last_ctrl.get(step), edit_steps, dev,
                                                   _key(sd, calls - 1)))
            r = s.step(s0.tick(s, s.controller_version()))
            if s.runtime.succeeded() or _body_pos(s, "cube")[2] < -0.05:
                break
        last_ctrl.update(mine)
        log(f"[window] {robot_key} seed {sd}: {sum(1 for r in rows if r.get('seed') == sd)} rows", flush=True)
    return rows


def _run_window(policy, R, s, snap, packet, window, dev):
    s.restore(snap)
    s0 = _s0(policy, R, s, dev)
    ok = _recv(s0, packet, s)
    tcp, cube, dis = [_tcp(s)], [_body_pos(s, "cube")], []
    has_d = any(o.sim_body == "distractor0" for o in s.detectables)
    if has_d:
        dis.append(_body_pos(s, "distractor0"))
    for _ in range(window):
        s.step(s0.tick(s, s.controller_version()))
        tcp.append(_tcp(s))
        cube.append(_body_pos(s, "cube"))
    return dict(accepted=ok, tcp=np.array(tcp), cube=np.array(cube), dis0=dis[0] if dis else None)


def _window_conditions(policy, R, P, s, p, robot_key, sd, step, window, delta_m, conditions, other, edit_steps,
                       dev, key):
    snap = s.snapshot()
    held = _held(s)
    ent = s.entity_slots.get("cube", _slot(s, "cube"))
    nS = len(s.detectables)
    packets = {"control_replay": deliver(p, s)}
    info = {}
    need = [c for c in ("rel+x", "rel-x", "rel+y", "rel+xy") if c in conditions or
            (c in ("rel+x", "rel+y") and "sum_xy" in conditions) or (c == "rel+x" and "rand" in conditions)]
    if need:
        zs, ds, inf = probe_edits(P, [p.z] * len(need), [ent] * len(need), [nS] * len(need),
                                  [np.array(DIRS[c]) * delta_m for c in need], steps=edit_steps, dev=dev)
        dz = dict(zip(need, ds))
        for c, z, i in zip(need, zs, inf):
            packets[c] = deliver(p, s, z, c)
            info[c] = i
        if "sum_xy" in conditions:
            packets["sum_xy"] = deliver(p, s, p.z + dz["rel+x"] + dz["rel+y"], "sum_xy")
            zz = torch.tensor(np.stack([p.z, p.z + dz["rel+x"] + dz["rel+y"]]), device=dev)
            with torch.no_grad():
                o = P(zz, torch.ones(2, zz.shape[2], dtype=torch.bool, device=dev), nS)
            info["sum_xy"] = dict(probe_shift_m=((o["rel_pos"][1, ent, 0, :3] - o["rel_pos"][0, ent, 0, :3]) / 10
                                                 ).cpu().tolist(),
                                  delta_norm=float(np.linalg.norm(dz["rel+x"] + dz["rel+y"])),
                                  delta_norm_ratio=float(np.linalg.norm(dz["rel+x"] + dz["rel+y"]) /
                                                         np.linalg.norm(p.z)))
        if "rand" in conditions:
            r = _rand_like(dz["rel+x"], key)
            packets["rand"] = deliver(p, s, p.z + r, "rand")
            info["rand"] = dict(delta_norm=float(np.linalg.norm(r)), delta_norm_ratio=float(np.linalg.norm(r) /
                                                                                              np.linalg.norm(p.z)))
    if "zero" in conditions:
        packets["zero"] = deliver(p, s, np.zeros_like(p.z), "zero")
    if "shuffle" in conditions and other is not None and other.z.shape == p.z.shape:
        packets["shuffle"] = deliver(other, s, other.z, "shuffle")
    has_d = any(o.sim_body == "distractor0" for o in s.detectables)
    if "focus_swap" in conditions and has_d and not held:
        cf = counterfactual_packets(policy, [s], [key])[0]
        packets["focus_swap"] = deliver(cf, s, cf.z, "focus_swap")
        info["focus_swap"] = dict(delta_norm=float(np.linalg.norm(cf.z - p.z)),
                                  delta_norm_ratio=float(np.linalg.norm(cf.z - p.z) / np.linalg.norm(p.z)))
    for c, dvec in CF_DIRS.items():
        if c in conditions and not held:
            cf = counterfactual_packets(policy, [s], [key], lambda ss, d=np.array(dvec) * delta_m: shift_state(ss, d))[0]
            packets[c] = deliver(cf, s, cf.z, c)
            zz = torch.tensor(np.stack([p.z, cf.z]), device=dev)
            with torch.no_grad():
                o = P(zz, torch.ones(2, zz.shape[2], dtype=torch.bool, device=dev), nS)
            info[c] = dict(probe_shift_m=((o["rel_pos"][1, ent, 0, :3] - o["rel_pos"][0, ent, 0, :3]) / 10).cpu().tolist(),
                           delta_norm=float(np.linalg.norm(cf.z - p.z)),
                           delta_norm_ratio=float(np.linalg.norm(cf.z - p.z) / np.linalg.norm(p.z)))
    runs = {c: _run_window(policy, R, s, snap, pk, window, dev) for c, pk in packets.items()}
    s.restore(snap)
    ctrl = runs["control_replay"]
    out = []
    for c, r in runs.items():
        row = dict(robot=robot_key, seed=sd, tick=step, condition=c, held=held, accepted=r["accepted"],
                   tcp_shift_final=(r["tcp"][-1] - ctrl["tcp"][-1]).tolist(),
                   tcp_shift_mean=(r["tcp"][1:] - ctrl["tcp"][1:]).mean(0).tolist(),
                   cube_shift_final=(r["cube"][-1] - ctrl["cube"][-1]).tolist(),
                   tcp_disp_control=(ctrl["tcp"][-1] - ctrl["tcp"][0]).tolist(),
                   tcp_disp=(r["tcp"][-1] - r["tcp"][0]).tolist(), **info.get(c, {}))
        if r["dis0"] is not None:
            u = r["dis0"] - ctrl["cube"][0]
            u[2] = 0
            row["cube_to_distractor_unit"] = (u / max(np.linalg.norm(u), 1e-9)).tolist()
            row["dist_tcp_distractor_change"] = float(np.linalg.norm(r["tcp"][-1] - r["dis0"]) -
                                                      np.linalg.norm(ctrl["tcp"][-1] - r["dis0"]))
        out.append(row)
    return out


# ------------------------------------------------------------------ episode protocol (whole closed loop)
def episode_protocol(policy, R, P, robot_key, seeds, *, conditions=EPISODE_CONDS, replan=8, max_steps=240,
                     delta_m=0.05, chain_t1=48, edit_steps=60, dev="cpu", log=print):
    from rrp.morphology.catalog import workbench_robots
    robot = workbench_robots()[robot_key]()
    feas = {sd: _session(robot, sd)[1] for sd in seeds}
    seeds = [sd for sd in seeds if feas[sd]]
    ctrl_packets: dict[int, list] = {}
    rows = []
    order = ["control"] + [c for c in conditions if c != "control"]
    for cond in order:
        res = _episodes(policy, R, P, robot, robot_key, seeds, cond, replan, max_steps, delta_m, chain_t1, edit_steps,
                        dev, ctrl_packets)
        rows.extend(res)
        log(f"[episode] {robot_key} {cond}: success {sum(r['success'] for r in res)}/{len(res)}", flush=True)
    return rows


def _episodes(policy, R, P, robot, robot_key, seeds, cond, replan, max_steps, delta_m, chain_t1, edit_steps, dev,
              ctrl_packets):
    from rrp.sim.scenario import BUILDERS
    from rrp.sim.native import Session
    S = [Session(BUILDERS["pick_place"](robot, sd, n_distractors=sd % 3), seed=sd) for sd in seeds]
    s0 = [_s0(policy, R, s, dev) for s in S]
    calls = [0] * len(S)
    done = [False] * len(S)
    has_d = [any(o.sim_body == "distractor0" for o in s.detectables) for s in S]
    traj = [dict(tcp=[_tcp(s)], cube=[_body_pos(s, "cube")],
                 dis0=[_body_pos(s, "distractor0")] if has_d[k] else None) for k, s in enumerate(S)]
    edits = [[] for _ in S]
    if cond == "control":
        for sd in seeds:
            ctrl_packets[sd] = []
    t0 = time.time()
    for step in range(max_steps):
        act = [k for k in range(len(S)) if not done[k]]
        if not act:
            break
        need = [k for k in act if step % replan == 0 or s0[k].packet is None]
        if need:
            keys = [_key(seeds[k], calls[k]) for k in need]
            base = policy.packets([S[k] for k in need], noise_keys=keys)
            if cond == "control":
                for k, p in zip(need, base):
                    ctrl_packets[seeds[k]].append(p)
            deliv = [deliver(p, S[k]) for k, p in zip(need, base)]
            if cond in DIRS or cond == "rand":
                c = "rel+x" if cond == "rand" else cond
                ents = [S[k].entity_slots.get("cube", _slot(S[k], "cube")) for k in need]
                zs, ds, inf = probe_edits(P, [p.z for p in base], ents, [len(S[k].detectables) for k in need],
                                          [np.array(DIRS[c]) * delta_m] * len(need), steps=edit_steps, dev=dev)
                for j, k in enumerate(need):
                    if cond == "rand":
                        r = _rand_like(ds[j], keys[j])
                        deliv[j] = deliver(base[j], S[k], base[j].z + r, "rand")
                        edits[k].append(dict(delta_norm_ratio=float(np.linalg.norm(r) / np.linalg.norm(base[j].z))))
                    else:
                        deliv[j] = deliver(base[j], S[k], zs[j], cond)
                        edits[k].append(inf[j])
            elif cond == "zero":
                deliv = [deliver(p, S[k], np.zeros_like(p.z), "zero") for k, p in zip(need, base)]
            elif cond == "shuffle":
                for j, k in enumerate(need):
                    donor = ctrl_packets[seeds[(k + 1) % len(seeds)]]
                    q = donor[min(calls[k], len(donor) - 1)]
                    if q.z.shape == base[j].z.shape:
                        deliv[j] = deliver(q, S[k], q.z, "shuffle")
            elif cond in CF_DIRS:
                sw = [j for j, k in enumerate(need) if not _held(S[k])]
                if sw:
                    d = np.array(CF_DIRS[cond]) * delta_m
                    cf = counterfactual_packets(policy, [S[need[j]] for j in sw], [keys[j] for j in sw],
                                                lambda ss: shift_state(ss, d))
                    for j, q in zip(sw, cf):
                        deliv[j] = deliver(q, S[need[j]], q.z, cond)
                        edits[need[j]].append(dict(step=step))
            elif cond in ("focus_swap", "chain_A_then_B"):
                sw = [j for j, k in enumerate(need) if has_d[k] and not _held(S[k]) and
                      (cond == "focus_swap" or step >= chain_t1)]
                if sw:
                    cf = counterfactual_packets(policy, [S[need[j]] for j in sw], [keys[j] for j in sw])
                    for j, q in zip(sw, cf):
                        deliv[j] = deliver(q, S[need[j]], q.z, cond)
                        edits[need[j]].append(dict(step=step))
            for j, k in enumerate(need):
                calls[k] += 1
                _recv(s0[k], deliv[j], S[k])
        for k in act:
            s = S[k]
            s.step(s0[k].tick(s, s.controller_version()))
            traj[k]["tcp"].append(_tcp(s))
            traj[k]["cube"].append(_body_pos(s, "cube"))
            if has_d[k]:
                traj[k]["dis0"].append(_body_pos(s, "distractor0"))
            if s.runtime.succeeded() or _body_pos(s, "cube")[2] < -0.05:
                done[k] = True
    rows = []
    for k, s in enumerate(S):
        tcp, cube = np.array(traj[k]["tcp"]), np.array(traj[k]["cube"])
        row = dict(robot=robot_key, seed=seeds[k], condition=cond, success=bool(s.privileged_success()),
                   steps=len(tcp) - 1, system_i_calls=calls[k], packets_rejected=s0[k].stats.rejected,
                   fallback_holds=s0[k].stats.fallback_holds, cube_lift_m=float(cube[:, 2].max() - cube[0, 2]),
                   min_dist_tcp_cube=float(np.linalg.norm(tcp - cube, axis=1).min()),
                   tcp=tcp[::2].round(4).tolist(), cube_final=cube[-1].tolist(), n_edits=len(edits[k]),
                   edit_info=edits[k][:3])
        if has_d[k]:
            dis = np.array(traj[k]["dis0"])
            row.update(dis0_init=dis[0].tolist(), dis0_lift_m=float(dis[:, 2].max() - dis[0, 2]),
                       min_dist_tcp_dis0=float(np.linalg.norm(tcp - dis, axis=1).min()),
                       final_dist_tcp_dis0=float(np.linalg.norm(tcp[-1] - dis[-1])),
                       final_dist_tcp_cube=float(np.linalg.norm(tcp[-1] - cube[-1])))
        rows.append(row)
    return rows


# ------------------------------------------------------------------ summaries (seed = statistical unit)
def boot_ci(x, n=4000, seed=0):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0:
        return dict(mean=None, lo=None, hi=None, n=0)
    g = np.random.default_rng(seed)
    bs = x[g.integers(0, len(x), (n, len(x)))].mean(1) if len(x) > 1 else np.array([x[0]])
    return dict(mean=float(x.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                n=int(len(x)))


def _per_seed(rows, f):
    by = {}
    for r in rows:
        v = f(r)
        if v is not None:
            by.setdefault((r["robot"], r["seed"]), []).append(v)
    return [float(np.mean(v)) for v in by.values()]


def summarize_window(rows, delta_m=0.05):
    rows = [r for r in rows if "condition" in r]
    out = {}
    for c in sorted({r["condition"] for r in rows}):
        rc = [r for r in rows if r["condition"] == c]
        e = dict(n_windows=len(rc))
        if c in DIRS or c in CF_DIRS:
            dv = DIRS.get(c) or CF_DIRS[c]
            u = np.array(dv) / np.linalg.norm(dv)
            e["tcp_shift_along_edit_m"] = boot_ci(_per_seed(rc, lambda r: float(np.dot(r["tcp_shift_final"], u))))
            e["gain_tcp_shift_per_requested_m"] = boot_ci(_per_seed(
                rc, lambda r: float(np.dot(r["tcp_shift_final"], u)) / (delta_m * np.linalg.norm(dv))))
            e["sign_agreement"] = boot_ci(_per_seed(rc, lambda r: float(np.dot(r["tcp_shift_final"], u) > 0)))
            e["probe_shift_along_edit_m"] = boot_ci(_per_seed(rc, lambda r: float(np.dot(r["probe_shift_m"], u))))
            e["cube_shift_along_edit_held_m"] = boot_ci(_per_seed(
                rc, lambda r: float(np.dot(r["cube_shift_final"], u)) if r["held"] else None))
        if c in ("rand", "sum_xy", "focus_swap") or c in DIRS or c in CF_DIRS:
            e["delta_norm_ratio"] = boot_ci(_per_seed(rc, lambda r: r.get("delta_norm_ratio")))
        if c == "rand":
            e["tcp_shift_along_x_m"] = boot_ci(_per_seed(rc, lambda r: float(r["tcp_shift_final"][0])))
        if c == "sum_xy":
            u = np.array([1.0, 1.0, 0]) / np.sqrt(2)
            e["tcp_shift_along_edit_m"] = boot_ci(_per_seed(rc, lambda r: float(np.dot(r["tcp_shift_final"], u))))
            e["probe_shift_along_edit_m"] = boot_ci(_per_seed(rc, lambda r: float(np.dot(r["probe_shift_m"], u))))
        if c == "focus_swap":
            e["tcp_shift_toward_distractor_m"] = boot_ci(_per_seed(
                rc, lambda r: float(np.dot(r["tcp_shift_final"], r["cube_to_distractor_unit"]))))
            e["dist_tcp_distractor_change_m"] = boot_ci(_per_seed(rc, lambda r: r["dist_tcp_distractor_change"]))
        e["tcp_shift_norm_m"] = boot_ci(_per_seed(rc, lambda r: float(np.linalg.norm(r["tcp_shift_final"]))))
        e["control_tcp_displacement_norm_m"] = boot_ci(_per_seed(rc, lambda r: float(np.linalg.norm(
            r["tcp_disp_control"]))))
        out[c] = e
    # paired contrasts and composition (per decision point, then per seed)
    idx = {(r["robot"], r["seed"], r["tick"], r["condition"]): r for r in rows}
    pts = {(r["robot"], r["seed"], r["tick"]) for r in rows}

    def paired(a, b, f):
        vals = []
        for p in pts:
            ra, rb = idx.get(p + (a,)), idx.get(p + (b,))
            if ra and rb:
                vals.append(dict(robot=p[0], seed=p[1], v=f(ra, rb)))
        return boot_ci(_per_seed(vals, lambda r: r["v"]))
    ux = np.array([1.0, 0, 0])
    con = {}
    if "rel+x" in out and "rand" in out:
        con["rel+x_minus_rand_along_x_m"] = paired("rel+x", "rand", lambda a, b: float(
            np.dot(a["tcp_shift_final"], ux) - np.dot(b["tcp_shift_final"], ux)))
    if "cf+x" in out and "cf-x" in out:
        con["cf_antisymmetry_(+x)-(-x)_along_x_m"] = paired("cf+x", "cf-x", lambda a, b: float(
            np.dot(a["tcp_shift_final"], ux) - np.dot(b["tcp_shift_final"], ux)))
    if "rel+x" in out and "rel-x" in out:
        con["antisymmetry_(+x)-(-x)_along_x_m"] = paired("rel+x", "rel-x", lambda a, b: float(
            np.dot(a["tcp_shift_final"], ux) - np.dot(b["tcp_shift_final"], ux)))
    for comp in ("rel+xy", "sum_xy"):
        if comp in out and "rel+x" in out and "rel+y" in out:
            vals = []
            for p in pts:
                ra, rb, rc = idx.get(p + ("rel+x",)), idx.get(p + ("rel+y",)), idx.get(p + (comp,))
                if ra and rb and rc:
                    s_add = np.array(ra["tcp_shift_final"]) + np.array(rb["tcp_shift_final"])
                    s_c = np.array(rc["tcp_shift_final"])
                    den = np.linalg.norm(ra["tcp_shift_final"]) + np.linalg.norm(rb["tcp_shift_final"])
                    vals.append(dict(robot=p[0], seed=p[1], res=float(np.linalg.norm(s_c - s_add) / max(den, 1e-9)),
                                     cos=float(s_c @ s_add / max(np.linalg.norm(s_c) * np.linalg.norm(s_add), 1e-12)),
                                     ratio=float(np.linalg.norm(s_c) / max(np.linalg.norm(s_add), 1e-9))))
            con[f"composition_{comp}_vs_sum_of_single_edits"] = dict(
                relative_residual=boot_ci(_per_seed(vals, lambda r: r["res"])),
                cosine=boot_ci(_per_seed(vals, lambda r: r["cos"])),
                norm_ratio=boot_ci(_per_seed(vals, lambda r: r["ratio"])))
    return dict(conditions=out, contrasts=con)


def summarize_episodes(rows):
    ctrl = {(r["robot"], r["seed"]): r for r in rows if r["condition"] == "control"}
    out = {}
    for c in sorted({r["condition"] for r in rows}):
        rc = [r for r in rows if r["condition"] == c]
        e = dict(n=len(rc), successes=sum(r["success"] for r in rc),
                 cube_lifted=sum(r["cube_lift_m"] > 0.03 for r in rc))
        from rrp.evaluation.statistics import wilson
        e["success_wilson95"] = wilson(e["successes"], len(rc))

        def along(r, u):
            c0 = ctrl.get((r["robot"], r["seed"]))
            if c0 is None:
                return None
            a, b = np.array(r["tcp"]), np.array(c0["tcp"])
            n = min(len(a), len(b))
            return float(((a[:n] - b[:n]) @ u).mean())
        if c in DIRS or c in CF_DIRS or c == "rand":
            u = np.array(DIRS.get(c) or CF_DIRS.get(c) or (1.0, 0, 0))
            u = u / np.linalg.norm(u)
            e["mean_tcp_offset_vs_control_along_edit_m"] = boot_ci(_per_seed(rc, lambda r: along(r, u)))
        e["mean_tcp_deviation_from_control_m"] = boot_ci(_per_seed(rc, lambda r: None if (r["robot"], r["seed"]) not in
                                                                   ctrl else float(np.linalg.norm(
            np.array(r["tcp"])[:min(len(r["tcp"]), len(ctrl[(r["robot"], r["seed"])]["tcp"]))] -
            np.array(ctrl[(r["robot"], r["seed"])]["tcp"])[:min(len(r["tcp"]),
                                                                 len(ctrl[(r["robot"], r["seed"])]["tcp"]))],
            axis=1).mean())))
        e["min_dist_tcp_cube_m"] = boot_ci(_per_seed(rc, lambda r: r["min_dist_tcp_cube"]))
        rd = [r for r in rc if "min_dist_tcp_dis0" in r]
        if rd:
            e["with_distractor_n"] = len(rd)
            e["min_dist_tcp_distractor_m"] = boot_ci(_per_seed(rd, lambda r: r["min_dist_tcp_dis0"]))
            e["final_dist_tcp_distractor_m"] = boot_ci(_per_seed(rd, lambda r: r["final_dist_tcp_dis0"]))
            e["final_dist_tcp_cube_m"] = boot_ci(_per_seed(rd, lambda r: r["final_dist_tcp_cube"]))
            e["distractor_lifted"] = sum(r["dis0_lift_m"] > 0.03 for r in rd)
            e["final_dist_tcp_distractor_minus_control_m"] = boot_ci(_per_seed(
                rd, lambda r: r["final_dist_tcp_dis0"] - ctrl[(r["robot"], r["seed"])]["final_dist_tcp_dis0"]
                if (r["robot"], r["seed"]) in ctrl else None))
        out[c] = e
    return out
