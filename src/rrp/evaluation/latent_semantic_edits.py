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
PAIRED_CONDITIONS = CONDITIONS + ("control_replay",)
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
    """Displacement of the requested placement: among +-g along x/y, the candidate whose new goal stays inside the
    placement workspace and is farthest from every other object (a goal on top of a distractor is not valid)."""
    z = _body_pos(s, "target_zone")
    others = [_body_pos(s, o.sim_body) for o in s.detectables if o.sim_body != "target_zone"]
    best, score = None, -1.0
    for d in ([0, -g, 0], [0, g, 0], [-g, 0, 0], [g, 0, 0]):
        q = z + np.array(d)
        if not (0.28 <= q[0] <= 0.55 and -0.28 <= q[1] <= 0.28):
            continue
        m = min(float(np.linalg.norm((q - o)[:2])) for o in others)
        if m > score:
            best, score = np.array(d, float), m
    return best if best is not None else np.array([0.0, -np.sign(z[1] or 1.0) * g, 0.0])


def rebind_descriptor(s, body, ent="cube"):
    """Paired scenes: VALID rebinding of the task entity `ent` to the physical object `body`. Only public task
    context changes: the entity declaration's descriptor becomes `body`'s public descriptor (e.g. "red cube" ->
    "blue cube") and the public entity->slot binding follows it. Physics, tracker beliefs and runtime statuses are
    untouched (event statuses remain those of the real rollout; identical before any grasp)."""
    sn = s.snapshot()
    c = sn.components
    desc = s.scenario.object(body).descriptor
    for e in c["task_runtime"]["doc"]["entity_declarations"]:
        if e["id"] == ent:
            e["descriptor"] = desc
    c["belief_state"]["entity_slots"][ent] = _slot(s, body)
    s.restore(sn)


def rebind_actor(s, arm):
    """Dual-arm assign scenes: VALID manipulator-assignment edit: the actor of take/place is rebound to `arm`
    (the task graph of the other variant of the pair; same graph_version, same runtime statuses)."""
    from rrp.sim.dual_scenarios import assign_task
    sn = s.snapshot()
    c = sn.components
    doc = assign_task(arm)
    doc["graph_version"] = c["task_runtime"]["doc"].get("graph_version", 0)
    c["task_runtime"]["doc"] = doc
    s.restore(sn)


def context_edit(cond, s, goal_off):
    if cond == "rebind_obj" and getattr(s, "_sem_paired", False):
        rebind_descriptor(s, "distractor0")
    elif cond == "rebind_obj":
        _swap_tracks(s, "cube", "distractor0")
    elif cond == "swap_arm":
        rebind_actor(s, "right" if s._sem_assigned == "left" else "left")
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
        return _featurizer(s)

    @torch.no_grad()
    def packet(self, s, cond, teacher, goal_off, key=None):
        from rrp.model.batch import collate_inputs
        from rrp.model.semantic_latent import assembly_tokens
        f = self.featurizer(s)
        H = self.lcfg.horizon
        snap, tst = s.snapshot(), teacher.state()
        try:
            context_edit(cond, s, goal_off)
            o = s.observe()
            pi = f(o)
            cmds = []
            for _ in range(H):                          # privileged teacher demo in a discarded snapshot
                c = teacher.act()
                cmds.append(_flat(c))
                s.step(c)
                if getattr(teacher, "done", False):
                    break
        finally:
            s.restore(snap)
            teacher.load(tst)
        n = len(cmds)
        a = f.aspace.normalize(cmds + [cmds[-1]] * (H - n), pi.q0).astype(np.float16).astype(np.float32)
        b = collate_inputs([pi]).to(self.dev)
        af, am, ai = assembly_tokens(b)
        N = b.node_feats.shape[1]
        at = np.zeros((1, H, N), np.float32)
        vt = np.zeros((1, H, N), bool)
        at[0, :, :a.shape[1]] = a
        vt[0, :n, :a.shape[1]] = True
        mu, _ = self.E(b, torch.from_numpy(at).to(self.dev), torch.from_numpy(vt).to(self.dev), af, am, ai)
        M = int(am[0].sum())
        z = mu[0, :, :M].float().cpu().numpy()
        samp = dict(route="oracle_diagnostic", condition=cond, demo="scripted_teacher")
        if getattr(s, "_sem_dual", False):
            return dual_packet(f, s, o, z, lsv=self.lsv, rcv=self.rcv, knot_times=self.lcfg.knot_times,
                               source="target_encoder_oracle", name=self.name, sampling=samp)
        return build_packet(f, s, o, z, lsv=self.lsv, rcv=self.rcv, knot_times=self.lcfg.knot_times,
                            source="target_encoder_oracle", name=self.name, sampling=samp)


def _flat(c):
    """Teacher command -> flat group dict (single robot: NativeCommand.groups; multi: MultiFeaturizer.flatten)."""
    if isinstance(c, dict):
        from rrp.data.features_multi import MultiFeaturizer
        return MultiFeaturizer.flatten({i: x.groups for i, x in c.items()})
    return c.groups


def dual_packet(f, s, o, z, *, lsv, rcv, knot_times, source, name, sampling, validity=0.8):
    from rrp.contracts.latent_action import LatentActionChunk, EntityHandle
    from rrp.evaluation.dual_latent_eval import assembly_handles
    hs, mask = assembly_handles(f, z.shape[1])
    z = np.ascontiguousarray(z, np.float32)
    z[:, ~np.array(mask)] = 0.0
    now = float(s.data.time)
    return LatentActionChunk(
        latent_space_version=lsv, realizer_compat_version=rcv, z=z, knot_times=list(knot_times), assemblies=hs,
        assembly_mask=mask, entity_registry=[EntityHandle(handle=f"ent:{d.slot}") for d in o.object_descriptors],
        observation_id=o.observation_id, graph_version=s.runtime.graph_version,
        runtime_version=s.runtime.runtime_version, robot_spec_hash=f.spec_hash, generated_at=time.time(),
        valid_from=now, valid_until=now + validity, source=source, policy_version=name, sampling=sampling)


def _featurizer(s):
    f = getattr(s, "_rrp_featurizer", None)
    if f is None:
        if getattr(s, "_sem_dual", False):
            from rrp.evaluation.dual_latent_eval import multi_featurizer
            f = multi_featurizer(s)
        else:
            from rrp.data.collect import featurizer_for
            f = s._rrp_featurizer = featurizer_for(s)
    return f


class TeacherSource:
    """Reference rung (not a packet route): the scripted privileged teacher's native commands for the edited task."""
    label = "teacher"
    lsv = rcv = "n/a"

    def featurizer(self, s):
        return _featurizer(s)


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


# ------------------------------------------------------------------ approach-level measurement (privileged, eval only)
def robot_body_ids(s, robot_index=None):
    """Body ids of the robot(s) (links of the compiled specs; prefixed names resolved in the model)."""
    import mujoco
    ids = {}
    for ri, r in enumerate(s.robots):
        if robot_index is not None and ri != robot_index:
            continue
        pre = getattr(r, "prefix", "") or ""
        for l in r.spec.links:
            for nm in (l.name, pre + l.name):
                b = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_BODY, nm)
                if b >= 0:
                    ids[b] = ri
                    break
    return ids


def robot_contacts(s, obj_ids: dict, rb: dict) -> set:
    """{(object name, robot index)} currently in contact."""
    out = set()
    d, m = s.data, s.model
    for c in range(d.ncon):
        con = d.contact[c]
        b1, b2 = int(m.geom_bodyid[con.geom1]), int(m.geom_bodyid[con.geom2])
        for x, y in ((b1, b2), (b2, b1)):
            if x in obj_ids and y in rb:
                out.add((obj_ids[x], rb[y]))
    return out


def approach_metrics(tcp, objs0, first_contact, tcp0=None, *, near=0.06, move=0.04):
    """tcp [T,3]; objs0 {name: initial xyz}. first_contact {name: tick or None}. Returns per-object min distance,
    first tick within `near`, the object approached first, cosines of the initial xy motion direction (first tick at
    which the TCP has moved `move` in xy) toward each object, and the first-contacted object."""
    tcp = np.asarray(tcp, float)
    tcp0 = tcp[0] if tcp0 is None else np.asarray(tcp0, float)
    out = dict(min_tcp_dist={}, first_within={}, init_dir_cos={})
    for b, p in objs0.items():
        dd = np.linalg.norm(tcp - p, axis=1)
        out["min_tcp_dist"][b] = float(dd.min())
        w = np.nonzero(dd < near)[0]
        out["first_within"][b] = int(w[0]) if len(w) else None
    mv = np.linalg.norm(tcp[:, :2] - tcp0[:2], axis=1)
    k = np.nonzero(mv >= move)[0]
    out["init_move_tick"] = int(k[0]) if len(k) else None
    for b, p in objs0.items():
        if not len(k):
            out["init_dir_cos"][b] = None
            continue
        u = tcp[k[0], :2] - tcp0[:2]
        v = np.asarray(p)[:2] - tcp0[:2]
        out["init_dir_cos"][b] = float(u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-9))
    fw = {b: t for b, t in out["first_within"].items() if t is not None}
    out["approached_first"] = min(fw, key=fw.get) if fw else None
    fc = {b: t for b, t in first_contact.items() if t is not None}
    out["first_contact_tick"] = dict(first_contact)
    out["first_contact"] = min(fc, key=fc.get) if fc else None
    return out


# ------------------------------------------------------------------ episodes
def _scene(robot, key, scene):
    from rrp.sim.scenario import BUILDERS, build_pick_place_paired
    from rrp.sim.native import Session
    if scene == "paired":                              # key = 10 * scene_seed + patient (as eval-binding)
        sd, p = divmod(int(key), 10)
        s = Session(build_pick_place_paired(robot, sd, patient=p, n_objects=2 + sd % 2), seed=sd)
        s._sem_paired = True
        return s
    return Session(BUILDERS["pick_place"](robot, key, n_distractors=max(1, key % 3)), seed=key)


def paired_edit_keys(seed_start, n_scenes):
    """One episode per paired scene; the assigned patient alternates with the scene seed (balanced over physical
    objects); the rebind target distractor0 is the first other cube in physical order."""
    return [10 * sd + (sd % (2 + sd % 2)) for sd in range(seed_start, seed_start + n_scenes)]


def run_condition(src, R, P, robot, robot_key, seed, cond, *, max_steps=300, replan=8, dev="cpu", g=0.12,
                  scene="pick_place"):
    from rrp.control.latent_realizer import LatentSystem0
    s = _scene(robot, seed, scene)
    goal_off = goal_offset(s, g)
    ctrl_t = PickPlaceTeacher(s)
    base = dict(robot=robot_key, seed=seed, condition=cond, scene=scene)
    if not ctrl_t.feasibility()["feasible"]:
        return dict(base, skipped="infeasible_control")
    match = "rebind_obj" if scene == "paired" else "goal_shift"
    if cond == "control_replay" and src.label != "generated":
        return dict(base, skipped=f"n/a_deterministic_{src.label}")
    teachers = {}
    if src.label in ("oracle", "teacher"):
        need = {"orthogonal_matched": ("control", match), "control_replay": ("control",)}.get(cond, (cond,))
        for c in need:
            t = make_teacher(c, s, goal_off)
            if not t.feasibility()["feasible"]:
                return dict(base, skipped=f"infeasible_{c}")
            teachers[c] = t
    f = src.featurizer(s)
    s0 = LatentSystem0(R, f, latent_space_version=src.lsv, realizer_compat_version=src.rcv, device=dev)
    z0 = {b: _body_pos(s, b) for b in ("cube", "distractor0", "target_zone")}
    goal_new = z0["target_zone"] + goal_off
    lift = {"cube": 0.0, "distractor0": 0.0}
    objs = {s.model.body(b).id: b for b in lift}
    rb = robot_body_ids(s)
    fcon = {b: None for b in lift}
    tcp_tr, info, calls = [_tcp(s)], [], 0
    zlog = []

    def measure(step):
        for b in lift:
            lift[b] = max(lift[b], float(_body_pos(s, b)[2] - z0[b][2]))
        for b, _ in robot_contacts(s, objs, rb):
            if fcon[b] is None:
                fcon[b] = step
        tcp_tr.append(_tcp(s))

    for step in range(max_steps):
        if src.label == "teacher":                      # reference rung: the expert's native commands, no packet
            if cond == "orthogonal_matched":
                return dict(base, skipped="n/a_for_teacher")
            s.step(teachers[cond].act())
            measure(step)
            if teachers[cond].done:
                break
            continue
        if step % replan == 0 or s0.packet is None:
            key = _key(seed, calls) + (500 if cond == "control_replay" else 0)
            calls += 1
            if cond == "orthogonal_matched":
                pc = src.packet(s, "control", teachers.get("control"), goal_off, key)
                pg = src.packet(s, match, teachers.get(match), goal_off, key)
                nrm = float(np.linalg.norm(pg.z - pc.z))
                r, inf = probe_orthogonal(P, pc.z, len(s.detectables), key, nrm)
                p = deliver(pc, s, pc.z + r, "orthogonal_matched")
                info.append(dict(norm=nrm, rel_norm=nrm / float(np.linalg.norm(pc.z)), **inf))
            else:
                p = src.packet(s, "control" if cond == "control_replay" else cond,
                               teachers.get("control" if cond == "control_replay" else cond), goal_off, key)
            if len(zlog) < 4:
                zlog.append(float(np.linalg.norm(p.z)))
            _recv(s0, p, s)
        s.step(s0.tick(s, s.controller_version()))
        for t in teachers.values():                     # shadow experts follow the REAL state (commands discarded)
            t.act()
        measure(step)
        if s.runtime.succeeded() or _body_pos(s, "cube")[2] < -0.05:
            break
    held = {b: any(b in v for v in s.truth().held_by.values()) for b in lift}
    fin = {b: _body_pos(s, b) for b in lift}
    xy = lambda a, b: float(np.linalg.norm((a - b)[:2]))
    placed = lambda b, goal: bool(xy(fin[b], goal) < ZONE_R and fin[b][2] < 0.06 and not held[b])
    tcp_tr = np.array(tcp_tr)
    am = approach_metrics(tcp_tr, {b: z0[b] for b in lift}, fcon)
    row = dict(base, route=src.label, steps=step + 1, system_i_calls=calls,
               packets_rejected=s0.stats.rejected, privileged_success=bool(s.privileged_success()),
               lifted={b: v > 0.03 for b, v in lift.items()}, max_lift_m=lift,
               held_at_end=held,
               moved_xy_m={b: xy(fin[b], z0[b]) for b in lift},
               final_xy_to_zone={b: xy(fin[b], z0["target_zone"]) for b in lift},
               final_xy_to_shifted_goal={b: xy(fin[b], goal_new) for b in lift},
               placed_in_zone={b: placed(b, z0["target_zone"]) for b in lift},
               placed_at_shifted_goal={b: placed(b, goal_new) for b in lift},
               goal_offset=goal_off.tolist(), tcp=tcp_tr[::4].round(4).tolist(), edit_info=info[:3],
               packet_z_norm=zlog, objects0={b: z0[b].round(4).tolist() for b in z0}, **am)
    if scene == "paired":
        row["patient_color"] = s.scenario.meta["cube_color"]
        row["rebind_color"] = s.scenario.object("distractor0").descriptor
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
                   out_path: Path | None = None, scene="pick_place"):
    from rrp.morphology.catalog import workbench_robots
    robot = workbench_robots()[robot_key]()
    rows = []
    for sd in seeds:
        for c in conditions:
            t0 = time.time()
            r = run_condition(src, R, P, robot, robot_key, sd, c, max_steps=max_steps, dev=dev, scene=scene)
            r["wall_s"] = round(time.time() - t0, 1)
            rows.append(r)
            if out_path:
                with open(out_path, "a") as fh:
                    fh.write(json.dumps(r) + "\n")
            log(f"[semantic] {src.label} {robot_key} seed {sd} {c}: "
                + (r.get("skipped") or f"followed={r['followed']} lifted={r['lifted']} "
                                        f"approached_first={r['approached_first']} first_contact={r['first_contact']} "
                                        f"min_d={ {k: round(v, 3) for k, v in r['min_tcp_dist'].items()} } "
                                        f"placed_zone={r['placed_in_zone']} placed_shift={r['placed_at_shifted_goal']}"),
                flush=True)
    return rows


def _pref_min(r):          # >0: TCP came closer to the NEW binding (distractor0) than to the old one (cube)
    return r["min_tcp_dist"]["cube"] - r["min_tcp_dist"]["distractor0"]


def _pref_dir(r):          # >0: initial xy motion points more toward distractor0 than toward the cube
    c = r.get("init_dir_cos") or {}
    if c.get("cube") is None or c.get("distractor0") is None:
        return None
    return c["distractor0"] - c["cube"]


def _goal_pref(r):         # >0: the cube ended closer to the shifted goal than to the real zone (only if moved)
    if r["moved_xy_m"]["cube"] < 0.02:
        return None
    return r["final_xy_to_zone"]["cube"] - r["final_xy_to_shifted_goal"]["cube"]


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
        if rc and "approached_first" in rc[0]:        # approach-level (works before any grasp)
            for q in ("approached_first", "first_contact"):
                e[q] = {b: sum(r[q] == b for r in rc) for b in ("cube", "distractor0", None)}
                kk = sum(r[q] == "distractor0" for r in rc)
                e[f"{q}_new_frac"] = dict(k=kk, n=n, wilson95=wilson(kk, n))
            e["min_dist_pref_new_m"] = boot_ci([_pref_min(r) for r in rc])
            e["init_dir_pref_new"] = boot_ci([_pref_dir(r) for r in rc])
            e["cube_moved"] = sum(r["moved_xy_m"]["cube"] >= 0.02 for r in rc)
            e["goal_pref_new_m"] = boot_ci([_goal_pref(r) for r in rc])
        # paired with control: did the same physical outcome occur?
        pair = [(r, ctrl[(r["robot"], r["seed"])]) for r in rc if (r["robot"], r["seed"]) in ctrl]
        if pair and c != "control":
            e["paired_n"] = len(pair)
            e["same_lifted_object_as_control"] = sum(r["lifted"] == q["lifted"] for r, q in pair)
            e["same_cube_in_zone_as_control"] = sum(r["placed_in_zone"]["cube"] == q["placed_in_zone"]["cube"]
                                                    for r, q in pair)
            e["approach_preference_distractor_vs_control_m"] = boot_ci([_pref_min(r) - _pref_min(q) for r, q in pair])
            e["tcp_mean_deviation_from_control_m"] = boot_ci([float(np.linalg.norm(
                np.array(r["tcp"])[:min(len(r["tcp"]), len(q["tcp"]))] -
                np.array(q["tcp"])[:min(len(r["tcp"]), len(q["tcp"]))], axis=1).mean()) for r, q in pair])
            if "approached_first" in pair[0][0]:
                d = [_pref_dir(r) - _pref_dir(q) for r, q in pair if _pref_dir(r) is not None and
                     _pref_dir(q) is not None]
                e["init_dir_pref_vs_control"] = boot_ci(d)
                e["approached_first_changed_to_new"] = sum(r["approached_first"] == "distractor0" and
                                                           q["approached_first"] != "distractor0" for r, q in pair)
                e["approached_first_changed_from_new"] = sum(r["approached_first"] != "distractor0" and
                                                             q["approached_first"] == "distractor0" for r, q in pair)
                e["first_contact_changed_to_new"] = sum(r["first_contact"] == "distractor0" and
                                                        q["first_contact"] != "distractor0" for r, q in pair)
                g = [(_goal_pref(r) or 0.0) - (_goal_pref(q) or 0.0) for r, q in pair
                     if _goal_pref(r) is not None or _goal_pref(q) is not None]
                e["goal_pref_vs_control_m"] = boot_ci(g)
        out[c] = e
    # effect of the valid edit beyond the irrelevant edits (paired per scene; difference of paired differences)
    byc = {c: {(r["robot"], r["seed"]): r for r in ok if r["condition"] == c} for c in {r["condition"] for r in ok}}
    con = {}
    for valid, metric in (("rebind_obj", _pref_min), ("rebind_obj", _pref_dir), ("goal_shift", _goal_pref)):
        for ctl in ("irrelevant_distractor", "orthogonal_matched", "control_replay"):
            if valid not in byc or ctl not in byc:
                continue
            ks = sorted(set(byc[valid]) & set(byc[ctl]) & set(ctrl))
            d = []
            for k_ in ks:
                v, w, q = metric(byc[valid][k_]), metric(byc[ctl][k_]), metric(ctrl[k_])
                if v is None and w is None:
                    continue
                d.append(((v or 0.0) - (q or 0.0)) - ((w or 0.0) - (q or 0.0)))
            con[f"{valid}-{ctl}:{metric.__name__.strip('_')}"] = boot_ci(d)
    out["_contrasts"] = con
    return out


# ================================================================== manipulator assignment (dual-arm assign scenes)
ARM_CONDITIONS = ("control", "swap_arm", "swap_slots", "orthogonal_matched", "control_replay")
"""Arm edits on the paired manipulator-assignment scenes (research/pairs/assign_pick_place_v1.json, dev split):
  control             task as given: actor of take/place = the assigned arm A.
  swap_arm            VALID context edit: the actor of take/place is rebound to the other arm B (the task graph of
                      the pair's other variant; physics unchanged). The oracle's teacher demonstrates B.
                      Prediction: arm B moves to the bar and touches it; arm A stays staged.
  swap_slots          packet-level edit: the control packet with its two assembly slots' z exchanged (handles
                      unchanged). Prediction if meaning is slot-addressed: as swap_arm.
  orthogonal_matched  CONTROL: control z + r, ||r|| = ||z_swap_arm - z_control|| per packet, r orthogonal to all
                      probe readout gradients. Prediction: A still acts.
  control_replay      CONTROL (generated route only): unedited, different flow noise."""


def teacher_state(t):
    import copy
    st = {k: copy.deepcopy(v) for k, v in vars(t).items() if k not in ("s", "arms")}
    st["_arms"] = {e: a.state() for e, a in t.arms.items()}
    return st


def teacher_load(t, st):
    import copy
    for k, v in st.items():
        if k != "_arms":
            setattr(t, k, copy.deepcopy(v))
    for e, a in t.arms.items():
        a.load(st["_arms"][e])


class _DualTeacher:
    """Wrapper giving dual teachers the state()/load() interface used by OracleSource."""

    def __init__(self, t):
        self.t = t

    def act(self):
        return self.t.act()

    def state(self):
        return teacher_state(self.t)

    def load(self, st):
        teacher_load(self.t, st)

    @property
    def done(self):
        return getattr(self.t, "done", False)


def _dual_teacher(s, arm):
    """AssignedPickPlaceTeacher for actor `arm`, constructed inside that task context (then restored)."""
    from rrp.control.dual_teachers import AssignedPickPlaceTeacher
    sn = s.snapshot()
    try:
        if arm != s._sem_assigned:
            rebind_actor(s, arm)
        t = AssignedPickPlaceTeacher(s)
    finally:
        s.restore(sn)
    return _DualTeacher(t)


def _dual_system0(R, f, lsv, rcv, dev):
    from rrp.evaluation.dual_latent_eval import DualLatentSystem0
    return DualLatentSystem0(R, f, latent_space_version=lsv, realizer_compat_version=rcv, device=dev)


def run_arm_condition(src, R, P, pair, seed, cond, *, max_steps=400, replan=8, dev="cpu"):
    import mujoco
    from rrp.control.dual_validate import make_session
    arm = "left" if seed % 2 == 0 else "right"          # assigned arm alternates with the seed (balanced)
    other = "right" if arm == "left" else "left"
    s = make_session(f"assign_{arm}", pair, seed)
    s._sem_dual, s._sem_assigned = True, arm
    base = dict(pair=pair, seed=seed, condition=cond, assigned=arm, edited_to=other, route=src.label)
    if cond == "control_replay" and src.label != "generated":
        return dict(base, skipped=f"n/a_deterministic_{src.label}")
    teachers = {}
    if src.label in ("oracle", "teacher"):
        need = {"orthogonal_matched": ("control", "swap_arm"), "swap_slots": ("control",)}.get(cond, (cond,))
        for c in need:
            t = _dual_teacher(s, other if c == "swap_arm" else arm)
            if not t.t.feasibility()["feasible"]:
                return dict(base, skipped=f"infeasible_{c}")
            teachers[c] = t
    if src.label == "teacher" and cond in ("swap_slots", "orthogonal_matched"):
        return dict(base, skipped="n/a_for_teacher")
    f = src.featurizer(s)
    s0 = None if src.label == "teacher" else _dual_system0(R, f, src.lsv, src.rcv, dev)
    bar = s.model.body("bar").id
    bar0 = s.data.xpos[bar].copy()
    sites = {e: mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_SITE, h.tcp_site) for e, h in s.handles.items()}
    rb_arm = {}
    for e, h in s.handles.items():
        for b, ri in robot_body_ids(s, h.robot).items():
            rb_arm[b] = e
    tr = {e: [s.data.site_xpos[i].copy()] for e, i in sites.items()}
    fcon = {e: None for e in sites}
    lift, info, calls, zlog = 0.0, [], 0, []
    for step in range(max_steps):
        if src.label == "teacher":
            s.step(teachers[cond].act())
        else:
            if step % replan == 0 or s0.packet is None:
                key = _key(seed, calls) + (500 if cond == "control_replay" else 0)
                calls += 1
                if cond in ("orthogonal_matched", "swap_slots"):
                    pc = src.packet(s, "control", teachers.get("control"), None, key)
                    if cond == "swap_slots":
                        p = deliver(pc, s, np.ascontiguousarray(pc.z[:, ::-1]), "swap_slots")
                    else:
                        pg = src.packet(s, "swap_arm", teachers.get("swap_arm"), None, key)
                        nrm = float(np.linalg.norm(pg.z - pc.z))
                        r, inf = probe_orthogonal_dual(P, pc.z, len(s.detectables), key, nrm)
                        p = deliver(pc, s, pc.z + r, "orthogonal_matched")
                        info.append(dict(norm=nrm, rel_norm=nrm / float(np.linalg.norm(pc.z)), **inf))
                else:
                    c = "control" if cond == "control_replay" else cond
                    p = src.packet(s, c, teachers.get(c), None, key)
                if len(zlog) < 4:
                    zlog.append(float(np.linalg.norm(p.z)))
                _recv(s0, p, s)
            s.step(s0.tick(s))
            for t in teachers.values():
                t.act()
        for e, i in sites.items():
            tr[e].append(s.data.site_xpos[i].copy())
        for c in range(s.data.ncon):
            con = s.data.contact[c]
            b1, b2 = int(s.model.geom_bodyid[con.geom1]), int(s.model.geom_bodyid[con.geom2])
            for x, y in ((b1, b2), (b2, b1)):
                if x == bar and y in rb_arm and fcon[rb_arm[y]] is None:
                    fcon[rb_arm[y]] = step
        lift = max(lift, float(s.data.xpos[bar][2] - bar0[2]))
        if s.runtime.succeeded():
            break
        if src.label == "teacher" and teachers[cond].done:
            break
    T = {e: np.array(v) for e, v in tr.items()}
    path = {e: float(np.linalg.norm(np.diff(v, axis=0), axis=1).sum()) for e, v in T.items()}
    disp = {e: float(np.linalg.norm(v - v[0], axis=1).max()) for e, v in T.items()}
    mind = {e: float(np.linalg.norm(v - bar0, axis=1).min()) for e, v in T.items()}
    fc = {e: t for e, t in fcon.items() if t is not None}
    tot = path["left"] + path["right"]
    row = dict(base, steps=step + 1, system_i_calls=calls, privileged_success=bool(s.privileged_success()),
               packets_rejected=s0.stats.rejected if s0 else 0, path_m=path, max_disp_m=disp, min_dist_bar_m=mind,
               first_contact_tick=fcon, first_contact_arm=min(fc, key=fc.get) if fc else None,
               moving_arm=max(path, key=path.get), bar_lift_m=lift,
               right_path_share=path["right"] / tot if tot > 0 else None,
               bar_pref_right_m=mind["left"] - mind["right"], edit_info=info[:3], packet_z_norm=zlog,
               bar_moved_m=float(np.linalg.norm(s.data.xpos[bar][:2] - bar0[:2])),
               tcp={e: v[::8].round(4).tolist() for e, v in T.items()})
    exp = other if cond in ("swap_arm", "swap_slots") else arm
    row["expected_arm"] = exp
    row["followed"] = bool(row["moving_arm"] == exp and (row["first_contact_arm"] in (None, exp)))
    return row


def probe_orthogonal_dual(P, z, n_ent, key, norm):
    """As probe_orthogonal, over all packet slots (both assemblies)."""
    zt = torch.tensor(z[None], dtype=torch.float32, requires_grad=True)
    dev = next(P.parameters()).device
    out = P(zt.to(dev), torch.ones(1, z.shape[1], dtype=torch.bool, device=dev), n_ent)
    M = z.shape[1]
    feats = [out["focused_on"][0, :, 0], out["visible"][0, :, 0]]
    for m in range(M):
        feats += [out["held_by"][0, :, m, 0], out["acting_on"][0, :, m, 0], out["rel_pos"][0, :, m, :3].reshape(-1)]
    feats += [out["subtask"].reshape(-1)]
    y = torch.cat(feats)
    rows = []
    for i in range(len(y)):
        g, = torch.autograd.grad(y[i], zt, retain_graph=True, allow_unused=True)
        if g is not None:
            rows.append(g.reshape(-1).numpy())
    J = np.stack(rows)
    Q, _ = np.linalg.qr(J.T)
    r = np.random.default_rng(key).standard_normal(z.size)
    r = r - Q @ (Q.T @ r)
    r = r / max(np.linalg.norm(r), 1e-12) * norm
    return r.reshape(z.shape).astype(np.float32), dict(probe_span_dim=int(np.linalg.matrix_rank(J)),
                                                      z_dim=int(z.size))


def arm_suite(src, R, P, pairs, seeds, conditions=ARM_CONDITIONS, *, max_steps=400, dev="cpu", log=print,
              out_path: Path | None = None):
    rows = []
    for pair in pairs:
        for sd in seeds:
            for c in conditions:
                t0 = time.time()
                r = run_arm_condition(src, R, P, pair, sd, c, max_steps=max_steps, dev=dev)
                r["wall_s"] = round(time.time() - t0, 1)
                rows.append(r)
                if out_path:
                    with open(out_path, "a") as fh:
                        fh.write(json.dumps(r) + "\n")
                log(f"[arm] {src.label} {pair} seed {sd} {c}: " + (r.get("skipped") or
                    f"assigned={r['assigned']} expected={r['expected_arm']} moving={r['moving_arm']} "
                    f"first_contact={r['first_contact_arm']} path={ {k: round(v, 2) for k, v in r['path_m'].items()} } "
                    f"min_d={ {k: round(v, 3) for k, v in r['min_dist_bar_m'].items()} } success={r['privileged_success']}"),
                    flush=True)
    return rows


def summarize_arm(rows):
    """Arm preference is signed toward the arm the EDIT names (the other arm): for each episode
    pref = share of TCP path travelled by the edited-to arm; bar_pref = how much closer the edited-to arm came."""
    from rrp.evaluation.statistics import wilson
    from rrp.evaluation.latent_causal import boot_ci
    ok = [r for r in rows if "skipped" not in r]
    share = lambda r: (r["right_path_share"] if r["edited_to"] == "right" else 1 - r["right_path_share"]) \
        if r["right_path_share"] is not None else None
    bpref = lambda r: r["min_dist_bar_m"][r["assigned"]] - r["min_dist_bar_m"][r["edited_to"]]
    ctrl = {(r["pair"], r["seed"]): r for r in ok if r["condition"] == "control"}
    out = {}
    for c in sorted({r["condition"] for r in ok}):
        rc = [r for r in ok if r["condition"] == c]
        n = len(rc)
        mv = sum(r["moving_arm"] == r["edited_to"] for r in rc)
        fc = sum(r["first_contact_arm"] == r["edited_to"] for r in rc)
        e = dict(n=n, followed=sum(r["followed"] for r in rc),
                 moving_arm_is_edited_to=dict(k=mv, n=n, wilson95=wilson(mv, n)),
                 first_contact_is_edited_to=dict(k=fc, n=n, wilson95=wilson(fc, n)),
                 first_contact_none=sum(r["first_contact_arm"] is None for r in rc),
                 privileged_success=sum(r["privileged_success"] for r in rc),
                 edited_to_path_share=boot_ci([share(r) for r in rc]),
                 bar_pref_edited_to_m=boot_ci([bpref(r) for r in rc]),
                 skipped=sum(1 for r in rows if r["condition"] == c and "skipped" in r))
        pair = [(r, ctrl[(r["pair"], r["seed"])]) for r in rc if (r["pair"], r["seed"]) in ctrl]
        if pair and c != "control":
            e["paired_n"] = len(pair)
            e["path_share_vs_control"] = boot_ci([share(r) - share(q) for r, q in pair
                                                  if share(r) is not None and share(q) is not None])
            e["bar_pref_vs_control_m"] = boot_ci([bpref(r) - bpref(q) for r, q in pair])
            e["moving_arm_changed"] = sum(r["moving_arm"] != q["moving_arm"] for r, q in pair)
        out[c] = e
    byc = {c: {(r["pair"], r["seed"]): r for r in ok if r["condition"] == c} for c in {r["condition"] for r in ok}}
    con = {}
    for valid in ("swap_arm", "swap_slots"):
        for ctl in ("orthogonal_matched", "control_replay"):
            if valid in byc and ctl in byc:
                ks = sorted(set(byc[valid]) & set(byc[ctl]))
                con[f"{valid}-{ctl}:path_share"] = boot_ci([share(byc[valid][k]) - share(byc[ctl][k]) for k in ks
                                                            if share(byc[valid][k]) is not None and
                                                            share(byc[ctl][k]) is not None])
                con[f"{valid}-{ctl}:bar_pref_m"] = boot_ci([bpref(byc[valid][k]) - bpref(byc[ctl][k]) for k in ks])
    out["_contrasts"] = con
    return out
