"""Teacher data collection. Public featurized inputs and native actions go to *.public.pkl;
privileged labels (truth poses, held_by truth, contacts, visibility truth, completion truth,
teacher phase) go to *.private.pkl. Failed and infeasible attempts are recorded too."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from rrp.policies.base import Act
from rrp.policies.teachers.arm import PickPlaceTeacher
from rrp.policies.features.featurizer import Featurizer, featurizer_for  # noqa: F401  (featurizer_for moved to rrp.policies.features)
from rrp.envs.mujoco.session import Session
from rrp.envs.mujoco.sensors import camera_visibility
from rrp.tasks.spec import Judgement, TaskSpec

from rrp.core.provenance import FEATURIZER_VERSION, physics_provenance  # noqa: E402,F401  (single constant; alias kept)


def privileged_labels(session: Session, feat: Featurizer) -> dict:
    """Training/evaluation-only labels at the current step (never policy inputs)."""
    m, d = session.model, session.data
    t = session.truth()
    held = t.held_by
    slots = [o.sim_body for o in session.detectables]
    manip_ids = list(session.manip_map)
    tcp = {}
    for ent in manip_ids:
        ri, asm = session.manip_map[ent]
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, session.robots[ri].tcp_sites[asm])
        tcp[ent] = d.site_xpos[sid].copy()
    lab = dict(slot_pos=np.zeros((len(slots), 3), np.float32), slot_held=np.zeros((len(slots), len(manip_ids)), bool),
               slot_contact=np.zeros((len(slots), len(manip_ids)), bool), slot_visible=np.zeros(len(slots), bool),
               slot_rel_tcp=np.zeros((len(slots), len(manip_ids), 3), np.float32),
               slot_gaze_angle=np.zeros(len(slots), np.float32), tcp_pos=np.zeros((len(manip_ids), 3), np.float32))
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "front")
    cpos, cax = d.cam_xpos[cid], -d.cam_xmat[cid].reshape(3, 3)[:, 2]
    for i, o in enumerate(slots):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o)
        p = d.xpos[bid].copy()
        lab["slot_pos"][i] = feat._to_base(p)
        lab["slot_visible"][i] = camera_visibility(m, d, "front", o, p)
        v = p - cpos
        lab["slot_gaze_angle"][i] = math.degrees(math.acos(np.clip(v @ cax / np.linalg.norm(v), -1, 1)))
        for k, ent in enumerate(manip_ids):
            lab["slot_held"][i, k] = o in held.get(ent, [])
            touching = {c.body_a if c.body_b == o else c.body_b for c in t.contacts if o in (c.body_a, c.body_b)}
            ri, asm = session.manip_map[ent]
            asm_spec = next(a for a in session.robots[ri].spec.assemblies if a.id == asm)
            hand = {l.name for l in session.robots[ri].spec.links if l.address in asm_spec.members}
            lab["slot_contact"][i, k] = bool(touching & hand)
            lab["slot_rel_tcp"][i, k] = (p - tcp[ent]).astype(np.float32)
    for k, ent in enumerate(manip_ids):
        lab["tcp_pos"][k] = feat._to_base(tcp[ent])
    lab["event_completion_truth"] = dict(t.event_completion_truth)
    lab["predicates_truth"] = dict(t.predicates)
    return lab


DART_FREE_PHASES = ("pregrasp", "transport")      # D-118 phase-gated DART: hover approach and carry at 18 cm
# D-126 #5 (D-121 fallback): small-amplitude noise in the final descent, behind the kinematic proximity guard. Only with
# dart_safety "phase" and dart_descent_sigma > 0 (default 0 = the v6dart behaviour, unchanged).
DART_DESCENT_PHASES = ("descend",)


class DartProximityGuard:
    """D-118 contact-safe DART: before a perturbed arm command is executed, forward kinematics on the COMMANDED joints
    (other joints and objects at their current state) gives the robot geom poses; the noisy command is rejected for this
    tick (the clean command executes) if any robot geom would come within `margin` of the task cube or the table AND
    closer than the clean command would bring it (so contacts the clean teacher makes itself, e.g. the grasp or a held
    cube, do not suppress the noise). Uses a private MjData; never touches the simulation."""

    def __init__(self, session, margin: float = 0.005, obj: str = "cube", strict: bool = False):
        m = session.model
        self.s, self.m, self.margin, self.strict = session, m, float(margin), bool(strict)
        self.d = mujoco.MjData(m)
        r = session.robots[0]
        names = {l.name for l in r.spec.links}
        self.qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in r.arm_joints])
        self.robot_geoms = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in names
                            and (m.geom_contype[g] or m.geom_conaffinity[g])]
        self.cube = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, obj)
        self.cube_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.cube]
        self.table = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table")
        asm = next(a for a in r.spec.assemblies if a.kind in ("gripper", "hand"))
        hand = {l.name for l in r.spec.links if l.address in asm.members}
        self.hand_bodies = {b for b in range(m.nbody) if m.body(b).name in hand}
        self.tcp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, r.tcp_sites[asm.id])
        self._ft = np.zeros(6)
        # robot geoms that rest on / next to the table at the start (base, mount) are not checked against the table
        self.table_geoms = [a for a in self.robot_geoms if self.table < 0 or
                            mujoco.mj_geomDistance(m, session.data, a, self.table, 0.05, self._ft) > 0.02]

    def _held(self) -> bool:
        """Current state: the cube is in contact with >= 2 hand bodies (then it moves with the hand)."""
        d, m = self.s.data, self.m
        touching = set()
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 == self.cube and b2 in self.hand_bodies:
                touching.add(b2)
            elif b2 == self.cube and b1 in self.hand_bodies:
                touching.add(b1)
        if len(touching) < 2:
            return False
        # carried only once it is off the table: while it rests on the table the hand moves relative to it (grasping)
        for g in self.cube_geoms:
            if self.table >= 0 and mujoco.mj_geomDistance(m, d, g, self.table, 0.01, self._ft) > 0.003:
                return True
        return False

    def _min_dist(self, q_arm, held: bool) -> float:
        m, d = self.m, self.d
        d.qpos[:] = self.s.data.qpos
        d.qpos[self.qadr] = q_arm
        if held:             # a held cube moves rigidly with the hand: shift it by the commanded TCP displacement
            mujoco.mj_kinematics(m, d)
            dp = d.site_xpos[self.tcp] - self.s.data.site_xpos[self.tcp]
            ja = m.body_jntadr[self.cube]
            qa = m.jnt_qposadr[ja]
            d.qpos[qa:qa + 3] = self.s.data.qpos[qa:qa + 3] + dp
        mujoco.mj_kinematics(m, d)
        cap = self.margin + 0.05
        best = np.inf
        pairs = [(a, self.table) for a in self.table_geoms if self.table >= 0]
        if held:             # the carried cube must not be driven into the table
            pairs += [(c, self.table) for c in self.cube_geoms if self.table >= 0]
        else:
            pairs += [(a, c) for a in self.robot_geoms for c in self.cube_geoms]
        for a, b in pairs:
            best = min(best, mujoco.mj_geomDistance(m, d, a, b, cap, self._ft))
        return best

    def unsafe(self, clean, noisy) -> bool:
        held = self._held()
        dn = self._min_dist(noisy, held)
        if dn >= self.margin:
            return False
        return dn < self._min_dist(clean, held) - 1e-4

    def choose(self, clean, noisy, last):
        """Executed arm command for a DART tick: 'noisy' if safe; else 'clean'; and if the chosen command itself would
        drive a robot geom > 1 mm into the cube/table (after earlier perturbations displaced the cube, the teacher's own
        plan can do that), hold the last executed command instead. Returns (kind, command)."""
        held = self._held()
        dn = self._min_dist(noisy, held)
        if self.strict:      # no perturbation while the clean command itself is within the margin of the cube/table
            dc = self._min_dist(clean, held)
            ok = dn >= self.margin and dc >= self.margin
        else:
            dc = dn if dn >= self.margin else self._min_dist(clean, held)
            ok = dn >= self.margin or dn >= dc - 1e-4
        kind, q, dq = ("noisy", noisy, dn) if ok else ("clean", clean, dc)
        if dq < -0.001 and last is not None:
            dl = self._min_dist(last, held)
            if dl > dq:
                return "hold", last
        return kind, q


@dataclass
class EpisodeRecord:
    public: dict
    private: dict


class _TeacherTrace:
    """Rollout hook of `collect_teacher_episode`: per tick the featurized public input (previous-command feature = the
    normalized CLEAN teacher command), the clean teacher command as the label, the privileged labels, the teacher phase
    and the task-event statuses; the DART execution perturbation (on_act: the executed command differs, the recorded
    label never does) and the W6 motion recorder. `dart_stats` / `mrec` are read by the caller after the episode."""

    def __init__(self, session, teacher, feat, *, exec_noise, noise_seed, dart_safety, dart_descent_sigma):
        self.session, self.teacher, self.feat = session, teacher, feat
        self.exec_noise, self.descent_sigma = exec_noise, dart_descent_sigma
        self.inputs, self.actions, self.labels, self.phases, self.statuses, self.q0s = [], [], [], [], [], []
        self.prev, self.k, self.nz, self.last_exec = None, 0, None, None
        self.nrng = np.random.default_rng([noise_seed, 7])
        self.dart_stats = dict(mode=dart_safety or "none", ticks_by_phase={}, applied_by_phase={}, rejected_by_phase={},
                               held_by_phase={})
        self.ds_mode, _, ds_arg = (dart_safety or "").partition(":")      # "proximity" or "proximity:<margin_m>"
        self.guard = (DartProximityGuard(session, margin=float(ds_arg) if ds_arg else 0.005,
                                         strict=self.ds_mode == "proximity_strict")
                      if (exec_noise > 0 and self.ds_mode in ("proximity", "proximity_strict")) else None)
        if self.guard is not None:
            self.dart_stats["margin_m"] = self.guard.margin
        if self.ds_mode == "phase":
            self.dart_stats["free_phases"] = list(DART_FREE_PHASES)
        self.dguard = None
        if dart_descent_sigma and dart_descent_sigma > 0:
            if self.ds_mode != "phase":
                raise ValueError("dart_descent_sigma needs dart_safety 'phase' (D-121 phase-gated DART)")
            if exec_noise > 0:
                self.dguard = DartProximityGuard(session, margin=0.005, strict=False)
                self.dart_stats["descent"] = dict(sigma=float(dart_descent_sigma), phases=list(DART_DESCENT_PHASES),
                                                  margin_m=self.dguard.margin, guard="proximity")
        from rrp.envs.mujoco.motion_quality import ArmMotionRecorder
        # W6 gates: read-only, the CLEAN label is recorded
        self.mrec = ArmMotionRecorder(session, boundary_kind="phase_switch")

    def on_act(self, i, obs, act):
        session, teacher, feat, cmd, k = self.session, self.teacher, self.feat, act.command, self.k
        pi = feat(obs, self.prev)
        self.a = feat.aspace.normalize([cmd.groups], pi.q0)[0]
        self.inputs.append(pi)
        self.actions.append(cmd.groups)
        out = None
        if self.exec_noise > 0:
            if k % 5 == 0:
                self.nz = self.nrng.normal(0, self.exec_noise, len(cmd.groups["arm"]))
            g, nz, stats = feat.aspace, self.nz, self.dart_stats
            arm_lo = [lo for lo, grp in zip(g.lower, g.node_group) if grp == "arm"]
            arm_hi = [hi for hi, grp in zip(g.upper, g.node_group) if grp == "arm"]
            noisy = np.clip(np.array(cmd.groups["arm"]) + nz, arm_lo, arm_hi)
            ph = teacher.phase
            stats["ticks_by_phase"][ph] = stats["ticks_by_phase"].get(ph, 0) + 1
            if self.ds_mode == "phase":   # D-118 fallback: perturb only in the free-space phases
                kind, q_exec = ("noisy", noisy) if ph in DART_FREE_PHASES else ("clean", None)
                if self.dguard is not None and ph in DART_DESCENT_PHASES:     # D-126 #5: small, guarded descent noise
                    clean_arm = np.array(cmd.groups["arm"])
                    small = np.clip(clean_arm + nz * (self.descent_sigma / self.exec_noise), arm_lo, arm_hi)
                    kind, q_exec = self.dguard.choose(clean_arm, small, self.last_exec)
            else:
                kind, q_exec = ("noisy", noisy) if self.guard is None else self.guard.choose(
                    np.array(cmd.groups["arm"]), noisy, self.last_exec)
            key = dict(noisy="applied_by_phase", clean="rejected_by_phase", hold="held_by_phase")[kind]
            stats.setdefault(key, {})
            stats[key][ph] = stats[key].get(ph, 0) + 1
            if kind != "clean":      # execute the perturbed (or held) command; the recorded label stays clean
                cmd = cmd.model_copy(update={"groups": dict(cmd.groups, arm=np.asarray(q_exec).tolist())})
                out = Act(cmd, act.packet, act.chunk, act.info)
            self.last_exec = np.array(cmd.groups["arm"])
        self.q0s.append(pi.q0)
        self.labels.append(privileged_labels(session, feat))
        self.phases.append(teacher.phase)
        self.statuses.append({e: v.status for e, v in session.runtime.instances.items()})
        return out

    def on_step(self, i, env, act, step):
        self.mrec.tick(self.actions[-1], self.k > 0 and self.phases[-1] != self.phases[-2])
        self.prev = self.a
        self.k += 1


class _TeacherEnd:
    """Rollout hook: the episode ends when the scripted teacher's FSM is done (its own end rule); `settle` hold ticks
    then re-read the privileged verdict. (harness.data cannot import harness.eval.hooks: eval imports data; the
    generic EndWhen / Settle / budget_task there are the same hooks.)"""

    def __init__(self, teacher, settle: int = 0):
        self.teacher, self.settle = teacher, settle

    def on_step(self, i, env, act, step):
        return Judgement(True, "failure", "teacher_done") if self.teacher.done else None

    def on_end(self, i, env, ep):
        if ep.outcome == "crash":
            return {}
        for _ in range(self.settle):
            env.step(None)
        if self.settle:
            ep.success_privileged, ep.success_public = bool(env.privileged_success()), bool(env.runtime.succeeded())
        return {}


def run_teacher_rollout(session, teacher, *, version: str, hooks, max_steps: int, settle: int = 0):
    """One scripted-teacher episode as a `harness.rollout` (docs/architecture.md 14.1): the teacher acts as a privileged
    policy, `hooks` record / perturb, the episode ends with the teacher's FSM (or `max_steps`), then `settle` hold
    ticks. Returns the Episode; a crash raises."""
    from rrp.harness import rollout as R
    from rrp.policies.teachers import TeacherPolicy
    name, env_id = session.scenario.name, session.spec.env_id
    task = TaskSpec(name, {env_id: {}}, float("inf"),
                    lambda env, t, max_s: Judgement(t >= max_s, "timeout", "timeout" if t >= max_s else None),
                    note="budget only: end rules live in hooks")
    pol = TeacherPolicy(name, lambda e: teacher, version, ("joint_position", "gripper"))
    ep = R.rollout(lambda sd: session, pol, task, [session.seed], batch=1, max_steps=max_steps,
                   hooks=[*hooks, _TeacherEnd(teacher, settle)])[0]
    if ep.outcome == "crash":
        raise RuntimeError(ep.metrics.get("note") or ep.failure_reason)
    return ep


def collect_teacher_episode(session: Session, teacher_cls=PickPlaceTeacher, max_steps: int = 600,
                            episode_id: str = "", split_lineage: dict | None = None,
                            exec_noise: float = 0.0, noise_seed: int = 0,
                            teacher_version: str | None = None, dart_safety: str | None = None,
                            dart_descent_sigma: float = 0.0, teacher_kw: dict | None = None) -> EpisodeRecord:
    """exec_noise > 0 (DART): executed ARM command = teacher command + N(0, exec_noise) held for a few
    steps; the recorded LABEL is always the clean teacher command, so data covers recovery states.
    teacher_version (rrp.policies.teachers.arm_smooth.TEACHER_VERSIONS key) selects a registered arm teacher version;
    None keeps `teacher_cls` (the v1 default) and the historical meta.
    dart_descent_sigma > 0 (D-126 #5; needs dart_safety "phase"): in the DART_DESCENT_PHASES the executed arm command
    is clean + (dart_descent_sigma / exec_noise) x the SAME held noise draw (so the noise RNG stream is unchanged),
    passed through DartProximityGuard (5 mm, non-strict); episode meta dart.descent records sigma, phases and counts.
    teacher_kw: extra keyword arguments for the teacher version (e.g. ik_limit_margin for v2lim; recorded in meta)."""
    from rrp.policies.teachers.arm_smooth import make_arm_teacher, teacher_source, teacher_version_id
    feat = featurizer_for(session)
    if teacher_kw and not teacher_version:
        raise ValueError("teacher_kw needs a teacher_version")
    teacher = (make_arm_teacher(session, teacher_version, **(teacher_kw or {})) if teacher_version
               else teacher_cls(session))
    f = teacher.feasibility() if hasattr(teacher, "feasibility") else {"feasible": True}
    t0 = time.time()
    trace = _TeacherTrace(session, teacher, feat, exec_noise=exec_noise, noise_seed=noise_seed, dart_safety=dart_safety,
                          dart_descent_sigma=dart_descent_sigma)
    dart_stats, mrec = trace.dart_stats, trace.mrec
    status, steps = "infeasible", 0
    if f["feasible"]:
        # the episode is a harness.rollout of the scripted teacher (docs/architecture.md 14.1): the trace records and
        # executes the DART perturbation, the teacher's end ends the episode, one hold tick then the privileged verdict
        ep = run_teacher_rollout(session, teacher, version="collect", hooks=[trace], max_steps=max_steps, settle=1)
        status, steps = ("success" if ep.success_privileged else "failure"), ep.steps
    inputs, actions, labels, phases, statuses, q0s = (trace.inputs, trace.actions, trace.labels, trace.phases,
                                                      trace.statuses, trace.q0s)
    rs = session.scenario.robots[0].robot_spec
    meta = dict(episode_id=episode_id, robot=rs.name, spec_hash=rs.spec_hash, lineage=rs.lineage,
                controller_version=session.robots[0].controller.version, task=session.scenario.name,
                task_hash=hashlib.sha256(json.dumps(session.scenario.task, sort_keys=True).encode()).hexdigest()[:16],
                seed=session.seed, control_dt=session.dt, physics_dt=float(session.model.opt.timestep),
                steps=steps, status=status, feasibility=f, source="scripted_teacher", privileged_teacher=True,
                public_runtime_success=bool(session.runtime.succeeded()), featurizer=FEATURIZER_VERSION,
                wall_s=time.time() - t0, split_lineage=split_lineage or {}, exec_noise=exec_noise,
                n_distractors=session.scenario.meta.get("n_distractors", 0),
                physics=physics_provenance(session.model).to_dict())
    if exec_noise > 0 and dart_safety:   # D-118: contact-safe DART variant + its coverage (v1-default meta unchanged)
        meta["dart"] = dart_stats
    if teacher_version:                  # new selectable versions record themselves; v1-default meta is unchanged
        meta.update(source=teacher_source(teacher_version), teacher_version=teacher_version_id(teacher_version))
    if teacher_kw:
        meta["teacher_kw"] = dict(teacher_kw)
    if session.scenario.meta.get("object_spec"):     # D-126 #35 task-object variant (absent for the default scene)
        meta["object_spec"] = dict(session.scenario.meta["object_spec"])
    if f["feasible"]:
        meta["motion"] = mrec.summary()  # rrp.envs.motion_quality (W6 dataset gates, rrp.harness.eval.gates)
    public = dict(meta=meta, inputs=inputs, actions=actions, q0=q0s, statuses=statuses,
                  action_space=dict(node_group=feat.aspace.node_group, node_col=feat.aspace.node_col,
                                    lower=feat.aspace.lower, upper=feat.aspace.upper,
                                    is_gripper=feat.aspace.is_gripper, open_value=feat.aspace.open_value,
                                    closed_value=feat.aspace.closed_value, delta_scale=feat.aspace.delta_scale))
    private = dict(meta=dict(episode_id=episode_id, kind="privileged_labels"), labels=labels, phases=phases,
                   manipulators=list(session.manip_map), slots=[o.sim_body for o in session.detectables])
    return EpisodeRecord(public, private)


def write_episode(rec: EpisodeRecord, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    eid = rec.public["meta"]["episode_id"]
    pub = out_dir / f"{eid}.public.pkl.gz"
    prv = out_dir / f"{eid}.private.pkl.gz"
    hashes = {}
    for path, obj in ((pub, rec.public), (prv, rec.private)):
        data = gzip.compress(pickle.dumps(obj, protocol=5), compresslevel=3)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        hashes[path.name] = hashlib.sha256(data).hexdigest()[:16]
    return dict(rec.public["meta"], files=hashes)


# On-disk data written before the D-140 refactor pickled classes under their old module paths. This remap applies
# to reading existing files only (it is not a code alias); extend it when a pickled class moves.
LEGACY_PICKLE_MODULES = {"rrp.data.features": "rrp.policies.features.featurizer"}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(LEGACY_PICKLE_MODULES.get(module, module), name)


def load_pickle(fh_or_bytes) -> object:
    """pickle.load with LEGACY_PICKLE_MODULES applied (every dataset/DAgger pickle read goes through here)."""
    import io
    fh = io.BytesIO(fh_or_bytes) if isinstance(fh_or_bytes, (bytes, bytearray)) else fh_or_bytes
    return _Unpickler(fh).load()


def read_episode(path: Path) -> dict:
    return load_pickle(gzip.decompress(Path(path).read_bytes()))
