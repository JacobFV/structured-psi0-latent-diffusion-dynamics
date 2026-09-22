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

from rrp.control.teachers import PickPlaceTeacher
from rrp.data.features import Featurizer
from rrp.sim.native import Session
from rrp.sim.sensors import camera_visibility

FEATURIZER_VERSION = "feat-v2"


def featurizer_for(session: Session, robot: int = 0) -> Featurizer:
    mr = session.scenario.robots[robot]
    return Featurizer(session.model, mr.robot_spec, mr.prefix, mr.meta, mr.base_pos, mr.base_yaw,
                      mr.manipulator_bindings, robot_index=robot)


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


@dataclass
class EpisodeRecord:
    public: dict
    private: dict


def collect_teacher_episode(session: Session, teacher_cls=PickPlaceTeacher, max_steps: int = 600,
                            episode_id: str = "", split_lineage: dict | None = None,
                            exec_noise: float = 0.0, noise_seed: int = 0) -> EpisodeRecord:
    """exec_noise > 0 (DART): executed ARM command = teacher command + N(0, exec_noise) held for a few
    steps; the recorded LABEL is always the clean teacher command, so data covers recovery states."""
    feat = featurizer_for(session)
    teacher = teacher_cls(session)
    f = teacher.feasibility() if hasattr(teacher, "feasibility") else {"feasible": True}
    t0 = time.time()
    inputs, actions, labels, phases, statuses, q0s = [], [], [], [], [], []
    obs = session.observe()
    prev = None
    status = "infeasible" if not f["feasible"] else "running"
    nrng = np.random.default_rng([noise_seed, 7])
    nz = None
    steps = 0
    if f["feasible"]:
        for k in range(max_steps):
            pi = feat(obs, prev)
            cmd = teacher.act()
            a = feat.aspace.normalize([cmd.groups], pi.q0)[0]
            inputs.append(pi)
            actions.append(cmd.groups)
            if exec_noise > 0:
                if k % 5 == 0:
                    nz = nrng.normal(0, exec_noise, len(cmd.groups["arm"]))
                g = feat.aspace
                arm_lo = [lo for lo, grp in zip(g.lower, g.node_group) if grp == "arm"]
                arm_hi = [hi for hi, grp in zip(g.upper, g.node_group) if grp == "arm"]
                noisy = np.clip(np.array(cmd.groups["arm"]) + nz, arm_lo, arm_hi)
                cmd = cmd.model_copy(update={"groups": dict(cmd.groups, arm=noisy.tolist())})
            q0s.append(pi.q0)
            labels.append(privileged_labels(session, feat))
            phases.append(teacher.phase)
            statuses.append({e: v.status for e, v in session.runtime.instances.items()})
            res = session.step(cmd)
            obs = res.observation
            prev = a
            steps += 1
            if teacher.done:
                break
        session.step(None)
        status = "success" if session.privileged_success() else "failure"
    rs = session.scenario.robots[0].robot_spec
    meta = dict(episode_id=episode_id, robot=rs.name, spec_hash=rs.spec_hash, lineage=rs.lineage,
                controller_version=session.robots[0].controller.version, task=session.scenario.name,
                task_hash=hashlib.sha256(json.dumps(session.scenario.task, sort_keys=True).encode()).hexdigest()[:16],
                seed=session.seed, control_dt=session.dt, physics_dt=float(session.model.opt.timestep),
                steps=steps, status=status, feasibility=f, source="scripted_teacher", privileged_teacher=True,
                public_runtime_success=bool(session.runtime.succeeded()), featurizer=FEATURIZER_VERSION,
                wall_s=time.time() - t0, split_lineage=split_lineage or {}, exec_noise=exec_noise,
                n_distractors=session.scenario.meta.get("n_distractors", 0))
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


def read_episode(path: Path) -> dict:
    return pickle.loads(gzip.decompress(Path(path).read_bytes()))
