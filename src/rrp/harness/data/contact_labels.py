"""Privileged per-tick contact-frame recorder (W12; LABELS and evaluation diagnostics only, never policy inputs).

`ContactFrameRecorder(session)` records, per control tick, from simulator truth:
  TCP pose of every task manipulator (site pose), pose of every detectable object, per (a, b) pair contact flag, mean
  contact point and mean contact normal (unit, pointing from b into a), and held_by truth.
Pairs: (manipulator, object) for every manipulator and object, then (object, object) for every object pair.
`recording()` returns an rrp.data.contact_segments.ContactRecording.

Read-only: it never writes to the simulation. Works for rrp.envs.dual.DualSession and the native arm Session (both
expose manip_map, robots[..].tcp_sites, detectables and truth()).
"""
from __future__ import annotations

import mujoco
import numpy as np

from rrp.harness.data.contact_segments import ContactRecording

CONTACT_LABEL_VERSION = "rrp.data.contact_labels/v1"


class ContactFrameRecorder:
    def __init__(self, session):
        s = self.s = session
        m = s.model
        self.manips = list(s.manip_map)
        self.objects = [o.sim_body for o in s.detectables]
        self.tcp_sid = []
        body_owner = {}
        for ent in self.manips:
            ri, asm = s.manip_map[ent]
            r = s.robots[ri]
            self.tcp_sid.append(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, r.tcp_sites[asm]))
            asm_spec = next(a for a in r.spec.assemblies if a.id == asm)
            for l in r.spec.links:
                if l.address in asm_spec.members:
                    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, l.name)
                    if bid >= 0:
                        body_owner[bid] = ent
        self.obj_bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o) for o in self.objects]
        for o, bid in zip(self.objects, self.obj_bid):
            body_owner[bid] = o
        self.body_owner = body_owner
        self.pairs = [(a, o) for a in self.manips for o in self.objects] + \
                     [(self.objects[i], self.objects[j]) for i in range(len(self.objects))
                      for j in range(i + 1, len(self.objects))]
        self.pair_idx = {p: i for i, p in enumerate(self.pairs)}
        self.rows = dict(tcp_pos=[], tcp_R=[], obj_pos=[], obj_R=[], contact=[], contact_point=[], contact_normal=[],
                         held=[])

    def tick(self):
        s, m, d = self.s, self.s.model, self.s.data
        C = len(self.pairs)
        self.rows["tcp_pos"].append(np.array([d.site_xpos[i] for i in self.tcp_sid]))
        self.rows["tcp_R"].append(np.array([d.site_xmat[i].reshape(3, 3) for i in self.tcp_sid]))
        self.rows["obj_pos"].append(np.array([d.xpos[b] for b in self.obj_bid]).reshape(-1, 3))
        self.rows["obj_R"].append(np.array([d.xmat[b].reshape(3, 3) for b in self.obj_bid]).reshape(-1, 3, 3))
        flag = np.zeros(C, bool)
        pt = np.zeros((C, 3))
        nrm = np.zeros((C, 3))
        cnt = np.zeros(C)
        for i in range(d.ncon):
            c = d.contact[i]
            o1 = self.body_owner.get(int(m.geom_bodyid[c.geom1]))
            o2 = self.body_owner.get(int(m.geom_bodyid[c.geom2]))
            if o1 is None or o2 is None or o1 == o2:
                continue
            n = np.array(c.frame[:3])                       # from geom1 to geom2
            if (o1, o2) in self.pair_idx:                   # pair (a=o1, b=o2): want b -> a = -n
                k, n = self.pair_idx[(o1, o2)], -n
            elif (o2, o1) in self.pair_idx:                 # pair (a=o2, b=o1): b -> a = n
                k = self.pair_idx[(o2, o1)]
            else:
                continue
            flag[k] = True
            pt[k] += c.pos
            nrm[k] += n
            cnt[k] += 1
        pt[cnt > 0] /= cnt[cnt > 0, None]
        nn = np.linalg.norm(nrm, axis=1)
        nrm[nn > 1e-9] /= nn[nn > 1e-9, None]
        self.rows["contact"].append(flag)
        self.rows["contact_point"].append(pt)
        self.rows["contact_normal"].append(nrm)
        held = s.truth().held_by
        self.rows["held"].append(np.array([[o in held.get(ent, []) for ent in self.manips] for o in self.objects],
                                          bool).reshape(len(self.objects), len(self.manips)))

    def recording(self) -> ContactRecording:
        r = {k: np.asarray(v) for k, v in self.rows.items()}
        return ContactRecording(dt=float(self.s.dt), manipulators=list(self.manips), tcp_pos=r["tcp_pos"],
                                tcp_R=r["tcp_R"], objects=list(self.objects), obj_pos=r["obj_pos"], obj_R=r["obj_R"],
                                pairs=list(self.pairs), contact=r["contact"].astype(bool),
                                contact_point=r["contact_point"], contact_normal=r["contact_normal"],
                                held=r["held"].astype(bool))
