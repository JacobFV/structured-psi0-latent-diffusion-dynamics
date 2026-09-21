"""Semantic inspection queries. Every answer carries its source (public estimator, runtime,
or learned model readout) and distinguishes null, multiple and unknown answers."""
from __future__ import annotations

import math

import mujoco
import numpy as np

from rrp.contracts.task import EntityBinding


def _objects(s):
    return s.sim.tracker.tracks


def _entity_of_slot(s, slot):
    for ent, sl in s.sim.entity_slots.items():
        if sl == slot:
            return ent
    return None


def _cam_axis(s, cam="front"):
    m, d = s.sim.model, s.sim.data
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    return d.cam_xpos[cid].copy(), -d.cam_xmat[cid].reshape(3, 3)[:, 2]


def run_probe(s, req) -> dict:
    q = req.query
    sim = s.sim
    rt = sim.runtime
    base = dict(query=q, subject=req.subject, object=req.object, source="public_estimator", t=float(sim.data.time))
    if q == "visible":
        ans = [dict(slot=tr.slot, descriptor=tr.descriptor, entity=_entity_of_slot(s, tr.slot))
               for tr in _objects(s) if tr.visible]
        return dict(base, answers=ans, null=not ans, multiple=len(ans) > 1)
    if q == "looking_at":
        cam = req.subject or "front"
        try:
            p, ax = _cam_axis(s, cam)
        except Exception:  # noqa: BLE001
            return dict(base, answers=[], null=True, unknown_reason="no such camera")
        best = []
        for tr in _objects(s):
            if tr.mean is None:
                continue
            v = np.array(tr.mean) - p
            ang = math.degrees(math.acos(np.clip(v @ ax / np.linalg.norm(v), -1, 1)))
            if ang < 6.0:
                best.append(dict(slot=tr.slot, descriptor=tr.descriptor, angle_deg=round(ang, 2),
                                 visible=tr.visible))
        return dict(base, answers=best, null=not best, multiple=len(best) > 1,
                    note="gaze = camera optical axis; independent of visibility and task focus")
    if q == "focused_on":
        ans = []
        for eid, inst in rt.instances.items():
            if inst.status == "active":
                ev = rt.compiled.event(eid)
                for sl in ev.roles:
                    if sl.role in ("patient", "target", "destination") and isinstance(sl.binding, EntityBinding):
                        ans.append(dict(event=eid, role=sl.role, ordinal=sl.ordinal, entity=sl.binding.entity.id))
        return dict(base, source="task_runtime", answers=ans, null=not ans, multiple=len(ans) > 1,
                    note="task focus from active events' role bindings; not visibility or contact")
    if q in ("acting_on", "held_by", "contact_mode"):
        manips = [req.subject] if req.subject else list(sim.manip_map)
        ans = []
        for m in manips:
            if m not in sim.manip_map:
                continue
            r = sim.robots[sim.manip_map[m][0]]
            touch = sim._touch_values(r)
            held = [ent for ent in sim.entity_slots
                    if sim.estimate("held_by", [ent, m])[0] is True]
            if q == "held_by":
                ans.append(dict(manipulator=m, holds=held, null=not held, multiple=len(held) > 1))
            elif q == "acting_on":
                acting = held or ([] if touch.max(initial=0) < 0.2 else ["<unidentified contact>"])
                ans.append(dict(manipulator=m, acting_on=acting))
            else:
                mode = "grasp" if held else ("contact" if touch.max(initial=0) > 0.2 else "free")
                ans.append(dict(manipulator=m, contact_mode=mode, touch=touch.round(2).tolist()))
        return dict(base, answers=ans, null=not ans)
    if q == "manipulator_tasks":
        manips = [req.subject] if req.subject else list(sim.manip_map)
        ans = []
        for m in manips:
            evs = []
            for e in rt.compiled.definition.events:
                roles = [(sl.role, sl.ordinal) for sl in e.roles
                         if isinstance(sl.binding, EntityBinding) and sl.binding.entity.id == m]
                if roles:
                    evs.append(dict(event=e.id, operator=e.operator, roles=roles, status=rt.status(e.id)))
            ans.append(dict(manipulator=m, events=evs, active=[x["event"] for x in evs if x["status"] == "active"]))
        return dict(base, source="task_runtime", answers=ans)
    if q in ("relative_pose", "distance"):
        a, b = req.subject, req.object
        def pos_of(x):
            if x in sim.manip_map:
                return sim.tcp_estimate(x)[0], None
            p, v = sim._track(x)
            return p, v
        pa, va = pos_of(a)
        pb, vb = pos_of(b)
        if pa is None or pb is None:
            return dict(base, answers=[], null=True, unknown_reason="no estimate for one argument")
        d = pb - pa
        return dict(base, answers=[dict(frame="world", translation_m=d.round(4).tolist(),
                                        distance_m=round(float(np.linalg.norm(d)), 4),
                                        std_m=None if vb is None else [round(math.sqrt(x), 4) for x in vb])])
    if q == "active_frame":
        ans = [dict(event_id=r.event_id, attempt=r.attempt, output=r.output_name, type=r.type, version=r.version,
                    valid=r.valid, value=r.value, invalid_reason=r.invalid_reason,
                    age_s=round(float(sim.data.time) - r.created_at, 3))
               for r in rt.receipts.all() if r.type in ("frame_estimate", "contact_anchor")]
        return dict(base, source="receipts", answers=ans, null=not any(a["valid"] for a in ans))
    if q == "desired_delta":
        ans = []
        for eid, inst in rt.instances.items():
            if inst.status == "active":
                for c in rt.compiled.event(eid).desired_effects:
                    ans.append(dict(event=eid, predicate=c.predicate, value=c.value,
                                    args=[a.entity.id if isinstance(a, EntityBinding) else a.event_id
                                          for a in c.arguments], observed=False))
        return dict(base, source="task_definition", answers=ans, null=not ans,
                    note="desired effects are targets, not observed facts")
    if q == "feasibility":
        ans = []
        for m in sim.manip_map:
            for ent in sim.entity_slots:
                v, known, conf = sim.estimate("reachable", [m, ent])
                ans.append(dict(manipulator=m, entity=ent, reachable=v, known=known))
        return dict(base, answers=ans)
    if q == "uncertainty":
        ans = [dict(slot=tr.slot, descriptor=tr.descriptor, visible=tr.visible,
                    std_m=None if tr.var is None else [round(math.sqrt(x), 4) for x in tr.var],
                    unknown=tr.mean is None) for tr in _objects(s)]
        return dict(base, answers=ans)
    if q == "object_qa":
        model = getattr(s, "qa_model", None)
        if model is None:
            return dict(base, source="none", answers=[], null=True,
                        unknown_reason="no object-QA model loaded in this session (runs only on request)")
        return dict(base, source="learned_qa_readout", answers=[model.answer(s, req.object, req.question)])
    return dict(base, answers=[], null=True, unknown_reason="unsupported query")
