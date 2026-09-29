"""Feature-centric coordination metrics (W12; re-exported by rrp.evaluation.contact_metrics): relative-frame orientation drift, contact-sequence fidelity,
re-anchoring latency, stance drift. PRIVILEGED evaluation-only measurements (simulator truth); nothing feeds a policy.

Keys are merged into the W6 `motion` field of eval rows with the prefix `cf_` (plus cf_version) -- only when enabled
(RRP_CONTACT_METRICS=1 or an explicit flag), so rows of existing evaluations are unchanged by default.

Primitives (unit-tested on synthetic trajectories, tests/unit/test_contact_frames.py):
  relative_drift(Rr, pr, Ro, po, segs)   per segment, drift of the pose of o in frame r relative to its value at the segment
                                         start: rotation (geodesic, rad) and position (m); max / mean-of-max / final
  contact_sequence(times, spec, ref)     precedence violations, maintained-contact violations, missing events, Kendall
                                         order error and timing error vs a reference sequence (e.g. the teacher, same seed)
  settle_latency(...)                    re-anchoring: time from a new contact until the TCP is at rest (< v_tol for
                                         `hold` ticks) in the new anchor frame; receipt latency: time until the runtime
                                         publishes a contact_anchor receipt for that actor
  stance_drift(foot_pos, foot_yaw, st)   per debounced stance: horizontal displacement and |yaw change| of the foot from
                                         its touchdown pose (legged; "slip" measured in the stance-contact frame)
Composite rows: dual_contact_motion (dual arm), arm_contact_motion (single arm, held object vs gripper),
legged_contact_motion (from LeggedMotionRecorder(record_stance=True).stance_trace()).
"""
from __future__ import annotations

import math
import os
from itertools import combinations

import numpy as np

from rrp.harness.data.contact_segments import (PHASE_ID, ContactRecording, SegmentParams, anchor_at, debounced_intervals,
                                       segment)
from rrp.policies.features.anchor_frame import geodesic_angle

CF_VERSION = "rrp.evaluation.contact_metrics/v1"


def contact_metrics_enabled(flag: bool | None = None) -> bool:
    return bool(flag) if flag is not None else os.environ.get("RRP_CONTACT_METRICS", "0") not in ("", "0", "false")


# ----------------------------------------------------------------------------------------------- drift
def relative_drift(R_ref, p_ref, R_obj, p_obj, segments) -> dict:
    """Pose of `obj` in the frame of `ref` over each segment [s, e), relative to its value at s."""
    R_ref, R_obj = np.asarray(R_ref, float), np.asarray(R_obj, float)
    p_ref, p_obj = np.asarray(p_ref, float), np.asarray(p_obj, float)
    per = []
    for s, e in segments:
        if e - s < 2:
            continue
        Rrel = np.einsum("tji,tjk->tik", R_ref[s:e], R_obj[s:e])
        prel = np.einsum("tji,tj->ti", R_ref[s:e], p_obj[s:e] - p_ref[s:e])
        rot = geodesic_angle(Rrel[:1], Rrel)
        pos = np.linalg.norm(prel - prel[:1], axis=1)
        per.append(dict(s=int(s), e=int(e), rot_max=float(rot.max()), rot_final=float(rot[-1]),
                        pos_max=float(pos.max()), pos_final=float(pos[-1])))
    if not per:
        return dict(n=0, rot_max=None, rot_mean=None, rot_final=None, pos_max=None, pos_mean=None, segments=[])
    return dict(n=len(per), rot_max=max(p["rot_max"] for p in per), rot_mean=float(np.mean([p["rot_max"] for p in per])),
                rot_final=float(np.mean([p["rot_final"] for p in per])), pos_max=max(p["pos_max"] for p in per),
                pos_mean=float(np.mean([p["pos_max"] for p in per])), segments=per)


def _trim(iv, a: int, b: int):
    return [(s + a, e - b) for s, e in iv if e - b - (s + a) >= 2]


def _intersect(A, B):
    out = []
    for s1, e1 in A:
        for s2, e2 in B:
            s, e = max(s1, s2), min(e1, e2)
            if e - s >= 2:
                out.append((s, e))
    return out


# ----------------------------------------------------------------------------------------------- sequence
# Contact-sequence specs derived from the task graphs (tasks/support_and_insert.json, tasks/handover.json). Keys are
# "make:<a>|<b>" / "break:<a>|<b>" with the recorder's pair names (manipulator entity | sim body).
#   precedence: (x, y) = x must happen before y
#   maintained: (pair, until) = once made, the pair stays in contact (no debounced break) until event `until`
TASK_CONTACT_SPECS = {
    "support_insert": dict(
        events=["make:left|fixture", "make:right|peg", "break:right|peg"],
        precedence=[("make:left|fixture", "break:right|peg"), ("make:right|peg", "break:right|peg")],
        maintained=[("left|fixture", "break:right|peg")]),          # support maintained_during align + insert
    "handover": dict(
        events=["make:left|bar", "make:right|bar", "break:left|bar"],
        precedence=[("make:left|bar", "make:right|bar"), ("make:right|bar", "break:left|bar")],
        maintained=[("left|bar", "make:right|bar")]),               # giver holds until the receiver has it
}


def swap_hands(spec: dict) -> dict:
    """The contact-sequence edit "the other hand first": left <-> right in every key (for the edit suite)."""
    sw = lambda k: k.replace("left", "\0").replace("right", "left").replace("\0", "right")
    return dict(events=[sw(k) for k in spec["events"]], precedence=[(sw(a), sw(b)) for a, b in spec["precedence"]],
                maintained=[(sw(a), sw(b)) for a, b in spec["maintained"]])


def event_times(rec: ContactRecording, seg: dict) -> dict[str, float]:
    """First occurrence time (s) of every make/break key."""
    out: dict[str, float] = {}
    for ev in seg["events"]:
        if ev.kind in ("make", "break"):
            k = f"{ev.kind}:{ev.a}|{ev.b}"
            out.setdefault(k, ev.t * rec.dt)
    return out


def _intervals_s(rec, seg, pair_key):
    a, b = pair_key.split("|")
    if (a, b) not in rec.pairs:
        return []
    c = rec.pairs.index((a, b))
    return [(s * rec.dt, e * rec.dt) for s, e in seg["intervals"][c]]


def contact_sequence(times: dict[str, float], spec: dict, ref_times: dict[str, float] | None = None,
                     intervals: dict[str, list] | None = None) -> dict:
    """times: first occurrence (s) per key; intervals: pair_key -> [(start_s, end_s)] for maintained checks."""
    missing = [k for k in spec["events"] if k not in times]
    ev = [(a, b) for a, b in spec["precedence"] if a in times and b in times]
    viol = [(a, b) for a, b in ev if not times[a] < times[b]]
    mviol, mev = [], 0
    for pair, until in spec.get("maintained", []):
        mk = f"make:{pair}"
        if intervals is None or mk not in times or until not in times:
            continue
        mev += 1
        t0, t1 = times[mk], times[until]
        # the first interval of the pair must cover [t0, t1]
        first = next(((s, e) for s, e in intervals.get(pair, []) if s <= t0 + 1e-9), None)
        if first is None or first[1] < t1 - 1e-9:
            mviol.append((pair, until))
    out = dict(n_precedence=len(ev), precedence_violations=len(viol), n_maintained=mev,
               maintained_violations=len(mviol), missing=len(missing), missing_keys=missing,
               violated=[f"{a}<{b}" for a, b in viol] + [f"{p} maintained until {u}" for p, u in mviol])
    n_ok = len(ev) + mev
    out["order_error"] = (len(viol) + len(mviol)) / n_ok if n_ok else None
    if ref_times is not None:
        keys = [k for k in spec["events"] if k in times and k in ref_times]
        pairs = list(combinations(keys, 2))
        disc = sum(1 for x, y in pairs if (times[x] - times[y]) * (ref_times[x] - ref_times[y]) < 0)
        out["kendall_order_error"] = disc / len(pairs) if pairs else None
        dts = [abs(times[k] - ref_times[k]) for k in keys]
        out["timing_mae_s"] = float(np.mean(dts)) if dts else None
        out["timing_max_s"] = float(np.max(dts)) if dts else None
    return out


# ----------------------------------------------------------------------------------------------- re-anchoring
def settle_latency(rec: ContactRecording, pair: int, make_tick: int, end_tick: int, v_tol: float = 0.01,
                   hold: int = 5, convention: str = "runtime_v1") -> float | None:
    """Seconds from the make tick until the TCP speed in the new anchor frame stays below v_tol for `hold` ticks
    (within the contact interval); None if it never settles."""
    a = rec.pairs[pair][0]
    if a not in rec.manipulators:
        return None
    m = rec.manipulators.index(a)
    f = anchor_at(rec, pair, make_tick, convention)
    p = np.einsum("ij,tj->ti", f.R.T, rec.tcp_pos[make_tick:end_tick, m] - f.pos)
    if len(p) < hold + 1:
        return None
    v = np.linalg.norm(np.diff(p, axis=0), axis=1) / rec.dt
    run = 0
    for i, x in enumerate(v):
        run = run + 1 if x < v_tol else 0
        if run >= hold:
            return float((i - hold + 1) * rec.dt)
    return None


def receipt_latency(make_time: float, actor: str, receipt_log: list[dict]) -> float | None:
    """Seconds from a physical make to the first contact_anchor receipt put for that actor at or after it (the
    runtime's re-anchoring); receipt_log = TaskRuntime.receipts.log (put records carry created_at, participants)."""
    ts = [r["created_at"] for r in receipt_log if r.get("op") == "put" and r.get("type") == "contact_anchor"
          and actor in (r.get("participants") or []) and r["created_at"] >= make_time - 1e-6]
    return float(min(ts) - make_time) if ts else None


# ----------------------------------------------------------------------------------------------- composites
def _held_segments(rec, o, m, P):
    return _trim(debounced_intervals(rec.held[:, o, m], P.min_on, P.min_off), P.make_ticks, P.release_ticks)


def arm_contact_motion(rec: ContactRecording, params: SegmentParams | None = None) -> dict:
    """Held-object pose drift relative to the gripper during maintained holds (every manipulator/object)."""
    P = params or SegmentParams()
    rot, pos, n = [], [], 0
    for m in range(len(rec.manipulators)):
        for o in range(len(rec.objects)):
            d = relative_drift(rec.tcp_R[:, m], rec.tcp_pos[:, m], rec.obj_R[:, o], rec.obj_pos[:, o],
                               _held_segments(rec, o, m, P))
            if d["n"]:
                rot.append(d["rot_max"]); pos.append(d["pos_max"]); n += d["n"]
    return dict(cf_version=CF_VERSION, cf_held_rot_drift_grip_max_rad=max(rot) if rot else None,
                cf_held_pos_drift_grip_max_m=max(pos) if pos else None, cf_n_held_segments=n)


def dual_contact_motion(rec: ContactRecording, task: str | None = None, ref_times: dict | None = None,
                        receipt_log: list[dict] | None = None, params: SegmentParams | None = None,
                        spec: dict | None = None) -> dict:
    P = params or SegmentParams()
    seg = segment(rec, P)
    out = arm_contact_motion(rec, P)
    # (b) held object relative to the maintained support: frame = the supporting hand's current TCP, over ticks where m
    # holds o and another manipulator keeps a live contact with an object o is not
    rot_s, pos_s, slip = [], [], []
    for m, ent in enumerate(rec.manipulators):
        for o in range(len(rec.objects)):
            held = _held_segments(rec, o, m, P)
            if not held:
                continue
            for c, (a, b) in enumerate(rec.pairs):
                if a == ent or a not in rec.manipulators or b == rec.objects[o]:
                    continue
                ms = rec.manipulators.index(a)
                sup = _trim(seg["intervals"][c], P.make_ticks, P.release_ticks)
                d = relative_drift(rec.tcp_R[:, ms], rec.tcp_pos[:, ms], rec.obj_R[:, o], rec.obj_pos[:, o],
                                   _intersect(held, sup))
                if d["n"]:
                    rot_s.append(d["rot_max"]); pos_s.append(d["pos_max"])
    # support-anchor slip: TCP displacement from the anchor over maintained (non-held) manipulator contacts
    for c, (a, b) in enumerate(rec.pairs):
        if a not in rec.manipulators:
            continue
        m = rec.manipulators.index(a)
        o = rec.objects.index(b) if b in rec.objects else None
        for s, e in _trim(seg["intervals"][c], P.make_ticks, P.release_ticks):
            if o is not None and rec.held is not None and rec.held[s:e, o, m].mean() > 0.5:
                continue                                              # a grasp, not a support contact
            f = anchor_at(rec, c, s)
            p = np.einsum("ij,tj->ti", f.R.T, rec.tcp_pos[s:e, m] - f.pos)
            slip.append(float(np.linalg.norm(p - p[:1], axis=1).max()))
    out.update(cf_held_rot_drift_support_max_rad=max(rot_s) if rot_s else None,
               cf_held_pos_drift_support_max_m=max(pos_s) if pos_s else None,
               cf_support_anchor_slip_max_m=max(slip) if slip else None)
    # re-anchoring
    lat, rlat = [], []
    for ev in seg["events"]:
        if ev.kind != "make" or ev.a not in rec.manipulators:
            continue
        end = next(e for s, e in seg["intervals"][ev.pair] if s == ev.t)
        x = settle_latency(rec, ev.pair, ev.t, end)
        if x is not None:
            lat.append(x)
        if receipt_log is not None:
            y = receipt_latency(ev.t * rec.dt, ev.a, receipt_log)
            if y is not None:
                rlat.append(y)
    out.update(cf_reanchor_settle_mean_s=float(np.mean(lat)) if lat else None,
               cf_reanchor_settle_max_s=float(np.max(lat)) if lat else None, cf_n_makes=len(lat))
    if receipt_log is not None:
        out["cf_anchor_receipt_latency_mean_s"] = float(np.mean(rlat)) if rlat else None
    # contact sequence
    sp = spec or TASK_CONTACT_SPECS.get(task or "")
    if sp is not None:
        times = event_times(rec, seg)
        iv = {k.split(":", 1)[1]: _intervals_s(rec, seg, k.split(":", 1)[1]) for k in sp["events"]}
        iv.update({p: _intervals_s(rec, seg, p) for p, _ in sp.get("maintained", [])})
        cs = contact_sequence(times, sp, ref_times, iv)
        out.update(cf_contact_order_error=cs["order_error"], cf_contact_missing=cs["missing"],
                   cf_contact_violations=cs["violated"], cf_contact_event_times=times)
        if ref_times is not None:
            out.update(cf_contact_kendall_error=cs["kendall_order_error"], cf_contact_timing_mae_s=cs["timing_mae_s"])
    out["cf_phase_share"] = {p: float((seg["phase"] == i).mean()) for p, i in PHASE_ID.items()}
    return out


def stance_drift(foot_pos, foot_yaw, stance, min_on: int = 2, min_off: int = 2, skip: int = 1) -> dict:
    """foot_pos [T,L,3] (contact point or foot origin), foot_yaw [T,L], stance [T,L] bool. For each debounced stance
    interval (first `skip` ticks after touchdown ignored: impact settling), the max horizontal displacement and max
    |yaw change| relative to the pose at touchdown + skip."""
    P, Y, S = np.asarray(foot_pos, float), np.asarray(foot_yaw, float), np.asarray(stance, bool)
    pos, yaw = [], []
    for l in range(S.shape[1]):
        for s, e in debounced_intervals(S[:, l], min_on, min_off):
            s2 = s + skip
            if e - s2 < 2:
                continue
            dp = np.linalg.norm(P[s2:e, l, :2] - P[s2, l, :2], axis=1)
            dy = np.abs((Y[s2:e, l] - Y[s2, l] + math.pi) % (2 * math.pi) - math.pi)
            pos.append(float(dp.max()))
            yaw.append(float(dy.max()))
    if not pos:
        return dict(n_stances=0, pos_max=None, pos_mean=None, yaw_max=None, yaw_mean=None)
    return dict(n_stances=len(pos), pos_max=max(pos), pos_mean=float(np.mean(pos)), yaw_max=max(yaw),
                yaw_mean=float(np.mean(yaw)))


def legged_contact_motion(trace: dict) -> dict:
    d = stance_drift(trace["foot_pos"], trace["foot_yaw"], trace["stance"])
    return dict(cf_version=CF_VERSION, cf_stance_pos_drift_max_m=d["pos_max"], cf_stance_pos_drift_mean_m=d["pos_mean"],
                cf_stance_yaw_drift_max_rad=d["yaw_max"], cf_stance_yaw_drift_mean_rad=d["yaw_mean"],
                cf_n_stances=d["n_stances"])
