"""Contact-event segmentation, event-aligned knots and anchor-relative targets (W12, LABEL side only).

Everything here reads a privileged `ContactRecording` (simulator truth recorded by rrp.data.contact_labels) and returns
supervision targets / diagnostics. Nothing here is a policy input; the deploy-time counterpart is
rrp.features.anchor_frame.anchor_inputs (FK of measured joints + runtime receipts only).

Versions (a new target/packet version; existing packets and configs do not use any of this):
  SEGMENT_VERSION  = "contact-seg-v1"
  KNOT_VERSION     = "knots-contact-v1"    (event-aligned knot times; the uniform default is "knots-uniform-v0")
  TARGET_VERSION   = "anchor-targets-v1"

Contact phases per manipulator (PHASES): free, approach, make, maintain, slide, pivot, release.
  make      first `make_ticks` ticks of a debounced contact interval
  release   last `release_ticks` ticks before the debounced break
  slide     in contact and the TCP moves tangentially relative to the contact anchor faster than `slide_v` (m/s)
  pivot     in contact, not sliding, TCP relative rotation rate faster than `pivot_w` (rad/s)
  maintain  any other in-contact tick
  approach  `approach_ticks` ticks before a make
  free      otherwise
Debounce: a contact interval starts after `min_on` consecutive contact ticks (dated at its first tick) and ends after
`min_off` consecutive non-contact ticks (dated at the first non-contact tick) -- the analogue of the runtime's
ANCHOR_LOSS_S debounce, so chatter does not create events.

Anchor convention `runtime_v1` (default; matches rrp.envs.dual contact_anchor receipts): anchor pos = TCP position at
the make tick, normal = -tool z at the make tick, tangent unknown (yaw about the normal referenced to world +x).
Convention `contact_point`: the recorded mean contact point and contact normal at the make tick.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rrp.features.anchor_frame import AnchorFrame, frame_from_normal, geodesic_angle, pose_vec, relative_pose

SEGMENT_VERSION = "contact-seg-v1"
KNOT_VERSION = "knots-contact-v1"
UNIFORM_KNOT_VERSION = "knots-uniform-v0"
TARGET_VERSION = "anchor-targets-v1"
PHASES = ("free", "approach", "make", "maintain", "slide", "pivot", "release")
PHASE_ID = {p: i for i, p in enumerate(PHASES)}
EVENT_KINDS = ("make", "break", "slide_start", "slide_end", "pivot_start", "pivot_end")


@dataclass
class ContactRecording:
    """Per-tick privileged recording of one episode. Pairs are (a, b) entity names; a pair is a manipulator contact when
    a is in `manipulators`. contact_normal[t, c] points from b into a (unit, world) where known."""
    dt: float
    manipulators: list[str]
    tcp_pos: np.ndarray                       # [T, M, 3]
    tcp_R: np.ndarray                         # [T, M, 3, 3]
    objects: list[str] = field(default_factory=list)
    obj_pos: np.ndarray | None = None         # [T, O, 3]
    obj_R: np.ndarray | None = None           # [T, O, 3, 3]
    pairs: list[tuple[str, str]] = field(default_factory=list)
    contact: np.ndarray | None = None         # [T, C] bool
    contact_point: np.ndarray | None = None   # [T, C, 3]
    contact_normal: np.ndarray | None = None  # [T, C, 3]
    held: np.ndarray | None = None            # [T, O, M] bool

    @property
    def T(self) -> int:
        return int(self.tcp_pos.shape[0])

    def to_dict(self) -> dict:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "ContactRecording":
        arr = lambda x, dt=float: None if x is None else np.asarray(x, dt)
        return cls(dt=float(d["dt"]), manipulators=list(d["manipulators"]), tcp_pos=arr(d["tcp_pos"]),
                   tcp_R=arr(d["tcp_R"]), objects=list(d.get("objects", [])), obj_pos=arr(d.get("obj_pos")),
                   obj_R=arr(d.get("obj_R")), pairs=[tuple(p) for p in d.get("pairs", [])],
                   contact=arr(d.get("contact"), bool), contact_point=arr(d.get("contact_point")),
                   contact_normal=arr(d.get("contact_normal")), held=arr(d.get("held"), bool))


@dataclass(frozen=True)
class ContactEvent:
    t: int                     # tick
    kind: str                  # EVENT_KINDS
    pair: int                  # index into recording.pairs
    a: str
    b: str


@dataclass
class SegmentParams:
    min_on: int = 2
    min_off: int = 3
    make_ticks: int = 3
    release_ticks: int = 3
    approach_ticks: int = 10
    slide_v: float = 0.02      # m/s tangential TCP speed relative to the anchor
    pivot_w: float = 0.5       # rad/s relative rotation rate
    smooth: int = 2            # half-window (ticks) of the moving average applied to rates


# ----------------------------------------------------------------------------------------------- intervals
def debounced_intervals(x, min_on: int = 2, min_off: int = 3) -> list[tuple[int, int]]:
    """[(start, end_exclusive)] of contact intervals: on after `min_on` consecutive True ticks (dated at the first),
    off after `min_off` consecutive False ticks (dated at the first False tick). An interval open at the end is closed
    at len(x)."""
    x = np.asarray(x, bool)
    out, start, on_run, off_run, active = [], None, 0, 0, False
    for t, v in enumerate(x):
        if not active:
            on_run = on_run + 1 if v else 0
            if on_run >= min_on:
                active, start, off_run = True, t - on_run + 1, 0
        else:
            off_run = off_run + 1 if not v else 0
            if off_run >= min_off:
                out.append((start, t - off_run + 1))
                active, on_run = False, 0
    if active:
        out.append((start, len(x)))
    return out


def _movavg(x, h):
    if h <= 0 or len(x) == 0:
        return np.asarray(x, float)
    k = np.ones(2 * h + 1) / (2 * h + 1)
    xp = np.pad(np.asarray(x, float), (h, h), mode="edge")
    return np.convolve(xp, k, mode="valid")


def _runs(mask) -> list[tuple[int, int]]:
    m = np.asarray(mask, bool)
    out, s = [], None
    for t, v in enumerate(m):
        if v and s is None:
            s = t
        elif not v and s is not None:
            out.append((s, t))
            s = None
    if s is not None:
        out.append((s, len(m)))
    return out


def anchor_at(rec: ContactRecording, pair: int, t: int, convention: str = "runtime_v1") -> AnchorFrame | None:
    a = rec.pairs[pair][0]
    if a not in rec.manipulators:
        return None
    m = rec.manipulators.index(a)
    if convention == "contact_point" and rec.contact_point is not None and rec.contact_normal is not None \
            and np.linalg.norm(rec.contact_normal[t, pair]) > 0.5:
        f = frame_from_normal(rec.contact_point[t, pair], rec.contact_normal[t, pair])
    else:
        f = frame_from_normal(rec.tcp_pos[t, m], -rec.tcp_R[t, m][:, 2])
    f.source = dict(pair=pair, t=int(t), convention=convention)
    return f


def _rel_rates(rec: ContactRecording, pair: int, s: int, e: int, convention: str, h: int):
    """Tangential speed (m/s) and rotation rate (rad/s) of the TCP relative to the anchor made at s, over [s, e)."""
    m = rec.manipulators.index(rec.pairs[pair][0])
    f = anchor_at(rec, pair, s, convention)
    p = np.einsum("ij,tj->ti", f.R.T, rec.tcp_pos[s:e, m] - f.pos)
    R = np.einsum("ij,tjk->tik", f.R.T, rec.tcp_R[s:e, m])
    if e - s < 2:
        return np.zeros(e - s), np.zeros(e - s)
    v_t = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1) / rec.dt
    w = geodesic_angle(R[:-1], R[1:]) / rec.dt
    v_t = np.r_[v_t[:1], v_t]
    w = np.r_[w[:1], w]
    return _movavg(v_t, h), _movavg(w, h)


# ----------------------------------------------------------------------------------------------- segmentation
def segment(rec: ContactRecording, params: SegmentParams | None = None, convention: str = "runtime_v1") -> dict:
    """Returns dict(version, events [ContactEvent], intervals {pair: [(s, e)]}, phase [T, M] int (PHASE_ID),
    pair_phase [T, C] int)."""
    P = params or SegmentParams()
    T, M, C = rec.T, len(rec.manipulators), len(rec.pairs)
    events: list[ContactEvent] = []
    intervals: dict[int, list] = {}
    pair_phase = np.zeros((T, C), np.int64)
    for c, (a, b) in enumerate(rec.pairs):
        iv = debounced_intervals(rec.contact[:, c], P.min_on, P.min_off)
        intervals[c] = iv
        for s, e in iv:
            events.append(ContactEvent(s, "make", c, a, b))
            if e < T:
                events.append(ContactEvent(e, "break", c, a, b))
            pair_phase[s:e, c] = PHASE_ID["maintain"]
            if a in rec.manipulators:
                v_t, w = _rel_rates(rec, c, s, e, convention, P.smooth)
                sl = v_t > P.slide_v
                pv = (w > P.pivot_w) & ~sl
                pair_phase[s:e, c][sl] = PHASE_ID["slide"]
                pair_phase[s:e, c][pv] = PHASE_ID["pivot"]
                for kind, msk in (("slide", sl), ("pivot", pv)):
                    for rs, re_ in _runs(msk):
                        events.append(ContactEvent(s + rs, f"{kind}_start", c, a, b))
                        if s + re_ < e:
                            events.append(ContactEvent(s + re_, f"{kind}_end", c, a, b))
            if e < T:                                     # a real release (not the episode end)
                pair_phase[max(s, e - P.release_ticks):e, c] = PHASE_ID["release"]
            pair_phase[s:min(e, s + P.make_ticks), c] = PHASE_ID["make"]
            pair_phase[max(0, s - P.approach_ticks):s, c] = np.where(
                pair_phase[max(0, s - P.approach_ticks):s, c] == PHASE_ID["free"], PHASE_ID["approach"],
                pair_phase[max(0, s - P.approach_ticks):s, c])
    # manipulator phase = highest-priority phase among its pairs (make > release > slide > pivot > maintain > approach)
    prio = np.array([0, 1, 6, 2, 4, 3, 5])        # indexed by PHASE_ID
    phase = np.zeros((T, M), np.int64)
    for m, ent in enumerate(rec.manipulators):
        cs = [c for c, (a, _b) in enumerate(rec.pairs) if a == ent]
        if not cs:
            continue
        pp = pair_phase[:, cs]
        phase[:, m] = np.take_along_axis(pp, np.argmax(prio[pp], axis=1)[:, None], 1)[:, 0]
    events.sort(key=lambda ev: (ev.t, EVENT_KINDS.index(ev.kind), ev.pair))
    return dict(version=SEGMENT_VERSION, events=events, intervals=intervals, phase=phase, pair_phase=pair_phase)


# ----------------------------------------------------------------------------------------------- knots
def uniform_knot_times(K: int = 4, horizon: float = 0.8) -> np.ndarray:
    """The existing arm default (0.1, 0.3, 0.5, 0.7) for K=4, horizon 0.8 s: centres of K equal bins."""
    return (np.arange(K) + 0.5) * horizon / K


def event_aligned_knot_times(event_ticks, t0: int, dt: float, K: int = 4, horizon: float = 0.8,
                             min_gap: float | None = None) -> tuple[np.ndarray, list[str]]:
    """Knot times (s after t0, strictly increasing, in (0, horizon]) aligned to contact events.
    event_ticks: [(tick, kind)] of events. Events in (t0, t0 + horizon] are placed first (make/break before
    slide/pivot, then earliest), keeping at least `min_gap` (default horizon / (2K)) between knots; remaining knots are
    filled at the uniform positions farthest from the chosen ones. Returns (times, tags) with tag = event kind or
    'uniform'. With no events this equals uniform_knot_times."""
    gap = min_gap if min_gap is not None else horizon / (2 * K)
    cand = []
    for tk, kind in event_ticks:
        tau = (tk - t0) * dt
        if 0 < tau <= horizon + 1e-9:
            cand.append((0 if kind in ("make", "break") else 1, tau, kind))
    cand.sort()
    chosen: list[tuple[float, str]] = []
    for _, tau, kind in cand:
        if len(chosen) >= K:
            break
        if all(abs(tau - c) >= gap for c, _ in chosen):
            chosen.append((tau, kind))
    grid = list(uniform_knot_times(K, horizon))
    while len(chosen) < K:
        best = max(grid, key=lambda u: min([abs(u - c) for c, _ in chosen] or [np.inf]))
        grid.remove(best)
        chosen.append((best, "uniform"))
    chosen.sort()
    times = np.array([c for c, _ in chosen])
    # strictly increasing (the packet contract): nudge exact ties
    for i in range(1, K):
        if times[i] <= times[i - 1]:
            times[i] = times[i - 1] + 1e-3
    return times, [k for _, k in chosen]


# ----------------------------------------------------------------------------------------------- anchor targets
def active_anchors(rec: ContactRecording, seg: dict, t: int, m: int) -> dict[str, int | None]:
    """Pair index of the anchors that are active for manipulator slot m at tick t:
    own     = m's most recently made contact whose interval contains t, else m's most recent contact made before t
              (the "most recent contact point", kept after release as in the runtime's anchor until re-anchored);
    support = the most recently made contact of ANOTHER manipulator whose interval contains t: another contact point,
              i.e. a maintained support (support_insert: the left hand on the fixture) or the partner's grasp
              (handover: the giver's grip on the bar)."""
    ent = rec.manipulators[m]
    own, own_live, sup, t_own, t_own_live, t_sup = None, None, None, -1, -1, -1
    for c, iv in seg["intervals"].items():
        a = rec.pairs[c][0]
        if a not in rec.manipulators:
            continue
        for s, e in iv:
            if s > t:
                continue
            live = t < e
            if a == ent:
                if live and s > t_own_live:
                    own_live, t_own_live = c, s
                if s > t_own:
                    own, t_own = c, s
            elif live and s > t_sup:
                sup, t_sup = c, s
    return dict(own=own_live if own_live is not None else own, support=sup,
                own_make=t_own_live if own_live is not None else t_own, support_make=t_sup)


def anchor_relative_targets(rec: ContactRecording, seg: dict, t0: int, knot_ticks, convention: str = "runtime_v1") -> dict:
    """Targets per knot k (tick knot_ticks[k], clipped to the episode) and manipulator slot m:
      tcp_in_own [K,M,9] + mask      TCP pose in m's own most recent contact anchor frame (pos dm + 6D rot)
      tcp_in_support [K,M,9] + mask  TCP pose in the other anchor frame (another manipulator's live contact: support or grasp)
      held_in_tcp [K,M,9] + mask     pose of the object m holds, in m's TCP frame
      held_in_support [K,M,9] + mask pose of the object m holds, in the support anchor frame
      normal_in_tcp [K,M,3] + mask   live contact normal of m's own anchor pair, in m's TCP frame (recorded normals)
      tangent_in_own [K,M,3]         unit tangential TCP displacement direction since the anchor was made (0 if < 1 mm)
      phase [K,M]                    contact phase id (PHASES)
      contact_active [K,C]           pair in contact at the knot (debounced)
      contact_persist [K,C]          pair in contact continuously from t0 through the knot
    """
    K, M, C = len(knot_ticks), len(rec.manipulators), len(rec.pairs)
    T = rec.T
    z9 = lambda: np.zeros((K, M, 9), np.float32)
    out = dict(version=TARGET_VERSION, convention=convention, tcp_in_own=z9(), tcp_in_own_mask=np.zeros((K, M), bool),
               tcp_in_support=z9(), tcp_in_support_mask=np.zeros((K, M), bool), held_in_tcp=z9(),
               held_in_tcp_mask=np.zeros((K, M), bool), held_in_support=z9(), held_in_support_mask=np.zeros((K, M), bool),
               normal_in_tcp=np.zeros((K, M, 3), np.float32), normal_in_tcp_mask=np.zeros((K, M), bool),
               tangent_in_own=np.zeros((K, M, 3), np.float32), phase=np.zeros((K, M), np.int64),
               contact_active=np.zeros((K, C), bool), contact_persist=np.zeros((K, C), bool))
    live = np.zeros((T, C), bool)
    for c, iv in seg["intervals"].items():
        for s, e in iv:
            live[s:e, c] = True
    for k, tk in enumerate(knot_ticks):
        t = int(min(max(tk, 0), T - 1))
        out["contact_active"][k] = live[t]
        out["contact_persist"][k] = live[min(t0, T - 1):t + 1].all(axis=0) if t >= t0 else live[t]
        out["phase"][k] = seg["phase"][t]
        for m in range(M):
            p, R = rec.tcp_pos[t, m], rec.tcp_R[t, m]
            act = active_anchors(rec, seg, t, m)
            fo = fs = None
            if act["own"] is not None:
                fo = anchor_at(rec, act["own"], act["own_make"], convention)
                out["tcp_in_own"][k, m] = pose_vec(*relative_pose(fo, p, R))
                out["tcp_in_own_mask"][k, m] = True
                d = fo.R.T @ (p - fo.pos)
                d[2] = 0.0
                if np.linalg.norm(d) > 1e-3:
                    out["tangent_in_own"][k, m] = d / np.linalg.norm(d)
                if rec.contact_normal is not None and live[t, act["own"]]:
                    n = rec.contact_normal[t, act["own"]]
                    if np.linalg.norm(n) > 0.5:
                        out["normal_in_tcp"][k, m] = R.T @ (n / np.linalg.norm(n))
                        out["normal_in_tcp_mask"][k, m] = True
            if act["support"] is not None:
                fs = anchor_at(rec, act["support"], act["support_make"], convention)
                out["tcp_in_support"][k, m] = pose_vec(*relative_pose(fs, p, R))
                out["tcp_in_support_mask"][k, m] = True
            if rec.held is not None and rec.obj_pos is not None:
                held = np.flatnonzero(rec.held[t, :, m])
                if len(held):
                    o = int(held[0])
                    tf = AnchorFrame(np.asarray(p, float), np.asarray(R, float))      # the TCP frame itself
                    out["held_in_tcp"][k, m] = pose_vec(*relative_pose(tf, rec.obj_pos[t, o], rec.obj_R[t, o]))
                    out["held_in_tcp_mask"][k, m] = True
                    if fs is not None:
                        out["held_in_support"][k, m] = pose_vec(*relative_pose(fs, rec.obj_pos[t, o], rec.obj_R[t, o]))
                        out["held_in_support_mask"][k, m] = True
    return out


def event_ticks(seg: dict) -> list[tuple[int, str]]:
    return [(ev.t, ev.kind) for ev in seg["events"]]
