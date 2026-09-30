"""Anchor frames: poses relative to contact anchors (W12, feature-centric coordination).

Two uses, one geometry:
1. LABEL side (rrp.harness.data.contact_segments): anchor-relative targets per knot from privileged recordings.
2. DEPLOY side (`anchor_inputs`): an optional system-0 input computed ONLY from public information, namely the FK of
   measured joints (TCP pose) and the task runtime's receipts (`contact_anchor`: pos, normal, validity, age, covariance;
   `frame_estimate`: pos, quat). No simulator ground truth enters; `anchor_inputs` takes no session and no truth.

Conventions (unit-tested, tests/unit/test_contact_frames.py):
- Rotation matrices act on column vectors; a pose (p, R) maps local -> world: x_w = p + R x_l.
- 6D rotation = the first two COLUMNS of R, concatenated [R[:,0], R[:,1]] (Zhou et al. 2019); `rot6d_to_mat` inverts
  it by Gram-Schmidt, so any 6-vector maps to a valid rotation.
- Anchor frame of a contact anchor: origin = anchor pos; z = the anchor normal (as recorded by the producer; the dual
  runtime records -tool_z, i.e. pointing from the tool into the surface); x = the anchor tangent if given, else the
  projection of a declared reference direction (world +x, then world +y if nearly parallel) onto the tangent plane.
  `yaw_defined` is False in the second case: the rotation ABOUT the normal is then conventional, not measured.
- Frame-estimate anchors use their own quaternion (wxyz).

Positions in features are decimetres (x10), as in the arm probes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

ANCHOR_FRAME_VERSION = "rrp.features.anchor_frame/v1"
ANCHOR_INPUT_VERSION = "anchor-in-v1"
ANCHOR_ROLES = ("own", "other", "frame")        # per manipulator slot: own last contact / other contact point / frame
ANCHOR_BLOCK = 15                               # valid, rel pos(3), rel rot6d(6), normal in tcp(3), log age, log cov
ANCHOR_INPUT_DIM = ANCHOR_BLOCK * len(ANCHOR_ROLES)
POS_SCALE = 10.0


# ----------------------------------------------------------------------------------------------- rotations
def quat_wxyz_to_mat(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / max(np.linalg.norm(q), 1e-12)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def axis_angle(axis, ang: float) -> np.ndarray:
    k = np.asarray(axis, float)
    k = k / max(np.linalg.norm(k), 1e-12)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def rot6d(R) -> np.ndarray:
    """[..., 3, 3] -> [..., 6] (first two columns)."""
    R = np.asarray(R, float)
    return np.concatenate([R[..., :, 0], R[..., :, 1]], -1)


def rot6d_to_mat(x) -> np.ndarray:
    """[..., 6] -> [..., 3, 3] by Gram-Schmidt (any input gives a proper rotation)."""
    x = np.asarray(x, float)
    a, b = x[..., :3], x[..., 3:6]
    c0 = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    b = b - (c0 * b).sum(-1, keepdims=True) * c0
    c1 = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    c2 = np.cross(c0, c1)
    return np.stack([c0, c1, c2], -1)


def geodesic_angle(R1, R2) -> np.ndarray:
    """Angle (rad) of R1^T R2, batched over leading dims."""
    R1, R2 = np.asarray(R1, float), np.asarray(R2, float)
    tr = np.einsum("...ji,...ji->...", R1, R2)
    return np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))


def yaw_about(R_rel, axis=(0.0, 0.0, 1.0)) -> float:
    """Twist angle of R_rel about `axis` (swing-twist decomposition), in (-pi, pi]."""
    R = np.asarray(R_rel, float)
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    # quaternion of R (w, v); twist = (w, (v.a) a)
    tr = np.trace(R)
    w = math.sqrt(max(0.0, 1.0 + tr)) / 2.0
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    v = v / (4 * w) if w > 1e-6 else _axis_from_pi_rotation(R) * 1.0
    p = float(v @ a)
    return float(2.0 * math.atan2(p, w)) if w > 1e-6 else (math.pi if abs(p) > 0.5 else 0.0)


def _axis_from_pi_rotation(R):
    M = (R + np.eye(3)) / 2.0
    i = int(np.argmax(np.diag(M)))
    v = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
    return v


# ----------------------------------------------------------------------------------------------- frames
@dataclass
class AnchorFrame:
    pos: np.ndarray
    R: np.ndarray
    yaw_defined: bool = True
    kind: str = "contact"                     # contact | frame
    source: dict = field(default_factory=dict)  # event_id, attempt, version, participants (provenance)


def frame_from_normal(pos, normal, tangent=None, ref=(1.0, 0.0, 0.0)) -> AnchorFrame:
    n = np.asarray(normal, float)
    n = n / max(np.linalg.norm(n), 1e-12)
    yaw_defined = tangent is not None
    t = np.asarray(tangent if tangent is not None else ref, float)
    t = t - (t @ n) * n
    if np.linalg.norm(t) < 1e-3:                  # reference parallel to the normal
        t = np.array([0.0, 1.0, 0.0]) - n[1] * n
        yaw_defined = False if tangent is None else yaw_defined
    x = t / np.linalg.norm(t)
    y = np.cross(n, x)
    return AnchorFrame(np.asarray(pos, float), np.stack([x, y, n], 1), yaw_defined=yaw_defined)


def relative_pose(frame: AnchorFrame, p, R) -> tuple[np.ndarray, np.ndarray]:
    """Pose (p, R) expressed in the anchor frame."""
    return frame.R.T @ (np.asarray(p, float) - frame.pos), frame.R.T @ np.asarray(R, float)


def pose_vec(p_rel, R_rel) -> np.ndarray:
    """9-vector: position (dm) + 6D rotation."""
    return np.concatenate([np.asarray(p_rel, float) * POS_SCALE, rot6d(R_rel)]).astype(np.float32)


# ----------------------------------------------------------------------------------------------- deploy-time input
def anchors_from_receipts(receipts) -> list[dict]:
    """Public anchor observations from runtime receipts (rrp.tasks.receipts.Receipt or dicts with the same fields).
    Only `contact_anchor` and `frame_estimate` receipts; invalid ones are kept with valid=False (validity is an input)."""
    out = []
    for r in receipts:
        g = (lambda k, d=None: r.get(k, d)) if isinstance(r, dict) else (lambda k, d=None: getattr(r, k, d))
        typ = g("type")
        if typ not in ("contact_anchor", "frame_estimate"):
            continue
        v = g("value") or {}
        if "pos" not in v:
            continue
        out.append(dict(type=typ, pos=np.asarray(v["pos"], float), normal=v.get("normal"), tangent=v.get("tangent"),
                        quat_wxyz=v.get("quat_wxyz"), valid=bool(g("valid", True)), created_at=float(g("created_at", 0.0)),
                        participants=list(g("participants", []) or []), event_id=g("event_id"),
                        attempt=g("attempt"), version=g("version"),
                        cov=g("covariance_diag") or v.get("covariance_diag")))
    return out


def anchor_to_frame(a: dict) -> AnchorFrame:
    if a["type"] == "frame_estimate" or a.get("normal") is None:
        R = quat_wxyz_to_mat(a.get("quat_wxyz") or [1, 0, 0, 0])
        f = AnchorFrame(np.asarray(a["pos"], float), R, yaw_defined=a.get("quat_wxyz") is not None, kind="frame")
    else:
        f = frame_from_normal(a["pos"], a["normal"], a.get("tangent"))
    f.source = {k: a.get(k) for k in ("event_id", "attempt", "version", "participants")}
    return f


def select_anchors(anchors: list[dict], manipulator: str) -> dict[str, dict | None]:
    """Per role in ANCHOR_ROLES the most recently created VALID anchor:
    own   = contact anchor produced by an event whose actor is `manipulator` (its most recent contact point);
    other = contact anchor produced by any other actor (another contact point / maintained support);
    frame = frame estimate (e.g. the located hole frame)."""
    pick: dict[str, dict | None] = {r: None for r in ANCHOR_ROLES}
    for a in sorted(anchors, key=lambda a: a["created_at"]):
        if not a["valid"]:
            continue
        if a["type"] == "frame_estimate":
            role = "frame"
        else:
            role = "own" if manipulator in a["participants"] else "other"
        pick[role] = a
    return pick


def anchor_block(tcp_pos, tcp_R, a: dict | None, now: float) -> np.ndarray:
    out = np.zeros(ANCHOR_BLOCK, np.float32)
    if a is None:
        return out
    f = anchor_to_frame(a)
    p_rel, R_rel = relative_pose(f, tcp_pos, tcp_R)
    out[0] = 1.0
    out[1:10] = pose_vec(p_rel, R_rel)
    out[10:13] = np.asarray(tcp_R, float).T @ f.R[:, 2]              # anchor normal in the tool frame
    out[13] = math.log1p(max(now - a["created_at"], 0.0))
    cov = a.get("cov")
    out[14] = math.log(max(float(np.sum(cov[:3])), 1e-8)) / 10.0 if cov else 0.0
    return out


def anchor_inputs(tcp_poses: dict[str, tuple], receipts, manipulators: list[str], now: float,
                  max_m: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Deploy-time system-0 input variant: [max_m, ANCHOR_INPUT_DIM] features + [max_m] slot mask, in packet slot
    order `manipulators` (the declared role order). tcp_poses[ent] = (pos, R) from FK of MEASURED joints; receipts =
    the runtime's receipts. Nothing else is read."""
    anchors = anchors_from_receipts(receipts)
    X = np.zeros((max_m, ANCHOR_INPUT_DIM), np.float32)
    mask = np.zeros(max_m, bool)
    for m, ent in enumerate(manipulators[:max_m]):
        if ent is None or ent not in tcp_poses:
            continue
        mask[m] = True
        p, R = tcp_poses[ent]
        sel = select_anchors(anchors, ent)
        X[m] = np.concatenate([anchor_block(p, R, sel[r], now) for r in ANCHOR_ROLES])
    return X, mask
