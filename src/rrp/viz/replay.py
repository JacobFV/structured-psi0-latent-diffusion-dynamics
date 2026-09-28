"""Replay files for the live visualization room (viz/CONTRACT.md, schema `rrp-viz/replay/v1`, D-131).

Pure data code: static MuJoCo geometry export (meshes decimated to <= 2000 faces, vertices inlined), a read-only
per-frame collector, schema validation, gzip JSON writing and the index. Nothing here steps physics or changes
simulator state: the collector only READS mjData fields (xpos/xquat/qpos/contact list), so recording cannot change
behaviour (tests/unit/test_viz_record.py checks recorded == unrecorded actions on a tiny case).

The simulation side (which harness, seeds, checkpoints) lives in rrp.viz.record and runs on the PEER only.
"""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = "rrp-viz/replay/v1"
INDEX_SCHEMA = "rrp-viz/replays-index/v1"
MAX_FPS = 30
MAX_MESH_FACES = 2000
GEOM_TYPES = ("plane", "box", "sphere", "capsule", "cylinder", "ellipsoid", "mesh")
META_KEYS = ("family", "task", "body", "route", "source_label", "ckpt_sha", "variant", "seed", "condition", "success",
             "failure_stage", "physics", "decision_refs")
PHYSICS_KEYS = ("contact_version", "grasp_contact_version", "actuator_limits_version", "actuator_mode")
# per-frame signals (length n_frames); probe.* likewise; task_events is a sparse event list
FRAME_SIGNALS = ("joint_target", "joint_pos", "contacts", "packet_pca", "phase", "edit_active", "forward_progress",
                 "object_pose", "penetration_mm", "slip",
                 # v1.2 (rich run signals; all optional, omitted where the quantity does not exist)
                 "joint_vel", "actuator_force", "contact_force", "contact_pos", "power_w", "energy_j", "cot",
                 "base_vel", "object_vel", "packet_norm", "edit_dz_norm", "gripper_aperture", "grasp_state",
                 "hand_contact", "grip_drift")
NESTED_SIGNALS = ("probe", "probe_truth")          # {key: per-frame list}
SPARSE_SIGNALS = ("task_events", "packet_events")  # [{t, ...}]

# mjtGeom enum -> contract name (hfield becomes a mesh; sdf/flex are not exported)
_MJ_GEOM = {0: "plane", 1: "hfield", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh"}


def r4(x):
    """Round floats to 4 decimals (contract: float, 4 dp) for nested lists / arrays."""
    a = np.asarray(x, dtype=float)
    return np.round(a, 4).tolist()


# ------------------------------------------------------------------ mesh decimation (vertex clustering, numpy only)
def decimate(vertices: np.ndarray, faces: np.ndarray, max_faces: int = MAX_MESH_FACES):
    """Vertex-clustering decimation to <= max_faces triangles. Returns (vertices float32 [n,3], faces int32 [m,3]).
    Cells of a uniform grid over the bounding box merge their vertices (mean position); degenerate and duplicate faces
    are dropped. The grid is the finest one (binary search) that meets the face budget. Visual fidelity only."""
    V = np.asarray(vertices, np.float64).reshape(-1, 3)
    F = np.asarray(faces, np.int64).reshape(-1, 3)
    if len(F) <= max_faces:
        return V.astype(np.float32), F.astype(np.int32)
    lo, span = V.min(0), np.maximum(V.max(0) - V.min(0), 1e-9)

    def cluster(n):
        cell = (np.floor((V - lo) / span * n).clip(0, n - 1)).astype(np.int64)
        key = cell[:, 0] * n * n + cell[:, 1] * n + cell[:, 2]
        uk, inv = np.unique(key, return_inverse=True)
        cnt = np.bincount(inv, minlength=len(uk)).astype(np.float64)
        nv = np.stack([np.bincount(inv, weights=V[:, d], minlength=len(uk)) for d in range(3)], 1) / cnt[:, None]
        nf = inv[F]
        keep = (nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])
        nf = nf[keep]
        if len(nf):
            srt = np.sort(nf, 1)
            _, first = np.unique(srt, axis=0, return_index=True)
            nf = nf[np.sort(first)]
        used = np.unique(nf) if len(nf) else np.zeros(0, np.int64)
        remap = -np.ones(len(nv), np.int64)
        remap[used] = np.arange(len(used))
        return nv[used], remap[nf] if len(nf) else nf

    best = None
    lo_n, hi_n = 2, 256
    while lo_n <= hi_n:
        mid = (lo_n + hi_n) // 2
        nv, nf = cluster(mid)
        if len(nf) <= max_faces:
            best = (nv, nf)
            lo_n = mid + 1
        else:
            hi_n = mid - 1
    if best is None:
        best = cluster(2)
    return best[0].astype(np.float32), best[1].astype(np.int32)


# ------------------------------------------------------------------ static geometry
def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def export_geoms(model, *, groups=(0, 1, 2), max_faces: int = MAX_MESH_FACES) -> tuple[list[dict], list[str]]:
    """Visible geoms of a compiled MjModel -> (contract geoms, body names that carry them, in id order).
    Each geom: {name, body, type, size, rgba, pos, quat (wxyz, relative to its body), mesh?}. Meshes are decimated
    to <= max_faces and inlined (menagerie assets are never copied as files). Height fields become meshes."""
    import mujoco
    geoms, bodies_used = [], set()
    mesh_cache: dict[int, dict] = {}
    for g in range(model.ngeom):
        if int(model.geom_group[g]) not in groups:
            continue
        rgba = np.array(model.geom_rgba[g], float)
        matid = int(model.geom_matid[g])
        if matid >= 0:
            rgba = np.array(model.mat_rgba[matid], float)
        if rgba[3] <= 0.0:
            continue
        t = _MJ_GEOM.get(int(model.geom_type[g]))
        if t is None:
            continue
        b = int(model.geom_bodyid[g])
        bname = model.body(b).name or ("world" if b == 0 else f"body{b}")
        gname = model.geom(g).name or f"geom{g}"
        d = dict(name=gname, body=bname, type=t, size=r4(model.geom_size[g]), rgba=r4(rgba),
                 pos=r4(model.geom_pos[g]), quat=r4(model.geom_quat[g]))
        if t == "mesh":
            mid = int(model.geom_dataid[g])
            if mid not in mesh_cache:
                va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
                fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
                V = np.array(model.mesh_vert[va:va + vn])
                F = np.array(model.mesh_face[fa:fa + fn])
                v2, f2 = decimate(V, F, max_faces)
                mesh_cache[mid] = dict(vertices=r4(v2.reshape(-1)), faces=f2.reshape(-1).astype(int).tolist(),
                                       source_faces=int(fn), name=model.mesh(mid).name)
            d["mesh"] = mesh_cache[mid]
        elif t == "hfield":
            d["type"] = "mesh"
            d["mesh"] = _hfield_mesh(model, int(model.geom_dataid[g]), max_faces)
            d["hfield"] = True
        geoms.append(d)
        bodies_used.add(b)
    order = sorted(bodies_used)
    return geoms, [model.body(b).name or ("world" if b == 0 else f"body{b}") for b in order]


def _hfield_mesh(model, hid: int, max_faces: int) -> dict:
    nr, nc = int(model.hfield_nrow[hid]), int(model.hfield_ncol[hid])
    sx, sy, sz, _base = (float(v) for v in model.hfield_size[hid])
    adr = int(model.hfield_adr[hid])
    H = np.array(model.hfield_data[adr:adr + nr * nc]).reshape(nr, nc)
    # subsample the grid so 2*(r-1)*(c-1) <= max_faces
    step = 1
    while 2 * ((nr - 1) // step) * ((nc - 1) // step) > max_faces:
        step += 1
    rows, cols = np.arange(0, nr, step), np.arange(0, nc, step)
    Hs = H[np.ix_(rows, cols)]
    ys = -sy + 2 * sy * rows / max(nr - 1, 1)
    xs = -sx + 2 * sx * cols / max(nc - 1, 1)
    X, Y = np.meshgrid(xs, ys)
    V = np.stack([X, Y, Hs * sz], -1).reshape(-1, 3)
    R, C = len(rows), len(cols)
    idx = np.arange(R * C).reshape(R, C)
    a, b_, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    F = np.concatenate([np.stack([a, b_, d], -1).reshape(-1, 3), np.stack([a, d, c], -1).reshape(-1, 3)])
    return dict(vertices=r4(V.reshape(-1)), faces=F.reshape(-1).astype(int).tolist(), source_faces=int(2 * (nr - 1) * (nc - 1)),
                name=model.hfield(hid).name)


# ------------------------------------------------------------------ per-frame collector (read-only)
class FrameCollector:
    """Collects body poses and signals at <= MAX_FPS from a (model, data) pair. `stride` = sim ticks per frame.
    Only reads mjData. Signals are appended by the family adapter via `add(**signals)` in the same call as `frame()`."""

    def __init__(self, model, bodies: list[str], control_dt: float, max_fps: int = MAX_FPS):
        import mujoco
        self.model = model
        self.bodies = list(bodies)
        self.bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) if b != "world" else 0 for b in bodies]
        if any(b < 0 for b in self.bids):
            raise ValueError(f"unknown bodies: {[n for n, b in zip(bodies, self.bids) if b < 0]}")
        self.stride = max(1, math.ceil((1.0 / control_dt) / max_fps - 1e-9))
        self.fps = (1.0 / control_dt) / self.stride
        self.t, self.pos, self.quat = [], [], []
        self.signals: dict[str, list] = {}
        self.probe: dict[str, list] = {}
        self.probe_truth: dict[str, list] = {}
        self.sparse: dict[str, list] = {}
        self.task_events: list[dict] = []
        self.annotations: list[dict] = []
        self._tick = 0
        self._last_status: dict[str, str] = {}

    def due(self) -> bool:
        """True when the current tick is a recorded frame (call once per control tick, then tick())."""
        return self._tick % self.stride == 0

    def tick(self):
        self._tick += 1

    def frame(self, data, t: float | None = None, **signals):
        """One frame. A signal (or probe key) absent in some frames is None there (e.g. before the first packet)."""
        n = len(self.t)
        self.t.append(round(float(data.time if t is None else t), 4))
        self.pos.append(r4(data.xpos[self.bids]))
        self.quat.append(r4(data.xquat[self.bids]))
        for k, v in signals.items():
            if k in ("probe", "probe_truth"):
                dst = self.probe if k == "probe" else self.probe_truth
                for pk, pv in (v or {}).items():
                    dst.setdefault(pk, [None] * n).append(_clean(pv))
            else:
                self.signals.setdefault(k, [None] * n).append(_clean(v))
        for d in (self.signals, self.probe, self.probe_truth):
            for v in d.values():
                if len(v) < n + 1:
                    v.append(None)

    def events(self, t: float, statuses: dict[str, str]):
        """Task-event status changes (sparse): statuses = {event_id: status}."""
        for e, st in statuses.items():
            if self._last_status.get(e) != st:
                self.task_events.append(dict(t=round(float(t), 4), event=e, status=str(st)))
                self._last_status[e] = st

    def add_sparse(self, name: str, t: float, **fields):
        """A sparse event (e.g. a new packet) at time t; `name` must be in SPARSE_SIGNALS."""
        self.sparse.setdefault(name, []).append(dict(t=round(float(t), 4), **{k: _clean(v) for k, v in fields.items()}))

    def annotate(self, t: float, text: str):
        self.annotations.append(dict(t=round(float(t), 4), text=text))

    def result(self) -> dict:
        n = len(self.t)
        sig = {}
        for k, v in self.signals.items():
            if len(v) != n or all(x is None for x in v):
                continue                          # missing signals are omitted, never faked
            sig[k] = v
        for name, src in (("probe", self.probe), ("probe_truth", self.probe_truth)):
            keep = {k: v for k, v in src.items() if len(v) == n and not all(x is None for x in v)}
            if keep:
                sig[name] = keep
        for name, lst in self.sparse.items():
            if lst:
                sig[name] = lst
        if self.task_events:
            sig["task_events"] = self.task_events
        return dict(fps=round(self.fps, 4), n_frames=n, bodies=self.bodies,
                    frames=dict(t=self.t, body_pos=self.pos, body_quat=self.quat), signals=sig,
                    annotations=self.annotations)


def _clean(v):
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else round(float(v), 4)
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    a = np.asarray(v)
    if a.dtype == bool:
        return a.astype(bool).tolist()
    if a.dtype.kind in "iu":
        return a.astype(int).tolist()
    return r4(np.nan_to_num(a.astype(float)))


# ------------------------------------------------------------------ packet PCA (per bundle, fitted on training packets)
def fit_pca(Z: np.ndarray, k: int = 3) -> dict:
    """PCA basis on flattened training packets Z [n, D]. Returns {mean, components [k, D], explained_variance_ratio}."""
    Z = np.asarray(Z, np.float64).reshape(len(Z), -1)
    mu = Z.mean(0)
    U, S, Vt = np.linalg.svd(Z - mu, full_matrices=False)
    var = S ** 2
    return dict(mean=mu.astype(np.float32).tolist(), components=Vt[:k].astype(np.float32).tolist(),
                explained_variance_ratio=(var[:k] / max(var.sum(), 1e-12)).round(4).tolist(), n_fit=int(len(Z)), dim=int(Z.shape[1]))


def project_pca(basis: dict, z) -> list[float]:
    z = np.asarray(z, np.float64).reshape(-1)
    return r4((np.asarray(basis["components"]) @ (z - np.asarray(basis["mean"]))))


# ------------------------------------------------------------------ assembly, validation, IO
def build_replay(rid: str, meta: dict, geoms: list[dict], collected: dict) -> dict:
    doc = dict(schema=SCHEMA, id=rid, meta=meta, fps=collected["fps"], n_frames=collected["n_frames"], geoms=geoms,
               frames=collected["frames"], bodies=collected["bodies"], signals=collected["signals"],
               annotations=collected.get("annotations", []))
    validate_replay(doc)
    return doc


class ReplaySchemaError(ValueError):
    pass


def validate_replay(doc: dict) -> None:
    """Checks a replay document against viz/CONTRACT.md (rrp-viz/replay/v1). Raises ReplaySchemaError."""
    def need(c, msg):
        if not c:
            raise ReplaySchemaError(msg)
    need(doc.get("schema") == SCHEMA, f"schema must be {SCHEMA}")
    for k in ("id", "meta", "fps", "n_frames", "geoms", "frames", "bodies", "signals"):
        need(k in doc, f"missing key {k}")
    m = doc["meta"]
    for k in META_KEYS:
        need(k in m, f"meta.{k} missing")
    need(isinstance(m["physics"], dict) and all(k in m["physics"] for k in PHYSICS_KEYS), "meta.physics incomplete")
    need(isinstance(m["decision_refs"], list) and m["decision_refs"], "meta.decision_refs must be a non-empty list")
    need(isinstance(m["success"], bool) or m["success"] is None, "meta.success must be bool (or null if undefined)")
    need(0 < float(doc["fps"]) <= MAX_FPS + 1e-6, f"fps must be in (0, {MAX_FPS}]")
    n, nb = int(doc["n_frames"]), len(doc["bodies"])
    need(n > 0, "n_frames must be > 0")
    fr = doc["frames"]
    need(len(fr["t"]) == n and len(fr["body_pos"]) == n and len(fr["body_quat"]) == n, "frames lengths != n_frames")
    need(all(len(p) == nb and all(len(x) == 3 for x in p) for p in fr["body_pos"]), "body_pos shape != [n, bodies, 3]")
    need(all(len(q) == nb and all(len(x) == 4 for x in q) for q in fr["body_quat"]), "body_quat shape != [n, bodies, 4]")
    need(all(b >= a for a, b in zip(fr["t"], fr["t"][1:])), "frames.t must be non-decreasing")
    bset = set(doc["bodies"])
    for g in doc["geoms"]:
        for k in ("name", "body", "type", "size", "rgba"):
            need(k in g, f"geom missing {k}")
        need(g["type"] in GEOM_TYPES, f"geom type {g['type']} not in {GEOM_TYPES}")
        need(g["body"] in bset, f"geom {g['name']} body {g['body']} not in bodies")
        need(len(g["rgba"]) == 4, "rgba must have 4 values")
        if g["type"] == "mesh":
            me = g.get("mesh") or {}
            need(len(me.get("vertices", [])) % 3 == 0 and len(me.get("faces", [])) % 3 == 0, "mesh arrays not triples")
            need(len(me["faces"]) // 3 <= MAX_MESH_FACES, f"mesh {g['name']} has > {MAX_MESH_FACES} faces")
            nv = len(me["vertices"]) // 3
            need(not me["faces"] or (0 <= min(me["faces"]) and max(me["faces"]) < nv), "mesh face index out of range")
    s = doc["signals"]
    for k, v in s.items():
        if k in FRAME_SIGNALS:
            need(len(v) == n, f"signals.{k} length {len(v)} != n_frames {n}")
        elif k in NESTED_SIGNALS:
            for pk, pv in v.items():
                need(len(pv) == n, f"signals.{k}.{pk} length != n_frames")
        elif k == "task_events":
            need(all({"t", "event", "status"} <= set(e) for e in v), "task_events entries need t, event, status")
        elif k in SPARSE_SIGNALS:
            need(all("t" in e for e in v), f"{k} entries need t")
        else:
            raise ReplaySchemaError(f"unknown signal {k}")
    if "packet_pca" in s:
        need("packet_pca_basis" in m, "packet_pca needs meta.packet_pca_basis")
        need(all(x is None or len(x) == 3 for x in s["packet_pca"]), "packet_pca entries must be [pc1, pc2, pc3]")
    for a in doc.get("annotations", []):
        need({"t", "text"} <= set(a), "annotations need t and text")


def write_replay(doc: dict, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{doc['id']}.json.gz"
    raw = json.dumps(doc, separators=(",", ":"), allow_nan=False).encode()
    with gzip.open(p, "wb", compresslevel=9) as fh:
        fh.write(raw)
    return p


def read_replay(p: Path) -> dict:
    p = Path(p)
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt") as fh:
        return json.load(fh)


def index_entry(doc: dict, file: str, video: str | None = None) -> dict:
    m = doc["meta"]
    e = dict(id=doc["id"], family=m["family"], task=m["task"], body=m["body"], route=m["route"],
             source_label=m["source_label"], variant=m["variant"], seed=m["seed"], condition=m["condition"],
             success=m["success"], n_frames=doc["n_frames"], fps=doc["fps"], file=file)
    for k in ("recorded_success", "reproduced", "decision_refs", "physics", "failure_stage", "ckpt_sha", "caveat"):
        if k in m:
            e[k] = m[k]
    if video:
        e["video"] = video
    return e


def write_index(root: Path, generated_at: str, git_sha: str | None = None) -> dict:
    """Scan <root>/**/*.json.gz and write <root>/index.json (rrp-viz/replays-index/v1)."""
    root = Path(root)
    rows, bytes_ = [], 0
    for p in sorted(root.rglob("*.json.gz")):
        doc = read_replay(p)
        rows.append(index_entry(doc, str(p.relative_to(root))))
        bytes_ += p.stat().st_size
    idx = dict(schema=INDEX_SCHEMA, generated_at=generated_at, git_sha=git_sha, sources=[str(root)],
               n=len(rows), total_bytes=bytes_, replays=rows)
    (root / "index.json").write_text(json.dumps(idx, indent=1))
    return idx
