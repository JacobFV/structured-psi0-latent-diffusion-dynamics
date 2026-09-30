"""Geometry labels + scene parts (D-144 unit R14; design: docs/relations.md sections 2-5, section 10 row R14;
catalog: research/relations_catalog.md "metric position" / "depth" / "orientation" / "surface directions").

Labels, written ONCE against `rrp.envs.base.StateView` (section 5.1) and registered into `rrp.harness.data.relgen.
LABELS` -- every one runs in any env whose `caps` cover its `needs` (`label_runs_in`):

- `pos3d`          true world position of each token's entity                              (needs: poses)
- `cam_uvd`        projection of the true position through the declared scene camera; depth is read from
                   `render_depth` at the projected pixel when the view declares "depth_render" (so occlusion /
                   render artifacts show up in the label), else the analytic camera-frame depth (needs: poses,
                   camera)
- `orient`         true orientation of the entity: the 3x3 world rotation matrix, row-major flattened to 9 dims,
                   matching `FieldDef("orient", 9, "orientation", ...)`                     (needs: poses)
- `contact_normal` the true contact-surface normal at the entity, oriented OUTWARD from that entity's own surface
                   (i.e. `ContactState.normal`, which points a -> b, for the `a` side; its negation for the `b`
                   side); the mean of all of a step's contacts touching that entity, renormalized (needs: contacts)

Every label is arity 1 (one value per token): a `TokenIndex` may name several token sets at once (`TokenIndex.sets`
is a dict), and the labels here are computed over the flattened token order (all sets, in dict-iteration order,
slot order within each set) so `Label.value` / `.valid` have `T = sum(len(slots) for slots in idx.sets.values())`
rows, matching how one `TokenSet` of tokens is addressed elsewhere in the registry (docs section 2).

Scene parts, registered into `PARTS`, mutate a declarative `SceneDraft` only (`relgen.compose`, unit R11, is not
implemented yet, so these never touch a live simulator):

- `table_objects` places `n_objects` on a table plane at random (x, y) and yaw; `vary(draft, rng, "orient")`
  returns decoupling copies (docs 5.2) that differ from `draft` ONLY in every object's yaw -- position, ids and
  everything else stay byte-identical, isolating orientation as in docs 6's `geo.orient` row.
- `camera_depth` places one object along one camera's view ray at a chosen pixel and depth;
  `vary(draft, rng, "depth")` returns decoupling copies that keep the SAME projected (u, v) pixel and differ ONLY
  in depth, as docs 6's `geo.depth3d` row requires ("same pixel, other depth"). The camera geometry is either
  supplied by the caller (`draft.kwargs["camera"]`, a `rrp.envs.base.Camera`) or defaults to a literal
  reconstruction of `rrp.bodies.generators.workspace_spec`'s declared "front" camera (pos, xyaxes, fovy=55) --
  the camera every arm/dual table scenario declares (verified against a real session's `sv.camera("front")` in
  `tests/unit/test_relations_r14.py`), so the part is usable before a live env exists.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from rrp.envs.base import Camera, StateView
from rrp.harness.data.relgen import (Label, LabelDef, SceneDraft, ScenePart, TokenIndex, register_label,
                                     register_part)

__all__ = ["DEFAULT_CAMERA", "front_camera", "project", "unproject", "TABLE_OBJECTS", "CAMERA_DEPTH"]

DEFAULT_CAMERA = "front"   # the one camera name every arm/dual table scenario declares (rrp.bodies.generators)


# ------------------------------------------------------------------ shared: token flattening
def _token_ids(idx: TokenIndex) -> list:
    return [eid for slots in idx.sets.values() for eid in slots]


def _entities_by_id(view: StateView) -> dict:
    return {e.id: e for e in view.entities()}


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """wxyz unit quaternion -> 3x3 world rotation matrix (standard formula; matches MuJoCo's `xquat`/`mju_quat2Mat`
    convention, which is what every `StateView.entities()` implementation supplies)."""
    w, x, y, z = (float(c) for c in q)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([[1 - (yy + zz), xy - wz, xz + wy],
                     [xy + wz, 1 - (xx + zz), yz - wx],
                     [xz - wy, yz + wx, 1 - (xx + yy)]])


# ------------------------------------------------------------------ label: pos3d
def _label_pos3d(view: StateView, idx: TokenIndex) -> Label:
    ents = _entities_by_id(view)
    ids = _token_ids(idx)
    value = np.zeros((len(ids), 3), dtype=np.float64)
    valid = np.zeros((len(ids),), dtype=bool)
    for i, eid in enumerate(ids):
        e = ents.get(eid) if eid is not None else None
        if e is not None:
            value[i] = e.pos
            valid[i] = True
    return Label(value=value, valid=valid, prov="gt", version="1")


register_label(LabelDef(name="pos3d", version="1", arity=1, needs=frozenset({"poses"}), fn=_label_pos3d, prov="gt"))


# ------------------------------------------------------------------ label: orient
def _label_orient(view: StateView, idx: TokenIndex) -> Label:
    ents = _entities_by_id(view)
    ids = _token_ids(idx)
    value = np.zeros((len(ids), 9), dtype=np.float64)
    valid = np.zeros((len(ids),), dtype=bool)
    for i, eid in enumerate(ids):
        e = ents.get(eid) if eid is not None else None
        if e is not None and e.quat is not None:
            value[i] = _quat_to_mat(e.quat).reshape(-1)
            valid[i] = True
    return Label(value=value, valid=valid, prov="gt", version="1")


register_label(LabelDef(name="orient", version="1", arity=1, needs=frozenset({"poses"}), fn=_label_orient, prov="gt"))


# ------------------------------------------------------------------ label: contact_normal
def _label_contact_normal(view: StateView, idx: TokenIndex) -> Label:
    ids = _token_ids(idx)
    value = np.zeros((len(ids), 3), dtype=np.float64)
    valid = np.zeros((len(ids),), dtype=bool)
    outward: dict = {}
    for c in view.contacts():
        n = np.asarray(c.normal, dtype=np.float64)
        outward.setdefault(c.a, []).append(n)          # c.normal points a -> b: outward from a
        outward.setdefault(c.b, []).append(-n)          # outward from b is the opposite direction
    for i, eid in enumerate(ids):
        ns = outward.get(eid) if eid is not None else None
        if ns:
            v = np.mean(ns, axis=0)
            m = float(np.linalg.norm(v))
            if m > 1e-9:
                value[i] = v / m
                valid[i] = True
    return Label(value=value, valid=valid, prov="gt", version="1")


register_label(LabelDef(name="contact_normal", version="1", arity=1, needs=frozenset({"contacts"}),
                        fn=_label_contact_normal, prov="gt"))


# ------------------------------------------------------------------ label: cam_uvd (+ shared projection math)
def project(cam: Camera, points_world: np.ndarray) -> np.ndarray:
    """(u, v, depth) of world points through `cam`, from `Camera.K` / `Camera.T_world_cam` alone: normalized
    image-plane coordinates at the vertical FOV (`FieldDef("cam_uvd", 3, ..., units="uv[-1,1], m")`; `u = v = 0` on
    the principal ray) plus metric depth along the camera's forward (-Z) axis. `StateView`-only (backend-agnostic)
    -- this reproduces `rrp.envs.mujoco.sensors.project_points`'s convention for MuJoCo since both derive from the
    same pinhole model (`Camera.K`'s `fy = (h/2) / tan(fovy/2)`, `T_world_cam`'s columns are the camera's world
    axes), but never imports it, so it works for any backend that fills in a `Camera` the same way. Points behind
    the camera get `depth <= 0` and `u = v = 0`; callers gate on `depth > 0`."""
    R, t = cam.T_world_cam[:3, :3], cam.T_world_cam[:3, 3]
    pts = np.atleast_2d(np.asarray(points_world, dtype=np.float64))
    local = (pts - t) @ R                     # world -> camera-local (R's columns are the camera axes in world frame)
    depth = -local[:, 2]
    f = cam.K[1, 1] / (cam.height / 2.0)       # = 1 / tan(fovy / 2)
    safe = np.where(depth > 1e-9, depth, 1.0)
    u = np.where(depth > 1e-9, f * local[:, 0] / safe, 0.0)
    v = np.where(depth > 1e-9, f * local[:, 1] / safe, 0.0)
    return np.stack([u, v, depth], axis=-1)


def unproject(cam: Camera, u: float, v: float, depth: float) -> np.ndarray:
    """Exact inverse of `project`: the world point at normalized (u, v) at the given forward `depth` along `cam`'s
    ray. `project(cam, unproject(cam, u, v, depth)[None]) == (u, v, depth)` for any `depth > 0` (used by
    `camera_depth` to place objects at a chosen pixel and to keep that pixel fixed while varying depth)."""
    f = cam.K[1, 1] / (cam.height / 2.0)
    local = np.array([u * depth / f, v * depth / f, -depth])
    R, t = cam.T_world_cam[:3, :3], cam.T_world_cam[:3, 3]
    return t + R @ local


def _pixel_of(cam: Camera, u: float, v: float):
    """(row, col) of a normalized (u, v) in a `render_depth`-sized image of `cam`, or `None` if outside its frame."""
    col = int(round((u * 0.5 + 0.5) * (cam.width - 1)))
    row = int(round((v * 0.5 + 0.5) * (cam.height - 1)))
    if 0 <= row < cam.height and 0 <= col < cam.width:
        return row, col
    return None


def _label_cam_uvd(view: StateView, idx: TokenIndex, *, camera: str = DEFAULT_CAMERA) -> Label:
    ents = _entities_by_id(view)
    ids = _token_ids(idx)
    value = np.zeros((len(ids), 3), dtype=np.float64)
    valid = np.zeros((len(ids),), dtype=bool)
    if "camera" not in view.caps:
        return Label(value=value, valid=valid, prov="gt", version="1")
    cam = view.camera(camera)
    depth_img = view.render_depth(camera) if "depth_render" in view.caps else None
    for i, eid in enumerate(ids):
        e = ents.get(eid) if eid is not None else None
        if e is None:
            continue
        u, v, d = project(cam, e.pos[None, :])[0]
        if d <= 0:
            continue
        if depth_img is not None:
            px = _pixel_of(cam, u, v)
            if px is not None:
                d = float(depth_img[px])
        value[i] = (u, v, d)
        valid[i] = True
    return Label(value=value, valid=valid, prov="gt", version="1")


register_label(LabelDef(name="cam_uvd", version="1", arity=1, needs=frozenset({"poses", "camera"}),
                        fn=_label_cam_uvd, prov="gt"))


# ------------------------------------------------------------------ scene part: table_objects (docs 6 `geo.orient`)
def front_camera() -> Camera:
    """Literal reconstruction of `rrp.bodies.generators.workspace_spec`'s declared "front" camera (`pos=[1.25, 0,
    0.65]`, `xyaxes=[0,1,0, -0.45,0,0.9]`, `fovy=55`) at the default 64x64 `state_view()` depth size -- the camera
    every arm/dual table scenario declares -- so `camera_depth` can place objects along its ray without a live
    simulator. Cross-checked against a real arm session's `sv.camera("front")` in
    `tests/unit/test_relations_r14.py`."""
    pos = np.array([1.25, 0.0, 0.65])
    x_axis = np.array([0.0, 1.0, 0.0])
    y_axis = np.array([-0.45, 0.0, 0.9])
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    R = np.stack([x_axis, y_axis, z_axis], axis=1)     # columns = camera axes expressed in the world frame
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, pos
    w = h = 64
    fy = (h / 2.0) / math.tan(math.radians(55.0) / 2.0)
    K = np.array([[fy, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]])
    return Camera(name=DEFAULT_CAMERA, K=K, T_world_cam=T, width=w, height=h)


def _table_objects_build(draft: SceneDraft, rng: np.random.Generator) -> None:
    n = int(draft.kwargs.get("n_objects", 3))
    table_z = float(draft.kwargs.get("table_z", 0.0))
    x_lo, x_hi = draft.kwargs.get("x_range", (0.2, 0.5))
    y_lo, y_hi = draft.kwargs.get("y_range", (-0.3, 0.3))
    for i in range(n):
        draft.entities.append({"id": f"obj{i}", "kind": "object",
                               "pos": np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi), table_z]),
                               "yaw": float(rng.uniform(-math.pi, math.pi))})
    draft.active = draft.active | TABLE_OBJECTS.activates
    draft.parts = draft.parts + ("table_objects",)
    draft.provenance.setdefault("parts", []).append({"part": "table_objects", "version": TABLE_OBJECTS.version})


def _table_objects_vary(draft: SceneDraft, rng: np.random.Generator, factor: str, *, k: int = 2) -> list:
    """Decoupling pairs for `geo.orient` (docs 5.2/5.4/6): `k` copies of `draft` whose objects keep their `id` and
    `pos` but get a freshly-drawn `yaw` -- position and everything else stay byte-identical to `draft`."""
    if factor not in ("orient", "geo.orient"):
        raise ValueError(f"table_objects.vary: unsupported factor {factor!r} (expected 'orient')")
    out = []
    for _ in range(k):
        entities = [dict(e, yaw=float(rng.uniform(-math.pi, math.pi))) if e["kind"] == "object" else dict(e)
                   for e in draft.entities]
        out.append(replace(draft, entities=entities,
                           provenance={**draft.provenance, "vary": {"part": "table_objects", "factor": factor}}))
    return out


TABLE_OBJECTS = ScenePart(name="table_objects", version="1", activates=frozenset({"orient", "above"}),
                          build=_table_objects_build, vary=_table_objects_vary,
                          envs=("mujoco/arm", "mujoco/dual"))
register_part(TABLE_OBJECTS)


# ------------------------------------------------------------------ scene part: camera_depth (docs 6 `geo.depth3d`)
def _camera_depth_build(draft: SceneDraft, rng: np.random.Generator) -> None:
    cam = draft.kwargs.get("camera") or front_camera()
    u = float(draft.kwargs.get("pixel_u", rng.uniform(-0.5, 0.5)))
    v = float(draft.kwargs.get("pixel_v", rng.uniform(-0.5, 0.5)))
    depth = float(draft.kwargs.get("depth", rng.uniform(0.3, 1.5)))
    draft.entities.append({"id": "depth_probe", "kind": "object", "pos": unproject(cam, u, v, depth),
                           "pixel": (u, v), "depth": depth})
    draft.kwargs.setdefault("camera", cam)
    draft.active = draft.active | CAMERA_DEPTH.activates
    draft.parts = draft.parts + ("camera_depth",)
    draft.provenance.setdefault("parts", []).append({"part": "camera_depth", "version": CAMERA_DEPTH.version})


def _camera_depth_vary(draft: SceneDraft, rng: np.random.Generator, factor: str, *, k: int = 2,
                       depth_range: tuple = (0.3, 1.5)) -> list:
    """Decoupling pairs for `geo.depth3d` (docs 5.2/5.4/6: "same pixel, other depth"): `k` copies of `draft` whose
    `depth_probe` keeps its `pixel` (u, v) but moves to a freshly-drawn `depth` along the SAME camera ray
    (`unproject` is `project`'s exact inverse, so the projected pixel is unchanged by construction, not by luck)."""
    if factor not in ("depth", "geo.depth3d"):
        raise ValueError(f"camera_depth.vary: unsupported factor {factor!r} (expected 'depth')")
    probe = next((e for e in draft.entities if e["id"] == "depth_probe"), None)
    if probe is None:
        raise ValueError("camera_depth.vary: draft has no 'depth_probe' entity (call .build first)")
    cam = draft.kwargs.get("camera") or front_camera()
    u, v = probe["pixel"]
    out = []
    for _ in range(k):
        d = float(rng.uniform(*depth_range))
        entities = [dict(e, pos=unproject(cam, u, v, d), depth=d) if e["id"] == "depth_probe" else dict(e)
                   for e in draft.entities]
        out.append(replace(draft, entities=entities,
                           provenance={**draft.provenance, "vary": {"part": "camera_depth", "factor": factor}}))
    return out


CAMERA_DEPTH = ScenePart(name="camera_depth", version="1", activates=frozenset({"depth"}),
                         build=_camera_depth_build, vary=_camera_depth_vary,
                         envs=("mujoco/arm", "mujoco/dual"))
register_part(CAMERA_DEPTH)
