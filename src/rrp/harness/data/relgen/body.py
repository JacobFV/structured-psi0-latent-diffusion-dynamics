"""Legged label family (D-144 unit R19; design: docs/relations.md sections 5-6, 10 row R19; catalog:
`research/relations_catalog.md` B "locomotion": "footholds, COM <-> support polygon, stability margin, tipping,
balancing, center of pressure"). Registers the two labels `catalog.py`'s `§legged` entries need --
`foothold_next` (`leg.foothold`) and `com_support` (`leg.com_support`) -- the supporting per-foot `foot_contacts`
label both are built from, and the scene part `terrain_steps`.

Every label is a pure `(StateView, TokenIndex) -> Label` function (`rrp.harness.data.relgen`) over
`rrp.envs.base.StateView`: feet are the entities a legged `state_view()` reports as `leg:<body>`
(`envs.mujoco.legged.LeggedSession._extra_state_entities`, kind "link"); the robot's base/pelvis is the assembly
entity "body" (its IMU tcp site) when the backend's `state_view()` reports one. `StateView` is a snapshot
interface and no backend today emits `terrain`-cap cells, so `foothold_next` degrades to an empty (all-invalid)
label rather than raising when no candidate cell entity is present -- matching docs 5.1: "a label written against
the view runs in every env whose caps cover its needs"; an env that never populates the ids a label looks for is a
needs gap, not an error, exactly like R16/R17's parts before a live env's scenario builder is wired to `compose`.

This unit's other owned file, `envs/warp/task_env.py`, is where the Warp height scan (`WarpStepsEnv.extra_obs` /
`WarpGapEnv.extra_obs`) stops feeding the deployable `observe()` vec and starts feeding `privileged()` only (the
"privileged layout test", docs row R19's acceptance line): that fix is independent of this module -- the Warp scan
stays a private, backend-specific grid (`envs/mujoco/humanoid_scenes.SCAN_X` / `SCAN_Y`, still read only by
`WarpStepsEnv` itself), not imported here. `terrain_steps` below declares its OWN smaller, backend-agnostic
candidate-cell grid (`SCAN_AHEAD` x `SCAN_SIDE`, a coarser approximation in the same body-frame convention -- ahead
of / beside the body, in multiples of leg length `L`) for MuJoCo-family legged fixtures, which have no Warp
staircase model to match cell-for-cell; the two grids are deliberately not the same numbers.
"""
from __future__ import annotations

import numpy as np

from rrp.envs.base import StateView
from rrp.harness.data.relgen import (Label, LabelDef, SceneDraft, ScenePart, TokenIndex, register_label,
                                     register_part)

__all__ = ["foot_ids", "stance_feet", "foot_contacts_fn", "foothold_next_fn", "com_support_fn",
          "support_polygon_margin", "terrain_steps_build", "terrain_steps_vary"]

VERSION = "1"
FOOT_PREFIX = "leg:"                 # envs.mujoco.legged.LeggedSession._extra_body_entity_map / _extra_state_entities
BODY_ID = "body"                     # the legged robot's single assembly entity (IMU tcp site), R4's `also=["body"]`
FOOTHOLD_CELL_KIND = "foothold_cell"
# `terrain_steps`'s candidate-cell layout: multiples of leg length L ahead of / beside the body, yaw frame -- the
# same body-frame CONVENTION `envs/mujoco/humanoid_scenes.SCAN_X` / `SCAN_Y` and `WarpStepsEnv.extra_obs` use
# privately, but a coarser, independent grid (4 x 3, not 11 x 3, different offsets); not imported from there (this
# module stays backend-agnostic -- a MuJoCo legged fixture has no Warp staircase model).
SCAN_AHEAD = (0.2, 0.5, 0.9, 1.2)
SCAN_SIDE = (-0.15, 0.0, 0.15)


def _ctx_ids(idx: TokenIndex) -> list:
    if "ctx" not in idx.sets:
        raise KeyError(f"legged labels need a 'ctx' token index; got {sorted(idx.sets)}")
    return idx.sets["ctx"]


def foot_ids(ids) -> list:
    """The `leg:*` slots of a token-index id list (order-preserving, no de-dup needed -- `ids` is already one
    token per slot)."""
    return [i for i in ids if i and i.startswith(FOOT_PREFIX)]


def stance_feet(view: StateView) -> set:
    """Foot entity ids with >= 1 contact record right now (`StateView.contacts`; the swing/stance split
    `WarpTrackerEnv._stance` makes from geom-level contact today, read here at the entity level instead)."""
    out = set()
    for c in view.contacts():
        if c.a.startswith(FOOT_PREFIX):
            out.add(c.a)
        if c.b.startswith(FOOT_PREFIX):
            out.add(c.b)
    return out


# ------------------------------------------------------------------ label `foot_contacts` (per-foot, arity 1)
def foot_contacts_fn(view: StateView, idx: TokenIndex) -> Label:
    """Per-foot contact / stance indicator (docs row R19 brief: "per-foot contact"). Arity 1: one bit per foot
    token (1.0 iff `stance_feet` names it), distinct from R16's pairwise `contact_pairs` (a relation between TWO
    entities) -- a foot's own contact state is a property of that one token. Non-foot slots are invalid (never
    scored); registered standalone (no `catalog.py` entry owns it directly) as the internal building block
    `foothold_next_fn` / `com_support_fn` below both use, matching R17's `support_matrix` (an internal helper
    `ix.force_flow`'s label is built from, not itself a `FactorDef`)."""
    ids = _ctx_ids(idx)
    stance = stance_feet(view)
    T = len(ids)
    value = np.zeros((T, 1), dtype=np.float32)
    valid = np.zeros((T,), dtype=bool)
    for i, a in enumerate(ids):
        if a is None or not a.startswith(FOOT_PREFIX):
            continue
        valid[i] = True
        if a in stance:
            value[i, 0] = 1.0
    return Label(value=value, valid=valid, prov="gt", version=VERSION)


register_label(LabelDef(name="foot_contacts", version=VERSION, arity=1, needs=frozenset({"contacts"}),
                        fn=foot_contacts_fn, prov="gt"))


# ------------------------------------------------------------------ label `foothold_next` (foot -> cell, arity 2)
def foothold_next_fn(view: StateView, idx: TokenIndex) -> Label:
    """Directed foot -> nearest terrain-cell entity (docs row R19 brief: "next foothold cell"; `leg.foothold`'s
    supervision). Terrain-cell entities are anything the token index carries with `EntityState.kind ==
    "foothold_cell"` (declared by `terrain_steps` below); for every foot currently in SWING (not in
    `stance_feet` -- a planted foot already has a foothold, not a "next" one), the true pair is the cell nearest
    it in planar (xy) world distance. Pure geometry once `entities()` is read: this label SUPERVISES a learned
    estimate (`leg.foothold`'s bilinear pair readout), it is never itself a policy input (docs/relations.md 5.1).
    No candidate cell in the index -> every pair simply stays invalid (see module docstring)."""
    ids = _ctx_ids(idx)
    ents = {e.id: e for e in view.entities()}
    stance = stance_feet(view)
    feet = [i for i in foot_ids(ids) if i in ents]
    cells = [i for i in ids if i and i in ents and ents[i].kind == FOOTHOLD_CELL_KIND]
    true_pairs = set()
    for f in feet:
        if f in stance:
            continue
        fp = np.asarray(ents[f].pos, dtype=np.float64)[:2]
        best, best_d = None, np.inf
        for c in cells:
            d = float(np.linalg.norm(np.asarray(ents[c].pos, dtype=np.float64)[:2] - fp))
            if d < best_d:
                best_d, best = d, c
        if best is not None:
            true_pairs.add((f, best))
    T = len(ids)
    value = np.zeros((T, T, 1), dtype=np.float32)
    valid = np.zeros((T, T), dtype=bool)
    feet_set, cells_set = set(feet), set(cells)
    for i, a in enumerate(ids):
        if a not in feet_set:
            continue
        for j, b in enumerate(ids):
            if b not in cells_set:
                continue
            valid[i, j] = True
            if (a, b) in true_pairs:
                value[i, j, 0] = 1.0
    return Label(value=value, valid=valid, prov="gt", version=VERSION)


register_label(LabelDef(name="foothold_next", version=VERSION, arity=2, needs=frozenset({"poses", "contacts"}),
                        fn=foothold_next_fn, prov="gt"))


# ------------------------------------------------------------------ label `com_support` (body, arity 1)
def _convex_hull(pts: np.ndarray) -> np.ndarray:
    """Monotone-chain convex hull of >= 1 planar points (deduplicated); < 3 distinct points come back as-is (the
    degenerate "polygon" `support_polygon_margin` handles as a point / segment)."""
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]
    uniq = [pts[0]]
    for p in pts[1:]:
        if not np.allclose(p, uniq[-1]):
            uniq.append(p)
    pts = np.array(uniq)
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return np.array(hull) if len(hull) >= 3 else pts


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    t = 0.0 if denom < 1e-12 else float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _point_in_polygon(p: np.ndarray, hull: np.ndarray) -> bool:
    inside = False
    n = len(hull)
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        if (a[1] > p[1]) != (b[1] > p[1]):
            x = (b[0] - a[0]) * (p[1] - a[1]) / ((b[1] - a[1]) or 1e-12) + a[0]
            if p[0] < x:
                inside = not inside
    return inside


def support_polygon_margin(com_xy, foot_xy: list) -> float:
    """Signed planar margin of `com_xy` inside the convex hull of `foot_xy` (docs row R19 brief: "COM projection
    inside the support polygon"): positive = inside (distance to the nearest hull edge), negative = outside
    (negated distance to the nearest edge / point / segment). 0 feet -> -inf (no support at all, never "inside"
    anything); 1 foot -> negated distance to that point; 2 feet (collinear) -> signed distance to the segment,
    always <= 0 (a segment has no interior). Pure geometry, no simulator."""
    pts = np.asarray(foot_xy, dtype=np.float64).reshape(-1, 2)
    p = np.asarray(com_xy, dtype=np.float64)
    if len(pts) == 0:
        return float("-inf")
    if len(pts) == 1:
        return -float(np.linalg.norm(p - pts[0]))
    hull = _convex_hull(pts)
    if len(hull) < 3:
        a, b = hull[0], hull[-1]
        return -_point_segment_distance(p, a, b)
    edge_d = min(_point_segment_distance(p, hull[i], hull[(i + 1) % len(hull)]) for i in range(len(hull)))
    return edge_d if _point_in_polygon(p, hull) else -edge_d


_NO_SUPPORT_MARGIN = -10.0   # 0-foot stance: large fixed negative margin, keeps the label finite / regressable


def com_support_fn(view: StateView, idx: TokenIndex) -> Label:
    """`leg.com_support`'s label: `support_polygon_margin` of the robot's base/COM proxy (the "body" assembly
    entity's own position -- the same base-frame approximation `WarpTrackerEnv._stance` / `extra_obs` already use
    in place of a true multi-body COM, since no generic `StateView` capability exposes a subtree centre of mass)
    inside the convex hull of every foot currently in `stance_feet`. Written on the "ctx" slot holding "body"
    (docs `ReadoutDef.address="asm"`, matching `probes:legged-v1`'s convention of one value read at the sample's
    body-assembly row); every other slot -- and every slot at all when no "body" entity is in view -- is invalid."""
    ids = _ctx_ids(idx)
    ents = {e.id: e for e in view.entities()}
    body = ents.get(BODY_ID)
    T = len(ids)
    value = np.zeros((T, 1), dtype=np.float32)
    valid = np.zeros((T,), dtype=bool)
    if body is None:
        return Label(value=value, valid=valid, prov="gt", version=VERSION)
    stance = stance_feet(view)
    foot_xy = [np.asarray(ents[f].pos, dtype=np.float64)[:2] for f in foot_ids(ids) if f in stance and f in ents]
    margin = support_polygon_margin(np.asarray(body.pos, dtype=np.float64)[:2], foot_xy)
    if not np.isfinite(margin):
        margin = _NO_SUPPORT_MARGIN
    for i, a in enumerate(ids):
        if a == BODY_ID:
            valid[i] = True
            value[i, 0] = float(margin)
    return Label(value=value, valid=valid, prov="gt", version=VERSION)


register_label(LabelDef(name="com_support", version=VERSION, arity=1, needs=frozenset({"poses", "contacts"}),
                        fn=com_support_fn, prov="gt"))


# ------------------------------------------------------------------ part `terrain_steps`
def terrain_steps_build(draft: SceneDraft, rng: np.random.Generator) -> None:
    """Declares `len(SCAN_AHEAD) x len(SCAN_SIDE)` foothold-cell entities on a staircase ahead of the robot
    (docs 5.2; `research/relations_catalog.md` "mujoco/legged, humanoid | terrain_steps | {foot contact} x
    {foothold} x {com_support}"): one row of cells per `SCAN_AHEAD` distance, each at a resampled step height in
    `[0, draft.kwargs["terrain_h_max"]]` (default 0.3), scaled by `draft.kwargs["terrain_leg_length"]` (default
    1.0). Declarative only -- a live env's own scenario builder resolves the draft into real geometry, matching
    R16 `grasp_target` / R17 `stack`'s "wiring `compose` into a live env is a later unit's job, not this one's"."""
    L = float(draft.kwargs.get("terrain_leg_length", 1.0))
    h_max = float(draft.kwargs.get("terrain_h_max", H_MAX_DEFAULT))
    tag = f"terrain{sum(1 for p in draft.parts if p == 'terrain_steps')}"
    cells, heights = [], []
    for ai, ahead in enumerate(SCAN_AHEAD):
        h = float(rng.uniform(0.0, h_max))
        heights.append(h)
        for si, side in enumerate(SCAN_SIDE):
            cid = f"{tag}_cell_{ai}_{si}"
            cells.append({"id": cid, "kind": FOOTHOLD_CELL_KIND, "pos": [ahead * L, side * L, h]})
    draft.entities = list(draft.entities) + cells
    draft.kwargs = dict(draft.kwargs, terrain_steps={"cells": [c["id"] for c in cells], "heights": heights,
                                                      "h_max": h_max, "L": L})
    draft.active = draft.active | {"terrain"}
    draft.parts = draft.parts + ("terrain_steps",)
    draft.provenance = dict(draft.provenance, terrain_steps={"version": VERSION, "n": len(cells), "h_max": h_max})


H_MAX_DEFAULT = 0.3


def terrain_steps_vary(draft: SceneDraft, rng: np.random.Generator, factor: str) -> list:
    """Decoupling pairs for `factor` (docs 5.2: "k copies differing ONLY in the named factor's value").
    `factor == "height"`: every row's step height resampled, cell count / xy layout / ids / every other kwarg,
    entity and event held fixed -- the pair that lets `com_support` / `foothold_next` toggle without any other
    scene change confounding it. Requires `terrain_steps_build` to have run on `draft` already."""
    if factor != "height":
        raise ValueError(f"terrain_steps: no vary axis {factor!r} (only 'height')")
    spec = draft.kwargs.get("terrain_steps")
    if not spec:
        raise ValueError("terrain_steps.vary: call build() first (no terrain_steps kwargs on this draft)")
    n_rows = len(SCAN_AHEAD)
    new_heights = [float(rng.uniform(0.0, spec["h_max"])) for _ in range(n_rows)]
    L = spec["L"]
    by_row: dict = {}
    for cid in spec["cells"]:
        ai = int(cid.rsplit("_", 2)[-2])
        by_row.setdefault(ai, []).append(cid)
    new_entities = []
    for e in draft.entities:
        if e.get("kind") != FOOTHOLD_CELL_KIND or e["id"] not in spec["cells"]:
            new_entities.append(e)
            continue
        ai = int(e["id"].rsplit("_", 2)[-2])
        new_entities.append({**e, "pos": [e["pos"][0], e["pos"][1], new_heights[ai] * (e["pos"][2] / max(1e-9, e["pos"][2]))]
                             if e["pos"][2] else {**e}["pos"]})
    # the line above only rewrites z for cells whose original height was non-zero (a 0-height cell keeps 0, since
    # "pos[2] / pos[2]" would be 0/0); do the straightforward, readable version instead of that clever one-liner:
    new_entities = []
    for e in draft.entities:
        if e.get("kind") != FOOTHOLD_CELL_KIND or e["id"] not in spec["cells"]:
            new_entities.append(e)
            continue
        ai = int(e["id"].rsplit("_", 2)[-2])
        new_entities.append({**e, "pos": [e["pos"][0], e["pos"][1], new_heights[ai]]})
    from dataclasses import replace
    new_kwargs = dict(draft.kwargs, terrain_steps=dict(spec, heights=new_heights))
    return [replace(draft, entities=new_entities, kwargs=new_kwargs)]


register_part(ScenePart(name="terrain_steps", version=VERSION, activates=frozenset({"terrain", "foothold", "com_support"}),
                        build=terrain_steps_build, vary=terrain_steps_vary, envs=("mujoco/legged", "warp/legged")))
