"""R20 (D-144, docs/relations.md section 10 row R20; section 6 first wave / section 5.5 "composable now" list:
"ComputerWorld {label_for, contains, focus, z-order, drag}"): the PRIVILEGED labels of `catalog.py`'s `ui.*`
entries, written once against `rrp.envs.base.StateView` (cap "ui_tree", `+"poses"` for `drag_to`).

`label_for` / `focus_next` recompute, from `StateView.ui_tree()` alone, exactly the same relation
`envs.computerworld.ui_edges` builds from the raw scene (the deployable "given" side of the SAME `ui.*` entries) --
an independent StateView-side implementation, the same duplication R14's geometry labels accept for `geo.*`'s
"given" fields (docs 10 row R14: "labels on arm / dual fixtures match hand-computed values"), here cross-checked
against `ui_edges` itself (`tests/unit/test_relations_r20.py`). `same_window` / `above` recompute, from the same
`ui_tree()`, the `parent_id` / `zlayer` fields `envs.computerworld.ui_public_fields` builds from the raw scene --
`ui.same_window` / `ui.above` (catalog.py) read those PUBLIC fields directly through the generic `same` / `order`
operators (D-144 addendum), so these two labels are a StateView-side cross-check of `ui_public_fields`, not of the
retired `ui_edges` "contains" / "above" channels. `drag_to` has no "given" counterpart (docs 3.1: the bias IS the
pair probe for a drag destination) and is privileged-only: its label is the drag the scripted TEACHER performs (D-146 round 2
PC; the focus-widget-to-nearest-widget proxy it replaces labelled no real drag): handle -> the widget outside the dragged
window nearest to where the drag ends. The teacher's goal is not in `CWStateView`, so `teacher_drag_view(env)` wraps the
view with cap `teacher_drag`.

Never imports `rrp.policies.relations` (labels stay on the privileged side of the deploy boundary, docs section 7;
`policies/` never imports `relgen`)."""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from rrp.envs.base import CapabilityError, StateView
from rrp.harness.data.relgen import (Label, LabelDef, Sample, SceneDraft, ScenePart, TokenIndex, register_label,
                                     register_part)

UI_LABELS_VERSION = "1"
DRAG_TO_VERSION = "2"      # 1 = focused widget -> nearest widget (a proxy); 2 = the scripted teacher's drag target


def _ui_nodes(view: StateView) -> dict[str, dict]:
    """`{entity id -> ui_tree node}` (`docs 5.1`: id / parent / z / focus_rank / role / label / bounds / order --
    the last `order` filled by `envs.computerworld.cw_state_view`, this unit's own addition to it, D-144 R20)."""
    return {n["id"]: n for n in view.ui_tree()}


def _pairwise(index: TokenIndex, nodes: dict[str, dict]) -> tuple[list, np.ndarray]:
    """Ctx entity ids and the `[T, T]` "both slots are real, known ui_tree widgets" candidacy mask every arity-2 ui.*
    label shares (a pair is a scored candidate whenever both its tokens are; the VALUE says which pairs are true)."""
    ids = index.sets.get("ctx", [])
    known = np.array([i is not None and i in nodes for i in ids], dtype=bool)
    return ids, known[:, None] & known[None, :]


def _label2(value: np.ndarray, valid: np.ndarray, version: str = UI_LABELS_VERSION) -> Label:
    return Label(value=value[..., None], valid=valid, prov="gt", version=version)


# ------------------------------------------------------------------------------------------------ same_window
def same_window_fn(view: StateView, index: TokenIndex) -> Label:
    """1[i, j share a window] (reflexive, symmetric); desktop-level widgets (`parent is None`) never match each
    other -- mirrors `envs.computerworld.ui_public_fields`'s `parent_id` field exactly (`ui.same_window`'s generic
    `same` operator, D-144 addendum), off `ui_tree` instead of the raw scene table."""
    nodes = _ui_nodes(view)
    ids, valid = _pairwise(index, nodes)
    T = len(ids)
    value = np.zeros((T, T), dtype=np.float64)
    for a in range(T):
        ia = ids[a]
        if ia is None or ia not in nodes or nodes[ia]["parent"] is None:
            continue
        for b in range(T):
            ib = ids[b]
            if ib is not None and ib in nodes and nodes[ib]["parent"] == nodes[ia]["parent"]:
                value[a, b] = 1.0
    return _label2(value, valid)


# ------------------------------------------------------------------------------------------------ label_for
def label_for_fn(view: StateView, index: TokenIndex) -> Label:
    """role="label" widget i -> the widget immediately after it (`order`) in the same window (or both desktop-level,
    `parent is None`), that is itself not a label. Mirrors `ui_edges`'s "label_for" channel."""
    nodes = _ui_nodes(view)
    ids, valid = _pairwise(index, nodes)
    T = len(ids)
    value = np.zeros((T, T), dtype=np.float64)
    groups: dict = {}
    for t, i in enumerate(ids):
        if i is not None and i in nodes:
            groups.setdefault(nodes[i]["parent"], []).append(t)
    for members in groups.values():
        ordered = sorted(members, key=lambda t: nodes[ids[t]]["order"])
        for a, b in zip(ordered, ordered[1:]):
            if nodes[ids[a]]["role"] == "label" and nodes[ids[b]]["role"] != "label":
                value[a, b] = 1.0
    return _label2(value, valid)


# ------------------------------------------------------------------------------------------------ focus_next
def focus_next_fn(view: StateView, index: TokenIndex) -> Label:
    """i, j consecutive in the public tab order (focusable, enabled, boxed widgets by `order`; cyclic). Mirrors
    `ui_edges`'s "focus_next" channel; `StateView.ui_tree()` carries no `disabled` / `focusable` / `bounds is None`
    flags of its own beyond `bounds`, so this reads them off `EntityState.attrs` (`entities()`, cap "poses")."""
    nodes = _ui_nodes(view)
    ents = {e.id: e for e in view.entities()}
    ids, valid = _pairwise(index, nodes)
    T = len(ids)
    value = np.zeros((T, T), dtype=np.float64)
    order = []
    for t, i in enumerate(ids):
        if i is None or i not in nodes:
            continue
        a = ents[i].attrs or {}
        if a.get("focusable") == "true" and a.get("disabled") != "true" and nodes[i]["bounds"] is not None:
            order.append(t)
    order.sort(key=lambda t: nodes[ids[t]]["order"])
    for a, b in zip(order, order[1:] + order[:1]):
        value[a, b] = 1.0
    return _label2(value, valid)


# ------------------------------------------------------------------------------------------------ above
def above_fn(view: StateView, index: TokenIndex) -> Label:
    """zlayer_j > zlayer_i (which of the pair renders on top), by `zlayer` order alone. Mirrors
    `envs.computerworld.ui_public_fields`'s `zlayer` field (`ui.above`'s generic `order` operator, D-144
    addendum), not the retired `ui_edges` "above" channel."""
    nodes = _ui_nodes(view)
    ids, valid = _pairwise(index, nodes)
    z = np.array([nodes[i]["z"] if (i is not None and i in nodes) else -1 for i in ids], dtype=np.float64)
    value = (z[None, :] > z[:, None]).astype(np.float64)
    return _label2(value, valid)


# ------------------------------------------------------------------------------------------------ drag_to
TEACHER_DRAG_CAP = "teacher_drag"


def nearest_outside_window(dest_px, nodes: Sequence[dict], window: str | None) -> str | None:
    """The drop target of a window drag: among the `ui_tree` nodes that have a box and are not in `window` (nor the
    window's own node), the id whose box centre is nearest to the drag's destination pixel. Ties go to the lower scene
    `order`, then the id, so the choice is a pure function of the scene. None when nothing qualifies."""
    best, key = None, None
    for n in nodes:
        if n["bounds"] is None or n["parent"] == window or n["id"] == window:
            continue
        x0, y0, x1, y1 = n["bounds"]
        d = float(np.hypot((x0 + x1) / 2 - dest_px[0], (y0 + y1) / 2 - dest_px[1]))
        k = (d, n["order"], n["id"])
        if key is None or k < key:
            best, key = n["id"], k
    return best


def teacher_drag_target(env) -> dict | None:
    """The drag the scripted teacher of `env` performs (`policies.teachers.computerworld._drag`), as widget ids:
    `{"src": title-bar handle, "dst": nearest widget outside the dragged window to where the drag ends, "dest_px",
    "src_slot", "dst_slot"}`; None when the task has no drag (the teacher never drags: a real negative). Read from the
    env's goal and scene, i.e. privileged: a label, never a policy input. `env.state_view()` assigns the slots the
    policy's own tables use (`SlotRegistry`), so the slots are the stored packs' widget indices."""
    from rrp.envs.computerworld import find_widget
    if env.task_name != "cw/drag_window":
        return None
    g = env.goal
    w = find_widget(env, lambda w: w["interaction"] == f"window:{g['window']}:drag")
    if w is None or w["box"] is None:
        raise ValueError(f"{env.task_name}: no drag handle for window {g['window']!r} in the scene")
    view = env.state_view()
    x0, y0, x1, y1 = w["box"]
    dest = ((x0 + x1) // 2 + g["dx"], (y0 + y1) // 2 + g["dy"])
    src_slot = env.slots.slots[w["key"]]
    dst = nearest_outside_window(dest, view.ui_tree(), f"window:{w['window']}")
    dst_slot = None if dst is None else next(sl for sl in env.slots.slots.values()
                                              if view.token_entity("widgets", sl) == dst)
    return dict(src=view.token_entity("widgets", src_slot), dst=dst, dest_px=[int(dest[0]), int(dest[1])],
                src_slot=int(src_slot), dst_slot=None if dst_slot is None else int(dst_slot))


class TeacherDragView:
    """A `StateView` plus cap `teacher_drag` and `teacher_drag()` (-> `teacher_drag_target`'s dict, or None when the
    episode's teacher never drags). Everything else is the wrapped view's own; `drag_to_fn` reads it."""

    def __init__(self, view: StateView, drag: dict | None):
        self._view, self._drag = view, drag

    @property
    def caps(self) -> frozenset:
        return frozenset(self._view.caps) | {TEACHER_DRAG_CAP}

    def teacher_drag(self) -> dict | None:
        return None if self._drag is None else dict(self._drag)

    def __getattr__(self, name):
        return getattr(self._view, name)


def teacher_drag_view(env) -> TeacherDragView:
    """`env.state_view()` with the env's teacher drag attached (the ComputerWorld StateView has no goal of its own)."""
    return TeacherDragView(env.state_view(), teacher_drag_target(env))


def drag_to_fn(view: StateView, index: TokenIndex) -> Label:
    """`ui.drag_to` from the TEACHER's drag (`view.teacher_drag()`): pair (handle, drop target) is true. Every pair of
    known widgets is a scored candidate; an episode whose teacher never drags has no true pair (a real negative). When the
    teacher drags but its handle or drop target is not among the context tokens, the sample carries no label (nothing is
    valid): a missing target is never a negative."""
    if TEACHER_DRAG_CAP not in view.caps:
        raise CapabilityError(f"ui.drag_to needs the teacher's drag target (cap {TEACHER_DRAG_CAP!r}); "
                              f"view caps {sorted(view.caps)}")
    nodes = _ui_nodes(view)
    ids, valid = _pairwise(index, nodes)
    value = np.zeros((len(ids), len(ids)), dtype=np.float64)
    td = view.teacher_drag()
    if td is not None:
        if td["src"] in ids and td["dst"] in ids:
            value[ids.index(td["src"]), ids.index(td["dst"])] = 1.0
        else:
            valid = np.zeros_like(valid)
    return _label2(value, valid, DRAG_TO_VERSION)


register_label(LabelDef(name="same_window", version=UI_LABELS_VERSION, arity=2, needs=frozenset({"ui_tree"}),
                        fn=same_window_fn, prov="gt"))
register_label(LabelDef(name="label_for", version=UI_LABELS_VERSION, arity=2, needs=frozenset({"ui_tree"}),
                        fn=label_for_fn, prov="gt"))
register_label(LabelDef(name="focus_next", version=UI_LABELS_VERSION, arity=2, needs=frozenset({"ui_tree", "poses"}),
                        fn=focus_next_fn, prov="gt"))
register_label(LabelDef(name="above", version=UI_LABELS_VERSION, arity=2, needs=frozenset({"ui_tree"}),
                        fn=above_fn, prov="gt"))
register_label(LabelDef(name="drag_to", version=DRAG_TO_VERSION, arity=2, needs=frozenset({"ui_tree", TEACHER_DRAG_CAP}),
                        fn=drag_to_fn, prov="gt"))


# ------------------------------------------------------------------------------------------------ reveal / surprise glue (docs 5.4; R9 transforms reused unmodified)
def label_for_sample(control_id: str, candidates: Sequence[str], view: StateView, *,
                     prior: dict | None = None, evidence: Sequence[dict] | None = None) -> Sample:
    """A `Sample`-shaped dict ready for `TRANSFORMS["reveal"]` / `TRANSFORMS["surprise"]` (R9, unmodified):
    `candidates` names the label-role widget ids one window offers as `control_id`'s possible `label_for` source
    (docs 5.4's own worked surprise example, "UI label changes" -- which candidate a control is actually bound to
    is exactly this: a classification over co-windowed label-role widgets, not a fixed fact). When the caller does
    not supply `evidence`, the CURRENT ground truth (`label_for_fn`'s own relation, restricted to `control_id`'s
    row) supplies a single `t=0` exclusion event ruling out every OTHER candidate -- but only once one is actually
    bound (matching `harness.data.relgen.task.next_contact_sample`'s identical "no evidence yet" vs. "evidence
    excludes everything" distinction)."""
    nodes = _ui_nodes(view)
    bound = None
    if control_id in nodes:
        window = nodes[control_id]["parent"]
        same_window = [c for c in candidates if c in nodes and nodes[c]["parent"] == window]
        ordered = sorted(same_window, key=lambda c: nodes[c]["order"])
        prior_label = [c for c in ordered if nodes[c]["order"] < nodes[control_id]["order"]]
        if prior_label and nodes[prior_label[-1]]["role"] == "label":
            bound = prior_label[-1]
    if evidence is None:
        evidence = []
        if bound is not None:
            excluded = [c for c in candidates if c != bound]
            if excluded:
                evidence = [{"t": 0, "excludes": excluded}]
    return {"inputs": {"candidates": list(candidates), "prior": prior, "evidence": list(evidence)},
            "labels": {}, "provenance": {"label": "label_for", "prov": "gt", "version": UI_LABELS_VERSION,
                                         "control": control_id}}


# ------------------------------------------------------------------------------------------------ ComputerWorld scene parts
# The scene-level knobs `envs.computerworld.make_env(scene=...)` takes (`width`, `height`, `m_per_px`, `depth`, `dz`): the
# `compose` side of the ui family (D-146 C3, docs/relations.md 5.2). A CW scene is authored by the task (its widgets and
# windows), so these parts add no entities: they set the SCENE kwargs a draft hands to `make_env`, and their `vary` gives
# the decoupling pairs (same task and widgets, only the varied factor differs). `activates` names the dynamics a target
# active set asks `compose` for (`viewport`, `zstack`); the catalog's `ui.*` `gen` lists the PART names.
CW_ENV = "computerworld"
CW_VIEWPORTS = ((960, 640, 0.001), (1280, 800, 0.00075), (800, 600, 0.0012))   # (width px, height px, m per px)
CW_DEPTHS = ("stack", "constant")


def _stamp(draft: SceneDraft, part: ScenePart) -> None:
    """Record the part on the draft once (`compose` has already listed the chosen parts; a direct `.build` has not)."""
    draft.active = draft.active | part.activates
    if part.name not in draft.parts:
        draft.parts = draft.parts + (part.name,)
    if not any(p["part"] == part.name for p in draft.provenance.setdefault("parts", [])):
        draft.provenance["parts"].append({"part": part.name, "version": part.version})


def _viewport_build(draft: SceneDraft, rng: np.random.Generator) -> None:
    if "width" not in draft.kwargs:
        w, h, m = CW_VIEWPORTS[int(rng.integers(len(CW_VIEWPORTS)))]
        draft.kwargs.update(width=w, height=h, m_per_px=m)
    _stamp(draft, CW_VIEWPORT)


def _viewport_vary(draft: SceneDraft, rng: np.random.Generator, factor: str) -> list[SceneDraft]:
    """One draft per OTHER viewport: the same widgets rendered into a different pixel frame (the relations are
    invariant, the pixel geometry is not)."""
    if factor not in ("viewport", "ui.viewport"):
        raise ValueError(f"cw_viewport.vary: unsupported factor {factor!r} (expected 'viewport')")
    cur = (draft.kwargs.get("width"), draft.kwargs.get("height"), draft.kwargs.get("m_per_px"))
    return [replace(draft, kwargs={**draft.kwargs, "width": w, "height": h, "m_per_px": m},
                    provenance={**draft.provenance, "vary": {"part": "cw_viewport", "factor": factor}})
            for w, h, m in CW_VIEWPORTS if (w, h, m) != cur]


def _depth_build(draft: SceneDraft, rng: np.random.Generator) -> None:
    if "depth" not in draft.kwargs:
        draft.kwargs["depth"] = CW_DEPTHS[int(rng.integers(len(CW_DEPTHS)))]
    draft.kwargs.setdefault("dz", 0.002)
    _stamp(draft, CW_DEPTH)


def _depth_vary(draft: SceneDraft, rng: np.random.Generator, factor: str) -> list[SceneDraft]:
    """The same scene with the other z-encoding: "stack" ranks the z-layers (`ui.above` is readable off the pose z),
    "constant" flattens them (only the public `zlayer` field carries the order)."""
    if factor not in ("zstack", "ui.above"):
        raise ValueError(f"cw_depth.vary: unsupported factor {factor!r} (expected 'zstack')")
    cur = draft.kwargs.get("depth")
    return [replace(draft, kwargs={**draft.kwargs, "depth": d},
                    provenance={**draft.provenance, "vary": {"part": "cw_depth", "factor": factor}})
            for d in CW_DEPTHS if d != cur]


CW_VIEWPORT = ScenePart(name="cw_viewport", version=UI_LABELS_VERSION, activates=frozenset({"viewport"}),
                        build=_viewport_build, vary=_viewport_vary, envs=(CW_ENV,))
CW_DEPTH = ScenePart(name="cw_depth", version=UI_LABELS_VERSION, activates=frozenset({"zstack"}),
                     build=_depth_build, vary=_depth_vary, envs=(CW_ENV,))
register_part(CW_VIEWPORT)
register_part(CW_DEPTH)
