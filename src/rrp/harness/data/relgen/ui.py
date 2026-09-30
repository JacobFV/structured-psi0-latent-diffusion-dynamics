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
pair probe for a drag destination) and is privileged-only.

Never imports `rrp.policies.relations` (labels stay on the privileged side of the deploy boundary, docs section 7;
`policies/` never imports `relgen`)."""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from rrp.envs.base import StateView
from rrp.harness.data.relgen import (Label, LabelDef, Sample, SceneDraft, ScenePart, TokenIndex, register_label,
                                     register_part)

UI_LABELS_VERSION = "1"


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


def _label2(value: np.ndarray, valid: np.ndarray) -> Label:
    return Label(value=value[..., None], valid=valid, prov="gt", version=UI_LABELS_VERSION)


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
def dragged_widget_id(view: StateView) -> str | None:
    """The widget CW's own focus currently names (`scene.focus.interaction`, `cw_state_view`'s `focus_rank == 0`):
    the only public/privileged signal of "the widget the pointer is actively interacting with", so the one
    candidate for "currently being dragged" a 2D desktop UI exposes without task-specific plumbing."""
    for n in view.ui_tree():
        if n["focus_rank"] == 0:
            return n["id"]
    return None


def drag_to_fn(view: StateView, index: TokenIndex) -> Label:
    """The dragged widget (`dragged_widget_id`) -> the nearest OTHER widget by on-screen position: the candidate
    drop target of an in-progress drag (`leg.foothold`'s own nearest-candidate pattern, R19, applied here). Every
    pair of known widgets is a scored candidate (`valid`); only the dragged widget's row ever has a true (1.0)
    entry, and only when at least one other widget exists."""
    nodes = _ui_nodes(view)
    ents = {e.id: e for e in view.entities()}
    ids, valid = _pairwise(index, nodes)
    T = len(ids)
    value = np.zeros((T, T), dtype=np.float64)
    dragged = dragged_widget_id(view)
    if dragged is not None and dragged in ents and dragged in ids:
        a = ids.index(dragged)
        best_b, best_d = None, None
        for b, ib in enumerate(ids):
            if ib is None or ib not in ents or ib == dragged:
                continue
            d = float(np.linalg.norm(ents[dragged].pos - ents[ib].pos))
            if best_d is None or d < best_d:
                best_b, best_d = b, d
        if best_b is not None:
            value[a, best_b] = 1.0
    return _label2(value, valid)


register_label(LabelDef(name="same_window", version=UI_LABELS_VERSION, arity=2, needs=frozenset({"ui_tree"}),
                        fn=same_window_fn, prov="gt"))
register_label(LabelDef(name="label_for", version=UI_LABELS_VERSION, arity=2, needs=frozenset({"ui_tree"}),
                        fn=label_for_fn, prov="gt"))
register_label(LabelDef(name="focus_next", version=UI_LABELS_VERSION, arity=2, needs=frozenset({"ui_tree", "poses"}),
                        fn=focus_next_fn, prov="gt"))
register_label(LabelDef(name="above", version=UI_LABELS_VERSION, arity=2, needs=frozenset({"ui_tree"}),
                        fn=above_fn, prov="gt"))
register_label(LabelDef(name="drag_to", version=UI_LABELS_VERSION, arity=2, needs=frozenset({"ui_tree", "poses"}),
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
