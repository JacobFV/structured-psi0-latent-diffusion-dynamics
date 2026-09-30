"""ComputerWorld as a 3D rrp environment (docs/architecture.md section 5; D-140). Optional extra `computerworld`.

The `computerworld` PyO3 wheel (github.com/JacobFV/computerworld, MIT; pinned 0.2.0) is imported lazily in the env
constructor; the mapping functions (screen <-> metre, widgets -> descriptors with depth, occlusion and stable slots,
command -> CW actions) never import it. The env builds a one-machine Ubuntu-themed desktop world inline (plus a simulated
static web form for `cw/fill_form`); tasks set up owner-side (launching an app is task setup, not a policy action).
Task setups and judges live here (they read CW scenes); `rrp.tasks.spec` registers the `cw/*` TaskSpecs, and the
scripted teachers are `rrp.policies.teachers.computerworld`.

Measured on computerworld 0.2.0 (see research/tracks/cworld.md): scene node ids change across layout revisions, so a
slot is keyed on (interaction, role, label, occurrence); depth "stack" ranks the scene's distinct z-layers (dense rank x
dz); within a layer later insertion occludes, as CW's own hit test does.
"""
from __future__ import annotations

import random
import re
import string
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np

from rrp.bodies.fixtures import cw_pointer_spec
from rrp.core.action import NativeCommand
from rrp.core.base import content_hash
from rrp.core.observation import (ImageObs, NodeState, ObjectDescriptor, PolicyObservation, PrivilegedTruth,
                                  SensorChannel)
from rrp.core.refs import EntityRef
from rrp.envs.base import ActionSpace, BodyInfo, CapabilityError, EntityState, EnvSpec, StepResult
from rrp.tasks.spec import Judgement

ADAPTER_VERSION = "cw_env.v2"          # v2 (D-142): 24 words / 24 names for cw/open_type, cw/fill_form
CONTROLLER_VERSION = "cw_pointer.v1"
CW_VERSION = "0.2.0"          # pinned wheel; snapshots and pixels are only valid within one engine version
MACHINE, ACTOR = "pc", "ada"


# ------------------------------------------------------------------------------------------------ screen geometry
@dataclass(frozen=True)
class ScreenFrame:
    """World frame: x right, y up, z toward the viewer, origin at the screen centre, 1 px = m_per_px metres."""
    width: int = 960
    height: int = 640
    m_per_px: float = 0.001
    depth: Literal["constant", "stack"] = "stack"
    dz: float = 0.002
    hover_z: float = 0.05            # the pointer's fixed slide z (above every stacked layer)

    def px_to_m(self, u: float, v: float) -> tuple[float, float]:
        return ((u - self.width / 2) * self.m_per_px, (self.height / 2 - v) * self.m_per_px)

    def m_to_px(self, x: float, y: float) -> tuple[int, int]:
        u = int(round(x / self.m_per_px + self.width / 2))
        v = int(round(self.height / 2 - y / self.m_per_px))
        return min(max(u, 0), self.width - 1), min(max(v, 0), self.height - 1)

    def limits_m(self) -> tuple[list[float], list[float]]:
        (x0, y1), (x1, y0) = self.px_to_m(0, 0), self.px_to_m(self.width - 1, self.height - 1)
        return [x0, y0], [x1, y1]

    def as_dict(self) -> dict:
        return {"units": "m", "up": "+y", "toward_viewer": "+z", "origin": "screen_centre", "m_per_px": self.m_per_px,
                "screen_px": [self.width, self.height], "depth": self.depth, "dz": self.dz, "hover_z": self.hover_z}


def node_box(node: dict) -> tuple[int, int, int, int]:
    """Scene-pixel box (x0, y0, x1, y1): CW bounds are local, the transform maps them into the scene."""
    b, t = node["bounds"], node.get("transform") or {"a": 1024, "b": 0, "c": 0, "d": 1024, "tx": 0, "ty": 0}
    xs, ys = [], []
    for lx, ly in ((b["x"], b["y"]), (b["x"] + b["width"], b["y"]), (b["x"], b["y"] + b["height"]),
                   (b["x"] + b["width"], b["y"] + b["height"])):
        xs.append((t["a"] * lx + t["c"] * ly) // 1024 + t["tx"])
        ys.append((t["b"] * lx + t["d"] * ly) // 1024 + t["ty"])
    return min(xs), min(ys), max(xs), max(ys)


def _clip(box, clip: dict | None):
    if not clip:
        return box
    x0, y0 = max(box[0], clip["x"]), max(box[1], clip["y"])
    x1, y1 = min(box[2], clip["x"] + clip["width"]), min(box[3], clip["y"] + clip["height"])
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


OCCLUDER_KINDS = {"box", "rounded_box", "backdrop", "asset_image"}   # opaque fills; text, paths, shadows never occlude


def scene_widgets(scene: dict) -> list[dict]:
    """Semantic nodes of a CW scene -> widget records, occluded ones included with visible=False.

    Record: key (slot key), entity ("role:label"), role, label, value, disabled, focusable, focused, interaction,
    window, box (clipped scene px or None), layer (dense rank of z), order (scene index), visible."""
    nodes = scene["nodes"]
    layers = {z: i for i, z in enumerate(sorted({n["z"] for n in nodes}))}
    occluders = []
    for i, n in enumerate(nodes):
        if n["primitive"].get("kind") in OCCLUDER_KINDS and n.get("opacity", 255) == 255:
            box = _clip(node_box(n), n.get("clip"))
            if box:
                occluders.append(((n["z"], i), box))
    focus = (scene.get("focus") or {}).get("interaction")
    seen: dict[str, int] = {}
    out = []
    for i, n in enumerate(nodes):
        sem = n.get("semantic")
        if not sem:
            continue
        inter = n.get("interaction") or ""
        base = f"{inter}|{sem.get('role')}|{sem.get('label')}"
        k = seen.get(base, 0)
        seen[base] = k + 1
        box = _clip(node_box(n), n.get("clip"))
        rank = (n["z"], i)
        covered = box is None or any(r > rank and o[0] <= box[0] and o[1] <= box[1] and o[2] >= box[2] and o[3] >= box[3]
                                     for r, o in occluders)
        m = re.match(r"window:(\d+):", inter)
        out.append(dict(key=base + (f"#{k}" if k else ""), entity=f"{sem.get('role')}:{sem.get('label')}",
                        role=sem.get("role") or "", label=sem.get("label") or "", value=sem.get("value"),
                        disabled=bool(sem.get("disabled")), focusable=bool(sem.get("focusable")),
                        focused=bool(focus) and focus == inter, interaction=inter,
                        window=int(m.group(1)) if m else None, box=box, layer=layers[n["z"]], order=i,
                        visible=not covered))
    return out


def widget_position(w: dict, frame: ScreenFrame) -> list[float] | None:
    if w["box"] is None:
        return None
    x0, y0, x1, y1 = w["box"]
    x, y = frame.px_to_m((x0 + x1) / 2, (y0 + y1) / 2)
    return [x, y, 0.0 if frame.depth == "constant" else w["layer"] * frame.dz]


class SlotRegistry:
    """Stable slots per episode: a key keeps its slot; new keys are appended; a vanished key leaves a null slot."""

    def __init__(self, slots: Mapping[str, int] | None = None):
        self.slots: dict[str, int] = dict(slots or {})

    def assign(self, widgets: Sequence[dict]) -> list[dict | None]:
        for w in widgets:
            self.slots.setdefault(w["key"], len(self.slots))
        table: list[dict | None] = [None] * len(self.slots)
        for w in widgets:
            table[self.slots[w["key"]]] = w
        return table


def widget_attributes(w: dict) -> dict[str, str]:
    return {"role": w["role"], "label": w["label"], "value": "" if w["value"] is None else str(w["value"]),
            "disabled": str(w["disabled"]).lower(), "focusable": str(w["focusable"]).lower(),
            "focused": str(w["focused"]).lower(), "window": "" if w["window"] is None else str(w["window"]),
            "interaction": w["interaction"]}


def descriptors(table: Sequence[dict | None], frame: ScreenFrame, t: float,
                bindings: Mapping[str, str] | None = None) -> list[ObjectDescriptor]:
    """Slot table -> ObjectDescriptors; `bindings` maps a widget entity ("button:Submit") to a task entity id."""
    bindings = bindings or {}
    out = []
    for slot, w in enumerate(table):
        if w is None:
            out.append(ObjectDescriptor(slot=slot, descriptor="null", visible=False, timestamp=t))
            continue
        out.append(ObjectDescriptor(
            slot=slot, descriptor=w["role"], bbox_xyxy=[float(v) for v in w["box"]] if w["box"] else None,
            position_estimate=widget_position(w, frame), visible=w["visible"], timestamp=t,
            bound_entity=EntityRef(id=bindings[w["entity"]], version=0) if w["entity"] in bindings else None,
            attributes=widget_attributes(w)))
    return out


# ------------------------------------------------------------------------------------------------ StateView (D-144 R8)
def _widget_id(slot: int, w: dict) -> str:
    return f"slot{slot}:{w['key']}"


@dataclass(frozen=True)
class CWStateView:
    """Privileged StateView over a ComputerWorld scene (docs/relations.md 5.1, rrp.envs.base.StateView). Labels only;
    never imported by policies/. Built once per `state_view()` call from the same `scene_widgets` table `observe()`
    uses, so a slot's entity id matches its `descriptors()` slot exactly (`slot{slot}:{widget-key}`)."""
    caps: frozenset
    time: float
    gravity: "np.ndarray"
    _entities: tuple
    _ui_tree: tuple
    _token_map: Mapping[tuple, str]

    def entities(self) -> list[EntityState]:
        return list(self._entities)

    def contacts(self):
        raise CapabilityError("computerworld StateView has no contacts (a 2D UI world)")

    def joints(self):
        raise CapabilityError("computerworld StateView has no joints (a 2D UI world)")

    def camera(self, name: str):
        raise CapabilityError("computerworld StateView has no camera")

    def ui_tree(self) -> list[dict]:
        return [dict(n) for n in self._ui_tree]

    def token_entity(self, token_set: str, slot) -> str | None:
        return self._token_map.get((token_set, slot))


def cw_state_view(scene: dict, frame: ScreenFrame, slots: SlotRegistry, t: float) -> CWStateView:
    """Pure builder (testable on an inline scene, no `computerworld` wheel needed): `scene_widgets(scene)` assigned
    into `slots`' stable slot table -> one EntityState + one ui_tree node per occupied slot."""
    table = slots.assign(scene_widgets(scene))
    entities, ui_tree, token_map = [], [], {}
    for slot, w in enumerate(table):
        if w is None:
            continue
        wid = _widget_id(slot, w)
        parent = f"window:{w['window']}" if w["window"] is not None else None
        pos = widget_position(w, frame) or [0.0, 0.0, 0.0]
        extent = None
        if w["box"]:
            x0, y0, x1, y1 = w["box"]
            extent = np.array([(x1 - x0) * frame.m_per_px, (y1 - y0) * frame.m_per_px, 0.0])
        entities.append(EntityState(id=wid, kind="widget", name=w["label"] or w["role"], pos=np.array(pos, np.float64),
                                    quat=None, vel=None, extent=extent, mass=None, friction=None, material=None,
                                    parent=parent, assembly=None, body=None, visible=w["visible"],
                                    attrs=widget_attributes(w)))
        ui_tree.append(dict(id=wid, parent=parent, z=w["layer"], focus_rank=0 if w["focused"] else None,
                            role=w["role"], label=w["label"], bounds=w["box"], order=w["order"]))
        token_map[("widgets", slot)] = wid
    return CWStateView(caps=frozenset({"poses", "ui_tree"}), time=t, gravity=np.zeros(3), _entities=tuple(entities),
                       _ui_tree=tuple(ui_tree), _token_map=token_map)


# ------------------------------------------------------------------------------------------------ public UI fields / ui-rel-v1 edges (D-144 R20)
# The pointer / ComputerWorld family's own public fields (docs/relations.md section 2's family row: "+pos3d (screen
# mm + z-layer depth), +zlayer, +parent_id, +focus_rank, entity_id") and the "ui-rel-v1" edge vocabulary
# (`catalog.py` section 10 row R20's `ui.*`), built PURELY from the same `scene_widgets` slot table
# `cw_state_view` / `descriptors` already use -- so a slot's field / edge row lines up with its `ObjectDescriptor`
# slot and its `CWStateView` entity 1:1. Plain numpy only: `envs` sits below `policies` in the layer order
# (`tests/unit/test_layering.py`), so this file never imports `rrp.policies.relations` -- the actual `TokenSet` /
# `EdgeSet` wrapping of this data is a collate-path concern (a future unit's `nets/batch.py`-style file, matching
# how `ix.force_flow`'s "edges:support-v1" is built in `nets/batch.py` from `relgen.support.support_matrix`'s plain
# dict output, not in the env adapter -- rel-geo, D-144 addendum). `focus_rank` matches `cw_state_view`'s own,
# already-merged convention exactly (0 = the currently focused widget via `scene.focus.interaction`, -1 = every
# other widget): a per-widget RANK beyond that binary needs an explicit UI tab-index CW does not expose.
UI_REL_VOCAB = ("label_for", "contains", "focus_next", "above")


def _tab_order(table: Sequence[dict | None]) -> list[int]:
    """Slots of focusable, enabled, boxed widgets, in scene (document) order -- the deterministic public tab
    sequence `ui_edges`'s `focus_next` walks cyclically. CW exposes no explicit tab index; `order` (scene node
    index, already in every `scene_widgets` record) is the same document-order proxy `label_for` uses below."""
    return [s for s, w in enumerate(table) if w is not None and w["focusable"] and not w["disabled"] and w["box"]]


def ui_public_fields(table: Sequence[dict | None]) -> dict[str, np.ndarray]:
    """Per-slot `parent_id` (window index, -1 = desktop-level or a null slot), `zlayer` (dense z-layer rank, `float`)
    and `focus_rank` (0 / -1, `cw_state_view`'s own convention) over a `SlotRegistry.assign` table -- every input is
    already in `ObjectDescriptor.attributes` / `scene_widgets`' own record, nothing privileged. Pixel-frame
    independent (unlike `widget_position`): no `ScreenFrame` argument needed."""
    n = len(table)
    parent_id = np.full(n, -1, np.int64)
    zlayer = np.zeros(n, np.float32)
    focus_rank = np.full(n, -1, np.int64)
    for slot, w in enumerate(table):
        if w is None:
            continue
        parent_id[slot] = -1 if w["window"] is None else w["window"]
        zlayer[slot] = float(w["layer"])
        focus_rank[slot] = 0 if w["focused"] else -1
    return dict(parent_id=parent_id, zlayer=zlayer, focus_rank=focus_rank)


def ui_edges(table: Sequence[dict | None]) -> np.ndarray:
    """`[W, W, len(UI_REL_VOCAB)]` bool public graph over the same slot table (`W = len(table)`); a null slot's row
    / column stays all-False.
      contains[i, j]    : i, j are widgets of the SAME window (reflexive, symmetric); desktop-level widgets
                          (window is None) never match each other, so unrelated top-level icons are not bundled
                          into one "container" -- there is no window entity token to hang a real containment edge
                          off (CW's window frame itself carries no `semantic` node, `scene_widgets`), so this is
                          "co-contained in one window" rather than a widget literally containing another.
      label_for[i, j]   : i has role "label", j does not, both are in the same window (or both desktop-level), and
                          j is the NEXT widget after i in scene (document) order within that group -- the standard
                          "the label immediately precedes the control it describes" layout convention.
      focus_next[i, j]  : i, j are consecutive slots of the public tab order (`_tab_order`, cyclic: the last widget
                          wraps to the first).
      above[i, j]       : zlayer_j > zlayer_i (which of the pair renders on top / would occlude the other).
    """
    n = len(table)
    out = np.zeros((n, n, len(UI_REL_VOCAB)), bool)
    i_label, i_contains, i_focus, i_above = range(len(UI_REL_VOCAB))

    windows: dict[int, list[int]] = {}
    for slot, w in enumerate(table):
        if w is not None and w["window"] is not None:
            windows.setdefault(w["window"], []).append(slot)
    for members in windows.values():
        for i in members:
            out[i, members, i_contains] = True
        ordered = sorted(members, key=lambda s: table[s]["order"])
        for a, b in zip(ordered, ordered[1:]):
            if table[a]["role"] == "label" and table[b]["role"] != "label":
                out[a, b, i_label] = True

    order = _tab_order(table)
    for a, b in zip(order, order[1:] + order[:1]):
        out[a, b, i_focus] = True

    layer = np.array([w["layer"] if w is not None else -1 for w in table], dtype=np.int64)
    present = np.array([w is not None for w in table], dtype=bool)
    out[..., i_above] = (layer[None, :] > layer[:, None]) & present[:, None] & present[None, :]
    return out


# ------------------------------------------------------------------------------------------------ actions
KEY_NAMES = ["Enter", "Backspace", "Tab", "Escape", "Delete", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown",
             "Home", "End", "Ctrl+a", "Ctrl+s", "Ctrl+z"]
PRINTABLE = [c for c in string.printable if c not in "\t\n\r\x0b\x0c"]     # 95 symbols, space .. ~
KEY_VOCAB = KEY_NAMES + PRINTABLE
WHEEL_PX_PER_NOTCH = 120


def key_index(symbol: str) -> int:
    return KEY_VOCAB.index(symbol)


@dataclass
class PointerState:
    u: int
    v: int
    button: bool = False


def command_to_actions(groups: Mapping[str, Sequence[float]], state: PointerState, frame: ScreenFrame
                       ) -> tuple[list[tuple[str, str, dict]], dict[str, list[float]], str | None]:
    """One control tick -> ordered CW actions [(family, op, payload)], executed groups, rejection code.

    Order within a tick: pointer move, button edge, wheel, key. Missing groups hold (pointer stays, button keeps its
    state, no wheel, no key). `state` is updated in place."""
    unknown = sorted(set(groups) - {"pointer", "button", "wheel", "key"})
    if unknown:
        return [], {}, f"bad_group:{unknown[0]}"
    acts: list[tuple[str, str, dict]] = []
    view = {"width": frame.width, "height": frame.height}
    executed: dict[str, list[float]] = {}
    if "pointer" in groups:
        p = list(groups["pointer"])
        if len(p) != 2 or not all(np.isfinite(p)):
            return [], {}, "bad_width:pointer"
        u, v = frame.m_to_px(*p)
        if (u, v) != (state.u, state.v):
            state.u, state.v = u, v
            acts.append(("pointer.v1", "move", {"x": u, "y": v, **view}))
        executed["pointer"] = list(frame.px_to_m(state.u, state.v))
    if "button" in groups:
        down = float(groups["button"][0]) >= 0.5
        if down != state.button:
            state.button = down
            acts.append(("pointer.v1", "down" if down else "up", {"x": state.u, "y": state.v, "button": 0, **view}))
        executed["button"] = [1.0 if state.button else 0.0]
    if "wheel" in groups:
        n = int(round(float(groups["wheel"][0])))
        if n:
            acts.append(("pointer.v1", "wheel", {"x": state.u, "y": state.v, "delta_y": n * WHEEL_PX_PER_NOTCH, **view}))
        executed["wheel"] = [float(n)]
    if "key" in groups:
        k = int(round(float(groups["key"][0])))
        if k >= len(KEY_VOCAB) or k < -1:
            return [], {}, "key_out_of_vocab"
        if k >= 0:
            sym = KEY_VOCAB[k]
            acts.append(("keyboard.v1", "key", {"key": sym}) if k < len(KEY_NAMES)
                        else ("keyboard.v1", "type", {"text": sym}))
        executed["key"] = [float(k)]
    return acts, executed, None


# ------------------------------------------------------------------------------------------------ world
FORM_HTML = ("<!doctype html><html><head><title>Sign up</title></head><body><h1>Sign up</h1><form>"
             "<label for=name>Name</label> <input id=name name=name><br>"
             "<label for=email>Email</label> <input id=email name=email><br>"
             "<button type=submit id=submit>Submit</button></form></body></html>")


def world_definition(apps: Sequence[str] = ("terminal", "editor", "files", "desktop", "calculator", "browser"),
                     theme: str = "virtual-ubuntu-24", form: bool = False) -> dict:
    """A one-machine desktop world; `form` adds a simulated static site (form.internal) on a second node."""
    nodes = [{"id": MACHINE, "address": "10.0.0.2", "zone": "local"}]
    links, services = [], []
    if form:
        nodes.append({"id": "web", "address": "10.0.0.3", "zone": "local"})
        links.append({"from": MACHINE, "to": "web", "bidirectional": True, "latency_us": 0, "loss_per_million": 0})
        services.append({"id": "form", "kind": "static-site", "node": "web", "domains": ["form.internal"], "port": 80,
                         "tls": False, "initial_state": {"files": {"/index.html": FORM_HTML}}})
    return {"schema_version": 1, "id": "rrp-cw-desktop",
            "profiles": [{"id": "ubuntu", "name": "Ubuntu", "family": "linux", "home": "/home/{user}",
                          "case_sensitive": True, "shell": "posix"}],
            "computers": [{"id": MACHINE, "profile": "ubuntu", "address": "10.0.0.2", "user": ACTOR,
                           "installed_apps": list(apps)}],
            "network": {"nodes": nodes, "links": links}, "services": services,
            "metadata": {"desktop_themes": {MACHINE: theme}}}


# ------------------------------------------------------------------------------------------------ task setups + judges
@dataclass(frozen=True)
class CWTaskDef:
    """Env-side half of a cw/* task: owner-side setup (returns the goal), the instruction, the judge and the task's
    entity bindings. The TaskSpec (rrp.tasks.spec) carries budget, teacher key and envs."""
    setup: Callable[["ComputerWorldEnv", random.Random], dict]
    instruction: Callable[[dict], str]
    judge: Callable[["ComputerWorldEnv", float, float], Judgement]
    bindings: Callable[[dict], dict] = lambda goal: {}
    form: bool = False


def find_widget(env: "ComputerWorldEnv", pred: Callable[[dict], bool]) -> dict | None:
    """First visible widget matching pred (else the first occluded one, else None)."""
    ws = [w for w in scene_widgets(env.scene()) if pred(w)]
    vis = [w for w in ws if w["visible"]]
    return (vis or ws or [None])[0]


def _elements(env: "ComputerWorldEnv") -> list[dict]:
    """Flattened semantic.v1 elements of the actor's machine (the actor's own public channel)."""
    stack = list(env.cw_env.observe()["channels"].get("semantic.v1", {}).get(MACHINE, {}).get("elements", []))
    out = []
    while stack:
        e = stack.pop(0)
        out.append(e)
        stack[0:0] = e.get("children", [])
    return out


def _running(t: float, budget: float, reason: str) -> Judgement:
    return Judgement(True, "timeout", reason, False, False) if t >= budget else Judgement(False)


def _calc_setup(env, rng):
    env.owner_act("application.v1", "launch", {"kind": "calculator"})
    return {"a": rng.randint(1, 9), "b": rng.randint(1, 9)}


def _calc_judge(env, t, budget):
    g = env.goal
    done = [s for s in (n["primitive"].get("text") or "" for n in env.scene()["nodes"])
            if re.fullmatch(r"-?[\d.]+ [+−×÷-] -?[\d.]+ = -?[\d.e]+", s)]
    if f"{g['a']} + {g['b']} = {g['a'] + g['b']}" in done:
        return Judgement(True, "success", None, True, True)
    if done:
        return Judgement(True, "failure", "wrong_result", False, False)
    return _running(t, budget, "no_result")


WORDS = ["hello", "robot", "pointer", "world", "relational", "window", "button", "cursor", "screen", "planet", "garden",
         "silver", "rocket", "puzzle", "marble", "violet", "harbor", "lantern", "meadow", "copper", "falcon", "summit",
         "canvas", "orbit"]         # 24 (was 5 before D-142): typing must copy characters from the instruction


def _type_setup(env, rng):
    return {"text": rng.choice(WORDS)}


def _type_judge(env, t, budget):
    vals = [e.get("value") for e in _elements(env) if e.get("id") == "editor-text"]
    val = vals[0] if vals else None
    if val is not None and val == env.goal["text"]:
        return Judgement(True, "success", None, True, True)
    if val and not env.goal["text"].startswith(val):
        return Judgement(True, "failure", "wrong_text", False, False)
    return _running(t, budget, "app_not_open" if val is None else "incomplete_text")


TOP_BAR_PX, DOCK_PX = 32, 64          # Ubuntu theme: windows clamp below the top bar and right of the dock


def _drag_setup(env, rng):
    wid = env.owner_act("application.v1", "launch", {"kind": "calculator"})["window"]
    ws = [w for w in scene_widgets(env.scene()) if w["window"] == wid and w["box"]]
    bar = next(w for w in ws if w["interaction"] == f"window:{wid}:drag")
    x0, y0 = min(w["box"][0] for w in ws), min(w["box"][1] for w in ws)
    x1, y1 = max(w["box"][2] for w in ws), max(w["box"][3] for w in ws)
    # feasible offsets keep the whole window on screen, clear of the top bar and dock (CW clamps windows there)
    lo_x, hi_x, lo_y, hi_y = DOCK_PX - x0 + 4, env.frame.width - x1 - 4, TOP_BAR_PX - y0 + 4, env.frame.height - y1 - 4
    while True:
        dx, dy = rng.randint(min(lo_x, 0), max(hi_x, 0)), rng.randint(min(lo_y, 0), max(hi_y, 0))
        if max(abs(dx), abs(dy)) >= 30:
            return {"window": wid, "start": list(bar["box"][:2]), "dx": dx, "dy": dy}


def _drag_judge(env, t, budget, tol=6):
    g = env.goal
    bar = find_widget(env, lambda w: w["interaction"] == f"window:{g['window']}:drag")
    if bar is None:
        return Judgement(True, "failure", "window_closed", False, False)
    err = max(abs(bar["box"][0] - g["start"][0] - g["dx"]), abs(bar["box"][1] - g["start"][1] - g["dy"]))
    if err <= tol and not env.pointer.button:
        return Judgement(True, "success", None, True, True)
    return _running(t, budget, "off_target")


NAMES = ["Ada", "Grace", "Alan", "Edsger", "Barbara", "Donald", "Frances", "John", "Margaret", "Linus", "Radia", "Ken",
         "Dennis", "Hedy", "Claude", "Katherine", "Tim", "Sophie", "Niklaus", "Anita", "Guido", "Ivan", "Shafi", "Leslie"]


def _form_setup(env, rng):
    env.owner_act("application.v1", "launch", {"kind": "browser", "argument": "http://form.internal/"})
    name = rng.choice(NAMES)
    return {"name": name, "email": f"{name.lower()}@example.org"}


def _form_judge(env, t, budget):
    for e in _elements(env):
        url = (e.get("action") or {}).get("url", "")
        if e.get("kind") == "form" and "?" in url:
            q = urllib.parse.parse_qs(url.split("?", 1)[1], keep_blank_values=True)
            bad = [f for f in ("name", "email") if q.get(f, [""])[0] != env.goal[f]]
            if bad:
                return Judgement(True, "failure", f"wrong_value:{bad[0]}", False, False)
            return Judgement(True, "success", None, True, True)
    return _running(t, budget, "not_submitted")


CW_TASKS: dict[str, CWTaskDef] = {
    "cw/calc_sum": CWTaskDef(_calc_setup, lambda g: f"Use the calculator to compute {g['a']} + {g['b']}.", _calc_judge),
    "cw/open_type": CWTaskDef(_type_setup, lambda g: f"Open the Text Editor and type '{g['text']}'.", _type_judge,
                              bindings=lambda g: {"button:Text Editor": "app"}),
    "cw/drag_window": CWTaskDef(_drag_setup,
                                lambda g: f"Drag the Calculator window {g['dx']} mm to the right and {-g['dy']} mm up.",
                                _drag_judge, bindings=lambda g: {"button:Move Calculator": "window"}),
    "cw/fill_form": CWTaskDef(_form_setup,
                              lambda g: f"Fill in the sign-up form with name '{g['name']}' and email '{g['email']}', "
                                        "then submit it.",
                              _form_judge, form=True,
                              bindings=lambda g: {"textbox:Name": "field_name", "textbox:Email": "field_email",
                                                  "button:Submit": "submit"}),
}


# ------------------------------------------------------------------------------------------------ env
@dataclass
class Snapshot:
    world: Any
    pointer: PointerState
    slots: dict[str, int]
    steps: int
    goal: dict
    seed: int


class ComputerWorldEnv:
    """rrp Env over one ComputerWorld desktop (docs/architecture.md section 5)."""

    def __init__(self, task: str, *, seed: int = 0, width: int = 960, height: int = 640, m_per_px: float = 0.001,
                 depth: Literal["constant", "stack"] = "stack", dz: float = 0.002, images: bool = False,
                 control_hz: float = 10.0, theme: str = "virtual-ubuntu-24"):
        import computerworld as cw                                  # optional extra, lazy
        if task not in CW_TASKS:
            raise KeyError(f"computerworld has no task {task!r}; known: {sorted(CW_TASKS)}")
        if cw.engine_version != CW_VERSION:
            raise RuntimeError(f"computerworld engine {cw.engine_version}; rrp pins {CW_VERSION}")
        self.task_name, self.task = task, CW_TASKS[task]
        self.frame = ScreenFrame(width, height, m_per_px, depth, dz)
        self.images, self.control_hz = images, control_hz
        self.definition = world_definition(form=self.task.form, theme=theme)
        self.world = cw.World(self.definition, seed)
        grants = ["application.v1", "keyboard.v1", "pointer.v1"] + (["browser.v1"] if self.task.form else [])
        self.cw_env = self.world.environment({"actor": ACTOR, "machines": [MACHINE], "actions": grants,
                                              "observations": ["semantic.v1", "pixels.v1"]})
        lo, hi = self.frame.limits_m()
        self.body = cw_pointer_spec(lo, hi, self.frame.hover_z, len(KEY_VOCAB), control_hz)
        caps = ["privileged_truth", "snapshot", "render", "language", "object_descriptors", "proprio", "deterministic"]
        self._spec = EnvSpec(
            env_id="computerworld", backend="computerworld", task=task,
            bodies=[BodyInfo.from_spec(0, self.body, "cw_pointer")], control_hz=control_hz,
            action_spaces=[ActionSpace(group="pointer", kind="cartesian_position", width=2, rate_hz=control_hz, low=lo,
                                       high=hi, units="m"),
                           ActionSpace(group="button", kind="button", width=1, rate_hz=control_hz, low=[0.0], high=[1.0]),
                           ActionSpace(group="wheel", kind="discrete", width=1, rate_hz=control_hz, low=[-10.0],
                                       high=[10.0], units="notch"),
                           ActionSpace(group="key", kind="discrete", width=1, rate_hz=control_hz, units="index",
                                       vocab=list(KEY_VOCAB))],
            capabilities=caps + (["images"] if images else []), frame=self.frame.as_dict(),
            provenance={"engine": f"computerworld/{cw.engine_version}", "package": cw.__version__,
                        "adapter": ADAPTER_VERSION, "world_digest": content_hash(self.definition), "theme": theme,
                        "viewport": [width, height]})
        self._initial: dict[int, Snapshot] = {}
        self.seed = seed
        self.reset(seed)

    @property
    def spec(self) -> EnvSpec:
        return self._spec

    # -- owner side (task setup, judges, teachers)
    def owner_act(self, family: str, op: str, payload: dict) -> Any:
        out = self.cw_env.step([{"family": family, "op": op, "machine": MACHINE, "payload": payload}])["outcomes"][0]
        if not out["success"]:
            raise RuntimeError(f"{family}.{op} failed: {out.get('error')}")
        return out.get("value")

    def scene(self) -> dict:
        return self.cw_env.scene(self.frame.width, self.frame.height)

    def cw_judge(self, t: float, budget: float) -> Judgement:
        return self.task.judge(self, t, budget)

    def state_hash(self) -> str:
        return self.world.state_hash()

    # -- Env protocol
    def reset(self, seed: int | None = None) -> PolicyObservation:
        seed = self.seed if seed is None else seed
        if seed not in self._initial:
            self.world.reset(seed)
            self.pointer = PointerState(self.frame.width // 2, self.frame.height // 2)
            self.owner_act("pointer.v1", "move", {"x": self.pointer.u, "y": self.pointer.v,
                                                  "width": self.frame.width, "height": self.frame.height})
            goal = self.task.setup(self, random.Random(seed))
            self._initial[seed] = Snapshot(self.world.snapshot(), PointerState(self.pointer.u, self.pointer.v), {}, 0,
                                           goal, seed)
        return self.restore(self._initial[seed])

    def snapshot(self) -> Snapshot:
        return Snapshot(self.world.snapshot(), PointerState(**vars(self.pointer)), dict(self.slots.slots), self.steps,
                        dict(self.goal), self.seed)

    def restore(self, s: Snapshot) -> PolicyObservation:
        self.world.restore(s.world)
        self.pointer = PointerState(**vars(s.pointer))
        self.slots = SlotRegistry(s.slots)
        self.steps, self.goal, self.seed = s.steps, dict(s.goal), s.seed
        self.instruction = self.task.instruction(self.goal)
        self._prev_q = self._qpos()
        return self.observe()

    @property
    def time(self) -> float:
        return self.steps / self.control_hz

    def _qpos(self) -> np.ndarray:
        return np.array([*self.frame.px_to_m(self.pointer.u, self.pointer.v), self.frame.hover_z])

    def observe(self) -> PolicyObservation:
        t, q = self.time, self._qpos()
        table = self.slots.assign(scene_widgets(self.scene()))
        imgs = [ImageObs(camera="screen", height=self.frame.height, width=self.frame.width, encoding="rgba8",
                         pixels=self.render(), timestamp=t)] if self.images else []
        return PolicyObservation(
            observation_id=f"cw:{self.seed}:{self.steps}", sensor_time=t, robot_spec_hash=self.body.spec_hash,
            sensor_images=imgs,
            measured_node_state=NodeState(joint_addresses=["p/j0", "p/j1", "p/j2"], qpos=q,
                                          qvel=(q - self._prev_q) * self.control_hz, qpos_mask=np.ones(3, bool),
                                          timestamp=t, units={"slide": "m", "vel": "per_s"}),
            declared_sensor_channels=[SensorChannel(name="button", kind="binary",
                                                    values=np.array([float(self.pointer.button)]),
                                                    mask=np.ones(1, bool), timestamp=t)],
            object_descriptors=descriptors(table, self.frame, t, self.task.bindings(self.goal)),
            instruction=self.instruction)

    def step(self, command: NativeCommand | Mapping[int, NativeCommand] | None) -> StepResult:
        if isinstance(command, Mapping):
            command = command.get(0)
        self._prev_q = self._qpos()
        rejected, executed = None, None
        if command is not None:
            acts, executed, rejected = command_to_actions(command.groups, self.pointer, self.frame)
            if acts:
                res = self.cw_env.step([{"family": f, "op": o, "machine": MACHINE, "payload": p} for f, o, p in acts])
                bad = [o for o in res["outcomes"] if not o["success"]]
                if bad:
                    fam, op, _ = acts[bad[0]["index"]]
                    rejected = f"cw_refused:{fam}.{op}:{(bad[0].get('error') or {}).get('code', '?')}"
        self.steps += 1
        obs = self.observe()
        return StepResult(observation=obs, qpos=obs.measured_node_state.qpos, time=self.time, rejected=rejected,
                          source=None if command is None else command.source, command=executed or None)

    def truth(self) -> PrivilegedTruth:
        ws = scene_widgets(self.scene())
        poses = {w["key"]: (widget_position(w, self.frame) or [0.0, 0.0, 0.0]) + [1.0, 0.0, 0.0, 0.0] for w in ws}
        return PrivilegedTruth(observation_id=f"cw:{self.seed}:{self.steps}", sim_time=self.time, object_poses=poses,
                               object_entity_map={}, contacts=[], held_by={}, predicates={}, event_completion_truth={},
                               labels={"state_hash": self.world.state_hash(), "goal": self.goal, "widgets": ws})

    def state_view(self) -> CWStateView:
        """`rrp.envs.base.StateView` (capability "privileged_truth"; D-144 R8). Labels only, never a policy input."""
        return cw_state_view(self.scene(), self.frame, self.slots, self.time)

    def render(self, camera: str | None = None, *, width: int | None = None, height: int | None = None) -> np.ndarray:
        fr = self.cw_env.render(width or self.frame.width, height or self.frame.height)
        return np.frombuffer(fr["rgba"], np.uint8).reshape(fr["height"], fr["width"], 4)

    def submit_chunk(self, *a, **k):
        raise CapabilityError("computerworld has no chunk_executor")

    def close(self) -> None:
        self.world = self.cw_env = None


def make_env(*, task: str, body: str | list[str] = "cw_pointer", seed: int = 0, scene: dict | None = None, **kw
             ) -> ComputerWorldEnv:
    """Registry factory (rrp.envs.base.ENVS["computerworld"]); returns an env already reset to `seed`."""
    if body not in ("cw_pointer", ["cw_pointer"]):
        raise KeyError(f"computerworld has one body, cw_pointer; got {body!r}")
    return ComputerWorldEnv(task, seed=seed, **{**(scene or {}), **kw})
