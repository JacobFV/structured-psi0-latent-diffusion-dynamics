"""ComputerWorld as a 3D rrp environment (D-140, docs/architecture.md section 5). PROTOTYPE before S3.

Everything ComputerWorld-specific lives here: the screen <-> metre mapping, widget descriptors with depth and
occlusion, the `cw_pointer` body, the env adapter, a small `cw/*` task set with judges, and one scripted teacher per
task (source `scripted_teacher`, privileged scene access). The S3 interface types (EnvSpec, ActionSpace, StepResult,
Judgement, TaskSpec, CapabilityError) are local stand-ins that mirror the design doc; at S3 they are replaced by
`rrp.envs.base` / `rrp.tasks` / `rrp.policies.base` and this file splits into envs/computerworld.py,
tasks (cw/*), policies/teachers/computerworld.py and bodies (cw_pointer).

Backend: the `computerworld` PyO3 wheel (github.com/JacobFV/computerworld, MIT), imported lazily in `make_env`.
Pure functions (mapping, depth, occlusion, slots, command -> CW actions) never import it.

Deviations from the design doc, measured on computerworld 0.2.0:
- Scene node ids are NOT stable across scene revisions (the same launcher button gets a new id after a window
  opens). Slot identity is therefore keyed on (interaction, role, label, occurrence), which is stable while the
  widget exists; a key that disappears leaves a null slot.
- Depth "stack" ranks the distinct `z` layers of the scene (dense rank x dz), so depth stays bounded (a desktop has
  ~5-10 layers but ~100 nodes); within a layer, later insertion wins for occlusion, as in CW's own hit test.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import string
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Literal, Mapping, Sequence

import numpy as np

from rrp.core.action import NativeCommand
from rrp.core.base import content_hash
from rrp.core.observation import ImageObs, NodeState, ObjectDescriptor, PolicyObservation, PrivilegedTruth, SensorChannel
from rrp.core.refs import EntityRef
from rrp.core.robot import (ActuatorSpec, AssemblySpec, CommandGroup, ControllerContract, FrameDef, JointSpec,
                            LinkSpec, RobotSpec, TypedEdge)

ADAPTER_VERSION = "cw_env.v1"
CONTROLLER_VERSION = "cw_pointer.v1"
MACHINE = "pc"
ACTOR = "ada"

# ------------------------------------------------------------------------------------------ S3 stand-ins (replace)
ActionKind = Literal["joint_position", "joint_velocity", "joint_torque", "base_velocity",
                     "cartesian_position", "gripper", "button", "discrete"]


class CapabilityError(RuntimeError):
    pass


@dataclass(frozen=True)
class ActionSpace:
    group: str
    kind: ActionKind
    width: int
    rate_hz: float
    robot: int = 0
    low: list[float] | None = None
    high: list[float] | None = None
    units: str = ""
    vocab: list[str] | None = None


@dataclass(frozen=True)
class BodyInfo:
    robot: int
    family: str
    key: str
    robot_spec_hash: str


@dataclass(frozen=True)
class EnvSpec:
    env_id: str
    backend: str
    task: str
    bodies: list[BodyInfo]
    control_hz: float
    action_spaces: list[ActionSpace]
    capabilities: frozenset[str]
    frame: dict
    provenance: dict
    batch: int = 1


@dataclass
class StepResult:
    observation: PolicyObservation
    time: float
    rejected: str | None = None
    executed: dict[int, dict[str, list[float]]] = field(default_factory=dict)
    reward: float | None = None


@dataclass(frozen=True)
class Judgement:
    done: bool
    outcome: Literal["success", "failure", "timeout", "fell", "infeasible", "rejected", "crash"] | None = None
    failure_reason: str | None = None
    success_public: bool | None = None
    success_privileged: bool | None = None


# ------------------------------------------------------------------------------------------------ screen geometry
@dataclass(frozen=True)
class ScreenFrame:
    """World frame: x right, y up, z toward the viewer, origin at the screen centre, 1 px = m_per_px metres."""
    width: int = 960
    height: int = 640
    m_per_px: float = 0.001
    depth: Literal["constant", "stack"] = "stack"
    dz: float = 0.002
    hover_z: float = 0.05            # pointer's fixed slide-z (above every stacked layer the adapter produces)

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
    """Scene-pixel box (x0, y0, x1, y1) of a node: CW bounds are local, the transform maps them to the scene."""
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
    """Semantic nodes of a CW scene -> widget records (all of them, occluded ones included, visible flag set).

    Record: key (stable slot key), entity ("role:label"), role/label/value/disabled/focusable/interaction/window,
    box (clipped scene px or None), layer (dense rank of z), order (scene index), visible."""
    nodes = scene["nodes"]
    layers = {z: i for i, z in enumerate(sorted({n["z"] for n in nodes}))}
    occluders = []
    for i, n in enumerate(nodes):
        if n["primitive"].get("kind") in OCCLUDER_KINDS and n.get("opacity", 255) == 255:
            box = _clip(node_box(n), n.get("clip"))
            if box:
                occluders.append(((n["z"], i), box))
    focus = scene.get("focus") or {}
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
                        focused=bool(focus.get("interaction")) and focus.get("interaction") == inter,
                        interaction=inter, window=int(m.group(1)) if m else None, box=box,
                        layer=layers[n["z"]], order=i, visible=not covered))
    return out


def widget_position(w: dict, frame: ScreenFrame) -> list[float] | None:
    if w["box"] is None:
        return None
    x0, y0, x1, y1 = w["box"]
    x, y = frame.px_to_m((x0 + x1) / 2, (y0 + y1) / 2)
    z = 0.0 if frame.depth == "constant" else w["layer"] * frame.dz
    return [x, y, z]


class SlotRegistry:
    """Stable slots per episode: a key keeps its slot; new keys are appended; a vanished key leaves a null slot."""

    def __init__(self):
        self.slots: dict[str, int] = {}

    def assign(self, widgets: Sequence[dict]) -> list[dict | None]:
        for w in widgets:
            self.slots.setdefault(w["key"], len(self.slots))
        table: list[dict | None] = [None] * len(self.slots)
        for w in widgets:
            table[self.slots[w["key"]]] = w
        return table


def descriptors(table: Sequence[dict | None], frame: ScreenFrame, t: float,
                bindings: Mapping[str, str] | None = None) -> list[ObjectDescriptor]:
    """Slot table -> ObjectDescriptors. `bindings` maps a widget entity ("button:Submit") to a task entity id."""
    bindings = bindings or {}
    has_attrs = "attributes" in ObjectDescriptor.model_fields     # added at S3
    out = []
    for slot, w in enumerate(table):
        if w is None:
            kw = dict(slot=slot, descriptor="null", visible=False, timestamp=t)
            if has_attrs:
                kw["attributes"] = {}
            out.append(ObjectDescriptor(**kw))
            continue
        kw = dict(slot=slot, descriptor=w["role"], bbox_xyxy=[float(v) for v in w["box"]] if w["box"] else None,
                  position_estimate=widget_position(w, frame), visible=w["visible"], timestamp=t,
                  bound_entity=EntityRef(id=bindings[w["entity"]], version=0) if w["entity"] in bindings else None)
        if has_attrs:
            kw["attributes"] = widget_attributes(w)
        out.append(ObjectDescriptor(**kw))
    return out


def widget_attributes(w: dict) -> dict[str, str]:
    return {"label": w["label"], "value": "" if w["value"] is None else str(w["value"]),
            "state": "focused" if w["focused"] else "", "disabled": str(w["disabled"]).lower(),
            "window": "" if w["window"] is None else str(w["window"]), "interaction": w["interaction"]}


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
    acts: list[tuple[str, str, dict]] = []
    unknown = sorted(set(groups) - {"pointer", "button", "wheel", "key"})
    if unknown:
        return [], {}, f"bad_group:{unknown[0]}"
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


# ------------------------------------------------------------------------------------------------ body
def cw_pointer_spec(frame: ScreenFrame) -> RobotSpec:
    """`cw_pointer`: base link at the screen origin, slide joints x, y (range = viewport), slide z fixed at hover."""
    lo, hi = frame.limits_m()
    ident = [1.0, 0.0, 0.0, 0.0]

    def link(a, name, parent):
        return LinkSpec(address=a, name=name, parent_joint=parent, mass=0.01, inertia_diag=[1e-6] * 3, com=[0, 0, 0],
                        pos_in_parent=[0, 0, 0], quat_in_parent_wxyz=ident)
    links = [link("p", "screen", None), link("p/0", "carriage_x", "p/j0"), link("p/0/0", "carriage_y", "p/j1"),
             link("p/0/0/0", "tip", "p/j2")]
    joints = [JointSpec(address="p/j0", name="x", type="slide", parent_link="p", child_link="p/0", axis=[1, 0, 0],
                        range=[lo[0], hi[0]], qpos_width=1, qvel_width=1),
              JointSpec(address="p/j1", name="y", type="slide", parent_link="p/0", child_link="p/0/0", axis=[0, 1, 0],
                        range=[lo[1], hi[1]], qpos_width=1, qvel_width=1),
              JointSpec(address="p/j2", name="z", type="slide", parent_link="p/0/0", child_link="p/0/0/0",
                        axis=[0, 0, 1], range=[frame.hover_z, frame.hover_z], qpos_width=1, qvel_width=1)]
    acts = [ActuatorSpec(address="p/a0", name="x", kind="position", joint="p/j0", ctrl_range=[lo[0], hi[0]]),
            ActuatorSpec(address="p/a1", name="y", kind="position", joint="p/j1", ctrl_range=[lo[1], hi[1]]),
            ActuatorSpec(address="p/a2", name="button", kind="general", joint=None, ctrl_range=[0, 1]),
            ActuatorSpec(address="p/a3", name="wheel", kind="general", joint=None),
            ActuatorSpec(address="p/a4", name="key", kind="general", joint=None, ctrl_range=[-1, len(KEY_VOCAB) - 1])]
    contract = ControllerContract(
        id="cw_pointer", version=CONTROLLER_VERSION, kind="scripted", rate_hz=10.0, state_required=["qpos"],
        command_groups=[CommandGroup(name="pointer", width=2, units="m", semantic="ee_pose", actuators=["p/a0", "p/a1"],
                                     lower=lo, upper=hi, hold="hold_last"),
                        CommandGroup(name="button", width=1, units="normalized", semantic="gripper",
                                     actuators=["p/a2"], lower=[0], upper=[1], hold="hold_last"),
                        CommandGroup(name="wheel", width=1, units="normalized", semantic="wholebody_command",
                                     actuators=["p/a3"], lower=[-10], upper=[10], hold="zero_velocity"),
                        CommandGroup(name="key", width=1, units="normalized", semantic="wholebody_command",
                                     actuators=["p/a4"], lower=[-1], upper=[len(KEY_VOCAB) - 1], hold="zero_velocity")])
    tool = AssemblySpec(id="tool", kind="tool", members=["p/0/0/0", "p/j0", "p/j1", "p/j2"],
                        frame=FrameDef(link="p/0/0/0", pos=[0, 0, 0], quat_wxyz=ident), capabilities=["push"])
    edges = [TypedEdge(src=j.parent_link, dst=j.child_link, type="parent_of") for j in joints]
    # TODO(S3): family "pointer" once RobotSpec.family admits it; "end_effector" is the closest current literal.
    return RobotSpec(name="cw_pointer", family="end_effector", asset_source={"kind": "computerworld", "key": "cw_pointer"},
                     lineage=["cw_pointer"], floating_base=False, links=links, joints=joints, actuators=acts,
                     sensors=[], assemblies=[tool], attachment_ports=[], controller_contracts=[contract],
                     typed_edges=edges, capability_tags=["pointer"], synthetic=True).with_hash()


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


# ------------------------------------------------------------------------------------------------ tasks
@dataclass(frozen=True)
class CWTask:
    """Stand-in for rrp.tasks.TaskSpec (S3). setup runs owner-side actions (app launch is task setup, not a policy
    action) and returns the goal; judge reads the env (public scene or privileged) and returns a Judgement."""
    name: str
    max_seconds: float
    setup: Callable[["ComputerWorldEnv", random.Random], dict]
    instruction: Callable[[dict], str]
    judge: Callable[["ComputerWorldEnv", float], Judgement]
    teacher: Callable[["ComputerWorldEnv"], Iterator[dict]]
    form: bool = False
    bindings: Callable[[dict], dict] = lambda goal: {}
    envs: Mapping[str, dict] = field(default_factory=lambda: {"computerworld": {}})

    @property
    def teacher_key(self) -> str:
        return f"teacher:{self.name}"


def _running(t: float, max_seconds: float, reason: str) -> Judgement:
    if t >= max_seconds:
        return Judgement(True, "timeout", reason, False, False)
    return Judgement(False)


def _find(env: "ComputerWorldEnv", pred: Callable[[dict], bool], scene: dict | None = None) -> dict | None:
    ws = [w for w in scene_widgets(scene or env.scene()) if pred(w)]
    vis = [w for w in ws if w["visible"]]
    return (vis or ws or [None])[0]


def _elements(env: "ComputerWorldEnv") -> list[dict]:
    """Flattened semantic.v1 elements of the actor's machine (public actor channel)."""
    root = env.cw_env.observe()["channels"].get("semantic.v1", {}).get(MACHINE, {}).get("elements", [])
    out, stack = [], list(root)
    while stack:
        e = stack.pop(0)
        out.append(e)
        stack[0:0] = e.get("children", [])
    return out


# calc_sum: click the named calculator buttons a, +, b, =
def _calc_setup(env, rng):
    env.owner_act("application.v1", "launch", {"kind": "calculator"})
    return {"a": rng.randint(1, 9), "b": rng.randint(1, 9)}


def _calc_judge(env, t):
    g = env.goal
    texts = [n["primitive"].get("text") or "" for n in env.scene()["nodes"]]
    done = [s for s in texts if re.fullmatch(r"-?[\d.]+ [+−×÷-] -?[\d.]+ = -?[\d.e]+", s)]
    if f"{g['a']} + {g['b']} = {g['a'] + g['b']}" in done:
        return Judgement(True, "success", None, True, True)
    if done:
        return Judgement(True, "failure", "wrong_result", False, False)
    return _running(t, env.task.max_seconds, "no_result")


def _calc_teacher(env):
    g = env.goal
    for label in (str(g["a"]), "+", str(g["b"]), "="):
        yield from _click(env, lambda w, l=label: w["interaction"].endswith(f":calc:{l}"))


# open_type: open the Text Editor from the dock, then type a word
WORDS = ["hello", "robot", "pointer", "world", "relational"]


def _type_setup(env, rng):
    return {"text": rng.choice(WORDS)}


def _editor_value(env) -> str | None:
    vals = [e.get("value") for e in _elements(env) if e.get("id") == "editor-text"]
    return vals[0] if vals else None


def _type_judge(env, t):
    val = _editor_value(env)
    if val is not None and val == env.goal["text"]:
        return Judgement(True, "success", None, True, True)
    if val and not env.goal["text"].startswith(val):
        return Judgement(True, "failure", "wrong_text", False, False)
    return _running(t, env.task.max_seconds, "app_not_open" if val is None else "incomplete_text")


def _type_teacher(env):
    yield from _click(env, lambda w: w["interaction"] == "shell:launch:editor")
    yield from _hold(env, 1)
    for ch in env.goal["text"]:
        yield env.teacher_command(key=key_index(ch))


# drag_window: drag the Calculator window by a given offset
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


def _drag_judge(env, t, tol=6):
    g = env.goal
    bar = _find(env, lambda w: w["interaction"] == f"window:{g['window']}:drag")
    if bar is None:
        return Judgement(True, "failure", "window_closed", False, False)
    err = max(abs(bar["box"][0] - g["start"][0] - g["dx"]), abs(bar["box"][1] - g["start"][1] - g["dy"]))
    if err <= tol and not env.pointer.button:
        return Judgement(True, "success", None, True, True)
    return _running(t, env.task.max_seconds, "off_target")


def _drag_teacher(env):
    g = env.goal
    yield from _goto(env, lambda w: w["interaction"] == f"window:{g['window']}:drag")
    yield env.teacher_command(button=1)
    u, v = env.pointer.u + g["dx"], env.pointer.v + g["dy"]
    yield from _goto_px(env, u, v, button=1)
    yield env.teacher_command(button=0)


# fill_form: fill Name and Email on a (simulated) web form and submit
NAMES = ["Ada", "Grace", "Alan", "Edsger", "Barbara"]


def _form_setup(env, rng):
    env.owner_act("application.v1", "launch", {"kind": "browser", "argument": "http://form.internal/"})
    name = rng.choice(NAMES)
    return {"name": name, "email": f"{name.lower()}@example.org"}


def _form_judge(env, t):
    for e in _elements(env):
        url = (e.get("action") or {}).get("url", "")
        if e.get("kind") == "form" and "?" in url:
            q = urllib.parse.parse_qs(url.split("?", 1)[1], keep_blank_values=True)
            bad = [f for f in ("name", "email") if q.get(f, [""])[0] != env.goal[f]]
            if bad:
                return Judgement(True, "failure", f"wrong_value:{bad[0]}", False, False)
            return Judgement(True, "success", None, True, True)
    return _running(t, env.task.max_seconds, "not_submitted")


def _form_teacher(env):
    for fld in ("name", "email"):
        yield from _click(env, lambda w, f=fld: w["role"] == "textbox" and w["interaction"].endswith(f":content:{f}"))
        for ch in env.goal[fld]:
            yield env.teacher_command(key=key_index(ch))
    yield from _click(env, lambda w: w["role"] == "button" and w["label"] == "Submit")


TASKS: dict[str, CWTask] = {t.name: t for t in [
    CWTask("cw/calc_sum", 15.0, _calc_setup, lambda g: f"Use the calculator to compute {g['a']} + {g['b']}.",
           _calc_judge, _calc_teacher),
    CWTask("cw/open_type", 12.0, _type_setup, lambda g: f"Open the Text Editor and type '{g['text']}'.",
           _type_judge, _type_teacher, bindings=lambda g: {"button:Text Editor": "app"}),
    CWTask("cw/drag_window", 8.0, _drag_setup,
           lambda g: f"Drag the Calculator window {g['dx']} mm to the right and {-g['dy']} mm up.",
           _drag_judge, _drag_teacher, bindings=lambda g: {"button:Move Calculator": "window"}),
    CWTask("cw/fill_form", 20.0, _form_setup,
           lambda g: f"Fill in the sign-up form with name '{g['name']}' and email '{g['email']}', then submit it.",
           _form_judge, _form_teacher, form=True,
           bindings=lambda g: {"textbox:Name": "field_name", "textbox:Email": "field_email", "button:Submit": "submit"}),
]}


def get_task(name: str) -> CWTask:
    if name not in TASKS:
        raise KeyError(f"unknown ComputerWorld task {name!r}; known: {sorted(TASKS)}")
    return TASKS[name]


# ------------------------------------------------------------------------------------------------ teacher helpers
MAX_STEP_PX = 60          # teacher pointer speed: 60 px/tick = 0.6 m/s at 1 mm/px and 10 Hz


def _goto_px(env, u, v, button=None) -> Iterator[dict]:
    while (env.pointer.u, env.pointer.v) != (u, v):
        du, dv = u - env.pointer.u, v - env.pointer.v
        n = max(1.0, np.hypot(du, dv) / MAX_STEP_PX)
        nu, nv = env.pointer.u + du / n, env.pointer.v + dv / n
        yield env.teacher_command(pointer=env.frame.px_to_m(round(nu), round(nv)), button=button)


def _goto(env, pred) -> Iterator[dict]:
    w = _find(env, pred)
    if w is None or w["box"] is None:
        raise RuntimeError("teacher target not found")      # teacher bug or unexpected scene; surfaces as crash
    x0, y0, x1, y1 = w["box"]
    yield from _goto_px(env, (x0 + x1) // 2, (y0 + y1) // 2)


def _click(env, pred) -> Iterator[dict]:
    yield from _goto(env, pred)
    yield env.teacher_command(button=1)
    yield env.teacher_command(button=0)


def _hold(env, n) -> Iterator[dict]:
    for _ in range(n):
        yield env.teacher_command()


class ScriptedTeacher:
    """Scripted teacher for a cw/* task: source `scripted_teacher`, privileged (reads the full scene through the env).
    Stand-in for a policies.teachers Policy (S3/S4): reset(envs) then act() one command per tick."""
    source = "scripted_teacher"
    privileged = True

    def __init__(self, task: str):
        self.task = get_task(task)
        self.name = self.task.teacher_key
        self._it: Iterator[dict] | None = None

    def reset(self, env: "ComputerWorldEnv") -> None:
        self._env = env
        self._it = self.task.teacher(env)

    def act(self, obs: PolicyObservation) -> NativeCommand | None:
        try:
            groups = next(self._it)
        except StopIteration:
            return None                    # plan finished: hold
        return NativeCommand(controller_version=CONTROLLER_VERSION, groups=groups, source="scripted_teacher")


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
    """rrp Env over one ComputerWorld desktop (spec: docs/architecture.md section 5)."""

    def __init__(self, task: str, *, seed: int = 0, width: int = 960, height: int = 640, m_per_px: float = 0.001,
                 depth: Literal["constant", "stack"] = "stack", dz: float = 0.002, images: bool = False,
                 control_hz: float = 10.0, theme: str = "virtual-ubuntu-24"):
        import computerworld as cw                                  # optional extra, lazy
        self._cw = cw
        self.task = get_task(task)
        self.frame = ScreenFrame(width, height, m_per_px, depth, dz)
        self.images = images
        self.control_hz = control_hz
        self.definition = world_definition(form=self.task.form, theme=theme)
        self.world = cw.World(self.definition, seed)
        grants = ["application.v1", "keyboard.v1", "pointer.v1"] + (["browser.v1"] if self.task.form else [])
        self.cw_env = self.world.environment({"actor": ACTOR, "machines": [MACHINE], "actions": grants,
                                              "observations": ["semantic.v1", "pixels.v1"]})
        self.body = cw_pointer_spec(self.frame)
        lo, hi = self.frame.limits_m()
        caps = {"privileged_truth", "snapshot", "render", "language", "object_descriptors", "proprio", "deterministic"}
        if images:
            caps.add("images")
        self.spec = EnvSpec(
            env_id="computerworld", backend="computerworld", task=task,
            bodies=[BodyInfo(0, "pointer", "cw_pointer", self.body.spec_hash)], control_hz=control_hz,
            action_spaces=[ActionSpace("pointer", "cartesian_position", 2, control_hz, low=lo, high=hi, units="m"),
                           ActionSpace("button", "button", 1, control_hz, low=[0.0], high=[1.0]),
                           ActionSpace("wheel", "discrete", 1, control_hz, low=[-10.0], high=[10.0], units="notch"),
                           ActionSpace("key", "discrete", 1, control_hz, units="index", vocab=list(KEY_VOCAB))],
            capabilities=frozenset(caps), frame=self.frame.as_dict(),
            provenance={"engine": f"computerworld/{cw.engine_version}", "package": cw.__version__,
                        "adapter": ADAPTER_VERSION, "world_digest": content_hash(self.definition), "theme": theme,
                        "viewport": [width, height]})
        self._initial: dict[int, Any] = {}
        self.reset(seed)

    # -- owner side
    def owner_act(self, family: str, op: str, payload: dict) -> Any:
        out = self.cw_env.step([{"family": family, "op": op, "machine": MACHINE, "payload": payload}])["outcomes"][0]
        if not out["success"]:
            raise RuntimeError(f"{family}.{op} failed: {out.get('error')}")
        return out.get("value")

    def scene(self) -> dict:
        return self.cw_env.scene(self.frame.width, self.frame.height)

    def teacher_command(self, pointer=None, button=None, key=None) -> dict:
        g: dict[str, list[float]] = {"pointer": list(pointer) if pointer is not None
                                     else list(self.frame.px_to_m(self.pointer.u, self.pointer.v)),
                                     "button": [float(self.pointer.button if button is None else button)]}
        if key is not None:
            g["key"] = [float(key)]
        return g

    # -- Env protocol
    def reset(self, seed: int | None = None):
        seed = self.seed if seed is None else seed
        if seed not in self._initial:
            self.world.reset(seed)
            self.pointer = PointerState(self.frame.width // 2, self.frame.height // 2)
            self.owner_act("pointer.v1", "move", {"x": self.pointer.u, "y": self.pointer.v,
                                                  "width": self.frame.width, "height": self.frame.height})
            self.goal = self.task.setup(self, random.Random(seed))
            self._initial[seed] = Snapshot(self.world.snapshot(), PointerState(self.pointer.u, self.pointer.v),
                                           {}, 0, dict(self.goal), seed)
        return self.restore(self._initial[seed])

    def snapshot(self) -> Snapshot:
        return Snapshot(self.world.snapshot(), PointerState(**vars(self.pointer)), dict(self.slots.slots), self.steps,
                        dict(self.goal), self.seed)

    def restore(self, s: Snapshot) -> PolicyObservation:
        self.world.restore(s.world)
        self.pointer = PointerState(**vars(s.pointer))
        self.slots = SlotRegistry()
        self.slots.slots = dict(s.slots)
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
        t = self.time
        q = self._qpos()
        table = self.slots.assign(scene_widgets(self.scene()))
        imgs = []
        if self.images:
            fr = self.cw_env.render(self.frame.width, self.frame.height)
            px = np.frombuffer(fr["rgba"], np.uint8).reshape(fr["height"], fr["width"], 4)
            rgba = "rgba8" in ImageObs.model_fields["encoding"].annotation.__args__      # S3 adds rgba8
            imgs.append(ImageObs(camera="screen", height=fr["height"], width=fr["width"],
                                 encoding="rgba8" if rgba else "rgb8", pixels=px if rgba else px[..., :3].copy(),
                                 timestamp=t))
        kw = dict(observation_id=f"cw:{self.seed}:{self.steps}", sensor_time=t, robot_spec_hash=self.body.spec_hash,
                  sensor_images=imgs,
                  measured_node_state=NodeState(joint_addresses=["p/j0", "p/j1", "p/j2"], qpos=q,
                                                qvel=(q - self._prev_q) * self.control_hz, qpos_mask=np.ones(3, bool),
                                                timestamp=t, units={"slide": "m", "vel": "per_s"}),
                  declared_sensor_channels=[SensorChannel(name="button", kind="binary",
                                                          values=np.array([float(self.pointer.button)]),
                                                          mask=np.ones(1, bool), timestamp=t)],
                  object_descriptors=descriptors(table, self.frame, t, self.task.bindings(self.goal)))
        if "instruction" in PolicyObservation.model_fields:                  # S3 adds instruction
            kw["instruction"] = self.instruction
        return PolicyObservation(**kw)

    def step(self, command: NativeCommand | Mapping[int, NativeCommand] | None) -> StepResult:
        if isinstance(command, Mapping):
            command = command.get(0)
        self._prev_q = self._qpos()
        rejected, executed = None, {}
        if command is not None:
            acts, executed, rejected = command_to_actions(command.groups, self.pointer, self.frame)
            if acts:
                res = self.cw_env.step([{"family": f, "op": o, "machine": MACHINE, "payload": p} for f, o, p in acts])
                bad = [o for o in res["outcomes"] if not o["success"]]
                if bad:
                    err = bad[0].get("error") or {}
                    a = acts[bad[0]["index"]]
                    rejected = f"cw_refused:{a[0]}.{a[1]}:{err.get('code', '?')}"
        self.steps += 1
        return StepResult(observation=self.observe(), time=self.time, rejected=rejected,
                          executed={0: executed} if executed else {})

    def truth(self) -> PrivilegedTruth:
        ws = scene_widgets(self.scene())
        poses = {w["key"]: (widget_position(w, self.frame) or [0.0, 0.0, 0.0]) + [1.0, 0.0, 0.0, 0.0] for w in ws}
        return PrivilegedTruth(observation_id=f"cw:{self.seed}:{self.steps}", sim_time=self.time, object_poses=poses,
                               object_entity_map={}, contacts=[], held_by={}, predicates={},
                               event_completion_truth={},
                               labels={"state_hash": self.world.state_hash(), "goal": self.goal,
                                       "widgets": [{k: v for k, v in w.items()} for w in ws]})

    def render(self, camera: str | None = None, *, width: int | None = None, height: int | None = None) -> np.ndarray:
        fr = self.cw_env.render(width or self.frame.width, height or self.frame.height)
        return np.frombuffer(fr["rgba"], np.uint8).reshape(fr["height"], fr["width"], 4)

    def submit_chunk(self, *a, **k):
        raise CapabilityError("computerworld has no chunk_executor")

    def judge(self) -> Judgement:
        return self.task.judge(self, self.time)

    def state_hash(self) -> str:
        return self.world.state_hash()

    def close(self) -> None:
        self.world = self.cw_env = None


def make_env(env_id: str = "computerworld", *, task: str, body: str = "cw_pointer", seed: int = 0, **kw
             ) -> ComputerWorldEnv:
    if env_id != "computerworld" or body != "cw_pointer":
        raise KeyError(f"computerworld factory: env {env_id!r} body {body!r}")
    return ComputerWorldEnv(task, seed=seed, **kw)


def run_episode(env: ComputerWorldEnv, policy: Callable[[PolicyObservation], NativeCommand | None] | ScriptedTeacher,
                seed: int, *, frames: list | None = None) -> dict:
    """Minimal pre-S5 rollout (harness.rollout replaces it): returns outcome, failure reason, steps, provenance."""
    obs = env.reset(seed)
    if isinstance(policy, ScriptedTeacher):
        policy.reset(env)
        act, source, name = policy.act, policy.source, policy.name
    else:
        act, source, name = policy, getattr(policy, "source", "mock"), getattr(policy, "name", "callable")
    rejections = 0
    while True:
        j = env.judge()
        if j.done:
            break
        try:
            cmd = act(obs)
        except RuntimeError as e:
            j = Judgement(True, "crash", f"policy_error:{e}", False, False)
            break
        st = env.step(cmd)
        rejections += st.rejected is not None
        obs = st.observation
        if frames is not None:
            frames.append(env.render())
    return {"task": env.task.name, "seed": seed, "policy": name, "source": source, "outcome": j.outcome,
            "failure_reason": j.failure_reason, "success_privileged": j.success_privileged, "steps": env.steps,
            "time": env.time, "rejections": rejections, "instruction": env.instruction,
            "provenance": {**env.spec.provenance, "seed": seed, "depth": env.frame.depth,
                           "state_hash": env.state_hash()}}
