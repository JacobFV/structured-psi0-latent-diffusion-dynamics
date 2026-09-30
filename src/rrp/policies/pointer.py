"""Pointer-body (ComputerWorld `cw_pointer`) system 0 and packet sources (docs/architecture.md sections 3 and 5).

The pointer body has one assembly (`tool`, M = 1) and the latent contract's knots (0.1, 0.3, 0.5, 0.7 s after
valid_from; validity 0.8 s). At the pointer's 10 Hz control rate knot k covers the control ticks that END in
(t_{k-1}, t_k] (t_{-1} = 0): one tick for k = 0 and two ("early", "late") for k >= 1, so a packet spans 7 ticks.

ENGINEERED packet encoding `cw_pointer_eng.v1` (z[4, 1, 14]; SCRIPTED, not learned). Per knot, two tick slots
(early at offset 0, late at offset 7; knot 0 uses only the late slot), each
    [x, y, depth, button, key, wheel, flag]
    x, y    pointer target (m, screen frame of rrp.envs.computerworld.ScreenFrame) at the end of that tick
    depth   stack depth (m) of the topmost widget under the target (0 = none); informational (the pointer hovers)
    button  +1 = held down during that tick, -1 = up
    key     (KEY_VOCAB index + 1) / len(KEY_VOCAB) typed during that tick, 0 = none
    wheel   wheel notches during that tick
    flag    1 = the slot is planned, 0 = no plan (system 0 holds)
`EngineeredSystem0` (source label: scripted) realizes it every tick from the MEASURED pointer: it moves toward the
slot's target by at most MAX_STEP_PX per tick (the scripted teachers' speed), sets the button and emits the key and
wheel events; an unplanned slot, an expired packet or no packet holds (declared fallback).

`TeacherOracleSource` (ORACLE DIAGNOSTIC, privileged) encodes the scripted teacher's next 7 commands: it runs the
teacher on a shadow twin of each env (same task, seed and frame; CW is deterministic) ahead of the real episode and
counts shadow/real state-hash divergences at every packet. `make_pointer_oracle` = oracle packets -> engineered
system 0 behind the Policy interface (rrp.policies.latent.LatentStackPolicy).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from rrp.core.action import NativeCommand
from rrp.core.latent_action import AssemblyHandle, EntityHandle, LatentActionChunk
from rrp.core.system0 import System0Base

KNOT_TIMES = (0.1, 0.3, 0.5, 0.7)          # the latent contract's knots (rrp.policies.nets.semantic_latent)
VALIDITY_S = 0.8
ENG_VERSION = "cw_pointer_eng.v1"
SLOT_FIELDS = ("x", "y", "depth", "button", "key", "wheel", "flag")
SLOT_W = len(SLOT_FIELDS)
ENG_DIM = 2 * SLOT_W
MAX_STEP_PX = 60                           # = rrp.policies.teachers.computerworld.MAX_STEP_PX
POINTER_KINDS = frozenset({"cartesian_position", "button", "discrete"})
# D-144 R6: the pointer body's `RelBlock`s carry no relation factors by DEFAULT (single "tool" assembly, no
# cross-assembly routing to restrict). `_save` (harness/train/pointer) records its `compat_hash` into every
# checkpoint's `versions["factors"]` for provenance / future compatibility checks, same as every other family's
# factor hash. `PolicyConfig.factors=None` resolves to this empty preset, so every existing checkpoint's `UICtx`
# stays byte-identical (docs/relations.md 3.2 zero-bias equivalence: an empty-spec `FactorSite` adds no parameters
# and `.bias()` / `.augment()` return `None` / `(None, None)`).
POINTER_FACTORS_PRESET = "none"
# D-144 R20 follow-up (archived research/tracks/rel-r20.md "lead_questions"): what `UICtx`'s widget self-attention (`ctx>ctx`
# site) offers factors -- R20's own `ui-rel-v1` edge vocab, token hiddens (for `ui.drag_to`'s bilinear pair probe)
# and the screen-geometry fields `geo.*` (R13) reads, so `factors=["preset:ui"]` and/or `geo.pos3d` / `geo.depth3d`
# resolve here. Naming a field only ADDS what a factor is allowed to read at this site (`ops._applies`); it is a
# no-op for every config that does not ask for it, same as rel-geo's own `CTX_CARRIES` extension for the arm.
UI_CARRIES = ("edges:ui-rel-v1", "hidden", "pos3d", "cam_uvd", "zlayer", "parent_id")


@dataclass
class PolicyConfig:
    """Pointer-net relation factors (docs/relations.md; D-144 R20 follow-up). `factors=None` (the default) resolves
    to `POINTER_FACTORS_PRESET` ("none"): unchanged nets, byte-identical checkpoints. `factors=["preset:ui"]` turns
    on `ui.label_for` / `ui.same_window` / `ui.focus_next` / `ui.above` / `ui.drag_to` (`catalog.py` §ui) at `UICtx`'s
    widget self-attention; add `geo.*` names (or `"geo.*"` itself) to also enable the PaPE / bilinear geometry
    factors over the same tokens' `pos3d` / `cam_uvd` fields."""
    factors: list | None = None

    def specs(self):
        from rrp.policies.relations.base import resolve
        return resolve(self.factors, default=POINTER_FACTORS_PRESET)


def tick_slot(phase: float, dt: float, knot_times=KNOT_TIMES) -> tuple[int, int] | None:
    """(knot, slot) of the tick that starts at `phase` s after valid_from and ends at phase + dt; slot 1 = late (ends
    at the knot time), 0 = early. None past the last knot."""
    end = phase + dt
    for k, t in enumerate(knot_times):
        if end <= t + 1e-6:
            return k, 1 if abs(end - t) < 1e-6 else 0
    return None


def packet_ticks(dt: float, knot_times=KNOT_TIMES) -> list[tuple[int, int]]:
    """(knot, slot) of the ticks j = 0, 1, ... a packet covers (tick j starts at phase j * dt)."""
    out, j = [], 0
    while (ks := tick_slot(j * dt, dt, knot_times)) is not None:
        out.append(ks)
        j += 1
    return out


def encode_commands(cmds, dt: float, n_keys: int, depths=None, knot_times=KNOT_TIMES) -> np.ndarray:
    """Tick command groups (None = no plan) -> engineered z [K, 1, ENG_DIM]. cmds[j] is the command of tick j."""
    z = np.zeros((len(knot_times), 1, ENG_DIM), np.float32)
    for j, (k, s) in enumerate(packet_ticks(dt, knot_times)):
        if j >= len(cmds) or cmds[j] is None:
            continue
        g, o = cmds[j], s * SLOT_W
        x, y = g["pointer"]
        key = int(round(g.get("key", [-1])[0]))
        z[k, 0, o:o + SLOT_W] = [x, y, 0.0 if depths is None else depths[j], 1.0 if g["button"][0] >= 0.5 else -1.0,
                                 (key + 1) / n_keys, g.get("wheel", [0.0])[0], 1.0]
    return z


def decode_slot(z: np.ndarray, k: int, s: int, n_keys: int) -> dict | None:
    """One tick slot of an engineered z -> {x, y, depth, button, key, wheel} (None = unplanned)."""
    v = np.asarray(z)[k, 0, s * SLOT_W:(s + 1) * SLOT_W]
    if v[6] < 0.5:
        return None
    return dict(x=float(v[0]), y=float(v[1]), depth=float(v[2]), button=bool(v[3] > 0),
                key=int(round(float(v[4]) * n_keys)) - 1, wheel=int(round(float(v[5]))))


def pointer_packet(env, obs, z, *, lsv: str, rcv: str, source: str, name: str, sampling=None,
                   validity: float = VALIDITY_S, knot_times=KNOT_TIMES) -> LatentActionChunk:
    """Single-assembly packet for a ComputerWorld env at its current observation (graph/runtime version 0: cw/* tasks
    have no task graph)."""
    now = float(env.time)
    return LatentActionChunk(
        latent_space_version=lsv, realizer_compat_version=rcv, z=np.ascontiguousarray(z, np.float32),
        knot_times=list(knot_times),
        assemblies=[AssemblyHandle(handle=f"asm:{env.body.spec_hash}:{env.body.assemblies[0].frame.link}",
                                   robot_index=0)],
        assembly_mask=[True], entity_registry=[EntityHandle(handle=f"ent:{d.slot}") for d in obs.object_descriptors],
        observation_id=obs.observation_id, graph_version=0, runtime_version=0, robot_spec_hash=env.body.spec_hash,
        generated_at=time.time(), valid_from=now, valid_until=now + validity, source=source, policy_version=name,
        sampling=dict(sampling or {}))


def _cmd_source(packet) -> str:
    return {"target_encoder_oracle": "oracle"}.get(packet.source, packet.source)


class EngineeredSystem0(System0Base):
    """SCRIPTED system 0 for the pointer: realizes a `cw_pointer_eng.v1` packet (module docstring) every tick from the
    measured pointer (qpos) and button sensor. Inputs: packet z, phase since valid_from, pointer x/y, button."""
    label = f"scripted:{ENG_VERSION}"

    def __init__(self, env, *, max_step_px: int = MAX_STEP_PX):
        super().__init__(latent_space_version=ENG_VERSION, realizer_compat_version=ENG_VERSION)
        self.spec_hash, self.frame = env.body.spec_hash, env.frame
        self.n_keys = len(env.spec.space("key").vocab)
        self.dt = 1.0 / env.spec.control_hz
        self.max_step_m = max_step_px * env.frame.m_per_px

    @property
    def robot_spec_hash(self) -> str:
        return self.spec_hash

    def tick(self, env, controller_version=None) -> NativeCommand | None:
        now = float(env.time)
        p = self.packet
        if p is None or now > p.valid_until - 1e-9:
            if p is not None:
                self.invalidate("expired", now)
            self.stats.fallback_holds += 1
            return None
        ks = tick_slot(now - p.valid_from, self.dt, p.knot_times)
        sl = None if ks is None else decode_slot(p.z, *ks, self.n_keys)
        if sl is None:
            self.stats.fallback_holds += 1
            return None
        obs = env.observe()
        q = np.asarray(obs.measured_node_state.qpos[:2], float)
        d = np.array([sl["x"], sl["y"]]) - q
        n = float(np.linalg.norm(d))
        tgt = q + d * min(1.0, self.max_step_m / n) if n > 1e-12 else q
        g = {"pointer": [float(tgt[0]), float(tgt[1])], "button": [1.0 if sl["button"] else 0.0]}
        if 0 <= sl["key"] < self.n_keys:
            g["key"] = [float(sl["key"])]
        if sl["wheel"]:
            g["wheel"] = [float(sl["wheel"])]
        self.stats.ticks += 1
        return NativeCommand(controller_version=controller_version or "cw_pointer.v1", groups=g, source=_cmd_source(p))


def _twin(env):
    from rrp.envs.computerworld import ComputerWorldEnv
    f = env.frame
    return ComputerWorldEnv(env.task_name, seed=env.seed, width=f.width, height=f.height, m_per_px=f.m_per_px,
                            depth=f.depth, dz=f.dz, control_hz=env.control_hz, theme=env.spec.provenance["theme"])


def target_depth(env, x: float, y: float) -> float:
    """Stack depth (m) of the topmost visible widget whose box contains (x, y); 0 if none."""
    from rrp.envs.computerworld import scene_widgets, widget_position
    u, v = env.frame.m_to_px(x, y)
    hits = [w for w in scene_widgets(env.scene()) if w["visible"] and w["box"]
            and w["box"][0] <= u < w["box"][2] and w["box"][1] <= v < w["box"][3]]
    if not hits:
        return 0.0
    return float(widget_position(max(hits, key=lambda w: (w["layer"], w["order"])), env.frame)[2])


def env_widget_table(env) -> list[dict | None]:
    """The env's current slot-indexed `scene_widgets` table (`SlotRegistry.assign`, `env.slots`): 1:1 with
    `obs.object_descriptors` when called right after `env.observe()` with no `env.step()` in between, matching
    `widget_features`' `table` argument (D-144 R20 follow-up). Used only at LIVE rollout call sites
    (`TeacherOracleSource._encode`, `PointerSystemI.packets`, `PointerBCPolicy.act`) so `preset:ui` factors see real
    edges/fields at inference when a factor set enables them; the training data pipeline
    (`harness.train.pointer.collect_episode`) stays on `table=None`, unchanged."""
    from rrp.envs.computerworld import scene_widgets
    return env.slots.assign(scene_widgets(env.scene()))


class TeacherOracleSource:
    """ORACLE DIAGNOSTIC packet source (privileged): the scripted teacher's next ticks, run on a shadow twin env, encoded
    with the engineered encoding, or (`encoder` = a frozen PointerEncoder with its versions) as z = E(public features
    at t, teacher chunk) for the LEARNED system 0 (the arm's target-encoder oracle). Metric: shadow/real divergences
    (state hash at the packet tick)."""
    name = "cw_teacher_oracle"

    def __init__(self, validity: float = VALIDITY_S, encoder=None, versions: dict | None = None, device="cpu"):
        self.validity, self.E, self.device = validity, encoder, device
        self.lsv = versions["latent_space_version"] if versions else ENG_VERSION
        self.rcv = versions["realizer_compat_version"] if versions else ENG_VERSION
        self.calls = 0

    def reset(self, envs):
        from rrp.policies.teachers.computerworld import CWTeacher
        self.st, self.hist = {}, {}
        for e in envs:
            sh = _twin(e)
            self.st[id(e)] = dict(shadow=sh, teacher=CWTeacher(sh, e.task_name), cmds=[], depths=[], ptrs=[],
                                  btns=[], hashes=[sh.state_hash()], done=False, divergences=0)
            self.hist[id(e)] = EventHistory()

    def featurizer(self, env):
        return None

    def executed(self, env, command) -> None:
        self.hist[id(env)].push(env.steps, None if command is None else command.groups, screen_half(env.spec))

    def _advance(self, st, upto: int):
        sh = st["shadow"]
        while len(st["cmds"]) < upto:
            c = None if st["done"] else st["teacher"].act()
            st["ptrs"].append(sh._qpos()[:2].copy())
            st["btns"].append(float(sh.pointer.button))
            if c is None:
                st["done"] = True
                st["cmds"].append(None)
                st["depths"].append(0.0)
                continue
            g = c.groups
            st["depths"].append(target_depth(sh, *g["pointer"]))
            sh.step(c)
            st["cmds"].append(g)
            st["hashes"].append(sh.state_hash())

    def _encode(self, e, obs, st, n, H):
        import torch
        half = screen_half(e.spec)
        f = public_features(obs, half, self.hist[id(e)], n, table=env_widget_table(e))
        cmds, ptrs, btns = st["cmds"][n:n + H], st["ptrs"][n:n + H], st["btns"][n:n + H]
        a = dict(dxy=np.zeros((H, 2), np.float32), xy=np.zeros((H, 2), np.float32), btn=np.zeros(H, np.float32),
                 key=np.zeros(H, np.int64), valid=np.zeros(H, bool))
        for j, g in enumerate(cmds):
            if g is None:
                continue
            xy = np.asarray(g["pointer"], np.float32)
            a["dxy"][j] = (xy - ptrs[j]) / STEP_M
            a["xy"][j] = xy / half
            a["btn"][j] = float(g["button"][0] >= 0.5)
            a["key"][j] = int(round(g.get("key", [-1])[0])) + 1
            a["valid"][j] = True
        b = collate_public([f], self.device)
        at = {k: torch.from_numpy(v)[None].to(self.device) for k, v in a.items()}
        with torch.no_grad():
            return self.E(b, at)[0][0].float().cpu().numpy()

    def packets(self, envs) -> list[LatentActionChunk]:
        out = []
        for e in envs:
            st, n = self.st[id(e)], e.steps
            dt = 1.0 / e.spec.control_hz
            H = len(packet_ticks(dt))
            self._advance(st, n + H)
            if n < len(st["hashes"]) and st["hashes"][n] != e.state_hash():
                st["divergences"] += 1
            obs = e.observe()
            if self.E is None:
                z = encode_commands(st["cmds"][n:n + H], dt, len(e.spec.space("key").vocab), st["depths"][n:n + H])
            else:
                z = self._encode(e, obs, st, n, H)
            out.append(pointer_packet(e, obs, z, lsv=self.lsv, rcv=self.rcv,
                                      source="oracle" if self.E is None else "target_encoder_oracle", name=self.name,
                                      validity=self.validity))
        self.calls += len(envs)
        return out


def pointer_requirements(*, privileged: bool = False, tasks=None):
    from rrp.policies.base import Requirements
    return Requirements(POINTER_KINDS, groups=frozenset({"pointer", "button", "key"}),
                        observations=frozenset({"proprio", "object_descriptors", "language"}),
                        body_families=frozenset({"pointer"}), tasks=frozenset(tasks) if tasks else None,
                        privileged=privileged)


def make_pointer_oracle(*, replan_ticks: int = 4, name: str = "pointer_oracle", representation: str | None = None,
                        device: str = "cpu"):
    """ORACLE DIAGNOSTIC: scripted-teacher look-ahead -> engineered packet -> SCRIPTED engineered system 0; with
    `representation` (a pointer rep checkpoint): -> z = E(public features, teacher chunk) -> its LEARNED system 0."""
    from rrp.policies.latent import LatentStackPolicy
    if representation is None:
        return LatentStackPolicy(TeacherOracleSource(), None, replan_ticks=replan_ticks, name=name, source="oracle",
                                 version=f"oracle:teacher->{ENG_VERSION}|system0={EngineeredSystem0.label}",
                                 make_s0=EngineeredSystem0, requires=pointer_requirements(privileged=True))
    rb = load_pointer_bundle(representation, device)
    v = rb["versions"]
    R = rb["modules"]["R"]
    return LatentStackPolicy(TeacherOracleSource(encoder=rb["modules"]["E"], versions=v, device=device), R,
                             replan_ticks=replan_ticks, name=name, source="oracle", variant=rb["config"]["variant"],
                             version=f"oracle:E({representation})|system0=learned:{v['realizer_compat_version']}",
                             make_s0=lambda e: LearnedSystem0(R, e, latent_space_version=v["latent_space_version"],
                                                              realizer_compat_version=v["realizer_compat_version"],
                                                              device=device),
                             requires=pointer_requirements(privileged=True))


# =============================================================================================== learned route (step 2)
# Public featurization: the ONLY definition of what the learned pointer policies (system i, BC) may see. Widget
# descriptors (slot-indexed; label/role/box/depth/visible/focused/disabled/focusable/bound task entity), the instruction
# text, the pointer (qpos x, y) and button sensor, the clock, and an efference copy of the policy's OWN executed
# events (button edges with the pointer position, typed keys) - CW does not show typed text in the scene, so without it
# typing progress is unobservable. No env.truth(), no teacher state, no CW scene internals.
NW, LC, LI, NH, WF = 80, 20, 112, 24, 11
PHASES = ("idle", "move", "press", "release", "drag", "type")
ROLE_IDS = {"null": 0, "button": 1, "textbox": 2, "label": 3, "form": 4, "heading": 5, "text": 6, "link": 7,
            "checkbox": 8}
N_ROLE, N_BOUND = 10, 16


def _vocab():
    from rrp.envs.computerworld import KEY_NAMES, KEY_VOCAB, PRINTABLE
    return KEY_NAMES, KEY_VOCAB, PRINTABLE


def sym_of_char(c: str) -> int:
    """Shared symbol codes: 0 pad, 1..95 printable ASCII, 96 other character, 97.. named keys."""
    _, _, pr = _vocab()
    return pr.index(c) + 1 if c in pr else 96


def sym_of_key(k: int) -> int:
    names, vocab, _ = _vocab()
    return 97 + k if k < len(names) else sym_of_char(vocab[k])


N_SYM = 97 + 14
N_KEYCLS = 110                 # 0 = no key, 1 + KEY_VOCAB index


def codes(text: str, n: int) -> np.ndarray:
    out = np.zeros(n, np.int16)
    for i, c in enumerate((text or "")[:n]):
        out[i] = sym_of_char(c)
    return out


def _bound_id(e) -> int:
    import zlib
    return 0 if e is None else 1 + zlib.crc32(e.id.encode()) % (N_BOUND - 1)


def screen_half(spec) -> np.ndarray:
    """Half extents (m) of the screen: normalization of every position feature to [-1, 1]."""
    w, h = spec.frame["screen_px"]
    return np.array([w, h], np.float32) * spec.frame["m_per_px"] / 2


def ui_widget_fields(table) -> dict:
    """R20's PUBLIC `ui_public_fields` / `ui_edges` (`rrp.envs.computerworld`, vocab `ui-rel-v1`), sliced/padded to
    NW slots -- `table` is the env's slot-indexed `scene_widgets` record list (`SlotRegistry.assign`, the exact same
    table `ComputerWorldEnv.observe()` passes to `descriptors()`), so slot `s` here is slot `s` of
    `obs.object_descriptors` / `widget_features` 1:1. `wzlayer` / `wparent` / `wfocusrank` and the `[NW, NW, 4]`
    `wuiedges` graph are what `UICtx`'s `preset:ui` factors read at the widget self-attention (`_relctx`)."""
    from rrp.envs.computerworld import UI_REL_VOCAB, ui_edges, ui_public_fields
    n = min(len(table), NW)
    t = list(table[:n])
    f, e = ui_public_fields(t), ui_edges(t)
    zlayer, parent, focusr = np.zeros(NW, np.float32), np.full(NW, -1, np.int64), np.full(NW, -1, np.int64)
    zlayer[:n], parent[:n], focusr[:n] = f["zlayer"], f["parent_id"], f["focus_rank"]
    edges = np.zeros((NW, NW, len(UI_REL_VOCAB)), bool)
    edges[:n, :n] = e
    return dict(wzlayer=zlayer, wparent=parent, wfocusrank=focusr, wuiedges=edges)


def widget_features(obs, half, table=None) -> dict:
    """Descriptor slots -> fixed arrays (slot index = row; slots >= NW are dropped). `wpos3d` / `wcamuvd` (D-144
    R20 follow-up): the same screen-geometry `widget_position` already puts in `d.position_estimate` (the 1 mm/px
    `ScreenFrame` mapping, `+` z-layer depth when `depth="stack"`) -- `pos3d` world-frame metres, `cam_uvd` the
    already-computed normalized screen (u, v) `+` the same depth, matching `geo.*`'s field kinds (`catalog.py`).
    `table` (optional; the env's raw `scene_widgets` slot table, `env_widget_table(env)`): when given, also adds
    R20's public UI fields / `ui-rel-v1` edges (`ui_widget_fields`) that `preset:ui` factors read; omitted (as at
    the training data pipeline, `harness.train.pointer.collect_episode`), `UICtx` runs those factors as a harmless
    no-op (zero edges) -- never a silent fabrication, since `factors` must be explicitly turned on for them to be
    read at all (`PolicyConfig`, `POINTER_FACTORS_PRESET` stays the default)."""
    ch = np.zeros((NW, LC), np.int16)
    role = np.zeros(NW, np.int8)
    bound = np.zeros(NW, np.int8)
    wf = np.zeros((NW, WF), np.float32)
    m = np.zeros(NW, bool)
    pos3d = np.zeros((NW, 3), np.float32)
    camuvd = np.zeros((NW, 3), np.float32)
    for d in obs.object_descriptors[:NW]:
        if d.descriptor == "null" or d.bbox_xyxy is None:
            continue
        a = d.attributes or {}
        s = d.slot
        ch[s] = codes(a.get("label", ""), LC)
        role[s] = ROLE_IDS.get(d.descriptor, 9)
        bound[s] = _bound_id(d.bound_entity)
        x, y, z = d.position_estimate
        x0, y0, x1, y1 = d.bbox_xyxy
        wf[s] = [x / half[0], y / half[1], (x1 - x0) * 1e-3 / half[0], (y1 - y0) * 1e-3 / half[1], z / 0.01,
                 float(d.visible), a.get("focused") == "true", a.get("disabled") == "true", a.get("focusable") == "true",
                 float(d.bound_entity is not None), 0.0]
        m[s] = True
        pos3d[s] = [x, y, z]
        camuvd[s] = [x / half[0], y / half[1], z]
    out = dict(wch=ch, wrole=role, wbound=bound, wf=wf, wmask=m, wpos3d=pos3d, wcamuvd=camuvd)
    if table is not None:
        out.update(ui_widget_fields(table))
    return out


class EventHistory:
    """Efference copy of the policy's own executed events: (kind 1 down / 2 up / 3 key, symbol, x, y, tick)."""

    def __init__(self):
        self.ev: list[tuple[int, int, float, float, int]] = []
        self.button = False

    def push(self, tick: int, groups: dict | None, half) -> None:
        if not groups:
            return
        x, y = groups.get("pointer", [np.nan, np.nan])
        if "button" in groups:
            b = groups["button"][0] >= 0.5
            if b != self.button:
                self.ev.append((1 if b else 2, 0, x / half[0], y / half[1], tick))
                self.button = b
        k = int(round(groups.get("key", [-1])[0]))
        if k >= 0:
            self.ev.append((3, sym_of_key(k), x / half[0], y / half[1], tick))

    def array(self, tick: int) -> np.ndarray:
        """[NH, 5] (kind, symbol, x, y, age in ticks) of the last NH events, oldest first; kind 0 = padding."""
        out = np.zeros((NH, 5), np.float32)
        ev = self.ev[-NH:]
        for i, (k, s, x, y, t) in enumerate(ev):
            out[i] = [k, s, np.nan_to_num(x), np.nan_to_num(y), tick - t]
        return out


def public_features(obs, half, hist: EventHistory, tick: int, table=None) -> dict:
    """One tick of public input (numpy, unbatched). `table`: see `widget_features`."""
    f = widget_features(obs, half, table)
    q = obs.measured_node_state.qpos
    btn = float(obs.declared_sensor_channels[0].values[0]) if obs.declared_sensor_channels else 0.0
    f.update(instr=codes(obs.instruction or "", LI), ptr=np.array([q[0] / half[0], q[1] / half[1]], np.float32),
             btn=np.float32(btn), tick=np.float32(tick), hist=hist.array(tick))
    return f


def collate_public(fs: list[dict], device) -> dict:
    import torch
    out = {}
    for k in fs[0]:
        a = np.stack([np.asarray(f[k]) for f in fs])
        t = torch.from_numpy(a)
        out[k] = (t.long() if a.dtype.kind in "iu" else t if a.dtype == bool else t.float()).to(device)
    return out


# ---------------------------------------------------------------------------------------------------------- nets
def _nets():
    """torch modules (built lazily: `import rrp.policies.pointer` stays torch-free for the scripted route)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from rrp.envs.computerworld import UI_REL_VOCAB
    from rrp.policies.nets.attention import MHA, RelBlock
    from rrp.policies.nets.flow import MLP, sinusoidal
    from rrp.policies.relations.base import EdgeSet, RelCtx, TokenSet
    from rrp.policies.relations.ops import FactorSite

    def _zero_mha_out(m: MHA) -> MHA:
        """Zero-init an MHA's output projection so it contributes exactly 0 to a residual sum: the same
        zero-bias-equivalence idiom docs/relations.md 3.2 uses for factor gates/aug, reused here (D-144 R6) for an
        entire unused `RelBlock` attention stage."""
        nn.init.zeros_(m.o.weight)
        nn.init.zeros_(m.o.bias)
        return m

    def cross_relblock(D, heads) -> RelBlock:
        """`RelBlock` standing in for the former pointer `Block(cross=True)` (cross-attention + MLP only, no
        self-attention among the query tokens): the unused self stage (`.s`) is zero-init-output, an exact no-op
        both fresh (bit-identical to the old 2-stage forward) and from an old checkpoint (nothing is loaded into
        `.s`, so it stays zero-init -- `_remap_block_state` below)."""
        b = RelBlock(D, heads)
        _zero_mha_out(b.s)
        return b

    def self_relblock(D, heads) -> RelBlock:
        """`RelBlock` standing in for the former pointer `Block(cross=False)` (self-attention + MLP only): the
        unused cross stage (`.x`) is zero-init-output, same reasoning as `cross_relblock`."""
        b = RelBlock(D, heads)
        _zero_mha_out(b.x)
        return b

    class LabelBag(nn.Module):
        """Widget label -> one vector: mean of char and (char, position) embeddings (embedding_bag: the [B, NW, LC, D]
        tensor is never materialized)."""

        def __init__(self, D):
            super().__init__()
            self.c = nn.Embedding(N_SYM, D, padding_idx=0)
            self.cp = nn.Embedding(N_SYM * LC + 1, D, padding_idx=0)
            self.register_buffer("pos", torch.arange(LC))

        def forward(self, ch):
            B, W, L = ch.shape
            c = ch.reshape(B * W, L)
            cp = torch.where(c > 0, c * LC + self.pos + 1, torch.zeros_like(c))
            e = F.embedding_bag(c, self.c.weight, mode="mean", padding_idx=0) + \
                F.embedding_bag(cp, self.cp.weight, mode="mean", padding_idx=0)
            return e.reshape(B, W, -1)

    class UICtx(nn.Module):
        """Public context tokens: NW widget tokens (label chars + role + bound entity + geometry, pointer-relative
        centre), LI instruction characters, NH own-event tokens and one proprio token; `layers` self-attention
        blocks. `specs` (D-144 R20 follow-up, docs/relations.md 10 row R6's own lead_question): resolved
        `FactorSpec`s (`PolicyConfig(factors=...).specs()`) applied at the widget self-attention (`ctx>ctx`) via a
        `TokenSet`/`RelCtx` built from R20's public UI fields/edges + the screen-geometry fields (`_relctx`); the
        default `()` (== `resolve(None, default=POINTER_FACTORS_PRESET)`, the empty preset) makes every per-layer
        `FactorSite` parameter-free, so `.bias()` / `.augment()` return `None` / `(None, None)` and an existing
        checkpoint loads and runs byte-identically."""

        def __init__(self, D=128, heads=4, layers=3, specs=()):
            super().__init__()
            self.sym = nn.Embedding(N_SYM, D)
            self.cpos = nn.Embedding(LI, D)
            self.bag = LabelBag(D)
            self.role, self.bound = nn.Embedding(N_ROLE, D), nn.Embedding(N_BOUND, D)
            self.lab, self.wf = MLP(D, D), MLP(WF + 2, D)
            self.hk, self.hf = nn.Embedding(4, D), MLP(5, D)
            self.prop = MLP(4, D)
            self.typ = nn.Embedding(4, D)
            self.blocks = nn.ModuleList([self_relblock(D, heads) for _ in range(layers)])
            self.rel = nn.ModuleList([FactorSite(heads, D, "ctx>ctx", specs, UI_CARRIES) for _ in range(layers)])
            self.D = D

        def label_tokens(self, b):
            return self.lab(self.bag(b["wch"]))

        def _relctx(self, b, T):
            """Widget-token `TokenSet`/`RelCtx` at the `ctx>ctx` self-attention site: R20's `ui-rel-v1` edges
            (`b["wuiedges"]`, all-zero -- a no-op -- when the batch has none, e.g. no `table` was passed to
            `widget_features`) and the screen-geometry fields `pos3d` / `cam_uvd` / `zlayer` / `parent_id`
            (`b["wpos3d"]` etc., `ui.same_window` / `ui.above`'s own `same` / `order` ops over the LAST two), padded
            past the NW widget slots to the full `[instr, hist, prop]` token count `T` this site's self-attention
            actually spans (those tokens carry no widget geometry / UI-graph membership; `parent_id` pads with -1,
            `same_window`'s own "never matches" sentinel, matching `ui_public_fields`' null-slot convention)."""
            wmask = b["wmask"]
            B, device, pad = wmask.shape[0], wmask.device, T - NW
            pos3d = F.pad(b["wpos3d"], (0, 0, 0, pad))
            camuvd = F.pad(b["wcamuvd"], (0, 0, 0, pad))
            zlayer = F.pad(b.get("wzlayer", torch.zeros(B, NW, device=device)), (0, pad))[..., None]
            parent = F.pad(b.get("wparent", torch.full((B, NW), -1, dtype=torch.long, device=device)),
                           (0, pad), value=-1)[..., None]
            mask = torch.cat([wmask, torch.ones(B, pad, dtype=torch.bool, device=device)], 1)
            geo_ok = b.get("wgeo_ok", wmask)        # a pack without stored geometry (zeros) says so: `Demos.batch`
            geo_ok = F.pad(geo_ok, (0, pad))
            ts = TokenSet("ctx", mask, fields={"pos3d": pos3d, "pos3d.valid": geo_ok, "cam_uvd": camuvd,
                                               "cam_uvd.valid": geo_ok, "zlayer": zlayer, "parent_id": parent})
            wuiedges = b.get("wuiedges")
            edges = (torch.zeros(B, NW, NW, len(UI_REL_VOCAB), device=device) if wuiedges is None
                     else wuiedges.to(pos3d.dtype))
            edges = F.pad(edges, (0, 0, 0, pad, 0, pad))
            return RelCtx(sets={"ctx": ts}, edges={"ctx>ctx": EdgeSet(UI_REL_VOCAB, edges)})

        def forward(self, b):
            B = b["ptr"].shape[0]
            rel = b["wf"][..., :2] - b["ptr"][:, None]
            w = (self.label_tokens(b) + self.role(b["wrole"]) + self.bound(b["wbound"])
                 + self.wf(torch.cat([b["wf"], rel], -1)) + self.typ.weight[0])
            ins = self.sym(b["instr"]) + self.cpos.weight[:LI] + self.typ.weight[1]
            h = b["hist"]
            hr = h[..., 2:4] - b["ptr"][:, None]
            ht = (self.hk(h[..., 0].long()) + self.sym(h[..., 1].long())
                  + self.hf(torch.cat([h[..., 2:4], hr, h[..., 4:5] / 20.0], -1))
                  + self.typ.weight[2])
            p = self.prop(torch.stack([b["ptr"][:, 0], b["ptr"][:, 1], b["btn"], b["tick"] / 100.0], -1))[:, None] \
                + self.typ.weight[3]
            x = torch.cat([w, ins, ht, p], 1)
            m = torch.cat([b["wmask"], b["instr"] > 0, h[..., 0] > 0, torch.ones(B, 1, dtype=torch.bool,
                                                                                device=x.device)], 1)
            rc = self._relctx(b, x.shape[1])
            for L, site in zip(self.blocks, self.rel):
                xn = L.n2(x)                                  # the exact pre-self-attention hidden (`.x` is a
                                                                # zero-init no-op, docs 3.2, so this equals n2 of L's
                                                                # own input): kernel-compatible q/k augmentation
                bias_s, aug_s = site.bias(rc), site.augment(rc, xn, xn)
                x = L(x, kv=x, q_mask=m, bias_s=bias_s, aug_s=aug_s)
            return x, m

    class Knots(nn.Module):
        """K knot queries (M = 1 assembly: the pointer tool)."""

        def __init__(self, D, K=len(KNOT_TIMES)):
            super().__init__()
            self.q = nn.Parameter(torch.randn(K, D) * 0.02)
            self.t = MLP(D, D)
            self.register_buffer("kt", torch.tensor(KNOT_TIMES, dtype=torch.float32))

        def forward(self, B):
            return (self.q + self.t(sinusoidal(self.kt, self.q.shape[1])))[None].expand(B, -1, -1)

    def tick_tokens(D):
        return nn.ModuleDict(dict(f=MLP(6, D), key=nn.Embedding(N_KEYCLS, D), j=nn.Embedding(8, D)))

    def embed_ticks(mod, a):
        """a: dict of demo chunk tensors dxy [B,H,2], xy [B,H,2], btn [B,H], key [B,H] (class), valid [B,H]."""
        H = a["btn"].shape[1]
        x = mod["f"](torch.cat([a["dxy"], a["xy"], a["btn"][..., None], a["valid"][..., None].float()], -1))
        return x + mod["key"](a["key"]) + mod["j"].weight[:H]

    class PointerEncoder(nn.Module):
        """E (training only): public context at t + demonstrated tick commands a[t : t+7] -> z [B, K, 1, dz]."""

        def __init__(self, dz=16, D=128, heads=4, layers=2, factors=None):
            super().__init__()
            self.ctx = UICtx(D, heads, 2, specs=PolicyConfig(factors).specs())
            self.knots, self.ticks = Knots(D), tick_tokens(D)
            self.blocks = nn.ModuleList([cross_relblock(D, heads) for _ in range(layers)])
            self.out = nn.Linear(D, 2 * dz)
            self.dz = dz

        def forward(self, b, a):
            t, m = self.ctx(b)
            kv = torch.cat([t, embed_ticks(self.ticks, a)], 1)
            km = torch.cat([m, a["valid"]], 1)
            q = self.knots(t.shape[0])
            for L in self.blocks:
                q = L(q, kv=kv, kv_mask=km)
            mu, lv = self.out(q).chunk(2, -1)
            return mu[:, :, None], lv.clamp(-8, 4)[:, :, None]

    class PointerRealizer(nn.Module):
        """Learned system 0. Inputs ONLY: z, knot times, phase since valid_from, measured pointer (x, y), button.
        Outputs for this tick: pointer step (dxy / 60 px), button logit, key logits (0 = none)."""

        def __init__(self, dz=16, D=128, heads=4, layers=2):
            super().__init__()
            self.z_in, self.dt = nn.Linear(dz, D), MLP(D, D)
            self.loc = MLP(3, D)
            self.ph = MLP(D, D)
            self.blocks = nn.ModuleList([cross_relblock(D, heads) for _ in range(layers)])
            self.xy, self.btn, self.key = nn.Linear(D, 2), nn.Linear(D, 1), nn.Linear(D, N_KEYCLS)
            self.register_buffer("kt", torch.tensor(KNOT_TIMES, dtype=torch.float32))
            self.D = D

        def forward(self, z, phase, ptr, btn):
            tok = self.z_in(z[:, :, 0]) + self.dt(sinusoidal(self.kt[None] - phase[:, None], self.D))
            q = (self.loc(torch.cat([ptr, btn[:, None]], -1)) + self.ph(sinusoidal(phase, self.D)))[:, None]
            for L in self.blocks:
                q = L(q, kv=tok)
            q = q[:, 0]
            return self.xy(q), self.btn(q)[:, 0], self.key(q)

    class PointerFlow(nn.Module):
        """System i: rectified flow over the standardized packet, conditioned on the public context only."""

        def __init__(self, dz=16, D=128, heads=4, layers=3, factors=None):
            super().__init__()
            self.ctx = UICtx(D, heads, 3, specs=PolicyConfig(factors).specs())
            self.knots = Knots(D)
            self.z_in, self.t_in = nn.Linear(dz, D), MLP(D, D)
            self.blocks = nn.ModuleList([cross_relblock(D, heads) for _ in range(layers)])
            self.self_blocks = nn.ModuleList([self_relblock(D, heads) for _ in range(layers)])
            self.out = nn.Linear(D, dz)
            self.register_buffer("z_mean", torch.zeros(dz))
            self.register_buffer("z_std", torch.ones(dz))
            self.dz = dz

        def velocity(self, zt, tau, cache):
            t, m = cache
            h = self.knots(zt.shape[0]) + self.z_in(zt[:, :, 0]) + self.t_in(sinusoidal(tau, t.shape[-1]))[:, None]
            for X, S in zip(self.blocks, self.self_blocks):
                h = X(h, kv=t, kv_mask=m)
                h = S(h, kv=h)
            return self.out(h)[:, :, None]

        def loss(self, b, z_target, probe_fn=None, w_sem=0.0, tau_min=0.6):
            cache = self.ctx(b)
            x1 = (z_target - self.z_mean) / self.z_std
            eps = torch.randn_like(x1)
            tau = torch.rand(x1.shape[0], device=x1.device)
            t_ = tau[:, None, None, None]
            zt = (1 - t_) * eps + t_ * x1
            v = self.velocity(zt, tau, cache)
            fl = ((v - (x1 - eps)) ** 2).mean()
            logs = dict(flow=float(fl.detach()))
            loss = fl
            if probe_fn is not None and w_sem > 0:
                zc = (zt + (1 - t_) * v) * self.z_std + self.z_mean
                keep = (tau >= tau_min)[:, None, None, None]
                pl, pl_logs = probe_fn(torch.where(keep, zc, zc.detach()))
                loss = loss + w_sem * pl
                logs.update({f"zhat_{k}": x for k, x in pl_logs.items()})
            return loss, logs

        @torch.no_grad()
        def sample(self, b, nfe=8, generator=None):
            cache = self.ctx(b)
            B = b["ptr"].shape[0]
            z = torch.randn(B, len(KNOT_TIMES), 1, self.dz, device=b["ptr"].device, generator=generator)
            for k in range(nfe):
                tau = torch.full((B,), k / nfe, device=z.device)
                z = z + self.velocity(z, tau, cache) / nfe
            return z * self.z_std + self.z_mean

    class PointerBC(nn.Module):
        """BC baseline (same public inputs): 7 tick queries -> absolute pointer xy (normalized), button, key logits."""

        def __init__(self, D=128, heads=4, layers=3, H=7, factors=None):
            super().__init__()
            self.ctx = UICtx(D, heads, 3, specs=PolicyConfig(factors).specs())
            self.q = nn.Parameter(torch.randn(H, D) * 0.02)
            self.blocks = nn.ModuleList([cross_relblock(D, heads) for _ in range(layers)])
            self.self_blocks = nn.ModuleList([self_relblock(D, heads) for _ in range(layers)])
            self.xy, self.btn, self.key = nn.Linear(D, 2), nn.Linear(D, 1), nn.Linear(D, N_KEYCLS)

        def forward(self, b):
            t, m = self.ctx(b)
            h = self.q[None].expand(t.shape[0], -1, -1)
            for X, S in zip(self.blocks, self.self_blocks):
                h = X(h, kv=t, kv_mask=m)
                h = S(h, kv=h)
            return self.xy(h), self.btn(h)[..., 0], self.key(h)

    return dict(UICtx=UICtx, PointerEncoder=PointerEncoder, PointerRealizer=PointerRealizer,
                PointerFlow=PointerFlow, PointerBC=PointerBC, F=F,
                cross_relblock=cross_relblock, self_relblock=self_relblock)


_NETS: dict | None = None


def nets() -> dict:
    global _NETS
    if _NETS is None:
        _NETS = _nets()
    return _NETS


# ------------------------------------------------------------------------------------------- D-144 R6: relation-factor probe
# Former `PointerProbe` (content-keyed: cross-attended real widget label/role/geometry) -> `nets.probes.ReadoutProbe`
# configured by `probes:pointer-v1` (catalog.py). M = 1 (the pointer body's single "tool" assembly), so every query
# addresses `knot×asm` (psi0/legged precedent, docs/relations.md 4): the generic head reads only z's own K*M packet
# tokens plus fixed random handle codes, no widget content -- the "opaque codes only" design every other family's
# probe already follows (nets/probes.py's own docstring), which this migration now brings the pointer probe into
# line with. `target_slot` (query name `slot`, kept identical to the old dict key) is a fixed NW=80-way
# classification of the packet's target widget by SLOT INDEX (stable within an episode, docs/architecture.md
# ComputerWorld notes) rather than by content -- a probe that can no longer partly cheat off widget text/role, a
# strictly harder and more honest test of what `z` encodes than the pre-migration probe was.
POINTER_PROBE_PRESET = "probes:pointer-v1"


def pointer_probe_specs(lv_min: float | None = None):
    """`preset:probes:pointer-v1`, with the Gaussian `rel` query's `params.lv_min` overridden when given (former CLI
    `--lv-min`; D-085 bounded NLL, as the arm's semfix). Mirrors `policies.psi0.nets._with_lv_min`."""
    from dataclasses import replace
    from rrp.policies.relations.base import get_factor, resolve
    specs = resolve([f"preset:{POINTER_PROBE_PRESET}"])
    if lv_min is None:
        return specs
    out = []
    for s in specs:
        if get_factor(s.name).readout.loss == "gauss":
            p = dict(s.p); p["lv_min"] = lv_min
            s = replace(s, params=tuple(sorted(p.items())))
        out.append(s)
    return tuple(out)


def new_pointer_probe(dz=16, D=96, heads=4, metadata_only=False, seed=1234, lv_min: float | None = None):
    """The pointer packet probe: a `ReadoutProbe` (docs/relations.md 4) on `pointer_probe_specs(lv_min)`. Old
    `PointerProbe` checkpoints do not strictly load (different architecture); `load_pointer_probe_state` drops them
    and refits, the same fallback psi0 (R5) uses for the identical situation (D-144 addendum a, which names R6)."""
    from rrp.policies.nets.probes import ReadoutProbe
    return ReadoutProbe(dz, len(KNOT_TIMES), specs=pointer_probe_specs(lv_min), width=D, heads=heads,
                        max_assemblies=1, metadata_only=metadata_only, seed=seed)


def run_pointer_probe(P, z) -> dict:
    """P(z) narrowed to the former `PointerProbe` output shapes (M = 1 squeezed out of every `knot×asm` output):
    slot [B,K,NW] classification logits, rel [B,K,4] (Gaussian mu(2)+logvar(2)), phase [B,K,n_phases] logits."""
    import torch
    B = z.shape[0]
    zmask = torch.ones(B, 1, dtype=torch.bool, device=z.device)
    out = P(z, zmask)
    return {k: v[:, :, 0] for k, v in out.items()}


def _is_old_pointer_probe_state(sd: dict) -> bool:
    """True for a pre-D-144-R6 `PointerProbe` state dict (module names `bag` / `wf` / `role` / `rel` / `phase` /
    bare `kq`, none of which the new `ReadoutProbe` layout uses)."""
    return any(k.startswith(("bag.", "wf.", "role.", "rel.", "phase.")) or k == "kq" for k in sd)


def load_pointer_probe_state(sd: dict, *, dz=16, D=96, heads=4, metadata_only=False, seed=1234,
                             lv_min: float | None = None, device="cpu"):
    """`new_pointer_probe(...)`, loaded from `sd`: an old-architecture `PointerProbe` state is dropped (fresh init,
    refit -- see `new_pointer_probe`'s docstring); a post-D-144 `ReadoutProbe` state loads strictly."""
    P = new_pointer_probe(dz=dz, D=D, heads=heads, metadata_only=metadata_only, seed=seed, lv_min=lv_min).to(device)
    if _is_old_pointer_probe_state(sd):
        return P
    missing, unexpected = P.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"PointerProbe: key mismatch missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}")
    return P


# ------------------------------------------------------------------------------------- D-144 R6: RelBlock checkpoint map
# `_BLOCK_PATHS[cls]`: every (dotted path prefix, mode) of a `blocks`-style ModuleList the class holds, `mode` being
# which `RelBlock` stage the former bespoke `Block`'s one attention module becomes ("cross" -> `.x`, old `n1` stays
# `n1`; "self" -> `.s`, old `n1` becomes `n2`, the LayerNorm immediately before `.s`). Old `n2` (pre-MLP LayerNorm)
# always becomes `n3`; old `m` always stays `m`. See `cross_relblock` / `self_relblock` above for why the OTHER
# stage needs no entry (it is left at its zero-init-output construction, an exact no-op).
_BLOCK_PATHS = {
    "PointerEncoder": (("ctx.blocks.", "self"), ("blocks.", "cross")),
    "PointerRealizer": (("blocks.", "cross"),),
    "PointerFlow": (("ctx.blocks.", "self"), ("blocks.", "cross"), ("self_blocks.", "self")),
    "PointerBC": (("ctx.blocks.", "self"), ("blocks.", "cross"), ("self_blocks.", "self")),
}


def _remap_block_state(sd: dict, prefix: str, mode: str) -> dict:
    """One (prefix, mode) entry of `_BLOCK_PATHS`: the old `Block` keys under `prefix` -> their `RelBlock` layout
    (see `_BLOCK_PATHS`'s docstring). Raises on a key it does not recognize (never silently drops data)."""
    ln1_new = "n1" if mode == "cross" else "n2"
    att_new = "x" if mode == "cross" else "s"
    out = {}
    for k, v in sd.items():
        if not k.startswith(prefix):
            continue
        i, sub = k[len(prefix):].split(".", 1)
        if sub.startswith("n1."):
            out[f"{prefix}{i}.{ln1_new}.{sub[len('n1.'):]}"] = v
        elif sub.startswith("a."):
            out[f"{prefix}{i}.{att_new}.{sub[len('a.'):]}"] = v
        elif sub.startswith("n2."):
            out[f"{prefix}{i}.n3.{sub[len('n2.'):]}"] = v
        elif sub.startswith("m."):
            out[f"{prefix}{i}.{sub}"] = v
        else:
            raise ValueError(f"_remap_block_state: unmapped old block key {k!r}")
    return out


def load_pointer_module(cls_name: str, ctor, sd: dict, device="cpu"):
    """Construct `ctor()` and load `sd`, remapping every `_BLOCK_PATHS[cls_name]` path whose checkpoint predates
    D-144 R6 (detected per-path by the old block's `.a.` attention key, which the new layout never has; a
    post-D-144 checkpoint has no such key anywhere and passes through unremapped). The only keys allowed to stay
    missing after a remap are the OTHER (zero-init, never-loaded) stage's parameters."""
    m = ctor().to(device)
    sd = dict(sd)
    expect_missing: set[str] = set()
    for prefix, mode in _BLOCK_PATHS.get(cls_name, ()):
        old = any(k.startswith(prefix) and k[len(prefix):].split(".", 1)[1].startswith("a.") for k in sd)
        if not old:
            continue
        remapped = _remap_block_state(sd, prefix, mode)
        for k in [k for k in sd if k.startswith(prefix)]:
            del sd[k]
        sd.update(remapped)
        unused = ("n2.", "s.") if mode == "cross" else ("n1.", "x.")   # the OTHER stage: its LayerNorm too
        expect_missing |= {k for k in m.state_dict()
                          if k.startswith(prefix) and k[len(prefix):].split(".", 1)[1].startswith(unused)}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    bad = [k for k in missing if k not in expect_missing] + list(unexpected)
    if bad:
        raise RuntimeError(f"{cls_name}: key mismatch {bad[:5]}")
    return m


# ------------------------------------------------------------------------------------------------ learned runtime
STEP_M = MAX_STEP_PX * 1e-3          # realizer pointer-step unit (m at the default 1 mm/px)


class LearnedSystem0(System0Base):
    """Learned pointer system 0 (PointerRealizer): z + phase + measured pointer/button -> this tick's command."""

    def __init__(self, R, env, *, latent_space_version: str, realizer_compat_version: str, device="cpu"):
        super().__init__(latent_space_version=latent_space_version, realizer_compat_version=realizer_compat_version)
        self.R, self.device = R.eval(), device
        self.spec_hash = env.body.spec_hash
        self.half = screen_half(env.spec)
        self.dt = 1.0 / env.spec.control_hz
        self.step_m = MAX_STEP_PX * env.spec.frame["m_per_px"]

    @property
    def robot_spec_hash(self) -> str:
        return self.spec_hash

    def tick(self, env, controller_version=None) -> NativeCommand | None:
        import torch
        now = float(env.time)
        p = self.packet
        if p is None or now > p.valid_until - 1e-9:
            if p is not None:
                self.invalidate("expired", now)
            self.stats.fallback_holds += 1
            return None
        ph = now - p.valid_from
        if tick_slot(ph, self.dt, p.knot_times) is None:
            self.stats.fallback_holds += 1
            return None
        obs = env.observe()
        q = np.asarray(obs.measured_node_state.qpos[:2], np.float32)
        btn = float(obs.declared_sensor_channels[0].values[0])
        dev = self.device
        with torch.no_grad():
            dxy, bl, kl = self.R(torch.from_numpy(np.asarray(p.z, np.float32))[None].to(dev),
                                 torch.tensor([ph], dtype=torch.float32, device=dev),
                                 torch.from_numpy(q / self.half)[None].to(dev), torch.tensor([btn], device=dev))
        d = np.clip(dxy[0].float().cpu().numpy(), -1.0, 1.0) * self.step_m
        g = {"pointer": [float(q[0] + d[0]), float(q[1] + d[1])], "button": [1.0 if float(bl[0]) > 0 else 0.0]}
        k = int(kl[0].argmax())
        if k > 0:
            g["key"] = [float(k - 1)]
        self.stats.ticks += 1
        return NativeCommand(controller_version=controller_version or "cw_pointer.v1", groups=g, source=_cmd_source(p))


class PointerSystemI:
    """Learned system i: public features (+ own event history) -> flow sample -> packet. `target="latent"` emits the
    learned latent (for LearnedSystem0); `target="eng"` emits the engineered encoding (for EngineeredSystem0)."""

    def __init__(self, flow, *, lsv: str, rcv: str, device="cpu", nfe=8, seed=0, name="pointer_system_i",
                 validity=VALIDITY_S):
        import torch
        self.flow, self.lsv, self.rcv, self.device, self.nfe = flow.eval(), lsv, rcv, device, nfe
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.name, self.validity = name, validity
        self.calls = 0
        self.hist: dict[int, EventHistory] = {}

    def reset(self, envs):
        self.hist = {id(e): EventHistory() for e in envs}

    def featurizer(self, env):
        return None

    def executed(self, env, command) -> None:
        self.hist[id(env)].push(env.steps, None if command is None else command.groups, screen_half(env.spec))

    def packets(self, envs) -> list[LatentActionChunk]:
        import torch
        obs = [e.observe() for e in envs]
        fs = [public_features(o, screen_half(e.spec), self.hist[id(e)], e.steps, table=env_widget_table(e))
              for e, o in zip(envs, obs)]
        with torch.no_grad():
            z = self.flow.sample(collate_public(fs, self.device), nfe=self.nfe, generator=self.gen).float().cpu().numpy()
        self.calls += len(envs)
        return [pointer_packet(e, o, z[i], lsv=self.lsv, rcv=self.rcv, source="learned", name=self.name,
                               sampling=dict(nfe=self.nfe, sampler="euler"), validity=self.validity)
                for i, (e, o) in enumerate(zip(envs, obs))]


def load_pointer_bundle(path: str, device="cpu") -> dict:
    """A pointer checkpoint (rrp.harness.train.pointer): {'kind', 'config', 'state' (module -> state_dict), 'versions'}.
    `E` / `R` / `S` / `BC` load through `load_pointer_module` (RelBlock checkpoint map, D-144 R6: strict for a
    post-R6 checkpoint, key-mapped for a pre-R6 one); `P` (the packet probe) through `load_pointer_probe_state`
    (different architecture pre/post R6: key-mapped load is a refit, not a strict load, for an old checkpoint)."""
    import torch
    st = torch.load(path, map_location=device, weights_only=False)
    N, cfg = nets(), st["config"]
    mods = {}
    for k, sd in st["state"].items():
        if k == "P":
            mods[k] = load_pointer_probe_state(sd, device=device, **cfg["arch"].get("P", {})).eval()
            continue
        cls = {"E": "PointerEncoder", "R": "PointerRealizer", "S": "PointerFlow", "BC": "PointerBC"}[k]
        mods[k] = load_pointer_module(cls, lambda cls=cls, kw=cfg["arch"].get(k, {}): N[cls](**kw), sd,
                                      device=device).eval()
    return dict(st, modules=mods)


def make_pointer_latent(*, flow: str, representation: str | None = None, device: str = "cpu", replan_ticks: int = 4,
                        nfe: int = 8, seed: int = 0, name: str = "pointer_latent"):
    """Learned route: system i (PointerFlow checkpoint `flow`) -> packet -> learned system 0 (the realizer of the flow's
    frozen representation), or, for a flow trained on the engineered encoding, -> the SCRIPTED engineered system 0."""
    from rrp.policies.latent import LatentStackPolicy
    fb = load_pointer_bundle(flow, device)
    variant, target = fb["config"]["variant"], fb["config"]["target"]
    tasks = fb["config"].get("tasks")
    if target == "eng":
        si = PointerSystemI(fb["modules"]["S"], lsv=ENG_VERSION, rcv=ENG_VERSION, device=device, nfe=nfe, seed=seed,
                            name=f"{name}:{flow}")
        return LatentStackPolicy(si, None, replan_ticks=replan_ticks, name=name, source="learned", variant=variant,
                                 version=f"learned:{flow}|system0={EngineeredSystem0.label}", make_s0=EngineeredSystem0,
                                 requires=pointer_requirements(tasks=tasks))
    rb = load_pointer_bundle(representation or fb["config"]["representation"], device)
    lsv, rcv = rb["versions"]["latent_space_version"], rb["versions"]["realizer_compat_version"]
    if fb["versions"]["latent_space_version"] != lsv:
        from rrp.core.errors import ControllerRejection
        raise ControllerRejection(f"flow {flow} was trained on latent space {fb['versions']['latent_space_version']}, "
                                  f"representation is {lsv}", code="latent_space_mismatch")
    si = PointerSystemI(fb["modules"]["S"], lsv=lsv, rcv=rcv, device=device, nfe=nfe, seed=seed, name=f"{name}:{flow}")
    R = rb["modules"]["R"]
    return LatentStackPolicy(si, R, replan_ticks=replan_ticks, name=name, source="learned", variant=variant,
                             version=f"learned:{flow}|system0=learned:{rcv}",
                             make_s0=lambda e: LearnedSystem0(R, e, latent_space_version=lsv,
                                                              realizer_compat_version=rcv, device=device),
                             requires=pointer_requirements(tasks=tasks))


class PointerBCPolicy:
    """BC baseline (source bc): the same public features; every `replan_ticks` ticks a 7-tick chunk of absolute
    pointer / button / key commands, executed tick by tick (CW has no chunk executor)."""

    def __init__(self, bundle: dict, *, path: str, replan_ticks: int = 4, device: str = "cpu", name: str = "pointer_bc"):
        from rrp.policies.base import PolicyInfo
        self.net, self.replan, self.device = bundle["modules"]["BC"], int(replan_ticks), device
        self.info = PolicyInfo(name, "bc", f"bc:{path}",
                               pointer_requirements(tasks=bundle["config"].get("tasks")), "bc")

    def reset(self, spec, task, seeds, *, envs=None):
        self.envs = list(envs)
        self.hist = [EventHistory() for _ in self.envs]
        self.chunk = [None] * len(self.envs)
        self.t = [0] * len(self.envs)
        self.t0 = [0] * len(self.envs)

    def act(self, obs):
        import torch
        from rrp.policies.base import Act
        idx = sorted(obs)
        need = [i for i in idx if self.t[i] % self.replan == 0 or self.chunk[i] is None]
        if need:
            fs = [public_features(obs[i], screen_half(self.envs[i].spec), self.hist[i], self.envs[i].steps,
                                  table=env_widget_table(self.envs[i])) for i in need]
            with torch.no_grad():
                xy, bl, kl = self.net(collate_public(fs, self.device))
            for n, i in enumerate(need):
                half = screen_half(self.envs[i].spec)
                self.chunk[i] = [(xy[n, j].float().cpu().numpy() * half, float(bl[n, j]), int(kl[n, j].argmax()))
                                 for j in range(xy.shape[1])]
                self.t0[i] = self.t[i]
        out = {}
        for i in idx:
            j = self.t[i] - self.t0[i]
            (x, y), b, k = self.chunk[i][min(j, len(self.chunk[i]) - 1)]
            g = {"pointer": [float(x), float(y)], "button": [1.0 if b > 0 else 0.0]}
            if k > 0:
                g["key"] = [float(k - 1)]
            self.hist[i].push(self.envs[i].steps, g, screen_half(self.envs[i].spec))
            out[i] = Act(NativeCommand(controller_version="cw_pointer.v1", groups=g, source="bc"))
            self.t[i] += 1
        return out


def make_pointer_bc(*, checkpoint: str, device: str = "cpu", replan_ticks: int = 4, name: str = "pointer_bc"):
    return PointerBCPolicy(load_pointer_bundle(checkpoint, device), path=checkpoint, replan_ticks=replan_ticks,
                           device=device, name=name)
