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


class TeacherOracleSource:
    """ORACLE DIAGNOSTIC packet source (privileged): the scripted teacher's next ticks, run on a shadow twin env, encoded
    with the engineered encoding. Metric: shadow/real divergences (state hash at the packet tick)."""
    name = "cw_teacher_oracle"
    lsv = rcv = ENG_VERSION

    def __init__(self, validity: float = VALIDITY_S):
        self.validity = validity
        self.calls = 0

    def reset(self, envs):
        from rrp.policies.teachers.computerworld import CWTeacher
        self.st = {}
        for e in envs:
            sh = _twin(e)
            self.st[id(e)] = dict(shadow=sh, teacher=CWTeacher(sh, e.task_name), cmds=[], depths=[],
                                  hashes=[sh.state_hash()], done=False, divergences=0)

    def featurizer(self, env):
        return None

    def _advance(self, st, upto: int):
        sh = st["shadow"]
        while len(st["cmds"]) < upto:
            c = None if st["done"] else st["teacher"].act()
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

    def packets(self, envs) -> list[LatentActionChunk]:
        out = []
        for e in envs:
            st, n = self.st[id(e)], e.steps
            dt = 1.0 / e.spec.control_hz
            H = len(packet_ticks(dt))
            self._advance(st, n + H)
            if n < len(st["hashes"]) and st["hashes"][n] != e.state_hash():
                st["divergences"] += 1
            z = encode_commands(st["cmds"][n:n + H], dt, len(e.spec.space("key").vocab), st["depths"][n:n + H])
            out.append(pointer_packet(e, e.observe(), z, lsv=self.lsv, rcv=self.rcv, source="oracle", name=self.name,
                                      validity=self.validity))
        self.calls += len(envs)
        return out


def pointer_requirements(*, privileged: bool = False, tasks=None):
    from rrp.policies.base import Requirements
    return Requirements(POINTER_KINDS, groups=frozenset({"pointer", "button", "key"}),
                        observations=frozenset({"proprio", "object_descriptors", "language"}),
                        body_families=frozenset({"pointer"}), tasks=frozenset(tasks) if tasks else None,
                        privileged=privileged)


def make_pointer_oracle(*, replan_ticks: int = 4, name: str = "pointer_oracle"):
    """ORACLE DIAGNOSTIC: scripted-teacher look-ahead -> engineered packet -> SCRIPTED engineered system 0."""
    from rrp.policies.latent import LatentStackPolicy
    return LatentStackPolicy(TeacherOracleSource(), None, replan_ticks=replan_ticks, name=name, source="oracle",
                             version=f"oracle:teacher->{ENG_VERSION}|system0={EngineeredSystem0.label}",
                             make_s0=EngineeredSystem0, requires=pointer_requirements(privileged=True))
