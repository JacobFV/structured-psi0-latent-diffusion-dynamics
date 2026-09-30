"""Pointer-body (ComputerWorld `cw_pointer`) packet encoding, packet construction and the SCRIPTED system 0
(docs/architecture.md sections 3 and 5).

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
ENGINEERED packet encoding `cw_pointer_eng.v2` (z[4, 1, 26]; D-146 C2, architecture 14.6): the same slots with the
scalar key replaced by a 7-bit +-1 code of the key CLASS (KEY_VOCAB index + 1, 0 = none; little-endian bits). A scalar
`key` is a 1-of-109 quantity squeezed onto one axis, where a flow's blur lands between unrelated keys; a bit code
puts a blur one bit away. Per slot [x, y, depth, button, wheel, flag, key0 .. key6]. The packet's
`latent_space_version` tag picks the layout (`spec.eng_layout`): v1 stays decodable, nothing is guessed from shape.
`EngineeredSystem0` (source label: scripted) realizes it every tick from the MEASURED pointer: it moves toward the
slot's target by at most MAX_STEP_PX per tick (the scripted teachers' speed), sets the button and emits the key and
wheel events; an unplanned slot, an expired packet or no packet holds (declared fallback).
"""
from __future__ import annotations

import time

import numpy as np

from rrp.core.action import NativeCommand
from rrp.core.latent_action import AssemblyHandle, EntityHandle, LatentActionChunk
from rrp.core.system0 import System0Base
from rrp.policies.pointer.spec import ENG_VERSION, KNOT_TIMES, MAX_STEP_PX, VALIDITY_S, eng_layout


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


def key_bits(cls: int, n: int = 7) -> list[float]:
    """Key class (0 = none, 1 + KEY_VOCAB index) -> its `n`-bit +-1 code, least significant bit first."""
    return [1.0 if (cls >> i) & 1 else -1.0 for i in range(n)]


def key_from_bits(bits) -> int:
    """Inverse of `key_bits` (a bit is set when its value is > 0): the key class."""
    return sum(1 << i for i, b in enumerate(bits) if float(b) > 0)


def encode_commands(cmds, dt: float, n_keys: int, depths=None, knot_times=KNOT_TIMES, *,
                    version: str = ENG_VERSION) -> np.ndarray:
    """Tick command groups (None = no plan) -> engineered z [K, 1, layout.dim]. cmds[j] is the command of tick j."""
    lay = eng_layout(version)
    if lay.key_bits and n_keys + 1 > 2 ** lay.key_bits:
        raise ValueError(f"{n_keys} keys do not fit a {lay.key_bits}-bit code")
    z = np.zeros((len(knot_times), 1, lay.dim), np.float32)
    for j, (k, s) in enumerate(packet_ticks(dt, knot_times)):
        if j >= len(cmds) or cmds[j] is None:
            continue
        g, o = cmds[j], s * lay.slot_w
        x, y = g["pointer"]
        key = int(round(g.get("key", [-1])[0]))
        base = dict(x=x, y=y, depth=0.0 if depths is None else depths[j],
                    button=1.0 if g["button"][0] >= 0.5 else -1.0, wheel=g.get("wheel", [0.0])[0], flag=1.0)
        if lay.key_bits:
            v = [base[f] for f in lay.fields[:lay.key]] + key_bits(key + 1, lay.key_bits)
        else:
            v = [base["x"], base["y"], base["depth"], base["button"], (key + 1) / n_keys, base["wheel"], 1.0]
        z[k, 0, o:o + lay.slot_w] = v
    return z


def decode_slot(z: np.ndarray, k: int, s: int, n_keys: int, *, version: str = ENG_VERSION) -> dict | None:
    """One tick slot of an engineered z -> {x, y, depth, button, key, wheel} (None = unplanned). A v2 key code past
    the vocabulary decodes to no key (-1), never to another key."""
    lay = eng_layout(version)
    v = np.asarray(z)[k, 0, s * lay.slot_w:(s + 1) * lay.slot_w]
    if v[lay.idx("flag")] < 0.5:
        return None
    if lay.key_bits:
        key = key_from_bits(v[lay.key:lay.key + lay.key_bits]) - 1
        key = key if key < n_keys else -1
    else:
        key = int(round(float(v[4]) * n_keys)) - 1
    return dict(x=float(v[0]), y=float(v[1]), depth=float(v[2]), button=bool(v[3] > 0), key=key,
                wheel=int(round(float(v[lay.idx("wheel")]))))


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


def cmd_source(packet) -> str:
    return {"target_encoder_oracle": "oracle"}.get(packet.source, packet.source)


class EngineeredSystem0(System0Base):
    """SCRIPTED system 0 for the pointer: realizes a `cw_pointer_eng.v1` / `.v2` packet (`version`; module docstring)
    every tick from the measured pointer (qpos) and button sensor. Inputs: packet z, phase since valid_from, pointer
    x/y, button. It accepts only packets tagged with its own version (`System0Base.receive`)."""
    label = f"scripted:{ENG_VERSION}"            # the default version's label; `label_for(version)` for any

    @staticmethod
    def label_for(version: str) -> str:
        return f"scripted:{eng_layout(version).version}"

    def __init__(self, env, *, max_step_px: int = MAX_STEP_PX, version: str = ENG_VERSION):
        super().__init__(latent_space_version=version, realizer_compat_version=version)
        self.version, self.label = eng_layout(version).version, self.label_for(version)
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
        sl = None if ks is None else decode_slot(p.z, *ks, self.n_keys, version=self.version)
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
        return NativeCommand(controller_version=controller_version or "cw_pointer.v1", groups=g, source=cmd_source(p))
