"""Joint-target controller (system 0 for fixed arms/grippers) and chunk executor.

The controller owns specific actuators per its contract. It validates every command:
version, group names/widths, finiteness, bounds (small tolerance clipped, larger rejected).
Targets are linearly interpolated over the control period on physics substeps. Stale or
mismatched action chunks are rejected, never silently clipped into a new plan.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import mujoco
import numpy as np

from rrp.contracts.action import ActionChunk, NativeCommand
from rrp.contracts.errors import ControllerRejection, StaleActionError
from rrp.contracts.robot import ControllerContract, RobotSpec


@dataclass
class ControllerStats:
    accepted: int = 0
    rejected: int = 0
    clipped: int = 0
    reasons: list = field(default_factory=list)


class JointTargetController:
    def __init__(self, model: mujoco.MjModel, robot: RobotSpec, contract: ControllerContract, prefix: str,
                 bound_tolerance: float = 0.05):
        self.model = model
        self.robot = robot
        self.contract = contract
        self.version = f"{contract.id}:{contract.version}:{robot.spec_hash}"
        self.tol = bound_tolerance
        self.groups = {g.name: g for g in contract.command_groups}
        addr_to_act = {a.address: a.name for a in robot.actuators}
        self.act_ids = {}
        for g in contract.command_groups:
            ids = []
            for addr in g.actuators:
                nm = addr_to_act[addr]
                full = nm if nm.startswith(prefix) else prefix + nm
                u = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, full)
                if u < 0:
                    raise ControllerRejection(f"actuator {full} not in model", code="controller_mismatch")
                ids.append(u)
            self.act_ids[g.name] = np.array(ids, dtype=int)
        self.prev = None
        self.target = None
        self.stats = ControllerStats()

    # ---------------------------------------------------------------- command validation
    def validate(self, cmd: NativeCommand) -> dict[str, np.ndarray]:
        if cmd.controller_version != self.version:
            raise StaleActionError(f"controller version {cmd.controller_version} != {self.version}")
        out = {}
        for name, vals in cmd.groups.items():
            if name not in self.groups:
                raise ControllerRejection(f"unknown command group {name}", code="unknown_group")
            g = self.groups[name]
            v = np.asarray(vals, dtype=float)
            if v.shape != (g.width,):
                raise ControllerRejection(f"group {name}: width {v.shape} != {g.width}", code="wrong_width")
            if not np.isfinite(v).all():
                raise ControllerRejection(f"group {name}: non-finite command", code="nonfinite")
            lo, hi = np.array(g.lower), np.array(g.upper)
            span = np.maximum(hi - lo, 1e-6)
            if np.any(v < lo - self.tol * span) or np.any(v > hi + self.tol * span):
                raise ControllerRejection(f"group {name}: command outside bounds", code="out_of_bounds")
            if np.any(v < lo) or np.any(v > hi):
                self.stats.clipped += 1
            out[name] = np.clip(v, lo, hi)
        return out

    def current_targets(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        return {n: data.ctrl[ids].copy() for n, ids in self.act_ids.items()}

    def measured_hold(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        out = {}
        for n, ids in self.act_ids.items():
            vals = []
            for u in ids:
                j = self.model.actuator_trnid[u, 0]
                vals.append(data.qpos[self.model.jnt_qposadr[j]])
            g = self.groups[n]
            out[n] = np.clip(np.array(vals), g.lower, g.upper) if g.hold == "hold_measured" else data.ctrl[ids].copy()
        return out

    def begin_step(self, data: mujoco.MjData, cmd: NativeCommand | None):
        """Set interpolation endpoints for this control period."""
        self.prev = self.current_targets(data)
        tgt = dict(self.prev)
        if cmd is not None:
            try:
                tgt.update(self.validate(cmd))
                self.stats.accepted += 1
            except ControllerRejection as e:
                self.stats.rejected += 1
                self.stats.reasons.append(e.code)
                raise
        self.target = tgt

    def apply_substep(self, data: mujoco.MjData, alpha: float):
        for n, ids in self.act_ids.items():
            data.ctrl[ids] = (1 - alpha) * self.prev[n] + alpha * self.target[n]

    def state(self) -> dict:
        return dict(prev={k: v.tolist() for k, v in (self.prev or {}).items()},
                    target={k: v.tolist() for k, v in (self.target or {}).items()},
                    stats=dict(self.stats.__dict__))

    def load(self, st: dict):
        self.prev = {k: np.array(v) for k, v in st["prev"].items()} or None
        self.target = {k: np.array(v) for k, v in st["target"].items()} or None
        self.stats = ControllerStats(**st["stats"])


class ChunkExecutor:
    """Queue of chunk rows. Rejects stale chunks; drops queued rows on invalidation."""

    def __init__(self):
        self.queue: deque = deque()
        self.meta: dict | None = None
        self.log: list[dict] = []

    def submit(self, chunk: ActionChunk, *, graph_version: int, robot_spec_hash: str, controller_version: str,
               now: float, execute_prefix: int | None = None):
        reasons = []
        if chunk.graph_version != graph_version:
            reasons.append("stale_graph_version")
        if chunk.robot_spec_hash != robot_spec_hash:
            reasons.append("robot_spec_mismatch")
        if chunk.controller_version != controller_version:
            reasons.append("controller_version_mismatch")
        if chunk.expires_at is not None and now > chunk.expires_at:
            reasons.append("expired")
        if reasons:
            self.log.append(dict(t=now, event="chunk_rejected", reasons=reasons, obs=chunk.observation_id,
                                 source=chunk.source))
            raise StaleActionError(f"action chunk rejected: {reasons}", reasons=reasons)
        n = chunk.horizon if execute_prefix is None else min(execute_prefix, chunk.horizon)
        self.queue.clear()
        for h in range(n):
            self.queue.append({g.group: g.values[h].tolist() for g in chunk.command_groups})
        self.meta = dict(observation_id=chunk.observation_id, source=chunk.source, policy=chunk.policy_version,
                         graph_version=chunk.graph_version, runtime_version=chunk.runtime_version)
        self.log.append(dict(t=now, event="chunk_accepted", obs=chunk.observation_id, rows=n, source=chunk.source))

    def invalidate(self, reason: str, now: float = 0.0):
        if self.queue:
            self.log.append(dict(t=now, event="queue_dropped", reason=reason, rows=len(self.queue)))
        self.queue.clear()
        self.meta = None

    def pop(self) -> dict | None:
        return self.queue.popleft() if self.queue else None

    def state(self) -> dict:
        return dict(queue=list(self.queue), meta=self.meta)

    def load(self, st: dict):
        self.queue = deque(st["queue"])
        self.meta = st["meta"]
