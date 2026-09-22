"""Authoritative task runtime: guards, lifecycle, receipts, retries, rejection memory.

Only the runtime changes execution status. Planned effects are never observed facts:
completion requires KNOWN observation estimates or validated receipts, held for the
declared persistence time. Unknown is not false. Requesting a later event never marks
its predecessors complete.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Protocol

from rrp.contracts.errors import ProvenanceError, VersionConflict
from rrp.contracts.observation import PolicyObservation, RuntimeEventView, ReceiptView, TaskInput
from rrp.contracts.task import Condition, EntityBinding, OutputBinding, TaskDefinition
from .compiler import CompiledTask, compile_task
from .interventions import GraphStore, EditReceipt
from .receipts import Receipt, ReceiptStore

RECEIPT_VALIDITY_PREDICATES = ("receipt_valid", "anchor_valid", "frame_valid")
STATUSES = ("pending", "ready", "active", "succeeded", "failed", "cancelled", "blocked")


@dataclass
class Estimate:
    value: bool | float | None
    known: bool
    confidence: float = 1.0
    source: str = "unknown"


UNKNOWN = Estimate(None, False, 0.0, "none")


class OutputProvider(Protocol):
    def __call__(self, event_id: str, output_name: str, output_type: str,
                 observation: PolicyObservation | None) -> dict | None: ...


@dataclass
class ExecutionDecision:
    accepted: bool
    event_id: str
    reason_code: str | None = None
    detail: str | None = None
    graph_version: int = 0
    runtime_version: int = 0


@dataclass
class Instance:
    event_id: str
    attempt: int = 0
    status: str = "pending"
    reason: str | None = None
    activated_at: float | None = None
    finished_at: float | None = None
    persistence_start: dict = field(default_factory=dict)
    requested: bool = False


@dataclass
class TaskContext:
    graph_version: int
    runtime_version: int
    statuses: dict
    active: list
    ready: list
    receipts: list
    changed: bool


class TaskRuntime:
    def __init__(self, task: CompiledTask | TaskDefinition | dict, *, clock: Callable[[], float] | None = None,
                 auto_advance: bool = False, output_provider: OutputProvider | None = None,
                 on_invalidate: Callable[[str], None] | None = None):
        if isinstance(task, CompiledTask):
            task = task.definition
        self.store = GraphStore(task)
        self.clock = clock or time.monotonic
        self.auto_advance = auto_advance
        self.output_provider = output_provider
        self.on_invalidate = on_invalidate
        self.receipts = ReceiptStore()
        self.runtime_version = 0
        self.instances: dict[str, Instance] = {e.id: Instance(e.id) for e in self.compiled.definition.events}
        self.attempt_log: list[dict] = []
        self.rejections: dict[str, list[dict]] = {}
        self.transitions: list[dict] = []
        self._last_obs: PolicyObservation | None = None
        self.store.on_commit(self._reconcile)
        self._refresh_ready()

    # ---------------------------------------------------------------- views
    @property
    def compiled(self) -> CompiledTask:
        return self.store.compiled

    @property
    def graph_version(self) -> int:
        return self.store.version

    def status(self, event_id: str) -> str:
        return self.instances[event_id].status

    def attempt(self, event_id: str) -> int:
        return self.instances[event_id].attempt

    def succeeded(self) -> bool:
        return all(self.instances[e].status == "succeeded" for e in self.compiled.definition.success_events)

    def failed_terminal(self) -> bool:
        return any(self.instances[e].status in ("failed", "cancelled") for e in self.compiled.definition.success_events)

    def resources_held(self) -> dict[str, str]:
        held = {}
        for eid, inst in self.instances.items():
            if inst.status == "active":
                for r in self.compiled.event(eid).resources:
                    if r.mode == "exclusive_control":
                        held[r.entity.id] = eid
        return held

    def controller_owner(self, entity_id: str) -> str | None:
        return self.resources_held().get(entity_id)

    def actor_of(self, event_id: str) -> list[str]:
        ev = self.compiled.event(event_id)
        return [s.binding.entity.id for s in sorted(ev.roles, key=lambda s: s.ordinal)
                if s.role in ("actor", "cooperating_actor") and isinstance(s.binding, EntityBinding)]

    def public_view(self) -> TaskInput:
        evs = []
        for eid, inst in self.instances.items():
            rej = self.rejections.get(eid, [])
            evs.append(RuntimeEventView(event_id=eid, attempt=inst.attempt, status=inst.status,
                                        reason_code=inst.reason, rejection_count=len(rej),
                                        last_rejection_reasons=[r["reason"] for r in rej[-4:]]))
        now = self.clock()
        recs = [ReceiptView(event_id=r.event_id, attempt=r.attempt, output_name=r.output_name, type=r.type,
                            version=r.version, valid=r.valid, value=r.value, covariance_diag=r.covariance_diag,
                            age_s=max(0.0, now - r.created_at)) for r in self.receipts.all()]
        return TaskInput(definition=TaskDefinition.model_validate(self.store.document()),
                         graph_version=self.graph_version, runtime_version=self.runtime_version,
                         events=evs, receipts=recs,
                         interventions=[h.request_id for h in self.store.history])

    # ---------------------------------------------------------------- transitions
    def _set(self, inst: Instance, status: str, reason: str | None = None):
        if inst.status != status or inst.reason != reason:
            self.transitions.append(dict(t=self.clock(), event=inst.event_id, attempt=inst.attempt,
                                         frm=inst.status, to=status, reason=reason,
                                         graph_version=self.graph_version))
            inst.status, inst.reason = status, reason
            self.runtime_version += 1

    def _reject(self, event_id: str, code: str, detail: str | None = None) -> ExecutionDecision:
        self.rejections.setdefault(event_id, []).append(dict(t=self.clock(), reason=code, detail=detail,
                                                              attempt=self.instances.get(event_id, Instance(event_id)).attempt,
                                                              graph_version=self.graph_version))
        return ExecutionDecision(False, event_id, code, detail, self.graph_version, self.runtime_version)

    # ---------------------------------------------------------------- guards
    def _binding_ok(self, event_id: str) -> tuple[bool, str | None, str | None]:
        ev = self.compiled.event(event_id)
        for slot in ev.roles:
            b = slot.binding
            if isinstance(b, OutputBinding):
                typ = self.compiled.outputs[(b.event_id, b.output_name)]
                try:
                    self.receipts.require(event_id=b.event_id, attempt=b.attempt, output_name=b.output_name, type=typ)
                except ProvenanceError as e:
                    return False, "binding_unavailable", f"{slot.role}[{slot.ordinal}]: {e.code}: {e}"
        return True, None, None

    def _prereqs(self, event_id: str) -> tuple[bool, str | None, str | None]:
        ev = self.compiled.event(event_id)
        for d in ev.requires_completed:
            if self.instances[d].status != "succeeded":
                return False, "prerequisite_unsatisfied", f"{d} is {self.instances[d].status}"
        for d in ev.requires_active:
            if self.instances[d].status != "active":
                return False, "maintained_prerequisite_inactive", f"{d} is {self.instances[d].status}"
        return True, None, None

    def _resources_ok(self, event_id: str) -> tuple[bool, str | None, str | None]:
        held = self.resources_held()
        for r in self.compiled.event(event_id).resources:
            if r.mode == "exclusive_control" and held.get(r.entity.id) not in (None, event_id):
                return False, "resource_claimed", f"{r.entity.id} held by {held[r.entity.id]}"
        return True, None, None

    def evaluate(self, cond: Condition, obs: PolicyObservation | None) -> Estimate:
        if cond.source == "validated_receipt":
            if cond.predicate == "event_succeeded":
                for a in cond.arguments:
                    if isinstance(a, OutputBinding):
                        r = self.receipts.latest(a.event_id, a.attempt, a.output_name)
                        return Estimate(bool(r and r.valid), True, 1.0, "receipt")
            if cond.predicate in RECEIPT_VALIDITY_PREDICATES:
                # validity of a bound receipt (e.g. a maintained support anchor): known from the
                # receipt store; a missing receipt is known-invalid, never silently true
                for a in cond.arguments:
                    if isinstance(a, OutputBinding):
                        r = self.receipts.latest(a.event_id, a.attempt, a.output_name)
                        return Estimate(bool(r and r.valid), True, 1.0, "receipt")
            return UNKNOWN
        if cond.source == "desired_effect":
            return UNKNOWN   # a target, never an observed fact
        if obs is None:
            return UNKNOWN
        args = [a.entity.id if isinstance(a, EntityBinding) else f"{a.event_id}#{a.attempt}.{a.output_name}"
                for a in cond.arguments]
        for pe in obs.predicate_estimates:
            if pe.predicate == cond.predicate and pe.args == args:
                return Estimate(pe.value, pe.known and pe.value is not None, pe.confidence, pe.estimator)
        return UNKNOWN

    @staticmethod
    def compare(cond: Condition, est: Estimate) -> bool | None:
        """True/False when known; None when unknown (unknown is not false)."""
        if not est.known or est.value is None:
            return None
        v = est.value
        if cond.comparison == "eq":
            return v == cond.value
        if cond.comparison == "le":
            return float(v) <= float(cond.value)
        if cond.comparison == "ge":
            return float(v) >= float(cond.value)
        lo, hi = cond.value
        return lo <= float(v) <= hi

    def _preconditions(self, event_id: str, obs) -> tuple[bool, str | None, str | None]:
        for c in self.compiled.event(event_id).preconditions:
            r = self.compare(c, self.evaluate(c, obs))
            if r is None:
                return False, "precondition_unknown", c.predicate
            if not r:
                return False, "precondition_false", c.predicate
        return True, None, None

    # ---------------------------------------------------------------- requests
    def request_event(self, event_id: str, expected_version: int, *, source: str = "user") -> ExecutionDecision:
        if expected_version != self.graph_version:
            return self._reject(event_id, "graph_version_conflict",
                                f"expected {expected_version}, current {self.graph_version}")
        if event_id not in self.instances:
            return ExecutionDecision(False, event_id, "unknown_event", None, self.graph_version, self.runtime_version)
        inst = self.instances[event_id]
        if inst.status in ("active", "succeeded"):
            return self._reject(event_id, f"already_{inst.status}")
        if inst.status in ("cancelled",):
            return self._reject(event_id, "cancelled")
        if inst.status == "failed":
            return self._reject(event_id, "failed_no_attempts_left")
        for check in (self._prereqs, self._binding_ok, self._resources_ok):
            ok, code, detail = check(event_id)
            if not ok:
                return self._reject(event_id, code, detail)
        ok, code, detail = self._preconditions(event_id, self._last_obs)
        if not ok:
            return self._reject(event_id, code, detail)
        inst.requested = True
        inst.activated_at = self.clock()
        inst.persistence_start = {}
        self._set(inst, "active", f"requested_by_{source}")
        return ExecutionDecision(True, event_id, None, None, self.graph_version, self.runtime_version)

    def cancel_event(self, event_id: str, reason: str = "user_cancel"):
        inst = self.instances[event_id]
        if inst.status not in ("succeeded", "failed", "cancelled"):
            self._set(inst, "cancelled", reason)
            self._invalidate_outputs(event_id, inst.attempt, "cancelled")

    # ---------------------------------------------------------------- tick
    def tick(self, observation: PolicyObservation | None) -> TaskContext:
        v0 = self.runtime_version
        self._last_obs = observation
        now = self.clock()
        for eid in self.compiled.topo_order:
            inst = self.instances[eid]
            if inst.status != "active":
                continue
            ev = self.compiled.event(eid)
            # maintained prerequisites must stay active
            for d in ev.requires_active:
                if self.instances[d].status != "active" and not (self.instances[d].status == "succeeded"):
                    self._fail(eid, "maintained_support_lost")
                    break
            if inst.status != "active":
                continue
            if inst.activated_at is not None and now - inst.activated_at > ev.timeout_seconds:
                self._fail(eid, "timeout")
                continue
            violated = None
            for c in ev.invariants:
                r = self.compare(c, self.evaluate(c, observation))
                if r is False:
                    violated = c.predicate
                    break
            if violated:
                self._fail(eid, f"invariant_violated:{violated}")
                continue
            # maintained outputs while active and valid
            for o in ev.produces:
                if o.availability == "while_active_and_valid":
                    self._publish(eid, o.name, o.type, observation, maintained=True)
            done = True
            for i, c in enumerate(ev.completion):
                r = self.compare(c, self.evaluate(c, observation))
                key = i
                if r:
                    inst.persistence_start.setdefault(key, now)
                    if now - inst.persistence_start[key] < c.persistence_seconds:
                        done = False
                else:
                    inst.persistence_start.pop(key, None)
                    done = False
            if done:
                ok = True
                for o in ev.produces:
                    if o.availability == "on_success":
                        ok = self._publish(eid, o.name, o.type, observation, maintained=False) and ok
                if ok:
                    self._set(inst, "succeeded", None)
                    inst.finished_at = now
                    for o in ev.produces:
                        if o.availability == "while_active_and_valid":
                            self._invalidate_outputs(eid, inst.attempt, "producer_finished", only=o.name)
                else:
                    inst.reason = "output_unavailable"
        self._refresh_ready()
        if self.auto_advance:
            self._advance()
        return TaskContext(self.graph_version, self.runtime_version,
                           {k: v.status for k, v in self.instances.items()},
                           [k for k, v in self.instances.items() if v.status == "active"],
                           [k for k, v in self.instances.items() if v.status == "ready"],
                           [r.key() for r in self.receipts.all() if r.valid], self.runtime_version != v0)

    def _advance(self):
        order = sorted(self.compiled.topo_order, key=lambda e: -self.store.priorities.get(e, 0))
        for eid in order:
            if self.instances[eid].status == "ready":
                self.request_event(eid, self.graph_version, source="auto")

    def _refresh_ready(self):
        for eid in self.compiled.topo_order:
            inst = self.instances[eid]
            if inst.status in ("pending", "ready", "blocked"):
                ok, code, _ = self._prereqs(eid)
                if ok:
                    ok, code, _ = self._binding_ok(eid)
                new = "ready" if ok else "pending"
                if new != inst.status or (not ok and inst.reason != code):
                    self._set(inst, new, None if ok else code)

    def _publish(self, eid, name, typ, obs, maintained: bool) -> bool:
        if self.output_provider is None:
            return False
        inst = self.instances[eid]
        value = self.output_provider(eid, name, typ, obs)
        latest = self.receipts.latest(eid, inst.attempt, name)
        if value is None:
            if maintained and latest and latest.valid:
                self.receipts.invalidate(eid, inst.attempt, name, "estimate_invalid")
                self.runtime_version += 1
            return False
        if maintained and latest and latest.valid and latest.value == value:
            return True
        if latest and latest.valid:
            self.receipts.invalidate(eid, inst.attempt, name, "superseded")
        cov = value.pop("covariance_diag", None) if isinstance(value, dict) else None
        self.receipts.put(Receipt(eid, inst.attempt, name, typ, (latest.version + 1) if latest else 0, value,
                                  participants=self.actor_of(eid), created_at=self.clock(),
                                  source_observation_id=obs.observation_id if obs else None,
                                  covariance_diag=cov, graph_version=self.graph_version))
        self.runtime_version += 1
        return True

    def _invalidate_outputs(self, eid, attempt, reason, only: str | None = None):
        for o in self.compiled.event(eid).produces:
            if only is None or o.name == only:
                self.receipts.invalidate(eid, attempt, o.name, reason)

    def _fail(self, eid: str, reason: str):
        inst = self.instances[eid]
        ev = self.compiled.event(eid)
        self._set(inst, "failed", reason)
        inst.finished_at = self.clock()
        self._invalidate_outputs(eid, inst.attempt, "producer_failed")
        self.attempt_log.append(dict(event=eid, attempt=inst.attempt, reason=reason, t=self.clock()))
        self.rejections.setdefault(eid, []).append(dict(t=self.clock(), reason=reason, attempt=inst.attempt,
                                                        graph_version=self.graph_version, detail="attempt_failed"))
        # dependents that were active and need this event active/its outputs fail too
        for other in self.compiled.definition.events:
            if eid in other.requires_active and self.instances[other.id].status == "active":
                self._fail(other.id, "maintained_support_lost")
        if ev.recovery.on_failure == "cancel_dependents":
            for other in self.compiled.definition.events:
                if eid in other.requires_completed and self.instances[other.id].status in ("pending", "ready"):
                    self._set(self.instances[other.id], "cancelled", f"dependency_failed:{eid}")
            return
        if ev.recovery.on_failure != "fail" and inst.attempt + 1 < ev.recovery.max_attempts:
            old = inst.attempt
            fresh = Instance(eid, attempt=old + 1)
            self.instances[eid] = fresh
            self.transitions.append(dict(t=self.clock(), event=eid, attempt=old + 1, frm="failed", to="pending",
                                         reason=f"retry_after:{reason}", graph_version=self.graph_version))
            self.runtime_version += 1
            # explicit, recorded rebinding of downstream consumers to the fresh attempt
            if any(o for o in ev.produces) and self._has_consumers(eid, old):
                self.store.apply_edit([{"op": "rebind_output", "event_id": eid, "from_attempt": old,
                                        "to_attempt": old + 1}],
                                      expected_version=self.graph_version,
                                      request_id=f"recovery:{eid}:{old}->{old + 1}:{self.runtime_version}",
                                      provenance="recovery")

    def _has_consumers(self, eid, attempt) -> bool:
        if any(i.source_event == eid and i.source_attempt == attempt for i in self.compiled.incidences):
            return True
        # outputs consumed only by guard conditions or frame bindings (e.g. receipt_valid(align#0.aligned))
        for e in self.compiled.definition.events:
            fb = e.frame_binding
            if isinstance(fb, OutputBinding) and fb.event_id == eid and fb.attempt == attempt:
                return True
            for c in e.preconditions + e.invariants + e.desired_effects + e.completion:
                if any(isinstance(a, OutputBinding) and a.event_id == eid and a.attempt == attempt
                       for a in c.arguments):
                    return True
        return False

    # ---------------------------------------------------------------- edits
    def apply_edit(self, operations: list[dict], expected_version: int, request_id: str,
                   provenance: str = "user") -> EditReceipt:
        return self.store.apply_edit(operations, expected_version=expected_version, request_id=request_id,
                                     provenance=provenance)

    def _reconcile(self, rec: EditReceipt, compiled: CompiledTask):
        ids = {e.id for e in compiled.definition.events}
        for eid in list(self.instances):
            if eid not in ids:
                inst = self.instances.pop(eid)
                self.transitions.append(dict(t=self.clock(), event=eid, attempt=inst.attempt, frm=inst.status,
                                             to="removed", reason="graph_edit", graph_version=rec.graph_version))
        for eid in ids:
            if eid not in self.instances:
                self.instances[eid] = Instance(eid)
        # active events whose bindings/resources became invalid are cancelled, not silently kept
        for eid, inst in self.instances.items():
            if inst.status == "active":
                ok, code, detail = self._binding_ok(eid)
                if ok:
                    ok, code, detail = self._resources_ok(eid)
                if not ok:
                    self._set(inst, "cancelled", f"edit_invalidated:{code}")
        self.runtime_version += 1
        if self.on_invalidate:
            self.on_invalidate(f"graph_edit:{rec.request_id}")
        self._refresh_ready()

    # ---------------------------------------------------------------- snapshot
    def snapshot(self) -> dict:
        return copy.deepcopy(dict(doc=self.store.document(), version=self.store.version,
                                  priorities=self.store.priorities,
                                  instances={k: asdict(v) for k, v in self.instances.items()},
                                  receipts=[asdict(r) for r in self.receipts.all()],
                                  runtime_version=self.runtime_version, rejections=self.rejections,
                                  attempt_log=self.attempt_log,
                                  history=[asdict(h) for h in self.store.history]))

    def restore(self, snap: dict):
        snap = copy.deepcopy(snap)
        store = GraphStore(snap["doc"])
        store.version = snap["version"]
        store.priorities = snap["priorities"]
        store.history = [EditReceipt(**h) for h in snap["history"]]
        store._by_request = {h.request_id: h for h in store.history}
        store._listeners = []
        store.on_commit(self._reconcile)
        self.store = store
        self.instances = {k: Instance(**v) for k, v in snap["instances"].items()}
        self.receipts = ReceiptStore()
        for r in snap["receipts"]:
            hist = self.receipts._by_key.setdefault((r["event_id"], r["attempt"], r["output_name"]), [])
            hist.append(Receipt(**r))
        self.runtime_version = snap["runtime_version"]
        self.rejections = snap["rejections"]
        self.attempt_log = snap["attempt_log"]
