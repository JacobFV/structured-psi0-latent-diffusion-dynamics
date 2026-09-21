"""Versioned, transactional graph edits with idempotency and provenance.

apply_edit validates on a deep copy and commits atomically: a stale expected_version or
any invalid operation leaves the graph byte-identical. Commit listeners invalidate
context caches and drop unsent action chunks. Layout (node screen positions) is NOT part
of the graph and never goes through this path.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from rrp.contracts.base import content_hash
from rrp.contracts.errors import ContractError, VersionConflict, RRPError
from rrp.contracts.task import TaskDefinition
from .compiler import compile_task, CompiledTask


class EditRejected(RRPError):
    code = "edit_rejected"


@dataclass
class EditReceipt:
    request_id: str
    accepted: bool
    graph_version: int
    content_hash: str
    operations: list
    provenance: str
    t: float = field(default_factory=time.time)
    reason: str | None = None


OPS = {"add_event", "remove_event", "add_dependency", "remove_dependency", "bind_role", "unbind_role",
       "set_priority", "rebind_output", "set_success_events", "add_entity", "replace_event"}


class GraphStore:
    def __init__(self, payload: dict | TaskDefinition):
        task = payload if isinstance(payload, TaskDefinition) else TaskDefinition.model_validate(payload)
        self._compiled: CompiledTask = compile_task(task)
        self._doc = task.model_dump(mode="json")
        self.version = self._doc["graph_version"]
        self.priorities: dict[str, int] = {}
        self.history: list[EditReceipt] = []
        self._by_request: dict[str, EditReceipt] = {}
        self._listeners: list[Callable[[EditReceipt, CompiledTask], None]] = []

    @property
    def compiled(self) -> CompiledTask:
        return self._compiled

    def content_hash(self) -> str:
        return content_hash({"doc": self._doc, "priorities": self.priorities})

    def document(self) -> dict:
        return copy.deepcopy(self._doc)

    def on_commit(self, fn):
        self._listeners.append(fn)

    def apply_edit(self, operations: list[dict], *, expected_version: int, request_id: str,
                   provenance: str = "user") -> EditReceipt:
        if request_id in self._by_request:
            return self._by_request[request_id]           # idempotent replay
        if expected_version != self.version:
            raise VersionConflict(f"expected graph version {expected_version}, current {self.version}",
                                  expected=expected_version, current=self.version)
        if not operations:
            raise EditRejected("empty edit", code="empty_edit")
        doc = copy.deepcopy(self._doc)
        prio = dict(self.priorities)
        try:
            for op in operations:
                _apply_op(doc, prio, op)
            doc["graph_version"] = self.version + 1
            compiled = compile_task(doc)
        except (ContractError, KeyError, ValueError, TypeError, IndexError) as e:
            code = getattr(e, "code", "invalid_operation")
            raise EditRejected(f"edit rejected: {e}", code=code if isinstance(code, str) else "invalid_operation",
                               request_id=request_id) from e
        # commit (atomic swap)
        self._doc, self.priorities, self._compiled = doc, prio, compiled
        self.version += 1
        rec = EditReceipt(request_id, True, self.version, self.content_hash(), copy.deepcopy(operations), provenance)
        self.history.append(rec)
        self._by_request[request_id] = rec
        for fn in self._listeners:
            fn(rec, compiled)
        return rec


def _event(doc, eid):
    for e in doc["events"]:
        if e["id"] == eid:
            return e
    raise KeyError(f"unknown event {eid}")


def _apply_op(doc: dict, prio: dict, op: dict[str, Any]):
    kind = op.get("op")
    if kind not in OPS:
        raise ValueError(f"unknown operation {kind!r}")
    if kind == "add_event":
        ev = copy.deepcopy(op["event"])
        if any(e["id"] == ev["id"] for e in doc["events"]):
            raise ValueError(f"event {ev['id']} exists")
        doc["events"].append(ev)
    elif kind == "replace_event":
        ev = copy.deepcopy(op["event"])
        idx = [i for i, e in enumerate(doc["events"]) if e["id"] == ev["id"]]
        if not idx:
            raise KeyError(ev["id"])
        doc["events"][idx[0]] = ev
    elif kind == "remove_event":
        eid = op["event_id"]
        _event(doc, eid)
        doc["events"] = [e for e in doc["events"] if e["id"] != eid]
        for e in doc["events"]:
            if eid in e["requires_completed"] or eid in e["requires_active"]:
                raise ValueError(f"cannot remove {eid}: {e['id']} depends on it (remove dependency first)")
        doc["success_events"] = [s for s in doc["success_events"] if s != eid]
        if not doc["success_events"]:
            raise ValueError("graph would have no success events")
    elif kind == "add_dependency":
        e = _event(doc, op["dst"])
        _event(doc, op["src"])
        key = {"completed": "requires_completed", "active": "requires_active"}[op["mode"]]
        if op["src"] in e[key]:
            raise ValueError("dependency exists")
        e[key].append(op["src"])
    elif kind == "remove_dependency":
        e = _event(doc, op["dst"])
        key = {"completed": "requires_completed", "active": "requires_active"}[op["mode"]]
        e[key].remove(op["src"])
    elif kind == "bind_role":
        e = _event(doc, op["event_id"])
        slot = {"role": op["role"], "ordinal": int(op["ordinal"]), "binding": copy.deepcopy(op["binding"])}
        e["roles"] = [r for r in e["roles"] if not (r["role"] == slot["role"] and r["ordinal"] == slot["ordinal"])]
        e["roles"].append(slot)
        # resource claims follow actor rebinding
        if slot["role"] == "actor" and slot["binding"]["kind"] == "entity" and op.get("move_resources", True):
            for r in e["resources"]:
                if r["mode"] == "exclusive_control":
                    r["entity"] = copy.deepcopy(slot["binding"]["entity"])
    elif kind == "unbind_role":
        e = _event(doc, op["event_id"])
        before = len(e["roles"])
        e["roles"] = [r for r in e["roles"] if not (r["role"] == op["role"] and r["ordinal"] == int(op["ordinal"]))]
        if len(e["roles"]) == before:
            raise KeyError("no such role slot")
    elif kind == "set_priority":
        _event(doc, op["event_id"])
        prio[op["event_id"]] = int(op["priority"])
    elif kind == "rebind_output":
        # explicit rebinding of downstream references to a fresh producer attempt
        src, old, new = op["event_id"], int(op["from_attempt"]), int(op["to_attempt"])
        for e in doc["events"]:
            for r in e["roles"]:
                b = r["binding"]
                if b["kind"] == "event_output" and b["event_id"] == src and b["attempt"] == old:
                    b["attempt"] = new
            for cond_key in ("preconditions", "invariants", "desired_effects", "completion"):
                for c in e[cond_key]:
                    for a in c["arguments"]:
                        if a["kind"] == "event_output" and a["event_id"] == src and a["attempt"] == old:
                            a["attempt"] = new
    elif kind == "set_success_events":
        doc["success_events"] = list(op["events"])
    elif kind == "add_entity":
        if any(d["id"] == op["entity"]["id"] for d in doc["entity_declarations"]):
            raise ValueError("entity exists")
        doc["entity_declarations"].append(copy.deepcopy(op["entity"]))
