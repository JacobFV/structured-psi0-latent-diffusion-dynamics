"""Compile a supplied TaskDefinition into validated incidence records and typed edges.

Semantic validation beyond JSON shape: unique ids, declared entities, role type/arity and
ordinal contiguity, dependency references, requires_completed acyclicity, output
provenance (consumer must be ordered after / concurrent with the producer as the output's
availability requires), and exclusive-control conflicts between events that can be
simultaneously active.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rrp.contracts.errors import ContractError
from rrp.contracts.task import TaskDefinition, EventDef, OutputBinding, EntityBinding

ACTOR_TYPES = {"manipulator", "sensor", "body"}
ROLE_ENTITY_TYPES = {
    "actor": ACTOR_TYPES, "cooperating_actor": ACTOR_TYPES,
    "patient": {"object", "feature", "manipulator", "body"},
    "instrument": {"object", "manipulator"},
    "target": {"object", "feature", "body"}, "source": {"object", "feature", "manipulator", "body"},
    "destination": {"object", "feature", "body", "manipulator"}, "reference": {"object", "feature", "body", "sensor"},
    "support": {"object", "feature", "manipulator", "body"},
}


@dataclass(frozen=True)
class Incidence:
    """Hyperedge incidence record: event --(role, ordinal)--> entity or output reference."""
    event_id: str
    role: str
    ordinal: int
    kind: str                 # entity | event_output
    entity_id: str | None
    entity_version: int | None
    source_event: str | None
    source_attempt: int | None
    output_name: str | None


@dataclass(frozen=True)
class TypedEdge:
    src: str
    dst: str
    type: str                 # enables | maintained_during | output_to | conflicts


@dataclass
class CompiledTask:
    definition: TaskDefinition
    incidences: list[Incidence]
    edges: list[TypedEdge]
    topo_order: list[str]
    entity_types: dict[str, str]
    outputs: dict[tuple[str, str], str] = field(default_factory=dict)       # (event, name) -> type
    availability: dict[tuple[str, str], str] = field(default_factory=dict)

    def event(self, eid: str) -> EventDef:
        return self.definition.event(eid)

    def retrieval_edges(self) -> list[tuple[str, str, str]]:
        """Attention routing orientation: query row = consumer, key column = prerequisite.
        semantic e_i -> e_j (i enables j) becomes retrieval (j reads i)."""
        return [(e.dst, e.src, e.type) for e in self.edges if e.type in ("enables", "maintained_during", "output_to")]


def _ancestors(eid, req):
    seen, stack = set(), list(req.get(eid, []))
    while stack:
        x = stack.pop()
        if x not in seen:
            seen.add(x)
            stack.extend(req.get(x, []))
    return seen


def compile_task(payload) -> CompiledTask:
    task = payload if isinstance(payload, TaskDefinition) else TaskDefinition.model_validate(payload)
    ids = [e.id for e in task.events]
    if len(set(ids)) != len(ids):
        raise ContractError("duplicate event ids", code="duplicate_event")
    decl = {}
    for d in task.entity_declarations:
        if d.id in decl:
            raise ContractError(f"duplicate entity {d.id}", code="duplicate_entity")
        decl[d.id] = d.type
    events = {e.id: e for e in task.events}
    for s in task.success_events:
        if s not in events:
            raise ContractError(f"success event {s} undefined", code="unknown_event")

    outputs, avail = {}, {}
    for e in task.events:
        names = [o.name for o in e.produces]
        if len(set(names)) != len(names):
            raise ContractError(f"{e.id}: duplicate output names", code="duplicate_output")
        for o in e.produces:
            outputs[(e.id, o.name)] = o.type
            avail[(e.id, o.name)] = o.availability

    req_c = {e.id: list(e.requires_completed) for e in task.events}
    for e in task.events:
        for dep in e.requires_completed + e.requires_active:
            if dep not in events:
                raise ContractError(f"{e.id} depends on unknown event {dep}", code="unknown_event")
            if dep == e.id:
                raise ContractError(f"{e.id} depends on itself", code="dependency_cycle")
        if set(e.requires_completed) & set(e.requires_active):
            raise ContractError(f"{e.id}: event both required completed and active", code="dependency_conflict")

    # topological order over requires_completed (must be a DAG within an active plan)
    order, temp, perm = [], set(), set()

    def visit(n, path):
        if n in perm:
            return
        if n in temp:
            raise ContractError(f"prerequisite cycle: {' -> '.join(path + [n])}", code="dependency_cycle",
                                cycle=path + [n])
        temp.add(n)
        for m in req_c[n]:
            visit(m, path + [n])
        temp.discard(n)
        perm.add(n)
        order.append(n)

    for e in task.events:
        visit(e.id, [])

    incidences, edges = [], []
    for e in task.events:
        seen_slots = {}
        for slot in e.roles:
            key = (slot.role, slot.ordinal)
            if key in seen_slots:
                raise ContractError(f"{e.id}: duplicate role slot {key}", code="role_arity")
            seen_slots[key] = slot
            b = slot.binding
            if isinstance(b, EntityBinding):
                et = decl.get(b.entity.id)
                if et is None:
                    raise ContractError(f"{e.id}: undeclared entity {b.entity.id}", code="unknown_entity")
                allowed = ROLE_ENTITY_TYPES[slot.role]
                if et not in allowed:
                    raise ContractError(f"{e.id}: role {slot.role} cannot bind {et} {b.entity.id}",
                                        code="role_type")
                incidences.append(Incidence(e.id, slot.role, slot.ordinal, "entity", b.entity.id,
                                            b.entity.version, None, None, None))
            else:
                _check_output_binding(e, b, events, outputs, avail, req_c)
                incidences.append(Incidence(e.id, slot.role, slot.ordinal, "event_output", None, None,
                                            b.event_id, b.attempt, b.output_name))
                edges.append(TypedEdge(b.event_id, e.id, "output_to"))
        by_role = {}
        for (r, o) in seen_slots:
            by_role.setdefault(r, []).append(o)
        for r, ords in by_role.items():
            if sorted(ords) != list(range(len(ords))):
                raise ContractError(f"{e.id}: role {r} ordinals must be contiguous from 0", code="role_arity")
        if not any(s.role == "actor" for s in e.roles):
            raise ContractError(f"{e.id}: event needs an actor", code="role_arity")
        for c in e.preconditions + e.invariants + e.desired_effects + e.completion:
            for a in c.arguments:
                if isinstance(a, OutputBinding):
                    if a.event_id not in events:
                        raise ContractError(f"{e.id}: condition references unknown event {a.event_id}",
                                            code="unknown_event")
                    if (a.event_id, a.output_name) not in outputs:
                        raise ContractError(f"{e.id}: condition references undeclared output", code="unknown_output")
                elif a.entity.id not in decl:
                    raise ContractError(f"{e.id}: condition references undeclared entity {a.entity.id}",
                                        code="unknown_entity")
        for r in e.resources:
            if r.entity.id not in decl:
                raise ContractError(f"{e.id}: resource on undeclared entity", code="unknown_entity")
        for d in e.requires_completed:
            edges.append(TypedEdge(d, e.id, "enables"))
        for d in e.requires_active:
            edges.append(TypedEdge(d, e.id, "maintained_during"))

    # exclusive control conflicts between events that must be simultaneously active
    for e in task.events:
        mine = {r.entity.id for r in e.resources if r.mode == "exclusive_control"}
        for d in e.requires_active:
            other = {r.entity.id for r in events[d].resources if r.mode == "exclusive_control"}
            clash = mine & other
            if clash:
                raise ContractError(f"{e.id} and maintained {d} both need exclusive control of {sorted(clash)}",
                                    code="resource_conflict")
    return CompiledTask(task, incidences, edges, order, decl, outputs, avail)


def _check_output_binding(consumer, b: OutputBinding, events, outputs, avail, req_c):
    if b.event_id not in events:
        raise ContractError(f"{consumer.id}: binds output of unknown event {b.event_id}", code="unknown_event")
    if (b.event_id, b.output_name) not in outputs:
        raise ContractError(f"{consumer.id}: {b.event_id} does not produce {b.output_name}", code="unknown_output")
    if b.attempt >= events[b.event_id].recovery.max_attempts:
        raise ContractError(f"{consumer.id}: attempt {b.attempt} exceeds producer max_attempts",
                            code="unknown_output")
    a = avail[(b.event_id, b.output_name)]
    if a == "on_success" and b.event_id not in _ancestors(consumer.id, req_c):
        raise ContractError(f"{consumer.id} consumes on_success output of {b.event_id} without requiring it "
                            f"completed", code="output_dependency_not_ordered")
    if a == "while_active_and_valid" and b.event_id not in consumer.requires_active:
        raise ContractError(f"{consumer.id} consumes maintained output of {b.event_id} without requiring it active",
                            code="output_dependency_not_ordered")
