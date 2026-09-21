"""Supplied event-hypergraph definition (mirrors docs/handoff/contracts/task_graph.schema.json).

Events are hyperedges realized as event nodes + ordered role-slot incidence records.
Semantic validation (referential integrity, cycles, role arity, output provenance,
controller-ownership overlap) lives in rrp.tasks.compiler; this module is the shape.
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, field_validator, model_validator

from .base import Strict
from .refs import EntityRef

RoleName = Literal["actor", "patient", "instrument", "target", "source", "destination", "reference",
                   "support", "cooperating_actor"]
EntityType = Literal["manipulator", "object", "feature", "sensor", "body"]
OutputType = Literal["frame_estimate", "contact_anchor", "alignment_receipt", "completion_receipt"]


class EntityBinding(Strict):
    kind: Literal["entity"]
    entity: EntityRef


class OutputBinding(Strict):
    """Binds a SPECIFIC event instance attempt's output (never 'latest value of a type')."""
    kind: Literal["event_output"]
    event_id: str = Field(min_length=1)
    attempt: int = Field(ge=0)
    output_name: str = Field(min_length=1)


Binding = Annotated[Union[EntityBinding, OutputBinding], Field(discriminator="kind")]


class Condition(Strict):
    predicate: str = Field(min_length=1)
    arguments: list[Binding]
    comparison: Literal["eq", "le", "ge", "inside_interval"]
    value: bool | float | str | list[float]
    source: Literal["observation_estimate", "validated_receipt", "declared_spec", "desired_effect"]
    persistence_seconds: float = Field(default=0.0, ge=0)

    @field_validator("value")
    @classmethod
    def _interval(cls, v):
        if isinstance(v, list) and len(v) != 2:
            raise ValueError("interval value must have exactly two numbers")
        return v

    @model_validator(mode="after")
    def _interval_matches(self):
        if self.comparison == "inside_interval" and not isinstance(self.value, list):
            raise ValueError("inside_interval requires a [lo, hi] value")
        if isinstance(self.value, list) and self.value[0] > self.value[1]:
            raise ValueError("interval lower bound exceeds upper bound")
        return self


class RoleSlot(Strict):
    role: RoleName
    ordinal: int = Field(ge=0)
    binding: Binding


class ResourceClaim(Strict):
    entity: EntityRef
    mode: Literal["exclusive_control", "shared_support", "observe"]


class OutputDecl(Strict):
    name: str
    type: OutputType
    availability: Literal["on_success", "while_active_and_valid"]


class Recovery(Strict):
    max_attempts: int = Field(ge=1)
    on_failure: Literal["fail", "reobserve_then_retry", "release_then_retry", "cancel_dependents"]


class EventDef(Strict):
    id: str = Field(min_length=1)
    operator: str = Field(min_length=1)
    roles: list[RoleSlot]
    requires_completed: list[str]
    requires_active: list[str]
    preconditions: list[Condition]
    invariants: list[Condition]
    desired_effects: list[Condition]
    completion: list[Condition]
    resources: list[ResourceClaim]
    produces: list[OutputDecl]
    frame_binding: Binding | None = None
    timeout_seconds: float = Field(gt=0)
    recovery: Recovery

    @field_validator("requires_completed", "requires_active")
    @classmethod
    def _unique(cls, v):
        if len(set(v)) != len(v):
            raise ValueError("dependency lists must be unique")
        return v


class EntityDecl(Strict):
    id: str
    type: EntityType
    descriptor: str


class TaskDefinition(Strict):
    schema_version: Literal["1.0"]
    task_id: str = Field(min_length=1)
    graph_version: int = Field(ge=0)
    entity_declarations: list[EntityDecl]
    events: list[EventDef] = Field(min_length=1)
    success_events: list[str] = Field(min_length=1)

    @field_validator("success_events")
    @classmethod
    def _unique(cls, v):
        if len(set(v)) != len(v):
            raise ValueError("success_events must be unique")
        return v

    def event(self, event_id: str) -> EventDef:
        for e in self.events:
            if e.id == event_id:
                return e
        raise KeyError(event_id)
