"""Typed request/message schemas shared with the UI (exported to TypeScript)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MESSAGE_KINDS = ("session_snapshot", "state_update", "graph_committed", "command_rejected", "command_accepted",
                 "probe_result", "resource_update", "mode_changed", "intervention", "event_request",
                 "event_cancelled", "debug_override", "session_reset", "frame")


def public_message_kinds() -> tuple[str, ...]:
    return MESSAGE_KINDS


class _Req(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateSession(_Req):
    robot: str = Field(max_length=64, pattern=r"^[a-z0-9_]+$")
    task: str = Field(default="pick_place", max_length=64, pattern=r"^[a-z0-9_]+$")
    seed: int = Field(default=0, ge=0, le=2 ** 31)


class Command(_Req):
    type: Literal["step", "run", "pause", "reset", "joint_target", "ee_target", "teleport", "request_event",
                  "cancel_event", "set_mode", "debug_force_success", "replay"]
    n: int = Field(default=1, ge=1, le=2000)
    group: str | None = Field(default=None, max_length=32)
    values: list[float] | None = Field(default=None, max_length=64)
    pos: list[float] | None = Field(default=None, min_length=3, max_length=3)
    yaw: float = 0.0
    body: str | None = Field(default=None, max_length=64)
    event_id: str | None = Field(default=None, max_length=64)
    expected_version: int | None = None
    mode: str | None = Field(default=None, max_length=32)
    policy: str | None = Field(default=None, max_length=64)
    seed: int | None = None
    confirm: bool = False
    request_id: str | None = Field(default=None, max_length=80)


class GraphEditRequest(_Req):
    expected_version: int
    request_id: str = Field(min_length=1, max_length=80)
    operations: list[dict] = Field(min_length=1, max_length=50)


class LayoutRequest(_Req):
    positions: dict[str, list[float]] = Field(max_length=500)


class ProbeRequest(_Req):
    query: Literal["visible", "looking_at", "focused_on", "acting_on", "held_by", "manipulator_tasks",
                   "relative_pose", "distance", "active_frame", "contact_mode", "desired_delta", "feasibility",
                   "uncertainty", "object_qa"]
    subject: str | None = Field(default=None, max_length=64)
    object: str | None = Field(default=None, max_length=64)
    question: str | None = Field(default=None, max_length=200)
