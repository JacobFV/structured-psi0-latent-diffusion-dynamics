"""Test fixture helpers. These never substitute for the implementation under test."""
from __future__ import annotations

import json
from pathlib import Path

HANDOFF = Path(__file__).resolve().parents[1] / "docs" / "handoff"


def fake_enforcement_backend():
    from rrp.ops.cgroup import FakeEnforcementBackend
    return FakeEnforcementBackend()


def supplied_task_payload() -> dict:
    """Exact copy of the package example (not an invented simplified graph)."""
    return json.loads((HANDOFF / "examples" / "support_and_insert.task.json").read_text())
