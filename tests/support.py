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


def make_observation(estimates=(), t=0.0, oid="obs"):
    """Public observation carrying declared-estimator predicate estimates only."""
    import numpy as np
    from rrp.contracts.observation import PolicyObservation, NodeState, PredicateEstimate
    ns = NodeState(joint_addresses=["r/0"], qpos=np.zeros(1), qvel=np.zeros(1), qpos_mask=np.ones(1, bool),
                   timestamp=t)
    pes = [PredicateEstimate(predicate=p, args=list(a), value=v, known=k, confidence=1.0 if k else 0.0,
                             estimator="fixture_estimator", timestamp=t) for (p, a, v, k) in estimates]
    return PolicyObservation(observation_id=oid, sensor_time=t, robot_spec_hash="fixture",
                             sensor_images=[], measured_node_state=ns, declared_sensor_channels=[],
                             predicate_estimates=pes)


def observation_with_no_grasp():
    return make_observation([("held_by", ("peg", "right"), False, True)])
