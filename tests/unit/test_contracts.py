import json
import numpy as np
import pytest
from pydantic import ValidationError

from rrp.contracts.refs import EntityRef
from rrp.contracts.task import TaskDefinition
from rrp.contracts.observation import PolicyObservation, PrivilegedTruth, NodeState
from rrp.contracts.action import ActionChunk, GroupCommand
from rrp.contracts.channels import serialize_public, deserialize_observation, PrivateBus
from rrp.contracts.errors import PrivilegedLeakError
from tests.support import supplied_task_payload


def test_example_and_identity_validation():
    graph = TaskDefinition.model_validate(supplied_task_payload())
    assert graph.task_id == "support_and_insert"
    assert EntityRef(id="peg", version=0).version == 0
    with pytest.raises(ValueError):
        EntityRef(id="peg", version=-1)


def test_task_roundtrip_and_unknown_field_rejection():
    p = supplied_task_payload()
    g = TaskDefinition.model_validate(p)
    assert TaskDefinition.model_validate_json(g.model_dump_json()) == g
    p2 = json.loads(json.dumps(p))
    p2["events"][0]["secret_truth"] = 1
    with pytest.raises(ValidationError):
        TaskDefinition.model_validate(p2)


def test_example_also_validates_against_supplied_json_schema():
    import jsonschema
    from tests.support import HANDOFF
    schema = json.loads((HANDOFF / "contracts" / "task_graph.schema.json").read_text())
    jsonschema.validate(supplied_task_payload(), schema)


def test_output_binding_preserves_event_attempt_and_name():
    g = TaskDefinition.model_validate(supplied_task_payload())
    align = g.event("align")
    refs = [r.binding for r in align.roles if r.binding.kind == "event_output"]
    assert {(b.event_id, b.attempt, b.output_name) for b in refs} == {("locate", 0, "hole_frame"),
                                                                        ("support", 0, "anchor")}


def test_interval_conditions_validated():
    p = supplied_task_payload()
    ev = p["events"][0]
    ev["completion"] = [dict(predicate="x", arguments=[], comparison="inside_interval", value=[2.0, 1.0],
                             source="observation_estimate")]
    with pytest.raises(ValidationError):
        TaskDefinition.model_validate(p)


def _obs(**kw):
    ns = NodeState(joint_addresses=["r/0"], qpos=np.zeros(1), qvel=np.zeros(1), qpos_mask=np.ones(1, bool),
                   timestamp=0.0)
    d = dict(observation_id="o1", sensor_time=0.0, robot_spec_hash="h", sensor_images=[],
             measured_node_state=ns, declared_sensor_channels=[])
    d.update(kw)
    return PolicyObservation(**d)


def test_nonfinite_state_rejected():
    with pytest.raises(ValidationError):
        NodeState(joint_addresses=["a"], qpos=np.array([np.nan]), qvel=np.zeros(1), qpos_mask=np.ones(1),
                  timestamp=0.0)


def test_public_transport_roundtrip_and_privileged_refusal(tmp_path):
    o = _obs()
    back = deserialize_observation(serialize_public(o))
    assert back.observation_id == "o1" and np.array_equal(back.measured_node_state.qpos, o.measured_node_state.qpos)
    t = PrivilegedTruth(observation_id="o1", sim_time=0.0, object_poses={"peg": [0] * 7},
                        object_entity_map={"peg": "peg_body"}, contacts=[], held_by={}, predicates={},
                        event_completion_truth={"insert": True})
    with pytest.raises(PrivilegedLeakError):
        serialize_public(t)
    # hostile payload smuggles privileged truth into a public observation
    d = json.loads(serialize_public(o))
    d["belief_history"] = [{"object_poses": {"peg": [1, 2, 3]}}]
    with pytest.raises(PrivilegedLeakError):
        deserialize_observation(json.dumps(d))
    d = json.loads(serialize_public(o))
    d["object_poses"] = {}
    with pytest.raises((PrivilegedLeakError, ValidationError)):
        deserialize_observation(json.dumps(d))
    with pytest.raises(PrivilegedLeakError):
        PrivateBus(tmp_path / "episode.public.jsonl")
    PrivateBus(tmp_path / "episode.private.jsonl").write(t)


def test_policy_observation_type_has_no_privileged_fields():
    assert not (set(PolicyObservation.model_fields) & {"object_poses", "contacts", "reward", "labels",
                                                        "event_completion_truth"})


def test_action_chunk_shape_and_finiteness():
    g = GroupCommand(group="arm", values=np.zeros((4, 7)), mask=np.ones((4, 7), bool))
    ActionChunk(observation_id="o", graph_version=1, runtime_version=1, robot_spec_hash="h", controller_version="c",
                policy_version="p", codec_version=None, start_time=0.0, dt=0.05, horizon=4, command_groups=[g],
                sampling_seed=0, source="learned")
    with pytest.raises(ValidationError):
        ActionChunk(observation_id="o", graph_version=1, runtime_version=1, robot_spec_hash="h",
                    controller_version="c", policy_version="p", codec_version=None, start_time=0.0, dt=0.05,
                    horizon=5, command_groups=[g], sampling_seed=0, source="learned")
    with pytest.raises(ValidationError):
        GroupCommand(group="arm", values=np.full((4, 7), np.inf), mask=np.ones((4, 7), bool))
