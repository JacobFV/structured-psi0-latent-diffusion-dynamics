import copy
import pytest
from rrp.tasks.interventions import GraphStore, EditRejected
from rrp.contracts.errors import VersionConflict
from tests.support import supplied_task_payload


def test_stale_edit_cannot_partially_mutate_graph():
    store = GraphStore(supplied_task_payload())
    before = store.content_hash()
    with pytest.raises(VersionConflict):
        store.apply_edit([], expected_version=0, request_id="r1")
    assert store.content_hash() == before


def test_failed_edit_leaves_graph_unchanged_and_cycle_rejected():
    store = GraphStore(supplied_task_payload())
    before = store.content_hash()
    ops = [{"op": "set_priority", "event_id": "locate", "priority": 5},
           {"op": "add_dependency", "src": "insert", "dst": "locate", "mode": "completed"}]
    with pytest.raises(EditRejected) as e:
        store.apply_edit(ops, expected_version=1, request_id="r2")
    assert e.value.code == "dependency_cycle"
    assert store.content_hash() == before and store.version == 1 and store.priorities == {}


def test_idempotent_request_id_and_version_increment():
    store = GraphStore(supplied_task_payload())
    seen = []
    store.on_commit(lambda rec, c: seen.append(rec.graph_version))
    r1 = store.apply_edit([{"op": "set_priority", "event_id": "acquire", "priority": 3}], expected_version=1,
                          request_id="same")
    r2 = store.apply_edit([{"op": "set_priority", "event_id": "acquire", "priority": 3}], expected_version=1,
                          request_id="same")
    assert r1 is r2 and store.version == 2 and seen == [2]


def test_unknown_operation_and_bad_reference_rejected():
    store = GraphStore(supplied_task_payload())
    with pytest.raises(EditRejected):
        store.apply_edit([{"op": "execute_python", "code": "rm -rf"}], expected_version=1, request_id="a")
    with pytest.raises(EditRejected):
        store.apply_edit([{"op": "remove_event", "event_id": "locate"}], expected_version=1, request_id="b")
    assert store.version == 1
