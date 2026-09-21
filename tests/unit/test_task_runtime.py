import copy
import pytest

from rrp.tasks.compiler import compile_task
from rrp.tasks.runtime import TaskRuntime
from rrp.contracts.errors import ContractError
from tests.support import supplied_task_payload, observation_with_no_grasp, make_observation


class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


def provider_factory(values):
    def provider(eid, name, typ, obs):
        v = values.get((eid, name))
        return copy.deepcopy(v) if v is not None else None
    return provider


def test_insert_cannot_skip_prerequisites():
    runtime = TaskRuntime(compile_task(supplied_task_payload()))
    runtime.tick(observation_with_no_grasp())
    decision = runtime.request_event("insert", expected_version=1)
    assert not decision.accepted
    assert decision.reason_code in {"prerequisite_unsatisfied", "binding_unavailable"}
    assert runtime.status("insert") != "succeeded"
    assert runtime.status("align") == "pending" and runtime.status("locate") == "ready"


def test_unknown_precondition_is_not_false():
    rt = TaskRuntime(supplied_task_payload())
    rt.tick(make_observation([]))
    d = rt.request_event("support", 1)
    assert not d.accepted and d.reason_code == "precondition_unknown"
    rt.tick(make_observation([("reachable", ("left", "fixture"), False, True)]))
    d = rt.request_event("support", 1)
    assert not d.accepted and d.reason_code == "precondition_false"
    rt.tick(make_observation([("reachable", ("left", "fixture"), True, True)]))
    assert rt.request_event("support", 1).accepted
    assert rt.status("support") == "active"


def test_completion_needs_known_estimate_held_for_persistence():
    clk = Clock()
    rt = TaskRuntime(supplied_task_payload(), clock=clk)
    rt.tick(make_observation([]))
    assert rt.request_event("acquire", 1).accepted
    rt.tick(make_observation([("held_by", ("peg", "right"), None, False)]))
    assert rt.status("acquire") == "active"
    rt.tick(make_observation([("held_by", ("peg", "right"), True, True)]))
    assert rt.status("acquire") == "active"          # persistence 0.1 s not yet met
    clk.t = 0.05
    rt.tick(make_observation([("held_by", ("peg", "right"), False, True)]))  # flicker resets persistence
    clk.t = 0.10
    rt.tick(make_observation([("held_by", ("peg", "right"), True, True)]))
    clk.t = 0.25
    rt.tick(make_observation([("held_by", ("peg", "right"), True, True)]))
    assert rt.status("acquire") == "succeeded"


def test_desired_effect_is_never_an_observed_fact():
    rt = TaskRuntime(supplied_task_payload())
    rt.tick(make_observation([("reachable", ("left", "fixture"), True, True)]))
    rt.request_event("support", 1)
    for _ in range(3):
        rt.tick(make_observation([("supported", ("fixture",), True, True)]))
    assert rt.status("support") == "active"   # completion needs insert receipt, not the desired effect


def _drive_to_align(rt, clk, anchor=True):
    rt.tick(make_observation([("reachable", ("left", "fixture"), True, True)]))
    for e in ("locate", "support", "acquire"):
        assert rt.request_event(e, rt.graph_version).accepted, e
    obs = [("frame_estimate_valid", ("hole",), True, True), ("held_by", ("peg", "right"), True, True),
           ("reachable", ("left", "fixture"), True, True)]
    rt.tick(make_observation(obs))
    clk.t += 0.2
    rt.tick(make_observation(obs))
    return obs


def test_functional_output_binding_and_maintained_anchor():
    clk = Clock()
    values = {("locate", "hole_frame"): {"pos": [0.5, 0.1, 0.2], "quat_wxyz": [1, 0, 0, 0]},
              ("support", "anchor"): {"pos": [0.4, -0.1, 0.1], "normal": [0, 0, 1]}}
    rt = TaskRuntime(supplied_task_payload(), clock=clk, output_provider=provider_factory(values))
    _drive_to_align(rt, clk)
    assert rt.status("locate") == "succeeded" and rt.status("acquire") == "succeeded"
    assert rt.status("support") == "active"
    assert rt.status("align") == "ready"
    r = rt.receipts.require(event_id="locate", attempt=0, output_name="hole_frame", type="frame_estimate")
    assert r.value["pos"] == [0.5, 0.1, 0.2]
    # anchor becomes invalid -> align loses its binding (no silent reuse)
    values.pop(("support", "anchor"))
    rt.tick(make_observation([]))
    assert rt.status("align") == "pending"
    assert rt.instances["align"].reason == "binding_unavailable"


def test_exclusive_resource_conflict_and_actor_rebinding_changes_owner():
    p = supplied_task_payload()
    rt = TaskRuntime(p)
    rt.tick(make_observation([]))
    assert rt.request_event("acquire", 1).accepted
    assert rt.controller_owner("right") == "acquire"
    # add a second event that also needs 'right'
    ev = copy.deepcopy([e for e in p["events"] if e["id"] == "acquire"][0])
    ev["id"] = "acquire2"
    rec = rt.apply_edit([{"op": "add_event", "event": ev}], expected_version=1, request_id="e1")
    assert rec.graph_version == 2
    d = rt.request_event("acquire2", 2)
    assert not d.accepted and d.reason_code == "resource_claimed"
    # rebinding the actor to 'left' moves the exclusive claim; now it is allowed
    rt.apply_edit([{"op": "bind_role", "event_id": "acquire2", "role": "actor", "ordinal": 0,
                    "binding": {"kind": "entity", "entity": {"id": "left", "version": 0}}}],
                  expected_version=2, request_id="e2")
    assert rt.request_event("acquire2", 3).accepted
    assert rt.controller_owner("left") == "acquire2"


def test_timeout_retry_creates_fresh_instance_and_explicit_rebinding():
    clk = Clock()
    values = {("locate", "hole_frame"): {"pos": [0.5, 0.1, 0.2], "quat_wxyz": [1, 0, 0, 0]}}
    rt = TaskRuntime(supplied_task_payload(), clock=clk, output_provider=provider_factory(values))
    rt.tick(make_observation([]))
    assert rt.request_event("locate", 1).accepted
    clk.t = 31.0
    rt.tick(make_observation([]))
    assert rt.attempt("locate") == 1 and rt.status("locate") in ("ready", "pending")
    assert rt.attempt_log[-1]["reason"] == "timeout"
    # downstream bindings were rebound through a recorded, versioned edit
    assert rt.graph_version == 2
    last = rt.store.history[-1]
    assert last.provenance == "recovery" and last.operations[0]["op"] == "rebind_output"
    b = [s.binding for s in rt.compiled.event("align").roles if s.role == "reference"][0]
    assert (b.event_id, b.attempt) == ("locate", 1)
    # the old attempt's output cannot satisfy the new binding
    assert rt.rejections["locate"][-1]["reason"] == "timeout"


def test_rejection_memory_persists_across_unrelated_actions():
    rt = TaskRuntime(supplied_task_payload())
    rt.tick(make_observation([]))
    rt.request_event("insert", 1)
    rt.request_event("acquire", 1)   # unrelated action succeeds
    rt.tick(make_observation([("held_by", ("peg", "right"), False, True)]))
    rt.request_event("insert", 1)
    view = {e.event_id: e for e in rt.public_view().events}
    assert view["insert"].rejection_count == 2
    assert view["insert"].last_rejection_reasons == ["prerequisite_unsatisfied"] * 2


def test_graph_version_conflict_on_request():
    rt = TaskRuntime(supplied_task_payload())
    d = rt.request_event("locate", 0)
    assert not d.accepted and d.reason_code == "graph_version_conflict"


def test_edit_invalidation_listener_and_active_cancellation():
    events = []
    p = supplied_task_payload()
    rt = TaskRuntime(p, on_invalidate=events.append)
    rt.tick(make_observation([]))
    rt.request_event("acquire", 1)
    ev = copy.deepcopy([e for e in p["events"] if e["id"] == "acquire"][0])
    ev["id"] = "acquire_other"
    rt.apply_edit([{"op": "add_event", "event": ev}], 1, "x1")
    assert events and events[-1].startswith("graph_edit")
    assert rt.status("acquire") == "active"  # unaffected event keeps running


def test_snapshot_restore_roundtrip():
    clk = Clock()
    values = {("locate", "hole_frame"): {"pos": [0.5, 0.1, 0.2]}}
    rt = TaskRuntime(supplied_task_payload(), clock=clk, output_provider=provider_factory(values))
    rt.tick(make_observation([]))
    rt.request_event("locate", 1)
    rt.tick(make_observation([("frame_estimate_valid", ("hole",), True, True)]))
    snap = rt.snapshot()
    rt.request_event("acquire", 1)
    rt.restore(snap)
    assert rt.status("acquire") == "ready" and rt.status("locate") == "succeeded"
    assert rt.receipts.require(event_id="locate", attempt=0, output_name="hole_frame",
                               type="frame_estimate").value["pos"] == [0.5, 0.1, 0.2]


def test_repeated_participant_in_different_roles_keeps_separate_incidences():
    p = supplied_task_payload()
    ev = [e for e in p["events"] if e["id"] == "acquire"][0]
    ev["roles"].append({"role": "destination", "ordinal": 0,
                        "binding": {"kind": "entity", "entity": {"id": "right", "version": 0}}})
    c = compile_task(p)
    inc = [(i.role, i.entity_id) for i in c.incidences if i.event_id == "acquire"]
    assert ("actor", "right") in inc and ("destination", "right") in inc


@pytest.mark.parametrize("mutate,code", [
    (lambda p: p["events"][0]["requires_completed"].append("insert") or
     [e for e in p["events"] if e["id"] == "insert"][0]["requires_completed"].append("locate") or
     [e for e in p["events"] if e["id"] == "align"][0]["requires_completed"].append("insert"), "dependency_cycle"),
    (lambda p: [e for e in p["events"] if e["id"] == "insert"][0].update(requires_completed=[]),
     "output_dependency_not_ordered"),
    (lambda p: [e for e in p["events"] if e["id"] == "align"][0].update(requires_active=[]),
     "output_dependency_not_ordered"),
    (lambda p: p["events"][2]["roles"][0].update(binding={"kind": "entity", "entity": {"id": "peg", "version": 0}}),
     "role_type"),
    (lambda p: p["events"][2]["roles"][1].update(ordinal=3), "role_arity"),
    (lambda p: p["events"][2]["roles"].append(copy.deepcopy(p["events"][2]["roles"][1])), "role_arity"),
    (lambda p: [e for e in p["events"] if e["id"] == "align"][0]["resources"][0].update(
        entity={"id": "left", "version": 0}), "resource_conflict"),
    (lambda p: p["events"][0]["roles"][1]["binding"]["entity"].update(id="ghost"), "unknown_entity"),
])
def test_compiler_rejects_invalid_graphs(mutate, code):
    p = supplied_task_payload()
    mutate(p)
    with pytest.raises(ContractError) as ei:
        compile_task(p)
    assert ei.value.code == code
