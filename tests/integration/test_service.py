import json
import pytest
from fastapi.testclient import TestClient
from rrp.service.app import create_test_app

H = {"x-rrp-token": "test-token"}


def test_loopback_api_rejects_unauthed_mutation():
    client = TestClient(create_test_app())
    assert client.get("/health").status_code == 200
    response = client.post("/api/sessions", json={"robot": "fixture_arm"})
    assert response.status_code in {401, 403}


def _session(client, robot="parm5_pg2", seed=1):
    r = client.post("/api/sessions", json={"robot": robot, "task": "pick_place", "seed": seed}, headers=H)
    assert r.status_code == 200, r.text
    return r.json()["session_id"], r.json()["snapshot"]


def test_selectors_reject_paths_and_bad_origin():
    c = TestClient(create_test_app())
    assert c.post("/api/sessions", json={"robot": "../etc/passwd"}, headers=H).status_code == 422
    assert c.post("/api/sessions", json={"robot": "nope"}, headers=H).status_code == 400
    r = c.post("/api/sessions", json={"robot": "parm5_pg2"}, headers={**H, "origin": "http://evil.example"})
    assert r.status_code == 403
    r = c.post("/api/sessions", content="x", headers={**H, "content-type": "text/plain"})
    assert r.status_code == 415


def test_physics_backed_step_teacher_labels_and_replay():
    c = TestClient(create_test_app())
    sid, snap = _session(c)
    q0 = snap["observation"]["joints"]["qpos"]
    r = c.post(f"/api/sessions/{sid}/commands", json={"type": "set_mode", "mode": "scripted_teacher"}, headers=H)
    assert "SCRIPTED TEACHER" in r.json()["label"]
    steps = c.post(f"/api/sessions/{sid}/commands", json={"type": "step", "n": 40}, headers=H).json()["steps"]
    assert all(s["source"] == "scripted_teacher" for s in steps)
    snap2 = c.get(f"/api/sessions/{sid}/snapshot").json()
    assert snap2["observation"]["joints"]["qpos"] != q0
    rep = c.post(f"/api/sessions/{sid}/commands", json={"type": "replay"}, headers=H).json()
    assert rep["final_state_match"], rep


def test_joint_target_validated_and_event_guard():
    c = TestClient(create_test_app())
    sid, snap = _session(c)
    bad = c.post(f"/api/sessions/{sid}/commands", json={"type": "joint_target", "group": "arm", "values": [9] * 5},
                 headers=H)
    assert bad.status_code == 422 and bad.json()["code"] == "out_of_bounds"
    ok = c.post(f"/api/sessions/{sid}/commands", json={"type": "joint_target", "group": "arm",
                                                        "values": [0.3, 0.5, 1.0, 1.4, 0.0]}, headers=H)
    assert ok.status_code == 200
    c.post(f"/api/sessions/{sid}/commands", json={"type": "step", "n": 5}, headers=H)
    v = snap["graph"]["version"]
    r = c.post(f"/api/sessions/{sid}/commands", json={"type": "request_event", "event_id": "place",
                                                       "expected_version": v}, headers=H).json()
    assert not r["accepted"] and r["reason_code"] == "prerequisite_unsatisfied"
    s3 = c.get(f"/api/sessions/{sid}/snapshot").json()
    assert s3["runtime"]["events"]["grasp"]["status"] != "succeeded"


def test_graph_edit_conflict_cycle_and_layout_does_not_change_version():
    c = TestClient(create_test_app())
    sid, snap = _session(c)
    v = snap["graph"]["version"]
    lay = c.put(f"/api/sessions/{sid}/graph/layout", json={"positions": {"grasp": [10, 20]}}, headers=H).json()
    assert lay["graph_version"] == v
    cyc = c.post(f"/api/sessions/{sid}/graph", json={"expected_version": v, "request_id": "a",
                                                      "operations": [{"op": "add_dependency", "src": "place",
                                                                      "dst": "grasp", "mode": "completed"}]}, headers=H)
    assert cyc.status_code == 422 and cyc.json()["code"] == "dependency_cycle"
    ok = c.post(f"/api/sessions/{sid}/graph", json={"expected_version": v, "request_id": "b",
                                                     "operations": [{"op": "set_priority", "event_id": "grasp",
                                                                     "priority": 2}]}, headers=H)
    assert ok.status_code == 200 and ok.json()["graph_version"] == v + 1
    stale = c.post(f"/api/sessions/{sid}/graph", json={"expected_version": v, "request_id": "c",
                                                        "operations": [{"op": "set_priority", "event_id": "grasp",
                                                                        "priority": 3}]}, headers=H)
    assert stale.status_code == 409


def test_teleport_contaminates_and_probes_answer_with_sources():
    c = TestClient(create_test_app())
    sid, _ = _session(c)
    c.post(f"/api/sessions/{sid}/commands", json={"type": "step", "n": 2}, headers=H)
    vis = c.post(f"/api/sessions/{sid}/probes", json={"query": "visible"}, headers=H).json()
    assert vis["source"] == "public_estimator" and len(vis["answers"]) >= 2
    held = c.post(f"/api/sessions/{sid}/probes", json={"query": "held_by", "subject": "gripper"}, headers=H).json()
    assert held["answers"][0]["null"] is True
    qa = c.post(f"/api/sessions/{sid}/probes", json={"query": "object_qa", "object": "cube",
                                                      "question": "what color?"}, headers=H).json()
    assert qa["null"] and "no object-QA model" in qa["unknown_reason"]
    r = c.post(f"/api/sessions/{sid}/commands", json={"type": "teleport", "body": "cube", "pos": [0.4, 0.1, 0.05]},
               headers=H).json()
    assert r["contaminated"]
    assert c.get(f"/api/sessions/{sid}/snapshot").json()["contaminated"] is True
    snap = c.get(f"/api/sessions/{sid}/snapshot?privileged=true").json()
    assert "PRIVILEGED" in snap["privileged_overlay"]["label"]


def test_websocket_requires_token_and_rehydrates():
    c = TestClient(create_test_app())
    sid, _ = _session(c)
    with pytest.raises(Exception):
        with c.websocket_connect(f"/ws/sessions/{sid}?token=wrong") as ws:
            ws.receive_text()
    with c.websocket_connect(f"/ws/sessions/{sid}?token=test-token") as ws:
        first = json.loads(ws.receive_text())
        assert first["kind"] == "session_snapshot"
        c.post(f"/api/sessions/{sid}/commands", json={"type": "step", "n": 1}, headers=H)
        msg = json.loads(ws.receive_text())
        assert msg["kind"] == "state_update" and msg["seq"] >= 1


def test_ui_protocol_exposes_state_and_intervention_receipts():
    from rrp.service.schemas import public_message_kinds
    kinds = set(public_message_kinds())
    assert {"session_snapshot", "graph_committed", "command_rejected",
            "probe_result", "resource_update"} <= kinds
