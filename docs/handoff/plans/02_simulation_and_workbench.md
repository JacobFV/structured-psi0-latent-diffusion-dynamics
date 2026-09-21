# simulation and workbench implementation plan

> for agentic workers: use installed subagent-driven-development or executing-plans to implement each task. write and run the failing test before implementation. the user has selected autonomous execution; review internally and proceed after evidence-backed gates.

**goal:** produce real robot/task interaction and breadth-ready data generation

**architecture:** implement the boundaries in `../docs/01_architecture.md` and `../docs/02_interfaces.md`; preserve simulator/controller/public-observation separation.

**tech stack:** Python/PyTorch/MuJoCo and React/TypeScript as appropriate, with verified local dependency locks.

**spec:** `../docs/00_scope_and_acceptance.md` and the relevant technical document linked by task.

## global constraints

The parent resource budget is aggregate ≤50% host free resources. All significant work requires a lease. No paid services, global changes, reference-repo mutation, or physical actuation. Commands below are target implementation commands, not a claim that product code ships in this handoff.

## review focus

Test malformed/empty inputs, version/provenance conflicts, variable dimensions and masks, interrupted/restarted jobs, and leakage or false completion. Extend the test examples with the full cases specified in `../docs/08_tests_and_release.md`.

---
## P07: native primitive simulation and sensors

**files:** create `src/rrp/sim/{native,fixtures,sensors,truth,render}.py`

**interfaces:** implement `Simulator.reset/step/snapshot/restore` with actual MuJoCo dynamics; separate public sensors and truth

- [ ] write the following failing test in `tests/integration/test_native_sim.py` (create any explicitly imported fixture helper in the same task; it must not substitute for the implementation under test):

```python
import numpy as np
from rrp.sim.fixtures import make_arm_session

def test_physics_steps_and_full_restore():
    sim = make_arm_session(seed=7)
    snap = sim.snapshot()
    command = sim.fixture_command()
    first = sim.step(command)
    sim.restore(snap)
    second = sim.step(command)
    assert np.allclose(first.qpos, second.qpos, atol=1e-8)
    assert np.isfinite(first.qpos).all()
```

- [ ] run `uv run pytest tests/integration/test_native_sim.py -q`; confirm failure comes from the missing behavior, not an unrelated environment/import issue.
- [ ] implement the behavior below and its additional failure cases.

Build primitive XML in project-owned fixtures with valid mass/inertia and bounded controllers. Verify installed MuJoCo/renderer on the actual architecture. Implement sensor timestamps, image rendering and separate truth labels. Snapshot both controller and runtime, not only physics positions. Test rate/units mismatches, missing cameras, nonfinite actions, constraints and contact state. GPU rendering runs on the peer; host tests use software/reference paths. An image-only mock scene does not pass.

- [ ] rerun that test file and all upstream contract tests; perform the specified integration check under a lease. save the command/output and artifact path in the task ledger.
- [ ] self-review against the spec, have an independent reviewer inspect high-risk changes when available, then make a local scoped commit. do not claim downstream experiments passed merely because this task's unit test passed.

## P08: controller adapters and first teacher task

**files:** create `src/rrp/control/{base,joint_targets,psi_original,sonic,teachers}.py`, `src/rrp/data/collect.py`

**interfaces:** validated controller contracts and teacher trace through the same event runtime later used by policies

- [ ] write the following failing test in `tests/integration/test_teacher.py` (create any explicitly imported fixture helper in the same task; it must not substitute for the implementation under test):

```python
from rrp.control.teachers import run_fixture_pick_place

def test_real_teacher_trace_has_actions_and_success_evidence():
    result = run_fixture_pick_place(seed=4, max_control_steps=600)
    assert len(result.actions) > 0
    assert result.controller_source == "scripted_teacher"
    assert result.success == result.privileged_evaluator_success
    assert result.success
```

- [ ] run `uv run pytest tests/integration/test_teacher.py -q`; confirm failure comes from the missing behavior, not an unrelated environment/import issue.
- [ ] implement the behavior below and its additional failure cases.

Implement and validate joint-target arm/gripper tracking plus task teachers. Pin original ψ₀ and SONIC action/group/state contracts separately; inspect actual code/checkpoints instead of padding blindly. Scripted teachers may use privileged data only with labels. Record failed attempts. Test controller ownership, invalid width/unit/version, limits, hold and reset. Produce a first actual episode/video before importing the entire robot catalogue. No teacher trace is described as learned control.

- [ ] rerun that test file and all upstream contract tests; perform the specified integration check under a lease. save the command/output and artifact path in the task ledger.
- [ ] self-review against the spec, have an independent reviewer inspect high-risk changes when available, then make a local scoped commit. do not claim downstream experiments passed merely because this task's unit test passed.

## P09: catalogue import, physical validation and module surgery

**files:** create `src/rrp/morphology/{registry,importers,ports,surgery,validation,generators}.py`

**interfaces:** resolve candidate assets to pinned RobotSpec; create physically validated synthetic bodies and fresh hashes

- [ ] write the following failing test in `tests/unit/test_surgery.py` (create any explicitly imported fixture helper in the same task; it must not substitute for the implementation under test):

```python
import pytest
from rrp.morphology.fixtures import arm_with_port, gripper_module
from rrp.morphology.surgery import attach, AttachmentError

def test_attachment_rebinds_names_and_preserves_positive_mass():
    a = attach(arm_with_port(), gripper_module(), port_id="wrist")
    assert len(a.joint_names) == len(set(a.joint_names))
    assert all(link.mass > 0 for link in a.rigid_links)
    with pytest.raises(AttachmentError):
        attach(arm_with_port(), gripper_module(), port_id="absent")
```

- [ ] run `uv run pytest tests/unit/test_surgery.py -q`; confirm failure comes from the missing behavior, not an unrelated environment/import issue.
- [ ] implement the behavior below and its additional failure cases.

Implement fixtures in `src/rrp/morphology/fixtures.py` with actual model records. Resolve each candidate's license/model file without claiming its controller is ready. Surgery rebinds named references, sensors, constraints and actuator maps; validates frames/payload/inertia/collision and tracker stability. Add force/inertia/DOF-changing tests and generated legged models. Persist rejection/evidence states. Keep source/composition lineage for splits. A URDF load alone is not controller validation.

- [ ] rerun that test file and all upstream contract tests; perform the specified integration check under a lease. save the command/output and artifact path in the task ledger.
- [ ] self-review against the spec, have an independent reviewer inspect high-risk changes when available, then make a local scoped commit. do not claim downstream experiments passed merely because this task's unit test passed.

## P10: authoritative service and bounded streaming

**files:** create `src/rrp/service/{app,sessions,commands,graphs,probes,stream}.py`

**interfaces:** loopback HTTP/WebSocket APIs, authentication/origin checks, versions, idempotency and backpressure

- [ ] write the following failing test in `tests/integration/test_service.py` (create any explicitly imported fixture helper in the same task; it must not substitute for the implementation under test):

```python
from fastapi.testclient import TestClient
from rrp.service.app import create_test_app

def test_loopback_api_rejects_unauthed_mutation():
    client = TestClient(create_test_app())
    assert client.get("/health").status_code == 200
    response = client.post("/api/sessions", json={"robot": "fixture_arm"})
    assert response.status_code in {401, 403}
```

- [ ] run `uv run pytest tests/integration/test_service.py -q`; confirm failure comes from the missing behavior, not an unrelated environment/import issue.
- [ ] implement the behavior below and its additional failure cases.

Create real session service around the simulator/runtime. Implement all endpoints in the workbench doc and small bounded test fixtures. Mutations have per-session authentication and validated inputs, no arbitrary filesystem/script execution. Bounded queues drop render frames rather than control commands. Handle disconnect/reconnect and stale graph versions without duplicated actuation. Add integration tests with authenticated actual physics sessions. Expose resource telemetry read-only through the broker.

- [ ] rerun that test file and all upstream contract tests; perform the specified integration check under a lease. save the command/output and artifact path in the task ledger.
- [ ] self-review against the spec, have an independent reviewer inspect high-risk changes when available, then make a local scoped commit. do not claim downstream experiments passed merely because this task's unit test passed.

## P11: working robot and dependency-graph GUI

**files:** create `ui/src/{scene,graph,inspector,playback,resources,transport}/`, `tests/browser/workbench.spec.ts`

**interfaces:** live scene selection and command/graph/probe interactions via service; no UI-owned execution state

- [ ] write the following failing test in `tests/integration/test_ui_contract.py` (create any explicitly imported fixture helper in the same task; it must not substitute for the implementation under test):

```python
from rrp.service.schemas import public_message_kinds

def test_ui_protocol_exposes_state_and_intervention_receipts():
    kinds = set(public_message_kinds())
    assert {"session_snapshot", "graph_committed", "command_rejected",
            "probe_result", "resource_update"} <= kinds
```

- [ ] run `uv run pytest tests/integration/test_ui_contract.py -q`; confirm failure comes from the missing behavior, not an unrelated environment/import issue.
- [ ] implement the behavior below and its additional failure cases.

Implement full UI behavior and a real Playwright suite: launch/step, target movement, robot/object/manipulator selection, graph add/bind/connect/edit, cycle rejection, unsatisfied future-event request, changed actor routing, replay, reconnect and visible control-mode labels. The Python protocol test is only a starter; browser acceptance is mandatory. Include software/streaming mode for host safety. Persist a short actual browser video. Use shared typed backend schemas; graph layout moves never change physics.

- [ ] rerun that test file and all upstream contract tests; perform the specified integration check under a lease. save the command/output and artifact path in the task ledger.
- [ ] self-review against the spec, have an independent reviewer inspect high-risk changes when available, then make a local scoped commit. do not claim downstream experiments passed merely because this task's unit test passed.

## P12: broad controllers, dataset streams and replay

**files:** create `src/rrp/control/{legged,tracker_training}.py`, `src/rrp/data/{dataset,manifest,chunking,features}.py`, `src/rrp/evaluation/replay.py`

**interfaces:** validated family trackers, generated/recorded episodes and source-hashed replayable chunks

- [ ] write the following failing test in `tests/unit/test_dataset_split.py` (create any explicitly imported fixture helper in the same task; it must not substitute for the implementation under test):

```python
from rrp.data.manifest import assert_disjoint_lineages
import pytest

def test_near_duplicate_morphology_cannot_leak():
    with pytest.raises(ValueError):
        assert_disjoint_lineages(train={"bodyA/handB"}, test={"bodyA/handB"})
```

- [ ] run `uv run pytest tests/unit/test_dataset_split.py -q`; confirm failure comes from the missing behavior, not an unrelated environment/import issue.
- [ ] implement the behavior below and its additional failure cases.

Build/validate the nominal procedural hexapod tracker and independent humanoid/quadruped control integrations required by breadth. Log training/calibration cost. Implement dataset/public-private separation, masks/dt, source-only normalization and versioned feature caches. Test all declared snapshot components and deterministic replay on a fixed backend. Controller and teacher eligibility is frozen separately from policy success. Never delay the first learned arm experiment until every breadth candidate works, but complete breadth tasks before final closure.

- [ ] rerun that test file and all upstream contract tests; perform the specified integration check under a lease. save the command/output and artifact path in the task ledger.
- [ ] self-review against the spec, have an independent reviewer inspect high-risk changes when available, then make a local scoped commit. do not claim downstream experiments passed merely because this task's unit test passed.
