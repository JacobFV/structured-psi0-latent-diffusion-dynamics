# ops and task-runtime implementation plan

> for agentic workers: use installed subagent-driven-development or executing-plans to implement each task. write and run the failing test before implementation. the user has selected autonomous execution; review internally and proceed after evidence-backed gates.

**goal:** create safe execution and precise information/semantic contracts before simulation or training

**architecture:** implement the boundaries in `../docs/01_architecture.md` and `../docs/02_interfaces.md`; preserve simulator/controller/public-observation separation.

**tech stack:** Python/PyTorch/MuJoCo and React/TypeScript as appropriate, with verified local dependency locks.

**spec:** `../docs/00_scope_and_acceptance.md` and the relevant technical document linked by task.

## global constraints

The parent resource budget is aggregate ≤50% host free resources. All significant work requires a lease. No paid services, global changes, reference-repo mutation, or physical actuation. Commands below are target implementation commands, not a claim that product code ships in this handoff.

## review focus

Test malformed/empty inputs, version/provenance conflicts, variable dimensions and masks, interrupted/restarted jobs, and leakage or false completion. Extend the test examples with the full cases specified in `../docs/08_tests_and_release.md`.

---
## P01: schemas, package and deterministic fixture support

**files:** create `pyproject.toml`, `src/rrp/contracts/{refs,robot,observation,action,task,runs}.py`, `tests/conftest.py`, `tests/support.py`, `src/rrp/cli.py`

**interfaces:** produce validated `EntityRef`, `RobotSpec`, `PolicyObservation`, `PrivilegedTruth`, `ActionChunk`, `TaskDefinition`; initialize `rrp --help` and schema export

- [ ] write the following failing test in `tests/unit/test_contracts.py` (create any explicitly imported fixture helper in the same task; it must not substitute for the implementation under test):

```python
import pytest
from rrp.contracts.refs import EntityRef
from rrp.contracts.task import TaskDefinition
from tests.support import supplied_task_payload

def test_example_and_identity_validation():
    graph = TaskDefinition.model_validate(supplied_task_payload())
    assert graph.task_id == "support_and_insert"
    assert EntityRef(id="peg", version=0).version == 0
    with pytest.raises(ValueError):
        EntityRef(id="peg", version=-1)
```

- [ ] run `uv run pytest tests/unit/test_contracts.py -q`; confirm failure comes from the missing behavior, not an unrelated environment/import issue.
- [ ] implement the behavior below and its additional failure cases.

Pin isolated Python/frontend dependency locks after audit; establish CPU-only pytest and lint/type-check commands. Implement `supplied_task_payload()` by loading an exact copied package example, not an invented simplified graph. Enforce finite numeric values, declared units, bounded shapes and typed errors. Generate JSON schemas and frontend types from one authority. Check the provided task schema and example. Add serialization round trips, unknown field rejection and no privileged fields in the public observation transport. Create directories and docs as part of this deliverable, not a separate scaffolding-only milestone.

- [ ] rerun that test file and all upstream contract tests; perform the specified integration check under a lease. save the command/output and artifact path in the task ledger.
- [ ] self-review against the spec, have an independent reviewer inspect high-risk changes when available, then make a local scoped commit. do not claim downstream experiments passed merely because this task's unit test passed.

## P02: machine discovery and truthful telemetry

**files:** create `src/rrp/ops/{discovery,telemetry,budget}.py`, `configs/resources.local.json`

**interfaces:** consume measured `Snapshot`; produce conservative host/peer `ResourceBudget` and discovered source-locked capability profile

- [ ] write the following failing test in `tests/unit/test_budget.py` (create any explicitly imported fixture helper in the same task; it must not substitute for the implementation under test):

```python
from rrp.ops.budget import host_budget

def test_half_free_not_half_total():
    b = host_budget(total_ram_gib=128, available_ram_gib=40,
                    free_cpu_cores=6, free_disk_gib=200,
                    total_disk_gib=1000)
    assert b.memory_gib <= 20
    assert b.cpu_cores <= 3
    assert b.new_disk_gib <= 100
    assert not b.host_gpu_enabled
```

- [ ] run `uv run pytest tests/unit/test_budget.py -q`; confirm failure comes from the missing behavior, not an unrelated environment/import issue.
- [ ] implement the behavior below and its additional failure cases.

Implement the formulas and lower bounds in the resource contract; never round fractional free CPU upward. Respect existing cgroup/affinity limits and shared unified memory. Discover only existing relevant SSH aliases with bounded batch probes; verify different machines and safe routes. Write `rrp doctor` and a redacted host/peer manifest. Test missing/N/A telemetry, swap pressure, zero/negative/nonfinite capacities, no peer, same-machine alias and foreign workload. This is measurement, not workload authorization.

- [ ] rerun that test file and all upstream contract tests; perform the specified integration check under a lease. save the command/output and artifact path in the task ledger.
- [ ] self-review against the spec, have an independent reviewer inspect high-risk changes when available, then make a local scoped commit. do not claim downstream experiments passed merely because this task's unit test passed.

## P03: aggregate enforcement, leases and watchdog

**files:** create `src/rrp/ops/{broker,cgroup,jobs,watchdog}.py`, `tests/integration/test_resource_enforcement.py`

**interfaces:** produce `ResourceBroker.acquire/heartbeat/release`; all children share one parent allocation; owned-only shutdown

- [ ] write the following failing test in `tests/unit/test_broker.py` (create any explicitly imported fixture helper in the same task; it must not substitute for the implementation under test):

```python
import pytest
from rrp.ops.broker import ResourceBroker, ResourceRequest, CapacityError
from tests.support import fake_enforcement_backend

def test_children_cannot_each_claim_the_parent_budget():
    broker = ResourceBroker(cpu_limit=2.0, memory_limit_bytes=1_000_000,
                            backend=fake_enforcement_backend())
    broker.acquire(ResourceRequest(cpu_cores=1.5, memory_bytes=600_000))
    with pytest.raises(CapacityError):
        broker.acquire(ResourceRequest(cpu_cores=1.0, memory_bytes=600_000))
```

- [ ] run `uv run pytest tests/unit/test_broker.py -q`; confirm failure comes from the missing behavior, not an unrelated environment/import issue.
- [ ] implement the behavior below and its additional failure cases.

The fake backend is only for arithmetic/state-machine unit tests. Implement a real delegated user cgroup or existing isolated backend, then use bounded subprocesses to verify CPU/memory aggregation and enforcement. No heavy host work without proof. Include lease expiry, telemetry failure, external load rise, emergency pressure, checkpoint reserve, start-time/PID reuse, job resume and owned-only cleanup. Simulate watchdog signals on small owned fixtures; never pressure-test the user's whole machine. Ensure dependency builds and managed browsers also run under the parent quota.

- [ ] rerun that test file and all upstream contract tests; perform the specified integration check under a lease. save the command/output and artifact path in the task ledger.
- [ ] self-review against the spec, have an independent reviewer inspect high-risk changes when available, then make a local scoped commit. do not claim downstream experiments passed merely because this task's unit test passed.

## P04: event compiler and runtime guards

**files:** create `src/rrp/tasks/{compiler,guards,runtime,resources}.py`

**interfaces:** consume validated `TaskDefinition`; produce ready/active states and `ExecutionDecision` without auto-completing effects

- [ ] write the following failing test in `tests/unit/test_task_runtime.py` (create any explicitly imported fixture helper in the same task; it must not substitute for the implementation under test):

```python
from rrp.tasks.compiler import compile_task
from rrp.tasks.runtime import TaskRuntime
from tests.support import supplied_task_payload, observation_with_no_grasp

def test_insert_cannot_skip_prerequisites():
    runtime = TaskRuntime(compile_task(supplied_task_payload()))
    runtime.tick(observation_with_no_grasp())
    decision = runtime.request_event("insert", expected_version=1)
    assert not decision.accepted
    assert decision.reason_code in {"prerequisite_unsatisfied", "binding_unavailable"}
    assert runtime.status("insert") != "succeeded"
```

- [ ] run `uv run pytest tests/unit/test_task_runtime.py -q`; confirm failure comes from the missing behavior, not an unrelated environment/import issue.
- [ ] implement the behavior below and its additional failure cases.

Validate IDs, arity, ordered/multiple roles, dependency orientation, cycles and declared controller ownership. Implement pending/ready/active/succeeded/failed/cancelled/blocked and required-completed versus required-active relations. Invariants apply in the specified event phase; missing outputs prevent dependent activation rather than making hidden success true. A support event can acquire contact before exporting a valid anchor. Produce desired deltas separately from current estimates. Expose reasons, not only false/true booleans.

- [ ] rerun that test file and all upstream contract tests; perform the specified integration check under a lease. save the command/output and artifact path in the task ledger.
- [ ] self-review against the spec, have an independent reviewer inspect high-risk changes when available, then make a local scoped commit. do not claim downstream experiments passed merely because this task's unit test passed.

## P05: functional composition, receipts and retry memory

**files:** create `src/rrp/tasks/{receipts,binding,recovery}.py`

**interfaces:** produce scoped output records with event/attempt/participant/version provenance, validity and persistent rejection history

- [ ] write the following failing test in `tests/unit/test_receipts.py` (create any explicitly imported fixture helper in the same task; it must not substitute for the implementation under test):

```python
import pytest
from rrp.tasks.receipts import ReceiptStore, Receipt, ProvenanceError

def test_same_type_foreign_frame_is_not_accepted():
    store = ReceiptStore()
    store.put(Receipt(event_id="other", attempt=0, output_name="hole_frame",
                      type="frame_estimate", version=1, value={"x": 0.0}))
    with pytest.raises(ProvenanceError):
        store.require(event_id="locate", attempt=0, output_name="hole_frame",
                      type="frame_estimate", version=1)
```

- [ ] run `uv run pytest tests/unit/test_receipts.py -q`; confirm failure comes from the missing behavior, not an unrelated environment/import issue.
- [ ] implement the behavior below and its additional failure cases.

Implement output publication on success or while active-and-valid; retain output participants and frame uncertainty. Add separate tests for wrong type, same-type foreign, stale same-event, old-but-valid outputs and duplicate appearances. Recovery creates fresh instances and explicitly rebinds downstream references; it never silently reuses an old successful result. Persist rejection counts/reasons across unrelated actions. Build encoded-input counterfactual tests: provenance/history changes must reach public policy tensors when the behavior should differ. Add a two-event data-dependent example where changing the estimated target frame changes downstream control.

- [ ] rerun that test file and all upstream contract tests; perform the specified integration check under a lease. save the command/output and artifact path in the task ledger.
- [ ] self-review against the spec, have an independent reviewer inspect high-risk changes when available, then make a local scoped commit. do not claim downstream experiments passed merely because this task's unit test passed.

## P06: public/private channels, snapshots and graph edits

**files:** create `src/rrp/contracts/channels.py`, `src/rrp/tasks/interventions.py`, `src/rrp/sim/snapshot_contract.py`

**interfaces:** versioned transactional graph edits; serializable public observation and full continuation snapshot contracts

- [ ] write the following failing test in `tests/unit/test_interventions.py` (create any explicitly imported fixture helper in the same task; it must not substitute for the implementation under test):

```python
import pytest
from rrp.tasks.interventions import GraphStore, VersionConflict
from tests.support import supplied_task_payload

def test_stale_edit_cannot_partially_mutate_graph():
    store = GraphStore(supplied_task_payload())
    before = store.content_hash()
    with pytest.raises(VersionConflict):
        store.apply_edit([], expected_version=0, request_id="r1")
    assert store.content_hash() == before
```

- [ ] run `uv run pytest tests/unit/test_interventions.py -q`; confirm failure comes from the missing behavior, not an unrelated environment/import issue.
- [ ] implement the behavior below and its additional failure cases.

Graph edit commit atomically increments versions, invalidates KV and unsent action queues, and logs intervention provenance. Distinguish undoing graph edits from physical rewind. Implement idempotency, partial-edit rejection and reconnect rehydration. Public policy transport cannot serialize PrivilegedTruth or private object poses; include hostile unknown-field tests. Snapshot contract enumerates runtime/controller/sensor/belief/RNG state so later physics restoration does not omit it.

- [ ] rerun that test file and all upstream contract tests; perform the specified integration check under a lease. save the command/output and artifact path in the task ledger.
- [ ] self-review against the spec, have an independent reviewer inspect high-risk changes when available, then make a local scoped commit. do not claim downstream experiments passed merely because this task's unit test passed.
