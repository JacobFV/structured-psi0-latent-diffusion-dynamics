# interfaces, event semantics, and information boundaries

Implement typed Python models in `src/rrp/contracts/` and generate frontend types from backend schemas. Public CLI package: `rrp`. Schemas and API version use `1.0`; versions change explicitly when incompatible.

## domain contracts

`EntityRef(id: str, version: int)` is an opaque runtime identity; version is nonnegative. It is not a target label, simulator-body address, or product name. Binding validity depends on identity, scope, type, provenance, and state version. IDs may be randomized jointly to test invariance. Names are presentation metadata.

`RobotSpec` fields: schema_version, spec_hash, asset_source, geometry_hashes, nodes, typed_edges, actuators, transmissions, sensors, assemblies, attachment_ports, controller_contracts, capability_tags. `NodeState` contains valid measured coordinates, masks, timestamps and units. Controller contracts define independent command groups, widths, units, normalization, bounds, frequency, ownership, hold/stop behavior, state required, and reset/snapshot hooks.

`PolicyObservation` fields: observation_id, sensor_time, robot_spec_hash, sensor_images, measured_node_state, declared_sensor_channels, task_input, belief_history. It must NOT contain `PrivilegedTruth`. `PrivilegedTruth` carries simulator object identity/pose, contacts, reward/completion truth and training labels on a separate bus. Serialize each type independently and test that the deployment transport cannot reference the private bus.

`TaskContext` carries supplied task definition, visible descriptors, runtime states inferred from allowed observations, observation-backed receipts, and user interventions. An oracle-active-event flag derived from hidden completion state is privileged too. In an explicitly labeled oracle diagnostic, an alternate context type carries truth and is prevented from being exported as deployable-policy evidence.

`ActionChunk` fields: observation_id, graph_version, runtime_version, robot_spec_hash, controller_version, policy_version, codec_version, start_time, dt, horizon, command_groups, masks, sampling_seed. Controller rejects stale ownership/version, nonfinite, wrong-width/unit, or expired chunks. A stale chunk is not silently clipped into a new plan.

`Snapshot` includes physics state, actuator/controller internal state, task-runtime status, sensor filters/history, entity tracker, frame/belief state, outstanding command queue, sampler RNG, environment RNG, source and backend numerical versions. Round-trip continuation must be tested. Remote replay within floating-point tolerance is distinct from bitwise reproducibility.

## event model

An event is an instance of an operator with typed role slots. Roles have names and ordinal indices. Repeated participants in different roles remain separate edges. Event nodes plus role-slot incidence implement a hypergraph without losing argument order or multiplicity.

An event stores:

```text
id, operator, bindings[role, ordinal, entity/input-ref],
preconditions[], invariants[], desired_effects[], completion[],
requires_completed[], requires_active[], resource_claims[],
produces[output-name, type], frame_binding?, timeout,
recovery_policy, status, attempt, graph_version
```

Statuses: pending, ready, active, succeeded, failed, cancelled, blocked. Only the runtime changes execution status. Desired effects describe targets. Completion evaluates observable estimates with declared thresholds/time persistence. Failed guards preserve their reason and scope; repeated identical retries must be detectable after intervening actions.

`requires_completed` is a directed acyclic graph within an active plan. `requires_active` captures a maintained support event while another event executes. Resource claims prevent conflicting actuator commands; shared support claims differ from exclusive control claims. Alternatives are explicit branches; recovery creates fresh event instances. Scene/contact graphs may have cycles. Do not impose a universal DAG on all graphs.

Preconditions are checks, not deltas. Invariants must continue holding while active. Effects can be boolean target changes, scalar intervals, relative poses or contact-mode changes. A rigid-transform delta is composition in a declared frame, not subtraction of two quaternion vectors.

## functional composition and provenance

An event may produce a typed result from its actual estimator/controller execution, e.g. an estimated hole frame. A later event input binds `event_id + attempt/version + output_name`. Runtime validates the receipt, type, source participant and validity. Never resolve only “the latest value of that type.” Wrong-type, same-type foreign, stale, and still-valid older results are separate test cases.

Example: `locate_hole` emits an observed frame; `establish_support` emits a maintained contact anchor; `align` consumes those estimates and emits a feasible aligned configuration; `insert` uses the same frame version while support remains active. Randomizing hole geometry makes the upstream output actually necessary. A supplied fixed sequence alone demonstrates sequencing, not functional composition.

Persist reasons such as “out of reach,” “grasp confidence insufficient,” “resource claimed,” “anchor slipped,” “stale receipt,” or “timeout.” Store attempt counts and reason summaries in the policy's declared history. Before expecting different choices, test that encoded inputs distinguish relevant histories. This implements the lesson from topoformer's extended-02 identifiability audit rather than copying its policy.

## attention routing versus execution authority

Semantic dependency e_i -> e_j means i enables j; retrieval edge j -> i means query j reads prerequisite i. Compile these orientations explicitly. Typed structural bias operates on latent associations: `b_r = p_query a_r p_key^T`, optionally after incidence-message encoding. Null associations prevent unrelated tokens being forcibly assigned to graph nodes.

Soft bias can guide attention; it does not enforce a precondition or resource constraint. The backend runtime enforces known schema/execution rules. Keep general content attention so missing graph edges do not falsely forbid physical coupling. Do not derive a causal conclusion solely from attention concentration.

## method signatures to implement

```python
class RobotCompiler:
    def compile(self, source: AssetSource) -> RobotSpec: ...
    def attach(self, body: RobotSpec, module: RobotSpec, port: AttachmentPort) -> RobotSpec: ...

class Simulator:
    def reset(self, robot: RobotSpec, task: TaskDefinition, seed: int) -> PolicyObservation: ...
    def step(self, command: NativeCommand) -> StepResult: ...
    def snapshot(self) -> Snapshot: ...
    def restore(self, snapshot: Snapshot) -> PolicyObservation: ...

class TaskRuntime:
    def tick(self, observation: PolicyObservation) -> TaskContext: ...
    def request_event(self, event_id: str, expected_version: int) -> ExecutionDecision: ...
    def apply_edit(self, edit: GraphEdit, expected_version: int) -> EditReceipt: ...

class Policy:
    def prepare(self, observation: PolicyObservation, robot: RobotSpec) -> ContextCache: ...
    def sample(self, cache: ContextCache, seed: int, nfe: int) -> ActionChunk: ...

class ResourceBroker:
    def acquire(self, request: ResourceRequest) -> Lease: ...
    def heartbeat(self, lease_id: str, usage: Usage) -> LeaseDecision: ...
    def release(self, lease_id: str) -> None: ...
```

These are target interfaces, not implementations supplied in the handoff. Concrete data classes live in the contracts module; no cross-module dicts of silently guessed shapes. Every exception has a typed code and an operator-readable message. Invalid states are rejected before starting physics/control work.

## task definition schema

`contracts/task_graph.schema.json` validates the serializable plan. Semantic validation additionally checks referential integrity, role type/arity, prerequisite cycles, overlapping controller ownership and output provenance. The example in `examples/support_and_insert.task.json` demonstrates maintained support and data-dependent event outputs; implement runtime lowering rather than executing the JSON as Python or arbitrary code.
