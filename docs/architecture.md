# rrp architecture (D-140; replaces docs/core_api.md and docs/repo_structure_audit.md)

rrp is ONE repository for every approach and every environment: direct-action BC, the latent packet
(system i + system 0; semfix / nosem), learned trackers, Ψ₀ direct and Ψ₀ + structure (migrated from psi1z), on our
MuJoCo arm / dual-arm / legged / humanoid scenes (CPU MuJoCo and MuJoCo Warp), Ψ₀'s SIMPLE benchmark, and ComputerWorld.
Everything plugs in through three interfaces, `Env`, `Policy` and `Task`, and one harness that runs any
Policy × Env × Task that declares compatibility. This page is the design, the delete list and the migration plan; the
status column of the plan is kept current by the refactor lead.

## 1. layers

Each layer imports only layers above it in this list (checked by `tests/unit/test_layering.py`).

| layer | package | contents | heavy deps |
|---|---|---|---|
| 0 | `rrp.core` | contracts: arrays/Strict base, `Observation` (= `PolicyObservation`), `PrivilegedTruth`, `NativeCommand`, `ActionChunk`, `LatentActionChunk` (packet z[K,M,D] + assembly handles/mask + entity registry), `System0Base`, errors, `RobotSpec` (morphology graph), `TaskDefinition`, provenance + `Source` labels, `RunConfig`, paths, run/workload helpers, public/privileged channels, Ψ action contracts | numpy, pydantic |
| 0 | `rrp.ops` | broker, leases, cgroups, watchdog, jobs, telemetry, discovery (was `orchestration`) | – |
| 1 | `rrp.bodies` | procedural + imported bodies (arms, grippers, aloha, legged, humanoids, G1 hands), catalog/registry, compiler to MuJoCo, IK, surgery/variants, **physics versions** (contact, grasp contact, actuator; was `physics/`) | mujoco |
| 2 | `rrp.tasks` | task graph compiler/runtime/receipts/interventions (numpy/pydantic), task JSON specs, the `TaskSpec` registry (which envs a task exists in, success/termination, scripted teacher key, gates) | – |
| 3 | `rrp.envs` | `Env` protocol, `EnvSpec`, capabilities, `make_env` registry; `mujoco/` (Session, LeggedSession, DualSession, scenes, sensors, state estimation, perturbations, embedded legged trackers, snapshots), `warp/` (batched GPU legged envs), `simple/` (optional extra), `computerworld/` (optional extra) | mujoco, mujoco_warp, torch (trackers) |
| 4 | `rrp.policies` | `Policy` protocol, `PolicyInfo`, `Requirements`, `negotiate`, registry; `features/` (featurizers: the ONLY definition of what a policy may see), `nets/` (shared torch modules: attention, flow, codec, backbone, probes, checkpoint), `bc.py`, `latent/` (system i planners + system 0 realizers for arm, dual, legged), `trackers.py`, `teachers/` (scripted / privileged, labelled), `oracle.py`, `psi0/` (to be filled by the Ψ₀ migration) | torch |
| 5 | `rrp.harness` | `rollout` (the one episode loop), `evaluate`/`matrix`, hooks (packet edits, perturbations, recorders), statistics, gates, audits; `data/` (collect, pack, manifests), `train/` (rep, flow, bc, refit, dagger, sft, grpo, ppo), pipelines + run-dag, the `rrp` CLI | – |
| 6 | `rrp.viz` | record/replay, the room exporter (`python -m rrp.viz.export`, file scans only), the workbench service (was `service/`) | fastapi (extra) |

Rules:
- No compatibility shims or deprecated aliases. A moved name is imported from its new path everywhere in the same
  commit. The only legacy mapping kept is data-level: `rrp.harness.data.load_pickle` remaps old module paths inside
  already-written dataset pickles (e.g. `rrp.data.features.PolicyInput`), because those files are on disk and are not code.
- One module per concept; a new file needs a reason that an existing module cannot absorb it.
- Optional backends are optional extras: `rrp.envs.simple` and `rrp.envs.computerworld` import their third-party
  packages lazily inside the factory, so `import rrp.envs` never needs Isaac/SIMPLE or ComputerWorld.
- `rrp.core` is the only thing an external package may treat as stable. psi1z is retired (D-140), so there is no
  pinned consumer any more and no `CORE_API_VERSION` ceremony; `research/decisions.md` still records on-disk format changes.

## 2. `Env` (rrp.envs.base)

One interface for every simulator. An environment is a world with one or more bodies that accepts native commands at a
declared rate and produces public observations; privileged truth is a separate, capability-gated method.

```python
ActionKind = Literal["joint_position", "joint_velocity", "joint_torque", "base_velocity",
                     "cartesian_position", "gripper", "button", "discrete"]
Capability = Literal["privileged_truth", "snapshot", "render", "task_graph", "chunk_executor",
                     "reward", "deterministic", "batched", "images", "language", "object_descriptors",
                     "proprio", "vector_obs"]

class ActionSpace(Strict):
    group: str                      # command group name, e.g. "arm", "gripper", "legs", "base_velocity", "pointer", "key"
    kind: ActionKind
    width: int
    robot: int = 0                  # body index in multi-body envs (dual arm: 0, 1)
    rate_hz: float                  # the rate at which step() consumes this group
    low: list[float] | None = None
    high: list[float] | None = None
    units: str = ""                 # "rad", "m", "m/s", "px", "index"
    vocab: list[str] | None = None  # discrete spaces only: value i means vocab[i]; -1 = no event

class BodyInfo(Strict):
    robot: int
    family: str                     # "arm", "legged", "humanoid", "g1_hands", "pointer"
    key: str                        # catalog key ("panda", "go2", "phum_3", "g1_simple", "cw_pointer")
    robot_spec_hash: str            # RobotSpec content hash (morphology graph; stable identity)

class EnvSpec(Strict):
    env_id: str                     # registry key: "mujoco/arm", "mujoco/dual", "mujoco/legged", "warp/legged", "simple", "computerworld"
    backend: Literal["mujoco", "mujoco_warp", "mjx", "isaac_simple", "computerworld"]
    task: str                       # TaskSpec name the env was built for
    bodies: list[BodyInfo]
    batch: int = 1                  # >1 only for batched (vectorized) envs
    control_hz: float
    action_spaces: list[ActionSpace]
    capabilities: frozenset[Capability]
    frame: dict                     # {"units": "m", "up": "+z", ...}; ComputerWorld adds its screen mapping (section 5)
    provenance: dict                # physics/contact/actuator versions, tracker source, asset digests

class StepResult(Strict):
    observation: Observation
    time: float
    rejected: str | None = None      # controller rejection code (stale chunk, bad group, ...), never silent
    executed: dict[int, dict[str, list[float]]] = {}   # robot -> groups actually executed (exact replay)
    reward: float | None = None      # only with capability "reward" (training envs); privileged

class Env(Protocol):
    spec: EnvSpec
    def reset(self, seed: int | None = None) -> Observation: ...
    def observe(self) -> Observation: ...
    def step(self, command: NativeCommand | Mapping[int, NativeCommand] | None) -> StepResult: ...
    #   None = the env's declared hold/fallback (or the next queued chunk row with "chunk_executor")
    def close(self) -> None: ...
    # capability-gated (raise CapabilityError when absent):
    def truth(self) -> PrivilegedTruth: ...                       # "privileged_truth"
    def snapshot(self) -> Snapshot: ...; def restore(self, s) -> Observation: ...   # "snapshot"
    def submit_chunk(self, chunk: ActionChunk, robot=0, execute_prefix=None): ...   # "chunk_executor"
    def render(self, camera: str | None = None, *, width=320, height=240) -> np.ndarray: ...  # "render"

def make_env(env_id: str, *, task: str, body: str | list[str], seed: int = 0, **kw) -> Env   # registry, lazy imports
```

Observation (`rrp.core.Observation`, today `PolicyObservation`) stays the single public observation type for
single-world envs. It gains three optional fields so every env fits: `instruction: str | None` (language; SIMPLE,
ComputerWorld tasks), `ImageObs.encoding` adds `"rgba8"`, and `ObjectDescriptor.attributes: dict[str, str]` (e.g.
widget role/label/value/state). `measured_node_state` stays required: every body has a joint space, including the
ComputerWorld pointer (section 5). Batched envs (`capability "batched"`) return `VectorObservation(vec[N, D], layout:
list[(name, width)], time, robot_spec_hashes)` from `observe()` and take `NativeCommand` groups whose values are `[N, width]`;
their privileged state is `truth()` (never concatenated into `vec`: the current WarpSteps/Gap height-scan
concatenation becomes an explicitly privileged `layout` block that only policies with `requires.privileged` may consume).

Implementations after the refactor:

| env_id | class | bodies | action spaces | notes |
|---|---|---|---|---|
| `mujoco/arm` | `envs.mujoco.Session` | arm catalog (panda, ur, procedural ...) | `arm` joint_position, `gripper` | 20 Hz, task graph, chunk executor, snapshot |
| `mujoco/dual` | `envs.mujoco.DualSession` | 2 arms / aloha | `r0:arm`, `r0:gripper`, `r1:...` | multi-robot command dict |
| `mujoco/legged` | `envs.mujoco.LeggedSession` (+ LocoPick, Foothold scenes) | legged + humanoids | `base_velocity` (10 Hz, embedded tracker), `legs` joint_position (50 Hz, no tracker) | the latent/BC legged policies use `legs` directly instead of replacing the tracker slot |
| `warp/legged` | `envs.warp.TrackerEnv` (+ steps/gap task variants, morph-multi) | legged + humanoids | `legs` residual joint_position, batched | tracker PPO; `reward` capability |
| `simple` | `envs.simple.SimpleEnv` (Ψ₀ agent) | `g1_simple` | `psi0` (36-d Ψ₀ command, 50 Hz; decoupled WBC inside) | Isaac Sim in its own venv behind an RPC boundary |
| `computerworld` | `envs.computerworld.ComputerWorldEnv` (CW agent) | `cw_pointer` | `pointer` cartesian_position, `button`, `wheel`, `key` discrete | section 5 |

The prototypes `mjx_legged.MjxLegged` and `warp_legged.WarpLegged` (bake-off only) and the CPU `LeggedEnv`/`VecPool`
duplicate of the Warp tracker env are deleted; `tracker_validation` becomes a hook-based evaluation on `mujoco/legged`
with the `legs` space driven by the tracker policy.

## 3. `Policy` (rrp.policies.base)

A policy maps observations to native commands for one control tick, batched over parallel episodes. What happens
inside (plan a chunk every k ticks, generate a packet and realize it every tick, run a teacher) is the policy's business;
what it emits is logged through `Act`.

```python
ObsField = Literal["proprio", "images", "object_descriptors", "predicates", "task_graph", "language", "vector"]

@dataclass(frozen=True)
class Requirements:
    action_kinds: frozenset[ActionKind]           # every kind the policy emits must be offered by the env
    groups: frozenset[str] = frozenset()          # required group names ("arm", "gripper"); empty = any group of those kinds
    observations: frozenset[ObsField] = frozenset({"proprio"})
    body_families: frozenset[str] | None = None   # None = any family whose action spaces match
    bodies: frozenset[str] | None = None          # checkpoint trained on specific morphologies (robot keys), else None
    tasks: frozenset[str] | None = None           # trained for specific tasks, else None
    privileged: bool = False                      # teachers/oracles: needs env.truth()/snapshot; result is labelled
    env_capabilities: frozenset[Capability] = frozenset()

@dataclass(frozen=True)
class PolicyInfo:
    name: str                  # registry key: "bc", "latent", "legged_latent", "legged_bc", "tracker", "teacher:<task>", "oracle", "psi0_direct", "psi0_structured"
    source: Source             # rrp.core.provenance: scripted_teacher | privileged | oracle | learned:<ckpt> | bc | random | mock
    version: str               # weights digest / bundle versions (compatibility IDs)
    requires: Requirements
    variant: str | None = None # "semfix", "nosem", "sem" for the latent family

@dataclass
class Act:
    command: NativeCommand | dict[int, NativeCommand] | None   # applied this tick (None = env hold / queued chunk)
    packet: LatentActionChunk | None = None    # emitted this tick (system i), after any hook edit
    chunk: ActionChunk | None = None           # submitted this tick (chunk policies)
    info: dict = field(default_factory=dict)   # latencies, rejections, probe outputs, SDE records ...

class Policy(Protocol):
    info: PolicyInfo
    def reset(self, spec: EnvSpec, task: TaskSpec, seeds: Sequence[int], *,
              envs: Sequence[Env] | None = None) -> None: ...
    #   envs is passed ONLY when info.requires.privileged (teachers, oracles); learned policies never see it.
    def act(self, obs: Sequence[Observation]) -> list[Act]: ...
    #   hooks (packet edits, noise keys) are attributes set by the harness: policy.packet_hook(i, packet) -> packet

def negotiate(info: PolicyInfo, spec: EnvSpec, task: TaskSpec) -> Compat      # Compat(ok, reasons: list[str])
def make_policy(name: str, **kw) -> Policy                                   # registry, lazy imports
```

`negotiate` is the capability check the harness runs before any episode; a declined pair is recorded as
`n/a` with its reasons, never silently skipped. Examples:
- `latent` (arm system i + `LatentSystem0`) requires `joint_position` + `gripper`, body family `arm`, `object_descriptors`
  and `task_graph`: accepted by `mujoco/arm` on any arm; declined by `computerworld` ("needs joint_position; env
  offers cartesian_position, button, discrete"), by `mujoco/legged` ("body family legged"), by `simple` ("needs
  joint_position; env offers psi0").
- `legged_latent` requires `legs` joint_position at 50 Hz, family legged/humanoid: accepted by `mujoco/legged`.
- `teacher:pick_place` requires `privileged` and task `pick_place`: declined by any env without `privileged_truth`.
- `psi0_direct` requires the `psi0` space, `images` + `language`, body `g1_simple`: SIMPLE only.
- A pointer-space policy (e.g. BC on ComputerWorld) requires `cartesian_position` + `button`: accepted by
  `computerworld`; also by `mujoco/arm` once that env exposes its IK `tcp` cartesian space (planned, not in this refactor).

Mapping of today's code onto the interface (the adapters are thin; the nets and featurizers do not change):

| policy | today | after |
|---|---|---|
| `bc` | `controllers.policy_runner.LearnedPolicy` (+ SDEPolicy/Expo/VLM variants in training) | `policies.bc.BCPolicy` (chunk policy; `Act.chunk`) |
| `latent` (arm, dual) | `controllers.latent_runner.LatentPolicy` + `LatentSystem0` / `batched_ticks`; `DualLatentPolicy` + `DualLatentSystem0` in evaluation | `policies.latent.LatentPolicy(planner, system0)`; dual is the same class with the multi featurizer |
| `legged_latent`, `legged_bc` | `LatentLeggedController` / `BCController` hidden in the tracker slot via `System0Adapter`/`BCAdapter` | `policies.latent.LeggedLatentPolicy`, `policies.bc.LeggedBCPolicy` on the `legs` space |
| `tracker` | `envs.legged_tracker.LearnedTracker` / `CPGTracker` | stays embedded in `mujoco/legged`'s `base_velocity` space; `policies.trackers.TrackerPolicy` wraps the same object for `warp/legged` and validation |
| `teacher:<task>` | `teachers/*` `act()` (reads session internals) | `policies.teachers`, `requires.privileged`, source `scripted_teacher` |
| `oracle` | `OraclePacketPolicy`, `OracleSource`, `OracleShadow` (three copies) | `policies.oracle.OraclePolicy` (teacher look-ahead → encoder → packet → system 0), source `oracle` |
| `psi0_direct`, `psi0_structured` | psi1z `serve_psi0`/`serve_ours` + `system_i`/`structured` | `policies.psi0` (Ψ₀ agent) |

## 4. `Task` (rrp.tasks)

```python
@dataclass(frozen=True)
class Judgement:
    done: bool
    outcome: Literal["success", "failure", "timeout", "fell", "infeasible", "rejected", "crash"] | None
    failure_reason: str | None      # family-specific code: "dropped_off_table", "drift_a", "halt", "stall", ...
    success_public: bool | None     # from public estimators
    success_privileged: bool | None # from env.truth(); reported, never fed back to the policy

@dataclass(frozen=True)
class TaskSpec:
    name: str                               # "pick_place", "waypoint_contact", "h_gap_sidestep", "simple/TabletopGraspMP", "cw/<task>"
    definition: TaskDefinition | None       # task graph (tasks/*.json); None for tasks without a graph (SIMPLE success flag)
    envs: Mapping[str, dict]                # env_id -> scene/builder kwargs; the task exists only in these envs
    teacher: str | None                     # policy registry key of its scripted teacher
    max_seconds: float
    judge: Callable[[Env, Observation, StepResult | None, float], Judgement]
    gates: Mapping[str, Any] = {}           # task-specific acceptance thresholds (harness.gates reads them)

TASKS: dict[str, TaskSpec]; def get_task(name) -> TaskSpec
```

Scene builders (MuJoCo scenario builders) stay in the env implementation that owns them (`envs.mujoco.scenes`);
`TaskSpec.envs` refers to them by key. Teachers are policies (section 3), referenced by key, so `rrp.tasks` stays
below `rrp.envs`.

## 5. ComputerWorld as a 3D environment

ComputerWorld (github.com/JacobFV/computerworld, MIT; Rust engine, PyO3 wheel `computerworld`) exposes a scene
(flat list of nodes: bounds in px, semantic role/label/value/state, `z` order, window, interaction string), RGBA frames
(`env.render(w, h)`), and pointer/keyboard/application actions. The env adapter maps it into our 3D conventions:

- **Frame.** Screen of W × H px. World frame: x right, y up, z toward the viewer, origin at the screen centre, metric
  scale `s = spec.frame["m_per_px"]` (default 0.001: 1 px = 1 mm). Pixel (u, v) ↦ (x, y) = ((u − W/2)·s, (H/2 − v)·s).
- **Depth.** `spec.frame["depth"] = "constant"` puts every widget on the plane z = 0. `"stack"` sets z from the scene
  order: z = rank(node.z, window stacking, node order) · `dz` (default 0.002 m), so occluding widgets are above occluded
  ones; a node fully covered by a higher one is reported `visible=False` (it stays in the list: null identities and
  occlusion are part of the representation).
- **Widgets → ObjectDescriptor.** slot = stable index from the node id (ids are stable across revisions; new ids are
  appended, removed ids leave a null slot), `descriptor = role`, `attributes = {label, value, state, disabled, window,
  interaction}`, `bbox_xyxy` in px, `position_estimate = [x, y, z]` of the bounds centre, `visible`. Widgets are the
  entities the task graph binds (e.g. "click the Save button" binds entity `button:Save`).
- **Body `cw_pointer`** (RobotSpec, family `pointer`): one base link at the screen origin, slide joints `x`, `y`
  (range = viewport) and a slide `z` fixed at the hover height; one assembly (`tool`). `measured_node_state.qpos` =
  pointer (x, y, z). Buttons are a `gripper`-like binary group.
- **Action spaces.** `pointer` cartesian_position width 2 (absolute x, y in m; the env converts to px and emits
  `pointer.v1 move`); `button` width 1 (≥ 0.5 = down; edges emit `down`/`up`; click = down then up on consecutive ticks);
  `wheel` width 1 (notches); `key` discrete width 1 with `vocab` = the key names plus printable characters (text is typed
  one symbol per tick; -1 = none). Application launch/focus are task setup, not policy actions.
- **Observation.** `sensor_images=[ImageObs(camera="screen", encoding="rgba8", ...)]` when `pixels.v1` is granted;
  `object_descriptors` from `semantic.v1`; `instruction` from the task; `task_input` if the task has a graph.
- **Privileged truth / reset.** `truth()` returns the full scene (including occluded nodes' semantics) and task
  predicates; reset = `world.restore(initial_snapshot)` (deterministic), `snapshot()` = `world.snapshot()`.
- Capabilities: `privileged_truth`, `snapshot`, `render`, `images`, `language`, `object_descriptors`, `proprio`,
  `task_graph` (when the task has one), `deterministic`.

## 6. harness

```python
def rollout(make_env: Callable[[int], Env], policy: Policy, task: TaskSpec, seeds: Sequence[int], *,
            batch: int = 8, max_seconds: float | None = None, hooks: Sequence[Hook] = ()) -> list[Episode]

class Hook(Protocol):                         # every current special case of the eval loops is a hook
    def on_reset(self, i: int, env: Env, obs: Observation) -> None: ...
    def on_act(self, i: int, obs: Observation, act: Act) -> Act: ...       # packet/chunk edits, perturbation of commands
    def on_step(self, i: int, env: Env, act: Act, step: StepResult) -> None: ...  # recorders, perturbations, DAgger collection
    def on_end(self, i: int, env: Env, ep: Episode) -> dict: ...           # metrics merged into Episode.metrics

@dataclass
class Episode:
    seed: int; task: str; env_id: str; body: str; policy: str; source: str
    outcome: str; failure_reason: str | None; success_public: bool | None; success_privileged: bool | None
    steps: int; time: float; metrics: dict; provenance: dict

def evaluate(policy_spec, env_id, task, bodies, seeds, *, out: Path, hooks=()) -> dict   # JSONL rows + Wilson summary
def matrix(policies, envs, tasks, bodies, seeds, *, out) -> dict                         # negotiate() every cell; n/a with reasons
```

Hooks replace the special cases of the current loops: `Meter`/stage reached, label error vs shadow teacher, oracle
comparison, DAgger collection, command logs, frame capture/video, perturbations and robustness grids, motion quality,
contact metrics, deploy OOD/safety/latency, probe readouts, packet edits (`swap_slots`, probe-guided, zero/freeze,
context transforms), noise keys for paired edits. Training (`harness.train`) keeps the existing dataset trainers;
DAgger, GRPO and adaptation call `rollout`.

CLI (`rrp`): `ops …` (unchanged), `data {collect,pack}`, `train {rep,flow,bc,refit,probes,sft,grpo,ppo}`,
`eval`, `matrix`, `edits`, `run-dag`, `workbench`, `task validate`, `assets validate`. Per-module `python -m` entry points are
removed except `rrp.viz.export` and `rrp.viz.record` (room exporter/recorder).

## 7. delete list

| what | why |
|---|---|
| shim packages `control/`, `learning/`, `model/`, `morphology/`, `ops/` (old alias of orchestration), `policy/`, `sim/`, top-level `cli_*.py`, `data/features*.py`, evaluation re-export shims (`campaign`, `latent_campaign`, `baseline_campaign`, `system2_eval`, `bc_semantic_edits`, `latent_slice1_report`, `contact_metrics`, `motion_quality`), `research/system2.py` | W4 aliases; owner rule: no shims |
| `src/rrp/research/` (bc_semantic_edits, latent_slice1_report, legged_t1_diag, qa_train, system2_eval) | finished one-offs; nothing live imports them; git keeps them |
| `envs/mjx_legged.py` `MjxLegged`, `envs/warp_legged.py` (bake-off prototypes; `build_model`/`default_data` helpers move into `envs.warp`), `envs/legged_core.LeggedEnv` + `legged_vec.VecPool` + `training/tracker_training.py` CPU PPO (superseded by Warp PPO; keep `LeggedBinding` obs/reward math which the Warp env and trackers share) | duplicate env implementations |
| duplicate eval loops: `evaluation/runner.evaluate`, `latent_eval.evaluate_latent`, `dual_latent_eval.evaluate_dual_latent`, `training/rollout.drive`, `latent_grpo.run_episodes`, `latent_semantic_edits.run_condition`/`run_arm_condition` bodies, `legged_latent_eval.run_episode` special cases | replaced by `harness.rollout` + hooks |
| duplicate oracles (`ladder.OraclePacketPolicy`, `latent_semantic_edits.OracleSource`, `legged_latent_eval.OracleShadow`), duplicate `wilson` in ladder, duplicate teacher wrappers in evaluation (`ShadowTeacher`, `ShiftedGoalTeacher`, `TeacherSource`, `_DualTeacher`) | one implementation each |
| `tests/unit/test_restructure_compat.py` (shim tests), the `SHIMS`/`PLANNED` tables of `test_layering.py`, `tests/data/legacy_scripts/` once the parity tests compare against goldens | shims gone |
| `scripts/`: 107 experiment chain drivers (`*_chain.sh`, `arm*_`, `legged_*.sh`, `t1_diag_*`, `binding_chain_v*`, `w12/*.sh`, …), 69 analysis one-offs (`diag_*`, `*_compare.py`, `*_parity.py`, `humanoid_*.py`, `dev/`, `analysis/`, …) | finished experiments; git history keeps them; resume steps in track notes cite commits |
| `scripts/` kept: `peer_sync.sh`, `peer_run.sh`, `peer_bootstrap.sh`, `peer_gpu_retry.sh`, `fetch_menagerie.sh`, `export_ui_types.py`; `ladder.py` and `legged_{ladder_summary,edit_effects,mirror_effect}.py` fold into `rrp eval`/`rrp edits`; `demo/` + `render_*.py` fold into `rrp.viz.record` specs (kept until the fold lands) | infrastructure |
| `dags/`: keep templates, overlays and the DAGs tests use (`arm_lineage*`, `legged_v2_go2`, `legged_fixrep`, `armexpert_v6dart`, `arm_targets_*`); delete the unreferenced ones | provenance of completed lineages stays in git |
| `configs/`: delete directories nothing references (`vlm/`, `adapt/`, `t1_diag/`, `legged_latent/` script-only); keep what dags/pipelines/tests read | – |
| docs: `docs/core_api.md`, `docs/repo_structure_audit.md` (this page), README module paths | superseded |

## 8. staged migration (each stage: `pytest tests/unit` green with exit code checked, then merged to main)

Verification is by golden tests, not reruns (D-140): `tests/unit/test_golden.py` records SHA-256 digests of the
outputs of fixed, seeded inputs before any move (arm featurizer on a procedural-arm pick_place reset; a teacher's first
commands and resulting qpos; a seeded-random latent planner packet z and realizer output; legged public context and
realizer output; dual multi-featurizer; packet serialization) and must stay byte-identical through every stage.

| stage | content | est. | status |
|---|---|---|---|
| S0 | this document; golden digests on the pre-refactor code | 2 h | – |
| S1 | delete shim packages and `research/`; rewrite importers in src/tests/scripts; layering test without shim tables; pickle remap in the data loader | 1.5 h | – |
| S2 | re-layer into `core / ops / bodies / tasks / envs / policies / harness / viz` (codemod on module paths, merge tiny modules, new layering table) | 3 h | – |
| S3 | `envs.base` (Env, EnvSpec, capabilities, registry) implemented by the MuJoCo sessions and the Warp env; `policies.base` (Policy, Requirements, negotiate, registry); `tasks` registry; SIMPLE/ComputerWorld stubs → **Ψ₀ and ComputerWorld agents can start here** | 3 h | – |
| S4 | policy adapters (bc, latent arm/dual, legged latent/bc on the `legs` space, teachers, one oracle, trackers); delete duplicate envs | 4 h | – |
| S5 | `harness.rollout` + hooks; port the eval loops, delete the duplicates; `rrp eval` / `rrp matrix`; golden traces of tiny episodes (a few ticks, procedural bodies) | 5 h | – |
| S6 | scripts/dags/configs pruning, CLI consolidation, README/STATUS/related_repos, viz exporter check | 2 h | – |

Running work is paused (D-140); unmerged track branches rebase onto the new paths using the move table that each
stage's commit message and section 10 record.

## 9. hand-off for the Ψ₀ and ComputerWorld agents (after S3 is on main)

Ψ₀ migration (psi1z → rrp):
- [ ] `rrp/bodies`: G1 + Dex3 36-d command morphology from `psi1z/body_g1.py` as a `RobotSpec` + the static tables (`node_static`, `relation_matrix`, `asm_static`); key `g1_simple`.
- [ ] `rrp/envs/simple.py`: `SimpleEnv(Env)`: runs SIMPLE (Isaac Sim 5.1, its own venv) behind a local RPC boundary (127.0.0.1); `psi1z.simple_compat` import hooks move with it; `reset(seed)` = eval episode config (task, DR level, index) + WBC stabilization; action space `psi0` width 36 at 50 Hz (the decoupled WBC stays inside the env); observation = head RGB image, 43-d `joint_qpos` as `measured_node_state`, `instruction`; privileged truth = palm/object poses and contacts (labels only); success from SIMPLE's `_success`.
- [ ] `rrp/policies/psi0/`: `psi0_direct` (upstream Ψ₀ action head over the 30×36 chunk) and `psi0_structured` (packet z[5, 6, 64] + realizer), both `Policy`s emitting `Act.chunk`/`Act.packet`; features cache, `train.py` → `rrp train` stages; checkpoints stay outside git.
- [ ] `rrp/tasks`: `simple/<Task>` TaskSpecs (TabletopGraspMP, BendPickMP, HandoverTeleop, ...), judge = SIMPLE success.
- [ ] fold P-xxx decisions into `research/decisions.md` keeping their P-numbers; archive psi1z read-only with a pointer here.

ComputerWorld:
- [ ] optional extra `computerworld` in pyproject (wheel built from github.com/JacobFV/computerworld; local clone at ~/Documents/computerworld).
- [ ] `rrp/envs/computerworld.py`: `ComputerWorldEnv(Env)` exactly as section 5; unit tests of the px ↔ metre mapping, depth modes, null slots, button edges.
- [ ] `rrp/bodies`: `cw_pointer` RobotSpec.
- [ ] `rrp/tasks`: a few `cw/*` tasks with a task graph (open app, click labelled widget, type into field) and a scripted teacher in `rrp/policies/teachers` (source `scripted_teacher`, privileged scene access).
- [ ] demonstrate negotiation: `rrp matrix` shows arm/legged policies declined with reasons and a pointer BC trained on teacher data accepted.

## 10. moved paths

Filled in by S1/S2 (old module → new module), so track branches and notes can be ported.
