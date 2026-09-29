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
| 0 | `rrp.core` | contracts (was `contracts/`): arrays/Strict base, `Observation` (= `PolicyObservation`), `PrivilegedTruth`, `NativeCommand`, `ActionChunk`, `LatentActionChunk` (packet z[K,M,D] + assembly handles/mask + entity registry), `System0Base`, errors, `RobotSpec` (morphology graph), `TaskDefinition`, provenance + `Source` labels, `RunConfig`, paths, run/workload helpers, public/privileged channels, Ψ action contracts | numpy, pydantic |
| 0 | `rrp.ops` | broker, leases, cgroups, watchdog, jobs, telemetry, discovery, workload (GPU cap) (was `orchestration`) | – |
| 1 | `rrp.bodies` | procedural + imported bodies (arms, grippers, aloha, legged, humanoids, G1 hands), catalog/registry, compiler to MuJoCo, IK, surgery/variants, **physics versions** (contact, grasp contact, actuator; was `physics/`) | mujoco |
| 2 | `rrp.tasks` | task graph compiler/runtime/receipts/interventions (numpy/pydantic), task JSON specs, the `TaskSpec` registry (which envs a task exists in, success/termination, scripted teacher key, gates) | – |
| 3 | `rrp.envs` | `Env` protocol, `EnvSpec`, capabilities, `make_env` registry; `mujoco/` (Session, LeggedSession, DualSession, scenes, sensors, state estimation, perturbations, embedded legged trackers, snapshots), `warp/` (batched GPU legged envs), `simple/` (optional extra), `computerworld/` (optional extra) | mujoco, mujoco_warp, torch (trackers) |
| 4 | `rrp.policies` | `Policy` protocol, `PolicyInfo`, `Requirements`, `negotiate`, registry; `features/` (featurizers: the ONLY definition of what a policy may see), `nets/` (shared torch modules: attention, flow, codec, backbone, probes, checkpoint), `bc.py`, `latent/` (system i planners + system 0 realizers for arm, dual, legged), `trackers.py`, `teachers/` (scripted / privileged, labelled), `oracle.py`, `psi0/` (Ψ₀ direct / Ψ₀ + structure / demo replay, their nets, feature cache and training) | torch |
| 5 | `rrp.harness` | `rollout` (the one episode loop), `evaluate`/`matrix`, hooks (packet edits, perturbations, recorders), statistics, gates, audits; `data/` (collect, pack, manifests), `train/` (rep, flow, bc, refit, dagger, sft, grpo, ppo), pipelines + run-dag | – |
| 6 | `rrp.viz` | record/replay, the room exporter (`python -m rrp.viz.export`, file scans only), the workbench service `viz.workbench` (was `service/`) | fastapi (extra) |
| 7 | `rrp.cli` | the `rrp` command (`python -m rrp.cli ...`; kept at the top so every documented invocation stays valid) | – |

Rules:
- No compatibility shims or deprecated aliases. A moved name is imported from its new path everywhere in the same
  commit. The only legacy mapping kept is data-level: `rrp.harness.data.collect.load_pickle` remaps old module paths inside
  already-written dataset pickles (e.g. `rrp.data.features.PolicyInput`), because those files are on disk and are not code.
- One module per concept; a new file needs a reason that an existing module cannot absorb it.
- Optional backends are optional extras: `rrp.envs.simple` and `rrp.envs.computerworld` import their third-party
  packages lazily inside the factory, so `import rrp.envs` never needs Isaac/SIMPLE or ComputerWorld.
- `rrp.core` is the only thing an external package may treat as stable. psi1z is retired (D-140), so there is no
  pinned consumer any more and no `CORE_API_VERSION` ceremony; `research/decisions.md` still records on-disk format changes.

## 2. `Env` (rrp.envs.base) — implemented in S3

One interface for every simulator. An environment is a world with one or more bodies that accepts native commands at
declared rates and produces public observations; privileged truth is a separate, capability-gated method.
`rrp/envs/base.py` is the authoritative code; this is its shape:

```python
ActionKind = Literal["joint_position", "joint_velocity", "joint_torque", "gripper", "base_velocity", "wholebody_command",
                     "ee_pose", "cartesian_position", "button", "discrete", "psi0"]  # == rrp.core.robot.CommandGroup.semantic
Capability = Literal["privileged_truth", "snapshot", "render", "task_graph", "chunk_executor", "reward", "deterministic",
                     "batched", "images", "language", "object_descriptors", "predicates", "proprio", "vector_obs"]

class ActionSpace(Strict):            # ActionSpace.from_group(CommandGroup, robot=, rate_hz=)
    group: str; kind: ActionKind; width: int; robot: int = 0; rate_hz: float
    low: list[float] | None = None; high: list[float] | None = None; units: str = ""
    vocab: list[str] | None = None    # discrete spaces: value i means vocab[i]; -1 = no event

class BodyInfo(Strict):               # BodyInfo.from_spec(robot, RobotSpec, key)
    robot: int; family: str; key: str; robot_spec_hash: str     # family = RobotSpec.family ("arm", "quadruped", "humanoid", ...)

class EnvSpec(Strict):
    env_id: str; backend: Literal["mujoco", "mujoco_warp", "mjx", "isaac_simple", "computerworld"]; task: str
    bodies: list[BodyInfo]; batch: int = 1; control_hz: float
    action_spaces: list[ActionSpace]; capabilities: list[Capability]
    frame: dict = {"units": "m", "up": "+z"}   # ComputerWorld adds its screen mapping (section 5)
    provenance: dict = {}                       # physics settings, controller/tracker versions, asset digests
    def has(cap) -> bool; def action_kinds() -> set[str]; def space(group, robot=0) -> ActionSpace

@dataclass
class StepResult:                     # (moved here from the MuJoCo session; same fields)
    observation: Observation; qpos: np.ndarray | None; time: float
    rejected: str | None = None       # controller rejection code, never silent
    source: str | None = None; command: dict | None = None; commands: dict | None = None   # executed groups (replay)
    reward: Any = None                # capability "reward" only; privileged, never a policy input

@dataclass
class VectorObservation:              # batched envs: vec [N, D] (numpy or torch), named layout
    vec: Any; layout: list[tuple[str, int]]; time: float; robot_spec_hashes: list[str]

@dataclass
class BatchCommand:                   # batched envs: group -> [N, width] array/tensor
    groups: dict[str, Any]; source: str

Observation = PolicyObservation | VectorObservation

class Env(Protocol):
    spec: EnvSpec                                                      # property
    def reset(self, seed: int | None = None) -> Observation
    def observe(self) -> Observation
    def step(self, command: NativeCommand | Mapping[int, NativeCommand] | BatchCommand | None) -> StepResult
    #   None = the env's declared hold/fallback (or the next queued chunk row with "chunk_executor")
    def close(self) -> None
    # capability-gated (CapabilityError when absent): truth() ["privileged_truth"]; snapshot()/restore(s) ["snapshot"];
    # submit_chunk(chunk, robot=0, execute_prefix=None) ["chunk_executor"]; render(camera=None, width=, height=) ["render"]

ENVS: dict[str, str]                  # env_id -> "module:factory" (lazy); register_env(env_id, target)
def make_env(env_id, *, task: str, body: str | list[str], seed: int = 0, **kw) -> Env
#   factories return an env already reset to `seed`; kw: scene={...} builder kwargs + env constructor kwargs;
#   a declared env whose module does not exist yet raises NotImplementedError naming this document
```

Observation (`rrp.core.observation.PolicyObservation`) stays the single public observation type for single-world envs.
S3 added three optional fields so every env fits: `instruction: str | None` (language; SIMPLE, ComputerWorld tasks),
`ImageObs.encoding` `"rgba8"`, and `ObjectDescriptor.attributes: dict[str, str]` (e.g. widget role/label/value/state).
`measured_node_state` stays required: every body has a joint space, including the ComputerWorld pointer (section 5).
`CommandGroup.semantic` gained `cartesian_position`, `button`, `discrete` (units `index`) and `psi0` so a UI body declares its
command groups in its RobotSpec like any robot. Batched envs return `VectorObservation` and take `BatchCommand`; their
privileged state is `truth()`, never concatenated into `vec` (the WarpSteps/Gap height-scan concatenation becomes an
explicitly privileged layout block in S4).

Implementations:

| env_id | class (factory) | bodies | action spaces | status |
|---|---|---|---|---|
| `mujoco/arm` | `envs.mujoco.session.Session` (`make_arm_env`: `BUILDERS[task]`, workbench robot key) | arm catalog | `arm` joint_position, `gripper` (20 Hz) | S3 |
| `mujoco/dual` | `envs.mujoco.dual.DualSession` (`make_dual_env`: pair key or list of keys) | 2 arms / aloha | per robot: `arm`, `gripper` | S3 |
| `mujoco/legged` | `envs.mujoco.legged.LeggedSession`, `LocoPickSession`, `FootholdSession` (`make_legged_env`: waypoint_contact, loco_pick, foothold_steps, h_steps, h_gap) | legged + humanoids | `base_velocity` (10 Hz, embedded tracker); `legs` joint_position (50 Hz, no tracker) | base_velocity S3; `legs` S4 |
| `warp/legged` | `envs.warp.tracker_env.WarpEnv` over WarpTrackerEnv / WarpStepsEnv / WarpGapEnv (`make_warp_env`) | legged + humanoids | `legs` normalized residual joint_position, batched | S3 (adapter; PPO keeps the engine API) |
| `simple` | `envs.simple` package (Ψ₀ agent: `envs/simple/{__init__,compat,worker}.py`, `bodies/g1_simple.py`, `policies/psi0/`) | `g1_simple` | `psi0` kind (36-d Ψ₀ command, 50 Hz; decoupled WBC inside) | declared |
| `computerworld` | `envs.computerworld` (CW agent) | `cw_pointer` | `pointer` cartesian_position, `button`, `wheel`, `key` discrete | declared |

S4 deletes the bake-off prototypes (`envs.warp.mjx_legged.MjxLegged`, `envs.warp.warp_legged.WarpLegged`; their
model-building helpers stay) and turns `tracker_validation` into a hook-based evaluation on `mujoco/legged`. The CPU
`LeggedEnv` stays for now: the D-126 physics goldens pin its reward/observation streams.

## 3. `Policy` (rrp.policies.base) — implemented in S3

A policy maps observations to native commands for one control tick, batched over parallel episodes. What happens
inside (plan a chunk every k ticks, generate a packet and realize it every tick, run a teacher) is the policy's business;
what it emitted is reported through `Act`.

```python
@dataclass(frozen=True)
class Requirements:
    action_kinds: frozenset[str]                  # every kind the policy emits must be offered by the env
    groups: frozenset[str] = frozenset()          # required group names; empty = any group of those kinds
    observations: frozenset[str] = {"proprio"}    # proprio | images | object_descriptors | predicates | task_graph | language | vector
    body_families: frozenset[str] | None = None   # None = any family whose action spaces match
    bodies: frozenset[str] | None = None          # checkpoint trained on specific body keys
    tasks: frozenset[str] | None = None           # trained for / scripted for specific tasks
    privileged: bool = False                      # teachers / oracles: env.truth() and internals; result labelled
    env_capabilities: frozenset[str] = frozenset()
    batched_env: bool = False                     # consumes VectorObservation batches

@dataclass(frozen=True)
class PolicyInfo:
    name: str        # "bc", "latent", "legged_latent", "legged_bc", "tracker", "teacher:<task>", "oracle", "psi0_direct", "psi0_structured"
    source: str      # rrp.core.provenance Source kind: scripted_teacher | privileged | oracle | learned | bc | random | mock ...
    version: str     # weights digest / bundle compatibility IDs / teacher version
    requires: Requirements
    variant: str | None = None   # "semfix" | "nosem" | "sem"

@dataclass
class Act:
    command: NativeCommand | dict[int, NativeCommand] | BatchCommand | None   # applied this tick
    packet: LatentActionChunk | None = None     # emitted this tick by system i (after any hook edit)
    chunk: ActionChunk | None = None            # submitted this tick (env capability "chunk_executor")
    info: dict = {}                             # e.g. {"execute_prefix": 8}, latencies, probe readouts, SDE records

class Policy(Protocol):
    info: PolicyInfo
    def reset(self, spec: EnvSpec, task: TaskSpec, seeds: Sequence[int], *, envs: Sequence[Env]) -> None
    #   one env per seed. A policy without requires.privileged reads only the public surface: observe() through its
    #   featurizer (rrp.policies.features: the ONLY definition of what a policy may see), the body model for public FK,
    #   controller / task-graph versions and the clock; privileged ones (teachers, oracles) may read truth
    def act(self, obs: Mapping[int, Observation]) -> dict[int, Act]
    #   RUNNING episodes only, keyed by episode index (position in reset's seeds): per-episode state and random
    #   streams never depend on which other episodes are still running

def negotiate(info: PolicyInfo, spec: EnvSpec, task: TaskSpec | str | None = None) -> Compat   # Compat(ok, reasons)
POLICIES: dict[str, str]              # name -> "module:factory"; a family "teacher:*" receives arg="<task>"
def make_policy(name: str, **kw) -> Policy; def register_policy(name, target)
```

`negotiate` is the capability check the harness runs before any episode (`rollout` raises `Incompatible(reasons)`;
`matrix` records `n/a` with the reasons). Reasons use the env's own vocabulary, e.g. for the arm latent policy on
ComputerWorld: "needs gripper, joint_position; env offers button, cartesian_position, discrete", "needs observation
'task_graph'", "body family pointer not in ['arm']" (tests/unit/test_interfaces.py). Examples:
- `latent` (arm system i + `LatentSystem0`) requires `joint_position` + `gripper`, family `arm`, `object_descriptors`
  and `task_graph`: accepted by `mujoco/arm`; declined by `computerworld`, `mujoco/legged` ("body family quadruped"),
  `simple` ("needs gripper, joint_position; env offers psi0").
- `legged_latent` requires `legs` joint_position, families legged/humanoid: accepted by `mujoco/legged` (after S4).
- `teacher:pick_place` requires `privileged` and task `pick_place`: declined by any env without `privileged_truth`.
- `tracker` requires a batched env for PPO (`warp/legged`); its deployment is the embedded `base_velocity` layer.
- `psi0_direct` requires the `psi0` action kind, `images` + `language`, body `g1_simple`: SIMPLE only.
- A pointer-space policy requires `cartesian_position` + `button`: accepted by `computerworld`.

Mapping of today's code onto the interface (S4; the nets and featurizers do not change):

| policy | today | after |
|---|---|---|
| `bc` | `policies.bc.LearnedPolicy` (+ SDEPolicy/Expo/VLM variants in harness.train) | `policies.bc.BCPolicy` / `make_bc` (chunk policy; `Act.chunk`) — S4 done, parity golden |
| `latent` (arm, dual) | `policies.latent.LatentPolicy` + `policies.system0.LatentSystem0` / `batched_ticks`; `DualLatentPolicy` + `DualLatentSystem0` (moved from harness.eval into policies.latent / policies.system0) | `policies.latent.LatentStackPolicy` / `make_latent` (system i every `replan_ticks`, system 0 every tick, `packet_hook` for edits); dual via `DualLatentPolicy` — S4 done, parity golden |
| `legged_latent`, `legged_bc` | `LatentLeggedController` / `BCController` in the tracker slot via `System0Adapter`/`BCAdapter` (harness.eval.legged_latent_eval) | `policies.latent` / `policies.bc` on the `legs` space |
| `tracker` | `envs.mujoco.legged_tracker.LearnedTracker` / `CPGTracker` | stays embedded in `mujoco/legged`'s `base_velocity` space; a Policy wrapper for `warp/legged` and validation; saved-actor options (clock_gate, target_margin, ref_ff, extra_obs_dim, morph_v1) pinned by test_golden |
| `teacher:<task>` | `policies.teachers.*` `act()` (reads session internals) | `policies.teachers.TeacherPolicy` (`make_policy("teacher:<task>")`), `requires.privileged`, source `scripted_teacher` — S4 done, parity golden (arm, dual) |
| `oracle` | `OraclePacketPolicy`, `OracleSource`, `OracleShadow` (three copies in harness.eval) | one `policies.oracle`, source `oracle` |
| `psi0_direct`, `psi0_structured` | psi1z `serve_psi0` / `serve_ours` + `system_i` / `structured` | `policies.psi0` (Ψ₀ agent; registered names already declared) |

## 4. `Task` (rrp.tasks.spec) — implemented in S3

```python
@dataclass(frozen=True)
class Judgement:
    done: bool
    outcome: Literal["success", "failure", "timeout", "fell", "infeasible", "rejected", "crash"] | None = None
    failure_reason: str | None = None       # family-specific code: "dropped_off_table", "timeout", "fell", ...
    success_public: bool | None = None      # observation-backed task graph / public estimators
    success_privileged: bool | None = None  # privileged truth; reported, never fed back to the policy

@dataclass(frozen=True)
class TaskSpec:
    name: str                          # "pick_place", "support_insert", "waypoint_contact", "h_steps", "simple/<Task>", "cw/<task>"
    envs: Mapping[str, dict]           # env_id -> scene kwargs; the task exists only in these envs
    max_seconds: float                 # the budget the existing evaluations use (arm 15 s, dual 40 s, legged 60 s, h_steps 40 s, h_gap 30 s)
    judge: Callable[[Env, float, float], Judgement]   # (env, elapsed s, budget s); duck-typed, may read privileged state
    graph: str | None = None           # tasks/<graph>.json
    teacher: str | None = None         # policy registry key of the scripted teacher
    gates: Mapping[str, Any] = {}
    note: str = ""

TASKS: dict[str, TaskSpec]; def get_task(name) -> TaskSpec; def register_task(TaskSpec)
def graph_judge(*, dropped_below_m=None, dropped_body="cube") -> Judge   # the arm runner's rule, generalized
```

Scene builders stay in the env implementation that owns them (`envs.mujoco.scenario.BUILDERS`, `dual_scenarios`,
`legged_scenes`, `humanoid_scenes`); the env factories select them by task name. Teachers are policies (section 3),
referenced by key, so `rrp.tasks` stays below `rrp.envs`. The legged fall/drift/halt outcomes of
`harness.eval.legged_latent_eval.run_episode` become the legged judge in S5 (until then legged tasks use `graph_judge`).

## 5. ComputerWorld as a 3D environment

ComputerWorld (github.com/JacobFV/computerworld, MIT; Rust engine, PyO3 wheel `computerworld`) exposes a scene
(flat list of nodes: bounds in px, semantic role/label/value/state, `z` order, window, interaction string), RGBA frames
(`env.render(w, h)`), and pointer/keyboard/application actions. The env adapter maps it into our 3D conventions:

- **Frame.** Screen of W × H px. World frame: x right, y up, z toward the viewer, origin at the screen centre, metric
  scale `s = spec.frame["m_per_px"]` (default 0.001: 1 px = 1 mm). Pixel (u, v) ↦ (x, y) = ((u − W/2)·s, (H/2 − v)·s).
- **Depth.** `spec.frame["depth"] = "constant"` puts every widget on the plane z = 0. `"stack"` sets z per z-LAYER:
  z = rank of the node's `z` value among the scene's distinct layers · `dz` (default 0.002 m); within a layer, draw order
  decides occlusion. A node fully covered by a later-drawn / higher node is reported `visible=False` (it stays in the
  list: null identities and occlusion are part of the representation).
- **Widgets → ObjectDescriptor.** ComputerWorld 0.2.0 node ids are NOT stable across layout changes, so slot identity
  is keyed by (interaction string, role, label): a key seen before keeps its slot, a new key is appended, a vanished key
  leaves a null slot. `descriptor = role`; `attributes = {role, label, value, disabled, focusable, focused, window,
  interaction}` (widgets expose role/label/value/disabled/focusable only; `focused` comes from the scene's focus record;
  there is no widget `state`); `bbox_xyxy` in px; `position_estimate = [x, y, z]` of the bounds centre; `visible`.
  Widgets are the entities the task graph binds (e.g. "click the Save button" binds entity `button:Save`).
- **Body `cw_pointer`** (RobotSpec, family `pointer`, added to `RobotSpec.family` in S3): one base link at the screen origin, slide joints `x`, `y`
  (range = viewport) and a slide `z` fixed at the hover height; one assembly (`tool`). `measured_node_state.qpos` =
  pointer (x, y, z). Buttons are a `gripper`-like binary group.
- **Action spaces.** `pointer` cartesian_position width 2 (absolute x, y in m; the env converts to px and emits
  `pointer.v1 move`); `button` width 1 (≥ 0.5 = down; edges emit `down`/`up`; click = down then up on consecutive ticks);
  `wheel` width 1 (notches); `key` discrete width 1 with `vocab` = the key names plus printable characters (text is typed
  one symbol per tick; -1 = none). Application launch/focus are task setup, not policy actions.
- **Observation.** The optional RGBA frame is `sensor_images=[ImageObs(camera="screen", encoding="rgba8", ...)]`
  (when `pixels.v1` is granted); `object_descriptors` from `semantic.v1`; the goal text in `instruction`; `task_input`
  if the task has a graph.
- **Privileged truth / reset.** `truth()` returns the full scene (including occluded nodes' semantics) and task
  predicates; reset = `world.restore(initial_snapshot)` (deterministic), `snapshot()` = `world.snapshot()`.
- Capabilities: `privileged_truth`, `snapshot`, `render`, `images`, `language`, `object_descriptors`, `proprio`,
  `task_graph` (when the task has one), `deterministic`.

## 6. harness

```python
# rrp.harness.rollout (S3)
def rollout(make_env: Callable[[int], Env], policy: Policy, task: TaskSpec, seeds: Sequence[int], *,
            batch: int = 8, max_seconds: float | None = None, hooks: Sequence[Hook] = ()) -> list[Episode]
#   lock-step batches; negotiate() first (Incompatible(reasons)); make_env(seed) returns a reset env; the policy sees
#   only running episodes; Act.chunk -> env.submit_chunk (stale/rejected chunks counted); Act.command -> env.step;
#   task.judge(env, t, budget) ends an episode; env/policy exceptions end episodes as outcome "crash" with a note

class Hook(Protocol):                         # every special case of the old loops is a hook; all methods optional
    def on_reset(self, i, env, obs) -> None
    def on_act(self, i, obs, act: Act) -> Act            # packet / chunk / command edits (chained in hook order)
    def on_step(self, i, env, act: Act, step: StepResult) -> None   # recorders, perturbations, DAgger collection
    def on_end(self, i, env, ep: Episode) -> dict        # merged into Episode.metrics

@dataclass
class Episode:
    seed; task; env_id; body; policy; source; outcome; failure_reason; success_public; success_privileged
    steps; time; wall_s; metrics: dict (command_rejections, chunk_rejections, hook metrics); provenance: dict
    def row(self) -> dict

# S5
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
| S0 | this document; golden digests on the pre-refactor code | 2 h | done (c14ca55, 44e1f3d, 58d65c5) |
| S1 | delete shim packages and `research/`; rewrite importers in src/tests/scripts; layering test without shim tables; pickle remap in the data loader | 1.5 h | done (a951398) |
| S2 | re-layer into `core / ops / bodies / tasks / envs / policies / harness / viz` (codemod on module paths, new layering table); S2b: no re-export aliases | 3 h | done (cb0ea23, ba2ce79) |
| S3 | `envs.base` (Env, EnvSpec, capabilities, registry) implemented by the MuJoCo sessions and the Warp env; `policies.base` (Policy, Requirements, negotiate, registry); `tasks` registry; SIMPLE/ComputerWorld declared in the registries → **Ψ₀ and ComputerWorld agents can start here** | 3 h | done (this commit) |
| S4 | policy adapters (bc, latent arm/dual, legged latent/bc on the `legs` space, teachers, one oracle, trackers); delete duplicate envs | 4 h | refactor lead (touches `policies/**`, `envs/**`) |
| S5 | port the eval loops onto `harness.rollout` (S3) + hooks, arm/dual first, legged last; delete the duplicates; legged judge; `rrp eval` / `rrp matrix`; golden traces of tiny episodes (a few ticks, procedural bodies) | 5 h | parallel agent (touches `harness/**`, `cli/**`; not `policies/**`) |
| S6 | scripts/dags/configs pruning, CLI consolidation, README/STATUS, viz exporter check | 2 h | – |

Running work is paused (D-140); unmerged track branches rebase onto the new paths using the move table that each
stage's commit message and section 10 record.

## 9. hand-off for the Ψ₀ and ComputerWorld agents (S3 is on main: start here)

Ψ₀ migration (psi1z → rrp; done, map in research/tracks/psi0.md):
- [x] `rrp/bodies/g1_simple.py`: the G1 + Dex3 36-d command morphology tables and `spec_hash()` (the G1 is simulated inside SIMPLE, so there is no compiled `RobotSpec`); key `g1_simple`.
- [x] `rrp/envs/simple/` (`make_env`): `SimpleEnv` drives a SIMPLE worker process (its own venv, Isaac Sim 5.1) over 127.0.0.1; the worker runs the unmodified upstream agent with an injected chunk client; `compat.py` holds the import hooks; action space `psi0` [36] at 50 Hz with capability `chunk_executor`; `chunk_request` / `psi0_state` channels; truth = palms, pelvis, objects, contacts (labels/judging only).
- [x] `rrp/policies/psi0/` (`make_direct`, `make_structured`, `make_replay`): policies over the upstream `Server` object (transforms, normalization, RTC); `nets.py`, `data.py` (feature cache, dataset, `LabelRecorder` hook), `train.py` (`python -m rrp.policies.psi0.train`; `rrp train` wiring with S5).
- [x] `rrp/tasks/spec.py`: `simple/<Task>` for the six benchmark tasks (`SIMPLE_TASKS`: released run, published rates, step-1 status), judge = SIMPLE `_success` via `env.truth()`.
- [x] P-001..P-023 folded as appendix P of `research/decisions.md`; notes in `research/tracks/psi0.md`; `docs/related_repos.md` deleted (glossary: section 11). psi1z archived read-only after owner approval.

ComputerWorld (track cworld, research/tracks/cworld.md):
- [x] optional extra `computerworld = ["computerworld==0.2.0"]` (PyPI abi3 wheel incl. aarch64; no Rust build).
- [x] `rrp/envs/computerworld.py` (`ComputerWorldEnv`, `make_env`), `tests/unit/test_computerworld.py` (mapping, depth
  modes, occlusion, null slots, button edges, negotiation; rollout/judge/determinism tests marked `computerworld`).
- [x] `rrp/bodies/fixtures.cw_pointer_spec` (family `pointer`).
- [x] `cw/calc_sum`, `cw/open_type`, `cw/drag_window`, `cw/fill_form` in `rrp/tasks/spec.py` (judges env-side: outcome +
  failure reason; no task graph yet) and `rrp/policies/teachers/computerworld.py` (`teacher:cw/*`, `scripted_teacher`,
  privileged).
- [ ] `rrp matrix` (S5) listing arm/legged policies declined with reasons; a pointer BC trained on teacher data accepted.

## 10. moved paths

S1 (a951398) deleted the W4 shim packages; their real targets are the "old" column below (e.g. `rrp.learning.latent_train`
was a shim of `rrp.training.latent_train`, now `rrp.harness.train.latent_train`; `rrp.sim.native` → `rrp.envs.native` →
`rrp.envs.mujoco.session`; `rrp.control.legged_tracker` → `rrp.envs.legged_tracker` → `rrp.envs.mujoco.legged_tracker`;
`rrp.morphology.*` → `rrp.bodies.*`; `rrp.ops.*` → `rrp.orchestration.*` → `rrp.ops.*`). `rrp.core` (the W11 re-export
module) is gone: import each name from its module. S2 moved packages as follows (a package row covers every module in it
unless a more specific row exists); module names inside packages are unchanged.

| old | new |
|---|---|
| `rrp.contracts` | `rrp.core` |
| `rrp.contracts.workload` | `rrp.ops.workload` |
| `rrp.controllers` | `rrp.policies` |
| `rrp.controllers.anchor_realizer` | `rrp.policies.system0_anchor` |
| `rrp.controllers.latent_realizer` | `rrp.policies.system0` |
| `rrp.controllers.latent_runner` | `rrp.policies.latent` |
| `rrp.controllers.policy_runner` | `rrp.policies.bc` |
| `rrp.data` | `rrp.harness.data` |
| `rrp.envs.dual` | `rrp.envs.mujoco.dual` |
| `rrp.envs.dual_scenarios` | `rrp.envs.mujoco.dual_scenarios` |
| `rrp.envs.fixtures` | `rrp.envs.mujoco.fixtures` |
| `rrp.envs.humanoid_scenes` | `rrp.envs.mujoco.humanoid_scenes` |
| `rrp.envs.joint_targets` | `rrp.envs.mujoco.joint_targets` |
| `rrp.envs.legged` | `rrp.envs.mujoco.legged` |
| `rrp.envs.legged_core` | `rrp.envs.mujoco.legged_core` |
| `rrp.envs.legged_scenes` | `rrp.envs.mujoco.legged_scenes` |
| `rrp.envs.legged_tracker` | `rrp.envs.mujoco.legged_tracker` |
| `rrp.envs.legged_vec` | `rrp.envs.mujoco.legged_vec` |
| `rrp.envs.mjx_legged` | `rrp.envs.warp.mjx_legged` |
| `rrp.envs.morph_obs` | `rrp.envs.mujoco.morph_obs` |
| `rrp.envs.motion_quality` | `rrp.envs.mujoco.motion_quality` |
| `rrp.envs.native` | `rrp.envs.mujoco.session` |
| `rrp.envs.perturb` | `rrp.envs.mujoco.perturb` |
| `rrp.envs.scenario` | `rrp.envs.mujoco.scenario` |
| `rrp.envs.sensors` | `rrp.envs.mujoco.sensors` |
| `rrp.envs.state_estimator` | `rrp.envs.mujoco.state_estimator` |
| `rrp.envs.tracker_nets` | `rrp.envs.mujoco.tracker_nets` |
| `rrp.envs.warp_legged` | `rrp.envs.warp.warp_legged` |
| `rrp.envs.warp_task_env` | `rrp.envs.warp.task_env` |
| `rrp.envs.warp_tracker_env` | `rrp.envs.warp.tracker_env` |
| `rrp.evaluation` | `rrp.harness.eval` |
| `rrp.features` | `rrp.policies.features` |
| `rrp.models` | `rrp.policies.nets` |
| `rrp.orchestration` | `rrp.ops` |
| `rrp.orchestration.dag` | `rrp.harness.dag` |
| `rrp.orchestration.yamlmini` | `rrp.harness.yamlmini` |
| `rrp.physics` | `rrp.bodies` |
| `rrp.physics.snapshot` | `rrp.envs.mujoco.snapshot` |
| `rrp.pipelines` | `rrp.harness.pipelines` |
| `rrp.service` | `rrp.viz.workbench` |
| `rrp.teachers` | `rrp.policies.teachers` |
| `rrp.training` | `rrp.harness.train` |

`python -m` entry points follow the table (e.g. `python -m rrp.training.warp_tracker_ppo` → `python -m
rrp.harness.train.warp_tracker_ppo`, `python -m rrp.pipelines` → `python -m rrp.harness.pipelines`); `python -m rrp.cli`
(and the `rrp` console script) is unchanged. Pipeline families, stage names (`collect`, `pack`, `train_rep`, `train_flow`,
`dagger`, `refit`, `eval_r1`, `eval_r2`, `heldout`, `edits`, ...), DAG files, configs and `artifacts/runs/<track>/...`
output paths are unchanged, so armdiv's resume (`dags/arm_lineage_v7div*.yaml`, `dags/armdiv_bc_v7div*.yaml`, ledgers
under `artifacts/runs/armdiv/_dags/`) and W13's resume work as written after `scripts/peer_sync.sh push`; their
scripts (`armdiv_chain.sh`, `armdiv_pack.sh`, `humanoid_{steps_eval,steps_eval_grid,gap_smoke,gap_eval,tracker_gate,tracker_finalize}`,
`contact_waypoint_eval.py`, `render_contact_compare.py`) were kept for that and are rewritten to the new module paths.
Other deleted scripts: `git show a951398^:scripts/<name>`.

Legacy on-disk data: dataset/DAgger pickles written before D-140 reference `rrp.data.features.PolicyInput`;
`rrp.harness.data.collect.load_pickle` (used by `read_episode` and the generator-DAgger loader) remaps it. Checkpoints
store state dicts and plain containers (no rrp classes), so they load unchanged.

## 11. glossary (names that are easy to confuse; was docs/related_repos.md)

| name | what it is | what it is NOT |
|---|---|---|
| **rrp** | this repository and its Python package (`import rrp`) | – |
| **structured-psi0-latent-diffusion-dynamics** | the GitHub name of this repo (renamed 2026-09-25, D-030) | not the Ψ₀ upstream |
| **relational-robot-policy** | the local folder name of this repo (`~/work/relational-robot-policy`) | not a different repo |
| **Ψ₀ / psi0** | the UPSTREAM humanoid VLA + SIMPLE benchmark (physical-superintelligence-lab/Psi0, arXiv 2603.12263), used unmodified from `~/work/ext` | not our code; never edited (compat is import-time hooks in `rrp.envs.simple.compat`) |
| **psi1z** | the former separate repo of the Ψ₀ line (D-100 .. D-140), github.com/JacobFV/psi1z, archived; its code is `rrp.policies.psi0` / `rrp.envs.simple` now | not a fork of Ψ₀ |
| **system i / system 0** | OUR architecture: system i generates the latent packet z; system 0 (realizer) turns z into native commands | "system 0" has nothing to do with Ψ₀ (psi-zero) |
| **packet / z** | the structured latent z[knots × assemblies × 64] passed from system i to system 0 | not Ψ₀'s action tokens |
| **SIMPLE** | Ψ₀'s humanoid benchmark (MuJoCo physics + Isaac Sim rendering), env `simple` | not our MuJoCo scenes |
| **D-xxx / P-xxx** | decisions in `research/decisions.md`; P-xxx = the folded psi1z log (appendix P, closed) | – |
| `~/work/ext/runs/psi1z/` | historical directory name of Ψ₀ checkpoints, features and eval outputs (kept; paths in records point there) | not a code location |

