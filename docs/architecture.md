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
| 4 | `rrp.policies` | `Policy` protocol, `PolicyInfo`, `Requirements`, `negotiate`, registry; `features/` (featurizers: the ONLY definition of what a policy may see), `nets/` (shared torch modules: attention, flow, codec, backbone, probes, checkpoint), `relations/` (the relation-factor registry: token sets + field provenance, factor entries, operators / forms, `FactorSite`, `ReadoutProbe`; section 12), `bc.py`, `latent/` (system i planners + system 0 realizers for arm, dual, legged), `trackers.py`, `teachers/` (scripted / privileged, labelled), `oracle.py`, `psi0/` (Ψ₀ direct / Ψ₀ + structure / demo replay, their nets, feature cache and training) | torch |
| 5 | `rrp.harness` | `rollout` (the one episode loop), `hooks` (the one hooks module: feasibility, settle, recorders, packet edits, perturbations; `HOOKS` by name, `TASK_HOOKS` per task), `eval.evaluate` (`evaluate` / `matrix`), statistics, gates, audits; `data/` (collect, pack, manifests, `relgen/` label functions / scene parts / transforms / curriculum, `mix`), `train/` (rep, flow, bc, refit, dagger, sft, grpo, ppo), pipelines + run-dag | – |
| 6 | `rrp.viz` | record/replay, the room exporter (`python -m rrp.cli viz export`, file scans only), replay-spec generator (`rrp viz specs`); the loopback workbench service is RETIRED (D-145, `.old/`) | – |
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
| `mujoco/legged` | `envs.mujoco.legged.LeggedSession`, `LocoPickSession`, `FootholdSession` (`make_legged_env`: waypoint_contact, loco_pick, foothold_steps, h_steps, h_gap) | legged + humanoids | `base_velocity` (10 Hz, embedded tracker); `legs` joint_position (50 Hz, no tracker) | base_velocity S3; `legs` S4 (`make_env(..., control="legs")`: one step per 50 Hz tracker tick, 10 Hz observation / runtime / fall schedule unchanged, `boundary` flag, default-stance fallback) |
| `warp/legged` | `envs.warp.tracker_env.WarpEnv` over WarpTrackerEnv / WarpStepsEnv / WarpGapEnv (`make_warp_env`) | legged + humanoids | `legs` normalized residual joint_position, batched | S3 (adapter; PPO keeps the engine API) |
| `simple` | `envs.simple` package (Ψ₀ agent: `envs/simple/{__init__,compat,worker}.py`, `bodies/g1_simple.py`, `policies/psi0/`) | `g1_simple` | `psi0` kind (36-d Ψ₀ command, 50 Hz; decoupled WBC inside) | declared |
| `computerworld` | `envs.computerworld` (CW agent) | `cw_pointer` | `pointer` cartesian_position, `button`, `wheel`, `key` discrete | declared |

S4 deleted the bake-off prototypes (`MjxLegged`, `WarpLegged`); their model-building helpers are `envs.warp.model`.
`tracker_validation` becomes a hook-based evaluation in S5. The CPU `LeggedEnv` stays: the D-126 physics goldens pin its
reward/observation streams and CPU PPO still uses it.

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
| `legged_latent`, `legged_bc` | `LatentLeggedController` / `BCController` in the tracker slot via `System0Adapter`/`BCAdapter` (harness.eval.legged_latent_eval) | `policies.legged.LeggedLatentPolicy` / `LeggedBCPolicy` (`make_legged_latent`, `make_legged_bc`) on the `legs` space; controllers and adapters moved to `policies.legged`; one episode at a time — S4 done, parity with the tracker-slot route (trace, packets, final pose) in test_deploy_eval |
| `tracker` | `envs.mujoco.legged_tracker.LearnedTracker` / `CPGTracker` | stays the embedded controller of `mujoco/legged`'s `base_velocity` space (and `body_tracker` in legs mode); trained on the Warp engine by PPO directly; saved-actor options (clock_gate, target_margin, ref_ff, extra_obs_dim, morph_v1) pinned by test_golden. No separate Policy wrapper (nothing needs one yet) |
| `teacher:<task>` | `policies.teachers.*` `act()` (reads session internals) | `policies.teachers.TeacherPolicy` (`make_policy("teacher:<task>")`), `requires.privileged`, source `scripted_teacher` — S4 done, parity golden (arm, dual) |
| `oracle` | `OraclePacketPolicy`, `OracleSource`, `OracleShadow` (three copies in harness.eval) | `policies.oracle` (ShadowTeacher, BCLookahead, OraclePacketPolicy, one encoder `encode_demos` also used by the semantic-edit OracleSource); `make_oracle` = LatentStackPolicy(oracle packets, frozen system 0), privileged, source `oracle`; legged: `policies.legged.OracleShadow` (`LeggedLatentPolicy(oracle=True)`) — S4 done |
| `psi0_direct`, `psi0_structured` | psi1z `serve_psi0` / `serve_ours` + `system_i` / `structured` | `policies.psi0` (Ψ₀ agent; registered names already declared) |
| `pointer_oracle`, `pointer_latent`, `pointer_bc` | – (new, track pointer) | `policies.pointer`: ComputerWorld `cw_pointer` system 0s (SCRIPTED engineered `cw_pointer_eng.v1`; learned realizer), teacher-oracle / learned system i packets through `LatentStackPolicy(make_s0=...)`, pointer BC; trainers `rrp train pointer` |

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

# S5 (harness.eval.evaluate over harness.rollout; hooks in harness.hooks; CLI `rrp eval`, `rrp matrix`)
def evaluate(policy, env_id, task, body, seeds, *, scene=None, batch=8, max_seconds=None, max_steps=None,
             hooks=(), out=None, row_extra=None) -> list[Episode]          # JSONL rows (Episode.row() + row_extra)
def summarize(episodes) -> dict       # attempted/successes/Wilson95/infeasible/outcomes/agreement/policy_calls/control_steps
def matrix(policies, envs: [(env_id, body)], tasks, *, seeds=(), out=None, build_heavy=False) -> list[dict]
#   every cell: accepted | n/a with reasons (task not in env, env/policy unavailable, negotiate); accepted cells roll out
#   on `seeds`. Env specs come from the env module's static `env_spec(task=, body=)` when present; HEAVY_ENVS
#   (simple: starts Isaac Sim) are never built implicitly.
# rollout(): max_steps (exact tick budget: the judge is asked with the budget spent); a hook's on_reset may return a
#   done Judgement (harness.hooks.Feasibility -> "infeasible", 0 steps); Episode.metrics counts chunks and packets.
```

Hooks replace the special cases of the current loops: `Meter`/stage reached, label error vs shadow teacher, oracle
comparison, DAgger collection, command logs, frame capture/video, perturbations and robustness grids, motion quality,
contact metrics, deploy OOD/safety/latency, probe readouts, packet edits (`swap_slots`, probe-guided, zero/freeze,
context transforms), noise keys for paired edits. Training (`harness.train`) keeps the existing dataset trainers;
DAgger, GRPO and adaptation call `rollout`.

CLI (`rrp` = `python -m rrp.cli`, S6b): `eval` / `matrix` (harness.eval.evaluate over harness.rollout, default hooks
from harness.hooks), `run-dag`, `ops …`, `data …`, `train …`, `latent …`, `adapt`, `campaign …`, `latency`,
`analyze`, `task validate`, `assets validate`, and the tool commands of `rrp.cli.tools`. No library module
is a program any more (a test enforces it; the only `__main__` modules left are `rrp.cli` and the SIMPLE worker /
compat layer, which run inside the Isaac Sim venv as subprocess targets of `rrp.envs.simple`). A tool keeps its own
argument parser; `rrp <group> <tool> ARGS` passes ARGS through unchanged:

The command list is data, not prose: `rrp.cli.tools.TOOLS` maps `(group, name)` to `module:function` (formerly `python -m <module>`), and `rrp --help` lists the groups; a unit test keeps every entry importable. Groups: `data`, `train`, `suite` (evaluation suites, audits, validators, `sealed-log`), `eval` (`humanoid-transfer`), `stage`, `viz`, `video`, `factors`, `ops`.

The BC `rrp evaluate` is `rrp eval --policy 'bc={"checkpoint": ..., "nfe": 8, "execute_prefix": 8}' --env mujoco/arm
--task pick_place --body <robot> --seeds <a:b> --out ...` (one body per call).

## 7-10. D-140 refactor record (archived)

The delete list, the S0-S6 migration table, the Psi0 / ComputerWorld hand-off checklist and the old-to-new module path
map (old `rrp.controllers.*` / `rrp.training.*` / `rrp.evaluation.*` ... names) are in `.old/docs/architecture_s7-10.md`
(D-145 P8). Everything they describe is done; the current structure is sections 1-6 and 12-13.

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

## 12. relation factors (D-144; full design `docs/relations.md`, catalog `research/relations_catalog.md`)

Every relational trick (typed task-graph edges, morphology edges, routing masks, PaPE-style geometry, interaction and
task-conditioned edges, probe readouts, data mixes) is one declarative entry of `rrp.policies.relations.catalog`
(field × operator × form × algebra × conditioning × source, plus its label and data generators), switched in ONE run
config list `factors:` (`FactorSpec`: control on / off / zero / shuffled / rewired / reversed / gt / estimated, source,
sites, heads, gate, confidence, weight, mix). Nets consume `TokenSet`s whose fields carry provenance (public /
estimated; privileged values live only in `labels`) and apply factors through `FactorSite` inside the shared `MHA`
(additive bias or kernel-compatible q/k augmentation). `versions["factors"]` (compat hash) is in every checkpoint; a
deployable policy refuses `gt` sources. Labels are written once against `envs.base.StateView`; scene parts compose
(`compose`) under a progressive `Curriculum` in the one `relations_data` stage.

## 13. repository schema (D-145; manifest `schema.toml`, enforced by `tests/unit/test_layout.py`)

One file schema for the whole repository. Every tracked path must match `schema.toml`; anything else is either
re-expressed in the schema or moved under `.old/`. The layout test is xfail until the purge (13.6) lands.

### 13.1 top level

| path | holds | rule |
|---|---|---|
| `src/rrp/` | all code (sections 1–6, 12); task graphs as package data `src/rrp/tasks/graphs/*.json` | no program outside the `rrp` CLI |
| `tests/` | `unit/`, `integration/`, `gpu/`, `data/` (goldens and JSON fixtures only) | tests never read `.old/` |
| `recipes/` | every run definition: `templates/*.yaml` (DAG templates), `<track>/*.yaml` (thin instances), `presets/*.json` (shared `params` fragments: model sizes, eval protocols) | no hand-written per-run config |
| `docs/` | `architecture.md`, `relations.md`, `strategy.md`, `experiments_roadmap.md`, `robot_training_considerations.md` | fold, do not add |
| `research/` | `decisions.md` (append-only, never moved), `registry.jsonl`, `relations_catalog.md`, `sources*.json`, `tasks.json`, `splits/` (sealed splits and paired-data specs), `methods/`, `corrections/`, `reports/evidence_matrix.md`, `tracks/<track>.md` for the tracks in `schema.toml [tracks]` + `BRIEF.md` | a closed track's note moves to `.old/` |
| `artifacts/` | the append-only evidence store (13.4) | small raw results only; weights / datasets are never tracked |
| `ops/` | `resource-ledger.jsonl`, `resources.local.json`, `bin/*.sh` (peer transport, asset fetch, external setup: shell that must run without the python env) | runtime state is git-ignored |
| `viz/` | `room/` (the room), `specs/`, `CONTRACT.md`, `radar_axes.json` | |
| `.old/` | legacy (13.5) | never read by live code |
| files | `AGENTS.md`, `CLAUDE.md`, `README.md`, `STATUS.md`, `pyproject.toml`, `schema.toml`, `.gitignore` | |

Removed top-level areas: `configs/`, `dags/`, `scripts/`, `tasks/`, `ui/`.

### 13.2 ontology: what a run is

A run is one stage of an experiment on a point of the axes below; its identity is data, never a file name.

| axis | values come from | today's field |
|---|---|---|
| policy | `rrp.policies.base.POLICIES` (`bc`, `latent`, `legged_latent`, `psi0_structured`, `pointer_latent`, ...) | `family` + stage set |
| env | `rrp.envs.base.ENVS` | `family` |
| task | `rrp.tasks.spec.TASKS` | recipe `vars.task` |
| body set | `rrp.bodies` catalog keys / a split in `research/splits/` | recipe `vars.body` / `train_robots` |
| factor set | `rrp.policies.relations` presets / a `factors:` list; identity = `compat_hash` (`fx-…`). The former `variant` (sem / nosem / semfix) is a factor set (probe readout weights and `lv_min`) | `variant`, `params.*.factors` |
| seed | integer | `seed` |
| stage `[-tag]` | `rrp.core.runconfig.PIPELINE_STAGES` | `stage`, `tag` |
| track / lineage | `schema.toml [tracks]`; lineage = the chain of stages sharing data and representation, named `<policy>-<body set>-<factor set>[-<note>]` | `track`, `lineage` |

- **Run id** = `<track>/<lineage>/<stage>[-<tag>]_s<seed>`; **output** = `artifacts/runs/<run id>/` (already what
  `RunConfig.out` derives). The run directory holds `config.json` (the rendered `RunConfig`: all axes, flags, params,
  input run ids, `config_hash`, `versions["factors"]`, git sha) — written by the stage, never by hand.
- **A run config is rendered, not written.** `RunConfig` (`rrp.core.runconfig`, schema `runconfig-1`) is the one
  config type; recipes produce it by template + matrix expansion over the axes (`rrp run-dag <recipe> --dry-run`
  prints every node's config; `--point seed=…` selects). `load_legacy` / `classify_legacy` / `LEGACY_FLAG_DEFAULTS` /
  `LegacyInfo` are deleted with the configs they read.
- **Inputs are run ids**, resolved by `artifacts/run_index.json` (aliases of pre-pipeline runs, was
  `configs/run_index.json`) or directly to `artifacts/<id>`.

### 13.3 recipes

```
recipes/templates/<family>_<purpose>.yaml    dag-1 templates: arm_lineage, arm_targets_bc, arm_targets_latent, arm_grpo,
                                             dual_lineage, legged_lineage (gated), legged_heldout, tracker_gated,
                                             pointer_lineage, psi0_step2, relations_factor
recipes/<track>/<name>.yaml                  instance: `extends: ../templates/<x>.yaml` + header + vars / matrix only
recipes/presets/<name>.json                  shared params fragments (policy-small, codec-small, eval protocol latent_slice1)
```

Instance header (required keys, checked): `name`, `schema`, `track`, `policy`, `env`, `task`, `bodies`, `factors`,
`extends`; at most 80 lines — a pipeline belongs in a template. `extends` chains (today `arm_lineage_v7div` →
`v6` → `v2` → `arm_lineage`) are flattened into one template with vars. Per-body copies (`legged_v2_{anymal,go2,
t1,t1sl}`) and per-version copies (`armexpert_v{4,5,6}dart`, `arm_lineage_v2/v6`) are instances or legacy.
Overlays (`.old/dags/overlays/*`) become template vars. One-off drivers (`.old/scripts/*_chain.sh`, `humanoid_*.sh`) become
recipe nodes calling `rrp` commands.

### 13.4 artifacts (tracked vs not)

Tracked: `artifacts/runs/**` result files (`json, jsonl, md, txt, log, gz, out, csv, png`, ≤ 4 MB), `artifacts/video/*`
(+ `INDEX.md`), `artifacts/trackers/**` (tracker metadata), `artifacts/assets/**`, `artifacts/receipts/**`,
`artifacts/requirements.json`, `artifacts/run_index.json`. Never tracked: weights, datasets, packed data, episodes
(`.gitignore`; peer store / `~/work/rrp-data`). The store is append-only evidence: the 170 run directories that
predate the schema are frozen by name in `schema.toml` (decisions cite their paths; they are not renamed and do not
move to `.old/`); a new run directory must start with a track name from `[tracks]`.

### 13.5 current vs legacy

| area | CURRENT (stays / re-expressed) | LEGACY (→ `.old/<same path>`) |
|---|---|---|
| `configs/` (280) | `model/policy-small-structured.json`, `model/codec-small.json`, `eval/latent_slice1.json`, `eval/primary.json` → `recipes/presets/`; `run_index.json` → `artifacts/`; `resources.local.json` → `ops/` | everything else (ladder, latent, legged_*, adapt, data, t1_diag, vlm, other model files) |
| `dags/` (39) | `templates/*` → `recipes/templates/` (renamed per 13.3); instances needed by open / paused tracks re-expressed under `recipes/<track>/` | per-lineage and per-version copies, overlays, smoke and parity DAGs |
| `scripts/` (36) | `peer_*.sh`, `fetch_menagerie.sh`, `psi0_ext.sh` → `ops/bin/`; `export_ui_types.py`, `viz_record_specs.py`, `render_*episode.py`, `humanoid_*_eval.py`, `contact_waypoint_eval.py` → `rrp` tool commands | `demo/`, `armdiv_*`, `humanoid_*.sh`, `render_contact_compare.py` after their recipes exist |
| `research/` | `decisions.md`, `registry.jsonl`, `relations_catalog.md`, `sources*.json`, `tasks.json`, `splits/` (+ `pairs/` moved in), `methods/`, `corrections/`, `reports/evidence_matrix.md`, `tracks/{humanoid,armdiv,psi0,pointer,relations,BRIEF}.md` | every other `tracks/*.md` and `tracks/*/` (incl. `rel-r*`, `ladder/`, `armdiag/`), `scripts/`, other `reports/`, `naming.md` |
| `docs/` | the five files of 13.1; sections 7–10 of this page (D-140 delete list, stage table, hand-off, moved paths) move out | `handoff/`, `demo/`, the D-140 tables |
| `tests/` | everything except: | `.old/tests/data/legacy_scripts/` + `test_ladder_cli_parity.py` (RETIRED: the ports are pinned by goldens) |
| `ops/` | ledger | `host-preflight-initial.json` |
| `tasks/` | → `src/rrp/tasks/graphs/` (package data) | – |
| `ui/`, `src/rrp/viz/workbench/`, `.old/tests/browser/`, `tests/integration/test_{service,ui_contract}.py` | – (the loopback workbench is RETIRED, D-145 lead ruling: the viz room replaced it) | `.old/ui/`, `.old/src/rrp/viz/workbench/`, `.old/tests/` |

Tests and goldens that pin legacy configs:

| test | decision |
|---|---|
| `test_dag.py::test_arm_dag_reproduces_legacy_configs`, `test_dag_*`, `test_dag_templates`, `test_d126_*`, `test_dual_v3` | RE-ANCHOR: before any move, record `recipe.<name>` goldens (digest of every rendered node config of each kept recipe) on the current tree, where the existing tests still prove equality with the legacy files; then point the tests at `recipes/` and delete the legacy-equality assertions |
| `test_runconfig.py` legacy round-trip over `configs/` | RETIRE with `load_legacy`; keep the schema / flag / variant tests on rendered recipe nodes |
| `test_relations_r2_latent.py` frozen `LatentConfig.version()` table over `configs/**` | RE-ANCHOR to `tests/data/latent_versions.json` (the `latent` dicts + expected versions, generated once from the legacy files): bundle compatibility IDs of existing checkpoints stay pinned |
| `test_legged_frozen_latent.py` (`.old/configs/legged_fixsem`) | RE-ANCHOR to the rendered `legged_lineage` recipe nodes |
| `test_ladder_cli_parity.py` | RETIRE |
| `tests/data/golden.json` | unchanged except the added `recipe.*` keys |

Tracks (`schema.toml [tracks]`): `humanoid` (W13, paused), `armdiv` (paused), `psi0` (paused: structured-arm fix),
`pointer` (paused: follow-ups; note `research/tracks/pointer.md`, was `cworld.md`), `relations` (open). Each paused track's
RESUME section is rewritten as recipe commands (`rrp run-dag recipes/<track>/<name>.yaml [--point …]`) BEFORE its
legacy DAGs / scripts move. All other tracks are closed: their notes move to `.old/`, their decisions stay in
`research/decisions.md`.

### 13.6 purge units (each: layout test for its area green, unit suite green, `rrp run-dag --dry-run` of every kept recipe)

| id | unit | deps | owns |
|---|---|---|---|
| P1 | recipe goldens + templates | – | `dags/**` → `recipes/templates/**`, `harness/dag.py` (recipe root, `extends`), `tests/unit/test_dag*.py`, `test_d126_*.py`, `test_dual_v3.py`, `test_legged_frozen_latent.py`, `.old/dags/` |
| P2 | configs purge + RunConfig legacy removal | P1 | `configs/**`, `recipes/presets/`, `artifacts/run_index.json`, `ops/resources.local.json`, `core/runconfig.py`, `tests/unit/test_runconfig.py`, `test_relations_r2_latent.py`, `tests/data/latent_versions.json`, config path literals in `cli/{latent,train}.py`, `harness/eval/target_eval.py`, `harness/pipelines/arm.py`, `harness/train/{baseline_campaign,tracker_training}.py`, `.old/configs/` |
| P3 | scripts purge | – | `.old/scripts/**`, `ops/bin/`, `cli/tools.py` (new tool entries), script path literals in `harness/dag.py` (peer_run only, after P1) and `harness/pipelines/legged.py`, `tests/unit/test_provenance.py`, `tests/integration/test_ui_contract.py`, `tests/conftest.py`, `.old/scripts/` |
| P4a | humanoid recipes | P1, P3 | `recipes/humanoid/`, `research/tracks/humanoid.md` |
| P4b | armdiv recipes | P1, P3 | `recipes/armdiv/`, `research/tracks/armdiv.md` |
| P4c | psi0 + pointer recipes | P1, P3 | `recipes/psi0/`, `recipes/pointer/`, `recipes/templates/{psi0_step2,pointer_lineage}.yaml`, `research/tracks/{psi0,pointer}.md` (`cworld.md` renamed) |
| P4d | relations recipes | P1 | `recipes/relations/`, `recipes/templates/relations_factor.yaml`, `research/tracks/relations.md` (new; folds the open items of `rel-*.md`) |
| P5 | research purge | P4a–d | `research/**` except the P4 notes and `decisions.md`; `viz/export/{knowledge,scan,dags,psi0}.py` path literals; docstring references to moved notes in tests; `.old/research/` |
| P6 | tasks + workbench move | – | `tasks/` → `src/rrp/tasks/graphs/`; `ui/`, `src/rrp/viz/workbench/`, `.old/tests/browser/` and the workbench tests → `.old/`; `rrp workbench`; `pyproject.toml`, `core/paths.py`, `.old/ui/`, `.old/src/`, `.old/tests/{browser,integration}/` |
| P7 | tests + ops leftovers | – | `.old/tests/data/legacy_scripts/`, `tests/unit/test_ladder_cli_parity.py`, `ops/host-preflight-initial.json`, `.old/tests/`, `.old/ops/` |
| P8 | docs + top-level | P1–P7 | `docs/**`, `README.md`, `STATUS.md`, `AGENTS.md`, `CLAUDE.md`, `.old/docs/`, `.old/README.md` (final index), every remaining textual reference to a moved path |
| P9 | close | P8 | remove the xfail marker in `tests/unit/test_layout.py`; dry-run every recipe; record the result under D-145 |

Briefs:
- **P1.** First commit: add `recipe.<name>` goldens — for each DAG that 13.5 keeps (all of `dags/templates/*` and
  the instances the open / paused tracks name in their RESUME sections: `arm_lineage_v7div*`, `armdiv_*_v7div*`,
  `d126_tracker_*`, plus one instance per template for coverage), digest every rendered node config
  (`harness.dag` expansion, sorted JSON) into `tests/data/golden.json` on the UNCHANGED tree. Then `git mv` templates to
  `recipes/templates/` with the 13.3 names, flatten `extends` chains into template vars, make the DAG loader resolve
  recipes from `recipes/`, move every other DAG to `.old/dags/` (with `.old/dags/README.md`: group → what → citing
  decisions). The goldens must not change; delete the legacy-equality assertions only after that.
- **P2.** Move `configs/**` to `.old/configs/` (README by group: ladder / latent / legged_* / adapt / data / t1_diag /
  vlm / model, with the decisions that cite them). Before moving, generate `tests/data/latent_versions.json` from the
  latent configs and re-anchor `test_relations_r2_latent.py` on it. Presets, run index and resources move as in
  13.5; update the path literals listed. Delete `load_legacy`, `classify_legacy`, `_legacy_variant`,
  `iter_legacy_configs`, `LEGACY_FLAG_DEFAULTS`, `LegacyInfo`, `LEGACY_ONLY_STAGES` and their tests.
- **P3.** `git mv` the transport / fetch scripts to `ops/bin/` and fix callers; turn the listed python scripts into
  `rrp` tool commands (`cli/tools.py` table; module under the package that owns the logic; no `__main__`); move the
  rest to `.old/scripts/` with a README. Do not move a script a paused track's RESUME still names until P4 has
  replaced it (coordinate through the merge lock: P4 units delete those lines).
- **P4a–c.** For the track, read its note's RESUME section and write `recipes/<track>/*.yaml` instances (header per
  13.3) whose nodes are the exact commands of the resume steps, as `rrp` commands; rewrite the RESUME section as
  `rrp run-dag recipes/<track>/<name>.yaml …` lines; `--dry-run` each and paste the node list into the note. No
  training, no simulation (host rules).
- **P4d.** New track `relations`: a template running `relations_data` → arm flow training with a factor-set axis →
  probe / interference report; instances for the first-wave presets (`geo`, `ix`, `task`, `ui`); note
  `research/tracks/relations.md` with the open items collected from `rel-*.md` lead_questions.
- **P5.** Move closed-track notes and their subdirectories, `research/scripts/`, superseded reports and `naming.md`
  to `.old/research/` (README table: track → what → decisions → state at closure); move `pairs/` into `splits/`
  (update `.old/configs/data` consumers if any remain live); fix exporter path literals so the room reads only open tracks.
- **P6.** Task graphs become package data (`scenario.TASKS_DIR` reads `rrp/tasks/graphs`; `data_path` / `package_data` and the wheel force-include are dropped);
  the loopback workbench (`ui/`, `src/rrp/viz/workbench/`, its browser and service tests, the `rrp workbench` command, the `service` extra) is retired to `.old/` (the viz room replaced it).
- **P7.** Retire the ladder CLI parity test and its frozen scripts; move the initial preflight record.
- **P8.** Move `.old/docs/handoff/`, `.old/docs/demo/` and sections 7–10 of this page to `.old/docs/`; fold the still-binding
  handoff rules into `AGENTS.md`; rewrite `README.md` / `STATUS.md` to the schema; finish `.old/README.md`; grep the
  tree for every removed top-level path and fix or delete the reference.

## 14. pre-training contracts (D-146; plan and units: `research/readiness.md`)

### 14.1 task / eval contract (unit F2)

```python
@dataclass(frozen=True)
class TaskSpec:                        # rrp.tasks.spec (adds to section 4)
    ...
    build: Mapping[str, str] = {}      # env_id -> "module:builder" (scene builder owned by the task, no if/elif in factories)
    scene: Callable[[int], dict] | None = None   # seed -> scene kwargs; None = the env takes no scene
    hooks: tuple[str, ...] = ()        # default eval hooks by name (rrp.harness.hooks.HOOKS)
    teacher: str | None = None         # POLICIES key (factory owned by the policy registry; no tuples in teachers/__init__)
    max_steps: int | None = None       # explicit tick budget (replaces the judge max_steps hack)
    failure_reasons: tuple[str, ...] = ()   # the vocabulary the judge may emit
class Env(Protocol): def failure_reason(self) -> str | None   # optional; env-side public failure code in its own vocabulary
```

- `make_env` passes `scene` only when the task declares one; a task with a scene on an env whose factory takes none
  is a `negotiate` failure (reason text), never a TypeError. `evaluate()` / `rrp eval` take `--env-kw k=v` (level,
  split, render profile) and contain no env-id or task-name string comparisons. `rollout` negotiates every env of a
  group. One `DUAL_TASKS` definition.
- **Hooks** live in ONE module, `rrp.harness.hooks` (beside `rollout.py`; it imports nothing from `harness.eval` or
  `harness.data`, both import it): `HOOKS` (by name), `TASK_HOOKS` (the per-family defaults a `TaskSpec.hooks` entry
  selects, with the env-id prefix each applies to) and the generic hook classes (feasibility, settle, session record,
  displacement, recorders, end-when). Hooks that need a family's probe or metric code live beside it
  (`harness.eval.latent_eval`, `dual_latent_eval`, `harness.data.contact_metrics`). `rollout` refuses a failure reason
  outside `TaskSpec.failure_reasons` + rollout's own reasons + the `failure_reasons` of its hooks (`UndeclaredFailureReason`).
- **Bench envs**: a rig that is not a task world runs under `rollout` as a bare-model `Env` (no scene, no task runtime,
  one control tick per physics step): `grasp_rig` (`mujoco/grasp_rig`, `_RigEnv` + `_RigMeter`) and the tracker validation
  trials (`mujoco/tracker_bench`, `_BenchEnv` + a scripted command policy + kick / meter hooks). The quantities they report
  (contact forces, cost of transport, joint margins) are read from the model / data by the env or a hook, never by a
  private stepping loop. Energy and CoT stay substep-exact because the env integrates them and reports them in
  `StepResult`. Outside `rollout.py` and env modules `.step(` / `mj_step(` is a lint error
  (`tests/unit/test_step_lint.py`); the one allowlisted exception is the snapshot look-ahead of `ShadowTeacher.lookahead`.
- Humanoid judge (unit HJ, `rrp.tasks.humanoid`): `fell`, `wall_collision`, `wrong_heading`, `trip`, `missed_step`,
  `hold_lost`, `dropped`, `timeout` from `env.failure_reason()` + truth; tasks registered with `build`.

### 14.2 pipeline / recipe contract (unit F3)

- **Open stage registry**: `register_stage(family, stage, fn, *, flags=())` in `harness.pipelines.base`; `PIPELINE_STAGES`
  is derived. Families: `arm`, `dual` (parked: collect / eval only), `legged`, `humanoid`, `psi0`, `pointer`,
  `relations`. `relations_data` is a real stage usable as a node of any family's recipe.
- **One channel to child processes**: the rendered `RunConfig`. `apply_run_context(cfg)` (called at stage entry in
  parent and child) resolves `factors`, configures featurizer options from them (`feat.base_axes`; the
  `$RRP_KINFEAT` environment variable is deleted) and sets the deploy flag. `StageContext.env()` carries only
  resources and `$RRP_RUN_CONTEXT`, the path of the rendered context, so a grandchild process (a process a child
  spawns) finds the same `RunConfig` (not a second channel). Families load at plan time.
- **Trainer hook**: `relation_batches(cfg, out_dir)` (`harness.data.mix`) wraps `mixed_batches` + `Scheduler`, reads
  `steer.jsonl`, appends `schedule.jsonl`; run-config keys `factors:` and `curriculum:`. Each trainer calls it once.
- **Adoption** of a completed node requires equal `config_hash`, `versions["factors"]`, catalog version (hash of the
  registry entries the specs use) and `PIPELINE_VERSION` (one constant, bumped when a stage changes meaning). The
  ledger records git revision, tree hash and dirty flag (untracked files count as dirty) per attempt; a node completed
  under another revision with equal versions is `stale`: adopted only with `--adopt-stale`, and reported.
- **Manifest**: `pipeline_manifest.json` carries `provenance.versions.factors` + factor provenance; the room reads it.

### 14.3 trackers, experts, terrain (unit HT)

- `TRACKERS[(body, version)] -> TrackerEntry(sha256, obs_format, extra_obs, gate, decision, store)` loaded from
  `artifacts/trackers/<body>/<version>/meta.json`; `make_env("mujoco/legged", ..., tracker="<body>:<version>")`;
  no monkeypatching of `load_tracker`. One recipe registry, one `_TURN`.
- **Terrain is a public sensor** (D-146): `terrain_scan` = egocentric elevation grid (11 × 7 cells, 0.1 m, x from −0.2 m
  and y from −0.3 m) in the **yaw frame** (base position, heading-only rotation, gravity-aligned; not the full body
  frame), range 1.5 m, from a declared downward depth-sensor model with noise (σ 1 cm), 2 % dropout and one-tick latency,
  computed identically in Warp and MuJoCo (`mj_ray`); it reads the ground (floor + scene ground geoms) only, so walls,
  objects and the robot itself are not scanned. Capability `terrain_scan`; a `declared_sensor_channels` entry; layout
  and noise model in the actor meta (`terrain_scan_spec()`, version `terrain_scan_v1`). The actor and `rl_expert`
  consume it (`extra_obs_dim` = 77); the critic keeps the exact scan (privileged). Test: Warp actor obs dim == MuJoCo
  adapter dim, same cell layout.
- **`range_ring`** (HS1, the gap expert's second public sensor): 16 horizontal rays in the yaw frame (ray k at angle
  2πk/16 from +x, counter-clockwise) cast from the base origin (torso height) through every colliding geom except the
  robot's own; reading = hit distance, clipped to 3 m (no return = 3 m), noise σ 2 cm, 2 % dropout, one tick of latency;
  capability `range_ring`, declared channel `0:range_ring`, `range_ring_spec()` (`range_ring_v1`). A gap actor takes
  scan + ring (`extra_obs_dim` = 77 + 16 = 93, scan first); `make_env("mujoco/legged", ..., range_ring=None)` turns it on
  exactly when the actor takes it, so the expert is not blind to walls. Both sensors need `control="base_velocity"`.
- `rl_expert` Policy (`policies.teachers.humanoid`, key `rl_expert`): loads a registry actor, source
  `learned:rl_expert:<sha>` when its obs are public, `privileged_teacher:rl_expert:<sha>` otherwise.

### 14.4 humanoid family, sealed split, upper body (units H4, H5, H6, U1–U3)

- Stages: `collect` (task registry + `rl_expert`; EVENTS / public context from the TaskSpec) → `pack` → `train_rep`
  → `train_flow` → `eval_transfer` (bodies × methods × demo budgets × seeds; Level 1 reported apart from Level 2;
  one acquisition accounting) → `sealed_eval`.
- `rrp.core.sealed.SealedSplit` (hash-pinned `research/splits/humanoid_v1.json`): `assert_train_allowed(bodies,
  seeds)` in every data / train stage; `sealed_eval` needs `--sealed`, appends to
  `artifacts/runs/humanoid/sealed_log.jsonl` and refuses a repeat of a (method, body, task, seed set) cell.
- **Sealed cells** (`core/sealed.py`): `SealedSplit` is split-agnostic (`humanoid_v1`, `armdiv_v1`, `cworld_pointer_v2`;
  `SealedSplit.load(name)`); a cell is the native id `body|method|task|n<budget>|adaptation|s<train_seed>|seed_set` (absent
  parts `-`); each split has its own run-once log (`artifacts/runs/<track>/sealed_log.jsonl`); `rrp suite sealed-log list` shows the cells and
  `infra-failure` releases an OPEN cell after an infrastructure failure (never after a bad result). The wholebody and gap tracker trainers call
  `assert_train_allowed` before creating anything; humanoid adaptation stages write `acquisition.json` with their cell id.
- **Upper body**: control mode `wholebody` = the `legs` group (unchanged: `control="legs"` stays bit-identical,
  golden) + an `upper` joint_position group (arms, waist, head; 50 Hz PD) in both backends. Trackers for wholebody
  are trained with random upper-body targets and payload (recipes `*_ub`); system 0 realizes `upper` from the packet
  like arm joints. Teachers: scripted IK reach / grasp on the humanoid hands (public FK, labelled
  `scripted_teacher`) composed with `rl_expert` legs. Tasks: L0 stand / walk-to, L3 turn-in-place; M1 reach, M2
  squat-pick, M3 place; C1 carry (tray / box while walking), C2 loco_pick; held-out `h_steps_carry`, `h_gap_cart`.

### 14.5 Ψ₀ + structure fix (D-141; unit P1)

Diagnosis (D-141 addendum): system 0 bypassed the packet, and state dims constant in training went OOD in closed
loop. Mechanism: (a) **constant-input mask** — per-dim std of every R input over the feature cache; dims with std <
1e-4 are zeroed and their mask is stored in the checkpoint and applied at inference; (b) **forced packet use** in
`StageA.loss`: state dropout on R (each non-masked state dim dropped with p = 0.3, whole state with p = 0.1) and a
permuted-packet hinge `relu(m − (err(R(z_perm)) − err(R(z))))`, m = 0.05 in normalized action units; (c) **gate**:
the `heldout` stage reports err(R(z_mean)) − err(R(E(a))); the structured head refuses to train when the gap < m.
Red/green test reproduces the ±1 torso-command shift.

### 14.6 pointer copy mechanism and discrete key code (unit C2)

- Key head = mixture: `p(key) = (1 − g)·softmax(free) + g·Σ_pos α_pos·onehot(char_at(pos))`, α = attention from the
  knot token to instruction-character tokens, g = sigmoid gate; loss = NLL of the mixture; next-char supervision is
  `instr[n_typed]` (public).
- `cw_pointer_eng.v2`: the key is a 7-bit ±1 code over `KEY_VOCAB` (v1's scalar stays decodable by version tag).
- Procedural strings (seeded generator) and `research/splits/cworld_pointer_v2.json` declared before any demo;
  `cworld_pointer_v1` and its sealed rows are untouched.

### 14.7 compute block: precision, compile, CUDA graphs (D-147 addendum; unit prec)

- One module, `rrp/core/compute.py`, serves every trainer (behavior codec / policy / SFT, latent rep / flow / probe fit / refit, legged
  latent rep / flow, legged BC, joint adapt, pointer rep / flow / BC / probe, Psi0). A trainer calls `cx = compute.setup(name, dev)` once and
  uses `cx.autocast()`, `cx.compile(module_or_fn, name, methods=...)` and `cx.write_stamp(out)`. Loss reductions that must stay float32
  are decorated `@compute.f32` (`gaussian_nll`, `readout_loss`, `masked_mse`) or fed through `compute.upcast` (`estimates_loss`, the
  representation / flow / realizer losses). The adapt.py likelihood ratios keep their own fp32 code. Nothing else in the repo calls
  `torch.autocast` or sets `allow_tf32` (tests/unit/test_compute.py enforces it, adapt.py excepted).
- `RunConfig.compute` (`precision fp32|bf16`, `tf32`, `compile off|reduce-overhead|max-autotune`, `cuda_graphs`, `seeds_per_job`,
  `eval_backend cpu|warp`). ABSENT (None) is the default and is omitted from the serialised config, so `config_hash`, goldens and every
  pre-registered run are unchanged. A non-default block hashes into the config, is pinned in `stage_versions` and recorded in the stage
  manifest, so runs with different compute settings never adopt each other's outputs (`seeds_per_job` and `eval_backend` are read by the
  seeds and eval units from `compute.current()`).
- Defaults are today's behaviour per trainer: `precision: null` resolves to fp32 except pointer and Psi0 (bf16 autocast on CUDA, as
  before) and behavior (TF32 matmul, as before). Under an explicit `precision: bf16` the fp32 islands switch on: loss reductions,
  probe variance / log-variance terms and the likelihood path run in float32, optimizer state is always fp32.
- Compile is applied to module METHODS on the instance (`forward`, or `velocity` / `encode` / `decode`), so `state_dict` keys never carry
  `_orig_mod.`; the first call that raises falls back to eager and the stamp (`compute.json` next to each trainer's result) records
  `compiled[name].status` and the reason. `reduce-overhead` is the CUDA-graph mode and needs `cuda_graphs: true`.
- Measured effect: `research/tracks/compute.md`; the tool is `rrp train bench-compute` (peer, tiny real batches).


### 14.8 batched evaluation backend (unit warpeval; `research/tracks/compute.md`)

- `rollout(..., eval_backend="cpu"|"warp", device=None)` and `evaluate(...)`/`rrp eval ... --eval-backend` (default `cpu`: byte-identical rows, no
  new keys). With `warp`, the episodes stay CPU `Session`s (tasks, judges, hooks, sensing, controllers, trackers, policies are the CPU code); only
  the integration of a control step is batched: `Session.step` is a generator (`_step_gen`) yielding an `Integrate(n, ctrl schedule, energy
  actuators)` request where it used to loop `mj_step`; `run_cpu` drives it with CPU `mj_step` (the default path), `BatchStepper`
  (`envs/warp/batch_sim.py`) collects the requests of all running episodes, uploads state, advances substeps 0..n-2 on MuJoCo Warp (one CUDA graph
  launch per substep), downloads, and runs the LAST substep on CPU `mj_step` so xpos / contacts / sensors / actuator forces keep CPU semantics.
- Episodes are grouped by model (identical except per-world batchable fields such as `body_pos / body_quat / qpos0`); a different distractor count
  is another group. Unsupported combinations run on CPU and say why in every row of the group (`provenance.eval_backend = {requested, effective,
  fallback}`): perturbation hooks / instance-replaced `apply_substep` or `_tracker_tick`, actuator-model modes, dual sessions, envs that are not
  MuJoCo Sessions, a hook or policy declaring `batch_unsafe`, applied external forces, models `mujoco_warp` cannot build.
- Legged latent / BC policies run at batch > 1 with one shallow controller copy and one torch generator per episode (seeded from the controller's
  seed and the episode seed); at batch = 1 the controller is used as before (its generator continues across episodes).
- Cached results of one backend are never adopted by a run on the other (`humanoid-transfer` keys on `eval_backend`).
- Parity (`rrp suite warp-parity`): single-step, closed-loop outcome gate and divergence statistics under the tolerances pre-registered in
  `research/tracks/compute.md`; Warp is fp32, so trajectories are not bit-identical to the fp64 CPU.
