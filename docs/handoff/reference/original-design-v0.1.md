# relational robot policy and research workbench

architecture specification v0.1 · 2026-09-21, san francisco time

status: written for review. this is a design, not an implemented system or a report of experimental results. no training, remote repository changes, hardware access, or paid compute is authorized by this document.

## 1. purpose and agreed constraints

build a new research implementation of a ψ₀-inspired, morphology-conditioned vision-language-action policy. its shared representation should describe task-relevant entities, events, roles, effects, contact frames, and uncertainty, while a body-conditioned realization path produces usable control commands. the practical objective is to reduce the total incremental cost of deployment onto a new robot. gradient-free transfer is desirable, but small amounts of supervised adaptation or reinforcement learning are acceptable.

the implementation is new. that does not require randomly initializing a vision-language foundation model or rewriting physics engines. the default is to reuse ψ₀'s pretrained system-ii weights where compatible through a thin adapter, while implementing the morphology encoder, event representation, action codec, policy, interfaces, evaluation, and workbench afresh. a different pretrained backbone must be explicitly identified and separately ablated if the ψ₀ feature interface proves incompatible. a fully random-initialized small policy is a controlled experiment, not the foundation-model initialization strategy. describe the resulting architecture as ψ₀-inspired until experiments actually establish what transfers from ψ₀.

agreed requirements:

- variable morphology, including humanoids, quadrupeds, hexapods, standalone arms, and mobile or dual-arm manipulators; physically synthetic combinations are allowed.
- supplied task graphs, directly editable and inspectable in a robot-interaction gui; autonomous discovery of task decompositions is not a prerequisite.
- privileged simulator state may supervise representation learning, teachers, rewards, and evaluation, but must not silently enter the deployable policy observation.
- use the ψ₀ separation between high-level action prediction and lower-level tracking as the initial control pattern, rather than attempting universal torque control immediately.
- morphology-conditioned action tokens, queryable manipulator-specific task representations, canonical object identity, relational predicates, compositional events, active contact frames, and uncertainty remain first-class design goals.
- inference overhead, adaptation interactions, adaptation compute, and new-body integration work are measured rather than assumed.

## 2. decisions and alternatives

### selected design

use typed semantic memory plus one continuous flow-matched action stream. compile supplied task structure into an explicit runtime representation. encode current observations and body structure once per policy observation. cache fixed per-layer context key/value projections for the subsequent action sample. dynamically conditioned action queries read that memory during flow integration.

this retains the requested semantic decomposition without requiring a separate full transformer for every concept. event, scene, morphology, and interaction/frame data are distinct typed banks. uncertainty is initially attached to entities, relations, and frames, not implemented as a fifth general-purpose transformer. a separate belief module can be tested if the simpler representation cannot handle history or partial observability adequately.

### retained comparisons

an all-token, jointly updated multi-stream transformer is an expressive ablation, not the default low-latency design. a simple shared graph-conditioned policy without the event/affordance auxiliary objectives is the principal scientific baseline. both must receive the same permissible observation and task information.

a scalar or vector per robot with a new robot-specific output head is an additional practical adaptation baseline. it is not an appropriate zero-shot comparator on a robot for which that head does not exist.

### important source corrections

ψ₀ already uses multi-modal action conditioning. its original paper describes joint attention over action and vision-language representations, rather than an action-only sampler. therefore, adding semantic memory must be compared against the actual reference implementation, not a fictitious action-only baseline. [s1]

the newer ψ₀ sonic release uses an 80-dimensional interface, unlike the original 36-dimensional interface, and changes state handling and chunking behavior. these are separate controller/checkpoint contracts, not interchangeable settings. [s2]

topoformer's inspected attention core is a dense correctness implementation that returns attention weights. reuse its routing contract and validation philosophy, not an assumption that the reference function is already an optimized inference kernel. its composition report explicitly does not establish a general workspace efficiency advantage. [s3–s5]

## 3. component boundaries

| component | responsibility | boundary |
| --- | --- | --- |
| robot registry and morphology compiler | load assets, build hierarchical body graphs, attach compatible modules, validate physical/control structure | produces a versioned robot specification and controller contract |
| simulation and observation service | step physics, render sensor observations, expose declared sensors, snapshot and restore experiments | policy observations are separate from privileged truth |
| task runtime | compile supplied event graphs, maintain event instances, check guards, apply graph edits | produces current task context, not hidden future outcomes |
| semantic encoder | produce entity, morphology, event, relation, and frame representations | consumes only the observation channel selected for that experiment |
| action codec and system i | learn action targets, sample trajectories, decode into controller-native chunks | one shared parameterization across variable supported bodies |
| controller adapter and system 0 | track native commands at the controller rate | exposes capabilities, units, tracking errors, and reset semantics |
| workbench | robot visualization, graph editing, intervention, replay, probe inspection | all execution-changing edits are versioned backend transactions |
| training and evaluation | demonstrations, auxiliary learning, flow matching, adaptation, paired comparisons | fixed splits, manifests, measurements, and honest denominators |

these boundaries allow the first runnable experiment to use a small policy and deterministic teachers while the same interfaces later host the full pretrained perception stack.

## 4. robot and morphology representation

### static specification

a robot specification contains links, joints, actuator/transmission relations, sensors, geometry, and declared attachment interfaces. node features include type, axes, local transforms, limits, physical parameters where available, and sensor/actuator capability descriptors. independent controls, generalized coordinates, mimic joints, and virtual controller commands are explicitly distinguished.

nominal joint or product names may appear in the user interface and logs. they must not be required learned lookup keys for generalization. semantic roles and structural identity are generated from the graph, geometry, current task assignment, and declared interfaces.

coarse nodes represent assemblies such as an arm, gripper, manipulator, chassis, or whole body. they use permutation-invariant attention pooling with role and structural information preserved. a coarse node shares typed spatial interfaces with individual links, without pretending every aggregate is a rigid link.

### dynamic state

current node state is separate from static initialization: joint position and velocity, base pose/twist, declared effort or tactile measurements, and observation masks. relative positions change with joint configuration; a static learned initializer is not a substitute for forward kinematics.

compute exact body-relative geometry from measured proprioception and the known model when possible. learn uncertain object-relative geometry from available perception. compare learned spatial probes against analytic forward-kinematics baselines rather than presenting rediscovered kinematics as the research contribution.

### manipulation and morphology factorization

represent each manipulator with separate addresses for identity, semantic assignment, capability, physical subtree, and current interaction state. query the event bank using its role binding to obtain its current subtask representation. physical realization additionally sees morphology and capability.

the invariance target is conditional: preserve the intended object-level goal across compatible swaps, not every intermediate motion, grasp strategy, or feasible assignment. a new hand may require another grasp or may be unable to perform the task. capability information must remain available; adversarially deleting it from every representation would be counterproductive.

for coordinated manipulation, a task can bind several manipulators, and a manipulator can participate in several events over time. do not impose a one-hot event-to-hand assignment.

## 5. event hypergraph and execution semantics

### canonical event representation

an event has a stable instance handle, an operator identifier or embedding, typed role bindings, preconditions, invariants during execution, desired effects, a completion predicate, optional timeout/recovery policy, reference-frame bindings, and uncertainty. roles preserve order and multiplicity: actor, patient, instrument, source, destination, reference, support, and cooperating actor are not an unordered set of entity embeddings.

an event-instance node with typed incidence edges to entities and roles is the implementation representation of a hyperedge. this keeps n-ary semantics without reducing them to ambiguous pairwise similarity.

relations among events include prerequisites, overlap constraints, mutual exclusion, resource ownership, enabling effects, and explicit alternatives. the first runnable runtime uses an acyclic prerequisite graph for each active plan. contact and morphology graphs need not be acyclic. repeated actions and recovery attempts receive fresh event-instance handles rather than making their histories indistinguishable.

an example is: maintain support on a fixture while another manipulator aligns and inserts a peg. the support event overlaps the alignment/insertion events; it is not merely a completed predecessor. an insertion event binds a peg, a hole, a manipulating assembly, and a reference frame. its effects include insertion depth and alignment changes, while its invariants can include contact-force limits.

### plans, observations, and deltas

keep desired state changes distinct from observations. an effect says what an event should achieve; it does not certify that the effect occurred. an event's completion comes from the configured observable estimator, with a separate privileged evaluator available for diagnostics.

represent deltas according to their type: predicate transitions, relative transforms, scalar changes, or interval constraints. rotational and rigid-transform effects use composition or local error coordinates, not subtraction of arbitrary pose encodings.

a supplied graph is legitimate input in the initial experimental track. supplied future contacts, successful grasp labels, or the correct future branch are not legitimate observations. a vision-only task-discovery claim requires another track and is outside the first specification.

### topoformer-style routing

adopt the query-row/key-column convention: a read edge from event j to event i means j may preferentially read i. this is opposite the common causal drawing in which i enables j. the gui displays semantic dependencies; the compiler builds the correct read-oriented matrices. [s3]

use typed additive attention biases with zero-strength equivalence to the unstructured model. latent-to-node associations may later produce biases of the form `p_query · adjacency · p_keyᵀ`, with a null association for irrelevant tokens. learned associations are not treated as proof of correct planning.

hard rules constrain schema validity and runtime execution. supplied morphology edges are not a hard ban on all nonlocal physical influence. use soft routing for model communication by default and preserve a route for contact-mediated and whole-body dependencies.

train operator, ordered role-binding, prerequisite, effect, and completion probes separately. evaluate actual graph edits and role swaps through executed outcomes; an attractive attention visualization or a successful probe is not sufficient evidence that control uses the representation.

## 6. perception, predicates, contact frames, and uncertainty

canonical object slots are shared across queries. queries such as visible, looking-at, focused-on, acting-on, and held-by return pointers or distributions over those slots, including null and multiple-object cases. these relations are distinct, not a strict implication chain: a robot may intentionally act on an occluded object or visually inspect an object unrelated to the next manipulation.

visibility labels use declared camera geometry and occlusion during training. looking-at needs an operational definition such as optical-axis alignment; it is not inferred from a hypothetical human eye. task focus uses supplied active-event roles and, separately, future-interaction labels available only during training.

object question answering uses a learned projection into a separate frozen text model when enabled. the loss trains the projection and object representation; frozen text-model weights do not imply detached object features. text queries are auxiliary training or explicit inspection requests, never a compulsory extra language-model pass per action sample. prompt/view invariance preserves object identity without forcing all physically important geometry to mimic a text embedding space.

contact frames are active anchors, not automatically the most recent touch. each stores participants, local origin, observable surface directions, attachment to the contacted surface, age, validity, slip state, and pose uncertainty. a single point contact does not fully determine orientation or constrain every degree of freedom. unresolved tangent orientation remains uncertain or is resolved from an explicit feature, gravity, or another observation.

a remembered contact frame follows the estimated contacted object, rather than remaining incorrectly fixed in world space. loss of contact and uncertain object motion invalidate or widen the anchor belief. the controller may use this geometry directly; its usefulness is not restricted to shaping a denoiser's hidden representation.

uncertainty is attached to entities, poses, contact modes, predicate estimates, and feasibility. distinguish a false relation from an unavailable observation. do not multiply correlated per-head confidences and call the product a calibrated probability. evaluate calibration on held-out bodies and interventions. support contact-seeking or view-changing events as supplied actions first; autonomous information-gathering planning is a later experiment.

## 7. systems ii, i, and 0

### system ii and semantic memory

use a pretrained vision-language backbone through a thin feature adapter. keep the whole relevant visual/text feature sequence available through a bounded resampler; preserve direct node-addressed morphology context independently of what the language backbone retains. do not require the vlm to reconstruct a graph from arbitrary edge-name tokens.

construct four typed memory banks: task/event; scene/entity; morphology/state; interaction/frame. each carries masks, timestamps, and uncertainty where appropriate. shared encoding layers and small type-specific projections are the initial design; four banks do not mean four independently expensive foundation models.

static morphology can be cached across observations. current proprioception, scene beliefs, active task state, and contact frames cannot. encode their interactions once per new policy observation or explicit graph intervention.

### explicit continuous action targets

a learned action latent needs a training target. first train a graph-conditioned action codec on demonstration chunks. the encoder maps native action sequences and training-only effect annotations into a per-controlled-entity latent sequence; the shared decoder reconstructs native commands from latents, morphology, and current state. the decoder does not receive the original demonstrated actions or future truth.

codec objectives include masked action reconstruction, forward-kinematic/task-effect reconstruction, and compatible cross-embodiment effect alignment. effects and semantic projections are aligned across bodies, not arbitrary joint indices. require a held-out reconstruction and rollout gate before using the codec as a policy target.

freeze the codec for the first flow-learning and rl experiments. this prevents the target space moving underneath the policy and gives adaptation comparisons a stable action interface. a direct normalized-action flow baseline skips the codec and exposes whether the codec itself helps. later supervised codec adaptation is evaluated as its own intervention with its own cost.

### action stream

represent latent actions as `z ∈ r[h × n × d]`, where h is future action time, n is the number of exposed controllable entities, and d is a shared latent width. n is not necessarily the raw joint count. a controller that jointly owns the legs may expose a body-command group instead of simultaneous independent leg targets. mutually exclusive ownership prevents two controllers commanding the same actuator.

node identity, static morphology, dynamic state, and temporal position are supplied separately from gaussian noise. weights are shared across nodes. arbitrary row permutations must give corresponding output permutations when graph features and observations are permuted consistently.

fix one flow convention: `z(τ) = (1−τ)ε + τz_target`, with `τ=0` noise and `τ=1` data. train the velocity against `z_target−ε`; sampling integrates from 0 to 1. observation time, future action time, transformer depth, and flow time are distinct indices.

### cached context and latency contract

only the action stream is noised. clean context can influence every action block without being re-encoded every flow evaluation. per-layer context projections may be cached only when they are independent of flow time, current noisy actions, stochastic training operations, and any intervening context updates. frozen weights alone do not make a context stream cacheable.

bidirectional context updates, flow-time modulation of context, or task-graph edits invalidate those caches. the first architecture forbids action-to-context updates inside the sampler. bidirectional refresh is a separately measured ablation, not an invisible approximation to the original ψ₀ model.

use type-specific projections followed by a shared packed key/value bank where possible; do not automatically perform four full-width, sequential cross-attention passes per layer. relation metadata controls routing without necessarily materializing a dense bias tensor on the fast path.

with a = hn action tokens and c context tokens, dense all-token attention has quadratic term `(a+c)²`; static-context action self/cross attention has `a²+ac`, plus context construction and all projection/feed-forward costs. this is a structural comparison, not a runtime prediction.

tokenizing each node increases a relative to a model that uses one token per action timestep. therefore benchmark factorized temporal/node attention: its attention term is approximately `nh² + hn² + hnc` rather than `(hn)² + hnc`. feed-forward and projection cost still scales with hn. body-size bucketing, grouping coarse controllable entities, and bounded context are required scaling experiments.

start with a small architecture profile for correctness and learning-curve experiments, then scale only after measured bottlenecks. profile 1, 2, 4, and 8 flow evaluations; fewer evaluations are not assumed to preserve success. report synchronized p50/p95 sampling time, full observation-to-command latency, peak memory, context construction, codec decoding, controller time, and missed deadlines. gui rendering and human pauses are reported separately.

### native realization and system 0

preserve ψ₀'s high-level/low-level separation, not one universal fixed output vector. the original controller path used joint targets for arms/hands and a lower-body tracking interface; the sonic release uses a materially different command representation. [s1, s2]

provide typed adapters for direct joint-target tracking, the original ψ₀-style body interface, and the pinned sonic interface. use established controllers only on bodies/configurations for which their behavior has been validated. quadrupeds, hexapods, and modified humanoids need their own validated tracking or locomotion path; a urdf does not supply it.

count controller integration, calibration, controller training, and adaptation after manipulator surgery in deployment cost. distinguish policy transfer with an existing controller from end-to-end deployment on genuinely new hardware.

## 8. robot coverage and synthetic surgery

### asset policy

begin with native mujoco assets and a simulator-independent observation/action contract. simple integration is retained for ψ₀ comparison, but the core workbench must not require every experiment to use its rendering pipeline. support a checked accelerated backend later; do not assume every native model has the same accelerator compatibility. menagerie explicitly distinguishes ordinary models and optional mjx variants. [s6]

every asset entry records source revision, license, model file, geometry hashes, controllable coordinates, sensors, attachment ports, controller availability, task eligibility, and validation status. asset availability, successful import, stable tracking, teacher availability, and learned-policy evaluation are different states.

### candidate coverage, not claimed working integrations

verified candidate model families from the inspected menagerie inventory include: [s6]

| family | candidate inventory |
| --- | --- |
| humanoid/biped | unitree g1 and h1; talos; booster t1; toddlerbot 2xc and 2xm; adam lite; apptronik apollo; berkeley humanoid; fourier n1; cassie |
| quadruped | unitree a1, go1, go2; anymal b and c; barkour variants; spot |
| standalone/dual arm | franka panda and fr3; ur5e; unitree z1; lite6; sawyer; aloha; selected low-cost arms |
| mobile manipulator | stretch 2/3; tiago/tiago++; tidybot; google robot; rby1 |
| interchangeable end-effector | panda gripper; robotiq 2f-85; allegro; shadow hand; leap hand; dex-ee; existing validated dex3/inspire assets from the ψ₀/simple ecosystem |
| generated topology | procedural quadruped and hexapod bodies; parameterized arms; explicitly labeled synthetic body/arm/hand combinations |

these are not all manipulation-capable in their stock form, nor all independent morphology families. do not count controller variants or slight hand variants as independent bodies. generated hexapods avoid depending on an unverified third-party hexapod stack for the first integration.

### attachment and generation contract

cut and attach at declared interfaces with coordinate-frame, mass/inertia, actuation, collision, sensor, and control compatibility checks. preserve or explicitly update mimic/equality constraints and actuator limits. random geometric resizing must update physical quantities consistently or be labeled an intentionally nonphysical diagnostic.

use the simulator's actual physical and actuator model contracts rather than editing visual geometry alone. [s10] validate model compilation, finite dynamics, self-collision, controller stability, and teacher feasibility before adding a combination to the evaluation distribution. replacing a hand changes payload and contact dynamics; a previously stable whole-body controller must be rechecked.

train on diverse bodies and combinations; reserve whole attachment combinations and entire morphology families for evaluation. future target-body policy rollouts, demonstrations, and target normalization statistics are excluded from the zero-shot track. declared physical specifications such as joint limits remain legitimate input.

## 9. robot and task-graph workbench

use a browser workbench with a python simulation/policy service. the proposed frontend is react with a three-dimensional scene viewer and a react-flow task editor; the backend, not frontend graph propagation, owns execution semantics. react flow is an editor building block, not the task planner or safety authority. [s7]

first-run capabilities: load a robot and scene, pause/resume, step physics or one policy chunk, reset, move a target with a gizmo, inspect a joint/manipulator, visualize contacts/reference frames, record a trace, and inspect task-event status. training workers remain independent of gui frame rate.

show the event graph with actor/object bindings, preconditions, effects, completion estimates, confidence, and active/waiting/failed status. clicking a task node selects it for inspection. moving its screen position changes layout only. changing priority or requesting another active event is an explicit execution command.

execution requests validate preconditions and resource compatibility. requesting a future event does not mark predecessors complete or teleport the robot. invalid requests explain which guard is unsatisfied. a deliberate privileged debug override is visibly labeled and excludes that run from ordinary evaluation.

graph edits are versioned transactions. pause, validate schema and prerequisite cycles, commit a new graph version, invalidate affected latent/context caches, cancel stale unsent commands, and replan from the observed physical state. permit rollback of the graph edit without pretending that executed physical actions have been undone. full simulation rewind uses a separately stored snapshot.

record graph versions and edits, robot/controller/model revisions, observations, sampled action chunks, flow seeds, and completion decisions. deterministic replay is tested only within a fixed declared backend and numerical configuration. a snapshot includes simulator, task-runtime, controller, sensor-history, and policy-memory state when they affect continuation.

privileged overlays may be displayed for research, but are visibly separated from estimated policy beliefs and never sent into the policy channel accidentally.

## 10. learning and adaptation protocol

### representation and behavioral learning

first collect teacher trajectories from validated planners, scripted task controllers, or teleoperation. record failures and feasibility exclusions, not only successful demonstrations. annotations include canonical objects, task bindings, relation changes, contact modes, and relative geometry.

train the codec and shared graph-conditioned behavioral policy before adding every auxiliary objective. introduce predicate/geometry supervision, compositional-event supervision, and compatible-swap invariance as independently removable groups. use matched model capacities, data, and optimization budgets for attribution.

validate object and manipulator-role consistency with interventions, not only reconstruction. examples include exchanging actor bindings, replacing only an irrelevant object, swapping a hand while keeping a feasible goal, and suppressing a contact anchor while controlling for otherwise identical observations.

### supervised new-body adaptation

report the zero-update result first. then evaluate predeclared demonstration budgets of 5, 20, and 100 episodes, with their actual trajectory lengths, generation costs, and gradient work reported separately. begin with the realization adapter and selected action-policy modules. unfreeze perception only when development diagnostics identify a perception limitation, then keep that choice fixed for evaluation.

### reinforcement learning

grpo is a planned adaptation method, not a guaranteed sample-efficiency property. z-1's relevant ideas include shared-prefix rollout reuse, task-critical branching, and a flow-sde likelihood construction. [s8] use event boundaries as candidate branch points, while preserving the complete continuation state and excluding reused fixed prefixes from inappropriate gradient credit.

for a latent-flow grpo experiment, keep the codec decoder and controller fixed. optimize the stochastic latent-generation policy with recorded transition log probabilities. do not substitute flow-matching error for a likelihood ratio or assume a deterministic ode directly provides the needed probability calculation. changing the latent-to-control decoder during the same importance-ratio update requires a separately justified objective.

compare no adaptation, supervised adaptation, and grpo at matched data/interaction budgets. retain an expo-ft-style off-policy comparator when online sample efficiency is the limiting factor. expo-ft combines an edit policy with updates to the base vla in a unified loop; it is not merely a permanently frozen vla plus a residual controller. account for base-model update cost in that comparison. [s9] do not require grpo to win to consider the representation successful.

initial online checkpoints are 0, 10,000, 50,000, and 200,000 newly executed simulator transitions. separately report optimizer presentations, branching reuse, replay use, policy calls, rendering, and wall time. repeated replay is not new environment experience. shared prefixes are counted once in physically executed simulation work, while all policy-training reuse remains visible.

## 11. experimental scope and success criteria

### first runnable integration

use a fixed-base arm with two compatible end-effectors, plus a second arm morphology, on a short pick/place or insertion task. expose the supplied event graph in the gui. the initial visible demonstration may use a teacher; its label must clearly distinguish teacher execution from learned control. this validates the interfaces before expensive policy training.

include one known same-body hand-swap pair from the g1 ecosystem in the subsequent paired data test. do not make walking on every imported humanoid a prerequisite for testing the manipulation representation.

### first scientific claim

hold out an attachment combination and an entire arm/body morphology. compare a matched shared morphology-conditioned policy with and without event/affordance structuring. the primary quantity is new-body interaction required to reach 80% task success, together with adaptation compute and integration work. zero-shot success is a separate coordinate on that trade-off.

the initial aspirational go/no-go target is a twofold reduction in new-body interactions to that success threshold, without more than 25% p95 policy-inference overhead versus the matched unstructured model. these are chosen research targets, not source-backed predictions. absence of that gain is a valid result and triggers simplification rather than redefining the threshold after seeing outcomes.

freeze eligibility, tasks, target conditions, budgets, and analysis before the confirmatory run. use at least three independent training seeds and 100 evaluation episodes per reported task/body cell for the first confirmation; report intervals and individual cells rather than hiding failure behind a global mean. use calibration/development rollouts for tuning, not final evaluation episodes. all such rollouts incur cost.

when a method never reaches the threshold within budget, report it as censored and show its success curve; do not invent a finite crossing time. below-threshold exploratory results remain developmental. a high raw score on an impossible-task-filtered subset does not certify arbitrary-robot transfer.

### required comparisons and interventions

1. same model/observations with structural gates zeroed; graph-as-input without attention bias; structured routing with true, reversed, and rewired relations.
2. action-codec versus direct-native-action targets; state-supervised diagnostics versus deployable sensor observations.
3. no auxiliary heads versus geometry/contact heads versus full event/role supervision; keep extra privileged information out of deployment inputs in every arm.
4. no swap augmentation versus compatible counterfactual swaps; left/right and event-role reassignment tests; held-out whole-body and controller-interface changes labeled separately.
5. no adaptation, supervised fine-tuning, and grpo; original ψ₀ reproduction only on its supported controller/task contracts.

### engineering acceptance

model tests cover shape/mask validity, independent-control versus mimic handling, node permutation equivariance, null bindings, flow-time conventions, cached/uncached equivalence, and finite gradients. runtime tests cover missing observations, stale graph versions, cycles, unsatisfied guards, conflicting controller ownership, snapshot integrity, and no privileged-state leakage.

integration checks cover physics stepping, a valid teacher trace, object/manipulator selection, target movement, graph intervention, replay, and changed-body loading. the policy must tolerate variable node counts without a robot-id output table. supported tracking controllers must maintain their declared stability and command limits.

latency is measured on the actual execution host after warmup and with device synchronization where needed. no hardware throughput, gpu memory fit, universal controller availability, or accelerated-physics support is assumed from a repository readme alone.

## 12. scope boundaries and next artifact

this specification covers the research architecture and the first executable workbench/transfer slice. later work includes broad locomotion transfer, larger foundation-model training, learned event discovery, autonomous scheduling, production hardware deployment, and separately justified joint controller/policy adaptation.

before executing an implementation plan, perform a read-only local capability audit, record exact library/model/source revisions, and define a bounded compute envelope from available hardware. no automatic cloud spending or new hardware purchase is part of this design.

the next artifact is a file-level implementation plan with runnable tests and acceptance commands, beginning with robot/task/observation schemas and the first gui-controlled simulation trace. source integrations remain separate from modifications to topoformer's ongoing research repository.

## sources inspected

[s1] ψ₀ original paper, architecture, action contracts, and flow conditioning. https://arxiv.org/html/2603.12263v1

[s2] ψ₀ sonic release note, retrieved 2026-09-21 utc; inspected blob `988f77230eab5d8d7c48aeaed3d0b55914e8ec68`. https://github.com/physical-superintelligence-lab/Psi0/blob/main/examples/psi0_for_sonic.md

[s3] topoformer initial design and directed attention convention; inspected blob `cec933f8093c3acdd60ed4e1d77898891981dfbc`. https://github.com/JacobFV/topoformer/blob/main/research/stages/stage-01/design.md

[s4] topoformer attention correctness implementation; inspected blob `40b38a09ff9fd7410a5609f8fb01fedc772dacd7`. https://github.com/JacobFV/topoformer/blob/main/src/topoformer/attention.py

[s5] topoformer composition scope and efficiency boundaries; inspected blob `806ce975454a17ec62ff573b6f58cd25821c91d2`. https://github.com/JacobFV/topoformer/blob/main/research/campaigns/extended-01/composition/C04-claim-prerequisites-and-next.md

[s6] mujoco menagerie model inventory and native/mjx distinction, retrieved 2026-09-21 utc; inspected readme blob `4c4b894226748252a1c41062728da7493fe575b4`. availability is not a completed controller integration. https://github.com/google-deepmind/mujoco_menagerie/blob/main/README.md

[s7] react flow documentation for graph-data interactions. https://reactflow.dev/learn/advanced-use/computing-flows

[s8] z-1, shared-prefix and flow-sde grpo methodology. https://arxiv.org/html/2606.31846v1

[s9] expo-ft, sample-efficient adaptation comparison. https://arxiv.org/html/2605.25477v2

[s10] mujoco modeling reference for physical and actuator model contracts. https://mujoco.readthedocs.io/en/stable/modeling.html
