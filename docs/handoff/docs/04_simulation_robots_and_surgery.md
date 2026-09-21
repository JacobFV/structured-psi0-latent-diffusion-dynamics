# simulation, catalogue, controllers, and morphology surgery

## backend strategy

Native MuJoCo is the required reference backend. Use a separate renderer/sensor service and explicit observation/control boundaries. Preserve a SIMPLE adapter for ψ₀ reproduction, but do not force every experiment through an expensive or unsupported renderer. Use an accelerated backend only for assets it genuinely supports after numerical/control equivalence testing. ARM64 compatibility and GPU memory fit are measured locally, not inferred from repository labels.

Use primitive geometry for first CI physics tests; no internet/model weights needed. Adopt meters, seconds, kilograms, radians, right-handed frames, and explicitly documented quaternion component order. Each controller declares its own rate, and policy chunks declare dt. Tests detect implicit 20Hz/50Hz assumptions, wrong joint order, and normalized/physical unit confusion.

## catalogue statuses and candidates

Each row in `config/robot_candidates.json` starts as a candidate. Populate a pinned resolved registry with source URL/commit/license, downloaded assets and hashes, native file, backend compatibility, independent control count, sensor profile, ports, task eligibility, and evidence paths. Progress states: candidate -> source_verified -> imported -> physics_validated -> controller_validated -> teacher_validated -> policy_evaluated. A stage may fail with a reason; failure is not equivalent to absence.

Import breadth opportunistically under disk/network caps, but prioritize the first arm vertical slice. All candidates must receive a source/licensing/status audit. Do not describe every candidate as executable. Commercial body variants, hands, fixed-base versions, and controller versions are separately labeled. Record asset lineage so near-duplicates are not treated as independent holdouts.

Mandatory executed breadth is in the scope document. A stock lower-body humanoid without hands is eligible for locomotion, not dexterous insertion. Whole-body/floating-base evaluation requires a stable tracker; fixed-base upper-body execution has its own label. A quadruped plus a newly attached arm is synthetic and needs post-attachment balance validation.

## minimal task suite and teachers

`reach_pose`: body/task-frame target tracking; no-grasp control sanity baseline.
`pick_place`: acquire object, transport to target, release; object binding and canonical predicates.
`support_insert`: maintain support while aligning and inserting; contact frames, concurrency and two-role assignment. Use two manipulators or one manipulator plus an explicit fixture-support role; these are separate capability conditions.
`handover`: source holds until receiver establishes grasp, then release; multiple held-by states and overlapping events.
`move_base_then_reach`: locomotion/mobility followed by reachable manipulation; declared controller groups.
`waypoint_contact`: biped/quadruped/hexapod locomotion and stance/contact transitions, without claiming all bodies can grasp.
`occluded_align`: identical task with partial visual occlusion and available contact/proprioception; uncertainty and anchor utility.

Teachers can be tested IK/trajectory planners, finite-state contact controllers, existing RL trackers, or teleoperation. They supply demonstrations, not model results. Validate each task/body teacher independently before policy evaluation, including boundary conditions and perturbations. Preserve unsuccessful attempts and feasibility rejection counts. A teacher that uses privileged object poses remains explicitly privileged. Ensure tasks are solvable using the declared deployment sensors, even when training teachers know more.

Provide controller families: position/velocity targets for fixed arms and grippers; ψ₀ original and SONIC adapter contracts on validated bodies; body-specific locomotion trackers or compact trained trackers for the procedural legged families. Trackers can be RL-trained with privileged information, but deployment observations and retraining cost are recorded. Stock pretrained WBC support does not imply arbitrary-surgery support.

## module attachment

An attachment port specifies host link/frame, allowed mechanical dimensions, attachment transform, mass/payload bounds, collision exemptions only where justified, electrical/control capability contract, sensor mounts, and actuator ownership. A module exposes its root link and required conditions. Every assembly has a fresh asset/spec hash and a recorded source lineage.

Procedure: parse semantic asset -> choose validated port -> transform and attach subtree -> namespace link/joint/actuator/mesh/constraint names -> rebind transmissions/equalities/mimics/sensors -> reconcile dynamics and limits -> compile -> run collision and finite-dynamics tests -> calibrate/validate tracker -> run teacher feasibility -> enroll in the dataset. Do not just concatenate XML strings and ignore references.

Vary hand shape/DOF, gripper type, arm length, joint count/type/order, payload and base. Uniform scale s at constant density changes mass by s³ and inertia by s⁵; control gains and torque limits still need a justified policy. Nonuniform shape changes require appropriate inertia recomputation. An intentionally nonphysical diagnostic is allowed only when labeled and excluded from physically plausible transfer claims.

Start with same body + two valid hands. Then hold out body-hand combinations and whole arm modules. Split by morphology lineage before sampling episodes. Do not let near-identical generated variants cross train/test boundaries unnoticed. Feasibility may change under a swap; preserve goal meaning while allowing different strategies, or explicitly mark infeasible.

## generated hexapods and unusual topologies

Build a procedural 6-leg, 3-joint-per-leg fixture with positive inertias, bounded actuators, foot contacts and a validated baseline gait/controller. Add variable link lengths and 4/6/8-leg diagnostic generators only after the nominal one is stable. This removes reliance on an unverified external hexapod repository. Measure stability, slip, distance and energy proxies without claiming transfer of dexterous semantics to legged-only tasks.

A procedural robot is a real simulated dynamical system, not a 2D drawing. Include falling/failure termination and realistic controller state. Use new topology holdouts to test tensor/mask correctness separately from behavioral competence.

## sensor and dataset records

Store episode identifier, split lineage, robot/controller/task/graph hashes, seed, control/physics rates, observations, actions, event receipts, resource attribution, and private labels in separate files/fields. Canonical object labels and simulator contacts supervise training; inferred slots and declared sensors drive deployment.

Chunk datasets retain masks, dt and frame conventions. Normalize from training statistics only, or declared physical specs that are available for a new robot. A zero-shot target cannot contribute demonstration normalization. Frozen visual feature caches include backbone, processor, augmentation, image and source hashes. Do not use cached features across different camera observations.

Data generation and video rendering are independently throttled. Keep sparse diagnostic videos instead of every rollout. Stream low-bandwidth workbench views; learner throughput must not depend on an open browser tab. Exact snapshot replay includes controller and runtime state, not just qpos/qvel.
