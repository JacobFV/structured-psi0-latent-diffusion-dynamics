# robot training: problems, our status, and what we do about them (2026-09-26)

Living checklist. Status: ✅ handled · ◐ partial · ❌ missing. Every item names where it lives or who owns it.
Update the status line when something changes and cite the decision (D-xxx) or track note that verifies it.

## 0. Why this document exists
The user observed that the legged robots and humanoids "jitter across the floor rather than walk" and that the scripted
trajectories look poor. Our own tracker validation confirms the first point (below). This is a systematic list of the problems
robot training runs into, so they are handled deliberately rather than discovered one by one.

## 1. Diagnosis that triggered this: legged skating (D-093)
Source: `artifacts/trackers/<body>/validation_learned.json`, forward trial, 5 seeds.

| body | body speed (m/s) | stance-foot slip (m/s) | tracker gate |
|---|---|---|---|
| t1 | 0.36 | 0.40 | fail |
| g1 | 0.44 | 0.29 | fail |
| h1 | 0.33 | 0.21 | fail |
| go2 | 0.63 | 0.30 | pass |
| anymal_c | 0.47 | 0.26 | pass |
| hexapod6 (learned) | 0.16 | 0.16 | pass |
| hexapod6 (CPG) | 0.17 | 0.06 | pass |

Causes:
- Physics options for t1/h1/g1/procedural bodies: pyramidal cone, `impratio 1`, `noslip_iterations 0`, default soft contact
  (solref 0.02), floor friction 1.0. The menagerie go2 uses an elliptic cone with impratio 100.
- PPO reward `feet_slip` −0.05 (negligible). Friction is randomized only as one sliding scale per worker.
- The gate had no slip criterion, so skating trackers passed.
Fix in progress: contact model v2 + reward schedule + slip gate (track `contact`, research/tracks/contact.md).
All legged results up to D-092 were produced with contact v1 and must be read with that caveat.

## 2. Where expert behaviour comes from (scripted vs RL)
- **Legged:** the scripted part only emits base-velocity commands toward waypoints (`src/rrp/control/legged_teachers.py`).
  The gait itself is a PPO-trained tracker per body (`src/rrp/control/tracker_training.py`). This matches Ψ₀
  (arXiv 2603.12263), where the learned model commands an RL tracking policy (AMO) with base velocity, height and torso
  orientation, and the command stream comes from teleoperation. Our weakness is the tracker quality, not the command source.
- **Arms:** a scripted waypoint pick-place planner is the expert. It looks mechanical and fails on some bodies.
- **Literature (fetched 2026-09-26, via a summarising fetch tool):**
  - Ψ₀: ~829 h egocentric human video, then ~30 h real teleoperation on Unitree G1; no simulation.
  - arXiv 2601.15419 (unified cross-embodiment latent): HumanML3D motion retargeted kinematically; no contact or physics.
  - Z-1 (arXiv 2606.31846): SFT on 1,199 RoboCasa demos, then GRPO in simulation with sparse success reward (67.4% → 80.6% over 24 tasks).
- **Policy for this project:**
  - RL for low-level controllers.
  - Demonstrations (scripted, smoothed) as the arm bootstrap.
  - RL fine-tuning (GRPO with a success reward) to pass the script's ceiling.
  - Human data only if a licensed source becomes available (we have no teleoperation hardware).

## 3. Reward schedule: shaping priors → natural objectives (design, D-093)
Three groups of terms:
- **Permanent:** velocity tracking, falls/termination, orientation/height sanity, joint limits, **stance-foot slip** (never decays),
  the swing **clearance floor**, and **standing on a zero command** (both feet down, no joint drift, no body sway).
- **Shaping priors (decay to ~10%):** air time, swing height / foot clearance, contact phase / gait clock, biped symmetry.
- **Natural objectives (ramp up):** cost of transport Σ|τ·q̇| / (m·g·|v|), torque², action rate / jerk, foot impact force.

Schedule:
- One scalar α ∈ [0, 1] moves the weights.
- α advances only through performance gates (rolling tracking error, fall rate and slip ratio within thresholds) and backs off on regression.
- α is not observed by the actor.
- α and all effective weights are logged per iteration and in checkpoint metadata.

Test: compare α = 0 against the final α for slip ratio, tracking, falls, duty factor and CoT, with videos. Report a collapse to shuffling if it happens.

**Revision 2026-09-27 (W1, lead request): standing is permanent, and stops are ≥ 10% of commands.**
- *Change:* `stand_contact` (both feet down under a zero command) moved from the shaping priors to the permanent group, together with
  `stand_still` and a new `stand_vel` (−1.5 × |v_xy| at zero command). Every command sampler now issues at least 10% zero commands
  (`MIN_STOP_SHARE`; the W8 teacher-mix sampler had 3%).
- *Reason:* standing still on command is a task requirement, not a gait prior. The W8 `halt` event checks both feet down and body
  speed ≤ 0.1 m/s. With `stand_contact` at its 10% prior floor (α = 1) and few stop commands, the sourced-limit t1 swayed at the halt (3/15
  waypoint failures). A strong speed penalty alone (−4) made it step in place instead (9/20 failures).
- *Scope:* new trainings only. Installed trackers need no retraining unless they fail the halt check.
- Code: `rrp.envs.legged_core` (`PRIOR_TERMS`, `PERMANENT_STANDING_TERMS`, `MIN_STOP_SHARE`); test `tests/unit/test_reward_schedule.py`.

## 4. Checklist

### 4.1 Physics fidelity
- ❌→in progress: foot–floor contact (elliptic cone, impratio 10–100, noslip, compliant sole via solref/solimp). Owner: contact track.
- ❌ Static vs kinetic friction: MuJoCo has one Coulomb coefficient; approximate stiction with the cone and noslip, and document it.
- ◐ Contact compliance of soles and fingertips: defaults, never tuned; penetration unmeasured.
- ◐ Actuator model: ideal PD with effort limits. Missing: armature (reflected rotor inertia), joint friction and damping,
  torque–speed curves, saturation, backlash.
- ◐ Integrator and timestep: dt 0.002, no convergence check (dt 0.001) and no solver-iteration check under many contacts.
- ◐ Collision geometry and self-collision: taken from menagerie, not audited per body.
- ◐ Grasp physics: fixed cube friction and density; sim grasps too easy (penetration holding); slip under load untested.

### 4.2 Domain randomization and robustness
- ◐ Friction: one sliding scale per worker. Needed: a range for sliding, torsional and rolling friction.
- ❌ Mass, CoM, inertia, link-length randomization.
- ❌ Motor strength, PD gains, joint offsets (encoder calibration).
- ✅ Pushes (tracker training only). ❌ Pushes in arm and latent-route evals.
- ❌ Terrain: flat only. Needed: slopes, steps, rough and soft ground, a terrain curriculum.
- ❌ Object variety (arm): size, mass, friction, shape, distractors.
- ❌ Robustness sweeps: evaluate each policy on a physics-parameter grid and report the break-point. This substitutes for a real robot.

### 4.3 Sensing and state
- ◐ Privileged-information audit of deployable observations (lesson from B-1). Needs an automated check per input group.
- ◐ Sensor noise: observation noise in tracker training only. No IMU bias or drift, no encoder quantization, no dropout.
- ❌ State estimation: true base velocity and pose used in places; deployable policies should use estimated state.
- ❌ Vision in control (the VLM is a smoke test only); later: lighting, texture and camera-pose randomization, occlusion.
- ◐ Touch: binary contact; no force or slip sensing.

### 4.4 Timing
- ❌ Actuation and communication latency (5–30 ms) not modelled. Train and evaluate with randomized latency.
- ◐ Control-tick jitter and missed ticks not injected (a deadline-miss counter exists).
- ✅ Inference latency measured (p95 overhead 1.014×, D-058); not on edge hardware.

### 4.5 Expert and data quality
- ◐ Arm scripted expert: smooth it with time-parameterized, minimum-jerk trajectories; then RL fine-tuning.
- ◐ Legged tracker skating (see §1).
- ◐ Recovery coverage: DAgger only. No deliberate perturbation data, no failure or regrasp demonstrations.
- ✅ Multimodality handled by the flow model.
- ◐ Cross-body semantic label consistency is assumed, not verified.
- ✅ Provenance and licences; third-party assets excluded from the public repo.
- ❌ Physics/contact version in dataset and model metadata (required before contact v2 data exists).

### 4.6 Learning pathologies
- ✅ Covariate shift: DAgger (BC expert, generated-packet states).
- ✅ Shortcuts found: B-1 previous action (D-044/045), velocity copy. Keep causal input audits routine.
- ✅ Loss balancing: D-085 unbounded NLL starved the sem system 0. Make per-loss gradient-norm monitoring standard.
- ❌ Catastrophic forgetting during fine-tuning or RL: anchor evaluations needed before GRPO.
- ◐ Seed variance: 3 seeds legged, 1–2 arm. Report per-seed results and permutation tests.
- ◐ Simulator exploitation (skating now; watch penetration holding and contact vibration).

### 4.7 RL
- ◐ Reward hacking: gate on video review and gait statistics, not the reward curve.
- in progress: reward schedule (§3).
- ◐ Sparse-reward exploration (arm GRPO): start from a competent policy; group size and prefix branching as in Z-1.
- ◐ Throughput: CPU PPO. MJX on the peer GPU could give 10–100× more simulation steps.
- ✅ Asymmetric actor-critic. ✅ Matched budgets for competing methods (contract).

### 4.8 Cross-embodiment
- ❌ Sealed held-out target bodies untested on the latent route (BC: xarm7 0/100, D-064).
- ◐ Actuator and scale normalization across bodies.
- ◐ Morphology coverage: many bodies, few families per task.
- ◐ Negative transfer risk (t1 under the D-085 defect).

### 4.9 Task and environment diversity
- ❌ One arm task, one legged task; dual-arm teacher data only.
- ❌ Clutter, articulated or deformable objects, long-horizon multi-step tasks.
- ❌ Loco-manipulation (the humanoid use case in Ψ₀).

### 4.10 Evaluation
- ✅ Paired seeds, CIs, fresh seed sets, irrelevant-edit controls.
- ◐ Motion-quality metrics: add slip ratio, CoT, jerk, contact forces and joint-limit margin to every legged and arm eval, plus video review.
- ❌ Robustness sweeps (see 4.2). ❌ Long-duration runs (minutes) for drift.

### 4.11 Safety and hardware
- ◐ Joint and torque limits: reward penalties only; add a deployment safety layer (clamp, rate limit).
- ❌ Fall and impact handling, safe stop, fall recovery (t1 packet edits already cause falls, D-092).
- ◐ Packet admission exists; ❌ OOD packet detection with a fallback controller.
- ❌ Thermal and duty-cycle limits.

### 4.12 Infrastructure
- ✅ Reproducibility: fingerprinted bundles, recorded commands, raw artifacts.
- ◐ Compute contention: host memory shedding (external process); peer CPU is the simulation bottleneck.
- ◐ Repo structure: see the structure audit and consolidation plan (`docs/strategy.md`).

## 5. Priorities
1. Physics realism: contact v2 + reward schedule + slip gate, then actuator realism (armature, joint friction, motor limits) and randomized latency.
2. Randomization + robustness sweeps with break-points per policy.
3. Motion-quality gates on trackers and datasets.
4. Arm: smoothed scripted expert, then GRPO fine-tuning with anchor evaluations.
5. Rough terrain and GPU simulation (MJX) for trackers.
6. The untested claims: held-out target bodies, one loco-manipulation task.
