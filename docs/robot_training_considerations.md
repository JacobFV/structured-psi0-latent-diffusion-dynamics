# robot training: problems, our status, and what we do about them (2026-09-26; statuses revised 2026-09-27)

Living checklist. Status: ✅ handled · ◐ partial · ❌ missing. Every item names where it lives or who owns it.
Update the status line when something changes and cite the decision (D-xxx) or track note that verifies it.

**Revision 2026-09-27 21:30 (records agent, D-123 item 10):** every §4 status re-checked against the code on origin/main
(68a6657) and the "Checked and implemented" list in `docs/experiments_roadmap.md (backlog appendix)`. Items that are implemented but opt-in or
eval-only are ◐, with the reason. Open items are tracked with evidence and size in `docs/experiments_roadmap.md (backlog appendix)`.

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
Fixed: contact model v2 + reward schedule + slip gate (D-101, D-103; .old/research/tracks/contact.md). Stance slip v1 → v2: t1 0.86 → 0.151,
h1 0.57 → 0.13, g1 0.42 → 0.08, go2 0.26 → 0.02, anymal_c 0.34 → 0.03. W8 regenerated anymal_c and go2 on contact v2 (D-105, D-113).
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
- Code: `rrp.envs.mujoco.legged_core` (`PRIOR_TERMS`, `PERMANENT_STANDING_TERMS`, `MIN_STOP_SHARE`); test `tests/unit/test_reward_schedule.py`.

## 4. Checklist

### 4.1 Physics fidelity
- ✅ Foot–floor contact v2: elliptic cone, impratio 10, floor priority, compliant sole (solref 8 ms), slope stick/slide test
  (`src/rrp/bodies/contact.py`, `tests/unit/test_contact_model.py`; D-101). Installed trackers: anymal_c, go2, t1 w8d.
- ◐ Static vs kinetic friction: MuJoCo has one Coulomb coefficient; stiction approximated by the elliptic cone and high impratio (slope creep
  0.1 mm/s at 0.9 μ); `noslip_iterations` measured but left at 0; documented in `bodies/contact.py` (D-101). No kinetic-friction proxy, by choice.
- ✅ Contact compliance of soles and fingertips: sole solref/solimp tuned and penetration measured (0–2 mm, D-101); finger/object contact
  grasp_v2 (penetration 0.03 mm on the rig) and grasp_v2.1 (stiff contact on all robot geoms that can touch the cube) (D-110, D-118).
- ◐ Actuator model: `src/rrp/bodies/actuator.py` (armature, damping, friction, torque–speed, 0–30 ms latency) exists but is OPT-IN (D-103);
  sourced torque limits are the default (`sourced_v1`, D-107). Max joint speeds are still estimates; trackers do not hold their gaits
  under it (h1 no-fall 0.72). No backlash.
- ◐ Integrator and timestep: dt min(source, 0.002), 10 substeps per 50 Hz tick (contact v2); no dt 0.001 convergence or solver-iteration check.
- ◐ Collision geometry and self-collision: taken from menagerie, not audited per body.
- ✅ Grasp physics: grasp_v2 removes interpenetration holding (held penetration tf3 3.3 → 0.09 mm; episodes > 3 mm 2407 → 0); slip under load
  matches the friction prediction (1.06×); sourced grip forces (D-110). grasp_v2.1 + phase-gated DART for v6dart (D-118, D-121).
  `src/rrp/bodies/grasp_contact.py`, `tests/unit/test_grasp_contact.py`. Object friction/mass still fixed in training (see 4.2).

### 4.2 Domain randomization and robustness
- ✅ Friction (tracker training, contact v2): floor μ ~ U(0.4, 1.25) with torsional/rolling scaled, plus solref time constant / damping
  randomization (D-101). Arm: friction is an eval perturbation only.
- ◐ Mass, CoM, inertia: root mass ×U(0.9, 1.1) and CoM ±2 cm in tracker training (contact v2); whole-body mass/CoM as eval perturbations
  (`src/rrp/envs/mujoco/perturb.py`, D-108). ❌ Link-length randomization.
- ◐ Motor strength, PD gains: PD-gain scaling as an eval perturbation only (`perturb.py`); ❌ in training; ❌ joint offsets (encoder calibration).
- ✅ Pushes: tracker training; legged, arm and latent-route evals through the robustness harness (`push_impulse_Ns`, D-108).
- ◐ Terrain: heightfield bumps as an eval perturbation (`terrain_amp_m`, break-point 8 cm on anymal_c, D-112). ❌ Slopes, steps, soft
  ground and a terrain curriculum in training.
- ◐ Object variety (arm): distractors exist (`envs/mujoco/scenario.py`); cube size fixed at the default; object mass/friction are eval perturbations
  only; no shape variety.
- ✅ Robustness sweeps: `rrp.harness.eval.robustness` (one factor at a time + all_moderate, paired seeds, Wilson CIs, break-points) (D-108);
  variant-level anymal_c over 3 training seeds (D-112). Arm sweep under grasp_v1 was not meaningful (D-108) and has not been repeated under grasp_v2.

### 4.3 Sensing and state
- ◐ Privileged-information audit of deployable observations: `core/channels.py` rejects privileged keys on transport; no automated
  per-group causal/ablation audit (D-123).
- ◐ Sensor noise: observation noise in tracker training only. ❌ IMU bias or drift, encoder quantization, dropout.
- ◐ State estimation: tracker actors use IMU + encoders only (true base velocity goes to the critic); system i's speed comes from a declared
  localization = truth + Gaussian noise; `base_vel_estimate` is never filled (D-123).
- ❌ Vision in control (the VLM is a smoke test only; the Ψ₀ line in psi1z uses rendered images, D-120); later: lighting, texture and
  camera-pose randomization, occlusion.
- ◐ Touch: binary contact; no force or slip sensing.

### 4.4 Timing
- ◐ Actuation latency: 0–8 ms per-episode latency in contact-v2 tracker training (D-101); 0–30 ms in the opt-in actuator model and as an
  eval perturbation (no break-point on anymal_c, D-112). Not the default for W8 (D-103, D-107).
- ◐ Control-tick jitter and missed ticks not injected (a deadline-miss counter exists in `harness/eval/latency.py`).
- ✅ Inference latency measured (p95 overhead 1.014×, D-058); not re-measured on current routes, not on edge hardware.

### 4.5 Expert and data quality
- ✅ Arm scripted expert: teacher v2, minimum-jerk and touch-confirmed grasp, 4811/4811 feasible episodes, jerk 385–625 → 36–55 rad/s³
  (D-097, D-102); arm dataset gate enforced (D-112, D-114). RL fine-tuning not started (see 4.6).
- ✅ Legged tracker skating fixed on accepted v2 trackers (§1; D-101); tracker and dataset gates enforced (slip < 0.15, D-112, D-114).
  ◐ humanoids: t1 w8d fails the lab gate (forward 0.72, joint margin −0.053) and its dataset fails the slip gate (D-113 exception); h1/g1 parked.
- ◐ Recovery coverage: DAgger plus DART noise; v6dart DART is phase-gated (pregrasp + transport only), so no deliberate perturbation in the
  final descent/grasp (D-121). No failure or regrasp demonstrations.
- ✅ Multimodality handled by the flow model.
- ◐ Cross-body semantic label consistency is assumed, not verified.
- ✅ Provenance and licences; third-party assets excluded from the public repo.
- ✅ Physics version in dataset and model metadata: `contact_version`, `actuator_limits`, `grasp_contact_version` and the DART mode in
  provenance (`src/rrp/core/provenance.py`, W3; D-107, D-110, D-121); the legged pipeline refuses mixed contact versions.

### 4.6 Learning pathologies
- ✅ Covariate shift: DAgger (BC expert, generated-packet states).
- ✅ Shortcuts found: B-1 previous action (D-044/045), velocity copy. Keep causal input audits routine.
- ◐ Loss balancing: D-085 unbounded NLL starved the sem system 0; trainers log grad norms (`clip_grad_norm_`), but per-loss gradient-norm
  monitoring is not standard, and one logger was silently broken (psi1z, D-109).
- ❌ Catastrophic forgetting during fine-tuning or RL: anchor evaluations needed before GRPO (no anchor eval in `harness/train/latent_grpo.py`, D-123).
- ◐ Seed variance: 3 training seeds legged (D-105, D-112, D-113), 2 arm (D-095). Per-seed results and exact permutation tests reported.
- ◐ Simulator exploitation: skating fixed (D-101) and interpenetration holding fixed (D-110); DART noise crushing the cube found and
  gated (D-118). Keep watching contact vibration.

### 4.7 RL
- ◐ Reward hacking: gate on video review and gait statistics (tracker gate in code, D-114), not the reward curve.
- ✅ Reward schedule (§3): permanent / shaping-prior / natural-objective groups with a gated α (`envs/mujoco/legged_core.py`,
  `tests/unit/test_reward_schedule.py`; D-101); permanent clearance floor (D-103), permanent standing and `MIN_STOP_SHARE` (§3 revision),
  permanent joint-limit-margin term (D-114; weight untuned, no tracker trained with it yet).
- ◐ Sparse-reward exploration (arm GRPO): start from a competent policy; group size and prefix branching as in Z-1. Not started.
- ❌ Throughput: CPU PPO only; no MJX (D-123).
- ✅ Asymmetric actor-critic. ✅ Matched budgets for competing methods (contract).

### 4.8 Cross-embodiment
- ❌ Sealed held-out target bodies untested on the latent route (BC: xarm7 0/100, D-064; v2 BC panda_tf3 99/100, xarm7 0/100, D-102).
  Target demos are collected in v6dart; no DAG node evaluates the latent route on them (D-123 top 1).
- ◐ Actuator and scale normalization across bodies; sourced limits per body (D-107).
- ◐ Morphology coverage: many bodies, few families per task.
- ◐ Negative transfer risk (t1 under the D-085 defect; learned t1 routes fall on the sourced-limit tracker, W8 interim).

### 4.9 Task and environment diversity
- ❌ One arm task, one legged task; dual-arm teacher data only. W12 (D-122) adds a dual-arm support_insert plan.
- ❌ Clutter, articulated or deformable objects, long-horizon multi-step tasks.
- ❌ Loco-manipulation (the humanoid use case in Ψ₀); psi1z excludes walking-heavy SIMPLE tasks by its reproduction gate (D-120).

### 4.10 Evaluation
- ✅ Paired seeds, CIs, fresh seed sets, irrelevant-edit controls.
- ✅ Motion-quality metrics (slip ratio, CoT, jerk, contact forces, joint-limit margin, chunk-boundary steps) in every legged and arm eval
  row (`rrp.envs.mujoco.motion_quality`, D-108); tracker foot force is the 20 ms-filtered peak (D-114).
- ✅ Robustness sweeps (see 4.2). ✅ Gates in code for trackers, datasets and policy flags (`rrp.harness.eval.gates`, D-112, D-114); the
  `validate_tracker` stage is wired into the legged DAG template for future runs (`recipes/templates/legged_lineage.yaml`).
  ❌ Long-duration runs (minutes) for drift.

### 4.11 Safety and hardware
- ◐ Joint and torque limits: sourced torque limits (D-107) and a permanent joint-limit-margin reward (D-114); no deployment safety layer
  (clamp, rate limit) beyond `np.clip(±6)` in `policies/bc.py`.
- ❌ Fall and impact handling, safe stop, fall recovery (t1 packet edits cause falls, D-092; learned t1 routes fall, W8 interim).
- ◐ Packet admission (version and staleness, `hold_measured` fallback) exists; ❌ OOD packet detection with a fallback controller.
- ❌ Thermal and duty-cycle limits.

### 4.12 Infrastructure
- ✅ Reproducibility: fingerprinted bundles (legged weight fingerprints, `policies/bundles.py`), recorded commands, raw artifacts,
  one manifest writer with physics versions (W3), `Source` enum (`core/provenance.py`).
- ✅ Orchestration: `rrp run-dag` with leases, bounded retries, gate-aware failure and a JSON ledger (W5, D-096, D-114); chain scripts
  not yet retired.
- ◐ Compute contention: no heavy compute on the host (D-115); peer admission capped with declarations ≥ 1.35 × peak (D-106, D-117); watchdog
  counts non-reclaimable memory and reports memory.high throttling (D-116, D-117).
- ✅ Repo structure: restructure with shims completed (W4, f1db76d); shim removal (phase 5) pending.

## 5. Priorities
1. Physics realism: contact v2 + reward schedule + slip gate (done, D-101/D-103), then actuator realism (implemented, opt-in, D-103/D-107) and randomized latency.
2. Randomization + robustness sweeps with break-points per policy (harness done, D-108/D-112; randomization in training still partial).
3. Motion-quality gates on trackers and datasets (done, D-112/D-114).
4. Arm: smoothed scripted expert (done, D-097; grasp v2/v2.1, D-110/D-118/D-121), then GRPO fine-tuning with anchor evaluations (not started).
5. Rough terrain and GPU simulation (MJX) for trackers.
6. The untested claims: held-out target bodies, one loco-manipulation task.
