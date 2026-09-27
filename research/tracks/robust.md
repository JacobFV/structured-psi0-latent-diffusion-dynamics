# track robust (W6): robustness sweeps + motion-quality gates

Owner: W6 robustness agent. Worktree `~/work/rrp-wt/robust`, branch `track/robust` (from origin/main b89ca70).
Peer code dir `/dev/shm/rrp-brandonin/wt/robust` (pushed once; peer CPU admission was full, so everything ran on the host).
Started 2026-09-27 ~09:40.

Question: where does each policy break as the physics departs from training, and is the semantic packet route more or
less robust than nosem and BC?

## what was built (state: verified; commits on track/robust, merged to main)
### `rrp.envs.perturb` (physics perturbations; evaluation-only, nothing enters an observation)
`PhysicsPerturbation` (frozen dataclass; `PhysicsPerturbation()` = nominal, every code path unchanged):
- model level, applied after the scenario is compiled (so the robot spec / morphology features the policy sees stay
  NOMINAL): `friction_scale` (all geom friction; under contact_v2 the floor has priority, so robot-floor mu = 0.9 x s),
  `mass_scale` (mass + inertia of every robot body), `com_offset_m` (legged: root/trunk; arm: last arm link),
  `kp_scale`/`kd_scale` (every position servo), arm `object_mass_scale` (cube) and `object_friction_scale` (cube AND the
  finger geoms, because MuJoCo takes the max of the pair: grasp mu = 1.5 x s). `mj_setConst` after mass/CoM changes.
- step level: `latency_ms` = legged: the existing actuator mode `v1lat` (`rrp.physics.actuator.ActuatorModel`, ideal
  joints + SOURCED torque/speed limits + fixed latency; nominal = ideal actuators, so the 0 ms level isolates the
  sourced speed envelope); arm: `ctrl_delay_v0` (FIFO delay of the robot ctrl by round(latency/dt) substeps; the arm has
  no actuator model). `push_impulse_Ns`: constant horizontal force J/0.1 s on the legged root (t = 4 s) or the last arm
  link (t = 3 s), direction drawn from the episode seed only (paired). `terrain_amp_m`: see below.
- hooks: `install_legged` replaces `LeggedSession._tracker_tick` by an instance-level copy with the same operations
  plus the hooks (used by every legged route: teacher tracker, system-0 adapter, BC adapter); `install_arm` wraps each
  controller's `apply_substep` (called right before every `mj_step` of `Session.step`).
### legged terrain flag (`rrp.bodies.legged.legged_world(..., terrain=)`, `build_waypoint_contact(..., terrain=)`)
`bumps_v1`: smoothed white noise (sigma 0.1 m), 0..amp m, heightfield named `floor` (so foot-contact / fall logic keep
working) over +-8 m, flat within 0.6 m of the spawn, plane `floor_outer` beyond. Default None = the flat plane, unchanged.
The pattern is drawn from the episode seed. Recorded in the model (`terrain_version` text) and in the scenario meta.
### `rrp.evaluation.motion_quality` (in EVERY eval row as `motion`; read-only recording)
- legged (`LeggedMotionRecorder`, attached in `legged_latent_eval.run_episode`): `slip_ratio` / `slip_cp_mps` (contact-point
  stance slip, loaded feet, over ticks with true base speed > 0.1 m/s), `cot` (sum |tau qdot| dt per substep / (m g path)),
  `joint_jerk_rms/peak` (measured policy joints, 50 Hz), `peak_contact_force_bw` (max foot touch sensor per substep / weight),
  `joint_limit_margin_min`.
- arm (`ArmMotionRecorder`, attached in `ladder.run_ladder`): `joint_jerk_rms/peak` (measured arm joints, 20 Hz),
  `chunk_vel_step_max` / `vel_step_any_max` (commanded joint-velocity step at chunk/packet boundaries +-1 tick / anywhere),
  `penetration_max_m` (cube-robot contacts, as teacher_quality), `joint_limit_margin_min`.
- Tests (`tests/unit/test_motion_quality.py`, known values): jerk of a cubic is exact; chunk-boundary velocity step 2 rad/s
  found at the boundary and not elsewhere; limit margin incl. unlimited joints; slip ratio / CoT; a 3 N s push on a 2 kg box
  gives 1.5 m/s; model scaling + nominal no-op; ctrl delay shifts by n substeps; terrain hfield is `floor` and bounded.
- Behaviour unchanged (checked against the pre-change code, same seeds): legged teacher x2, BC, semfix flow rows identical
  except the new `motion` field; arm route learned (BC) and generated rows identical.
### `rrp.evaluation.robustness` (harness; CLI `run` / `report` / `video`)
- `run --family legged|arm --robots R --route NAME=SPEC ... --seeds S [--factors] [--shard i/n] [--workers k]`.
  Routes: legged `teacher` | `bc:<ckpt>` | `flow:<ckpt>`; arm `teacher` | `learned:<ckpt>` | `generated:<flow>@<rep>`.
  Source labels come from the eval code (`scripted_teacher`, `bc:<ckpt>`, `learned:<ckpt>`, `learned(system-i flow)`).
- Grid (one factor at a time + `all_moderate` + nominal; same seeds at every level):
  legged friction 0.4/0.6/0.8/1.25/1.5, mass 0.8/0.9/1.1/1.2, com_x +-1/+-2 cm, com_y +1/+2 cm, kp 0.7/0.85/1.15/1.3,
  latency (v1lat) 0/10/20/30 ms, push dv 0.25/0.5/0.75/1/1.5 m/s (x total mass = N s), terrain 1/2/3/5/8 cm; all_moderate =
  friction 0.7, mass 1.1, com_x +1 cm, kp 0.85, 20 ms, push 0.5 m/s, terrain 2 cm.
  arm friction 0.4..1.5, mass 0.8..1.2, com_x +-1/+-2 cm, kp 0.7..1.3, ctrl delay 10/20/30 ms, push 2/5/10/20/40 N s,
  object mass x0.5/2/5/10/20, object friction x0.05/0.1/0.2/0.4/0.7; all_moderate = friction 0.8, mass 1.1, com +1 cm,
  kp 0.85, 20 ms, push 2 N s, object mass x2, object friction x0.6.
- Outputs `<out>/<route>/<robot>/<factor>=<level>.{jsonl,summary.json}` + `manifest.json` (seeds, checkpoint sha256);
  resumable (legged per episode, arm per shard: a batch shares the flow RNG, so a partial arm shard is redone whole).
- `report`: per level success k/n, Wilson 95% CI, falls, failure stages, task metrics, motion medians, paired lost/gained
  seeds vs nominal, BREAK-POINT = first level moving away from nominal (each side) with success > 20 points below nominal.
- `video --family --route --robot --condition --seed`: re-runs one sweep episode exactly and writes a captioned clip + INDEX line.
- The legged controller is loaded once per shard and its seed state reset per episode (rows verified identical to a
  fresh construction per seed; ~15 s/episode of checkpoint loading saved).

## findings on the way
- Arm R2 rows depend on the DIAGNOSTIC oracle comparison (`compare_oracle`, default True in scripts/ladder.py): its
  look-ahead rolls the real session forward and restores it, and the restore recomputes sensors with mj_forward (after
  mj_step, sensordata lags one substep), so the next public observation differs in the last bits and the chaotic rollout
  diverges (same 2 seeds: 114/171 steps, success/success with compare_oracle vs 114/300 success/timeout without). Not a
  privileged-information leak, but R2 episodes are not reproducible across that flag; statistics should be equivalent.
  The sweep runs every arm level with compare_oracle=False (step hooks are not part of a snapshot; `run_ladder` now refuses
  step-level perturbations when a look-ahead is active).
- Arm results depend on the batch composition (the flow/BC sampling RNG is shared by a 16-session batch): 2-seed runs do
  not reproduce the 30-seed run's episodes. The harness always uses the ladder's batching (16) on the same seed list.
- Host: the memory-PSI watchdog shed/refused jobs repeatedly (external swap pressure); a process at 3 GB RSS ran at ~4% CPU
  during one such episode. Launcher `scripts/robust_host_run.sh` retries (bounded) and resumes.
