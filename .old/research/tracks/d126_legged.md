# D-126 legged/physics implementation (code only; nothing trained)

Worktree `~/work/rrp-wt/d126legged`, branch `track/d126legged`, peer code dir `/dev/shm/rrp-brandonin/wt/d126legged`.
Every option is additive and OFF by default. Default behaviour is unchanged, and this is proven by
`tests/unit/test_d126_legged.py::test_default_tracker_env_is_byte_identical_*`: the reward and observation streams of the tracker
env on go2 (contact_v2), t1 (turn terms), t1 (v1lat) and hexapod6 (v1) hash identically to the pre-D-126 code, origin/main 4181111.

## what landed
| item | code | state |
|---|---|---|
| #13 h1 clock gait | `RewardCfg.ref_gait=clock` with `ref_lift` / `ref_contact` (phase-locked foot-lift targets, `clock_lift_targets`). `--ref-gait clock --ref-gait-mode reward\|residual\|both`; residual = `ref_ff` at any speed through `--ref-ff-vmax`, applied by LearnedTracker from actor meta | complete; recipe `h1_clock_scratch` (from scratch, 2048 envs) |
| #13 g1 yaw cap | `yaw_progress_cap` (default 1.2 = old hard-coded clip), `yaw_lin_all_max`, `yaw_overshoot` | complete; recipe `g1_yawcap_ft` (from g1_src, sha f0a4daaa…) |
| #13 t1 turn+latency | recipe `t1_turn_latency_ft` (from installed w8d 36e91467…, actuator v1lat, teacher mix, turn terms, limit margin agg max); the DAG also validates at v1lat 30 ms | complete (not run) |
| #13 limit_margin | review: `mean` over joints makes one joint at its limit cost 1/n (0.083/step on t1's 12 joints, vs. a ~4.5/step tracking reward), while the D-112 gate checks the MIN margin. Exposed `--limit-margin`, `--limit-margin-agg mean\|max\|sum`; the recipes use -1.0 with `max` (1.0/step at the limit, 4 at 2% past). The default is unchanged (mean, gait_v2 -1.0) | complete |
| #14 actuator_mode | `rrp.physics.actuator.resolve_mode` (ideal\|v1lat\|v2; alias v1; `$RRP_ACTUATOR_MODE`; `ACTUATOR_MODE_DEFAULT="ideal"`). LeggedSession(actuator_mode=…) routes ticks through ActuatorModel with a fixed per-episode latency (`$RRP_ACTUATOR_LATENCY_MS`, else drawn from the seed); snapshot/restore includes the queue. Wired into perturb.install_legged (the session's model is kept unless a perturbation sets latency), eval rows (`actuator_mode` key only when non-ideal), collect metadata, `tracker_validation --actuator` (default from env), `tracker_training --actuator`, and the pipeline (`options.actuator_mode` / `actuator_latency_ms` → physics_env; rows and shards are checked against the declaration) | complete; default not flipped |
| #14 estimated speeds | `speed_sources` / `mode_record`: joints whose vmax is `estimate` are listed in rows, actor meta (`actuator_speed_estimated`) and validation JSON. Currently: ALL procedural bodies (pquad4, sprawl4/8, hexapod6*: 8 rad/s estimate); t1/h1/g1/go2/anymal_c are fully URDF-sourced | flagged |
| #15 terrain curriculum | `--terrain-curriculum gated`: `LeggedEnv(terrain=…)` builds the perturb/bodies `bumps_v1` heightfield at amp_max and rescales `hfield_size[2]` at run time. Workers are spread over (w+1)/W of the level. `TerrainCurriculum` is gated on fall rate and tracking error (+ optional alpha threshold), resumable, logged (`terrain_level`, `terrain_amp_m`) and recorded in meta. `floor_outer` counts as floor for contacts | complete; recipe `anymal_c_terrain_ft` (to 12 cm) |
| #15 MJX | `src/rrp/envs/mjx_legged.py` (PROTOTYPE, imported by nothing), peer venv `~/work/ext/venvs/mjx` (jax 0.11.2 + mujoco-mjx 3.14.0, CUDA works) | feasibility shown, see below |
| stage | new pipeline stage `train_tracker` (recipe + args + resume + actuator_mode); `validate_tracker` accepts `inputs.actor` | complete |
| #11 | `dags/templates/legged_v2_heldout.yaml`: validate_{a,b,h} gate → collect A, B (train) + H (50-episode budget) into one root → rep/flow on [A, B] → R2 on A, B, zero-shot `heldout` on H → DAgger (generated route) → refit on [A, B, H] (`dagger_present_bodies`, opt-in) → R2 on H tagged `adapt`. BC references: zero-shot [A, B] and the same-budget [A, B, H]. Defaults A=go2, B=hexapod6 (CPG, gate report-only; D-101(4)), H=anymal_c | template (not run) |
| gated template | `dags/templates/tracker_recipe_gated.yaml` (train_tracker → validate_tracker gate) + `dags/d126_tracker_{h1_clock_scratch,g1_yawcap_ft,t1_turn_latency_ft,anymal_c_terrain_ft}.yaml` | ready |
| #34 | `tasks/loco_pick.json`, `envs/legged_scenes.build_loco_pick` (spot_arm, the menagerie quadruped with an arm; opt-in `ensure_spot_arm_asset`; scene_version loco_pick_v0), `LocoPickSession` (public/truth `height_above_m`), `teachers/legged_loco.LocoPickTeacherStub` (**STUB**, scripted_teacher: walk to the standoff only; the arm raises NotImplementedError). Not registered (`register()` is opt-in). Spot torque limits are unsourced (menagerie ±1000, labelled). **No spot tracker exists** | task + scene complete; teacher STUB |
| #22 legged | `tasks/foothold_steps.json`, `build_foothold_steps` (visual footholds, optional stones), targets relative to the previous foothold (dx, dy, dyaw) plus world poses, foot order with null = any foot, `FootholdSession` public (encoder FK + localization, known only under load) / truth `foot_distance_m`, `failure_reason()` (fell / wrong_foot / missed_foothold / timeout; privileged, reports only) | complete (no teacher) |

## MJX peer smoke (go2, contact_v2, 25 ticks x 10 substeps; lowest priority; leases 1790581748_eeac75, 1790581761_60020e, 1790581852_b21747)
- The plain contact_v2 go2 model fails: `NotImplementedError: (CYLINDER, BOX) collisions not implemented` (robot self-collision).
  With the explicit adaptation `no_self_collision` (robot geoms collide with the floor only; its effect on the C reference is
  0.0 over the rollout), MJX accepts the elliptic cone, impratio 10 and condim 6 unchanged.
- Parity with C MuJoCo (float32): CPU max |Δqpos| 1.2e-5 rad; GPU 3.2e-3 rad (final height equal to 3e-6 m on CPU).
- Throughput (env-ticks/s; 1 tick = 10 substeps): C MuJoCo, one thread, 4518. MJX CPU: 38 (1 env), 48 (64). MJX GPU: 1.7 (1),
  102 (64), 512 (512 envs), with the GPU shared at ~96% utilisation by other jobs. Compile: 15–80 s. Peak RSS 1.3 GB (CPU),
  2.2 GB (GPU run).
- Reading: MJX runs this physics correctly, but at ≤512 envs on a contended GPU it is ~9x slower than ONE C thread. A throughput
  win needs thousands of envs, an idle GPU, and probably fewer solver iterations (adaptation `few_solver_iters`, untested). Not
  worth wiring into training until an idle-GPU scaling run (e.g. n_envs 2048/4096) shows > 16 C workers' worth.
- Reproduce: `export RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/d126legged; scripts/peer_sync.sh push; scripts/peer_run.sh --gpu --gpu-mem 4G --cpu 2 --mem 8G --label mjx --max-seconds 1800 -- env XLA_PYTHON_CLIENT_PREALLOCATE=false nice -n 19 /home/brandonin/work/ext/venvs/mjx/bin/python -m rrp.envs.mjx_legged --body go2 --contact v2 --device gpu --n-envs 1,64,512 --adapt no_self_collision`
  (use `--device cpu` and drop `--gpu` for CPU).

## launch later (lead decides; peer only, D-115/D-127)
- Trackers: `rrp run-dag dags/d126_tracker_<recipe>.yaml`. First measure a 20-iteration smoke of the recipe
  (`--iters 20` through `args`) and redeclare the train node's memory at ≥ 1.35 × peak. The declared resources are estimates.
  Or directly: `python -m rrp.training.tracker_training --recipe <name> --out artifacts/runs/<dir>`.
- Actuator realism on any legged DAG: add `actuator_mode: v1lat` (and `actuator_latency_ms`) to the node options (collect, eval,
  edits, validate). Tracker validation: `--actuator v1lat --latency-ms 30`.
- Held-out: a child DAG `extends: templates/legged_v2_heldout.yaml` sets name / label prefixes, `sha_a` (go2
  af3f06f4e029e9f92fafc50e6174ffdb171c5512fbab4cd6fac18e6ce23cf18b), `sha_h` (anymal_c
  2a16532bbd07f7abc662bdda10df6bfb70fca2ec11d59f9567142e92a2f22d95).
- #34 needs a spot_arm tracker (`python -m rrp.training.tracker_training --body spot_arm …` after `ensure_spot_arm_asset()`;
  not attempted) and a real manipulation phase.

## coordination
- `envs/legged.py`: `_sense` is untouched (deployment agent). Added: the `actuator_mode` constructor args, `_tracker_tick` routing,
  `actuator_record()`, and snapshot of the actuator queue.
- `envs/perturb.py` (robust track): install_legged keeps a session's own actuator model when no latency perturbation is set.
- `bodies/legged.py`: `standalone_model(terrain=None)` (additive).
- `contracts/runconfig.py`: `train_tracker` appended to PIPELINE_STAGES.
