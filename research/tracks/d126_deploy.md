# D-126 deployment credibility and infra (track notes, append-only)

## source labels
Backlog row "Canonical source labels everywhere" (sub-agent, branch `track/d126deploy-source`). State: **verified**
(unit tests + a real 4-step teacher-route ladder rollout on parm5_pg2, host, tiny).

- Switch: env `RRP_SOURCE_LABELS=canonical` or explicit `source_labels=True` (LadderConfig field,
  `evaluate_dual_latent(..., source_labels=)`, `run_audit_episode(..., source_labels=)`,
  `run_policy_quality_episode(..., ckpt=, source_labels=)`). Default OFF. Version tag `sl-1`.
- On: new rows gain `source_label` (strict canonical string) and `source_label_version: "sl-1"`; `source` unchanged.
  Canonical labels: arm ladder teacher `scripted_teacher:privileged`, oracle `oracle:teacher_future` or
  `oracle:learned_chunk:<label>`, generated `learned:<flow ckpt>`, learned `learned:<policy label>`; dual latent eval
  `learned:<policy>[+edit:<e>]`; dual teacher audit `scripted_teacher:dual_<task>`; arm teacher_quality policy rows
  `learned:<ckpt>` (or `bc:<ckpt>` if the label parses as bc). Teacher rows already wrote canonical strings.
- Byte identity (off): the parm5_pg2 teacher ladder row (seed 0, 4 steps, wall_s dropped) was byte-identical before
  (origin/main 733b02a) and after the change (`cmp`); `test_source_labels.py` freezes its key order and source string.
- Readers: `row_source` (contracts/provenance); `pipelines/arm._source_counts` (manifest `source_kinds`) uses it;
  `robustness.summarize_shard` adds `source_labels` only when some row has one (old summaries unchanged).
  Other readers read the unchanged legacy `source` key, so they need no change. The service/UI labels by mode and
  does not filter by row source.
- Wire: `contracts.action.Source` / `LatentActionChunk.source` widened with canonical kinds; `check_packet` and
  `contracts/channels.py` do not read the source (unaffected).
- Left: `evaluation/legged_latent_eval.py` (not touched, parallel edit) writes `row["source"]` in `run_episode`
  (`src = ...; row = dict(body=..., source=src, ...)`). Its legacy strings already parse with `row_source`
  (scripted_teacher[:arc_only], privileged_oracle_packet:<lsv> -> oracle, oracle_diagnostic:... -> oracle, ctl.policy_version
  = learned:/bc:<ckpt>). Writer change for the parent: after `row = dict(...)` add
  `lab = parse_legacy_source(src); stamp_source_label(row, "oracle" if lab.kind is Source.ORACLE else lab.kind,
  (f"privileged_packet:{ctl.lsv}" if src.startswith("privileged_oracle_packet") else lab.detail))`
  (import from rrp.contracts.provenance; default off, so rows stay byte-identical). The arm ladder `learned` route is
  a plain FlowPolicy BC baseline but its legacy string says `learned:`; relabelling it `bc:` is a semantic change
  left for a decision (the sl-1 stamp keeps kind `learned` so it agrees with the legacy string).
  → Applied by the parent (legged_latent_eval.run_episode, after the row dict); tested in
  `tests/unit/test_deploy_eval.py::test_legged_rows_canonical_source_label_when_switched_on`.

## state (D-126 agent; host crashed ~00:34 on 09-28 and was resumed; D-127: host = git, editing, unit suite only)
| item | state | code | tests |
|---|---|---|---|
| #27 base-state estimator | verified (unit) | `envs/state_estimator.py` (bse-1), `LeggedSession(base_state_source=)` | `test_state_estimator.py` (12) |
| #28 privileged audit | verified (unit) | `evaluation/privileged_audit.py` (paudit-1), `contracts/channels.schema_privileged_fields` | `test_privileged_audit.py` (10) |
| #29 packet OOD + fallback | verified (unit) | `controllers/packet_ood.py` (pood-1), `training/packet_ood_fit.py` | `test_safety_ood.py`, `test_deploy_eval.py` |
| #30 safety layer | verified (unit) | `controllers/safety.py` (safety-1) | `test_safety_ood.py`, `test_deploy_eval.py` |
| #31 system II harness | verified (unit; sub-agent) + wired | `evaluation/system2.py` (s2h-1), `--system2` in legged eval | `test_system2_harness.py`, `test_deploy_eval.py` |
| #32 long runs + latency | verified (unit) | `evaluation/deploy_eval.py` (deploy-1): `--eval-mode long`, `--measure-latency`; arm `rrp latent latency --representation` | `test_deploy_eval.py` |
| ladder.py / legged summaries → library | verified (sub-agent, parity tests) | `evaluation/ladder_cli.py`, `evaluation/legged_summaries.py` | `test_ladder_cli_parity.py` |
| source labels | verified (sub-agent) | sl-1 (above) | `test_source_labels.py` |
No experiment was run (code only). No real-weight smoke yet: the host may not run simulations (D-127); the peer smokes
are listed under "peer smokes" below.

### defaults are byte-identical (proof)
Goldens computed on origin/main 733b02a (before any D-126 change) and asserted in the suite:
- `test_state_estimator.py::test_default_session_byte_identical_to_pre_d126`: LeggedSession hexapod6 seed 3, 15 steps,
  sha256 over qpos + every public observation (minus observation_id) + speed_est.
- `test_deploy_eval.py::test_defaults_byte_identical_to_pre_d126`: run_episode rows (teacher route 2.5 s; latent route
  with a tiny random-weight bundle 1.5 s), sha256 of the row minus wall_s / checkpoint_provenance / packet timestamps.
- The pipelines/DAGs call the unchanged CLIs; every new flag defaults to the old behaviour; `DeployOptions()` == None.

### #27 estimator (bse-1)
Complementary filter: IMU strapdown (quat, specific force) predicts world velocity; leg odometry
v_b = mean over stance feet of -(J qd + ω × r) (feet from a PRIVATE MjData with the free joint pinned at identity, joint
encoders only) corrects it with gain dt/(τ+dt), τ = 0.1 s; position dead-reckoned. Stance = touch > 1 N.
`base_state_source=estimator`: NodeState.base_vel_estimate = body-frame velocity (units string names bse-1); the system-i
speed feature (public_context[20]) becomes 1 s differences of the dead-reckoned position (same baseline as the truth+noise
path, so the feature's meaning is unchanged). Localization still gives x, y, yaw for waypoint geometry (declared sensor).
Physics is unaffected (test: identical qpos stream). Unit tests with known answers: static, IMU-only integration closed
form, leg odometry with rotation, yaw-rotated frames, accelerometer-bias steady state b·dt(1-k)/k, real hexapod
kinematics vs truth, real-kinematics odometry.
Caveat for the experiment: routes were trained on the truth+noise speed; a mismatch under the estimator is expected and is
exactly what #27 measures (report both, and the `base_state` / `long_run` est_vel_rmse drift metrics).

### #28 privileged audit (paudit-1)
Groups: ctx.{gyro, gravity, osc, waypoint_a, waypoint_b, event, speed} (= the 22 public_context dims exactly, tested),
local.{q, qd, imu, touch}. Provenance per group; waypoint_* and speed are `truth_noise` (computed from sim truth + noise;
speed becomes `declared_estimator` under the estimator). Ablations zero / shuffle (same call index of another episode) /
noise (recorded mean/std), plus an `identity` rerun control. Paired by seed: success lost/gained (McNemar), trajectory
deviation (bootstrap CI), final shift. Flags: expected_irrelevant_but_used, privileged_derived_used,
nondeterministic_baseline. Static AST check of the deployable-input functions (0 violations; declared exceptions:
`_sense` truth+noise localization, `LeggedBinding.imu` == IMU sensors, `LegKinematics.feet` writes its private copy);
schema check of PolicyObservation field names; dynamic tripwire (privileged accessors raise while every deployable input is
computed; observation passes the public transport) for both base-state sources.

### #29 packet OOD (pood-1)
PPCA-style Gaussian score on standardized packet features (Mahalanobis in r PCs + residual / isotropic residual
variance), thresholds = calibration quantiles q0.99 (monitor) / q0.999 (enforce) of a held-out split. Stored as
<stem>.json + .npz, fingerprinted (sha256 over arrays + metadata; a tampered file is refused), bound to the bundle's
latent_space_version (another bundle is refused). Fit sources: `fit-rep` (encoder posterior means on the rep's training
split, calibration on its held-out episodes) or `fit-packets` (recorded unedited generated packets, split by episode).
Runtime `--packet-ood monitor|enforce`: enforce rejects like an incompatible packet (counted in stats.rejected, logged
`packet_ood`), then the declared fallback: hold_default (existing) | hold_measured | safe_stop (needs --safety enforce).
The D-092 t1 edit packets are NOT available locally (the t1_edits rows store probe readouts, not z; weights live on the
peer), so the t1 test is an experiment step (below), not a unit test.

### #30 safety layer (safety-1)
Per tick on the policy joints: position clamp (joint range ∩ ctrlrange − margin), rate limit, exact PD torque clamp
(u ∈ q + (kd qd ± effort)/kp), velocity guard (u = q when |qd| > qd_max), fall detection (tilt > 0.8 tilt_limit for 5
ticks → hook, default safe stop), safe stop (linear ramp to the default stance over stop_s). Limits SOURCED
(physics.actuator.SOURCED peak torque / URDF speed) for go2/anymal_c/t1/h1/g1, else model ranges + labelled VMAX estimate
(`limit_source` in the row). `--safety monitor` passes targets unchanged and counts would-be interventions.

### #32 eval modes
`--eval-mode long --long-s 180 --window-s 10`: the episode continues after task success (until long_s or a fall);
`long_run` = per-window displacement, path, yaw change, true vs estimated speed, height, tilt, falls, estimator velocity
RMSE / dead-reckoning error, post-task drift. `--measure-latency`: per-call wall-clock of system i (0.4 s budget) and the
system-0 tick (20 ms budget, includes the replan call) in the eval process. Arm: `rrp latent latency --checkpoint FLOW
--representation REFIT_REP` times the deployed refit route (packets addressed to the refit realizer, as the ladder).

## exact configs for the experiments (peer only; lowest priority for smokes; declare memory ≥ 1.35 × peak, D-117)
Every command below is `scripts/peer_run.sh --cpu 2 --mem 6G --label d126deploy_<x> --max-seconds N -- PY -m ...`
with `RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/d126deploy`. FLOW/REP = a W8 route, e.g. the anymal_c/go2 contact_v2
semfix/nosem flows of `dags/legged_v2_*` (their eval_r2 nodes name the exact policy.pt / representation.pt).
- #27: `python -m rrp.evaluation.legged_latent_eval --flow FLOW --bodies go2 --seeds 10000-10029 --out
  artifacts/runs/d126deploy_est/go2_r2_estimator.jsonl --base-state-source estimator` vs the same with the default
  (truth_noise); teacher reference: drop --flow. Compare success (newcombe_diff) and `base_state.windows[*].est_vel_rmse`.
- #28: `python -m rrp.evaluation.privileged_audit run --flow FLOW --bodies go2 --seeds 10000-10009 --max-s 30
  --out artifacts/runs/d126deploy_audit/go2_semfix [--expect-irrelevant ctx.osc]`; repeat with `--bc BC` (positive
  control) and `--base-state-source estimator`. `python -m rrp.evaluation.privileged_audit static` (host-safe, no sim).
- #29: (1) `python -m rrp.training.packet_ood_fit fit-rep --rep REP --body t1 --out
  artifacts/runs/d126deploy_ood/t1_s0` (4000 fit / 1000 cal packets); (2) record the D-092 suite with packets:
  `legged_latent_eval --flow T1FLOW --bodies t1 --seeds 10000-10019 --edit <e> --t-edit 2.0 --record-packets
  artifacts/runs/d126deploy_ood/t1_pk --out ...` for e in none, halt, probe_yaw:+0.6, contact:0:0, probe_halt, zero,
  rand_norm:<matched>; (3) `packet_ood_fit score --model .../t1_s0 --packets .../t1_pk --out .../scores.json`
  (auroc_fall, auroc_edited_vs_unedited, flagged falls); (4) closed loop with `--packet-ood enforce --ood-model
  .../t1_s0 --ood-fallback safe_stop --safety enforce` on the falling edits: fall rate vs (2).
- #30: `legged_latent_eval --flow FLOW --bodies go2 --seeds 10000-10029 --safety enforce --out ...` vs off (success
  cost); under perturbation use `rrp.evaluation.robustness` with the same rows (push/terrain) and `--safety enforce`.
- #31: `legged_latent_eval --flow FLOW --bodies go2 --seeds 10000-10019 --system2 oracle|default|vlm:<weights>`, or
  `python -m rrp.evaluation.system2` (ground_eval / closed_loop_eval).
- #32: `legged_latent_eval --flow FLOW --bodies go2 --seeds 10000-10004 --eval-mode long --long-s 180 --window-s 10
  --measure-latency --out artifacts/runs/d126deploy_long/go2.jsonl` (≈ 5 × 180 s sim per body); arm latency:
  `rrp latent latency --checkpoint FLOW --representation REFIT/representation.pt --out
  artifacts/runs/d126deploy_latency/arm_<route>.json` on an idle GPU lease.
