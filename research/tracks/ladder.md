# track: ladder — closed-loop failure localization (correction 2026-09-25, item 3)

Owner: ladder track agent. Branch `track/ladder`, worktree `~/work/rrp-wt/ladder`, peer dir `/dev/shm/rrp-brandonin/wt/ladder`.
Raw outputs live on the peer store `artifacts/runs/ladder_*` (copied summaries under `research/tracks/ladder/` when final).

## CURRENT ANSWER FOR ACCEPTANCE / GRPO / campaign (which route/checkpoint is competent)
- **No competent learned route exists yet, and none can with the current Stage-A system 0**: R0 teacher 30/30;
  R1 oracle (latent_sem_v1) 0/30 with deployment input, 0/30 with training-consistent own-prev input; R2 flow v2 0/30.
- **Root cause localized: bug B-1** (below). The Stage-A realizer is a copycat of the previous teacher command stored in
  node-feature column 28; at deployment that column is 0, which on the TRAINING pack itself makes system 0 no better than
  holding still (arm) and wrong on the gripper. This explains the acceptance finding (step-0 output -0.1 vs -1.0, packet
  fine) and the GRPO findings (small undirected steps, gripper stuck ~0.03 = "mode averaging" is actually copying a zero
  prev-action; no grasp/lift after a teacher prefix).
- **Fix in progress**: system 0 re-fit on the FROZEN sem_v1 encoder with column 28 zeroed (same latent space => existing
  flows stay usable; new realizer compat id). Running: peer lease 1790370807_bd05c2 `ladder_rz_sem_ft`
  (config `configs/ladder/rz_sem_v1_b1fix_ft.json`, warm start, 8k steps) -> `artifacts/runs/ladder_rz_sem_v1_b1fix_ft/`.
  Then R1/R2 are rerun on it. If the frozen z lacks what R needs without the crutch, Stage A must be retrained with
  `"zero_prev_action": true` (binding track: please set it for revised representations).
- Until then: do NOT treat any learned closed-loop number (acceptance, GRPO, baselines, 4-way campaign) as evidence
  about the architecture; they are dominated by B-1.

### B-1 on the training pack itself (`scripts/ladder_t0_check.py`, raw `artifacts/runs/ladder_smoke/t0_check_sem_panda.json`)
latent_sem_v1 E+R, panda_pg2 rows, j=0, 96 rows each; normalized MSE (arm hold-still reference = arm_label_sq):
| input | rows | arm err | arm hold-still | grip err | grip pred / label | z norm |
|---|---|---|---|---|---|---|
| stored (teacher prev cmd in col 28) | t=0 (col 28 is 0 there too) | 0.112 | 0.114 | 0.314 | -0.44 / -1.0 | 97.7 |
| stored | t=1 | 0.005 | 0.079 | 0.0002 | -1.01 / -1.0 | 96.5 |
| stored | t>=5 | 0.005 | 0.082 | 0.005 | -0.36 / -0.35 | 35.4 |
| col 28 zeroed (= deployment) | t=1 | 0.144 | 0.079 | 0.319 | -0.44 / -1.0 | 96.3 |
| col 28 zeroed (= deployment) | t>=5 | 0.083 | 0.082 | 0.741 | -0.59 / -0.35 | 34.6 |
The only training rows whose col 28 is 0 are t=0, and exactly there the realizer has no skill: the acceptance "step-0"
symptom is the same bug, not a knot-timing/phase-0/normalization/clipping problem (knot times, phase and normalization
are identical between pack and runtime; the +-6 output clip was not checked separately because the error is already present in the raw network output on the pack). The large step-0 z norm
is E's (98 at t<=1 vs 35 later), identical in pack and runtime. The acceptance parity check matched features because at
t=0 the column is 0 in both; from t=1 on it differs (`scripts/ladder_feature_parity.py`).

## bug B-1: previous-action feature zeroed in the wrong column
- Node features are `[static(26) | q, qd, PREV_ACTION, anchor(3), axis(3), jp(3), jr(3), lever(3)]` (NODE_DIM 44), so
  prev-action is column **28**. Datasets were collected before D-021 with the teacher's previous 1-step command there.
  D-021's load-time fix in `rrp.learning.data.episode_samples` tests/zeroes column **2** (a static column, always 0),
  so it never fires: packs built since then (incl. `latent_pp_v3dart_s1_H16`) carry the teacher's previous command in
  column 28 of node features and of the node rows of the morph bank. The deployed featurizer always writes 0 there.
- Evidence (all on the peer store):
  - Feature parity: replay of stored episode `pick_place_panda_pg2_s0` through the current featurizer/teacher
    (`scripts/ladder_feature_parity.py artifacts/datasets/pick_place_primary_v3dart pick_place_panda_pg2_s0`): every
    bank, relation, q0, local sensor and action is identical except node/morph column 28 (t >= 1). Packed column 28 is
    non-zero in 66% of node entries (first 20k rows).
  - Stage-A system 0 on the TEACHER's own clean trajectory (R0 route with a shadow system 0 fed oracle packets, not
    executed; panda_pg2 seeds 3000000-1; `artifacts/runs/ladder_smoke/teacher_shadow*.jsonl`), normalized 1-step MSE vs
    teacher label:
    | prev-action input | arm | gripper | reference: arm hold-still error |
    |---|---|---|---|
    | 0 (current deployment) | 0.030 | 0.78 (close 1.0, transport 1.8, lower 2.0: it OPENS while carrying) | 0.018 |
    | own previous command (training-consistent) | 0.0017 | 0.028 | 0.018 |
    | training pack, same robot (`packed_check_sem_panda.json`) | 0.004-0.011 | 0.001-0.03 | 0.03-0.09 |
    With the deployed input, system 0 is worse than holding still on the arm and inverts the gripper while carrying.
- Fix (opt-in, default keeps running jobs/resumes bit-identical): `PackedChunkDataset(..., zero_prev_action=True)` /
  config key `"zero_prev_action": true` for `train_representation` / `train_latent_flow` zeroes column 28 on node rows at
  load (`rrp.learning.packed.PREV_ACTION_COL`; guard test `tests/unit/test_prev_action_col.py`). Verified: only node
  column 28 and morph rows < n_nodes column 28 change. Stage A and Stage B must be retrained with it for a
  deployment-consistent model. Alternatively deploy with the prev-action input = own previous command (ladder
  `--prev-action own`); this is training-consistent in form but not in semantics (teacher vs own command, the D-021 copycat
  concern) — the ladder measures both.
- Affects also: old direct-action baselines (dev5/dev6), GRPO bases, binding-track representations (unless they set the
  flag), dualarm pack (if built from pre-D-021 datasets; check column 28).

## ladder table (panda_pg2 unless noted; dev seeds = first 30 feasible from 3,000,000; 300 ticks; replan 8; NFE 8)
Raw: peer `artifacts/runs/ladder_v1/<robot>/<route>_<tag>.jsonl` (+ `.summary.json`); summarize with
`scripts/ladder_peek.py <files>`. `prev` = what the deployed featurizer puts in node column 28 (B-1): `zero` = current
deployment; `own` = system 0's own previous command (training-consistent form). Label error = normalized 1-step MSE of
system 0 vs the shadow teacher's command at the visited state; on R0 it is the shadow system 0 on the teacher trajectory.
| rung | system 0 | prev | success (Wilson 95%) | failed stage | min TCP-cube | arm / grip label err | track q (rad) / TCP (m) |
|---|---|---|---|---|---|---|---|
| R0 teacher (privileged) | shadow: sem_v1 R | own | 30/30 (0.89-1.0) | - | 0.003 | 0.0017 / 0.023 | 0.025 / 0.021 |
| R0 teacher | shadow: sem_v1 R | zero | 30/30 (running) | - | | | |
| R0 teacher | shadow: refit R @2k (B-1 fix) | zero | 8/8 | - | 0.003 | 0.0074 / 0.056 | 0.025 / 0.020 |
| R1 oracle (ORACLE DIAGNOSTIC) | sem_v1 R | zero | 0/30 (0-0.11) | approach 30 | 0.42 | (diverged) | 0.006 / 0.005 (barely moves: 0.014 rad/tick) |
| R1 oracle | sem_v1 R | own | 0/30 (0-0.11) | approach 30 | 0.28 | copycat drift | 0.073 / 0.10 |
| R1 oracle | refit R @2k | zero | 0/16 | approach 14, grasp 2 | 0.084 | | 0.015 / 0.020 |
| R1 oracle, re-anchored expert | refit R @2k | zero | 0/16 | approach 13, grasp 1, lift 2 | 0.095 | | 0.017 / 0.022 |
| R2 flow v2@24543 | sem_v1 R | zero | 0/30 (0-0.11) | approach 30 | 0.44 | | 0.007 / 0.007 |
| R2 flow v2@24543 | sem_v1 R | own | 0/30 (0-0.11) | approach 30 | 0.33 | | 0.032 / 0.031 |
| R2 flow v2@24543 | refit R @2k | zero | 0/16 | approach 15, lift 1 | 0.22 | | 0.035 / 0.038 |
parm6_tf3 (seeds 3000003..3000041, 30 feasible): R0 30/30 (track 0.006 rad / 0.007 m); shadow sem_v1 R on the teacher
trajectory: prev own arm/grip 0.0018/0.0017, prev zero 0.0087/0.53 (arm hold-still reference 0.0048) -> B-1 on a
second body too.

Tracking: the joint tracker follows every route's commands closely (R0 0.025 rad mean lag at teacher speeds); failures are
never tracker failures.

Oracle vs generated z at the same states (R2 rows, `oracle_cmp`): probes read the generated packet as well as the oracle
packet (held_by/acting_on/subtask 1.0, rel_pos 0.045 vs 0.043 m). With the OLD system 0 the action from generated vs
oracle z differs by only 0.002 (it barely reads z; copycat of col 28). With the REFIT system 0 the difference is 0.89
(normalized arm MSE; hold-still 0.017): once system 0 actually uses z, the generator's z (flow v2, itself trained with
the B-1 input) drives it very differently from the oracle z -> the generator is the second failure point.

R1 failure mode with the refit realizer (trace `artifacts/runs/ladder_smoke/oracle_ticks_zero_rz2k_re.jsonl`): the arm
moves toward the cube but swings sideways first and the tool tilts progressively (tool z-axis 20-25 deg off vertical by
t=40-90); it descends next to/onto the cube and pushes it. Compounding 1-step error (covariate shift) + off-manifold
packets; the teacher's relabel at those states is a large wrist correction (0.4-0.6 rad).

## fixes being tested
1. Realizer refit on frozen E with col 28 zeroed (B-1): `configs/ladder/rz_sem_v1_b1fix_ft.json`, lease 1790370807_bd05c2.
2. System-0 DAgger: R1 (re-anchored expert) rollouts of the current refit realizer on all 13 source-train bodies, seeds
   3,200,000+ (24 feasible each), buffer = oracle posterior at each replan + learner-visited states with the shadow
   teacher's command; mixed 50/50 with the pack in `refit_realizer` (`"dagger": [...]`). Collection leases
   1790373150_81c015, 1790373151_61c3bd, 1790373151_2e8975, 1790373152_9c8793 -> `artifacts/runs/ladder_dagger_r1/<robot>.npz`.
3. Generator: flow retrain on the same frozen E with the B-1 fix (`configs/ladder/flow_sem_v2_b1fix.json`, 20k steps),
   lease 1790373182_289454 -> `artifacts/runs/ladder_flow_sem_v2_b1fix/`.

## method (code: `src/rrp/evaluation/ladder.py`, CLI `scripts/ladder.py`)
- Matched scenes: `feasible_seeds(robot, 3_000_000, n)` (same list for every rung), `n_distractors = seed % 3`.
- R0 `teacher`: scripted teacher (privileged) -> joint-target tracker. R1 `oracle`: ORACLE DIAGNOSTIC, frozen Stage-A E
  encodes the teacher's next 16 commands rolled out from the current closed-loop state (snapshot/restore; normalized with
  q0 at t, fp16-rounded like the pack), packet source `target_encoder_oracle`; frozen system 0 realizes it every tick.
  R2 `generated`: system-i flow (ODE Euler, NFE 8) from public observations. Replan every 8 ticks, validity 0.8 s,
  300 ticks.
- A shadow teacher advances once per tick at the executed state (DART-like) to give phase labels, the relabelled teacher
  command (label error `lab_err_*`, normalized MSE; `lab_step_arm` = hold-still reference), and in R2 the same-state oracle
  packet (z distance, system-0 action from each, probe readouts). It never controls R1/R2 except through E in R1.
- Tracking: `track_q_rad` mean |q_cmd - q_achieved| after the 50 ms tick; `track_tcp_m` |FK(q_cmd) - TCP achieved|.
- Failure stage from privileged geometry: approach (TCP within 2.5 cm of cube) -> grasp (held truth) -> lift (+4 cm while
  held) -> transport (cube within 4 cm of zone xy while held) -> place (privileged success).
- Checkpoint identities (sha256):
  - `latent_sem_v1/representation.pt` 8946aa9d2d0da5fca149786e867a4138ca9596976258abde2ef20aef4708db04 (ls-8db814f941f5)
  - `latent_nosem_v1/representation.pt` 49cea78db52c71c0cdbf3894453d0cdca59a5d329bbfac7d3a1a3b23388dd4cf
  - `ladder_ckpts/flow_latent_sem_v2_step24543.pt` (frozen copy of flow_latent_sem_v2/policy_interrupted.pt)
    ed886740e76ffe5f2638a169fc93c60164b5bafeddfbdcf1ae37cc1de4ab74c2
  - `grpo_base_snapshots/flow_latent_sem_v2_step22000.pt` bb6f454b93c4471708768adab34d9e05fef78da001738cd4da53d5c06a9b5df0

## log
- [verified] smoke: R0 panda 2/2 (track_q 0.026 rad, track_tcp 1.9 cm); R1 (prev=0) 0/2, never approaches (min TCP-cube
  0.47 m), label error 2.1 -> led to B-1. Commands in this file's bug section; raw `artifacts/runs/ladder_smoke/`.
- [running] lease 1790369958_1a7c76 `ladder_r01_panda`: R0 (shadow, prev own), R1 prev own, R1 prev zero; n=30 panda_pg2
  -> `artifacts/runs/ladder_v1/panda_pg2/{teacher_shadow_own,oracle_own,oracle_zero}.jsonl`.
- [running] lease 1790369958_fe952f `ladder_r2_panda`: R2 flow v2 step 24543, prev own / zero -> `.../generated_v2s24543_{own,zero}.jsonl`.
