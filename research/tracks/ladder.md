# track: ladder — closed-loop failure localization (correction 2026-09-25, item 3)

Owner: ladder track agent. Branch `track/ladder`, worktree `~/work/rrp-wt/ladder`, peer dir `/dev/shm/rrp-brandonin/wt/ladder`.
Raw outputs live on the peer store `artifacts/runs/ladder_*` (copied summaries under `research/tracks/ladder/` when final).

## CURRENT ANSWER FOR ACCEPTANCE (which route/checkpoint is competent)
- **No competent learned route yet** (being measured; see table below — updated as jobs land).
- **CRITICAL BUG B-1 found (2026-09-25): train/deploy input mismatch in EVERY model trained on the packed datasets
  since D-021.** Any closed-loop result of Stage-A realizers, Stage-B flows, GRPO, baselines evaluated with the current
  featurizer is confounded by it. Details and fix below.

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
