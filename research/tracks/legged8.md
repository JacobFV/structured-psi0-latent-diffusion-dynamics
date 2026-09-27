# track legged8 (W8): legged semantic-supervision study regenerated on contact v2

Owner: W8 legged agent. Worktree `~/work/rrp-wt/legged8`, branch `track/legged8`, peer code dir
`/dev/shm/rrp-brandonin/wt/legged8` (RRP_PEER_REPO). Started 2026-09-27 ~00:55.

Question: do the D-088/D-090/D-092 legged results (fixed sem = nosem on success and goal steering; a task-context
`halt` that slows only the semantic packet) hold when the data, trackers and evaluation physics are contact_v2 (D-093,
D-101)? Wave 1 = anymal_c (the only body whose contact_v2 tracker passes the full contact gate). go2 joins after the
contact agent's swing-floor fix, t1 after its turn-in-place fix (lead decides).

Sources (labels in every row/manifest): commands = `scripted_teacher` (privileged waypoint teacher); gait =
`learned_tracker:anymal_c:iter2499:contact_v2` (sha256 2a16532bbd07f7ab..., installed from the contact worktree into
`artifacts/trackers/anymal_c/contact_v2/` locally and in the shared peer store); our models = `learned:<ckpt>`; BC
positive control = `bc:<ckpt>`.

## how it runs
Everything goes through `rrp run-dag dags/legged_v2_anymal.yaml` (61 nodes; derived from dags/legged_fixrep.yaml):
- global nodes (planned once): `collect` (anymal_c seeds 0-599, shards of 100, DART sigmas 0/0.1/0.2/0.3, held-out =
  seed % 20 == 19: the go2 legged_latent_v1 counts/splits), `bc` (recipe of configs/legged_bc/bc_go2_v1.json, seed 0),
  `teacher` and `bcr2` (reference R2 evals, dev seeds 10000-10029);
- matrix variant {semfix (probe_lv_min -4), nosem, sem (original, optional)} x training seed {0, 1, 2}: rep ->
  [nosem probe] -> flow -> R2 snap_s4000 + final -> z-edit suite + task-context suite (dev 10000-10019, t_edit 2 s,
  window 2-5 s), exactly the D-090 recipe and suites.
- caps: <= 2 GPU nodes (defaults.max_parallel_gpu), summed declared CPU <= 16 (one 10-CPU eval + two 3-CPU GPU jobs).
- table: `python scripts/legged8_compare.py anymal_c` -> `artifacts/runs/legged8/legged8_compare_anymal_c.{md,json}`.

## rrp changes (small, tested; the legged pipeline's first real use)
1. `rrp.orchestration.dag`: `scope: global` nodes (planned once outside the matrix; point nodes reference them with
   "@collect"; global nodes cannot use axis values) — the fixrep DAG had no way to share a dataset/BC/teacher across
   points. Executor caps `max_parallel_gpu` / `max_cpu` (defaults or CLI `--max-parallel-gpu/--max-cpu`).
   Tests: test_global_scope_nodes_are_planned_once_and_shared, test_gpu_and_cpu_caps.
2. `rrp.pipelines.legged`:
   - GAP FOUND: flags.contact_version was recorded only; collect/evals/edits/DAgger subprocesses built their scenes from
     $RRP_CONTACT_MODEL (default v1), so a "contact_v2" DAG would have silently simulated contact_v1. Now every
     simulating subprocess gets RRP_CONTACT_MODEL = flags.contact_version (`physics_env`).
   - collect: sharded + parallel (`shard_size`, `workers`), verifies every shard manifest records the declared contact
     version and the episode count; output `data` root; metrics (success/fell, tracker versions).
   - eval_r*/edits: every row must report a scene built with the declared version and checkpoints whose recorded
     training-data physics match (`check_rows_contact`); eval rows now carry `contact_version` (legged_latent_eval).
   - check_contact_version: data with no recorded version (legacy = contact_v1) is refused for any other declared version.
   - new stage `train_bc` (legged BC positive control; moved from LEGACY_ONLY_STAGES to PIPELINE_STAGES).
   Test: test_legged_contact_v2_refuses_unversioned_data_and_mismatched_rows. Full suite on the peer: 323 passed, 2 skipped.

## findings so far
- Probe collection (40 episodes each, seeds 0-39, peer): v1 tracker in v1 physics 38/40 "success"; v2 tracker in v2
  physics 28/40, no falls in either. All v2 "failures" have the task runtime (public estimates) succeeded; the
  privileged end-of-episode check fails because the DART noise (sigma > 0) keeps making the v2 anymal_c lift a foot
  momentarily during the 1 s post-halt stand (truth stance_fraction 0.75 < 0.99 at the last tick; noise-free replays of
  those seeds stand with all four feet down). Training does not filter on status, so this changes no data; it is
  recorded as a property of the v2 gait under DART noise. The teacher reference (sigma 0) measures the clean rate.

## state
| step | state | evidence |
|---|---|---|
| worktree, tracker install, rrp changes + tests | verified | commit 7f39e66; peer pytest lease 1790496346_ab6066 rc=0 |
| gate 0 smoke (`dags/legged_v2_anymal_smoke.yaml`: 20 episodes, 300 steps, 2 eval episodes; semfix + nosem, seed 0) | verified | 17/17 nodes completed 01:03-01:10 (host log `artifacts/runs/legged8/smoke_rundag.log`, ledger `artifacts/runs/legged8/_dags/legged_v2_anymal_smoke/ledger.json`; outputs peer store `artifacts/runs/legged8/legged8smoke-anymal_c-*`). Every manifest records contact_version contact_v2 and git sha 7f39e66; eval rows: scene contact_v2, tracker `learned_tracker:anymal_c:iter2499:contact_v2`. After the checkpoint-contact check was added, 3 eval nodes were rerun (leases 1790496679_df690f, 1790496691_b94fe9, 1790496713_3e7266): checkpoint_contact flow/bc = contact_v2. Plumbing only (300-step models): teacher 2/2, BC 0/2 (falls), R2 0/2. Unit suite on the peer: 324 passed, 2 skipped (lease 1790496752_d09826). |
| full DAG, first launch (01:13) | failed (fixed) | collect lease 1790496817_215a50 shed by the peer watchdog (rc -10, available_memory_below_reserve): my 6 collectors grew to ~13 GB each in 80 s. Cause: every episode's LeggedSession (~0.45 GB in contact_v2) lives in reference cycles until a full GC. Fix: gc.collect() per episode in legged_latent_collect and legged_latent_eval (16-episode collector 5.8 -> 2.1 GB max RSS; 12-episode eval 2.1 GB). The retry was then refused by the broker's AdmissionStopped, which run-dag treated as a final launch error; it now waits on it like a capacity refusal. Commit 8236737 (main). |
| collect (lease 1790497229_6aa1e5, 01:20-01:24) | completed | 600 episodes s0-599, 433145 ticks, 0 falls, source scripted_teacher -> learned_tracker:anymal_c:iter2499:contact_v2, contact_v2 in every shard manifest. End-of-episode privileged "success" by DART sigma 0 / 0.1 / 0.2 / 0.3: 148/150, 133/150, 116/150, 67/150 (464/600): the noise-free rate matches go2 v1; the drop with sigma is the momentary foot lift under noise during the final stand (see findings). |
| Stage A semfix, clip-starvation check (lead request) | verified | `scripts/legged8_clipscale.py` (same statistic as legged_fixsem_geom.py: logged pre-clip grad norm every 200 steps, clip 1.0, scale = mean min(1, 1/gn)); raw `research/tracks/legged8/clipscale_stageA_semfix.json`. anymal_c semfix s0/s1/s2: median gn 14.4 / 10.2 / 11.3, mean update scale 0.125 / 0.132 / 0.138 (by thirds of training: 0.026-0.031 -> 0.083-0.101 -> 0.263-0.294). go2 references: D-088 s0 orig sem 466 / 0.0036, fixed sem 10.7 / 0.177, nosem 0.32 / 0.944; fixrep fixsem s1/s2 10.3 / 0.169, 9.9 / 0.167 (thirds 0.028-0.032 -> 0.11-0.12 -> 0.35-0.37); nosem s2 0.31 / 0.935. Verdict: above the lead's 0.05 flag threshold on every seed, about 25% less update than go2 fixed sem, 35x more than the D-085 defective recipe; the first third is clip-dominated in BOTH bodies (~0.03, the known early phase of the fixed recipe). No change to the recipe, so it stays identical to D-090. nosem values follow when its Stage A finishes. Held-out: realization MSE 0.0068 / 0.0072 / 0.0077 (zero-action 1.10), contact probe acc 0.898, swing 0.70-0.71. The coordinator was held to Stage A (`--only '^(rep|probe)@'`) until this check; flows started 02:22. |
| Stage A semfix s0/s1/s2 (leases 1790497473_476e4a, 1790497474_e2f4fa, 1790497475_f40b7f; 3 GPU nodes per the lead) | completed 02:22 | ~0.28 s/step with 3 concurrent (57 min). The original sem variant is dropped (lead); dags/legged_v2_anymal.yaml now has semfix x nosem only. |

## resume
`cd ~/work/rrp-wt/legged8; export RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/legged8`; never `scripts/peer_sync.sh push`
while DAG jobs run from that dir. Relaunch the coordinator (it resumes from the ledger and re-adopts running leases):
`PYTHONPATH=src ~/work/relational-robot-policy/.venv/bin/python -m rrp.cli run-dag dags/legged_v2_anymal.yaml [--point variant=semfix --point variant=nosem]`.
Outputs live in the peer store `/dev/shm/rrp-brandonin/repo/artifacts/runs/legged8/`.

## ANYMAL_C RESULT (contact_v2; wave 1). Table: `research/tracks/legged8/legged8_compare_anymal_c.{md,json}` (built by
`scripts/legged8_compare.py anymal_c` from the raw rows in the peer store `artifacts/runs/legged8/legged8-anymal_c-*`)
DAG `legged_v2_anymal`: 40/40 nodes completed (host log `artifacts/runs/legged8/full_rundag.log`). Sources:
learned:legged8-anymal_c-{semfix,nosem}/train_flow_s{0,1,2}/{snap_s4000,policy}.pt (DEPLOYABLE R2 route), gait
learned_tracker:anymal_c:iter2499:contact_v2 (sha 2a16532b...), commands scripted_teacher. Same protocol and statistics as D-090.
- Stage A clip scale (lead's check; `clipscale_stageA_all.json`): semfix 0.125 / 0.132 / 0.138 (median gn 14.4 / 10.2 / 11.3),
  nosem 0.968 / 0.978 / 0.970 (median gn 0.14 / 0.15 / 0.17). go2 D-088/D-090 references: fixed sem 0.17-0.18, nosem 0.94.
- Task-context HALT Δforward (t=2-5 s), training seeds 0 / 1 / 2 -> pooled 60: semfix -0.32 / -0.10 / -0.16 -> **-0.19 [-0.26, -0.13]**;
  nosem +0.08 / +0.43 / +0.12 -> **+0.21 [+0.14, +0.28]**. Every semfix seed below every nosem seed: exact one-sided permutation p = 0.05
  (two-sided 0.10); pooled difference -0.40 [-0.50, -0.30]. Relative size 7-26% of the unedited 1.2-1.5 m. Irrelevant control (mirror
  INACTIVE) Δforward +0.002 / -0.003. **D-088/D-090 replicates on a third body under contact_v2**, and nosem again walks FURTHER under halt.
- R2 success (30 dev seeds, no falls anywhere): semfix snap 28/26/30, final 26/26/26; nosem snap 29/29/27, final 30/28/25. Pooled final
  78/90 vs 83/90. Of the 25 non-successes over all 12 R2 runs, 22 are at the HALT stage with the task runtime reporting success
  (public success) but the privileged end check failing (a foot momentarily off the ground at the last tick: the same v2 anymal_c
  stance flicker as in collection); only 3 are real failures (60 s timeouts: semfix s0 snap 2 x drift toward waypoint a, nosem s2 snap 1).
  Counting public success, both variants are at 88-90/90 per checkpoint; the privileged-success gap is not a competence difference.
- Context goal steering (ACTIVE-INACTIVE toward): semfix +0.21 / +0.25 / +0.20 vs nosem +0.10 / +0.12 / +0.13 -> pooled +0.22 vs +0.12,
  every seed ordered (p = 0.05 one-sided). Unlike go2/hexapod6 (equal), on anymal_c semfix steers ~2x more.
- z edits: halt strong in both (-0.64 vs -0.76, not ordered); turn +0.6 +0.26 vs +0.20 (ordered, p 0.05), -0.6 -0.25 vs -0.20; goal readout
  +0.025 vs +0.011 (random |dz| 8: -0.006 both). Random |dz| 16/25 perturb semfix forward progress in both directions by seed (+0.14..-0.22).
- References on the same 30 dev seeds: scripted_teacher 30/30, bc:legged8-anymal_c-v2data/train_bc_s0/policy.pt 30/30, no falls
  (both run after the --point filter bug fix; the BC was shed once at 07:50 and rerun from scratch, lease 1790520661_7050cb).
- Clips (artifacts/video/INDEX.md): `2026-09-27_learned_ctxhalt_anymal_c_trainseed0_s10002_fixsem-vs-nosem_effect.mp4` (fixsem 2.12 -> 0.99 m,
  nosem 2.02 -> 1.80 m over t=2-8 s) and `..._s10017_..._noeffect.mp4` (fixsem 1.97 -> 1.70, nosem 2.39 -> 2.92). Reviewed a frame: the
  halted fixsem robot stops short of waypoint A; nosem reaches it.
- 08:28 second peer memory emergency (a W10 psi1z job grew ~23 GB in 2 min from 08:26 while the peer sat at ~114 GB): shed go2 rep
  semfix s1/s2 and t1 rep semfix s1 (the three old-code leases above the 2-lease allocation) at ~step 9000. They are retried with
  --retry-failed and resume from rep_last.pt, INEXACTLY (their checkpoints predate the CUDA-RNG fix; numpy/torch-CPU RNG restored).
  Recorded here; these runs are flagged in the wave-2 tables.

## WAVE 2 (D-103): go2 and t1 on contact_v2 (started 2026-09-27 06:02)
- Trackers (installed from the contact worktree into artifacts/trackers/<body>/contact_v2/ locally and in the peer store):
  go2 clearance-floor cf2 = learned_tracker:go2:iter799:contact_v2, sha256 af3f06f4e029e9f92fafc50e6174ffdb171c5512fbab4cd6fac18e6ce23cf18b
  (run artifacts/runs/contact_go2_cf2); t1 turn-trained = learned_tracker:t1:iter1199:contact_v2, sha256
  0d77322c0248e019403413da11aa118d782a08db42c8b7050cc40ed8b8f6c804 (run artifacts/runs/contact_t1_turn2).
  **t1 limitation (D-103):** wide lunging stance, forward ratio 0.84, CoT 1.77; every t1 result carries it.
- Provenance: collect episodes now record tracker_sha256, tracker_run and actuator; the dataset provenance versions carry
  tracker_sha256/actuator; eval rows carry tracker_sha256. The DAGs declare the sha (collect and teacher nodes) and the pipeline
  refuses a mismatch. Actuators: ideal PD servo (the default; the waypoint sim never applies the realistic actuator model of D-103).
- DAGs dags/legged_v2_{go2,t1}.yaml, generated from the anymal_c DAG. Differences: body, DART sigmas (go2 0/0.1/0.2/0.3, t1
  0/0.05/0.1/0.15, as each body's v1 data), names, declared tracker sha. Training seeds 0/1/2 for both (the D-087 t1 runs used 0/1/3).
- Data (verified): go2 600 episodes, 0 falls, end-check success 574/600; t1 600, 0 falls, 587/600. Teacher reference 30/30 on both
  (tracker sha as declared, contact_v2 rows).

## INCIDENTS / RESOURCE CHANGES (2026-09-27)
- 07:50 peer memory emergency (project memory ~115 GB of 119 across all agents): the watchdog shed 8 jobs, 3 of them W8 (anymal_c BC
  1790518634_6f0913, go2 flow s0 1790519727_dc576b, t1 rep s0 1790518687_411dc4). Just before the stop, my restart race let W8 hold 4 GPU
  leases (the old go2/t1 coordinators launched on the admission reopening while the anymal BC ran). Running leases were left to finish.
- Lead allocation (08:00): W8 <= 2 GPU leases, <= 8 CPU, <= 28 GiB declared in total. Implemented in rrp: run-dag `shared_budget`
  (caps over every DAG ledger of the track, so the three coordinators share them) + `max_mem_gib`; all DAGs capped 2/8/28.
  Declared memory from measurements: eval processes peak ~2.1 GB (gc fix) -> 4 workers = 10G; trainers' cgroup peak 1.6-2.0 GB, plus
  CUDA (unified memory; not in the cgroup): now logged as cuda_peak_mb every 200 steps, declarations to be tightened from it (7G for now).
- Exact resume: legged rep/flow/BC checkpoints now include the CUDA RNG state (numpy and torch CPU were already saved); resume prints
  exact/INEXACT (the anymal/go2/t1 runs already in flight were started before this; a resume of those would be INEXACT and is logged so).
- Bug fixed: run-dag `--point` filtering dropped scope-global nodes, so the first anymal_c run skipped the teacher/BC references.
  anymal_c teacher reference now run; BC (shed at 07:50) rerunning.
- Clips: `artifacts/runs/legged8/videos_claim.sh` claims one GPU slot in the shared budget, then renders anymal_c training seed 0,
  eval seeds 10002 (effect) and 10017 (no effect).

## D-106 (peer admission capped at 108 GiB RAM+GPU declared, 19.97 CPU, 8 GPU slots): W8 declarations and queue check
- Measured (08:30, new-code trainers): process peak (cgroup MemoryPeak) 1.95-2.09 GB; CUDA max_memory_reserved 1.96-2.14 GB (+ ~0.5 GB
  CUDA context not counted by torch). Declared per trainer/probe/BC node: mem 3G + gpu_mem 4G (was 7G + 5G). Eval/edit nodes: 4 workers x
  ~2.1 GB -> mem 11G. run-dag's track cap `max_mem_gib` now sums RAM + GPU, like the broker (test updated).
- Queue check on the live broker (scratch DAG, one teacher-eval node declaring 100 GiB, admission bound 150 s): run-dag logged
  "broker refused admission (CapacityError: memory 158913789952 > aggregate limit 115964116992); waiting (not an attempt)", made 0
  attempts, and failed only when my deliberately short 150 s bound expired. Real DAG nodes use the default 10800 s bound. Cosmetic fix:
  the refusal reason logged is now the broker's error line (was a traceback caret line).

## t1 PAUSED (D-107, 2026-09-27 ~08:45): "t1 torque limits 2-3x manufacturer"
The t1 body model's joint torque limits were 2-3x Booster's (knee 130 vs 60 N m, ankle 50 vs 20), so every t1 tracker, including the
turn-trained learned_tracker:t1:iter1199:contact_v2 (sha 0d77322c) used here, was trained on an unrealistically strong robot.
- The t1 coordinator is stopped (no new nodes). No t1 lease was running at that moment (the two t1 Stage A runs shed at 07:50/08:28 had
  not been relaunched), so nothing had to be stopped.
- Labelled "t1 torque limits 2-3x manufacturer": the ledger `artifacts/runs/legged8/_dags/legged_v2_t1/ledger.json` (top-level caveat +
  paused record + per node) and a `CAVEAT_t1_torque_limits.json` in every t1 output dir of the peer store (collect_s0 600 episodes,
  train_bc_s0, eval_r2-teacher_s0 30/30, eval_r2-bc_s0, partial train_rep_s0/s1 semfix). None of these enter any result.
- dags/legged_v2_t1.yaml carries a PAUSED header. A rerun starts from collection with the sourced-limit tracker; its sha must be
  declared anew (the pipeline refuses data and rows from any other tracker), with a fresh lineage/ledger.
- go2 keeps running and now has both W8 GPU slots (the track-wide cap is 2; t1 holds none).
