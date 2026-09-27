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
