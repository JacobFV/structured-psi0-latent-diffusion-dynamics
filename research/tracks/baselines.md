# track: baselines (latent_slice1 four-way comparison, baseline methods)

## SPRINT BC RESULT (2026-09-25, sprint_bc; live section, updated as checkpoints land)
**Plain behaviour cloning with deployment-consistent input (B-1 fixed) IS a competent source controller on the ladder's
matched dev scenes, already at mid-training.** Same data (latent_pp_v3dart_s1_H16, stride 2), same seed 1701, same
scenes/seeds as the ladder (30 feasible dev seeds from 3,000,000; n_distractors = seed % 3; prev-action input 0 as
deployed; replan / execute prefix 8; nfe 8; privileged success evaluator). So the latent path's closed-loop failures
(oracle route 0-1/30, D-046..D-049) do NOT come from the data, the teacher's demonstrations, or the simulator/tracker
setup: the same demonstrations yield a competent controller. They come from the latent path (Stage A + system 0) and/or
from the oracle rung's construction (its packets come from a stateful shadow teacher that stalls off-trajectory; see the
D-049 diagnostic below). The decisive latent test is R2 (system i's own packet) against this BC on the same seeds.

| checkpoint (label) | panda_pg2 | parm6_tf3 | failures by stage (panda / parm6) |
|---|---|---|---|
| learned:direct1701_u12000 (direct-action BC, 12k/26.3k updates) | 25/30 = 0.83 [0.66, 0.93] | 27/30 = 0.90 [0.74, 0.97] | grasp 2, lift 2, transport 1 / transport 2, place 1 |
| learned:codec1701_u13152 (action-only codec BC, 13.2k updates) | 28/30 = 0.93 [0.79, 0.98] | 25/30 = 0.83 [0.66, 0.93] | grasp 1, place 1 / lift 1, transport 4 |
| reference: R0 scripted_teacher (ladder track) | 30/30 | 30/30 | - |
| reference: R1 oracle route, best (D-048, jfdag1 re-anchored) | 1/30 | 0/30 | mostly approach |

**Provenance audit of the direct-action snapshots (lead request, 21:45).** All `direct1701_u*` snapshots come from ONE
B-1-fixed run: root `artifacts/runs/latent_slice1_b1fix/baseline_direct_action/seed1701/source`, config
`.../source/config.json` (`zero_prev_action: true`, `exact_resume: true`, packed_dir latent_pp_v3dart_s1_H16), name
baseline_direct_action_seed1701. Updates 0-12,869 ran on the host (lease 1790381771_8e7e1c, log
~/work/relational-robot-policy/ops/logs/1790381771_8e7e1c_b1fix_baseline_direct_action.log) until the host watchdog
stopped it (disk_below_reserve; SIGTERM checkpoint policy_interrupted.pt sha 2483e8cfe3f5c6f6, policy_last.pt sha
09263cfa3569e14e). The SAME directory was copied to the peer store and resumed exactly from update 12,869 (peer unit
rrp-b1fix-direct, lease 1790391035_b47fcd, log /dev/shm/rrp-brandonin/repo/ops/logs/1790391035_b47fcd_b1fix_baseline_direct_action.log;
train_log.jsonl continues at step 12,900). The host copy stays frozen at 12,869 (MOVED_TO_PEER.txt). Each snapshot's own
embedded config says zero_prev_action=True, out_dir = the b1fix root above, step = its tag:
| snapshot | sha256[:16] (file = ladder summaries' recorded sha) | node at snapshot | embedded step / zero_prev_action |
|---|---|---|---|
| direct1701_u12000 | 5e6586bd6000412c | host | 12000 / True |
| direct1701_u15000 | e6b211e19d09c7d1 | peer (resumed) | 15000 / True |
| direct1701_u18000 | 0063fa12427c4259 | peer (resumed) | 18000 / True |
scripts/bc_ckpt_watch.sh copies policy_last.pt and keeps the copy only if its sha256[:16] equals policy_last.json's
sha256_16 at that moment (otherwise it discards and retries), so every snapshot matched its run's policy_last.json.
No snapshot comes from a pre-fix run (the pre-fix sources are under `artifacts/runs/latent_slice1/`, never read here).

**Learning curve (ladder dev scenes, 30 matched seeds per body; regenerate: `python3 scripts/bc_curve_table.py`)**
| checkpoint | panda_pg2 | parm6_tf3 | failed stages panda / parm6 |
|---|---|---|---|
| learned:codec1701_u13152 | 28/30 [0.79, 0.98] | 25/30 [0.66, 0.93] | place 1, grasp 1 / transport 4, lift 1 |
| learned:codec1701_u17000 | 25/30 [0.66, 0.93] | 25/30 [0.66, 0.93] | lift 2, place 1, transport 2 / transport 5 |
| learned:codec1701_u20000 | 27/30 [0.74, 0.97] | 29/30 [0.83, 0.99] | grasp 1, lift 1, transport 1 / place 1 |
| learned:codec1701_u23000 | 27/30 [0.74, 0.97] | 29/30 [0.83, 0.99] | place 1, lift 2 / transport 1 |
| learned:direct1701_u12000 | 25/30 [0.66, 0.93] | 27/30 [0.74, 0.97] | lift 2, transport 1, grasp 2 / transport 2, place 1 |
| learned:direct1701_u15000 | 23/30 [0.59, 0.88] | 27/30 [0.74, 0.97] | grasp 2, place 2, transport 1, lift 2 / place 1, transport 2 |
| learned:direct1701_u18000 | 30/30 [0.89, 1.00] | 29/30 [0.83, 0.99] | none / transport 1 |

Held-out source bodies (NOT in BC training; the protocol's source-competence bodies and harness: rrp.evaluation.runner,
seeds 2,000,000.., 50 episodes each, infeasible excluded), learned:direct1701_u12000: parm5s_tf3 44/47 = 0.94
[0.83, 0.98], parm5l_pg2 30/33 = 0.91 [0.76, 0.97]; pooled 74/80 = 0.93. All 6 failures are timeouts. Raw:
`artifacts/runs/baselines_bc_ladder/heldout/direct1701_u12000.{jsonl,summary.json}` (peer lease 1790388547_71cadb).

Wilson 95% in brackets. All failures of BC are timeouts late in the task (grasp/lift/transport/place), none at approach.
Raw: `artifacts/runs/baselines_bc_ladder/<robot>/learned_<tag>.{jsonl,summary.json}` (committed; peer store same path).

**Diagnostic for D-049 (the teacher is stateful): the shadow teacher is not a valid expert on learner-visited states.**
In the same BC episodes the ladder's shadow teacher (advanced once per tick at the executed state, as in every rung) ends
far behind the task: among the 25 panda_pg2 BC successes its final phase is pregrasp 10, descend 11, lift 3,
transport 1; parm6_tf3 (27 successes): pregrasp 4, descend 5, lift 10, other 8. Codec BC: the same pattern (panda
pregrasp 7 / descend 11 of 28). So BC completes pick-and-place while the teacher FSM, whose look-ahead is what the R1
oracle packet encodes and what DAgger relabels with, still says "hover at pregrasp". BC is stateless (public obs ->
chunk) and does not need the teacher's waypoint; the oracle route inherits the FSM's stale phase. Consequence: R1's
0-1/30 (D-046..D-049) cannot separate "system 0 cannot realize packets" from "the oracle packets are wrong off the
teacher trajectory"; the DAgger labels from this shadow are suspect for the same reason. The clean test of the latent
architecture is R2 (system i's own packet, no teacher) on a B-1-fixed flow, compared with this BC on the same seeds.
Data/teacher/sim are NOT the bottleneck: the same demonstrations yield a competent BC controller.
Checkpoints: snapshots of `policy_last.pt` (sha256 prefix verified against policy_last.json) in
`artifacts/runs/baselines_bc_ckpts/<tag>.pt` (host + peer store; not committed). Caveat: panda_pg2 and parm6_tf3 are
source-TRAINING bodies (as for the ladder); competence on held-out source bodies (protocol parm5s_tf3/parm5l_pg2) is
below.

**Semantic edits on BC (same suite, conditions, edits and measurements as the acceptance track's
latent_semantic_edits; BC has no packet, so the edit is applied to the public context BC observes at each chunk;
physical scene unchanged).** learned:direct1701_u12000, pick_place dev scenes from 3,000,000 (n_distractors =
max(1, seed % 3)), 24 seeds per body (parm6_tf3: 7 infeasible, n = 17):

| condition | panda_pg2 (n=24) | parm6_tf3 (n=17) |
|---|---|---|
| control: cube lifted / cube in zone | 23 / 22 | 17 / 13 |
| rebind_obj (task belief -> distractor0): distractor0 lifted / cube lifted / distractor0 in zone | 20 / 0 / 7 | 14 / 0 / 9 |
| goal_shift (goal belief +12 cm): cube at shifted goal / cube in old zone | 19 / 1 | 10 / 0 |
| irrelevant_distractor (unbound belief moved 10 cm): same lifted object as control / cube in zone | 23 / 21 | 17 / 14 |

Paired approach preference toward distractor0 vs control: rebind +0.30 m [0.27, 0.33] (panda), +0.31 m [0.26, 0.35]
(parm6); irrelevant edit +0.001 m [0.000, 0.002] / +0.000 m. So BC's behaviour follows valid context edits (object and
goal) and ignores the matched irrelevant edit. Caveat: this rebind swaps the tracker BELIEFS of the two objects, so it
tests "act where the task slot's object is", not binding. The VALID binding edit (descriptor + public binding only,
`rebind_desc`) was run on the same checkpoint by the acceptance track (research/tracks/acceptance.md "BC reference
result", 32 seeds panda_pg2): BC IGNORES it (first approach to the new cube 0/32; original cube in zone 25/32; effect vs
irrelevant +0.6 cm). The source data never varies the binding, so BC learned "slot 0", not "the bound entity". BC is
also not competent on the binding-paired scenes (permuted slots/colours it never saw). So BC is the competent-control
reference for manipulation and goal edits, and a NEGATIVE reference for binding (what the latent v4 path must beat).
Rebind completes the placement less often (7/20, 9/14 of the lifts; the distractor is a same-size cube). Not diagnosed;
likely the public task runtime (grasp/hold events bound to the cube) disagrees with the edited belief after the lift.
Raw: `artifacts/runs/baselines_bcsem_u12000/<robot>/semantic_{rows,summary}_learned_pick_place.*` (committed).
Command: `python -m rrp.evaluation.bc_semantic_edits --policy artifacts/runs/baselines_bc_ckpts/direct1701_u12000.pt
--label direct1701_u12000 --robots <r> --episodes 24 --out artifacts/runs/baselines_bcsem_u12000/<r>` (peer leases
1790388716_62f535, 1790388716_bce2bf). Videos: artifacts/video/2026-09-25_bc_semantic_{control,rebind_obj,goal_shift,
irrelevant_distractor}_parm6_tf3_s3000005_direct1701_u12000_followed.mp4 (all four followed).

**Incident 19:16-19:52: the host direct-action source was stopped by the host watchdog (disk_below_reserve) at update
12,869 (SIGTERM checkpoint) and every retry was refused (memory limit, host busy).** I stopped the host supervisor unit
rrp-b1fix-baseline_direct_action and resumed the run EXACTLY (exact_resume; checkpoint sha 09263cfa3569e14e copied)
on the peer: unit rrp-b1fix-direct (peer, dir wt/baselines, same scripts/baselines_host_b1fix.sh with RRP_NODE=peer),
lease 1790391035_b47fcd, output now in the PEER store
`/dev/shm/rrp-brandonin/repo/artifacts/runs/latent_slice1_b1fix/baseline_direct_action/seed1701/source`. The host dir
carries MOVED_TO_PEER.txt; do not resume there. The unit runs the source-competence cell after training. The watcher and
the b0 launcher (`scripts/bc_b0_after.sh`, host loops) follow the peer path.

Code (verified by smoke + these runs): `run_ladder` route `learned` (src/rrp/evaluation/ladder.py; `scripts/ladder.py
--route learned --policy <ckpt> --policy-label <tag>`): a LearnedPolicy chunk is submitted every 8 ticks and its rows
executed; shadow teacher, Meter and failure stages exactly as the other rungs. Learning-curve watcher
`scripts/bc_ckpt_watch.sh` (host loop, pid in `pgrep -af bc_ckpt_watch`; log ~/work/rrp-wt/baselines-logs/bc_watch.log):
every 3,000 updates of either source (and the final policy.pt) it snapshots and launches a peer CPU lease
`baselines_bcl_<tag>` (4 CPU, 8G) that runs both robots.

Commands: `scripts/ladder.py --route learned --policy artifacts/runs/baselines_bc_ckpts/<tag>.pt --policy-label <tag>
--robot <r> --n 30 --tag <tag> --out artifacts/runs/baselines_bc_ladder/<r>` (peer, CUDA_VISIBLE_DEVICES=).

Owner: baselines track agent. Branch `track/baselines`, worktree `~/work/rrp-wt/baselines`,
peer code dir `/dev/shm/rrp-brandonin/wt/baselines` (its `artifacts/` is the shared peer store).

## goal
Run the two BASELINE methods of the sealed protocol `configs/eval/latent_slice1.json` (SEALED_BEFORE_RESULTS; not
modified): `baseline_direct_action`, `baseline_action_only_codec` x seeds 1701/1702/1703 x targets
xarm7_pg2/xarm7_tf3/panda_tf3 x SFT budgets 0/5/20/100, 100 eval episodes per cell on seeds 2,000,000+, Wilson CIs.
The latent methods (`latent_nosem`, `latent_sem`) join later with the binding track's revised representation; the
aggregation script already reads their `run_latent_cell` outputs.

## what existed (2026-09-25 survey)
- No baseline runs anywhere: host `artifacts/runs`, peer `/dev/shm/rrp-brandonin/repo/artifacts/runs`, peer snapshot
  `~/rrp-peer-data/artifacts-snapshot-20260921/runs` contain only latent/flow/adapt/dev runs. No codec checkpoint
  (`codec_dev_v3`) survived; no `pp_v3dart_s2_H16` pack exists (named by policy-small-structured.json).
- Only the latent cell was implemented (`rrp.evaluation.latent_campaign.run_latent_cell`). The old `campaign.run_cell`
  targets the retired `primary` protocol (different keys: `budgets`, `sft`, `prefix`), reads pickled datasets, and its
  `sft()` fed raw actions as the flow target even for a codec policy (would have been a silent bug for the codec
  baseline: the codec-latent policy would be fine-tuned towards raw actions).

## implementation (state: verified by smoke; running)
- `src/rrp/evaluation/baseline_campaign.py` `run_baseline_cell(protocol, method, seed, target, budget)`; CLI
  `rrp campaign baseline-cell --method M --seed S --target T[,T] --budget B[,B] [--train-only] [--smoke]`.
  Idempotent: source (per method/seed, under a file lock), SFT, eval each skipped when outputs exist; a partial eval
  JSONL is discarded and redone. Output: `artifacts/runs/latent_slice1/<method>/seed<k>/{codec,source,sft/<tgt>_b<b>,
  eval/<tag>.jsonl + .summary.json, cells/<tag>.json}`. `target=source` = source competence on the held-out source
  bodies (parm5s_tf3, parm5l_pg2), 50 episodes, as in run_latent_cell.
- `rrp.learning.sft.sft_packed`: baseline SFT with the latent methods' exact acquisition (see fairness).
- `PackedChunkDataset(stride=k)`: t % k == 0 row subset of a stride-1 pack (= the chunk set of a stride-k pack,
  because `episode_samples` uses `range(0, T, stride)`); `train_policy` reads `packed_stride`.
- `scripts/baselines_slot.sh <method> <seeds...>`: serial queue for one GPU slot (source, then target x budget cells);
  `scripts/baselines_supervise.sh` (host, lightweight) re-launches the slot lease in <=6 h segments (broker cap) until
  the slot writes `.slot_done_*` (only when no cell failed). The slot defers a seed while `source/REMOTE_TRAINING`
  exists (a hook for training a source model elsewhere; currently unused, and the host script was removed).
- `src/rrp/evaluation/latent_slice1_report.py` (`python -m rrp.evaluation.latent_slice1_report`): per method x target x
  budget pooled success, Wilson 95%, per-seed k/n, target demo episodes / transitions / SFT updates, source updates,
  first sustained crossing of 0.8. Works on both baseline and latent cell layouts.

## fairness (same information, same acquisition)
- Source data: all four methods train on `artifacts/packed/latent_pp_v3dart_s1_H16` (13 source bodies, DART incl.
  noisy-execution failures with clean labels). Baselines use its stride-2 rows (the chunk set named by
  policy-small-structured.json). The codec autoencoder itself trains on `pick_place_primary_v3` stride 2, exactly as
  codec-small.json says (action-only, clean teacher episodes).
- Target adaptation: identical to `sft_latent_flow`: same target pack (`artifacts/packed/latent_targets/<tgt>`, 150
  clean teacher episodes per target), same nested episode choice `nested_budget_indices(len(eps), [b], seed)` over the
  sorted pack episode ids, same row sampler (`rng.sample(pool, B)`), B = min(128, max(8, rows)), AdamW lr 1e-4 wd 1e-4,
  grad clip 1.0, updates {5: 150, 20: 300, 100: 600}. The whole FlowPolicy is trained (the analogue of system i); the
  codec stays frozen (analogue of the frozen realizer). Hidden-state auxiliaries stay on during SFT (aux_weight 0.1).
- Evaluation: same scene builder, seeds 2,000,000.., n_distractors = seed % 3, max_steps 300, success =
  privileged_success, infeasible scenes excluded from the denominator, batch 25, nfe 8, replan every 8 control ticks
  (LearnedPolicy execute_prefix 8 = latent replan_ticks 8 = 0.4 s).
- Reporting: budget 0 = zero-shot transfer of the source-trained controller; budgets > 0 = SFT of that controller
  (a new controller per cell, initialized from source). Source pre-training updates are reported separately from
  target acquisition.
- Compute is NOT matched by the sealed configs: baseline source = 6 epochs x 4,385 updates x 256 chunks (~26.3k
  updates, 6.7M chunk presentations); latent flow = 30k updates x 128 (3.8M) plus the stage-A representation. Reported,
  not changed.

## deviations from the sealed protocol text (implementation choices, recorded before any result)
1. Data location: policy-small-structured.json names `artifacts/packed/pp_v3dart_s2_H16`, which no longer exists.
   Used the stride-2 row subset of `latent_pp_v3dart_s1_H16` (same source dataset v3dart, same robots, include DART
   failures, H 16): the same chunk set, not a different recipe.
2. Codec-latent FlowPolicy config: the protocol only says "FlowPolicy on codec latent". Used the direct baseline's
   policy config (policy-small-structured: width 256, 6 blocks, aux on, same data/epochs/lr) with latent_dim = codec
   latent_dim 4 and the frozen per-seed codec (codec-small.json, trained with the same seed). Only the target space
   differs between the two baselines.
3. Source competence episodes: 50 per held-out source body (as run_latent_cell); the protocol has no source count.
4. The protocol's eval block has no execute_prefix/batch; used prefix = replan_ticks = 8 and batch 25 (as the latent
   cell).
5. Output path: the user asked for `artifacts/runs/latent_slice1/<method>/seed<k>` (shared with the latent cells),
   instead of the brief's `artifacts/runs/<track>_...` naming.
6. No registry writes (the brief reserves research/registry.jsonl for the lead); every cell writes its own JSON.

7. Source-training resume: the 6 h lease cap interrupts each ~8 h source run. The stock resume repeats the interrupted
   epoch with the step counter continuing (extra, seed-dependent updates). Baseline sources use `exact_resume`
   (per-epoch data order Random(seed*1000+epoch), batch-skip cursor saved every 1000 updates and on SIGTERM, flow
   noise generator state), so every source model gets exactly 6 epochs = the same update count. Verified: SIGTERM at
   update 9 + resume -> 24/24 updates, max weight difference 1.8e-6 vs an uninterrupted run (peer lease
   baselines_resume_test).

## incidents
- 12:27 and 13:18 (2026-09-25): both seed1701 source runs were stopped and restarted from scratch by me to pick up
  first the prefetch change, then exact resume (lost ~1 h of compute). Their partial logs are kept in
  `source/aborted_pre_exact_resume/`. No results existed. codec seed1701 was kept (finished, not affected).

## runs
Smoke (verified 2026-09-25): `rrp campaign baseline-cell --smoke --root artifacts/runs/baselines_smoke` on the peer,
both methods (codec: source + xarm7_pg2 b0,b5; direct: source + panda_tf3/xarm7_tf3 b0,b100). All paths ran: codec
training, source training (30 steps), SFT (10 steps), eval (3 episodes per robot), cell JSON, aggregation.
Throughput on the shared peer: ~0.6 s/update at batch 256 => ~4.4 h per source model; SFT ~0.33 s/update; eval
~1 s per episode.

Campaign (running): see "status" below.
- codec seed1701 (`.../baseline_action_only_codec/seed1701/codec/result.json`): 4,572 updates, 195,297 train chunks,
  train reconstruction MSE 0.00017 (shuffled-latent 0.41, zero-action 0.25; normalized actions).

## status (2026-09-25 13:50 PDT)
State: running, restricted by the lead's priorities (`research/corrections/2026-09-25-causal-semantics-priorities.md`):
seed 1701 only, source-competence cells plus budget-0 transfer cells. Seeds 1702/1703 and all SFT budget cells
(5/20/100) are ON HOLD until the lead says otherwise.
- Peer lease `1790369387_f24d3b` (baselines_direct_action) and `1790369387_158758` (baselines_action_only_codec), each
  running `BUDGETS=0 scripts/baselines_slot.sh <method> 1701` through `BUDGETS=0 scripts/baselines_supervise.sh <method> 1701`.
  Both seed-1701 sources resumed exactly from their SIGTERM checkpoints (direct update 924, codec 1060).
  After each source finishes, the slot runs `--target source` (competence on the held-out source bodies) and the
  three `<target>_b0` cells, then writes `.slot_done_<method>_1701_b0` and stops.
- The direct-action source checkpoint (the "expert native commands -> tracker" comparison for the ladder track) will be
  `/dev/shm/rrp-brandonin/wt/baselines/artifacts/runs/latent_slice1/baseline_direct_action/seed1701/source/policy.pt`
  on the peer (the same file is in the shared store `/dev/shm/rrp-brandonin/repo/artifacts/runs/latent_slice1/...`). NOT
  finished yet (~26.3k updates at 1-2 s/update on the shared peer).
- Results: none yet.

## resume
- Check: `ssh gb10-direct 'systemctl --user list-units "rrp-job-*" --no-legend | grep baselines_'`;
  `pgrep -af "baselines_(host_)?supervise"`; peer log `/dev/shm/rrp-brandonin/repo/ops/logs/<lease>_baselines_direct_action.log`.
- Restart the current (restricted) slots from the worktree (idempotent/resumable):
  `BUDGETS=0 setsid nohup scripts/baselines_supervise.sh baseline_direct_action 1701 > ~/work/rrp-wt/baselines-logs/supervise_baseline_direct_action.log 2>&1 < /dev/null &`
  `BUDGETS=0 setsid nohup scripts/baselines_supervise.sh baseline_action_only_codec 1701 > ~/work/rrp-wt/baselines-logs/supervise_baseline_action_only_codec.log 2>&1 < /dev/null &`
- Full grid, only once the lead releases it: the same commands with seeds `1701 1702 1703` and without `BUDGETS=0`.
- Single cell by hand: `PYTHONPATH=src python -m rrp.cli campaign baseline-cell --method M --seed S --target T --budget B`
  (prefix with `rrp.cli ops run ...` on the host, or `scripts/peer_run.sh ... -- PY -m rrp.cli ...` on the peer).
- Aggregate (after pulling peer JSONs: `rsync -a --include='*/' --include='*.json' --include='*.jsonl' --exclude='*'
  gb10-direct:/dev/shm/rrp-brandonin/wt/baselines/artifacts/runs/latent_slice1/baseline_direct_action/ artifacts/runs/latent_slice1/baseline_direct_action/`):
  `PYTHONPATH=src python -m rrp.evaluation.latent_slice1_report --out-md research/reports/latent_slice1_tables.md`.
- Latent methods: run_latent_cell writes `artifacts/runs/latent_slice1/latent_{nosem,sem}/seed<k>` on the peer; pull
  them the same way and the report picks them up.
