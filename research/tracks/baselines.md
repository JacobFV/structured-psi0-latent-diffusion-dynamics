# track: baselines (latent_slice1 four-way comparison, baseline methods)

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
  the slot writes `.slot_done_*` (only when no cell failed). `scripts/baselines_host_source.sh <method> <seed>`: optional
  host slot that trains one source model and rsyncs it to the peer; the peer slot defers that seed while
  `source/REMOTE_TRAINING` exists.
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

## runs
Smoke (verified 2026-09-25): `rrp campaign baseline-cell --smoke --root artifacts/runs/baselines_smoke` on the peer,
both methods (codec: source + xarm7_pg2 b0,b5; direct: source + panda_tf3/xarm7_tf3 b0,b100). All paths ran: codec
training, source training (30 steps), SFT (10 steps), eval (3 episodes per robot), cell JSON, aggregation.
Throughput on the shared peer: ~0.6 s/update at batch 256 => ~4.4 h per source model; SFT ~0.33 s/update; eval
~1 s per episode.

Campaign (running): see "status" below.

## status
(updated below as results arrive)
