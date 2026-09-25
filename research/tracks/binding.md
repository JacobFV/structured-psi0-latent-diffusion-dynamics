# track `binding`: make the Stage-A packet follow the supplied task binding (D-032)

Branch `track/binding`, worktree `~/work/rrp-wt/binding`, peer dir `/dev/shm/rrp-brandonin/wt/binding`.
State: **running**. Scope per `research/corrections/2026-09-25-causal-semantics-priorities.md` (D-037): items 1 (diagnostic
repair), 2 (whole-pipeline binding diversity, single-arm objects) and 5 (goal/predicted/observed effect).
Stage-A v2 (semantic-only counterfactuals) is an INTERMEDIATE result; the deliverable is the paired-task pipeline (section 5).

## 1. Is the v1 counterexample test valid? Partly not (fixed, see below)
Context layout (`data/features.py`): scene slots have structural edges ONLY to task-bank tokens: `role_points_to`
(role -> slot), `patient_of`/`destination_of` in BOTH orientations (event <-> slot), `pred_arg` in both orientations.
Pointers carry slot features into role/predicate tokens. The encoder is structured (`bias_mode=true`), so the binding
is visible through the relation bias and pointer messages; bank_text is unused.

Flaws of `counterexample` v1 (`src/rrp/evaluation/latent_counterfactuals.py`, now `counterexample_v1`):
1. **One-sided edit.** It retargeted only rows whose KEY is slot 0 (`R[:,2]==1 & R[:,3]==0`). Slot 0's own query-side
   edges (`slot0 -patient_of-> event`, `slot0 -pred_arg-> predicate`) stayed, so the edited graph bound the patient to
   slot 0 AND slot 2. Not a clean rebinding.
2. **Readout cannot flip.** In the whole training set the patient is canonical slot 0 and the destination slot 1:
   focus rate per slot index = [0.965, 0.186, 0, 0, ...] (200k packed rows). The v1 probes never saw a focused slot
   >= 2, so `focus_argmax_changed` could not become true even for a binding-aware z.
3. Minor: `argmax_changed` is a weak readout when two slots are focused.

Fix (`counterexample`, "v2_symmetric_rebind"): `model/binding_aug.rebind` swaps ALL structural edges of two scene slots
(symmetric row/column permutation of ctx_rel restricted to slots, pointers remapped); scene tokens, body and trajectory
are identical. The expected label is recomputed from the edited relations with the public focus rule
(`focus_from_batch`, identical to `learning/packed._focus`; checked in `tests/test_binding_aug.py`). Headline metric:
`focus_follows_changed` = probe focus ON for the newly bound slot and OFF for the old one, over pairs whose label
changes (an active event binds the patient). `counterexample_v1` still runs alongside for continuity with D-032.

## 2. Diagnosis: why the binding is ignored -> labels/data confound, not model capacity
- The patient is ALWAYS slot 0 with a fixed descriptor hash (scene-token hash dims have std 0 for slots 0 and 1).
  The binding is therefore perfectly predictable from slot content/index: the **metadata-only probe (no z) already gets
  focused_on 0.927** (`/dev/shm/.../latent_sem_v1/probe_metadata_only.json`).
- The demonstrated trajectory also goes to slot 0, so rel_pos/held/contact are trajectory-readable.
- Nothing in the objective ever rewards reading the role edges; any learner can ignore them. The structural bias
  weights are zero-initialised, so the edges stay unused.
- Evidence on v1 with the fixed test (host re-eval, `artifacts/runs/binding_v1_reeval/`): see table below.

Design 1 (chosen first): **counterfactual-binding augmentation.** Append rebound copies of 50% of each batch
(`LatentConfig.binding_cf=0.5`): same trajectory, body and scene tokens; slot edges swapped (src = a bound slot,
dst = an unbound slot when one exists, else the other bound slot); `focus` labels follow the binding; physical labels
(held, contact, rel_pos, future displacement, visibility, gaze) are unchanged because the body/world are unchanged;
realizer loss on factual rows only (a rebound context with the old trajectory has no valid action target); KL and
probe loss on all rows. sem and nosem get identical treatment (same augmented batches, same realizer/KL), differing only in
`semantic_weight`. Post-hoc probes for all variants (incl. v1) are refit with `fit-probes --binding-cf 0.5`, so the
probe has seen binding-dependent focus labels; the metadata-only control is refit the same way.
Designs 2/3 if needed: `binding_contrast` hinge (push E(cf) >= 0.25 rel. distance away), trajectory-feature dropout.
Known limitation for later: Stage B (flow) still trains on factual data only, where binding never varies.

## 3. Runs
| what | command | output | state |
|---|---|---|---|
| CPU smoke (30 steps) | `rrp latent train-representation --config <scratch smoke_sem.json>` | scratch | verified (runs; eval has `binding_counterfactual`) |
| v1 re-eval, fixed test | `scripts/binding_v1_reeval.sh`; host lease 1790363904_b7ef3b was killed by the host disk watchdog (another track's copy; old reserve) after the sem part; relaunched on the peer CPU-only (lease 1790366019_5303b7) | `artifacts/runs/binding_v1_reeval/` (peer) | running |
| Stage A sem v2 | peer lease 1790363882_585088, `configs/latent/rep-binding_latent_sem_v2.json` | `artifacts/runs/binding_latent_sem_v2` | running (~2.1 h) |
| Stage A nosem v2 | peer lease 1790363882_6525d1, `configs/latent/rep-binding_latent_nosem_v2.json` | `artifacts/runs/binding_latent_nosem_v2` | running |

## 4. Results
### v1 under the repaired test (encoded teacher targets, panda_pg2, 60 episodes x 2 samples, n=80 pairs with >= 3 slots)
| representation | probe | rel z change | focus_follows (label-changing pairs) | exact focus set, factual |
|---|---|---|---|---|
| latent_sem_v1 | factual post-hoc (D-031) | 0.047 | 0.0 (n=71) | 1.00 |
| latent_sem_v1 | post-hoc refit with `--binding-cf 0.5` | 0.047 | **0.0** (n=71; also 0.0 on its in-distribution held-out batches) | 0.95 |
| latent_nosem_v1 | factual post-hoc | 0.009 | 0.0 (n=71) | 0.775 |
| latent_nosem_v1 | post-hoc refit with `--binding-cf 0.5` | 0.009 | 0.0 (n=71; in-distribution 0.0) | 0.75 |
Raw: `artifacts/runs/binding_v1_reeval/{sem,nosem}_cf_probe_factual.json`, `sem_probe_bindcf.json` (peer store; small
JSONs mirrored in the host worktree). Even a probe trained on binding-varying labels cannot recover the binding from v1
z: the v1 packet does not carry it. The original D-032 numbers (4.9% / 0/14, asymmetric edit + slot-0 prior) stay
valid as a record of the flawed test and are reproduced by `counterexample_v1` in every counterfactuals.json.

### Stage-A v2 (design 1: semantic-only counterfactual augmentation on v3dart) -- intermediate, stopped
- `binding_latent_sem_v2` trained 8,700/30,000 steps (peer contention: ~1.3-2.4 s/step vs 0.1 in v1), then stopped
  with a resumable checkpoint (`rep_last.pt`, `representation_interrupted.pt`) to free GPU slots for v3;
  `binding_latent_nosem_v2` stopped at ~6,000. Train-log focus BCE plateaued at 0.40-0.42 from step 2,000 on
  (v1 reached 0.007 by step 4,000 without counterfactuals): the counterfactual rows were NOT being learned.
  Early check at step 6,000 (`scripts/binding_early_check.py`, jointly trained probe, 36 label-changing pairs):
  focus_follows 0.0, rel z change 6.1% (sem) / 0.2% (nosem). Raw: `artifacts/runs/binding_latent_sem_v2/train_log.jsonl`.
- Learnability diagnostic (`scripts/binding_diag_learnability.py`, focus-only loss, cf 0.5, fresh E+P, v3dart, demonstrated
  actions ZEROED): held-out cf_focus_follows 0.77 at 250 steps, 0.91 at 500, 1.00 at 2,000, 0.99 at 2,500
  (`artifacts/runs/binding_diag/learnability_no_traj.json`). So the structured encoder CAN read the supplied binding
  (pointers + relation bias) quickly; in v2 the demonstrated trajectory + slot/descriptor prior made the binding
  path not worth learning under the joint objective. (The with-trajectory arm of the diagnostic was stopped for GPU
  slots before it ran.)
- Also found: the scene bank has NO slot-address input. v1 z could name entities by slot index only because slot index
  == descriptor in v3dart (canonical order). With permuted slot order an encoder cannot address entities at all, so
  v3 adds `slot_handles` (learned embedding of the public tracker slot id on scene tokens; also in FlowPolicy).

## 5. Whole-pipeline binding diversity (item 2) and effect semantics (item 5)
- `sim/scenario.build_pick_place_paired` (+ BUILDERS `pick_place_paired`): the physical scene depends only on the seed
  (2-3 same-size cubes, distinct colors from 6, poses, green target zone); `patient` selects which physical cube the
  task assigns. Pairs therefore have IDENTICAL initial scenes and differ only in the supplied binding, each with its own
  scripted_teacher demonstration. Public slot order = seeded permutation over physical objects incl. the target zone,
  identical within a pair and independent of role (address permutation; patient/destination slots balanced).
  Verified on a smoke collection (parm6_tf3: identical slot tokens within a pair; focus label moves with the patient).
- `configs/data/pick_place_paired_v1.json`: 13 source robots x 200 scenes (seeds 2000000+), every assignment, + DART
  0.08 -> 13000 episodes. Peer CPU lease 1790369924_8948b2 (running, ~100 min).
- Outcome: 5,840 successes (clean 5,487/6,500; DART 353), 5,326 failures (5,230 DART), 1,834 infeasible (small arms,
  wider object region); 1,867/2,600 clean scenes have EVERY assignment succeed.
- Combined pack `artifacts/packed/binding_combined_v1_H16` (peer; `configs/latent/pack-binding_combined_v1.json`,
  multi-assembly format M=2): paired single-arm successes (790,610 rows; DART failures NOT included, unlike v1, since
  they average ~280 rows/episode and would have tripled the pack) + dual-arm assign_pick_place_v1 source_train
  successes with the 5 invalid "pushed, not grasped" DART episodes excluded (749,831 rows) = 1,540,441 rows.
  Arm-swap dev/held-out pairs are NOT in the pack (kept for evaluation).
- Stage A `binding_paired_{sem,nosem}_v3` (design 2 = paired factual data + binding_cf 0.5 + slot_handles +
  goal_effect probe; 20k steps; identical except semantic_weight) and the rest of the chain run as detached units
  `rrp-binding-chain-v3-{sem,nosem}` (`scripts/binding_chain_v3.sh`; logs `ops/logs/binding_chain_v3_*.log` on the
  peer): rep -> fit-probes (--binding-cf 0.5; + metadata-only for sem) -> counterfactuals on v3dart and paired data ->
  `latent_chain_v2.sh` (flow 20k, 20-episode dev eval on panda_pg2/parm6_tf3/parm5s_tf3/parm5l_pg2, disturbance,
  videos) -> `rrp latent eval-binding` (10 paired scenes x each assignment, panda_pg2 + parm6_tf3).
- Item 5: probe query `desired_delta` renamed `observed_effect` (it is realized future displacement, label
  future_disp); output alias `desired_delta` and a checkpoint key remap keep old probes/reps loading (checked on
  latent_sem_v1). New optional query `goal_effect` (probe cfg `goal_effect: true`): label from the task spec and public
  estimates = destination minus position of the object bound as patient of a not-yet-succeeded event, zero elsewhere
  (`binding_aug.goal_effect_from_batch`); it is recomputed after rebinding, so it follows the binding. Metrics:
  goal_effect_err_m, goal_effect_err_patient_m and the zero-prediction baseline.
- Infra: `scripts/peer_sync.sh` excluded `artifacts/`, which does not protect the peer SYMLINK; `--delete` removed it
  and the running collection recreated a local tmpfs `artifacts/` (294 episodes, merged back into the shared store,
  symlink restored). Fixed by excluding `/artifacts`. Other tracks using peer_sync push were exposed to the same race.

## resume
1. Poll: `ssh gb10-direct tail -n1 /dev/shm/rrp-brandonin/repo/artifacts/runs/binding_latent_{sem,nosem}_v2/train_log.jsonl`
   (training resumes from rep_last.pt if relaunched with the same command).
2. Then fit-probes (`--binding-cf 0.5`, plus `--metadata-only` for sem) and `counterfactuals --probe ... --n 60 --per-episode 2`.
