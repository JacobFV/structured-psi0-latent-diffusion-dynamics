# track `binding`: make the Stage-A packet follow the supplied task binding (D-032)

Branch `track/binding`, worktree `~/work/rrp-wt/binding`, peer dir `/dev/shm/rrp-brandonin/wt/binding`.
State: **running** (Stage-A v2 training on peer).

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
| v1 re-eval, fixed test | `scripts/binding_v1_reeval.sh` (host GPU lease) | `artifacts/runs/binding_v1_reeval/` | running |
| Stage A sem v2 | peer lease 1790363882_585088, `configs/latent/rep-binding_latent_sem_v2.json` | `artifacts/runs/binding_latent_sem_v2` | running (~2.1 h) |
| Stage A nosem v2 | peer lease 1790363882_6525d1, `configs/latent/rep-binding_latent_nosem_v2.json` | `artifacts/runs/binding_latent_nosem_v2` | running |

## 4. Results
(pending)

## resume
1. Poll: `ssh gb10-direct tail -n1 /dev/shm/rrp-brandonin/repo/artifacts/runs/binding_latent_{sem,nosem}_v2/train_log.jsonl`
   (training resumes from rep_last.pt if relaunched with the same command).
2. Then fit-probes (`--binding-cf 0.5`, plus `--metadata-only` for sem) and `counterfactuals --probe ... --n 60 --per-episode 2`.
