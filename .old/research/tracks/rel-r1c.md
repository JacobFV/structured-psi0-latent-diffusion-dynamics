# track: rel-r1c (relation-factor fanout, R1 deferred-scope follow-up: delete nets/latent_probes.py for real)

Branch `track/rel-r1c`, worktree `~/work/rrp-wt/rel-r1c`, from `origin/main` @ `566fccc6` (rebased onto later `main`
before merge, see below). Spec: `docs/relations.md` sections 2-5 and 10 (row R1, brief, section 8.2 delete list),
`research/tracks/rel-r1.md` (R1's original merge and its "open question for the lead"), `research/decisions.md`
D-144 addendum (R8 lead decisions, already on `main` when this unit started -- checked first, not duplicated here).
Host only (git / editing / unit suite; `CUDA_VISIBLE_DEVICES=` for every test run); no peer smoke needed (pure host
unit tests).

## scope (what R1 deferred, closed out here)

R1's original merge (`9be709dd`) swapped every construction site of `nets.latent_probes.PacketProbe` for
`nets.probes.ReadoutProbe` (preset `probes:arm-packet-v1`) but **kept `nets/latent_probes.py` on disk**, because
deleting it would have broken `tests/unit/test_golden.py`'s `_tiny_latent` fixture (still built a real `PacketProbe`)
and every out-of-scope caller of `probe_loss` / `probe_metrics` (unrenamed, kept as the literal acceptance
reference). This unit finishes that:

1. **`tests/unit/test_golden.py`'s `_tiny_latent` fixture** now builds `ReadoutProbe(8, 4,
   specs=["preset:probes:arm-packet-v1"], width=32, heads=2)` right after the same `torch.manual_seed(4)` the old
   `PacketProbe(8, 4, width=32, heads=2)` used. Same module names / parameter creation order (docs/relations.md 4;
   `nets/probes.py`'s own docstring states the layout contract) -> byte-identical state dict and forward outputs to
   the retired class, so every golden hash that exercises this fixture (`loop.arm.latent`, `loop.dual.latent`,
   `loop.semantic_edits.arm`, `loop.semantic_edits.dual`, the assignment-edit loops) is unaffected. Verified: full
   `pytest tests/unit/test_golden.py` green, `git status --short tests/data/golden.json` clean (no re-record).
2. **`nets/latent_probes.py` deleted** (`PacketProbe`, `probe_loss`, `probe_metrics`, `probe_loss_multi`,
   `probe_metrics_multi`, `gaussian_nll`, `_goal_terms`, `goal_metrics`) -- docs/relations.md 8.2 delete list.
3. **Every `probe_loss` / `probe_metrics` caller switched to `readout_loss` / `readout_metrics`**:
   `harness/eval/latent_eval.py`, `harness/eval/dual_latent_eval.py`, `harness/eval/ladder.py`,
   `harness/train/latent_train.py` (probe lines only, as R1 owned), `harness/train/joint_adapt.py` (an
   out-of-scope-for-R1 caller R1's own track note flagged; fixed here since deleting the module now breaks it
   too), `tests/unit/test_latent_boundary.py`. `gaussian_nll` callers (`tests/unit/test_probe_lv_min.py`) switched
   to the foundation's `nets.probes.gaussian_nll` (verified formula-identical: same clamp/NLL/mask reduction, `d=3`
   either way since every arm/dual target is a 3-vector).

   **Design deviation, flagged for the lead**: `readout_loss` / `readout_metrics` are NOT re-exports of
   `nets.probes.readout_loss` / `readout_metrics` (the generic, spec-driven foundation pair R5 added, now on
   `main`). They are the arm/dual math **ported unchanged** from the deleted `probe_loss` / `probe_metrics` (incl.
   the `_multi` dual/dispatch variants), renamed, now living in `harness/eval/latent_eval.py` (already the shared
   import hub for arm/dual probe-scoring utilities -- `dual_latent_eval.py` already pulled `acc_probe_counts` from
   there). Reason: the deferred-scope instructions are explicit -- **"metric key names reported by hooks must not
   change"** -- and the generic `nets.probes.readout_metrics` cannot satisfy that for this family:
     - key scheme differs (`<query>_acc` / `<query>_mae` / `<query>_cos_err` vs. the established `visible`,
       `rel_pos_err_m`, `desired_delta_err_m`, ...);
     - it has no positives-balanced `*_pos` views (rare-positive diagnostics every arm hook reports);
     - it has no goal-effect patient / zero-baseline breakdown;
     - `gauss`-loss queries: generic = per-component mean absolute error: arm's `rel_pos_err_m` /
       `observed_effect_err_m` = mean L2 (Euclidean) error in metres -- a different NUMBER, not just a different key;
     - the dual `held_m` / `contact_m` / `rel_tcp_m` / `subtask_m` per-slot (`@m`) addressing + pooling has no
       equivalent in the generic single-mask contract at all (each `ReadoutDef.label` is one fixed key, not one per
       packet slot), and the legacy single-assembly labels (`lab["held"]` etc., trimmed to manipulator 0) don't
       have the shape the generic path would need for `entity×asm` queries.
   Every caller therefore imports `readout_loss` / `readout_metrics` from `rrp.harness.eval.latent_eval` (not
   `rrp.policies.nets.probes`) -- literally satisfying "callers switch to `readout_loss`/`readout_metrics`" while
   keeping every reported key and value unchanged (checked directly in
   `tests/unit/test_relations_r1_probes.py::test_readout_loss_and_metrics_report_the_legacy_arm_key_set`, and
   indirectly by every golden hash that scores a packet probe staying byte-identical). This is analogous to R5's own
   precedent for Ψ₀ (`psi0/nets.py` keeps its own `probe_loss` / `probe_metrics` names, delegating to the generic
   pair only where the family's contract actually matches it).
4. **Bug found and fixed in the same commit** (surfaced only once the golden fixture stopped being a real
   `PacketProbe`, which still carried the deprecated `desired_delta` output alias): `harness/eval/
   latent_semantic_edits.py`'s `probe_orthogonal` (used by the `orthogonal_matched` condition, part of
   `PAIRED_CONDITIONS` and therefore exercised by `test_semantic_edit_loop_rows`) read `out["desired_delta"]`
   directly. That key was already gone from every OTHER `ReadoutProbe` construction site since R1's original merge
   (`load_representation`, `latent_train.py`'s fresh `P`, `latent_causal.load_probe`, ...) -- `latent_semantic_edits.py`
   just wasn't on R1's owned-file list, so nothing exercised this path with a real `ReadoutProbe` until this
   fixture moved. Fixed by reading `out["observed_effect"]` instead (`# D-144 R1: desired_delta output alias
   dropped`, matching the pattern `latent_causal.py` line ~108 already used). For the OLD `PacketProbe`,
   `out["desired_delta"]` and `out["observed_effect"]` were literally the same tensor (`out["desired_delta"] =
   out["observed_effect"]`), so this is a no-op for any *other* remaining `PacketProbe` caller and the golden hash
   this fixes (`loop.semantic_edits.arm`) is unaffected (checked: identical to the value recorded on unmodified
   `main`).
5. Old checkpoints still load strictly (decision (b) / R1's `_remap_probe_state_dict`, untouched;
   `test_relations_r1_probes.py`'s remap test now derives its synthetic "old" state dict from a `ReadoutProbe`
   itself, since `PacketProbe` no longer exists to instantiate -- the identical-layout guarantee means this is
   equivalent, not weaker).

## lead_questions / flags (not blockers)

- **Pre-existing, out-of-scope `desired_delta` dict-key references** found by `grep` (not exercised by any test, so
  not fixed here -- neither file is on R1's or this unit's owned-file list, and both already read a `ReadoutProbe`
  output since R1's original merge, so this is not something this unit introduced): `src/rrp/viz/workbench/
  sessions.py:438` (`out["desired_delta"]`, live interactive workbench view) and `src/rrp/harness/eval/
  latent_counterfactuals.py:159` (R2-owned file; `o["desired_delta"]`). Both will `KeyError` the first time they
  run against a real trained checkpoint's probe. Flagging for R2 (owns `latent_counterfactuals.py`) and whoever owns
  `viz/workbench/*` to fix with the same one-line swap to `observed_effect` used here.
- Whether the arm/dual `readout_loss` / `readout_metrics` in `harness/eval/latent_eval.py` should eventually be
  folded into the generic `nets.probes` contract (e.g. by letting `ReadoutDef` carry an explicit metric key /
  positives-balanced flag) is a foundation design question, not something this unit's file-ownership lets it decide
  -- flagged rather than acted on, per docs/relations.md 10's "a unit that needs an operator/foundation change
  reports it to the lead instead."

## tests

Full `pytest tests/unit -q` (host, `CUDA_VISIBLE_DEVICES=`, `PYTHONPATH=$PWD/src:$PWD`): 782 passed, 39 skipped
(environment-only: menagerie assets not fetched, `computerworld` extra not installed, `transformers` not installed --
none of this unit's concern), 0 failed. `tests/data/golden.json`: `git status --short` clean (byte-identical,
no re-record needed for this unit -- decision (a)'s re-record allowance was not needed here).

New/changed tests: `tests/unit/test_relations_r1_probes.py` rewritten (module docstring + `_pair`/`_probe` fixture
no longer need `PacketProbe`; acceptance-3 replaced by a direct key-set/value assertion on `readout_loss` /
`readout_metrics` since the old PacketProbe-vs-ReadoutProbe live comparison can no longer run once the class is
deleted; added `test_latent_probes_module_is_gone`). `tests/unit/test_relgen.py`: removed
`test_readout_probe_equals_packet_probe` (same reason). `tests/unit/test_latent_boundary.py`,
`tests/unit/test_probe_lv_min.py`: import swaps only, same assertions.
