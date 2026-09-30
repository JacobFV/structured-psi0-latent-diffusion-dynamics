# track: rel-r2 (relation-factor fanout, unit R2: arm latent config + flags -> specs)

Branch `track/rel-r2`, worktree `~/work/rrp-wt/rel-r2`, from `origin/main` @ `9be709dd` (D-144 R1, tip at the time
this unit started; R9 `a66959c5` and R1 were both already ancestors). Spec: `docs/relations.md` sections 2-5 and 10
(row R2, brief in "briefs"), `research/relations_catalog.md`, `AGENTS.md`. Host only (git / editing / unit suite;
CUDA hidden; no training beyond CPU unit fixtures).

## scope

Moved `LatentConfig.semantic_weight / probe_lv_min / binding_cf / binding_contrast / slot_handles`
(`nets/semantic_latent.py`) off their own ad hoc scalar fields onto the registry the foundation (F2-F4) and R1
already built:

- `LatentConfig.factors: tuple | None` (docs/relations.md 10: "Weight, lv_min ... become spec weight/params"): each
  arm probe query's loss weight and Gaussian log-variance floor is a `probe.arm.<query>` FactorSpec override
  (`weight`, `params.lv_min`); the public slot-address embedding is the `id.slot_handle` factor -- both already
  registered by the foundation / R1, not new catalog entries. `LatentConfig.weight` / `.lv_min` / `.slot_handles`
  are now properties reading `resolve(self.factors)`.
- `cf_mix` / `cf_contrast` (ex `binding_cf` / `binding_contrast`) stay plain scalars: the counterfactual-binding
  swap has no catalog entry of its own (it is a training-loop data augmentation, not a relation over token fields),
  and `catalog.py` is not a file this row owns (see "what I deliberately did not do" below).
- `__init__` accepts EITHER the new `factors=` (+`cf_mix`/`cf_contrast`) OR the flat legacy keys (mixing raises,
  same rule as `PolicyConfig.from_dict`); every stored checkpoint's embedded config, and two OTHER units' owned
  files (`bundles.load_representation` line 46: R1; `policies/latent.py` line 40: R3) call
  `LatentConfig(**some_dict)` directly and are not in this row's owned-file list, so backward-compatible
  construction is not optional here.
- `LatentConfig.version()` reproduces the pre-R2 hash bit-for-bit over every config under `configs/` that
  constructs a `LatentConfig` (frozen BEFORE this unit's changes with the origin/main code; test file has the frozen
  table). `_check_variant` (`core/runconfig.py`) now also accepts `params.latent.factors` (new-style), in addition
  to the untouched `params.latent.semantic_weight` branch (old-style; still exercised by
  `test_runconfig.py::test_variant_must_match_recipe`, which is not owned by this row).
- `nets/binding_aug.py` deleted; its functions moved to `nets/latent_batch.py` (an owned file that already existed
  for an unrelated helper, `assembly_batch`) with NO renames except at the one call site the R9 port required (see
  below) -- `rebind` is `binding_aug.rebind` verbatim, unchanged tensor math, for training-hot-path speed.
  `harness/train/latent_train.py`'s local `binding_cf_metrics` renamed to `cf_swap_metrics` (a private helper this
  row owns outright; no external caller of that name).
- Ported onto R9's `harness.data.relgen.transforms.cf_swap`: proven, not just asserted -- `rebind` and a reference
  bridge through the ACTUAL registered `cf_swap` transform (Batch row -> generic relgen `Sample` shape -> `cf_swap`
  -> back) agree on `ctx_rel` / `act_rel` on a synthetic fixture (`tests/unit/test_relations_r2_latent.py::
  test_rebind_matches_cf_swap_reference`). The bridge itself lives in the TEST, not in `nets/latent_batch.py`:
  `policies` is a lower architecture layer than `harness` (`tests/unit/test_layering.py`), so production code may
  not import `harness.data.relgen`; running the per-sample numpy transform on every row of a real GPU training
  batch, every step, would also be a real throughput regression the hot path cannot afford. Known, documented gap:
  `Batch.pointers` (index-pair incidence list) is outside cf_swap's `{tokens, edges, labels}` dict contract, so the
  equivalence fixture uses a batch with zero pointer rows; `rebind` keeps remapping `pointers` directly (the same
  permutation cf_swap would compute for the token axis).
- `configs/latent/*.json` (all 10 files with a `"latent"` block) codemodded: `semantic_weight` / `binding_cf` /
  `slot_handles` -> `factors: [...]` / `cf_mix`. Verified byte-identical `LatentConfig(...).version()` before/after
  and byte-identical `RunConfig.load_legacy(...).to_native()` round-trip (both proven by the test file, and were
  proven interactively before the codemod ran -- see commands below).
- Two files NOT in this row's owned list broke on import when `nets/binding_aug.py` was deleted
  (`harness/eval/latent_eval.py:31`, `harness/data/latent.py:76`, both `from rrp.policies.nets.binding_aug import
  goal_effect_from_batch`) and one integration test did too (`tests/integration/test_binding_aug.py`, not run by
  the host gate -- needs `artifacts/packed/...` -- but its import would fail for anyone with that data). Fixed all
  three (one-line import path only, `binding_aug` -> `latent_batch`; zero logic changes) rather than leave a known
  breakage: "protect existing work" outweighs the ownership boundary for a mechanical fix that is a direct,
  unavoidable consequence of a deletion this row's brief explicitly asks for.

## what I deliberately did NOT do (and why)

**`semantic_weight` / `probe_lv_min` / `binding_cf` are not literally zero-occurrence in `src/`.** Three narrowly
scoped, clearly commented exceptions remain, each one forced by a currently-passing test or a real backward-
compatibility requirement I verified concretely rather than assumed:

1. `LatentConfig.version()` (`nets/semantic_latent.py`) reconstructs the pre-R2 dict shape internally, using those
   three names as literal JSON object keys, ONLY to keep the hash byte-identical. A SHA-256 preimage cannot be
   reproduced under different key names; this is not stylistic, it is the acceptance criterion itself
   ("`LatentConfig.version()` identical for every config under configs/") forcing the other one's letter.
2. `core/provenance.py`'s `TRAINING_FLAG_KEYS` keeps `probe_lv_min` / `semantic_weight` / `packet_semantic_weight`.
   Confirmed concretely, not assumed: `tests/unit/test_provenance.py::
   test_legged_checkpoints_fingerprinted_and_legacy` builds a LEGGED checkpoint config with the flat
   `probe_lv_min`/`semantic_weight` keys and asserts `checkpoint_provenance(...).flags["probe_lv_min"] == -4.0`.
   Legged (fanout unit R4, not merged) reads these as plain dict keys, never through `LatentConfig`, and this row
   does not own `harness/train/legged_latent_train.py` or `policies/bundles.py`'s `LEGGED_FLAG_KEYS` to migrate it
   there instead. Removing the keys from this generic, family-agnostic key list would have silently stopped
   recording them for every still-legacy-style config, arm or legged, and broken that test. Only `binding_cf_weight`
   (an arm-only, never-actually-set entry in the same tuple) was safely renamed to `cf_mix`.
3. `core/runconfig.py`'s `FLAG_NAMES` / `Flags.probe_lv_min` / `FLAG_SPEC` / `LEGACY_FLAG_DEFAULTS` are untouched.
   Confirmed concretely: `test_runconfig.py::test_legacy_classification_and_flags` asserts
   `rep.flags.probe_lv_min == -8.0` for an arm ladder config this row does not own, and legged's `train_rep`/`refit`
   stages declare `probe_lv_min` / `qd_dropout` in the same shared `FLAG_SPEC`. `_check_variant` WAS extended (see
   above) rather than left a pure no-op, because that addition is purely additive (only taken when
   `"factors" in params.latent`) and does not touch the old branch those tests exercise.
4. `harness/train/latent_train.py::fit_probes_on_frozen`'s `binding_cf: float` parameter keeps its name: it is a
   plain CLI/pipeline float option (not `LatentConfig.cf_mix`), called by keyword from `cli/latent.py`
   (`a.binding_cf`, R1-owned) and `harness/pipelines/arm.py` (`o.get("binding_cf", ...)`, owned by neither this nor
   any other single fanout row I could find). Renaming it would break both call sites, which this row may not edit.

**`packet_semantic_weight` / `aux_weight` (top-level, Stage-B/BC-only knobs, distinct from `LatentConfig`) were not
migrated onto `factors:`.** They have no `probe.arm.*`-shaped registry entry to become (Stage B trains a FROZEN
probe on `z_hat`, not the representation probes the registry entries describe), the same `packet_semantic_weight`
key is read by `policies/bundles.py` and `harness/train/legged_latent_train.py` (both owned by other units), and
`aux_weight` (`harness/train/{behavior,sft,vlm_train}.py`) is a `FlowPolicy.loss` keyword with no `LatentConfig`
involvement at all. The brief paragraph names both as in-scope; the "owns" column and the actual call graph say
otherwise for this pass. Recorded here as real follow-up work, not silently dropped.

**`dags/**` was not codemodded.** `LatentConfig`'s legacy-kwargs backward compatibility (proven, not assumed: see
the version-identity tests) means every existing, UNMODIFIED dag continues to produce byte-identical
`latent_space_version` strings, so nothing is broken by leaving them. Dags template real, potentially in-flight
distributed training runs (`dags/arm_lineage.yaml`, `dags/templates/dual_lineage.yaml`, ...) with NO unit-test
coverage in this repo to verify a YAML template rewrite is exactly equivalent before it reaches a real pipeline run;
AGENTS.md's "protect existing work" weighs against an unverifiable rewrite of live pipeline definitions in a pass
with no way to execute or check them here (host = git/editing/unit suite only). Deferred, not dropped.

## commands run (host, CPU only)

```
export PYTHONPATH=$PWD/src:$PWD
pytest tests/unit -q                              # 621 passed, 39 skipped (pre-existing, unrelated: Menagerie
                                                   # assets / optional extras not fetched on this host)
pytest tests/unit/test_relations_r2_latent.py -q  # 26 passed (this row's red/green acceptance tests)
```

Version-identity freeze (before any change, against `git show origin/main:src/rrp/policies/nets/semantic_latent.py`
re-implemented inline since the real dataclass's `@dataclass` decorator needs its defining module in `sys.modules`)
and post-codemod re-check are both reproduced by `test_relations_r2_latent.py::test_version_identical_to_pre_r2`
and `test_frozen_set_covers_every_latent_config`.

## open questions for the lead

1. Is the "no semantic_weight / probe_lv_min / binding_cf in src/" acceptance line meant literally (repo-wide,
   zero substring occurrences), or as "not LatentConfig's own construction surface / RunConfig's arm-facing
   validation path" (what I implemented)? The literal reading is mathematically incompatible with
   "`LatentConfig.version()` identical" (a SHA-256 preimage) and with two currently-passing, out-of-scope tests
   (`test_provenance.py::test_legged_checkpoints_fingerprinted_and_legacy`,
   `test_runconfig.py::test_legacy_classification_and_flags`) that this row does not own and must not break. I
   resolved every instance in favor of NOT breaking those, with each exception narrowly scoped and commented at
   its source (see above). If the intent really was repo-wide and those two tests are meant to be updated too,
   that is a decision for whoever owns them (not this row: they are general infra tests, not R2/R4's file).
2. `packet_semantic_weight` / `aux_weight`: worth a small follow-up row once R4 (legged) merges and
   `harness/pipelines/arm.py` / `policies/bundles.py` ownership is clearer, so the migration can touch the shared
   readers without guessing at another unit's boundary.
3. `dags/**` latent-key codemod: deferred; safe to do once there is a way to validate a rendered dag's `RunConfig`
   end-to-end (e.g. from the peer, or once R11's `rrp steer` / pipeline tooling is available to dry-run one).
