# rel-r10 track (D-144 fanout unit R10: relgen stage + shards + mix)

Owner: Sonnet fanout agent. Worktree `~/work/rrp-wt/rel-r10`, branch `track/rel-r10` from `origin/main` @ `c5e7d45`
(D-144 foundation). Brief: `docs/relations.md` section 10, row R10 + its brief paragraph. Host only (git / editing /
unit suite; CUDA hidden; no training/simulation): no peer smoke was needed (the row's acceptance is a unit-test
fixture run, not a training/simulation run), so `scripts/peer_run.sh` was not used.

## what changed (owned files only)
- `src/rrp/harness/pipelines/relations.py` (new): the `relations_data` stage.
  - `EpisodeSnapshot(env, task, seed, view, index, step)`: what a caller (env-specific collection code, or a
    relabeller reading recorded snapshots) hands the stage. The stage itself never talks to a simulator.
  - `label_episode(ep, factor_specs, rng)`: for each resolved `FactorSpec`, looks up its `FactorDef` (foundation
    field, untouched); factors without a `.label`, or whose label's `needs` the view's `caps` don't cover, are
    skipped (not an error — lets the stage run correctly against catalog sections other units haven't filled in
    yet). A factor WITH a label computes it via `rrp.harness.data.relgen.LABELS[fdef.label].fn(view, index)`, then
    runs each name in `fdef.gen` that is a registered `TRANSFORMS` entry, in order, letting a transform multiply one
    sample into several (reveal / cf_swap-style). `ScenePart`/`compose` (scene construction) stays R11's job; this
    stage only ever applies `TRANSFORMS`, which need no new scene.
  - `relations_data(factors, episodes, out_root, seed)`: resolves the factor list (`relations.base.resolve`, so
    presets/globs/overrides all work unchanged), labels+transforms every episode, and writes one shard + manifest
    per producing factor.
  - `write_shard` / `read_shard_manifest` / `load_shard_rows`: the shard/manifest format this unit owns —
    `artifacts/relgen/<factor>/<version>/manifest.json` (`rrp.harness.data.manifest.write_manifest`, the one
    existing manifest writer, reused rather than inventing a second one) + `<shard_id>.npz` holding each row's label
    `value`/`valid` arrays. A manifest's rows accumulate across repeated `relations_data` calls into the same
    `<factor>/<version>` dir (e.g. one call per seed), each pointing at its own shard file.
- `src/rrp/harness/data/mix.py`: implemented `mixed_batches` (`allocate`, already F4's, is unchanged) and added
  `mask_missing_labels`. `mixed_batches(main, scheduler, shards, batch_size, rng)` draws `scheduler.sample(step, n)`
  each step, pools relgen rows from `shards[factor]` (an in-memory `{factor: [Sample, ...]}`, e.g.
  `pipelines.relations.load_shard_rows` per factor — composed multi-factor active sets are R11's `compose`; until
  then the active set's first (sorted) factor name stands in for its pool), fills the rest of the batch from `main`,
  and returns `sum(counts.values()) == batch_size` exactly every time (the same integer split `allocate` computes
  from the scheduler's current share). `mask_missing_labels` gives every sample in a batch every requested label
  name, `valid=False` where it doesn't already carry one, using the array shape of any label already present in the
  batch — so downstream stacking code never sees a missing key.
- `tests/unit/test_relgen_stage.py` (new, 8 tests, all red before this change since `relations_data` didn't exist
  and `mixed_batches` raised `NotImplementedError`): registers a fixture `LabelDef`/`TransformDef`/two `FactorDef`s
  (test-only entries; nothing added to `catalog.py`, which this unit does not own) and a duck-typed `StateView`.

## acceptance evidence (row R10, docs/relations.md section 10)
Command: `cd ~/work/rrp-wt/rel-r10 && export PYTHONPATH=$PWD/src:$PWD && .venv (main checkout)/bin/python -m pytest
tests/unit/test_relgen_stage.py -v` → 8 passed. Full suite: `pytest tests/unit -q` → 559 passed, 39 skipped
(pre-existing skips: Menagerie assets / packed data / optional extras not present in this checkout / peer-only), 0
failed.
- **"2-episode fixture run writes shards + manifest with per-sample provenance (active set, label versions)"**:
  `test_relations_data_writes_shard_and_manifest` runs `relations_data` over 2 fixture episodes, asserts the written
  manifest rows carry `active` (the full resolved factor-name list) and `labels` (`[{"label", "prov", "version"}]`),
  and that `artifacts/relgen/<factor>/<version>/{manifest.json,*.npz}` exist and round-trip
  (`read_shard_manifest` -> `hash_ok is True`; `load_shard_rows` -> the same `Label` arrays back).
  `test_relations_data_no_producers_is_not_an_error` and `test_relations_data_appends_across_shard_runs` cover the
  "no factor with a label" and "multiple stage runs share one manifest" edges.
- **"`mixed_batches` fractions exact and seed-deterministic"**: `test_mixed_batches_fractions_are_exact` asserts
  every batch's `counts` equals `allocate(batch_size, scheduler_share)` exactly (sums to `batch_size`).
  `test_mixed_batches_is_seed_deterministic` runs the same scheduler config + shards + `rng` seed twice and asserts
  byte-identical batch counts and relgen row provenance.
- **"missing labels masked"**: `test_mixed_batches_masks_missing_labels` (main rows, which carry no relgen label,
  come back with `valid.any() == False`, `prov == "none"` for the pooled factor's label; real relgen rows are
  untouched) and `test_mask_missing_labels_shapes_from_a_present_label` (the mask borrows its shape from a label
  already in the batch).

## what R10 explicitly did NOT do (by design / row scope)
- Did not touch `relations/base.py` or `relations/ops.py` (forbidden) or `catalog.py` (not owned by this row —
  no real factor has a `.label`/`.gen` yet; only test-local `FactorDef`s exercise the stage).
- Did not wire `relations_data` into `rrp.harness.pipelines.base`'s `Pipeline`/`StageSpec`/`register()` machinery or
  add `"relations_data"` to `rrp.core.runconfig.PIPELINE_STAGES`: that tuple, and `register_family`'s stage-name
  validation against it, are closed and live in `core/runconfig.py`, which this row does not own — doing so would
  be an interface change on a shared foundation file outside R10's scope. `relations_data` is therefore a plain
  function today (`harness.pipelines.relations.relations_data`), matching the row's owned-files list and acceptance
  criteria exactly (a fixture-run + manifest test, not a `rrp stage run relations_data` CLI test). Flagged in
  `lead_questions` below.
- Did not implement scene composition (`relgen.compose`, still `NotImplementedError`) or the scheduler's real
  decision policy (`relgen/curriculum.py`'s placeholder policy) — both are R11.
