# provenance track (W3: provenance and contracts, phase 1)

Owner: provenance agent. Worktree `~/work/rrp-wt/provenance`, branch `track/provenance`. Started 2026-09-26.
Scope: docs/strategy.md W3 and docs/repo_structure_audit.md section 3 / Phase 1. Additive only: no modules were moved (W4).

State: **verified** (code, unit tests, smoke runs below).

## API (`src/rrp/contracts/provenance.py`)
- `physics_provenance(model, contact_version=None) -> PhysicsProvenance`: mujoco_version, timestep,
  integrator, cone, impratio, solver, iterations, ls_iterations, noslip_iterations, contact_version.
  The contact track (W1) plugs in via `contact_version`: an explicit value, else the model's `contact_version` text
  element (track/contact `apply_world`), else contact_v1; a contradiction raises. Legged collectors pass the scenario's `meta["contact_model"]`
  (`rrp.morphology.contact.version_str`). Everything else defaults to `contact_v1`.
- `FEATURIZER_VERSION = "feat-v2"`: the single featurizer constant. `rrp.data.collect.FEATURIZER_VERSION` and
  `rrp.learning.behavior.FEAT_VERSION` are now aliases of it. The dual version stays derived: `feat-multi-v1+feat-v2`.
- `Source` enum: scripted_teacher, privileged_teacher, oracle (diagnostic), learned, bc, random, mock, cpg_tracker,
  learned_tracker, plus user, replay (legacy `contracts.action` values) and unknown (legacy reads only).
  `SourceLabel(kind, detail)` formats as `learned:<ckpt>`. `parse_source(s)` maps legacy strings: teacher,
  target_encoder_oracle, oracle, privileged_oracle_packet:*, oracle_diagnostic:*, scripted_controller→cpg_tracker,
  debug→mock, rep:*→learned. `parse_source(s, strict=True)` / `source_label(kind, ckpt)` are for new writes: they
  require a checkpoint for learned/bc and reject legacy aliases and unknown. On-disk strings of old runs are not rewritten.
- `Provenance` (pydantic, strict, `schema_version="provenance-1"`) has these fields: source, physics,
  featurizer_version, code (git_sha, dirty), weights (component→12-hex digest), bundle_fingerprint, versions, flags,
  legacy, created_at, notes. It supports `to_json` / `from_json` round-trips. Constructors: `make_provenance(...)`
  for new runs, `legacy_provenance(...)`, and `read_provenance(meta)` (the stored record, or legacy=True).
- `weights_digest(state_dict)`: the D-038 arm fingerprint, moved here. `control/latent_realizer.weights_digest`
  re-exports it, and a test checks that the algorithm is unchanged.
- `training_flags(config)`: zero_prev_action (None if unstated), realizer_drop_qd, realizer_anchor,
  realizer_qd_dropout, probe_lv_min, semantic weights, and similar flags.
- `resolve_zero_prev_action(cfg, where=, new_run=)`: raises `MissingFlagError` for a NEW run that omits it. A
  resumed or legacy reader gets a loud warning (warnings + stderr) and the legacy default False.

## wiring
- One manifest writer, `rrp.data.manifest.write_manifest(..., provenance=, filename=)` (hashed; the provenance
  is covered by the hash). It is used by arm `data/generate.py`, dual `data/collect_dual.py`, legged
  `data/legged_collect.py` (it replaces the unhashed hand-written manifest and keeps the old keys
  body/n/success/fell) and `data/legged_latent_collect.py` (per shard `<shard>.manifest.json`, plus `provenance`
  in `<shard>.json`). Every episode meta now carries `physics`. `dataset_provenance(episodes, ...)` builds the
  manifest record and records mixed or missing physics in `notes` instead of guessing.
- `read_manifest(path)` reads every format: hashed arm/dual manifests, old legged summaries, packed meta.json and
  legged shard .json. It returns `hash_ok` and `provenance` (legacy=True for old files). All real host manifests
  (pick_place_primary_v3/v3dart, support_insert_primary_v2, legged_latent_v2 shards, packed latent_pp_v3dart_s1_H16)
  read as legacy.
- Packed meta (`pack_dataset`) now stores `prev_action` (column 28 stored raw; zero_prev_action is the consumer's
  load-time flag) and `dataset_provenance`. `PackedChunkDataset` rejects `zero_prev_action=None`.
- `learning/checkpoint.save_checkpoint` stores `provenance` (the model weights fingerprint, versions and training
  flags) in the state and in the `.json` sidecar. `checkpoint_provenance(state, path)` marks older files as
  legacy/"unfingerprinted".
- Legged `_save` (legged_latent_train, reused by legged_bc and legged_dagger) keeps the bare-dict format and adds
  `_provenance`: weights of every state_dict entry (E/R/P/flow/model/R), git sha, legged flags (probe_lv_min,
  qd_dropout, semantic_weight, ..., prev_action_input=False), versions and the training-data physics. It also
  writes a `<name>.json` sidecar with the file sha256. `checkpoint_provenance` verifies the stored fingerprints on
  load (a mismatch raises) and marks bare checkpoints as legacy/unfingerprinted. `load_rep` prints a line for
  legacy files.
- Legged compatibility IDs: `legged_bundle_versions(base_lsv, E, R)` gives
  lsv = `legged-ls-<name>-w<E digest>` and rcv = `legged-rz-osc-v1-<lsv>-r<R digest>`. Both are computed from
  the weights actually loaded, so legacy checkpoints get the same IDs as fingerprinted copies and a refit realizer
  (`--realizer`) gets a new rcv. Eval rows now include latent_space_version, realizer_compat_version and
  checkpoint_provenance. Legged BC positive-control rows are now labelled `bc:<ckpt>`; before this change they
  were labelled `learned:<ckpt>`.
- zero_prev_action: `train_representation`, `train_latent_flow`, `refit_realizer` and `behavior.train_policy`
  (packed path) resolve it explicitly. A missing value errors for a new run and warns when resuming. The resolved
  value is written back into the config, so every checkpoint config states it. `fit_probes_on_frozen`,
  `sft_latent_flow` and `sft.py` warn when the source config omits it.

## config migration
`scripts/migrate_zero_prev_action.py --apply` (dry run without `--apply`) covers configs/latent (except pack-*),
configs/ladder and configs/model configs that use a packed_dir. It inserts `"zero_prev_action": false` textually
and checks that each file parses to exactly the old dict plus the new key.
Result: **22 configs migrated** (11 configs/latent flow_*, 8 configs/latent rep-*, 3 configs/model policy-*),
115 already explicit. `configs/ladder/armseed2/**` (33 files) was NOT touched; all 33 already state the flag, so
none needed migration. Legged configs are out of scope: the legged system 0 has no previous-action input
(recorded as `prev_action_input=False`).

## tests and smoke runs (raw outputs in `artifacts/runs/provenance_smoke/`)
- `tests/unit/test_provenance.py`: 11 tests, all pass. They cover physics extraction, JSON round-trip, legacy
  read, the Source mapping of every `contracts.action.Source` and `LatentActionChunk.source` literal, strict
  writes, the single featurizer constant, weights_digest equality with D-038, manifest write/read in new and old
  formats, legged fingerprint/legacy/tamper and compatibility IDs, save_checkpoint provenance, zero_prev_action
  errors and warnings, migration insertion, and a check that every in-scope repo config is explicit.
- Related suites pass under lease `1790473093_3bc997`: test_flow_model, latent_boundary, latent_contract,
  legged_latent, contracts, prev_action_col, psi_contracts, latent_grpo, surgery, interventions, legged.
- Collection smokes (host, `rrp ops run --cpu 1`):
  - arm: `rrp.cli data generate --config artifacts/runs/provenance_smoke/arm.json`. 4 episodes (parm5_pg2, 2
    clean + 2 DART; 2 success, 2 failure). The manifest provenance is scripted_teacher, feat-v2, physics
    implicitfast/elliptic/impratio 10/dt 0.002/contact_v1, git sha.
  - dual: `python -m rrp.data.collect_dual --config artifacts/runs/provenance_smoke/dual.json`. 2 support_insert
    episodes (2 success). Featurizer feat-multi-v1+feat-v2, same physics.
  - legged: `python -m rrp.data.legged_collect --body go2 --seeds 0-1`. 2 of 2 success. Physics: **euler,
    elliptic, impratio 100** (go2 menagerie option overrides, recorded for the first time), contact_v1,
    tracker_source learned_tracker, plus the tracker version.
  - legged latent: `python -m rrp.data.legged_latent_collect --body go2 --seeds 0-20`. 21 of 21 success,
    11525 ticks. The shard .json and s0-20.manifest.json carry provenance.
- Legged checkpoint smoke (`legged_eval_smoke.py`, output in `legged_eval_smoke.json`): a tiny rep (30 steps) and
  flow (60 steps) trained on the CPU. These tiny models are plumbing checks, not results. Three variants were
  tested: a fingerprinted checkpoint, a legacy bare copy (the exact pre-W3 format) and a stand-in refit
  realizer. All three load and generate packets with 0 rejections. The legacy and fingerprinted checkpoints give
  identical lsv/rcv. The refit gives the same lsv and a new rcv (`...-rccde349c4652` vs `...-r266ca84fb944`).

## left for W4 / follow-ups
- `scripts/peer_sync.sh` excludes `.git`, so peer runs record `git_sha=None`. A sync could write `.rrp_revision`
  (`{"git_sha":..., "dirty":...}`), which `code_provenance` already reads, or export `RRP_GIT_SHA`. I left the
  sync script untouched.
- Eval rows of other families (arm ladder, dual) still write free source strings. They parse with
  `parse_source`, but they should call `source_label()` when W4 moves them.
- `contracts.action.Source` / `LatentActionChunk.source` Literals are unchanged for wire compatibility. W4 could
  switch them to the enum.
- The old legged checkpoints on the peer store were not load-tested (per the brief, I did not touch the peer). The
  legacy path was tested on a bare checkpoint written in the exact old `_save` format. W4's load test over every
  peer `*.pt` should include `checkpoint_provenance`.
- Chain scripts that generate configs on the fly must now include `zero_prev_action`. New runs without it error
  by design.

## D-126 source labels (sl-1)
Switch: `RRP_SOURCE_LABELS=canonical` (env) or an explicit `source_labels=True` argument; **default off**
(unset / "" / "legacy"; any other value raises). `canonical_source_labels(enabled=None)` resolves it.
When on, `stamp_source_label(row, kind, detail)` ADDS `source_label` (strict `source_label()`) and
`source_label_version: "sl-1"` to a new row; the legacy `source` string is never changed and must agree in kind.
Off, rows are byte-identical to before. Readers use `row_source(row, default=None)`: `source_label` first, else the
legacy `source` via `parse_legacy_source` (also reads the ladder's "learned(system-i flow)" /
"target_encoder_oracle(...)" / "scripted_teacher(privileged)" forms). `contracts.action.Source` and
`LatentActionChunk.source` now also accept the canonical kinds (not "unknown"); old values serialize unchanged.
Notes: `research/tracks/d126_deploy.md` "source labels".
