# track: rel-r5 (relation-factor fanout, unit R5: Ψ₀ nets)

Branch `track/rel-r5`, worktree `~/work/rrp-wt/rel-r5`, from `origin/main` @ `c5e7d45` (D-144 foundation). Spec:
`docs/relations.md` sections 2-5 and 10 (row R5 + its brief), `research/relations_catalog.md`. Host only
(git / editing / unit suite; CUDA hidden via `CUDA_VISIBLE_DEVICES=""`); no peer smoke needed (the row's
acceptance is pure unit tests on fixtures, no training/simulation).

Owned files (per the row): `src/rrp/policies/psi0/{nets,train}.py`, `src/rrp/bodies/g1_simple.py`. (`psi0/data.py`
does not exist on `main`; nothing in the row's scope needed it.) Shared file touched in its own section only:
`src/rrp/policies/relations/catalog.py` (new `# R5:` section, `s0-psi0` route preset + `probes:psi0-v1` +
optional grasp-region factors). `relations/base.py` / `relations/ops.py` were never edited.

## scope

- **`Realizer`'s READS mask -> `route.assembly_reads` factor.** The legacy inline `-inf` fill of `Morph.read_mask`
  inside `Realizer.forward` is replaced by `FactorSite("dims>knots", ...)` built from the `s0-psi0` preset
  (`route.assembly_reads`, `params.reads = bodies.g1_simple.reads_table()`), applied through the shared `RelBlock`
  (attention.py) instead of the private `block`/`run_block` pair. `PacketEncoder` (unrelated to system 0's routing)
  keeps `block`/`run_block` unchanged. The exact -inf pattern is unchanged (`test_route_assembly_reads_matches_read_mask`)
  and the realizer's numerical output is unchanged (`test_realizer_output_matches_pre_r5_formula`, both green).
  `bodies/g1_simple.py` gained `READS` / `reads_table()` as the canonical source (moved out of `psi0/nets.py`,
  which now imports it); `psi0.nets.READS` is kept as an alias and `read_mask()` (legacy shape, still used by its
  own golden + by `Morph.read_mask`) is rebuilt from `reads_table()` instead of duplicating the mapping.
- **`PacketProbe` -> `ReadoutProbe` (`probes:psi0-v1`).** `catalog.py` registers the 7 core `probe.psi0.*` queries
  (`hand_dist`, `contact`, `lift`, `target_pos`, `active_hand`, `base_disp`, `base_cmd`) in the old `PROBE_QUERIES`
  order plus preset `probes:psi0-v1`, and the optional grasp-region pair (`probe.psi0.grasp_pt`, `.grasp_face`,
  default off, roadmap #24 / D-126) as standalone factors added by `nets.probe_specs(grasp=True)`. `new_probe()`
  builds a `ReadoutProbe` from this spec; `run_probe()` adapts the generic `knot×pair` addressing to psi0's
  per-knot-only queries (pair slot 0: `lift`, `target_pos`, `base_cmd`) and the grasp queries (last knot only).
  `probe_loss` / `probe_metrics` now mostly delegate to the foundation's generic `readout_loss` / `readout_metrics`
  (`nets/probes.py`), keeping only the psi0-specific bits the generic path can't infer: `active_hand`'s `-1 = none`
  mask, `hand_dist`'s label reshape, and the grasp pair's runtime `w_grasp` weight + validity mask.
  `base_cmd`'s old key-masked read (only `READS["base"]` tokens) is **not** reproduced by the generic probe
  (`ReadoutProbe` has one key mask per forward call, not per query) -- `base_cmd` now reads the full packet like
  every other `knot×pair` query. Flagged below; not blocking (no test on `main` asserted the old restricted read).
- **Checkpoint loading.** `load_tolerant` now drops every `P.*` key when it detects a pre-D-144 `PacketProbe`
  checkpoint (module keys `P.a1.`/`P.a2.`/`P.kcode.`/`P.code.`), so `R`/`E`/`morph` still load strictly and the
  probe is refit fresh (the two architectures are incompatible, so "load" for `P.*` was never a real option --
  tracked as the row's `load_tolerant` acceptance item). A post-D-144 checkpoint's `P.*` loads and is checked
  strictly like everything else. `load_stage_a` detects `grasp` from either the old (`P.grasp_*`) or new
  (`grasp_pt`/`grasp_face` in state dict / label names) key spelling.
- **`nets/probes.py` (foundation file, not in this row's `owns` list): added `readout_metrics`.** Pure addition
  (no existing line touched), the metrics-side twin of the foundation's existing `readout_loss` -- R1's brief
  already names `readout_metrics` as the target of psi0/arm's old `probe_metrics`, so this looked like a foundation
  gap rather than an R5-only need. Flagged in lead_questions since it's outside the row's file list; safe to keep
  (additive, no other unit touches `nets/probes.py` yet per a worktree sweep) or to move into F/R1 later.

## fixed during this session (continuing interrupted work)

The worktree had uncommitted D-144 R5 changes from an earlier, interrupted run. `pytest tests/unit -q` was RED on
resume: `test_probe_metrics_match_r5_baseline` failed because `_goldens_r5()` never called `N.add_cmd_labels(b)`
before probing, so `labels["base_cmd"]` was absent and the `base_cmd_mae` metric silently never appeared (readout
metrics skip a query when its label key is missing) -- a test-harness bug, not a `nets.py` bug (`train.py`'s own
training loop already calls `add_cmd_labels` correctly, confirmed by reading its diff). Fixed the harness to match
`train.py`'s usage and re-recorded `GOLDEN_R5` from the corrected, deterministic output (command in the file's
comment). Note for whoever next touches `ReadoutProbe`: enabling `grasp=True` changes `qtype`'s embedding size,
which shifts global-RNG-seeded init for every *later*-constructed submodule (`att`/`att2`/heads/`mlp`), so the
non-grasp metrics differ between `grasp=False` and `grasp=True` runs even at the same `torch.manual_seed` -- unlike
the old `PacketProbe`, which built the grasp head strictly last so shared modules kept identical init either way.
Not a correctness bug (nothing requires cross-config identity; `test_grasp_head_off_by_default_and_loss_only_when_weighted`
only checks `grasp=False` against the default, both False), but worth knowing before treating a `GOLDEN_R5` diff as
automatically an intentional change.

## tests (red then green)

`tests/unit/test_psi0.py`. Confirmed red-before-green from the interrupted state (`test_probe_metrics_match_r5_baseline`
failing per above, see below); all other new tests were already present and passing when this session resumed.
Spot-checked that `test_route_assembly_reads_matches_read_mask` actually discriminates (not a tautology) by
computing `R.route.bias(rc)` directly and diffing it against a deliberately wrong all-zero bias (mismatch, as
expected) and against the intended `-inf`-fill formula (exact match) -- did not need to touch `Realizer.forward`
itself to check this.

- `test_nets_match_psi1z_goldens`: the psi1z-original subset (`tables`, `read_mask`, `context_tokens`) still exact.
- `test_probe_metrics_match_r5_baseline`: regression baseline on the migrated `ReadoutProbe`-based probe (not
  psi1z-comparable; the probe architecture changed).
- `test_route_assembly_reads_matches_read_mask`: `route.assembly_reads`'s bias == the legacy `read_mask()` fill.
- `test_realizer_output_matches_pre_r5_formula`: `Realizer(...)` output byte-identical to the pre-R5 formula
  reimplemented inline against the same submodules (R5's "Realizer output byte-identical" acceptance item).
- `test_probes_psi0_v1_preset_and_grasp_optional`: `probes:psi0-v1` resolves to the 7 queries in order; grasp off
  by default, on when requested.
- `test_load_tolerant_drops_pre_r5_packet_probe_keeps_rest` / `_loads_post_r5_checkpoint_strictly`: both
  `load_tolerant` acceptance paths.
- Pre-existing tests updated for the new call signatures (`probe_loss`/`probe_metrics` now take `specs`;
  `PacketProbe`/`P(z)` -> `new_probe`/`run_probe`; `L["s"].forward` -> `L.s.forward` since blocks are `RelBlock`
  modules now, not `ModuleDict`s) without changing what they assert.

`pytest tests/unit -q`: 557 passed, 39 skipped (pre-existing skips: optional extras / Menagerie assets not
fetched / peer-only, unrelated to this row), 0 failed, exit 0.

## lead_questions

1. `base_cmd`'s old restricted read (only `READS["base"]` knot tokens, via a per-query key mask in the legacy
   `PacketProbe`) is not reproduced by the generic `ReadoutProbe` (one key mask per forward call, not per query);
   `base_cmd` now reads the whole packet like every other query. No existing test asserted the restricted read, so
   this did not need a `relations/ops.py` change to fix under this row's rules, but the lead may want a per-query
   key-mask capability added to `ReadoutProbe`/`FactorSite` later if `base_cmd`'s locomotion-only reasoning is a
   real requirement (that would be a `relations/base.py`/`ops.py` interface change, out of this row's scope).
   Not blocking merge.
   2. Added `readout_metrics` to `nets/probes.py` (foundation file, not in R5's `owns` list) as the metrics
   counterpart of the existing `readout_loss` -- R1's brief already assumes it exists. A worktree sweep found no
   other unit currently touching that file. Flagging for visibility, not blocking merge (pure addition, tested via
   `probe_metrics`/`readout_metrics` call sites in this row's own tests).
