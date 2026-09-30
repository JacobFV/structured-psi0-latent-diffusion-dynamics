# track: rel-r6 (relation-factor fanout, unit R6: pointer nets)

Branch `track/rel-r6`, worktree `~/work/rrp-wt/rel-r6`, from `origin/main` (D-144 foundation + the pointer track,
D-142, already merged). Spec: `docs/relations.md` sections 2-5 and 10 (row R6 + its brief), `research/decisions.md`
D-144 addendum, `research/tracks/cworld.md` "pointer policy". Host only (git / editing / unit suite; CUDA hidden via
`CUDA_VISIBLE_DEVICES=""`); no peer smoke run (the row's acceptance is pure unit tests on fixtures, no
training/simulation).

Owned files (per the row): `src/rrp/policies/pointer.py`, `src/rrp/harness/train/pointer.py`. Shared file touched
only in its own pre-created section: `src/rrp/policies/relations/catalog.py` (new comment block right after
`probes:legged-v1`, `probe.pointer.*` + `probes:pointer-v1`). `relations/base.py` / `relations/ops.py` and
`nets/attention.py` were never edited.

## scope

- **`Block` -> `RelBlock` (empty preset).** The pointer body has M = 1 (the single "tool" assembly), so there is
  no cross-assembly routing to restrict and no `route.*` factor is needed; its `RelBlock`s carry the registered
  empty preset `none` (declared once, `POINTER_FACTORS_PRESET`). The pre-migration `Block` class was a bespoke
  **2-stage** pre-LN block (module keys `n1, a, n2, m`; EITHER cross-attention (`cross=True`) OR self-attention
  (`cross=False`) + MLP), unlike legged's / psi0's pre-migration blocks, which were already 3-stage
  (cross+self+MLP) and so mapped onto `RelBlock` (module keys `n1, x, n2, s, n3, m`) as a trivial rename. Pointer's
  2-stage design does not decompose into `RelBlock`'s fixed 3 stages by renaming alone, so this unit adds one
  device for exact equivalence: `cross_relblock` / `self_relblock` build a `RelBlock` whose UNUSED stage (`.s` for
  a former `cross=True` usage, `.x` for a former `cross=False` usage) has its **output projection zero-initialized**
  (`_zero_mha_out`) -- the same zero-bias-equivalence idiom `docs/relations.md` 3.2 already uses for factor
  gates/aug, applied here to an entire attention stage. A zero output projection makes that stage's contribution to
  the residual sum exactly 0 regardless of its input, so:
  - **fresh-init parity**: `RelBlock`'s 3-stage forward, with the unused stage contributing 0, is *bit-identical*
    to the deleted 2-stage `Block`'s forward using the SAME submodules
    (`test_cross_relblock_self_stage_is_a_noop_matching_old_cross_only_block`,
    `test_self_relblock_cross_stage_is_a_noop_matching_old_self_only_block`, `atol=1e-6`).
  - **old-checkpoint-load parity**: nothing is loaded into the unused stage (it keeps its zero-init), so an old
    checkpoint loaded through the key map reproduces the old forward's output exactly, not just approximately
    (`test_pointer_realizer_old_checkpoint_matches_old_forward`, `atol=1e-5`, the unit's "fixed-input equivalence"
    acceptance item; `test_remap_block_state_prefixes_are_disjoint_and_correct` unit-tests the key map itself).
  All four net classes (`PointerEncoder`, `PointerRealizer`, `PointerFlow`, `PointerBC`) and the shared `UICtx`
  moved onto `cross_relblock` / `self_relblock`; `PointerFlow` / `PointerBC`'s former two-list-zip
  (`X(...)` then separately `S(...)`, each with its own MLP) becomes `X(...)` then `S(...)` with each being one
  `RelBlock` in the appropriate mode -- same parameter count and same math as before (each old `Block` already had
  its own single MLP; nothing is fused or dropped).
  `_BLOCK_PATHS` names every `blocks`-style path each class holds (`"blocks."`, `"ctx.blocks."`, `"self_blocks."`)
  and its mode; `load_pointer_module` detects a pre-R6 checkpoint per path (the old block's `.a.` attention key,
  which the new layout never has) and remaps only that path, leaving a post-R6 checkpoint untouched.
- **`PointerProbe` -> `ReadoutProbe` (`probes:pointer-v1`).** The pre-migration `PointerProbe` was NOT the "opaque
  random handle codes, no scene features" design every other family's probe already follows (`nets/probes.py`'s
  own docstring): it cross-attended the REAL widget descriptor content (label characters, role, bound entity,
  geometry) as keys, a genuine architecture difference from `ReadoutProbe`, not just a naming difference (unlike
  R4/R5's probes, which already matched the generic shape closely). `catalog.py` registers 3 `probe.pointer.*`
  queries with the query names kept IDENTICAL to the old output dict (`slot`, `rel`, `phase`) so nothing downstream
  needs renaming. M = 1, so every query addresses `knot×asm` (the psi0/legged precedent): `slot` is now a fixed
  NW=80-way classification of the target widget by SLOT INDEX (CW slot identity is stable within an episode,
  `research/tracks/cworld.md`) rather than by content -- the probe can no longer partly cheat off widget
  text/role, a strictly harder and more honest test of what `z` itself encodes, and exactly the simplification
  that brings pointer into line with the shared design. `rel` (out=4, Gaussian mu(2)+logvar(2)) and `phase`
  (out=6, CE) map directly. `new_pointer_probe` / `run_pointer_probe` / `pointer_probe_specs` (the former CLI
  `--lv-min` is now `params.lv_min` on the `rel` query's spec, mirroring `policies.psi0.nets._with_lv_min`) are the
  new pointer.py surface; `harness/train/pointer.py`'s `probe_loss` / `probe_metrics` now delegate almost entirely
  to the foundation's `readout_loss` / `readout_metrics`, keeping only the `-1 = missing` label-masking pointer
  needs (identical formula to the pre-R6 code: verified query by query against the deleted code's masking, and
  `slot_acc` / `phase_acc` are the exact same masked-argmax-equality computation, so those two numbers are
  unchanged; `rel_err` is kept in pixel units (not the generic MAE, which is in normalized units) for continuity
  with existing eval consumers). `--w-sem` stays a plain top-level float multiplying the packet loss against the
  main reconstruction/flow loss (same pattern psi0/legged use; not a factor-level `weight`, which would instead
  control relative weighting AMONG the probe's own 3 queries).
- **Checkpoint loading (probe).** `load_pointer_probe_state` drops every key of an old `PointerProbe` state
  (detected by module names `bag.` / `wf.` / `role.` / `rel.` / `phase.` / bare `kq`, none of which the new
  `ReadoutProbe` layout uses) and returns a fresh `ReadoutProbe` init -- the two architectures are genuinely
  incompatible (real content vs opaque codes), so, per addendum (a) (which names R6 explicitly for this), a refit
  is the correct fallback, the same one psi0 (R5) uses for the identical situation. A post-R6 checkpoint's `P.*`
  loads strictly (`test_old_pointer_probe_checkpoint_drops_and_refits_new_loads_strictly`).
- **`load_pointer_bundle`.** `E` / `R` / `S` / `BC` now load through `load_pointer_module` (the `RelBlock`
  checkpoint map); `P` through `load_pointer_probe_state`. Verified end to end (`_save` a fresh bundle, round-trip
  through `load_pointer_bundle`, bit-identical state on every module, `versions["factors"]` present) outside the
  committed test suite, ad hoc, on the host.
- **`versions["factors"]`.** `_save` now records `compat_hash(resolve(["preset:none"]))` into every pointer
  checkpoint's `versions` dict (the empty-preset provenance the row's brief asks for), matching how every other
  migrated family's factor hash enters its checkpoint (`docs/relations.md` 3.5).

## acceptance (docs/relations.md 10, row R6)

- "pointer unit tests green": `tests/unit/test_pointer.py` -- 12 passed (was 3 passed / 4 skipped;
  `computerworld`-marked tests still skip on host, unaffected by this unit), 4 skipped, exit 0.
- "PointerProbe checkpoints load": `test_old_pointer_probe_checkpoint_drops_and_refits_new_loads_strictly` (old ->
  refit, no crash; new -> bit-exact strict load) + the `load_pointer_bundle` round-trip above.
- "`probes:pointer-v1`": `test_probes_pointer_v1_preset_resolves_in_order`,
  `test_new_pointer_probe_output_shapes`.
- "fixed-input equivalence test (outputs within atol 1e-5 of the old code on a seeded fixture)":
  `test_pointer_realizer_old_checkpoint_matches_old_forward` (an old-format `PointerRealizer` checkpoint,
  key-mapped by `load_pointer_module`, reproduces the deleted `Block`-based forward's `(xy, btn, key)` output
  exactly, `atol=1e-5`, seed 7) plus the two block-mechanism no-op tests above (`atol=1e-6`).
- Full suite: `CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD/src:$PWD:$HOME/work/ext/cw-site
  ~/work/relational-robot-policy/.venv/bin/python -m pytest tests/unit -q -p no:cacheprovider` -> 827 passed, 25
  skipped, exit 0 (baseline on this worktree before the unit's changes: 818 passed, 25 skipped -- +9 new tests, 0
  regressions). `tests/data/golden.json` untouched (no `rel.pointer.*` entry existed and none was added; no golden
  needed re-recording, so addendum (a)'s re-record allowance was not exercised for this unit).

## deviations / notes for the lead

- `nets/probes.py` (a foundation file) was **not** edited: every address this unit needed (`knot×asm`) was already
  implemented there by R4/R5's work. No operator or address change was requested of `relations/base.py` /
  `relations/ops.py`.
- The probe's `slot` query is a behavior change from the pre-migration code (opaque slot-index classification
  instead of content-based classification), by design (see "scope" above) -- flagging per addendum (a)'s spirit
  even though no golden depended on it.
- `cmd_edit`'s per-episode widget-descriptor batch construction (`public_features` / `collate_public` / `half`)
  was dropped from `harness/train/pointer.py::cmd_edit`: the new probe no longer reads widget content, so those
  calls were dead code after the swap. Only the probe read sites changed; the causal-edit LOGIC (gradient steps on
  the `slot` logit toward a target widget, L2 anchor, closed-loop realization) is unchanged.
