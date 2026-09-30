# sweep-flags: close rel-r2c's open question 1 (probe_lv_min/semantic_weight off live paths for arm+dual+legged,
binding_cf fully deleted, packet_semantic_weight re-confirmed blocked)

Owner: sweep-flags agent (Sonnet). Worktree `~/work/rrp-wt/sweep-flags`, branch `track/sweep-flags`, from
`origin/main` `2bbbdd4b` (D-144 R20 merged; R4, R2c already ancestors). Scope: finish moving
`semantic_weight` / `probe_lv_min` / `binding_cf` / `packet_semantic_weight` / `aux_weight` off live code paths onto
`factors:`, per `research/tracks/rel-r2c.md`'s "open questions for the lead" item 1 and the D-144 addendum (decisions
(a)/(b)/(c)). Owned files: `harness/train/legged_latent_train.py`, `core/runconfig.py`, `harness/pipelines/arm.py`,
`cli/latent.py`, `harness/train/joint_adapt.py`, `dags/arm_lineage.yaml`, `dags/templates/dual_lineage.yaml`, plus
the `configs/ladder/**.json` (6 rep files) and test files this coordinated change touches. Host only (git / editing /
unit suite; `CUDA_VISIBLE_DEVICES=` throughout; no training, no simulation). Read `docs/relations.md`,
`research/tracks/rel-r2.md`, `research/tracks/rel-r2c.md`, `research/tracks/rel-r4.md` first. Never touched
`src/rrp/policies/relations/base.py` / `ops.py`; `tests/data/golden.json` unchanged.

## what changed

1. **`core/runconfig.py`: `probe_lv_min` retired from `FLAG_SPEC[("arm"|"dual", "train_rep")]`.** R2c left this in
   place because retiring it needed three things together it did not own: (1) legged's identical `Flags` schema
   (fanout unit R4 — merged since, but see item 2 below: legged's own dags still use it, for a DIFFERENT reason);
   (2) `test_runconfig.py::test_variant_must_match_recipe`; (3) `dags/arm_lineage.yaml` /
   `dags/templates/dual_lineage.yaml` + the 6 `configs/ladder/**.json` rep files locked together by
   `test_dag.py::test_arm_dag_reproduces_legacy_configs`. This row owns all of them:
   - `FLAG_SPEC[("arm", "train_rep")]` (and `dual`, which copies `arm`'s table) no longer maps `probe_lv_min`;
     `LEGACY_FLAG_DEFAULTS[("arm"|"dual", "probe_lv_min")]` removed (dead once FLAG_SPEC drops the key — every
     remaining `load_legacy` reader of that default is `("legged", "probe_lv_min")`, kept).
   - `_check_variant`: unchanged logic (it already preferred `params.latent.factors` when present, R2), but its
     old-style branch (`lat.get("semantic_weight")` / `rc.flags.probe_lv_min`) is now LIVE ONLY for `legged` — arm
     and dual dags always supply `factors:` now, and `rc.flags.probe_lv_min` is always `None` for those two families
     (documented in the function's own docstring).
   - `_legacy_variant` (the `load_legacy` variant classifier): dropped the
     `lat.get("probe_lv_min", -8.0) > -8.0 and lat.get("semantic_weight", 0) > 0` fallback that used to supplement
     the filename/name regex. Confirmed empirically (a script over every `configs/**/*.json` with a `"latent"`
     block, not just inspection) that NO file in the repo actually depends on it — the regexes alone already agree
     with it everywhere — so this is dead-code removal, not a behavior change.
   - Both retained the giant explanatory comment block (rewritten) documenting exactly why `probe_lv_min` STAYS in
     `FLAG_NAMES` / `Flags` / `FLAG_SPEC[("legged", "train_rep")]` / `LEGACY_FLAG_DEFAULTS[("legged", "probe_lv_min")]`
     forever: at least 9 `dags/legged_v2_*.yaml` / `dags/templates/legged_v2_*.yaml` / `legged_fixrep.yaml` /
     `smoke_legged.yaml` files (confirmed by grep, none owned by this row) render fresh, non-legacy legged
     `train_rep` RunConfigs with a literal `flags: {probe_lv_min: ..., ...}` + flat `params.latent.semantic_weight`
     block. Retiring it for legged too needs those dag files in a unit's owned-file list, which this row's brief
     does not include (only `dags/arm_lineage.yaml` / `dags/templates/dual_lineage.yaml`).
2. **`dags/arm_lineage.yaml` / `dags/templates/dual_lineage.yaml`: `stageA`'s `latent:` block is `factors:`-shaped.**
   `base.flags` (arm) / the node's own `config.flags` (dual) no longer set `probe_lv_min`. The base template sets
   every `probe.arm.<query>` weight to `'{sw}'` (0.0 nosem / 1.0 sem+semfix, `_ARM_PROBE_QUERIES` order, no `params`
   — matches the LatentConfig default `lv_min=-8.0`, omitted exactly like R2's own codemod of
   `configs/latent/*.json`); a new `per: {variant: {semfix: {params: {latent: {factors: [...]}}}}}` node override
   supplies the literal `weight: 1.0, params: {lv_min: -4.0}` shape for the one variant that differs. Verified
   against `test_dag.py::test_arm_dag_reproduces_legacy_configs` (all 6 arm/dual variant×seed points) and
   `test_dual_v3.py::test_dual_pipeline_guards_and_template`.
3. **`configs/ladder/**.json` (6 rep files) codemodded** — the exact transformation R2c wrote, verified and then
   reverted (`git checkout --`) when it hit the FLAG_SPEC/dag blocker: flat `{semantic_weight, [probe_lv_min]}` ->
   `factors: [{name: probe.arm.<q>, weight: <w>[, params: {lv_min: <lv>}]}, ...]` (`_ARM_PROBE_QUERIES` order, params
   omitted at the -8.0 default). `LatentConfig(**d["latent"]).version()` unchanged for all 6
   (`tests/unit/test_relations_r2_latent.py`'s frozen table, already carrying these exact 6 paths/hashes from R2c's
   original prepared-but-reverted attempt, passes unmodified).
4. **`harness/train/legged_latent_train.py`: the ONE conversion point for legged's own `latent` config dict.** New
   `_legged_probe_factors(lc)` (mirrors `nets/semantic_latent.py`'s `LATENT_LEGACY_KEYS`/`_probe_factors` shape, for
   `probe.legged.*` instead of `probe.arm.*`) and `_legged_probe_weight_lv(factors)` (mirrors `LatentConfig.weight`
   / `.lv_min`). `train_rep` calls both once and uses the derived `(w_sem, probe_lv_min)` everywhere `rep_step` is
   called (was: `lc["semantic_weight"]` / `lc.get("probe_lv_min", -8.0)` read directly at the call site) and passes
   `factors=` into `legged_probe(...)` (P's own spec set, previously never populated — inert for `probe_loss`'s
   custom uniform-weight computation, since that never read P's internal specs either, but keeps the checkpoint's
   provenance / factor list consistent with what actually trained it). `train_flow` hoists the SAME
   `_legged_probe_factors`/`_legged_probe_weight_lv` call (on `rcfg["latent"]`, the frozen rep's own saved config)
   out of the training loop, replacing a `rcfg["latent"].get("probe_lv_min", -8.0)` re-lookup on every step.
   On-disk config format is UNCHANGED (`configs/legged_latent/**`, `configs/legged_fixsem/**`, `configs/t1_diag/**`
   keep their flat `semantic_weight`/`probe_lv_min` keys, and every `dags/legged_v2_*` dag keeps rendering them, per
   item 1's FLAG_SPEC note) — verified equivalence by re-deriving `(weight, lv_min)` for every real on-disk legged
   rep config and asserting it matches `lc.get("semantic_weight", 1.0)` / `lc.get("probe_lv_min", -8.0)` exactly
   (all of `configs/legged_latent/rep_{sem,nosem}_v1.json` and `configs/legged_fixsem/rep_*.json`).
   `packet_semantic_weight` (flow stage) is UNCHANGED — see "what I deliberately did not do".
5. **`harness/pipelines/arm.py`'s `probes` stage: the `options.get("cf_mix", options.get("binding_cf", 0.0))`
   fallback is gone** — `cf_mix=float(o.get("cf_mix", 0.0))`. Confirmed by grep (repeated from R2c, re-verified):
   no dag or config sets either options key for this stage.
6. **`cli/latent.py`'s `fit-probes` command: the deprecated `--binding-cf` alias is gone** — `--cf-mix` is the only
   spelling now. R2c's own decision (b) had explicitly kept this as "the CLI-flag analogue of the loader's on-disk
   legacy-key remap"; this row's brief explicitly asks for its deletion, which supersedes that specific sub-decision
   (the `LATENT_LEGACY_KEYS` table itself, the actual "one legacy remap table," is untouched).
7. **Tests updated for the new reality** (all in owned/allowed files):
   `tests/unit/test_runconfig.py::test_legacy_classification_and_flags` (`rep.flags.probe_lv_min is None` for an
   arm `train_rep` legacy config now, not `-8.0`/absent-flagged), `::test_variant_must_match_recipe` (arm
   `train_rep` variant-checking is `factors:`-only now); `tests/unit/test_dual_v3.py::
   test_dual_pipeline_guards_and_template` (`LatentConfig(**...).lv_min` instead of a flat `native["latent"]
   ["probe_lv_min"]` key that no longer exists); `tests/unit/test_relations_r2_latent.py` — `_rc`'s default flags/
   params updated to the new arm reality, new `_rc_legged` + `test_check_variant_old_style_is_legged_only` (the old
   test asserted arm behavior that is now legged-only), `test_no_binding_cf_as_live_parameter_name` strengthened to
   check the CLI parser's actual `option_strings` and the `arm.py` `probes` stage's compiled bytecode (`co_names`/
   `co_consts`), not source text (so its own explanatory comments mentioning the retired name for history don't
   trip it).

## what I deliberately did NOT do (and why, concretely — not just by inspection)

**`packet_semantic_weight` full retirement (`harness/train/legged_latent_train.py::train_flow`,
`harness/train/joint_adapt.py`).** Re-confirmed by a repo-wide grep (not just re-asserting R2c's finding): this
exact key name is a literal, live `params`/checkpoint-config key in `harness/train/latent_train.py` (arm/dual
`train_flow`/`flow_ft`/`sft_latent_flow` — out of this row's owned files), in `harness/train/joint_adapt.py`'s own
read (from `st["config"]`, an ALREADY-TRAINED checkpoint's saved native config — written by `latent_train.py`, so
this read is inherently downstream of that file, matching decision (b)'s "reading an old pickle format forever"
class even discounting the joint_adapt.py-specific LEGACY_ONLY_STAGE point R2c already made), in at least 9
`dags/legged_v2_*.yaml` / `dags/templates/legged_v2_*.yaml` / `legged_fixrep.yaml` / `smoke_legged.yaml` files, and
in 30+ `configs/{ladder,latent,legged_latent,legged_fixsem,t1_diag}/flow_*.json` files on disk — none of the dag or
config files are in this row's owned-file list (only `dags/arm_lineage.yaml` / `dags/templates/dual_lineage.yaml`
are, and THEY don't render `train_flow`/`flow_ft` through a `packet_semantic_weight`-shaped `factors:` path in this
row's scope either — `packet_semantic_weight` isn't a `LatentConfig` field, it has no `probe.arm.*`/`probe.legged.*`
factor-spec analogue registered in `catalog.py`, which this row does not own). Renaming it in only the two files
this row does own (`legged_latent_train.py`, `joint_adapt.py`) while every other reader/writer keeps the old name
would fragment one config-key meaning across two spellings, with the checkpoint-config read in `joint_adapt.py`
actively BREAKING (silently falling through `.get(..., 0.0)` to "no semantic loss" instead of reading the value a
still-unrenamed `latent_train.py` wrote) rather than raising. Documented in-place at both read sites (updated,
verified comments, not the R2c-era ones). Exactly R2c's own open question 2/3, still blocked on the same two
out-of-scope file classes (`harness/train/latent_train.py` + the legged dags), now re-confirmed rather than merely
carried forward.

**`aux_weight` (SemanticReadout / BC-policy auxiliary loss weight, `harness/train/{vlm_train,behavior,sft}.py`,
`policies/nets/flow.py`).** This row's owned-file list does not include any of these four files, and none of the
five owned files reads `aux_weight` (confirmed by grep) — it is a distinct top-level BC/VLM policy config key from
this row's other four flags, unreachable from any file this row touches. Out of scope, not attempted.

**Legged's own `dags/legged_v2_*.yaml` / `dags/templates/legged_v2_*.yaml` / `legged_fixrep.yaml` /
`smoke_legged.yaml` codemod (would let `probe_lv_min` retire from `FLAG_SPEC[("legged", "train_rep")]` too).** Not
in this row's owned-file list (only `dags/arm_lineage.yaml` / `dags/templates/dual_lineage.yaml` are named). The
internal consumption of these flags inside `legged_latent_train.py` is migrated (item 4 above); the on-disk/dag
surface is not, by design/scope — a natural follow-up row, scoped explicitly to those files + `core/runconfig.py`'s
`FLAG_SPEC[("legged", ...)]`.

## commands run (host, CPU only)

```
export PYTHONPATH=$PWD/src:$PWD
CUDA_VISIBLE_DEVICES= ~/work/relational-robot-policy/.venv/bin/python -m pytest tests/unit -q -p no:cacheprovider
  # 884 passed, 41 skipped (full suite; re-run clean after every edit batch)
CUDA_VISIBLE_DEVICES= ~/work/relational-robot-policy/.venv/bin/python -m pytest \
  tests/unit/test_relations_r2_latent.py tests/unit/test_runconfig.py tests/unit/test_dag.py \
  tests/unit/test_d126_arm_dags.py tests/unit/test_dual_v3.py -q -p no:cacheprovider   # 68 passed
```
Plus one-off verification scripts (not left in the tree): a repo-wide scan proving no `configs/**/*.json`
classification depends on `_legacy_variant`'s dropped `probe_lv_min`/`semantic_weight` fallback; a script
re-deriving `(weight, lv_min)` via `_legged_probe_factors`/`_legged_probe_weight_lv` for every real on-disk legged
rep config and asserting exact agreement with the old flat-key reads; a `legged_probe(dz=8, factors=...)`
construction smoke check.

## open questions for the lead

1. `packet_semantic_weight` full retirement needs a SINGLE unit owning `harness/train/latent_train.py` (arm/dual)
   together with `harness/train/legged_latent_train.py`, `harness/train/joint_adapt.py`, and every
   `dags/legged_v2_*` / `configs/*/flow_*.json` file that renders the flat key — a much larger row than this one.
2. `probe_lv_min` retirement from `FLAG_SPEC[("legged", "train_rep")]` needs `dags/legged_v2_*.yaml` +
   `dags/templates/legged_v2_*.yaml` + `legged_fixrep.yaml` + `smoke_legged.yaml` (at least 9 files) in a unit's
   owned-file list alongside `core/runconfig.py`, mirroring exactly what this row did for arm/dual.

## follow-up (2026-09-30, unit `lm-legged`, worktree `~/work/rrp-wt/lm-legged`, branch `lm-legged`, from origin/main `6d4099a4`)
Closes both open questions above. Full write-up: `research/decisions.md`'s "D-144 addendum 2026-09-30: sweep-flags
follow-up" (this same date). Summary:
- Open question 2 (`probe_lv_min` off `FLAG_SPEC[("legged", "train_rep")]`): done. All 9 legged dag files + 27 of 29
  `configs/{legged_latent,legged_fixsem,t1_diag}/rep_*.json` now render `latent.factors`
  (`nets/semantic_latent.py::legacy_latent_factors`, shared with arm's own `_probe_factors`); the 2 exceptions
  (`rep_{nosem,sem}_v1.json`) stay flat on purpose (see the decisions.md entry -- they are also valid arm
  `LatentConfig(**...)` kwargs and are pinned by `test_relations_r2_latent.py`'s OWN frozen table under arm's
  reading). New frozen-hash regression test: `tests/unit/test_legged_frozen_latent.py`.
- Open question 1 (`packet_semantic_weight` full retirement): PARTIALLY closed, by design. The four readers
  (`latent_train.py` x2, `legged_latent_train.py`, `joint_adapt.py`) now share ONE conversion point
  (`nets/semantic_latent.py::packet_semantic_weight`/`packet_semantic_factor`), but no on-disk config changed --
  this row judged the FULL architecture (a registered `params.on`-carrying factor, docs/relations.md 10's R2 brief)
  to need `catalog.py`/`nets/flow.py`/`nets/probes.py` wiring, out of scope for a row whose brief named only the
  four training-code files + runconfig. `joint_adapt.py`'s read stays permanently flat by the D-144 addendum
  decision (b) "reading an old pickle format forever" rule (stage `adapt` is a permanent `LEGACY_ONLY_STAGE`) --
  this is not reopened, only reconfirmed. A real follow-up remains: registering `flow.packet_semantic` (or
  `params.on`) as a live catalog factor, IF a future unit owns `catalog.py` + `nets/flow.py` + `nets/probes.py`
  together with these four files and wants the on-disk key retired too (not required by anything currently -- no
  test or reader depends on the on-disk name changing).
