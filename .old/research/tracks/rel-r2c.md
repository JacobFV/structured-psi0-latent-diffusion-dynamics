# track: rel-r2c (relation-factor fanout, R2 deferred-scope follow-up)

Branch `track/rel-r2c`, worktree `~/work/rrp-wt/rel-r2c`, from `origin/main` @ `566fccc6` (R11 merged; unit R2 itself,
`86aba25f`, was already an ancestor -- R2 had already merged before this row started). Spec: `docs/relations.md`
sections 2-5 and 10 (row R2, brief in "briefs"), `research/tracks/rel-r2.md` ("open questions for the lead"),
`research/decisions.md` D-144 addendum (2026-09-30, decisions (a)/(b)/(c)), `AGENTS.md`. Host only (git / editing /
unit suite; CUDA hidden; no training beyond CPU unit fixtures).

## scope

R2's deferred scope per the addendum: no live code path reads `semantic_weight` / `probe_lv_min` / `binding_cf` /
`binding_contrast` / `slot_handles` / `packet_semantic_weight` / `aux_weight`, through
`core/runconfig.py`, `core/provenance.py`, `nets/semantic_latent.py`, `harness/train/latent_train.py`,
`cli/latent.py`, `harness/pipelines/arm.py`, `policies/bundles.py`; exactly one legacy-key mapping table in the
loader; codemod `configs/`/`dags/`. `LatentConfig.version()` must stay identical for every config under `configs/`
(frozen-versions test kept green); `tests/data/golden.json` unchanged (not touched).

## what changed

1. **One named legacy mapping table** (`nets/semantic_latent.py`): promoted the inline `legacy_keys` tuple in
   `LatentConfig.__init__` to a module-level constant `LATENT_LEGACY_KEYS`, documented as the loader-only remap
   (same status/pattern as `harness.data.collect.LEGACY_PICKLE_MODULES`, per decision (b)). No behaviour change
   (`test_new_and_legacy_construction_agree`, `test_frozen_set_covers_every_latent_config` unchanged, green).
2. **`binding_cf` -> `cf_mix`, everywhere it was still a live parameter/output-key name**: `harness/train/
   latent_train.py::fit_probes_on_frozen`'s own parameter (R2 could not do this: it did not own
   `cli/latent.py` / `harness/pipelines/arm.py`, the only two external call sites). rel-r2c owns all three files, so:
   - `fit_probes_on_frozen(binding_cf=...)` -> `fit_probes_on_frozen(cf_mix=...)`; its result dict key
     `binding_cf` -> `cf_mix` (confirmed by grep: no other reader of that result key in the repo).
   - `cli/latent.py`: `--binding-cf` -> `--cf-mix`, same `dest="cf_mix"`; `--binding-cf` kept as a deprecated
     alias (same dest) for any invocation still spelled the old way -- the CLI-flag analogue of an on-disk legacy
     remap, not a construction surface.
   - `harness/pipelines/arm.py`'s `probes` stage: reads `options.get("cf_mix", options.get("binding_cf", 0.0))`
     -- prefers the new options key, falls back to the old one for a not-yet-updated `options` dict (no dag
     currently sets either key for the arm `probes` stage; confirmed by grep of `dags/*.yaml`).
   - `tests/unit/test_relations_r2_latent.py`: updated its acceptance-criteria docstring and added
     `test_no_binding_cf_as_live_parameter_name` (asserts `cf_mix` in, `binding_cf` not in,
     `inspect.signature(fit_probes_on_frozen).parameters`).
3. **Documentation-only strengthening** of the three exceptions R2's open question 1 already identified
   (`core/provenance.py::TRAINING_FLAG_KEYS`, `core/runconfig.py::FLAG_NAMES`/`FLAG_SPEC`/`LEGACY_FLAG_DEFAULTS`):
   both comments now cite the D-144 addendum and the concrete, empirically-confirmed reason they cannot be retired
   within this unit's owned files (see "what I deliberately did not do").

## what I deliberately did NOT do (and why, concretely -- not just by inspection)

**`configs/ladder/**` codemod (6 rep-\*.json files with a flat `semantic_weight`/`probe_lv_min` "latent" block:
`rep-latent_sem_b1fix_anchor.json`, `armsemfix/rep-latent_semfix_b1fix_anchor.json`,
`armnosem/rep-latent_nosem_b1fix_anchor.json`, and the `armseed2/{sfjf2,sejf2,nsjf2}` seed-2 equivalents).**
I wrote and ran the same codemod R2 already applied to `configs/latent/**` (flat keys -> `factors:`/`cf_mix`, using
`_ARM_PROBE_QUERIES` / the same minimal-diff shape as the existing codemodded files: no `params.lv_min` when it is
the -8.0 default, no `cf_mix` key when 0), verified byte-identical `LatentConfig(**d["latent"]).version()` hashes
against the frozen table in `test_relations_r2_latent.py` for all 6 files (they matched exactly), then ran the full
suite: `tests/unit/test_dag.py::test_arm_dag_reproduces_legacy_configs` FAILED, 6 of 6 parametrizations. That test
(general infra, not owned by this row) asserts `dags/arm_lineage.yaml`'s rendered native config is dict-equal to
these exact 6 on-disk files. `dags/arm_lineage.yaml` (and `dags/templates/dual_lineage.yaml`, same pattern) render
a `flags: {..., probe_lv_min: '{lv}', ...}` block together with a flat `latent: {name: ..., semantic_weight: '{sw}'}`
block for these lineages; `RunConfig.to_native()` unconditionally injects `flags.probe_lv_min` at
`latent.probe_lv_min` for every FLAG_SPEC-applicable (family, stage) (`core/runconfig.py`, `_set_path` in the
`to_native` loop) -- there is no way to make the on-disk config `factors:`-shaped while the dag still emits the
flag-driven flat key, without ALSO removing `probe_lv_min` from `FLAG_SPEC[("arm", "train_rep")]` /
`FLAG_SPEC[("dual", "train_rep")]`. I reverted the config-file codemod (`git checkout --`) rather than touch
`FLAG_SPEC` (see next paragraph) or `dags/**` (a file class this addendum puts in scope, but only together with a
`FLAG_SPEC` change I cannot make safely -- see below); the suite is back to the pre-existing 783 passed / 39 skipped
with no diff under `configs/ladder/`.

**`FLAG_NAMES` / `FLAG_SPEC` / `LEGACY_FLAG_DEFAULTS` (`core/runconfig.py`): `probe_lv_min` stays a live,
arm+legged-shared Flag.** Removing it from `FLAG_SPEC[("arm"|"dual", "train_rep")]` would (1) break
`test_runconfig.py::test_variant_must_match_recipe` (general infra; constructs a RunConfig with
`flags.probe_lv_min` set for `_check_variant`'s old-style branch -- confirmed by reading the test, not just
inference), (2) break legged's use of the identical `Flags` schema (fanout unit R4, not merged: legged's own
`train_rep`/`refit` FLAG_SPEC entries also key off `probe_lv_min`), and (3), newly confirmed above, break
`test_dag.py::test_arm_dag_reproduces_legacy_configs`. All three are outside this row's owned files. Documented at
`FLAG_NAMES`'s definition site.

**`packet_semantic_weight` (Stage-B / BC-only top-level knob, `harness/train/latent_train.py::
{train_latent_flow,sft_latent_flow}`, `core/provenance.py`, `core/runconfig.py::_check_variant`).** Grepped every
read site in the repo: besides this row's 3 owned files, it is read live by `harness/train/joint_adapt.py` (stage
`adapt`, a permanent `LEGACY_ONLY_STAGE` -- by design never migrated to `factors:`, the same status as an
old pickle format read forever) and `harness/train/legged_latent_train.py` (fanout unit R4, not merged), both of
which read the SAME config key for their own configs. Renaming it in only this row's files would fragment one
config-key meaning into two spellings across files that must agree, with no test able to prove the split is
equivalent (no golden/fixture depends on the literal string here, so there is nothing to freeze against, unlike the
`configs/ladder/**` attempt above). Exactly R2's own open question 2; still blocked on the same two files, still
not this row's to touch.

## commands run (host, CPU only)

```
export PYTHONPATH=$PWD/src:$PWD
CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest tests/unit -q -p no:cacheprovider   # 783 passed, 39 skipped
                                                                                     # (identical to pre-existing;
                                                                                     # baseline re-checked before
                                                                                     # and after every edit)
CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest tests/unit/test_relations_r2_latent.py \
  tests/unit/test_runconfig.py tests/unit/test_provenance.py -q -p no:cacheprovider  # 43 passed
```
The `configs/ladder/**` codemod attempt (script, reverted) and its `LatentConfig(...).version()` hash verification
against the frozen table are reproducible from this track note's description; not left as a script file (no shims
/ scratch files beyond those listed, per the fanout rules).

## open questions for the lead

1. **CLOSED by unit `sweep-flags`** (`research/tracks/sweep-flags.md`, D-144 addendum 2026-09-30 "Applied by
   sweep-flags"): the `configs/ladder/**` + `dags/arm_lineage.yaml` / `dags/templates/dual_lineage.yaml` codemod +
   `core/runconfig.py`'s `FLAG_SPEC[("arm"|"dual", "train_rep")]` change landed together, once R4 had merged.
   `packet_semantic_weight` full retirement is still NOT closed (re-confirmed blocked on `harness/train/
   latent_train.py`, out of sweep-flags' scope too, plus 9+ legged dag files and 30+ legged flow configs) --
   sweep-flags' own open question 1 restates this as the next follow-up, now scoped precisely.
2. No new D-144 addendum items were needed beyond (a)/(b)/(c) already given; nothing added under D-144 that
   wasn't already the lead's own text.
