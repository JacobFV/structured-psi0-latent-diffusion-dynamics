# track rel-geo: wire geometry/interaction factors so they actually reach attention

Worktree `~/work/rrp-wt/rel-geo`, branch `track/rel-geo`, from `origin/main` (started at `566fccc6`, R11: record
final merge evidence in track notes). Follow-up unit (D-144 addendum), not one of the original 21 fanout rows:
closes the gap R13 and R17 both flagged in their own "for the lead" / "lead question" sections
(`research/tracks/rel-r13.md`, `rel-r17.md`) and adds R18's `task.next_contact` / `time.same_track` preset
(`rel-r18.md`). Read `docs/relations.md` §2-5, §10 first.

Never touched `relations/base.py` / `relations/ops.py` (imported read-only). No operator change was needed for
items 1 and 3; item 2's PROBE/estimate half genuinely needs one and was left as a lead question rather than made
unilaterally (see below and `research/decisions.md` D-144 addendum).

## (1) `nets/flow.py` `CTX_CARRIES`

`CTX_CARRIES = ("edges:arm-rel-v1", "hidden")` -> `(..., "cam_uvd", "pos3d", "orient", "normal")` (R13's exact
request). `ACT_CARRIES` left untouched (not asked for; nothing in R13/R17/R18 needs `geo.*` at `act>ctx`/`act>act`).

Evidence (`tests/unit/test_relations_geo_wiring.py`, new):
- `test_ctx_carries_names_the_r12_geometry_fields`, `test_geo_factors_now_apply_at_ctx_ctx_with_the_real_carries_tuple`
  -- all five `geo.*` names now resolve into a `FactorSite("ctx>ctx", ..., carries=CTX_CARRIES)`.
- `test_geo_depth3d_reaches_the_real_flowpolicy_ctx_ctx_site_without_any_runtime_patch` -- a real `FlowPolicy`
  (`factors=["preset:arm", "geo.depth3d"]`) picks it up with NO per-process attribute patch, unlike R13's own peer
  smoke (`flow_mod.CTX_CARRIES = flow_mod.CTX_CARRIES + ("cam_uvd",)`, done only in that smoke's own process).
- `test_geo_depth3d_changes_ctx_attention_logits_in_the_real_context_encoder` -- runs the real `ContextEncoder`,
  then compares layer-0 `ctx>ctx` attention WEIGHTS (softmax'd logits, `MHA(..., need_weights=True)`) at a fixed
  hidden state and a fixed (manually supplied, non-degenerate) `cam_uvd` probe estimate, before vs. after nudging
  `geo.depth3d`'s zero-init PaPE coefficients off zero (docs 3.2: an `aug` factor is a no-op at step 0). The
  estimate is supplied directly into `rc.estimates` rather than left to `FieldReadouts`' own freshly-constructed
  (zero-init) head, which predicts an identical `mu=0` for every token regardless of `h` at model construction --
  true of any freshly-registered probe field, not specific to this factor, and it would make BOTH q/k PaPE features
  collapse to the same all-zero point (a literally zero dot product no amount of nudging PaPE's own coefficients
  can move) -- an initialization artifact orthogonal to the CTX_CARRIES wiring under test.
- `test_default_arm_preset_factor_site_unchanged_by_the_wider_carries_tuple` -- `FactorSite("ctx>ctx", preset:arm,
  ...)` resolves to the IDENTICAL 17-name list under the new vs. the old (two-entry) `CTX_CARRIES` (`edge.*` keys on
  `"edges:arm-rel-v1"` only; `msg.incidence` is `form="message"`, filtered out before the carries check regardless
  of what carries names).
- `test_golden_suite_is_byte_identical_after_the_ctx_carries_change` -- runs `tests/unit/test_golden.py` as a
  subprocess (no `RRP_GOLDEN_RECORD`); asserts exit 0. Direct evidence for "golden.json unchanged," not just the
  structural argument above.

## (2) `ix.force_flow`'s "edges:support-v1" `EdgeSet`

New `src/rrp/policies/nets/batch.py` function `support_edges(inputs, batch, graphs, views) -> EdgeSet`
(`SUPPORT_REL_VOCAB = ("support",)`, new in `catalog.py`, also registered as `VOCABS["support-v1"]`). Builds the
DIRECT support graph (channel `"support"`) over `ctx` SCENE-bank token positions, from `ix.support`'s privileged
label side (`relgen.support.support_matrix`'s output) -- the GT/label half of R17's own "for the lead" note.

**Layering course-correction** (caught by `tests/unit/test_layering.py::test_import_layering`, which statically
scans every import including lazy/function-local ones): my first draft had `support_edges` import
`rrp.harness.data.relgen.support` directly (to call `support_matrix(view)` itself). `nets/batch.py` is `policies`
layer 4; `harness.data.relgen` is layer 5 -- an upward import, forbidden even lazily. Fixed by inverting the
dependency: `support_edges` takes the ALREADY-COMPUTED `graphs[i]` (one `support_matrix(view)` dict per batch item)
as a plain argument, computed by whichever harness-layer caller has both `nets.batch` (layer 4, a downward/allowed
import from harness) and `relgen.support` in scope; `nets/batch.py` itself imports neither `relgen` nor anything
harness-layer. `views[i]` (a `StateView`, layer 3, downward/allowed) is kept only for `token_entity("scene", slot)`
(unit R7) -- the scene-bank-slot -> privileged-entity-id map `support_matrix`'s graph is keyed by, resolved here
directly off a live `Batch` instead of an offline `TokenIndex`.

The `flow` operator itself (`ClosureOp`, unedited) computes the transitive closure of whichever channel it's
pointed at (`params.edge="support"`) at attention time; this function only needs to supply the one direct channel,
never the closure.

**What is NOT built**: the PROBE-sourced half of this same EdgeSet (from `ix.support`'s own learned bilinear pair
estimate, not the privileged label). `relations/ops.py`'s `BilinearOp.features` returns q/k KERNEL features for the
`aug` form, never the raw `[B,Q,K]` pair score, and `FieldReadouts` explicitly skips `op="bilinear"` -- exposing
that score as an `EdgeSet` needs a new read hook in `relations/ops.py`, an operator-level change this unit's rules
forbid making unilaterally. Flagged in `lead_questions` below (same shape as R17's original note, whose label/GT
half this unit resolves; the probe half remains open for whoever wires `ix.force_flow` into a real net).

Exposed as a pure function only (matching R18's `candidate_interaction_edges` precedent exactly) -- NOT wired into
`relation_token_sets` / any net's `RelCtx.edges`.

Evidence (`tests/unit/test_relations_geo_wiring.py`):
- `test_support_edges_direct_chain_matches_support_matrix` -- hand-built 4-entity chain (`_View`, a `StateView`
  stand-in with `token_entity` IMPLEMENTED, unlike `test_relations_support.py`'s `_StackView` stub) against a
  synthetic `Batch`: exactly the two direct edges land at the right `[q,k]` cells, nothing else (no closure, no
  cross-talk, side object gets nothing).
- `test_support_edges_channel_matches_ix_force_flow_edge_param`, `test_support_edges_prov_is_privileged`.
- `test_support_edges_no_view_or_too_few_entities_is_all_zero_never_errors`,
  `test_support_edges_unresolved_token_entity_ids_are_skipped_not_errors` -- degenerate-input coverage matching
  `candidate_interaction_edges`'s own "never crashes" precedent.
- `test_support_edges_on_a_real_arm_fixture_shape_and_range_smoke` -- real `Session.state_view()` (R7) + real
  featurizer/collate path: right shape, values in {0, 1}, no crash (the pick_place fixture has no real
  object-on-object stacking, so this is a shape/range smoke, not a correctness claim -- correctness against
  `support_matrix` is `test_support_edges_direct_chain_matches_support_matrix`'s job, and R17's own test suite
  already covers `support_matrix`/`support_closure` math in full).

## (3) Presets

`catalog.py`, additive only:
- `register_preset("ix", ...)` extended with `"ix.support"`, `"ix.force_flow"` (previously R16's three `ix.*`
  only). Forward-references the R17 `FactorDef`s registered later in the same file, same precedent R13/R18 already
  set for forward-referencing not-yet-registered names.
- New `register_preset("task", ["task.next_contact", "time.same_track"])` (R18's two entries; R18 itself never
  registered a preset for them).
- The third part of this item -- decide a `route.assembly_reads` preset for R5's Ψ₀ usage -- needed no action: a
  mid-unit `git fetch` showed R5 (`db8b603a`) had already landed on `main` with `register_preset("s0-psi0",
  [{"name": "route.assembly_reads", "params": {"reads": _G.reads_table().tolist()}}])`, `_G.reads_table()` a new
  `bodies.g1_simple` function -- the real `[M,M]` table, computed at the BODY layer specifically to sidestep the
  `catalog.py` -> `psi0.nets` circular import this unit had independently identified as blocking computing it from
  `catalog.py` itself. Recorded (not re-implemented) so the name/table are not re-litigated.

Evidence: `test_ix_preset_includes_support_and_force_flow`, `test_task_preset_is_next_contact_and_same_track`,
`test_route_assembly_reads_preset_already_resolved_by_r5`.

## lead decisions recorded

Checked `research/decisions.md` and every `research/tracks/rel-r*.md` for an existing D-144 addendum before adding
one. A `git fetch` mid-unit turned up `e622beda` ("D-144 addendum: readout-probe swap acceptance, gone-from-src
definition, merge lock (R8 lead decisions)") already on `main` with the exact (a)/(b)/(c) text this unit's own brief
carried -- so that block was NOT re-added (would have been a byte-identical duplicate); this unit's own scope
decisions (above) are recorded under a second, separate `## D-144 addendum` heading instead.

## commands run (host only: git / editing / unit suite; no training/simulation, no peer dispatch -- nothing in this
## unit's acceptance needs one: no new op/field, no new training run)

```
export PYTHONPATH=$PWD/src:$PWD
CUDA_VISIBLE_DEVICES="" ~/work/relational-robot-policy/.venv/bin/python -m pytest tests/unit/test_relations_geo_wiring.py tests/unit/test_layering.py -q
CUDA_VISIBLE_DEVICES="" ~/work/relational-robot-policy/.venv/bin/python -m pytest tests/unit -q -x -p no:cacheprovider
```
`test_relations_geo_wiring.py` + `test_layering.py`: 20 passed. Full suite: see the merge section below / this
run's transcript for the exact count and exit code recorded at merge time (re-run after every rebase per the fanout
rules).

## files touched (owned by this unit)

- `src/rrp/policies/nets/flow.py` (`CTX_CARRIES` only)
- `src/rrp/policies/nets/batch.py` (`support_edges`, `SUPPORT_REL_VOCAB` import)
- `src/rrp/policies/relations/catalog.py` (`SUPPORT_REL_VOCAB`, `VOCABS["support-v1"]`, two preset edits)
- `tests/unit/test_relations_geo_wiring.py` (new)
- `research/decisions.md`, `research/tracks/rel-geo.md` (this file)

## lead_questions

1. `ix.force_flow`'s PROBE-sourced `edges:support-v1` (from `ix.support`'s own learned bilinear pair score, not the
   privileged label) needs a new read hook in `relations/ops.py` -- `BilinearOp` has no existing way to expose its
   raw `[B,Q,K]` pair score, and `FieldReadouts` explicitly skips `op="bilinear"`. Out of this unit's rules
   (`relations/ops.py` never touched). Suggest a small hook analogous to `FieldReadouts` but for bilinear pair
   scores, or explicit lead sign-off on the shape of one, scoped to whichever unit next trains `ix.force_flow`
   end-to-end.
2. None -- R5 landed with its own `s0-psi0` preset + real `params.reads` table before this unit merged (see
   "(3) Presets" above); no open question remains on this item.

## resume steps (if interrupted before merge)

1. `cd ~/work/rrp-wt/rel-geo && export PYTHONPATH=$PWD/src:$PWD`
2. `CUDA_VISIBLE_DEVICES="" ~/work/relational-robot-policy/.venv/bin/python -m pytest tests/unit -q` -- must exit 0.
3. `git fetch origin && git rebase origin/main`; rerun the unit suite if the rebase brought new commits (especially
   watch for a sibling unit also touching `catalog.py`'s preset section or `nets/batch.py`'s tail, or a sibling
   `research/decisions.md` D-144 addendum landing first -- merge, don't duplicate).
4. `flock ~/work/rrp-data/main-merge.lock bash -c 'git fetch origin && git rebase origin/main && CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD/src:$PWD ~/work/relational-robot-policy/.venv/bin/python -m pytest tests/unit -q -x -p no:cacheprovider && git push origin HEAD:main'`
