# track: rel-r17 (D-144 relation-factor fan-out, unit R17: support / force flow)

Worktree `~/work/rrp-wt/rel-r17`, branch `track/rel-r17`, from `origin/main` (started at `ad9d422b`, D-144 R21).
Brief: `docs/relations.md` section 10, row R17 / briefs (line 675-676): "Support (`a` supports `b` if in contact
and the contact normal is within 30° of gravity-up at `a`'s top), force-flow closure (upstream / downstream along
support), part `stack` (n objects, vary order), entries." Deps: R7 (StateView: MuJoCo, merged `7cd5af6b`/`b2d15d7c`),
R12 (arm/dual fields, merged `f38b689a`) -- both present on `origin/main` before this worktree was created; no
rebase conflicts expected. Host only: no CUDA (`CUDA_VISIBLE_DEVICES=""`), no training / simulation beyond
unit-test fixtures; this unit's tests never touch a simulator at all (see "host python" below).

Not part of this unit's scope (left for the lead / a later unit, `lead_questions`): wiring the actual
`edges:support-v1` graph that `ix.force_flow`'s `flow` operator reads at forward time from `ix.support`'s bilinear
pair estimate. `ops.py`'s `flow` operator (`ClosureOp`, already implemented by the foundation, unedited here) reads
a real `RelCtx.edges[site]` `EdgeSet` with that vocab; producing that `EdgeSet` from a pair-probe estimate is
net-side plumbing outside `catalog.py` / `relgen/support.py` (R17's owned files) and outside `base.py` / `ops.py`
(never touched, per the fanout rule). Declared it anyway (the registry entry is valid and resolves; see
`test_factors_registered_with_the_documented_shape`) since every sibling `ix.*` bilinear + derived-closure factor
(e.g. a future `ix.contact` + closure, if ever added) will face the identical open question, and it is not an
operator/interface change (no edit to `base.py` / `ops.py`, no new op, no new form) -- just an entry that will need
its carrier wired by whichever unit first trains against it. Flagged in `lead_questions` for visibility, not
`status: blocked`, since it did not block finishing this row's declared acceptance criteria.

## host python
This host's `~/work/ext/venvs/rrpcore312` (pure-CPU torch wheel) reproduces `tests/data/golden.json` /
`GOLDEN_TEACHER` / `psi1z` goldens WRONG (3 pre-existing failures: `test_deploy_eval.py
::test_defaults_byte_identical_to_pre_d126`, `test_golden.py::test_semantic_edit_loop_rows`,
`test_psi0.py::test_nets_match_psi1z_goldens`) -- floating-point kernel differences between that wheel and the
`torch==2.14.0+cu130` wheel the goldens were recorded against (even with `CUDA_VISIBLE_DEVICES=""`, a cu130 build's
CPU fallback kernels are not bit-identical to a cpu-only build's). `~/work/relational-robot-policy/.venv` (the main
repo's checked-in venv, same versions but the cu130 wheel) reproduces every golden exactly with CUDA hidden, so
this track used `CUDA_VISIBLE_DEVICES="" ~/work/relational-robot-policy/.venv/bin/python` for every
`pytest tests/unit` run below, not the shared `rrpcore312` venv. This is a pre-existing, environment-only issue
(unrelated to this unit's files); not investigated further since it does not touch anything `ix.support` /
`ix.force_flow` / `relgen/support.py` own.

## what changed
- `src/rrp/policies/relations/catalog.py` (my section only, `R17: support / force flow`): two `FactorDef` entries.
  - `ix.support`: `field="hidden"`, `op="bilinear"`, `form="aug"`, `sources=("probe", "gt")`, `label="support_pairs"`,
    `gen=("stack",)`, `readout=ReadoutDef("support", "pair", 1, "bce", label="support_pairs", reads="tokens")`,
    `params={"rank": 8}` -- the pairwise learned kernel over token hiddens (docs 3.2 `bilinear`), its `pair` readout
    IS the supervised probe (docs 4: "the bias IS the probe").
  - `ix.force_flow`: `field="edges:support-v1"`, `op="flow"`, `form="bias"`, `algebra.transitive=True`,
    `sources=("probe", "gt")`, `label="support_closure"`, `gen=("stack",)`, `params={"edge": "support"}` -- the
    generic transitive-closure operator (`ops.ClosureOp`, already registered under both `"ancestor"` and `"flow"`
    by the foundation) over the support graph.
  - Neither entry touches `VOCABS`, `FIELDS`, any other unit's section, `base.py` or `ops.py`.
- `src/rrp/harness/data/relgen/support.py` (new; owned by this row):
  - `support_matrix(view, angle_deg=30.0) -> {entity_id: {entity_id: True}}`: direct support graph from
    `view.contacts()` (world normal within `angle_deg` of gravity-up, `_up = -gravity/|gravity|`) and `view.entities()`
    (poses corroborate "at a's top": a contact below the lower entity's own origin along "up" is rejected).
    Handles the contact's `.a`/`.b` reported in either order (normal within the angle band of up -> `.a` supports
    `.b`; within the band of down -> `.b` supports `.a`), OR-merges multiple contact points between one pair, and
    ignores degenerate (near-zero) normals.
  - `support_closure(direct) -> {entity_id: {entity_id: True}}`: pure transitive closure (iterative fixed point,
    terminates on a cycle) -- `out[a][b]` = `b` downstream of / `a` upstream of.
  - `support_pairs_fn` / `support_closure_fn` (`(StateView, TokenIndex) -> Label`, registered as `LabelDef`s
    `support_pairs` / `support_closure`, both `arity=2`, `needs={"contacts", "poses"}`): map the id-keyed graphs
    onto `[T, T, 1]` value / `[T, T]` valid arrays over whichever token set `TokenIndex.sets` carries (prefers
    `"ctx"`); a `None` (null / masked) slot is never valid, neither is a token against itself.
  - `stack` `ScenePart` (`activates={"support", "force_flow"}`, `envs=("mujoco/arm", "mujoco/dual")` per
    `research/relations_catalog.md`'s wave-1 env table): `build` adds `draft.kwargs["stack_n"]` (default 3) object
    descriptors (size/mass/color, one draw per object id) to `draft.entities` and records the sole
    bottom-to-top arrangement in `draft.kwargs["stack"]["order"]`; `vary(draft, rng, factor)` (docs 5.2's
    decoupling pairs) returns up to 3 seed-deterministic permutations of `order` alone -- same object ids, same
    per-object properties, same entities/events/every-other-kwarg, only which position each object occupies moves.
    `build` is safe to call more than once on the same draft (ids tagged by how many `"stack"` parts are already in
    `draft.parts`, so a second `stack` instance's ids never collide with the first's).

## acceptance evidence (row R17)
`tests/unit/test_relations_support.py` (new; not listed in the row's `owns` column, but the fanout rules ask every
unit to "write red/green tests for the acceptance criteria in your row" and every sibling row with a
labels/part-only surface, e.g. R9/R13, lists exactly this kind of dedicated test file -- treated as the missing-
from-the-table but clearly intended file, named after the sibling pattern `test_relations_<area>.py` /
`test_relgen_<area>.py`).
- **"stack fixture: support pairs ... correct"**: `test_support_matrix_direct_pairs_and_angle_threshold` (a hand-
  built 4-entity chain plus a side-lean distractor and a stray below-origin contact) and
  `test_support_pairs_label_registered_and_correct_on_the_fixture` (the registered `LabelDef` end to end: shape,
  direct pairs only, null-slot and self-pair invalidity). `test_support_matrix_rejects_contacts_past_the_angle_threshold`
  / `test_support_matrix_ignores_degenerate_zero_normals` isolate the 30° boundary and the zero-normal edge case.
- **"... and upstream / downstream closure correct"**: `test_support_closure_upstream_downstream_correct` (pure
  hand-computed graph), `test_support_closure_of_the_stack_fixture` (closure of the same fixture: `block_a` reaches
  every descendant though it only directly touches `block_b`), `test_support_closure_label_registered_and_matches_the_pure_closure`
  (the `LabelDef` end to end), `test_support_closure_terminates_on_a_cycle` / `_disjoint_components_stay_disjoint`
  (defensive: no infinite loop, no cross-component leakage).
- **"stack.vary changes only the order"**: `test_stack_vary_changes_only_the_order` (every field of the varied
  draft compared against the base one field at a time; only `kwargs["stack"]["order"]` may differ, and it always
  does, to a permutation of the same ids), `test_stack_vary_is_seed_deterministic`,
  `test_stack_vary_noop_below_two_objects`, `test_stack_build_produces_n_objects_and_activates_its_dynamics`,
  `test_stack_build_does_not_collide_ids_across_two_parts_in_one_draft`.
- Registry-level, matching docs section 7's "red/green" pattern ("`geo.pos3d source=gt` trains and raises in
  deploy mode"): `test_factors_registered_with_the_documented_shape`, `test_factors_resolve_and_default_deploy_safe`
  (probe-sourced by default: NOT privileged), `test_factor_gt_source_is_blocked_in_deploy_mode`
  (`ix.support source=gt` and `ix.force_flow control=gt` both raise `PrivilegedInput` via `assert_deployable`).
- Red/green discipline checked by hand: temporarily inverted the angle-threshold branch in `support_matrix`
  (`cosang >= cos_thr` -> `cosang >= -1.0`) and reran this file -- 5 of the 19 tests failed as expected
  (`test_support_matrix_*`, `test_support_closure_of_the_stack_fixture`, both `LabelDef` end-to-end tests), then
  reverted (`diff` against the pre-mutation copy confirmed byte-identical) and reran green.

`CUDA_VISIBLE_DEVICES="" ~/work/relational-robot-policy/.venv/bin/python -m pytest tests/unit/test_relations_support.py -q`:
19 passed, exit 0.
`CUDA_VISIBLE_DEVICES="" ~/work/relational-robot-policy/.venv/bin/python -m pytest tests/unit -q`: see the merge
section below for the exact count on the commit that landed.

## rules followed
- Never touched `relations/base.py` / `relations/ops.py` (imported read-only: `OPS`, `PrivilegedInput`,
  `assert_deployable`, `resolve`, `FactorSpec`, `get_factor` -- all pre-existing foundation API, nothing added or
  changed there).
- `catalog.py`: only the pre-created `R17: support / force flow` section touched; `VOCABS`, `FIELDS`, every other
  unit's section untouched (`git diff --stat` on this branch: `catalog.py | 19 +++++++++++++++++++`, one file,
  additions only, inside that section).
- `tests/data/golden.json` untouched (not read or written by anything in this unit).
- No shims; no new files beyond `harness/data/relgen/support.py`, `tests/unit/test_relations_support.py` and this
  note (the test file's placement is explained above).
