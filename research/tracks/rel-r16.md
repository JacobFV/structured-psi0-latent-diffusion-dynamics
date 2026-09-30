# R16: contact / grasp / handover (D-144 fanout, docs/relations.md section 10)

Worktree `~/work/rrp-wt/rel-r16`, branch `track/rel-r16` from `origin/main` (created at `ad9d422b`, well after the
D-144 foundation commit `c5e7d453`; deps R7 `state_view()` and R12 `ctx` token-set fields are both already merged
to `main`). Host only: git / editing / `pytest tests/unit`; no training, no simulation beyond unit-test fixtures.

Owned files (docs/relations.md section 10's table row): `src/rrp/policies/relations/catalog.py` §contact (own
section only, marked `# R16:` — every other unit's section is untouched), `src/rrp/harness/data/relgen/contact.py`
(new file: labels `contact_pairs`, `held_pairs`, `handover_pairs`; part `grasp_target`). Never touched
`relations/base.py` / `relations/ops.py` (no operator/interface change was needed). New test file:
`tests/unit/test_relations_contact.py` (not listed in the table row, but every unit needs its own red/green tests;
no other new files). `tests/data/golden.json` untouched.

## what changed

- **`policies/relations/catalog.py` §contact**: registers `ix.contact`, `ix.held_by`, `ix.handover`, all
  `field="hidden" op="bilinear" form="aug"` (docs 3.2: the bilinear kernel's aug features ARE the pair probe,
  `p_ij = sigmoid(<Ux_i,Vx_j> + c)`; `ops.BilinearOp` — foundation code, unedited). `sources=("probe", "gt")`:
  there is no public/estimated "given" field for contact state (unlike `geo.*`), so `probe` (the learned kernel) is
  the only deployable source and `gt` exists for training-time diagnostics / the deploy guard only (section 7).
  `readout=ReadoutDef(query, "pair", 1, "bce", label=..., reads="hidden")` — `"pair"` is the `ReadoutDef.address`
  literal docs section 4 names for bilinear factors. `params=(("rank", 8),)`. `register_preset("ix", [...])`.
- **`harness/data/relgen/contact.py`** (new): three `LabelDef`s, pure `(StateView, TokenIndex) -> Label` functions
  operating on the `ctx` token set (docs 2: every family with contact-relevant entities — arm, dual, legged — names
  it `ctx`), plus the scene part `grasp_target`.
  - `contact_pairs`: any two entities with >= 1 `StateView.contacts()` record between them.
  - `held_pairs`: reproduces today's `Session._held_truth` rule (an object touched by >= 2 distinct hand bodies of
    one manipulator assembly) AT THE STATEVIEW LEVEL. `StateView.contacts()` already resolves every hand body of
    one assembly to that ONE assembly entity id (`Session._body_entity_map`), so ">= 2 distinct hand bodies" shows
    up as ">= 2 separate contact records" between the same (object, assembly) entity pair (one MuJoCo contact per
    contacting geom pair). Also excludes assembly<->assembly and object<->object pairs (held is a manipulator<->
    object relation only).
  - `handover_pairs`: two manipulator-assembly entities simultaneously in contact with the same object entity — a
    handover in progress. A single-timestep `StateView` snapshot can't see a hand-to-hand transfer unfold, so this
    is the instantaneous proxy: "both hands on the object at once"; see lead question below.
  - `grasp_target`: `ScenePart(activates={"contact"}, envs=("mujoco/arm", "mujoco/dual"))`. `build` appends one
    graspable object entity + a `"grasp"` task event to a `SceneDraft`; `vary(draft, rng, "reach")` returns two
    drafts differing only in that object's position (inside vs. outside a manipulator's reach) — the decoupling
    pair docs 5.2 asks for, letting `ix.contact` / `ix.held_by` toggle without any other scene change confounding
    the comparison.

## acceptance (docs/relations.md section 10, R16 row)

| criterion | evidence |
|---|---|
| labels on grasp / dual-handover fixtures | `tests/unit/test_relations_contact.py`: `test_contact_pairs_*`, `test_held_pairs_*`, `test_handover_pairs_*` against a minimal fake `StateView` with hand-picked entities/contacts (values hand-computed, see docstrings); `test_contact_pairs_runs_on_a_real_mujoco_state_view` is a live-`Session.state_view()` smoke (shape/validity only — no physics-timing assertion, which would be flaky) |
| bilinear readout trains on a synthetic batch (loss decreases, host-cheap) | `test_ix_held_by_bilinear_readout_trains_on_a_synthetic_batch`: trains `OPS["bilinear"].build(...)`'s own `U/V/g/c` (the exact mechanism `ix.held_by`'s `aug` factor and pair-probe readout use) against a target realizable by that op family (`sign(x_i . x_j)` of fixed random token hiddens); CPU, `T=6`, 200 Adam steps, well under a second — loss strictly decreases (`final < initial - 0.1`) and converges (`final < 0.2`) |

Plus, beyond the literal acceptance bullets (kept as extra coverage, all red on pre-R16 `main`): registry-shape
tests (`ix.*` field/op/form/sources/readout/params), the deploy guard (`ix.*` defaults to `probe`, `source="gt"`
is blocked by `assert_deployable`; note `control="gt"` itself is REJECTED by `BilinearOp.controls()`, which only
allows `on/off/zero/rewired` regardless of form — `source="gt"` with `control="on"` (default) is the right way to
select the privileged source, `validate()` in `relations/base.py` catches the difference), `FactorSpec(source=
"given")` correctly rejected (no such source is declared), and full coverage of `grasp_target.build` / `.vary`.

Full suite on this branch: `PYTHONPATH=$PWD/src:$PWD CUDA_VISIBLE_DEVICES= pytest tests/unit -q` -> 651 passed, 39
skipped (pre-existing skips: optional extras / Menagerie assets not fetched / peer-only), 0 failed, exit 0.

## design decisions (autonomous, reversible; flagged here rather than blocking)

Two things the row's one-paragraph brief and the section-6/10 tables left underspecified; both are ordinary
registry-metadata choices inside this unit's OWN section of `catalog.py` and `contact.py` (no operator/interface
change, so not a "stop and ask" per AGENTS.md), but are worth a lead sanity-check:

1. **`handover_pairs` is single-timestep** ("two hands on the object at once"), not the literal "contacts over
   time" the section-6 table's gen column mentions for `ix.handover`. A real handover — hand A releases, hand B
   grips, released before gripped or slightly overlapping — needs an EPISODE, not one `StateView` snapshot; that
   belongs in the `relations_data` stage (R10, already merged) calling this per-frame label across a trajectory and
   is out of this unit's owned files. The per-frame co-contact proxy is the correct building block for that (it is
   exactly the frame where the sequence-level "handover" event is true) and is what R16's own acceptance bullet
   ("labels on ... dual-handover fixtures") tests.
2. **`gen=("grasp_target",)` on all three `ix.*` factors**: the section-6 table's compact `gen` column only lists
   `reveal` for `ix.contact` and `–` (nothing) for `ix.held_by`/`ix.handover`, while section 10's row explicitly
   assigns this unit ownership of `part grasp_target` without saying which factor's `gen` names it. Registered it
   on all three (they are the only factors that plausibly consume a graspable-object scene), not on any factor
   outside this unit's section, and not adding `reveal` (a docs 5.2 TRANSFORM, not this unit's `owns`) to keep the
   change strictly inside what R16 was asked to build.

Neither blocks a merge; no `lead_questions` entry needed for either (both are default-preserving registry choices,
reversible by any later unit's `catalog.py` edit to `gen=`).

## resume steps (if interrupted before merge)

1. `cd ~/work/rrp-wt/rel-r16 && export PYTHONPATH=$PWD/src:$PWD CUDA_VISIBLE_DEVICES=`
2. `pytest tests/unit -q` -- must exit 0 before merging.
3. `git fetch origin && git rebase origin/main`; rerun the unit suite if the rebase brought new commits.
4. `git push origin HEAD:main` (retry fetch/rebase/test/push up to 5x on a race).
