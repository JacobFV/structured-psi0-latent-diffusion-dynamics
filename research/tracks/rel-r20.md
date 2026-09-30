# track: rel-r20 (D-144 relation-factor fan-out, unit R20: UI factors)

Worktree `~/work/rrp-wt/rel-r20`, branch `track/rel-r20`, from `origin/main`. Brief: `docs/relations.md` section 10,
row R20 / briefs: "ComputerWorld public UI fields (parent, z-layer, focus rank) and `ui-rel-v1` edges in the env
adapter, UI entries and labels (label_for, contains, focus_next, above, drag_to from the teacher), `cf_swap(label)`."
Deps: R6 (pointer nets, merged), R8 (StateView: Warp / ComputerWorld / SIMPLE, merged). Host only:
`CUDA_VISIBLE_DEVICES=""`, no training / simulation beyond unit-test fixtures. Never edited `relations/base.py` /
`relations/ops.py`. Read `docs/relations.md` sections 2-5, 10 and `research/tracks/cworld.md` first, per the brief.

## what changed
- `src/rrp/envs/computerworld.py`:
  - `cw_state_view`'s `ui_tree` node dicts gain one key, `"order"` (the widget's raw scene index, already on every
    `scene_widgets` record) -- needed by `relgen/ui.py`'s labels to recompute document order from `StateView`
    alone; every existing consumer only reads specific keys by name (`test_state_view_envs.py`), so this is
    additive and does not touch that already-merged R8 test.
  - New pure functions (plain numpy; `envs` sits below `policies` in the layer order, `tests/unit/test_layering.py`,
    so this file never imports `rrp.policies.relations` -- the actual `TokenSet`/`EdgeSet` wrapping of this data is
    a collate-path concern, same split rel-geo's addendum records for `edges:support-v1`):
    - `UI_REL_VOCAB = ("label_for", "contains", "focus_next", "above")` -- the `ui-rel-v1` edge vocabulary.
    - `ui_public_fields(table) -> {"parent_id", "zlayer", "focus_rank"}`: `parent_id` = window index (-1 desktop
      level / null slot), `zlayer` = `w["layer"]`, `focus_rank` = 0/-1 matching `cw_state_view`'s OWN already-merged
      convention exactly (0 = the one currently-focused widget via `scene.focus.interaction`, -1 = every other
      widget) -- deliberately NOT a full tab-order rank, to stay consistent with that R8-authored, already-tested
      field rather than silently redefining it.
    - `ui_edges(table) -> [W, W, 4] bool`: `contains` (same-window membership, reflexive/symmetric; desktop-level
      widgets never match each other -- there is no window ENTITY token to hang a real containment edge off, CW's
      window frame carries no `semantic` node), `label_for` (role="label" widget -> the next widget after it in
      scene order within the same window), `focus_next` (consecutive slots of the public tab order: focusable,
      enabled, boxed widgets by scene order, cyclic), `above` (`zlayer_j > zlayer_i`).
- `src/rrp/policies/relations/catalog.py` (`R20: UI` section only): two new `FieldDef`s (`zlayer`, `focus_rank`,
  both `kind="scalar"`, `prov="public"`; `parent_id` / `entity_id` already existed), `VOCABS["ui-rel-v1"]
  = UI_REL_VOCAB` (imported from `envs.computerworld`: policies -> envs is the allowed, downward direction), five
  `FactorDef`s and `register_preset("ui", [...])`:
  - `ui.label_for` / `ui.contains` / `ui.focus_next` / `ui.above`: the foundation's `edge.*` pattern applied to
    `UI_REL_VOCAB` (`field="edges:ui-rel-v1"`, `op="edge"`, `form="bias"`), `sources=("given", "gt")` (public /
    deterministic by construction, like `edge.*`; "gt" additionally offered matching `geo.*`'s pattern so the
    privileged `relgen.ui` label of the same name can be diagnosed against it), `label=<name>`. `ui.contains` alone
    is `Algebra(direction="symmetric")` (same-window membership); the other three default to `directed`.
    `ui.label_for` alone carries `gen=("reveal", "surprise")`: which label-role widget a control is bound to is
    docs 5.4's own worked surprise example ("UI label changes") -- a candidate classification over a window's
    label-role widgets, not a fixed fact, unlike `contains`/`focus_next`/`above`.
  - `ui.drag_to`: the `ix.*`/`leg.*` bilinear pair-probe pattern (`field="hidden"`, `op="bilinear"`, `form="aug"`,
    `sources=("probe", "gt")`, `label="drag_to"`, `readout=ReadoutDef("drag_to", "pair", 1, "bce", ...)`) -- there
    is no public/estimated "given" source for a drag DESTINATION (docs 3.1: the bias IS the pair probe).
- `src/rrp/harness/data/relgen/ui.py` (new): privileged labels of the same five names, written once against
  `StateView.ui_tree()` (+ `entities()` for `focus_next`'s `focusable`/`disabled` flags and `drag_to`'s positions).
  `contains_fn` / `label_for_fn` / `focus_next_fn` / `above_fn` independently recompute exactly the relation
  `ui_edges` builds from the raw scene -- the same StateView-side duplication R14 accepts for `geo.*`'s "given"
  fields, cross-checked here against `ui_edges` itself. `dragged_widget_id` / `drag_to_fn`: the widget CW's own
  focus currently names -> the nearest OTHER widget by position (R19's `leg.foothold` nearest-candidate pattern).
  `label_for_sample(control, candidates, view, ...)`: `next_contact_sample`'s exact shape (docs 5.4 glue, R9's
  `reveal`/`surprise` reused unmodified), built around `label_for`'s own candidate set (co-windowed label-role
  widgets) instead of contact candidates.
- `tests/unit/test_relations_r20.py` (new, 31 cases, 1 `@pytest.mark.computerworld`-marked and auto-skipped without
  the optional wheel): `ui_public_fields` / `ui_edges` on a 2-window, 4-widget fixture scene (one enabled + one
  disabled textbox, each preceded by its label, at two z-layers) -- edge extraction per channel, directedness,
  null-slot rows/columns all-False; `relgen.ui`'s four structural labels cross-checked against `ui_edges` on a
  `cw_state_view` of the SAME scene; `drag_to_fn` nearest-candidate and "nothing focused -> all zero" cases;
  registry / catalog assertions (fields, factor shapes, sources, algebra, `gen`, preset, `VOCABS["ui-rel-v1"]`,
  deploy guard green on the default preset and red under a forced `gt` source); a `FactorSite` built on the `ui`
  preset resolves at a hand-built `ctx>ctx` site and runs one real `bias()` / `augment()` forward (finite,
  correctly-shaped logits that soften to a valid attention distribution) -- standing in for one rollout tick's
  attention layer on the SAME pure functions the real `ComputerWorldEnv.observe()` / `state_view()` build on,
  since this unit does not own `policies/pointer.py` (R6's own `RelBlock`s there still carry the empty preset
  `none` by design, per catalog.py's own R6 comment: "there is nothing here for R20 (UI factors) to do but ADD
  entries, not change this preset") and cannot wire the SAME check into a live checkpoint-free net without a
  collate-path unit this row does not own; `cf_swap` applied to a widget "label" field (`docs 5.2`'s literal
  `cf_swap(label)`) proven to move a UI-shaped arity-2 label (`contains`) consistently on both axes;
  `label_for_sample` + `reveal` (collapses onto the true binding) + `surprise` (docs 5.4's own worked example, "UI
  label changes": a later contradiction flips the collapsed candidate, `switch_at` recorded). Plus one
  `@pytest.mark.computerworld` test building a REAL `ComputerWorldEnv(body="cw_pointer")` tick and running the same
  `FactorSite` forward on its actual scene -- verified locally with the wheel on `PYTHONPATH`
  (`PYTHONPATH=$PWD/src:$PWD:$HOME/work/ext/cw-site`), skips (does not fail) under the merge gate's own command,
  which does not add that path.

`pytest tests/unit -q -x -p no:cacheprovider`: 884 passed, 41 skipped (all pre-existing menagerie/CUDA/optional-
extra skips, plus this unit's own `computerworld`-marked test skipping the same way `test_computerworld.py` /
`test_pointer.py` already do), exit 0. Ran with `CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD/src:$PWD
~/work/relational-robot-policy/.venv/bin/python`, per the merge protocol (no `~/work/ext/cw-site` on `PYTHONPATH`,
matching the exact command AGENTS.md's merge lock step runs). Separately re-ran
`tests/unit/test_relations_r20.py` alone with `~/work/ext/cw-site` appended to `PYTHONPATH`: 31 passed, 0 skipped
(confirms the `computerworld`-marked test genuinely exercises the real wheel and passes, not just "skips cleanly").
`tests/data/golden.json` untouched (not referenced by any file this unit touched).

## design notes (for the record, not litigated further)
- **Why `ui.above` is an `edges:ui-rel-v1` channel, not `field="zlayer"`/`op="order"`** (the `geo.above` pattern):
  `order`'s "gt" control reads `TokenSet.label(d.field)` -- i.e. a label keyed by the FIELD name ("zlayer"), not by
  the factor name ("above"). The brief names five LABELS by the FACTOR names (`label_for, contains, focus_next,
  above, drag_to`), which only lines up with the "gt" control's actual lookup key when the factor is edge-shaped
  (`EdgeOp`'s "gt" reads a whole separate privileged `EdgeSet`, keyed by the edge/channel name, matching a label of
  that same name by the established `_EDGE_DOC` convention) -- so `above` joined the other three structural
  channels instead of introducing a second, `order`-shaped path with a mismatched label key.
- **Why `contains` means "same window" rather than a literal container -> member edge**: CW's window frame node
  carries no `semantic` field (`scene_widgets` only emits nodes that do), so a window is never itself a widget /
  token in the pointer body's own `ctx` set -- there is no first token to put on the "contains" row of a directed
  containment edge. Documented in both `ui_edges`'s and `catalog.py`'s docstrings so nobody re-litigates it as a
  bug; a real container -> member edge needs a widget-shaped window token, out of this row's scope.
- **Why `focus_rank` stays binary (0 / -1), not a tab-order integer**: `cw_state_view`'s `ui_tree` entries already
  set `focus_rank = 0 if w["focused"] else None` (R8, already merged and tested,
  `test_state_view_envs.py::test_cw_state_view_on_fixture_scene`). Redefining it to a full tab-order rank here would
  have silently changed that already-accepted, already-tested field's meaning without owning that test file to fix
  it. `ui_edges`'s own `focus_next` channel gets its ordering from `_tab_order` (scene-order among focusable/enabled
  widgets) independently, so nothing needed the field itself to carry a rank.

## lead_questions
- No live net wires `ui.*` into a real `RelCtx` yet: `policies/pointer.py`'s `UICtx` (R6) builds its widget context
  tokens as a bare self-attention stack (`RelBlock(x, kv=x, q_mask=m)`), never constructing a `TokenSet`/`RelCtx` at
  all. Whoever next touches `policies/pointer.py` (out of this row's owned files) can wire `ui_public_fields` /
  `ui_edges` into that net's collate path and pass `factors=["preset:ui", ...]` through `PolicyConfig`, the same
  step R13/rel-geo's `CTX_CARRIES` wiring did for the arm's `geo.*`. Nothing here blocks it: `UI_REL_VOCAB`,
  `ui_public_fields` and `ui_edges` are ready to be called from that collate path exactly as tested.
- `ui.drag_to`'s label (`dragged_widget_id`) treats "the currently focused widget" as "the one being dragged" --
  the only public/privileged signal a 2D desktop UI's `StateView` exposes without task-specific plumbing (CW has no
  explicit "drag in progress" flag in the scene). A future unit with access to task/teacher internals (e.g. the
  `cw/drag_window` judge's own goal state) could sharpen this to a genuine "mid-drag" predicate; flagged, not
  blocking (same status as R17's own "for the lead" notes before rel-geo).

**Closed (worktree `sweep-pointer`, `track/sweep-pointer`, D-144 addendum, 2026-09-29):** `policies/pointer.py`'s
`UICtx` now builds a `TokenSet`/`RelCtx` at its widget self-attention (`ctx>ctx`) from this row's own
`ui_public_fields` / `ui_edges` (via a new `widget_features(obs, half, table=None)` argument -- `table` optional so
every existing call site is untouched) plus screen-geometry fields `pos3d` / `cam_uvd` (the same 1 mm/px
`ScreenFrame` mapping `widget_position` already used, always populated, no `table` needed) and `zlayer`; a new
`UI_CARRIES` tuple and per-layer `FactorSite` (mirroring rel-geo's `CTX_CARRIES`) let `PolicyConfig(factors=
["preset:ui"])` (new dataclass) and `geo.pos3d` / `geo.depth3d` resolve there. Default `factors=None` keeps every
`FactorSite` parameter-free (`.bias()` / `.augment()` exactly `None` / `(None, None)`) -- byte-identical checkpoints,
verified by `test_default_uictx_factors_are_a_zero_bias_no_op`. `test_preset_ui_changes_widget_self_attention_logits_
on_the_cw_fixture_scene` (`tests/unit/test_pointer.py`, this row's own CW fixture scene, no wheel needed) enables
`preset:ui`, perturbs the (off zero-init) `FactorSite` weights and shows the self-attention bias goes from `None` to
a real finite non-zero `[B,H,T,T]` term and the forward output changes. Full suite: 888 passed, 41 skipped, exit 0.
`relations/base.py` / `relations/ops.py` untouched; `tests/data/golden.json` untouched.
