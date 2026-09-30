# R14: geometry labels + parts (D-144 fanout, docs/relations.md section 10)

Worktree `~/work/rrp-wt/rel-r14`, branch `track/rel-r14` from `origin/main` (b2d15d7c, which includes the D-144
foundation c5e7d45 and R7's `state_view()` on the MuJoCo sessions -- this row's dependency). Host only: git /
editing / `pytest tests/unit`; no training, no simulation beyond unit-test fixtures.

Owned files (docs/relations.md section 10's table row): `src/rrp/harness/data/relgen/geometry.py`. New test file
`tests/unit/test_relations_r14.py` (not listed in the table row, but every unit needs red/green tests for its own
acceptance criteria, per the precedent set by R12's `test_relations_r12.py`; no other new files). `base.py` /
`ops.py` / `relgen/__init__.py` untouched -- no operator/interface change was needed for this row.

## what changed

`src/rrp/harness/data/relgen/geometry.py` registers, against the F4 registries (`LABELS`, `PARTS`):

- **labels** (`LabelDef.fn: (StateView, TokenIndex) -> Label`, arity 1, all `prov="gt"`), computed over the
  flattened token order across every set in `TokenIndex.sets` (dict order, then slot order):
  - `pos3d` (needs `poses`): the entity's true `pos`.
  - `orient` (needs `poses`): the entity's true orientation as a 3x3 world rotation matrix, row-major flattened to
    9 dims (`FieldDef("orient", 9, ...)`); invalid when the entity has no `quat`. `_quat_to_mat` is the standard
    wxyz -> matrix formula, checked directly against MuJoCo's own `xmat` on a real fixture (below) rather than
    trusted blind.
  - `contact_normal` (needs `contacts`): the mean, renormalized, of every contact's normal touching that entity --
    `ContactState.normal` points `a -> b`, so it is taken as-is for the `a` side and negated for the `b` side, so
    the label is always OUTWARD from the labelled entity's own surface. An entity touched by zero contacts is
    invalid (not zero-and-valid).
  - `cam_uvd` (needs `poses`, `camera`): `project()` (below) of the entity's true position through
    `view.camera(DEFAULT_CAMERA="front")` -- the one camera name every arm/dual table scenario declares
    (`rrp.bodies.generators.workspace_spec`). Per the brief, depth comes from `view.render_depth(...)` at the
    projected pixel when the view declares `"depth_render"` (so occlusion / render artifacts enter the label, not
    just the analytic straight-line distance), else from the analytic depth `project()` already computed. Points
    behind the camera (`depth <= 0`) and pixels outside the rendered frame fall back to / stay analytic-invalid.
  - `project(cam, points_world) -> (u, v, depth)` / `unproject(cam, u, v, depth) -> point` are a `Camera`-only
    (`K`, `T_world_cam`) pinhole pair, exact inverses of each other by construction. Deliberately independent of
    `rrp.envs.mujoco.sensors.project_points` (which needs a live `model`/`data`, MuJoCo-specific) so a `StateView`
    label stays backend-agnostic; cross-checked to agree with `sensors.project_points` on a real session (both
    derive from the same pinhole model and the same `Camera.K = [[fy,0,w/2],[0,fy,h/2],[0,0,1]]`,
    `fy = (h/2)/tan(fovy/2)` convention already fixed by R7's `MujocoStateView.camera()`).

- **parts** (`ScenePart.build(draft, rng) -> None`, `ScenePart.vary(draft, rng, factor, k=2) -> list[SceneDraft]`),
  operating purely on the declarative `SceneDraft` -- `relgen.compose` (unit R11) is not implemented yet
  (`NotImplementedError` on `main`), so nothing here ever touches a live simulator:
  - `table_objects` (`activates = {"orient", "above"}`): places `n_objects` (default 3, `draft.kwargs`) at random
    `(x, y, yaw)` on a table plane. `.vary(draft, rng, "orient")` returns `k` copies whose objects keep their `id`
    and `pos` byte-identical and get a freshly drawn `yaw` -- the docs 5.2 "decoupling pair" for `geo.orient`.
  - `camera_depth` (`activates = {"depth"}`): places one `depth_probe` object along a camera's ray at a chosen
    pixel `(pixel_u, pixel_v)` and `depth` (`draft.kwargs`, else drawn from `rng`), via `unproject`.
    `.vary(draft, rng, "depth")` returns `k` copies whose probe keeps the SAME `(u, v)` (re-derived from `project`
    after the move, not merely copied) and gets a freshly drawn depth -- docs 6's `geo.depth3d` row ("same pixel,
    other depth"), and the acceptance criterion this row names explicitly. The camera is `draft.kwargs["camera"]`
    if supplied, else `front_camera()`: a literal reconstruction of `workspace_spec`'s declared "front" camera
    (`pos=[1.25,0,0.65]`, `xyaxes=[0,1,0,-0.45,0,0.9]`, `fovy=55`, 64x64), checked to match a real arm session's
    `sv.camera("front")` (K and T_world_cam) to `atol=1e-6`.
  - Both `.vary`s reject an unrecognized `factor` (`ValueError`) and record `{"part", "factor"}` under
    `draft.provenance["vary"]`; `.build` appends `{"part", "version"}` to `draft.provenance["parts"]` and unions
    `activates` into `draft.active` -- provenance recorded, per the acceptance criterion.

**Design decision, not a base/ops change:** `LabelDef.fn`'s fixed two-argument signature
(`(StateView, TokenIndex) -> Label`) has no slot for a camera NAME (R12's live collate path instead threads
`cameras: list[(model, data, cam_name)]` in from its caller -- not available to a registry-level `LabelDef`).
`cam_uvd` therefore defaults to the one camera name every declared arm/dual table scenario actually has ("front"),
matching `rrp.envs.mujoco.sensors.DetectorConfig`'s own default and verified against a real session above. This is
a default, not an interface limitation that blocks the row (a future multi-camera factor could pass a different
name once R11/R18 exist to plumb one in), so nothing is reported to the lead as blocked.

## tests (red then green)

`tests/unit/test_relations_r14.py`, 22 cases. Verified red first: moved `geometry.py` aside and confirmed
`pytest tests/unit/test_relations_r14.py` fails on `ModuleNotFoundError: rrp.harness.data.relgen.geometry`, then
restored it and iterated to green.

- **registration**: `LABELS`/`PARTS` entries have the right `arity` / `needs` / `prov` / `activates` / `vary`.
- **pos3d**: matches `EntityState.pos` on real arm AND dual fixtures (`make_pick_place_session`, the dual
  `support_insert` session from `test_state_view.py`'s own fixture pattern), including a null-identity slot
  (invalid, zero); a synthetic multi-token-set `TokenIndex` (`{"ctx": [...], "act": [...]}`) checks the flattening
  order directly.
- **orient**: on the real arm fixture's cube, `.value[0].reshape(3,3)` matches MuJoCo's own `d.xmat[bid]` to
  `atol=1e-6` (an independent ground truth, not just our own formula fed back to itself); a no-quat entity is
  invalid; `_quat_to_mat` checked at identity and a hand-derived 90 degree z rotation.
- **contact_normal**: a fake `StateView` with two synthetic contacts on one entity checks the outward-orientation
  convention (`a`-side = `c.normal`, `b`-side = `-c.normal`) AND the mean-then-renormalize averaging by hand; an
  untouched entity is invalid (not zero-and-valid); on the real arm fixture, the settled cube's label is a unit
  vector and a never-touching "target" zone stays invalid.
- **cam_uvd**: on the real arm fixture (with `"depth_render"` hidden via a small wrapper view, so the analytic
  branch runs -- see below), `project()` matches `rrp.envs.mujoco.sensors.project_points` to `atol=1e-5` for two
  entities at different depths; a fake view with a controlled `render_depth` array checks the render-driven branch
  reads the right pixel and OVERRIDES the analytic depth; a point placed behind the camera is invalid; a view
  without the `"camera"` capability is all-invalid; `front_camera()` matches a real session's `sv.camera("front")`;
  `project`/`unproject` round-trip exactly over 20 random points.
- **table_objects**: `.build` places objects in the declared ranges with valid provenance/`active`; `.vary`
  changes ONLY yaw (positions/ids byte-identical, original draft untouched) and is seed-deterministic (same seed
  -> same yaws); an unknown factor raises.
- **camera_depth**: `.build` places the probe so `project(cam, probe.pos)` recovers the exact requested
  `(pixel_u, pixel_v, depth)`; `.vary` keeps the SAME projected pixel while depth changes (checked by
  re-projecting, not by trusting the move), across `k=3` copies, with the original draft's probe untouched; `.vary`
  before `.build` and with an unknown factor both raise.

**Host-safety note (docs/relations.md 10 row R7: "EGL only on the peer -- host tests use the camera math only")**:
a real MuJoCo `state_view()`'s `caps` unconditionally include `"depth_render"` (R7's `MujocoStateView.CAPS`), so
`cam_uvd`'s render-driven branch would otherwise construct a real `mujoco.Renderer` (a GL context) the first time
its test ran against a real session -- which it did, initially, and the resulting depth (correctly, since the
render sees whatever surface is actually closest along that ray, not necessarily the queried entity's own point)
did NOT match the analytic `project_points` cross-check, which is what caught this. Fixed by wrapping the real view
in a small `_NoDepthRenderView` (hides the `"depth_render"` cap; test-only, not shipped) for the analytic-path
test, and covering the render-driven branch exclusively through a fake `StateView` with a controlled depth array
(never a real `mujoco.Renderer`) -- the same host-safety pattern R7's own `test_render_depth_shape_matches_declared_size` established.

## commands run

```
cd ~/work/rrp-wt/rel-r14 && export PYTHONPATH=$PWD/src:$PWD
PY=~/work/relational-robot-policy/.venv/bin/python
$PY -m pytest tests/unit -q                                   # baseline (pre-change): 631 passed, 39 skipped, exit 0
mv src/rrp/harness/data/relgen/geometry.py /tmp/geometry.py.bak
$PY -m pytest tests/unit/test_relations_r14.py -q             # red: ModuleNotFoundError
mv /tmp/geometry.py.bak src/rrp/harness/data/relgen/geometry.py
$PY -m pytest tests/unit/test_relations_r14.py -q             # green: 22 passed
$PY -m pytest tests/unit -q                                   # full suite post-change: 653 passed, 39 skipped, exit 0
```

State: **completed** (implementation + tests green; merge to main pending per docs/relations.md 10's merge rule).
