# R12: arm / dual token-set fields (D-144 fanout, docs/relations.md section 10)

Worktree `~/work/rrp-wt/rel-r12`, branch `track/rel-r12` from `origin/main` (c5e7d45, the D-144 foundation commit).
Host only: git / editing / `pytest tests/unit`; no training, no simulation beyond unit-test fixtures.

Owned files (per docs/relations.md section 10's table row): `src/rrp/policies/features/featurizer.py`,
`src/rrp/policies/features/multi.py`, `src/rrp/policies/features/kinfeat.py`, `src/rrp/policies/nets/batch.py`,
`src/rrp/envs/mujoco/sensors.py`. New test file: `tests/unit/test_relations_r12.py` (not listed in the table row,
but every unit needs red/green tests for its own acceptance criteria; no other new files).

## what changed

- **`envs/mujoco/sensors.py`**: `camera_frame(model, data, cam)` (pos, axis-columns, fovy of a declared camera) and
  `project_points(model, data, cam, points_world) -> [N,3] (u, v, depth)`, a pure pinhole projection matching the
  MuJoCo camera convention already used by `camera_visibility` (`cam_dir = -cmat[:, 2]`): normalized image-plane
  `(u, v)` at the vertical FOV (`FieldDef("cam_uvd", 3, ..., units="uv[-1,1], m")`), depth along the camera's
  forward axis. Points behind the camera get `depth <= 0` and `u = v = 0` (callers gate on `depth > 0`).
- **`policies/features/featurizer.py`**: additive-only. Named slice constants (`STATIC_DIM`, `NODE_ANCHOR_SLICE`,
  `NODE_AXIS_SLICE`, `ASM_POS_SLICE`, `ASM_ZCOL_SLICE`, `ASM_XCOL_SLICE`, `SCENE_POS_SLICE`, `SCENE_STD_SLICE`,
  `SCENE_VISIBLE_COL`, `SCENE_KNOWN_COL`) documenting the token-vector layout so `nets/batch.py` can read fields
  off it without a second definition of the layout; an assertion in `_build_static` that `node_static`'s width
  matches `STATIC_DIM` (keeps the constants honest). `Featurizer.__init__` / `featurizer_for` gained an explicit,
  optional `base_axes: bool | None = None` kwarg (see "`feat.base_axes`" below); `None` (the default, used by every
  existing call site) reproduces `kinfeat.enabled()` (`$RRP_KINFEAT`) exactly, so **no value changes** -- confirmed
  by `tests/unit/test_golden.py` (31 passed, byte-identical) and by `test_base_axes_none_is_byte_identical_to_pre_r12`.
- **`policies/features/multi.py`**: `MultiFeaturizer.__init__` threads the new `base_axes` kwarg down to its
  per-robot `Featurizer`s (same None-default, same no-op-by-default guarantee). No other change; it already copies
  `Featurizer`'s per-token rows verbatim, so the new featurizer.py slice constants apply unchanged to dual/multi
  `PolicyInput`s too (`test_ctx_fields_on_dual_featurizer`).
- **`policies/nets/batch.py`**: the collate-path TokenSet field builders (docs/relations.md section 2), all derived
  from EXISTING `PolicyInput` columns (`tokens`, `token_kind`, `relations`, `pointers`) -- no dataset rewrite:
  - `ctx_geometry_fields(inputs, batch)`: `pos3d` (+`.var`) and `orient` of the `ctx` bank. Morph action / passive
    tokens -> the public-FK joint anchor (`NODE_ANCHOR_SLICE`), always valid, no `orient` (a point, not a frame).
    Morph assembly tokens -> the frame origin (`ASM_POS_SLICE`) and the 3x3 frame rotation, reconstructed from its
    stored `R[:, 2]` / `R[:, 0]` columns via `R[:, 1] = cross(R[:, 2], R[:, 0])` (right-handed orthonormal frame;
    `FieldDef("orient", 9, ...)`). Scene tokens -> the tracked position, valid iff `known`, `.var` recovered from
    the stored `log(std + 1e-4) / 5`; no `orient` (object frames are unknown to the tracker).
  - `ctx_id_fields(inputs, batch)`: `entity_id` (self-index by default; a token that is the source of EXACTLY ONE
    incidence pointer takes its destination's index instead, so two tokens referencing the same public entity
    compare equal under the `same` op -- a fan-out source, e.g. a multi-argument `pred_arg` predicate token, is
    not any one entity's alias and keeps its self id) and `assembly_id` (-1 by default; a morph assembly token's
    own id is its own index; any OTHER ctx token with a `node_in_assembly` edge TO one -- today: `interact`
    sensor tokens -- takes it). `node_in_assembly` is stored BOTH directions in `relations` (member->assembly AND
    the reverse, so both attention orderings see the bias); only the direction whose KEY is the assembly token
    means membership, so the reverse rows are filtered out (`R[:, 2] == MORPH_BANK_ID` + destination kind check)
    -- an early version of this got that backwards and briefly aliased an assembly token's OWN id to the sensor
    it senses; caught by `test_assembly_id_self_ids_assembly_tokens_and_reaches_interact_sensors`.
  - `act_assembly_id(inputs, batch)`: `assembly_id` of the `act` token set (action nodes) from the act>ctx
    `node_in_assembly` edge (`relations[:, 0] == -1`).
  - `attach_cam_uvd(pos3d, pos3d_valid, cameras)`: `cam_uvd` from an already-derived `pos3d`, projected per batch
    item through its own declared camera (`sensors.project_points`; `cameras[i] = (model, data, cam_name)` -- the
    collate path itself has no live MuJoCo state, so this is a separate step, unlike the other fields).
  - `relation_token_sets(inputs, batch, cameras=None)`: wraps the above into the `ctx` / `act` `TokenSet`s
    (docs/relations.md section 2) that a net can hand to `RelCtx`; `TokenSet.kind` maps the bank-local subtype ids
    already in `PolicyInput.token_kind` onto the shared `TOKEN_KINDS` vocabulary.

## acceptance (docs/relations.md section 10, R12 row)

| criterion | evidence |
|---|---|
| featurizer goldens unchanged | `pytest tests/unit/test_golden.py -q` -> 31 passed, byte-identical (ran on this branch) |
| `pos3d` (+`.var`), `cam_uvd`, `orient`, `entity_id`, `assembly_id` with correct provenance on the arm fixture | `tests/unit/test_relations_r12.py`: `test_pos3d_*`, `test_orient_*`, `test_entity_id_*`, `test_assembly_id_*`, `test_act_assembly_id_*`, `test_attach_cam_uvd_*`, `test_relation_token_sets_shapes_and_masks`, `test_orient_dim_matches_catalog_field_def` (dims/prov match `catalog.FIELDS`, foundation-registered) |
| projection test against MuJoCo camera math | `test_project_points_matches_mujoco_camera_math` (principal-ray / FOV-edge / inverse-depth identities against the fixture's own `cam_xpos` / `cam_xmat` / `cam_fovy`), `test_project_points_behind_camera_is_flagged_not_crashed`, `test_project_points_batches` |
| `$RRP_KINFEAT` gone, `feat.base_axes` reproduces its features | PARTIAL -- see below |

Full suite on this branch: `pytest tests/unit -q` (see PR / merge log for the exit code recorded at merge time).

## `$RRP_KINFEAT` -> `feat.base_axes`: scope conflict, flagged for the lead

The brief (docs/relations.md section 10, R12) reads "Replace `$RRP_KINFEAT` by the factor-list option
`feat.base_axes` ... `$RRP_KINFEAT` gone". Implemented, inside this unit's owned files: `Featurizer` /
`MultiFeaturizer` no longer read `os.environ` directly for their OWN decision -- they take an explicit
`base_axes: bool | None` kwarg (`None` = legacy env-var behaviour, for byte-identical goldens; `True` / `False`
pins it regardless of the environment). `test_base_axes_true_matches_legacy_kinfeat_env` /
`test_base_axes_false_ignores_env` show the explicit path reproduces (or overrides) the flag's features exactly.

What is NOT done, and can't be done without leaving this unit's owned files (AGENTS.md: "Edit ONLY the files your
row owns"): the `$RRP_KINFEAT` env var itself is still read in three places R12 does not own --
`harness/pipelines/base.py` (`rc.options.get("kinfeat")` -> `os.environ["RRP_KINFEAT"]`),
`harness/data/packed.py` / `harness/data/latent.py` (`kinfeat.enabled()` at load time), and
`policies/nets/checkpoint.py` (`kinfeat.VERSION` / `kinfeat.enabled()` in the checkpoint version guard). None of
those are in R12's `owns` column, and `kinfeat.py`'s `ENV` / `VERSION` / `enabled()` / `apply_rows()` /
`table_for_robot()` are kept EXACTLY as they were so those three unowned call sites keep working unmodified.
There is also no `feat.base_axes` entry in `relations.catalog.FACTORS` -- it isn't an attention factor (field x
op x form); it doesn't fit `FactorDef`, and `catalog.py` sections are owned per-unit too (R12's row lists no
`catalog.py` section, unlike R13/R15/...), so wiring an actual `factors: [feat.base_axes]` run-config entry
end-to-end also crosses into files this unit does not own (`core/runconfig.py` at least).

**Lead question**: finishing the literal "`$RRP_KINFEAT` gone" bullet needs either (a) a follow-up unit scoped to
`harness/pipelines/base.py` + `harness/data/packed.py` + `harness/data/latent.py` + `policies/nets/checkpoint.py`
(+ deciding where `feat.base_axes` should actually live if it's meant to be resolvable from `factors:`, given it
isn't an attention factor), or (b) explicit sign-off that R12 stops at the featurizer-level `base_axes` kwarg and
the env var deletion is out of scope for this unit. Not blocking: R12's other four acceptance bullets are fully
met and merged independently of this one.

## resume steps (if interrupted before merge)

1. `cd ~/work/rrp-wt/rel-r12 && export PYTHONPATH=$PWD/src:$PWD`
2. `pytest tests/unit -q` -- must exit 0 before merging (see AGENTS.md / docs/relations.md section 10 rules).
3. `git fetch origin && git rebase origin/main`; rerun the unit suite if the rebase brought new commits.
4. `git push origin HEAD:main` (retry fetch/rebase/test/push up to 5x on a race).
