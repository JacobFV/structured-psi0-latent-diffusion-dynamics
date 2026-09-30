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

## R12c (worktree `~/work/rrp-wt/rel-r12c`, branch `track/rel-r12c` from `origin/main`): `$RRP_KINFEAT` deleted

Resolves the "lead question" above under D-144 addendum decision (b) (research/decisions.md): the env var deletion
is IN scope, done now, across every file that read it (not just the featurizer). No `factors: [feat.base_axes]`
run-config wiring was added (still out of scope per (b) -- `feat.base_axes` is a featurizer option, not an
attention `FactorDef`; that stays a `core/runconfig.py` question for whoever needs it next).

- **`policies/features/kinfeat.py`**: `ENV` / `enabled()` deleted. Replaced by `resolved(explicit: bool | None =
  None) -> bool` (explicit wins; else the process-ambient value set by `set_base_axes`; else `False`) and
  `set_base_axes(value) -> prev` (plain module global, set/restored like the old env var but never touching
  `os.environ`). `legacy_bool(v)` is the ONE remaining string-vocabulary mapping table (`"v1"/"1"` -> True,
  `""/"0"/"off"/"none"` -> False), read only by loaders (decision b) -- never called to decide live behaviour.
  `table_for_robot` no longer does the env-var toggle/restore dance; it calls `featurizer_for(s, base_axes=True)`
  directly (the explicit kwarg already existed from R12's first pass).
- **`policies/features/featurizer.py` / `multi.py`**: `self.kinfeat = kinfeat.enabled() if base_axes is None else
  bool(base_axes)` -> `self.kinfeat = kinfeat.resolved(self.base_axes)`. Same net behaviour (an explicit kwarg
  still pins it; `None` still means "ambient default"), just no more env read. Docstrings updated; no functional
  change to any call site that already passed an explicit `base_axes`.
- **`harness/pipelines/base.py`**: the `options.kinfeat` stage-option block no longer sets/restores
  `os.environ["RRP_KINFEAT"]`; it calls `kinfeat.set_base_axes(kinfeat.legacy_bool(kf) if kf is not None else
  None)` before `spec.fn(ctx)` and restores the previous ambient value in the `finally`. The "contradicts
  $RRP_KINFEAT env" check is gone (nothing to contradict any more -- there is no external env var). Cross-process
  propagation was never actually needed here: every DAG node (`harness/dag.py Runner.command`) already serializes
  its OWN full `RunConfig` (`--config-b64`) into a fresh `rrp.cli stage run` subprocess, which re-enters
  `Pipeline.run()` and re-resolves `rc.options.get("kinfeat")` independently -- the old env var was redundant for
  subprocess nodes and only mattered for code running IN-PROCESS during that same stage call (data
  collection/training code in the same Python process), which the ambient module global covers identically.
- **`harness/data/packed.py`**: `self.kinfeat = kinfeat.enabled()` -> `kinfeat.resolved()`. `harness/data/latent.py`
  only reads `self.ds.kinfeat` (comment updated, no functional change).
- **`policies/nets/checkpoint.py`**: `save_checkpoint` no longer writes a standalone `versions["kinfeat"]` key;
  `kinfeat.resolved()` (when True) appends `kinfeat.VERSION` into the SAME combined `versions["factors"]` string
  used for attention-factor `compat_hash` (`"+".join([...])`; either component may be absent). `check_kinfeat`
  (still called unconditionally from `load_checkpoint`, preserving "never silently mixed") reads BOTH the legacy
  standalone `versions["kinfeat"]` key (old checkpoints, decision-b remap) and the new folded
  `versions["factors"]` string (`kinfeat.VERSION in factors.split("+")`), compared against `kinfeat.resolved()`
  (ambient default at load time, same role the env var played before).
- **`dags/arm_lineage_v7div_kinfeat.yaml`, `dags/armdiv_bc_v7div_kinfeat.yaml`**: `options: {kinfeat: v1}` (the
  RunConfig-level surface) is UNCHANGED -- only its internal implementation changed. Comments that mentioned
  `$RRP_KINFEAT=v1` reworded to `options.kinfeat=v1`; no functional YAML change. `research/tracks/armdiv.md`'s
  RESUME section and `scripts/armdiv_chain.sh` call these DAGs unmodified and need no changes: `armdiv_chain.sh`
  only invokes `rrp.cli run-dag <file>.yaml`, which builds each node's `RunConfig` from the YAML `options:` block
  the same way as before -- verified by `tests/unit/test_pipelines.py::
  test_kinfeat_option_sets_ambient_base_axes_for_the_stage_only` (options.kinfeat=v1 resolves to
  `kinfeat.resolved() is True` inside the stage function, restores to `False` after, never touches `os.environ`).
- **Tests**: `tests/unit/test_kinfeat.py` rewritten around the new API (ambient set/resolve, `legacy_bool`,
  `check_kinfeat` reading both the legacy and folded checkpoint version shapes); `tests/unit/test_relations_r12.py`'s
  three `base_axes` tests swapped `monkeypatch.setenv/delenv("RRP_KINFEAT", ...)` for `kinfeat.set_base_axes`;
  `tests/unit/test_pipelines.py` gained the stage-option integration test above. `grep -rn "RRP_KINFEAT\|kinfeat\.enabled"
  src/` returns only historical comments (no live reads). `pytest tests/unit -q` (see merge log for the exit code
  recorded at merge time); `tests/data/golden.json` untouched.

## resume steps (if interrupted before merge)

1. `cd ~/work/rrp-wt/rel-r12 && export PYTHONPATH=$PWD/src:$PWD`
2. `pytest tests/unit -q` -- must exit 0 before merging (see AGENTS.md / docs/relations.md section 10 rules).
3. `git fetch origin && git rebase origin/main`; rerun the unit suite if the rebase brought new commits.
4. `git push origin HEAD:main` (retry fetch/rebase/test/push up to 5x on a race).
