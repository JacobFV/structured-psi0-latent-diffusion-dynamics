# track: rel-r19 (D-144 relation-factor fan-out, unit R19: legged labels + foothold)

Worktree `~/work/rrp-wt/rel-r19`, branch `track/rel-r19`, from `origin/main`. Brief: `docs/relations.md` section 10,
row R19 / briefs (line 680-682): "Legged labels (next foothold cell, per-foot contact, COM projection inside the
support polygon), entries `leg.foothold` / `leg.com_support`, part `terrain_steps`; move the Warp height scan out
of the deployable `vec` into a label block (privileged layout test)." Deps: R4 (legged nets, merged), R7 (StateView:
MuJoCo, merged). Host only: `CUDA_VISIBLE_DEVICES=""`, no training / simulation beyond unit-test fixtures; no test
in this unit touches CUDA or `mujoco_warp` (see "WarpTrackerEnv on host" below). Never edited `relations/base.py` /
`relations/ops.py`.

On picking this worktree up, `src/rrp/envs/warp/task_env.py` (modified) and `src/rrp/harness/data/relgen/body.py`
(new, untracked) already carried a substantially complete draft of the privileged-layout fix and the label family
from an earlier session in this worktree; this session reviewed both against the actual base-class contracts
(`WarpTrackerEnv.observe` / `.privileged` / `.priv_dim`, `rrp.envs.base.StateView` / `EntityState` / `ContactState`,
`rrp.harness.data.relgen`'s `LabelDef` / `ScenePart` registries), fixed one inaccuracy found in `body.py`'s module
docstring and an inline comment (see "what changed" below), added the `catalog.py` §legged entries the draft's own
docstrings already assumed (`ReadoutDef.address="asm"` etc.), and wrote the test suite.

## what changed
- `src/rrp/policies/relations/catalog.py` (`R19: legged` section only): two `FactorDef` entries + a preset.
  - `leg.foothold`: `field="hidden"`, `op="bilinear"`, `form="aug"`, `algebra=Algebra(arity=2, direction="directed",
    value="prob", dynamic=True)`, `sources=("probe", "gt")`, `label="foothold_next"`, `gen=("terrain_steps",)`,
    `readout=ReadoutDef("foothold", "pair", 1, "bce", label="foothold_next", reads="hidden")`, `params={"rank": 8}`
    -- the same bilinear-pair pattern as R16's `ix.*` / R17's `ix.support` (no public/estimated "given" source for
    which terrain cell a swinging foot lands on next; `probe` deployable, `gt` training/diagnostics-only, deploy
    guard tested).
  - `leg.com_support`: `field="packet"`, `op="inert"`, `form="readout"`, `label="com_support"`,
    `readout=ReadoutDef("com_support", "asm", 1, "gauss", label="com_support")` -- same shape as `probe.legged.*`
    (R4's packet-probe entries): one scalar read at the sample's body-assembly row, not a pairwise bias (there is
    no second token for "the robot's own stability margin" to attend over). Default `sources=("given",)`
    (`FactorDef`'s default), matching `probe.legged.*`'s own default -- never privileged at the registry level since
    an `inert`/`readout` factor never resolves a field through `effective_source` in the forward pass either way.
  - `register_preset("legged-r19", ["leg.foothold", "leg.com_support"])` for convenience (not required by the brief,
    matching R16's `ix` / R17-adjacent presets' own convenience-preset habit).
- `src/rrp/harness/data/relgen/body.py` (new; already drafted, reviewed and two doc-accuracy fixes applied, see
  below): `foot_ids` / `stance_feet` helpers; label `foot_contacts` (arity 1, per-foot stance bit; internal building
  block, no `catalog.py` entry owns it directly -- same status as R17's `support_matrix`); label `foothold_next`
  (arity 2, `leg.foothold`'s supervision: swinging foot -> nearest `foothold_cell` entity by planar distance,
  planted feet never a true pair, degrades to all-invalid when no backend populates candidate cells); `_convex_hull`
  / `_point_segment_distance` / `_point_in_polygon` / `support_polygon_margin` (signed planar margin of a point
  inside a point set's convex hull: positive inside, negative outside, `-inf` for zero feet); label `com_support`
  (arity 1, `leg.com_support`'s supervision, written only on the `"body"` assembly slot); scene part `terrain_steps`
  (`SCAN_AHEAD x SCAN_SIDE` = 4 x 3 candidate `foothold_cell` entities on a staircase ahead of the robot, resampled
  step heights; `activates={"terrain", "foothold", "com_support"}`; `vary("height")` resamples every row's height,
  holding cell count / xy layout / ids fixed -- the decoupling pair `com_support` / `foothold_next` need to toggle
  without any other scene change confounding it).
- `src/rrp/envs/warp/task_env.py` (already drafted, reviewed, unchanged from the draft): `_layout_dims(base_obs_dim,
  base_priv_dim, extra_dim) -> (obs_dim, priv_dim)` -- pure arithmetic (no CUDA needed to test it): `obs_dim` stays
  the tracker's public width, UNCHANGED by the task's extra block; `extra_dim` widens `priv_dim` instead.
  `WarpStepsEnv` / `WarpGapEnv` no longer override `observe()` (they inherit `WarpTrackerEnv.observe`, already the
  full deployable vec); `privileged(fc)` is overridden to `cat([super().privileged(fc), self.extra_obs()], -1)`.
  This is a real behavior change for anyone who *trains* a new W13 task expert after this merge: the actor net's
  input width shrinks by `extra_dim` (no more height scan / gap geometry baked into the policy's own forward pass),
  matching the "trained with a privileged critic, deployed without it" invariant every other tracker in this repo
  already follows (`rrp.envs.mujoco.legged_tracker` module docstring) and AGENTS.md's "simulator ground truth may
  not silently enter deployable observations". Downstream effect verified, not touched (out of this unit's owned
  files, `harness/train/warp_tracker_ppo.py` is a shared trainer script): `meta["extra_obs_dim"] =
  int(getattr(env, "extra_dim", env.obs_dim - env.b.obs_dim))` at line ~204 now naturally computes `0` for a
  from-scratch W13 run (since `env.obs_dim == env.b.obs_dim` post-fix) -- correct, not stale: the actor genuinely
  gets zero extra input width now. `rrp.envs.mujoco.legged_tracker.LearnedTracker`'s `extra_fn` / `TrackerMismatch`
  machinery (checks `meta["obs_dim"] != binding.obs_dim + extra_obs_dim`) is untouched and still correct for BOTH
  old checkpoints (`extra_obs_dim > 0`, pre-R19 actors that really did read the scan) and new ones (`== 0`) -- the
  "on-disk legacy remap" shape D-144's addendum decision (b) describes, just via a numeric field instead of a name
  table. One stale artifact left alone (documentation only, no functional effect, flagged for the lead below):
  `warp_tracker_ppo.py`'s `meta["extra_obs"]` display string still reads `"...(expert only)"` for `--task steps`/
  `gap` regardless of `extra_obs_dim`, which is now always paired with `extra_obs_dim=0` for a fresh run -- mildly
  misleading provenance text, not a loaded value; did not touch it since it is outside this unit's owned files.
- **Doc-accuracy fixes to the draft** (found while reviewing, not new design): `body.py`'s module docstring
  originally claimed `terrain_steps` declares "the SAME fixed body-frame scan geometry `WarpStepsEnv.extra_obs`
  computes privately (11 x 3 cells ...)"; the actual `SCAN_AHEAD` / `SCAN_SIDE` constants are 4 x 3 with different
  offsets than `envs/mujoco/humanoid_scenes.SCAN_X` / `SCAN_Y` (11 x 3), and the very next inline comment on
  `SCAN_AHEAD` already said "not imported from there (backend-agnostic)" -- a direct self-contradiction. Reworded
  both the module docstring and the inline comment to state the true relationship: same body-frame CONVENTION
  (ahead of / beside the body, multiples of leg length `L`), deliberately different, independent numbers; a MuJoCo
  legged fixture has no Warp staircase model to match cell-for-cell. No code changed, only the two comments; see
  `test_warp_steps_extra_obs_width_matches_scan_grid` (asserts the Warp grid really is 11 x 3 + 1 = 34) alongside
  `test_terrain_steps_build_declares_grid_and_activates` (asserts the `terrain_steps` grid really is `len(SCAN_AHEAD)
  x len(SCAN_SIDE)` = 12) as the record that these are two different, intentionally independent numbers.
- `tests/unit/test_relations_r19.py` (new, 26 cases): a hand-built quadruped `StateView` fixture (3 stance feet, one
  swing foot) proves `stance_feet` / `foot_ids` / `foot_contacts_fn` / `foothold_next_fn` (including "no candidate
  cells -> all invalid", "swing foot pairs with the nearest cell", "stance feet never a true pair") and
  `com_support_fn` (written only on `"body"`, matches `support_polygon_margin` by hand, `-inf`/no-stance-feet fixed
  fallback, missing body entity -> all invalid); `support_polygon_margin` geometry cases (0/1/inside/outside);
  `terrain_steps_build` / `.vary("height")` (grid size, `activates`, provenance, vary touches only z, rejects an
  unknown factor / an un-built draft); registry resolution + shape assertions + the deploy guard for both `leg.*`
  entries and the `legged-r19` preset; the privileged-layout test (`_layout_dims` pure arithmetic,
  `WarpStepsEnv`/`WarpGapEnv` class-attribute inspection proving `observe` is not overridden and `privileged` is,
  the Warp scan grid size assertion above).

`pytest tests/unit -q`: 827 passed, 40 skipped (pre-existing menagerie/CUDA/optional-extra skips; one new skip
already existed before this unit, `test_state_view_envs.py`'s documented "WarpTrackerEnv itself always needs
CUDA" skip -- this unit added no new skips), exit 0. Ran with `CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD/src:$PWD
~/work/relational-robot-policy/.venv/bin/python` per the merge protocol.

## WarpTrackerEnv on host
Same situation R8's track already documented: `WarpTrackerEnv.__init__` hard-codes `self.dev` to a CUDA device, so
no test in this unit constructs a real `WarpStepsEnv` / `WarpGapEnv`. The privileged-layout test instead exercises
`_layout_dims` directly (pure arithmetic, exactly what its own docstring says it is for) and inspects
`WarpStepsEnv.__dict__` / `WarpGapEnv.__dict__` to prove `observe` is inherited unchanged and `privileged` is
overridden -- both classes import cleanly on host (their module-level imports are `mujoco` + `torch`, no
`mujoco_warp`), so this needs no simulator and no skip.

## lead_questions
- `harness/train/warp_tracker_ppo.py`'s `meta["extra_obs"]` display string (not `extra_obs_dim`, which is correct)
  still describes the height scan / gap geometry as reaching the actor ("expert only" implying "fed to the net"):
  worth a one-line wording fix by whoever next touches that file, to stop implying the new actor's own forward pass
  reads it. Not blocking; not in this unit's owned files (`envs/warp/*`, not `harness/train/*`).
- No live env populates `foothold_cell` / terrain-cap entities for `foothold_next` yet (`terrain_steps` is
  declarative-only, matching R16/R17's own parts before a live env's scenario builder is wired to `compose`, R11).
  Same open item R16/R17 already flagged: wiring a scene part's declared entities into an actual running env's
  `state_view()` is a later (R10/R11-adjacent) unit's job, not this row's.
