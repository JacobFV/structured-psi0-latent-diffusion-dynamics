# track rel-r8: StateView — Warp, ComputerWorld, SIMPLE (D-144 fanout row R8)

Owner: fanout unit R8 (`docs/relations.md` section 10). Deps: F only (foundation `c5e7d45`). Status: **verified**.
Worktree `~/work/rrp-wt/rel-r8`, branch `track/rel-r8`. Host: git / editing / unit suite only (CUDA hidden); no
training or simulation beyond unit-test fixtures.

## what changed

Implements `env.state_view()` (`rrp.envs.base.StateView`, capability `privileged_truth`, docs/relations.md 5.1) for
the three backends R7 does not own:

- **`src/rrp/envs/computerworld.py`**: `cw_state_view(scene, frame, slots, t)` (pure builder) + `CWStateView` +
  `ComputerWorldEnv.state_view()`. Entities are widgets (one `EntityState` per occupied slot, id `slot{n}:{key}`
  matching the slot `descriptors()` already uses); `ui_tree()` (`caps={"poses","ui_tree"}`) gives node/parent/z/
  focus_rank/role/label/bounds; `contacts()` / `joints()` / `camera()` raise `CapabilityError` (a 2D UI world has
  none). `token_entity("widgets", slot)` returns the entity id at that slot.
- **`src/rrp/envs/simple/__init__.py`**: `simple_state_view(truth)` (pure builder over the worker's `_sim_truth()`
  dict) + `SimpleStateView` + `SimpleEnv.state_view()`. Entities: `pelvis` (pos+quat), `palm:{left,right}`
  (pos only — SIMPLE's truth has no palm orientation), `object:{name}` (`attrs.is_target` flags `target_name`).
  `contacts()` (`caps={"poses","contacts"}`) from the hand×object boolean grid, one `ContactState` per active pair;
  `normal` is zeros (SIMPLE never reports a contact normal, only a boolean + mean point — left unfabricated).
  `joints()` / `camera()` / `ui_tree()` raise `CapabilityError`. `time = step / CONTROL_HZ` (SIMPLE's truth dict has no
  `time` field, only `step`; matches the observation path's `steps * 0.02`).
- **`src/rrp/envs/warp/tracker_env.py`**: `warp_state_view_from_arrays(m, xpos, c_geom, c_pos, c_world, index, t)`
  (pure builder over host numpy: `m` a `mujoco.MjModel` from `rrp.envs.warp.model.build_model`, no CUDA / mujoco_warp
  needed to build or read it) + `WarpStateView` + `WarpTrackerEnv.state_view(index)` (converts its torch tensors to
  numpy and calls the pure builder for that one world). Entities are every body of `m`, positioned by `xpos[index]`;
  contacts are the pool rows with `c_world == index` (a `-1` geom marks an unused slot); `caps={"poses","contacts"}`.
  `WarpTrackerEnv` hard-codes `self.dev = torch.device("cuda")` (pre-existing, not touched), so there is no way to
  construct a live instance on a CUDA-hidden host — only the pure builder is host-testable, exactly as R8's brief
  anticipates ("Warp on a 2-env CPU batch if available, else skip-marked"). `joints()` / `camera()` / `ui_tree()`
  raise `CapabilityError` (no joint data is read from the warp step today; adding it is future catalog work, not R8).

None of `relations/base.py`, `relations/ops.py`, or any file outside the three listed above were touched.

## tests

`tests/unit/test_state_view_envs.py` (new file — see "file not in the row's `owns` list" below):
- `test_cw_state_view_on_fixture_scene`, `test_cw_state_view_slots_are_stable_across_calls`: inline scene (same shape
  as `test_computerworld.py`'s `SCENE`), no `computerworld` wheel needed.
- `test_simple_state_view_on_recorded_truth_dict`, `test_simple_state_view_handles_missing_contact_point`,
  `test_simple_env_declares_state_view`: a synthetic dict matching `Worker._sim_truth()`'s exact keys/shapes, no
  Isaac / SIMPLE venv.
- `test_warp_state_view_on_two_env_cpu_batch`: `mujoco` + procedural `hexapod()` (no menagerie assets) host model,
  synthetic 2-world `xpos` / contact-pool arrays; skip-marked if the host model can't be built. Verifies per-world
  filtering (world 0 reads world 0's `xpos`, not world 1's; contacts split 2/1 across worlds; the `-1` slot is
  dropped).
- `test_warp_tracker_env_state_view_live` (`skipif(True, ...)`) + `test_warp_tracker_env_declares_state_view`:
  documents why a live `WarpTrackerEnv` contract test isn't possible here (hard-coded CUDA device) and instead checks
  the method exists.
- Every `StateView` built is checked with `isinstance(sv, rrp.envs.base.StateView)` (the protocol is
  `runtime_checkable`).

Red -> green checked directly: `git stash` the three source files, re-run the test file (collection `ImportError:
cannot import name 'cw_state_view'`), `git stash pop`, green again.

```
cd ~/work/rrp-wt/rel-r8 && export PYTHONPATH=$PWD/src:$PWD
~/work/relational-robot-policy/.venv/bin/python -m pytest tests/unit/test_state_view_envs.py -q
# 7 passed, 1 skipped
~/work/relational-robot-policy/.venv/bin/python -m pytest tests/unit -q
# see STATUS below for the exact count / exit code at merge time
```

## a note on the row's `owns` column (not blocking)

`docs/relations.md` section 10's table lists R8's owned files as exactly `envs/warp/tracker_env.py`,
`envs/computerworld.py` (`state_view` only), `envs/simple/__init__.py` (`state_view`) — no test file (unlike R7,
which explicitly owns `tests/unit/test_state_view.py`). Since R8's acceptance criteria are contract *tests* and the
top-level rules also require red/green tests, I added one new file, `tests/unit/test_state_view_envs.py`, deliberately
named to not collide with R7's `tests/unit/test_state_view.py`. This isn't an operator/interface change under
`relations/base.py` / `ops.py`, so it doesn't meet the "stop and ask" bar — flagged here (and in the merge summary)
as an FYI for the lead in case the table should be corrected for the remaining rows that have the same gap (R15-R18).

## commands run (host only; no peer smoke needed — R8 has no peer-smoke acceptance item)

```
cd ~/work/relational-robot-policy && git worktree add ~/work/rrp-wt/rel-r8 -b track/rel-r8 origin/main
cd ~/work/rrp-wt/rel-r8 && export PYTHONPATH=$PWD/src:$PWD
~/work/relational-robot-policy/.venv/bin/python -m pytest tests/unit -q   # full suite before merge
```

## D-144 addendum (lead decisions, recorded by this unit -- first to need them)

No prior D-144 addendum existed in `research/decisions.md` at merge time, so this unit added one with the lead's
three merge-protocol decisions (readout-probe swap acceptance criteria; the "gone from `src/`" definition; the
`~/work/rrp-data/main-merge.lock` serialization) so R4/R5/R6 and siblings needing "gone from src/" can cite it
without re-deriving it.

## merge

Committed the `StateView` feature (`493eded6b`) and the D-144 addendum (`afc0e553`) in the worktree, then ran the
documented `flock ~/work/rrp-data/main-merge.lock bash -c '...'` merge command exactly once. `git fetch origin &&
git rebase origin/main` rebased cleanly onto `566fccc6` (R11's merge) with no conflicts and no new commits landing
between fetch and rebase (no siblings merged concurrently during this window), so the full-suite re-run and push
happened on the first attempt -- no push race.

Final full-suite result before the push: `789 passed, 40 skipped, 0 failed` (exit code 0). Pushed to `origin/main`
at commit `e622bedae1eab1daab5fdc9d8fcd81d7aff3fd93` ("D-144 addendum: readout-probe swap acceptance, gone-from-src
definition, merge lock (R8 lead decisions)"), on top of `44ceba89` (the R8 feature commit itself, `493eded6b`
rebased).
