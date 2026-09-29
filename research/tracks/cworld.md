# track cworld: ComputerWorld as an rrp environment (D-140)

Owner decision D-140; design `docs/architecture.md` section 5. Status: **prototype verified (pre-S3)**. It lives in
`proto/cworld/` until S3 lands. After that it moves into `rrp.envs.computerworld`, `rrp.tasks` (cw/*),
`rrp.policies.teachers` and `rrp.bodies`, and `proto/` is deleted.

## install (exact; no Rust build needed)
PyPI publishes an aarch64 abi3 wheel for 0.2.0, so no Rust build is needed on either machine.
```sh
mkdir -p ~/work/ext && git clone https://github.com/JacobFV/computerworld ~/work/ext/computerworld   # read-only reference (HEAD 35753083)
python3 -m pip download computerworld==0.2.0 --no-deps -d ~/work/ext/wheels
#   computerworld-0.2.0-cp39-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
#   sha256 b6360952f6ec2c54921ccc6d71d99577842dc30c01b482de33849e976b748b8a (40 MB; never committed)
uv pip install --python ~/work/relational-robot-policy/.venv/bin/python --target ~/work/ext/cw-site --no-deps \
    ~/work/ext/wheels/computerworld-0.2.0-*.whl
PYTHONPATH=~/work/ext/cw-site python -c "import computerworld as c; print(c.__version__, c.engine_version)"   # 0.2.0 0.2.0
```
The shared host `.venv` is not modified: the wheel sits in its own `--target` dir and is added to `PYTHONPATH`. At S3 it
becomes the optional extra `computerworld = ["computerworld==0.2.0"]`. Pin the exact version. CW snapshots and pixels
are only valid within one engine version, and a git-HEAD build (0.2.0+246 commits) is a different engine from the wheel.
The world definitions under `worlds/` at HEAD are also not guaranteed to load in 0.2.0. Our env builds its own
one-machine world inline, so it does not use them.

## run the tests
```sh
cd ~/work/rrp-wt/cworld
PYTHONPATH=$PWD/src:$PWD:$PWD/proto/cworld:$HOME/work/ext/cw-site CUDA_VISIBLE_DEVICES= \
  ~/work/relational-robot-policy/.venv/bin/python -m pytest -q proto/cworld/test_cworld.py -p no:cacheprovider   # 20 passed
```
Without the wheel, the 7 pure tests still run: mapping, transform, depth modes, occlusion, null slots, button edges and
body. The other 13 skip.

## what is there (`proto/cworld/cworld.py`, one file)
- `ScreenFrame`: at 1 mm/px, (u, v) maps to ((u-W/2)s, (H/2-v)s) with z toward the viewer. Default viewport 960x640,
  the CW console default. `depth="stack"` (default) sets z = dense rank of the scene's z-layers × dz (2 mm), and
  `depth="constant"` sets z = 0.
- `scene_widgets` / `SlotRegistry` / `descriptors`: every semantic node becomes an `ObjectDescriptor` (descriptor =
  role; attributes are label/value/state/disabled/window/interaction once S3 adds the field). Occluded nodes are kept
  with `visible=False`: a node counts as occluded when a higher opaque box/rounded_box/backdrop/image fully covers its
  clipped box. Slots stay stable within an episode. A vanished widget leaves a `descriptor="null"` slot, and new
  widgets are appended. Task bindings set `bound_entity` (e.g. `textbox:Name` → `field_name`).
- `cw_pointer` RobotSpec: slide x, y (viewport range) and slide z fixed at hover (0.05 m). qpos = pointer (x, y, z).
- Action groups at 10 Hz:
  - `pointer`: cartesian_position, 2-wide, m, absolute.
  - `button`: ≥ 0.5 is down; only edges emit `down`/`up`, so a click takes two ticks.
  - `wheel`: notches × 120 px.
  - `key`: discrete over `KEY_VOCAB` (14 named keys + 95 printable characters; -1 = none).
  - Within a tick the order is move, button, wheel, key.
  - Rejections are returned as codes: `bad_group:*`, `key_out_of_vocab`, `cw_refused:<family>.<op>:<code>`.
- Tasks (judge → outcome + failure_reason):

  | task | setup | success | failure / timeout reasons |
  |---|---|---|---|
  | `cw/calc_sum` | calculator open | history shows `a + b = a+b` | `wrong_result`, `no_result` |
  | `cw/open_type` | empty desktop | editor text == word | `wrong_text`, `app_not_open`, `incomplete_text` |
  | `cw/drag_window` | calculator open | title bar moved by (dx, dy) ± 6 px, button up | `window_closed`, `off_target` |
  | `cw/fill_form` | browser on the simulated `form.internal` static site | submitted query == name/email | `wrong_value:<field>`, `not_submitted` |

  Drag goals are sampled from the feasible range: CW clamps windows below the 32 px top bar and right of the 64 px dock.
- `ScriptedTeacher(task)`: source `scripted_teacher`, privileged (it reads the full scene through the env). It moves the
  pointer at 60 px/tick toward the widget's centre, then clicks, drags or types one symbol per tick.
- `truth()` gives every widget's pose, including occluded ones, plus the goal and the CW `state_hash`. `reset(seed)`
  restores a cached per-seed initial snapshot, so it is deterministic. `snapshot`/`restore` wrap `world.snapshot()`
  and also carry pointer/slot state. Provenance records the engine version, package version, adapter version, world
  digest, theme and viewport.

## findings about CW 0.2.0 that correct the design doc
1. **Scene node ids are not stable across revisions.** The Files launcher gets a new id after a window opens, and CW's
   own docs say not to retain node ids across layout changes. Slot identity is therefore keyed on
   `interaction|role|label` plus an occurrence index.
2. There is no explicit `state` in the node semantics (only role/label/value/disabled/focusable). The "focused" state
   comes from `scene.focus.interaction`. The Text Editor's value appears only in the `semantic.v1` observation
   (`editor-text`), not in the scene node, so the judge reads the observation.
3. Stack depth ranks z-layers rather than nodes, because a desktop has ~100 nodes but ~5–10 layers. Within a layer,
   occlusion follows insertion order, as CW's hit test does.
4. Every themed-desktop GUI action needs `application.v1`, and browser content needs `browser.v1`. The mouse pointer
   is the last scene node (z 2e6, no semantics) and is never an occluder.

## smoke evidence (host, ~2 s total, vibe-check scale)
Scripted teacher, seeds 0–19: calc_sum 20/20, open_type 20/20, fill_form 20/20. drag_window: 50/50 on seeds 0–49
after the feasibility fix, 16/20 before it (the 4 failures were targets behind the dock). The idle policy times out on
every task with the reasons above. Speed: `scene()` ~1 ms, `step` ~0.7 ms, snapshot+restore ~0.01 ms, a 640x480 frame
~85 ms.

## after S3 (resume steps)
1. `git fetch origin && git rebase origin/main`. Then:
   - Move the env into `src/rrp/envs/computerworld.py` (drop the stand-in types; import them from `rrp.envs.base`).
   - Move the tasks into the `rrp.tasks` registry, the teacher into `rrp.policies.teachers.computerworld`
     (a `Policy` with `requires.privileged`) and `cw_pointer` into `rrp.bodies`.
   - Move the tests into `tests/unit/test_computerworld.py`.
2. Use the real `ObjectDescriptor.attributes`, `ImageObs` `rgba8`, `Observation.instruction` and RobotSpec family
   `pointer`, and remove the `model_fields` fallbacks.
3. Add the `computerworld==0.2.0` extra to pyproject.
4. `rrp matrix`: show arm/legged policies declined with reasons and a pointer policy accepted. Run the unit suite and
   check its exit code. Merge.
5. Demo video (peer, not host: D-115): one teacher success and one failure per task, labelled scripted_teacher.
