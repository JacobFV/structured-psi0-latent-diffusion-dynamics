# track: rel-r7 (D-144 relation-factor fan-out, unit R7: StateView on the MuJoCo sessions)

Worktree `~/work/rrp-wt/rel-r7`, branch `track/rel-r7`, from `origin/main` (c5e7d45, D-144 foundation). Brief:
`docs/relations.md` section 10, row R7 / briefs. Deps: F (foundation) only, merged. Host only: no CUDA, no
training/simulation beyond unit-test fixtures. All MuJoCo sessions here are CPU physics only; `render_depth` itself
is never invoked on host (docs/relations.md 5.1: "EGL only on the peer — host tests use the camera math only" —
this host has no GPU-free GL backend available either: no `libOSMesa`, and the default backend absent `MUJOCO_GL`
is `glfw`, which would open a real GLX/GPU context if one ran it here). The `render_depth` shape acceptance
criterion is met by faking `mujoco.Renderer` in the test (checks the declared-size plumbing only); a fix applied
on resume after finding the original test called the real renderer (see "resume fix" below).
No peer smoke: the brief does not ask for one for this unit.

## resume fix (2026-09-29, continuing this worktree)
The commit already on this branch (`6976aa1`) had `test_render_depth_shape_matches_declared_size` call
`sv.render_depth("front")` for real. That only passed because this shell happened to have `DISPLAY` set, so
`mujoco.Renderer` opened a real GLX context against the host GPU — exactly the host GPU/EGL rendering
docs/relations.md 5.1 reserves for the peer, and it would have failed outright on a true headless host (no
`libOSMesa` in this venv). Rewrote the test to monkeypatch `rrp.envs.mujoco.session.mujoco.Renderer` with a
fake that records `(height, width)` / `enable_depth_rendering` / `update_scene(camera=...)` and returns a
correctly-shaped array, so the declared-size plumbing is checked without ever constructing a GL context on host.
No production code changed for this fix, only the test.

## what changed
- `src/rrp/envs/mujoco/session.py`: `Session.state_view()` returns `MujocoStateView`, gated by the
  already-declared `"privileged_truth"` capability (same gate as `truth()`). `MujocoStateView` implements
  `rrp.envs.base.StateView` (`caps`, `time`, `gravity`, `entities`, `contacts`, `joints`, `camera`, `ui_tree`,
  `token_entity`), plus one extra capability-gated method beyond the protocol, `render_depth(name)` (`mujoco.Renderer`
  depth mode; capability `"depth_render"`). `caps = {"poses", "velocities", "contacts", "forces", "joints",
  "camera", "depth_render"}`; `ui_tree()` raises `CapabilityError` (MuJoCo has none).
  - `entities()`: one `EntityState` per scene object/feature (id = the task entity id when the object is bound,
    else `obj:<sim_body>`; true `xpos`/`xquat`/`cvel`, mass, first-geom friction) + `Session._manip_entities()`
    (one per bound manipulator assembly, id = the scenario's public manipulator entity key, e.g. `"gripper"` /
    `"left"` / `"right"` / `"body"`; true TCP site pose) + `Session._extra_state_entities()` (empty at this level;
    `LeggedSession` overrides it).
  - `contacts()`: one `ContactState` per `mjData` contact, geom pair resolved to entity ids via
    `Session._body_entity_map()` (objects + assembly member links, falling back to the raw mj body name when
    unmapped, e.g. `"world"`/ground); world normal = `contact.frame[:3]` (same source `Session.truth()` already
    uses); world force = `frame.T @ mj_contactForce(...)[:3]`.
  - `joints()`: one `JointState` per robot's hinge/slide joints (the same `RobotRuntime.joint_names/qadr/dadr`
    the rest of `Session` already uses), `JointSpec` supplies parent/child/axis/range.
  - `camera()`: `K` from `cam_fovy` + the requested (width, height) (`state_view(depth_width=, depth_height=)`,
    default 64x64), `T_world_cam` from `cam_xpos`/`cam_xmat`. `render_depth()` builds a `mujoco.Renderer` lazily.
  - `token_entity(token_set, slot)`: `slot` is the detector's track index (`self.detectables[slot]`); returns its
    entity id or `None` out of range / non-int (never raises).
  - New hooks on `Session` for subclasses to extend without duplicating the object/contact logic:
    `_manip_entities`, `_extra_state_entities`, `_body_entity_map`, `_extra_body_entity_map`.
- `src/rrp/envs/mujoco/dual.py`: `DualSession._manip_entities()` overrides the generic one with one entity per
  `ManipHandle` (still the true simulator TCP pose, never the public `_fk_site` estimate used elsewhere in this
  file for `tcp_pose`), enriched with `gripper_kind` / `reach_m` in `attrs`.
- `src/rrp/envs/mujoco/legged.py`: `LeggedSession._extra_state_entities()` adds one `EntityState` per foot link
  (`id="leg:<mj body name>"`, `kind="link"`, `parent="body"`); `_extra_body_entity_map()` resolves foot-floor
  contacts to those ids instead of the raw mj body name fallback.
- `tests/unit/test_state_view.py` (new; row-owned): contract tests on arm (`Session` + `build_pick_place`,
  `parm6_pg2`), dual (`DualSession` + `build_support_insert`, the same fixture as
  `test_dual_public_estimators.py`) and legged (`LeggedSession` + `build_waypoint_contact("pquad4", ...)`, the
  same fixture as `test_d126_legged.py`) fixtures.

## acceptance evidence (row R7)
- **entities with poses**: `test_arm_entities_have_correct_poses_and_ids`, `test_dual_entities_use_manip_handle_metadata`,
  `test_legged_entities_include_foot_links` — ids match the scenario's public entity keys (`"cube"`, `"target"`,
  `"gripper"` / `"left"`/`"right"` / `"body"` / `"leg:<body>"`); poses checked against `mjData` bit-for-bit
  (`atol=1e-9`).
- **contacts with normals**: `test_arm_cube_settled_on_table_has_upward_world_contact`,
  `test_legged_feet_contact_floor_with_upward_normal_when_standing` (unit normal, upward-supporting force on a
  settled/standing fixture) + `test_contact_force_direction_matches_truth_normal_force` (cross-checks against
  `Session.truth()`'s own `mj_contactForce` use — catches a wrong frame transpose).
- **joints**: `test_joints_match_qpos_qvel` (parametrized arm/dual/legged) — `q`/`qd` equal the live
  `data.qpos`/`data.qvel` at the `RobotRuntime` addresses the rest of the session already uses.
- **camera K/T**: `test_camera_intrinsics_and_extrinsics` (parametrized) — `K` reproduces the `cam_fovy` formula at
  a non-square (32, 48) size, `T_world_cam` equals `cam_xpos`/`cam_xmat` exactly; unknown camera name raises
  `KeyError`.
- **`render_depth` shape**: `test_render_depth_shape_matches_declared_size` — `mujoco.Renderer` faked (never
  constructs a real GL context on host, see "resume fix" above); checks `render_depth` passes the declared
  `(height, width)` into `Renderer(model, height, width)`, calls `enable_depth_rendering()`, forwards the
  camera name to `update_scene`, and returns the renderer's array unchanged (shape (16, 24) for a
  `depth_width=24, depth_height=16` view).
- **caps declared**: `test_caps_declared_and_protocol_satisfied` — `isinstance(sv, StateView)` (the protocol is
  `runtime_checkable`), expected caps present, `ui_tree()` raises `CapabilityError`.
- **views never reachable from the featurizer**: `test_state_view_never_reachable_from_the_featurizer` — greps
  `policies/features`, `policies/nets`, `policies/bundles.py` for `state_view`/`StateView` and asserts no hits.

## rules followed
- Never touched `relations/base.py` / `relations/ops.py` (not imported, not edited; no operator/interface change
  was needed for this unit).
- `tests/data/golden.json` untouched (not read or written by anything in this unit).
- No shims, no files beyond `envs/mujoco/{session,dual,legged}.py` + `tests/unit/test_state_view.py` + this note.
- `_manip_entities` / `_extra_state_entities` / `_body_entity_map` / `_extra_body_entity_map` are new hook points
  on `Session`, overridden (not duplicated) by `DualSession`/`LeggedSession` — the shared-file discipline the
  fanout plan asks for ("shared files are shared by disjoint functions only").

## commands
- `export PYTHONPATH=$PWD/src:$PWD && ~/work/relational-robot-policy/.venv/bin/python -m pytest tests/unit/test_state_view.py -q`
- Full suite before merge: `pytest tests/unit -q` (see STATUS below for the run this unit used).
