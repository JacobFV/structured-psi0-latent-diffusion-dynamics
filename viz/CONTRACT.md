# rrp live visualization room: data contract (v1)

Owner request (2026-09-28, D-131): "build a comprehensive live visualization room like we did for ../ibm-2 … i want to see
everything about our policy trained and evaled on all the tasks". Pattern mirrors IBM-2's workbench: a local web app, a
read-only local data API with a short cache, and a static snapshot fallback that says it is a snapshot.

## ground rules
- Listens on port 3013; the bind host is configurable with `RRP_ROOM_HOST` (default 127.0.0.1; set 0.0.0.0 to view it from other machines; public listeners are allowed per D-132). The room stays read-only.
- **Read-only.** The room never starts, stops or modifies jobs, experiments or files outside `viz/data/`
  and `~/work/rrp-data/viz/`.
- **Host-light (D-127).** The host exporter is pure file/JSON work: ≤ 1 CPU, ≤ 500 MB, runs at most every 15 s
  (live ops every 10 s), and does no simulation, no torch and no mujoco. Anything that simulates (replay recording) runs on the
  PEER through the broker (`scripts/peer_run.sh`, memory ≥ 1.35 × peak).
- **Honest labels everywhere.** Every number shows its source file, its decision id and a source label
  (scripted_teacher / oracle / learned:<ckpt> / bc:<ckpt> / learned_tracker:<body>:<ver>). Caveats (e.g. t1 D-113, grasp_v1)
  are shown, not hidden. Interim results are marked interim. There are no invented progress percentages and no fake live
  activity. A snapshot says "snapshot as of <time>".
- **Public repo.** Room source is committed. Generated data (`viz/data/`), replays, videos, weights and menagerie meshes
  are NOT committed (gitignored). The API serves them locally.

## layout
- `viz/room/`: frontend (Vite + React + TypeScript + three.js + recharts + tailwind). `npm run dev` serves on 127.0.0.1:3013
  with the Vite API plugin; `npm run build` produces a static build; `npm run snapshot` exports `viz/data` for offline use.
- `src/rrp/viz/export/` (a package: check `src/rrp/viz/export/__init__.py`, not `export.py`): `python -m rrp.cli viz export [--live] --out viz/data [--only <name>]`. Builds the JSON documents below from the repo,
  the local artifact copies and (with --live) small ssh reads of the peer broker, watchdog and ledgers.
- `src/rrp/viz/record.py`: PEER-ONLY replay recorder: `python -m rrp.cli viz record --spec <yaml> --out <dir>`. Writes replay
  files (below) for chosen episodes (task × body × route × seed × condition).
- `~/work/rrp-data/viz/replays/`: replay files pulled from the peer (rsync), served by the API.

## API (served by the Vite plugin; the same paths exist as static files in a snapshot)
All JSON. Every document has `{schema: "rrp-viz/<name>/v1", generated_at, git_sha, sources: [paths]}`.

| path | content |
|---|---|
| `/api/overview` | headline claims: established (with D-id), open (roadmap #), caveats; key numbers table; latest 15 decisions |
| `/api/live` | peer node (GPU util/temp, CPU temp, memory available, PSI, project memory, disk free, admission state + reason), active leases (id, label, workstream, declared mem/GPU/CPU, measured current/peak, memory.high events, age), last 200 watchdog samples (level, reasons), host basics; `stale: true` if older than 60 s |
| `/api/dags` | every run-dag ledger (host + peer copies): dag name, nodes (id, stage, state, lease, started/ended, rc, caveat), counts, a stated ETA if recorded in track notes (never invented) |
| `/api/results` | normalized results catalogue: rows `{id, family: arm/legged/dual/psi0, task, body, route, variant, seed, grasp/contact/actuator version, metric, k, n, rate, ci_lo, ci_hi, source_label, source_file, decision, interim, caveat}` from every summary.json / compare table / gate report |
| `/api/edits` | causal edit effects: `{body, variant, seed, edit (ctx_halt / mirror_active / mirror_inactive / z_halt / z_turn / goal / rebind…), control?, effect, ci, n_pairs, permutation p, decision}` |
| `/api/training` | training curves index + series: `{run, kind (stageA / flow / refit / bc / tracker / grpo / psi1z), step[], losses{…}, grad_norm, clip_scale, lr, alpha (reward schedule), gate state}`, downsampled to ≤ 2000 points |
| `/api/robustness` | sweep tables: route × factor × level success with CIs, break-points, motion-quality medians, variant-level paired diffs (D-108/D-112) |
| `/api/physics` | contact v2 slip / CoT tables per tracker, grasp rig results, tracker validation, gate backfill verdicts (D-093..D-114) |
| `/api/psi0` | the Ψ₀ line: reproduction table, step-2 results, psi1z P-decisions and crosswalk (reads ~/work/psi1z and its local results copies) |
| `/api/knowledge` | decisions (parsed D-entries: id, date, title, body markdown), roadmap items, backlog items, strategy workstreams, STATUS markdown, docs list |
| `/api/replays` | replay index: `{id, family, task, body, route, source_label, variant, seed, condition, success, n_frames, fps, file, video?}` |
| `/api/replay/<id>` | a replay file (below) |
| `/api/videos` | `artifacts/video/INDEX.md` entries with labels; files served from `/media/<name>` |
| `/api/doc?path=` | allowlisted markdown document (docs/, research/, STATUS.md, README.md, AGENTS.md, psi1z README/notes/decisions) |
| `/api/radar` | declared route radar (`rrp-viz/radar/v1`) built from `viz/radar_axes.json`: axes (metric, direction, floor, reference, protocol, decision) × series (teacher, bc, semfix, nosem, frozen_sem) with value, r (floor → reference), drawn (clamped), spread and per-value evidence; missing values stay gaps with a reason |
| `/api/training/<id>` | one training series (`rrp-viz/training-series/v1`: step[], losses{…}, grad_norm, clip_scale, lr, alpha, gate_state); ids from `/api/training` `runs[].id` |

Provenance fields on every row (results, edits, physics, robustness, training runs): `source_file` (path relative to its checkout),
`location` (`repo` = this checkout, i.e. in git / `main` / `wt:<worktree>` / `rrp-data:<dir>` / `peer:<host>`), `in_git`, `sha1`,
`copies` (identical copies found; deduplicated by content hash), `alt_paths`, `decision` (latest D-entry whose text names the file or
its run directory), `decisions`, `decision_match` (the token that matched), `track_notes`. Results rows also carry `key_path`
(location inside the file), `body_from` / `route_from` / `source_label_from` / `version_from` (json | key_path | path | missing),
`body_hint` (a short arm name such as `panda` when the file does not give the gripper), `interim_reason`, `ci_method`
(`recorded` | `wilson95_computed`). Unknown values are null, never guessed.

## plugin interface (exporter ↔ Vite plugin)
- `viz/data/<name>.json` for every name above, `viz/data/training/<id>.json`, and `viz/data/_manifest.json`
  (`documents.<name>: {generated_at, bytes, rows, seconds, error}`, `last_run.timings`). Runs are serialized by a flock on
  `viz/data/_cache/lock`; per-source caches live in `viz/data/_cache/`. One filesystem walk is reused for 10 s, so per-document
  calls in a row are cheap (~0.1–0.5 s each warm); a full export takes ~1.5–3 s warm, ~9 s cold.
- The plugin shells out with cwd = repo root, `PYTHONPATH=src`, `OMP_NUM_THREADS=1`, `.venv/bin/python`, a 20 s timeout and a
  15 s cache (live: 10 s), like IBM-2's progress plugin:
  `python -m rrp.cli viz api get <name> [--live] [--max-age S]` prints the absolute path of the fresh document (it re-exports only that
  document when older than S); then the plugin reads that file. `/api/live` uses `get live --live` (one bounded ssh read of the
  peer, ≤ 5 s; `stale` when the last good read is older than 60 s).
- `python -m rrp.cli viz api doc <path>` prints `{schema: rrp-viz/doc/v1, path, markdown, mtime}` (exit 2 if not allowlisted; optional,
  the plugin may serve `/api/doc` itself with the same allowlist, v1.1);
  `api replay <id>`, `api media <name>`, `api training <id>` print an absolute file path (exit 2 if unknown). `api routes` prints the table.
- `python -m rrp.cli viz export --sync-psi1z` rsyncs small psi1z summary files from the peer into `~/work/rrp-data/viz/psi1z/` (≤ 50 MB);
  it is never part of a periodic refresh.

## replay file (`rrp-viz/replay/v1`, JSON; gzip allowed)
```
{schema, id, meta: {family, task, body, route, source_label, ckpt_sha, variant, seed, condition, success, failure_stage,
        physics: {contact_version, grasp_contact_version, actuator_limits_version, actuator_mode}, decision_refs: [...]},
 fps (≤ 30), n_frames,
 geoms: [ {name, body, type: plane|box|sphere|capsule|cylinder|ellipsoid|mesh, size: [...], rgba: [...],
           mesh?: {vertices: float[] (decimated ≤ 2000 faces), faces: int[]}} ],        # static geometry
 frames: { t: [...], body_pos: [[x,y,z]...per body...], body_quat: [[w,x,y,z]...] },    # per frame, per body (float, 4 dp)
 bodies: [names],
 signals: { joint_target:[[...]], joint_pos:[[...]], contacts:[[foot/finger flags]], packet_pca:[[pc1,pc2,pc3]],
            probe:{contact:[[..]], halt:[..], goal:[[x,y]], held_by:[..]}, phase:[..], task_events:[{t, event, status}],
            edit_active:[bool], forward_progress:[..], object_pose:[[..]], penetration_mm:[..], slip:[..] },
 annotations: [{t, text}] }
```
Missing signals are omitted, never faked.

Frontend additions (v1.1, proposed by the frontend agent; all optional, the theatre degrades without them):
- `geoms[].pos` `[x,y,z]` and `geoms[].quat` `[w,x,y,z]`: the geom pose **in its body frame** (MuJoCo `geom_pos`/`geom_quat`);
  identity when absent. Sizes use MuJoCo semantics (box half-extents, capsule/cylinder `[r, half-length]` along local z,
  plane `[hx, hy, grid]`). World frame is z-up. `body` is a name in `bodies`; the world body may be omitted from `frames`.
- `meta.contact_bodies: [names]`: the body for each column of `signals.contacts` (enables 3D contact markers).
- `meta.base_body`, `meta.object_body`: bodies drawn with a ghost trail (defaults: first non-world body; `object_pose`).
- `meta.joint_names: [names]`: names for the columns of `joint_target` / `joint_pos`.
- `meta.packet_pca: {fit_on, explained_variance: [3]}`: provenance of the PCA basis.
- `/api/meta`: plugin status `{exporter_available, exporter, root, replays_dir, video_dir, cache_s}`; the UI uses it to decide
  between live data and clearly-labelled FIXTURES (fixtures are only used while the exporter module does not exist).
Run-history signals (v1.2, IMPLEMENTED by `rrp.viz.record` for the frontend's run-history panels). All optional, per frame
(same length as `frames.t`) unless noted; omitted when the quantity does not exist for that harness, `null` in frames where it
is undefined (e.g. before the first packet, a foot in swing). Units and privilege are stated per signal in
`meta.signal_notes`; PRIVILEGED signals are display-only (never an observation of any controller).
- `joint_vel: [[..]]` columns `meta.joint_names` (rad/s or m/s).
- `joint_torque: [[..]]` columns `meta.joint_torque_names` (actuator names): actuator force/torque applied (N·m or N).
- `contact_force: [[N per meta.contact_bodies]]` normal force; `contact_force_tangential: [[N ..]]`;
  `contact_pos: [[[x,y,z] | null per contact body]]` normal-force-weighted contact point, world. Legged: feet vs floor;
  arm/dual: finger (touch-sensor body subtree) vs non-robot bodies; rig: pads vs cube. PRIVILEGED.
- `probe_truth: {…}` the ground truth for the same keys/units as `probe` in the same frame. Legged: `contact` (feet, assembly
  order; the probe has one extra body assembly), `halt`, `goal` (body frame, /2 m), `subtask`. Arm: `held_by`, `contact`
  per detector slot. PRIVILEGED, display only.
- `packet_z: [[float]]` the executed packet downsampled to the first 8 latent dims per knot × assembly, flattened in
  `meta.packet_shape: [knots, assemblies, 8]` order; `packet_norm: [[[..]]]` full-dz L2 norm per knot × assembly;
  sparse `packet_events: [{t, z_norm, edit?, edit_dz_norm?, source?}]` one per new packet.
- `power: [W]` Σ|actuator force × actuator velocity|; `energy: [J per frame interval]` (legged/tracker/rig integrated over
  every physics substep; arm/dual sampled at control ticks, stated in the notes); `meta.energy_total_j`;
  `cot: [..]` legged only, energy / (m g horizontal path), null until 0.2 m.
- `base_vel: [[vx,vy,vz]]`, `base_ang_vel: [[wx,wy,wz]]` (legged root), `object_vel`, `object_ang_vel` (cube / first task
  object), world frame. PRIVILEGED.
- Edits: `edit_dz_norm: [..]` |edited − unedited packet| computed with the SAME flow noise at the same state (legged context
  edits: a copy of the generator state; arm edits: the same noise key); null before the edit. `meta.edit_onset_t`
  (legged: t_edit; arm edit replays: 0.0) and `meta.edit_first_packet_t` (legged: first edited packet).
- Arm: `gripper_aperture: [m]` (public width sensor, where one exists); `grasp_state: ["free"|"contact"|"held"]` PRIVILEGED.
- Dual: `hand_contact: [[bool per meta.hands]]` (public touch ≥ 0.2); `grip_drift: [[[pos_mm, rot_deg] | null per hand]]`
  held object's pose drift in the TCP frame since the grip formed (PRIVILEGED held truth).
- `meta.waypoints`, `meta.t_edit` are recorded for legged (top-down map).
- `/api/doc?path=` is served by the plugin directly from the allowlist (no exporter call). The packet PCA basis is fitted per bundle on training packets and stored in `meta`.

## views (frontend)
1. **Overview**: what is established vs open, key numbers, caveats, latest decisions.
2. **Live ops**: peer vitals timeline, admission, leases (declared vs measured, throttling), DAG progress per workstream, watchdog events.
3. **Results matrix**: task × body × route heatmap (rate + CI), filters for physics version, seeds and variants, and deltas (e.g. grasp v1 → v2).
4. **Episode theatre**: a three.js 3D replay with synchronized timelines (targets/positions, contacts, packet PCA trajectory, probe readouts, phase, task events, edits), side-by-side compare, and scrubbing. Video fallback.
5. **Causal edits**: effect plots with controls, per seed and body, with permutation tests.
6. **Training**: curves and gradient health (D-085 clip scale), reward-schedule α, DAgger rounds, probes.
7. **Robustness**: break-point curves, motion quality, variant-level diffs.
8. **Physics credibility**: contact/grasp/actuator tables, tracker validation, gates.
9. **Ψ₀ line**: reproduction and step-2 results; psi1z decisions.
10. **Knowledge**: decisions timeline (D ↔ P crosswalk), roadmap, backlog, strategy, docs reader.
