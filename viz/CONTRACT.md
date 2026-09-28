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
- `src/rrp/viz/export.py`: `python -m rrp.viz.export [--live] --out viz/data`. Builds the JSON documents below from the repo,
  the local artifact copies and (with --live) small ssh reads of the peer broker, watchdog and ledgers.
- `src/rrp/viz/record.py`: PEER-ONLY replay recorder: `python -m rrp.viz.record --spec <yaml> --out <dir>`. Writes replay
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
