# room track (D-131): live visualization room

## data
Owner: room data agent (worktree `~/work/rrp-wt/roomdata`, branch `track/roomdata`). Contract: `viz/CONTRACT.md`.

What exists
- `src/rrp/viz/export/` (`python -m rrp.viz.export [--live] --out viz/data [--only name] [--sync-psi1z]`): writes the 12 API
  documents + `training/<id>.json` + `_manifest.json` into `viz/data/` (gitignored). Stdlib only, one process.
- `src/rrp/viz/api.py` (`python -m rrp.viz.api get|doc|replay|media|training|routes`): the Vite plugin's helper; interface in
  the contract section "plugin interface" (contract edited in a separate commit: provenance fields, `/api/training/<id>`,
  plugin interface).
- `tests/unit/test_viz_export.py` (fixture trees: envelope/schema keys, provenance on every row, D-entry parser, result/edit/
  training extraction, doc allowlist, no ssh/rsync without --live). `tests/unit/test_layering.py`: `viz` = layer 12, nothing imports it.

Sources and how they are read
- Files: repo (= in git), `~/work/relational-robot-policy`, every `~/work/rrp-wt/*` (artifacts/runs, artifacts/trackers,
  research/tracks), `~/work/rrp-data/{git-untracked-2026-09-26,frozen}`. ~60 k candidate files, deduplicated by sha1
  (~2.5 k unique JSON, ~160 unique train logs); the canonical copy is repo > main > worktrees > rrp-data, `copies`/`alt_paths` kept.
- Cache: `viz/data/_cache/files.pkl` = (path, mtime_ns, size) -> sha1, and sha1 -> extracted products; `found.pkl` = the file
  list, reused for 10 s so the plugin's per-document calls share one walk. Full run: ~3 s with a cold cache (first ever run
  with a cold OS cache ~8.5 s), ~1.5 s warm; per-document calls 0.1–0.5 s; `--live --only live` ~0.7 s (one ssh; remote script
  time-boxed to 3.5 s, ssh timeout 5 s). Peak RSS ~200–270 MB.
- Decisions: a result's `decision` is the latest D-entry whose text names the file or one of its run directories (identifier-like
  tokens only; body names and stems shared by > 3 files never match). ~52% of result rows have a match; the rest are null.
- Caveats attached by rule: t1 D-113 label (legged t1, contact_v2/legged8 lineage), grasp version missing/grasp_v1 (arm),
  contact version missing/contact_v1 (legged), go2 semfix inexact resumes (D-113), partial/interim files.
  Interim: run-dag node outputs of incomplete DAGs, `armexpert_gc2eval` except `compare_gc2_final` (D-121), t1 sourced-limit
  lineage, psi1z runs without a summary.json.
- psi1z copies: `--sync-psi1z` rsyncs only summary.json/result.json/episodes.jsonl/eval_stats (≤ 5 MB each) from
  `gb10-direct:~/work/ext/runs/psi1z/cl/` into `~/work/rrp-data/viz/psi1z/` (368 KB on 2026-09-28).

Gaps (missing stays missing)
- clip_scale is not logged by the trainers; only grad_norm/gn is. D-085 grad health is shown from the recorded
  `research/tracks/legged8/clipscale_*.json` summaries (`training.grad_health`); no clip scale is derived.
- Peer run-dag ledgers: none exist under /dev/shm (the DAG runner writes ledgers on the host worktrees); the glob is still read.
- Many arm rows have no recorded grasp version and many older rows have no source label (source_label null,
  `source_label_from: missing`); legged8 lineage evals are .jsonl (not summarized per file) and enter via the compare JSONs.
- Replays: empty until the recorder writes `~/work/rrp-data/viz/replays/index.json`.

Frontend plugin note: it must test `src/rrp/viz/export/__init__.py` (the exporter is a package), stated in the contract.

Resume: `cd ~/work/rrp-wt/roomdata && PYTHONPATH=src .venv/bin/python -m rrp.viz.export --out viz/data` (add `--live`).

## replays

Owner of this section: the replay-recorder agent (branch `track/roomrec`, worktree `~/work/rrp-wt/roomrec`, peer code dir
`/dev/shm/rrp-brandonin/wt/roomrec`). Contract: `viz/CONTRACT.md` "replay file" (`rrp-viz/replay/v1`).

### what was built
- `src/rrp/viz/replay.py`: pure data code. Static geometry export (visible geoms of groups 0–2; meshes decimated by vertex
  clustering to ≤ 2000 faces and inlined; height fields become meshes; menagerie files are never copied), a read-only frame
  collector (≤ 30 fps; body poses 4 dp; signals absent in some frames are null, all-null signals are omitted), packet PCA
  (fit / project), schema validation, gzip JSON writer, `index.json` writer (`rrp-viz/replays-index/v1`).
- `src/rrp/viz/record.py`: `python -m rrp.viz.record --spec a.yaml[,b.yaml] --out <dir> [--only ids] [--shard i/n] [--skip-done]`.
  PEER ONLY (refuses unless `RRP_NODE=peer`). One subprocess per spec entry with the entry's env (e.g. `RRP_GRASP_CONTACT`,
  `RRP_CONTACT_MODEL`, `RRP_ACTUATOR_LIMITS`, `OMP_NUM_THREADS`) and CUDA hidden (every recorded eval ran on CPU).
  Harnesses are re-run through the SAME functions that produced the recorded rows and observed from outside:
  | harness | recorded rows came from | observation point |
  |---|---|---|
  | `ladder` (arm R2 / BC) | `rrp.evaluation.ladder_cli` (`scripts/ladder.py`) | `run_ladder(frame_cb=...)` (existing read-only callback); the full feasible seed list is re-run batch by batch (16) up to the target seed's batch, because the flow/BC noise generator is shared across batches |
  | `arm_teacher` | `rrp.evaluation.teacher_quality` | wrapper on `Session.step` (after the original), armed when `make_arm_teacher` returns |
  | `arm_edit` | `rrp.cli latent semantic-edits` | `latent_semantic_edits.run_condition(on_step=...)` (existing callback); source/probe built as the CLI does |
  | `legged`, `legged_robust` | `rrp.evaluation.legged_latent_eval` / `rrp.evaluation.robustness` | class wrappers on `LeggedMotionRecorder.on_tick/on_reset` (the read-only recorder `run_episode` always installs), 50 Hz ticks → 25 fps |
  | `tracker_val` | `rrp.evaluation.tracker_validation` | a `mujoco` proxy in that module's namespace (wraps `mj_step`, forwards everything else) |
  | `dual_teacher` | `rrp.evaluation.dual_teacher_quality.run_audit_episode` | class wrapper on `DualQualityRecorder.after_step` |
  | `grasp_rig` | `rrp.evaluation.grasp_rig.run` | `mujoco` proxy in that module |
  Packet taps (latent routes): a wrapper on `LatentSystem0.receive` remembers the packet system 0 executes (arm); legged uses
  the controller's own `record_z` list and its probe readouts in `ctl.packets` (generate() only appends to them).
  No harness file is edited; no RNG is consumed; only mjData reads plus `mj_contactForce` / `mj_objectVelocity`.
- `tests/unit/test_viz_record.py`: decimation budget, schema on a tiny synthetic model (3 mj_steps), 6 broken-document
  rejections, PCA, recorded == unrecorded executed commands on a 6-tick parm5_pg2 ladder case (host), and a peer-only legged
  check (final pose identical with and without recording). Host: 10 passed + 1 skipped; peer (`RRP_NODE=peer`): 11 passed.
- Specs: `viz/specs/{arm,legged,physics,dual}.yaml`, generated by `scripts/viz_record_specs.py` (episode choices documented
  there). Packet PCA bases (stored in each replay's `meta.packet_pca_basis` with `fit_source`):
  - arm: `mu` of the system-0 refit DAgger buffers the bundle's system 0 was trained on (`ladder_dagger_<lin>_gen3` for
    gendag3 R2 bundles, `_gen1` for the gendag1 edit bundles; the original sem lineage uses the unprefixed dirs), 4000 packets;
  - legged: `E(ctx, demonstrated targets)` on 2560 training rows of the bundle's own data (the flow's targets), per body.

### run
PEER, lowest priority: `nice -n 19`, ≤ 2 concurrent CPU leases (2 CPU each). Memory: the panda ladder batch (16 menagerie
sessions) peaks at 3.3 GB RSS, so leases declare 5G (≥ 1.35 × peak, D-117). A first launch at 3G was throttled at
memory.high (≈ 9k high events in < 1 min); both leases were stopped at once with `ops stop --owned-only --lease ...` and
relaunched at 5G (no live cgroup changes, D-117).

```
export RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/roomrec; scripts/peer_sync.sh push
S=viz/specs/arm.yaml,viz/specs/legged.yaml,viz/specs/physics.yaml,viz/specs/dual.yaml
for i in 0 1; do scripts/peer_run.sh --cpu 2 --mem 5G --label roomrec_record_$i --max-seconds 14400 --detach -- \
  nice -n 19 PY -m rrp.viz.record --spec $S --out artifacts/runs/roomrec_replays --skip-done --shard $i/2; done
```
Host copy: `rsync gb10-direct:/dev/shm/rrp-brandonin/repo/artifacts/runs/roomrec_replays/ ~/work/rrp-data/viz/replays/`.

### catalogue (2026-09-28, 160 replays, 48.5 MB; peer `artifacts/runs/roomrec_replays/`, host `~/work/rrp-data/viz/replays/`)
Outcome per seed as re-run (S/F). Every replay's meta carries the seed, checkpoint shas, physics versions, env, the
eval config (ladder: seed-list, batch index, position in batch) and decision_refs.
| family / task | route | cells | replays |
|---|---|---|---|
| arm pick_place, panda_pg2 + parm6_tf3 | R2 (flow gdag2h → system 0 gendag3_noqd): frozen sem s1/s2, semfix s1/s2, nosem s1/s2 × grasp_v1 (as recorded) and grasp_v2 (D-127) | 24 | 50 (2 per cell, 3 for frozen sem s1 parm6; S and F where the first batch has them) |
| arm pick_place | BC direct1701_final × grasp_v1/v2 (panda 3000000, 3000111; parm6 3000003, 3000004) | 4 | 8 |
| arm pick_place | scripted teacher v2 × grasp_v1/v2 (no recorded failure exists: 4810/4811) | 4 | 8 |
| arm pick_place, semantic edits (grasp_v1) | control / rebind_desc / goal_shift: semfix s1, nosem s1, semfix s2, nosem s2 on parm6 (3000003); semfix s1, nosem s2 on panda (3000000) | 6 | 18 |
| arm grasp rig | v1 vs v2 × pg2/tf3 × friction ×1 / ×0.05 (tf3 v1 at ×0.05 drops the cube) | 8 | 8 |
| legged W8 contact_v2 | semfix s0 / nosem s0 × {unedited, ctx halt, mirror_inactive control} × anymal_c (10002, 10017), go2 (10007, 10003), t1 (10019, 10000); 5 s edit windows, t_edit 2 s | 18 | 36 |
| legged W8 | teacher and BC references (60 s; 10002, t1 also 10019; t1 BC 0/30 as recorded) | 6 | 8 |
| legged robustness (anymal_c, policy.pt s0) | semfix vs nosem: push 1 m/s (10001: semfix falls, nosem succeeds), friction 0.4 (10010), terrain 8 cm (10003, 10000) | 6 | 8 |
| legged tracker validation | forward trial seed 1000: v1 tracker on contact_v1 vs accepted v2 tracker on contact_v2, anymal_c / go2 / t1 (t1 v1 under legacy_gains_v0) | 6 | 6 |
| dual (W12) | teacher v3 smoke under grasp_v2.1 (support_insert, handover × parm5_pg2×2 and panda_pg2+ur5e_pg2 × seeds 0, 1); peg case: support_insert v2 teacher parm5×2 seed 0, grasp_v1 success vs grasp_v2.1 failure (L:l_hold|R:r_transit) | 6 | 10 |
Totals: arm 92, legged 58, dual 10; 67 successes, 93 failures. Largest file 0.82 MB (t1 60 s BC episode).

### reproduction check
All 160 re-runs reproduce the recorded outcome of the original eval row for the same seed (meta.reproduced = true,
160/160), and every compared detail field matches exactly (414 fields: arm outcome + steps; legged fell, sim_time,
final pose to 1 mm; tracker fell, distance, slip; dual status, steps, failure phase; rig lift_held, slip mass,
penetration; edits privileged_success, first contact, lifted). This includes the grasp_v1 arm rows (older code
commits), the W8 rows (commits 82367379 / 4b030fd6 / 9bf74f95) and the D-108 robustness rows that ran on the host venv.
No episode could not be reproduced.

### caveats and notes
- Legged 5 s context-edit episodes have success = false by construction (the task cannot finish in the window; D-090
  protocol); `meta.success_definition` says so. The effect is in `signals.forward_progress` after `t_edit`.
- Arm edit replays define success as `followed` (lifted the new cube / placed at the shifted goal / placed in the zone);
  the D-074 rebind measure is "approached the new cube first", in `meta.edit_row.approached_first`.
- The grasp rig has no peg/cylinder case (D-125 item 1 is still open); the peg case is the dual support_insert pair.
- `phase` for arm R2 is the SHADOW scripted teacher's FSM phase at the real state (privileged diagnostic label).
- Host transfer: the host broker's admission has been latched `available_memory_below_reserve` since before the 00:34
  host crash (state.json last event 1790579657; no host watchdog process is running), so no host lease can be admitted.
  The 48 MB rsync was run directly with nice/ionice and a 20 MB/s limit (host disk 576 GB free); flagged for the lead.

### resume
Re-record anything: `--skip-done` skips entries logged ok in `<out>/record_log.jsonl`; `--index-only` rewrites
`index.json`. New episodes: edit `scripts/viz_record_specs.py`, regenerate the specs, push, run with `--only <ids>`.

## frontend

Owner: the room frontend agent (worktree `~/work/rrp-wt/roomui`, branch `track/roomui`). The app is in `viz/room/` (see its README).

What exists
- `viz/room/rrp-api-plugin.ts`: a Vite middleware that is read-only (GET/HEAD only). `/api/<doc>` calls
  `python -m rrp.viz.api get <doc> [--live] --max-age S` (niced, serialized, 15 s cache, live 10 s, 30 s timeout) and serves
  `viz/data/<doc>.json` with an ETag, which gives a 304 when nothing changed (results.json is about 10 MB). If an export fails
  and an older file exists, that file is served with `X-RRP-Export-Error`, and the UI shows STALE plus the error.
  `/api/training/<id>`, `/api/replay/<id>` (recursive id → file scan under the replay root, gzip passthrough),
  `/media/<name>` (regex + extension allowlist, Range support, falls back to the main checkout), `/api/doc?path=` and
  `/api/doclist` (the contract allowlist, including psi1z README/research notes/decisions), `/api/meta`.
  Bind: `RRP_ROOM_HOST` (default 127.0.0.1; D-132), port 3013, strictPort.
- Ten views with URL state (`#view?k=v`); see the README for what each one shows. Code split per view; three.js loads only in
  the theatre.
- Theatre: `components/Stage.tsx` (geoms → three meshes in MuJoCo semantics, z-up, body poses per frame, contact markers from
  `meta.contact_bodies`, trails from `object_pose`/`base_body`, follow camera, render on demand), `components/Timelines.tsx`
  (a shared clock, click/drag to scrub, edit-active shading, per-signal notes from `meta.signal_notes` with PRIVILEGED flags,
  joints matched by name via `joint_names`/`joint_target_names`, per-foot arrays drawn as one line per column, packet PCA 3D
  drawn only when A and B share a basis).
- Fixtures: `viz/room/fixtures/*.json` and 2 synthetic replays (made by `scripts/make_fixtures.py`; they pass
  `rrp.viz.replay.validate_replay`). Real data never falls back to fixtures.

Verification (2026-09-28): `npm run typecheck` and `npm run build` pass. An HTTP smoke test on 127.0.0.1:3013 gave 200 for all
12 documents, 304 on a matching ETag, 200 for `/api/training/<id>`, 200/206 for `/media` (Range), 400 for traversal attempts
on doc/media/training, 404 for an unknown replay or API path, and 405 for POST; the server was stopped afterwards. An SSR render
check (`scripts/render-check.tsx`; `REPLAY_DIR=<dir>` adds real replays) renders all ten views, with 48 URL states over the real documents and the fixtures, with no
exception and no NaN/undefined/[object Object] in the output. No browser was run on the host (D-127), so WebGL and chart
interaction are untested here.

Known gaps
- The recorder's first arm replays were on the peer but not yet in `~/work/rrp-data/viz/replays` when this was written. Three of
  them (panda gv1/gv2 and parm6, copied to a scratch dir) pass the SSR render check, alone and as A/B pairs, and every one of
  their 130 geoms (49 meshes each on panda) builds a three.js mesh with a finite bounding box. Until the index exists the
  theatre says "No replays yet" and offers an opt-in FIXTURE demo. Legged and dual replays have not been checked yet.
- Joint targets and positions are paired by name. When the names differ (arm: `arm[i]`/`gripper[0]` targets against
  `r0_joint*` positions) they are paired by column order, and each row says so (`r0_joint1 ↔ arm[0]`).
- clip_scale is not logged by the trainers (data notes), so the clip overlay appears only where a series has it. GPU
  utilisation history is only the readings a page has seen, because the watchdog samples carry no GPU utilisation.
- Results heatmap: a cell holding several rows shows the one with the largest n, or pools k/n when asked. The default metric
  filter is `success`, to avoid mixing metrics.
- No browser/visual QA was done on the host, only SSR; layout at narrow widths and the three.js scenes need a look in a real
  browser.

Redesign (owner: "more compact, visually-oriented, wall-street trader"; lessons from IBM-2's board):
- dark-first terminal styling: seams in place of cards, a 22px ticker and a 28px top bar with the views numbered 1–0;
- `#board` as the landing page (KPI strip plus about 14 dense panels, each drilling into its full view);
- theatre with a large stage and a right rail of compact timelines;
- keyboard shortcuts;
- a shared per-document fetch store, so the ticker, board and views share their polls.

Two aggregations are named on the board where they appear:
- the arm KPI and lineage panel sum the `[lineage, body, grasp_v2]` rows of `compare_gc2_final.json` (311/480 semfix vs
  40/480 nosem, INTERIM per D-121);
- the halt tiles show the median halt/ctx_halt effect over rows of the semantic variants against nosem, on the edit's most
  common metric.

Commits are not in any exporter document, so the ticker does not show them (the data layer is unchanged). The render check
now covers the ticker and the board (58 states, plus 6 on real replays).

Resume: `cd ~/work/rrp-wt/roomui/viz/room && npm install && npm run dev` (add `RRP_ROOM_HOST=0.0.0.0` for the network).

### frontend v3 (owner feedback: IBM-2 sidebar, one-screen board, visuals over text); merges ae0e470, 2b82fa4, 3e202a4, then Runs
- Shell: IBM-2's left sidebar (brand, a view picker, then that view's controls/filters), one workspace, a thin ticker, a
  narrow-screen drawer, and no top bar or nested tabs. Five entries:
  - Board;
  - Evidence (lenses: success matrix, causal effects, robustness, physics gates, Ψ₀);
  - Runs;
  - Ops (peer/leases/DAGs, training health);
  - Library (decisions, roadmap, docs, claims).
  Old #results/#edits/… addresses still resolve to their lens.
- Board: four tiles (claim, competence, open, live), each one chart plus one number with its CI, explanations on hover. The
  render check asserts a per-tile pixel budget at 1440×900 (a layout-model check, not a browser measurement).
- Evidence:
  - curated heatmaps with CI whiskers: legged contact v2 (summary_contact_v2, D-124) and arm grasp_v2 (compare_gc2_final,
    D-127);
  - a grasp v1→v2 Δ multiple;
  - a go2-only contact v1→v2 multiple (legged_fixrep_compare_go2 D-090 → legged8_compare_go2 D-113; anymal_c/t1 have no v1
    match);
  - forest plots, break-point curves, gate/tracker charts and Ψ₀ bars.
  Raw tables and prose sit behind the sidebar "show data tables" toggle.
- Runs: a vertical stack of linked panels with one shared time cursor, rendered lazily and shown only when the replay has the
  data:
  - 3D;
  - top-down map (paths, waypoints, edit onset, fall, contact points);
  - phase lanes; task-event Gantt;
  - contact/gait diagram with duty factors;
  - progress;
  - edit and B − A difference traces (edit_dz_norm);
  - joint heatmap (position/target/tracking error/velocity);
  - phase portrait;
  - forces/slip/penetration/torque; velocities; grasp (aperture, state, hand contact, drift); energy/power/CoT;
  - packet (PCA 3D, change rate, packet_z and packet_norm heatmaps with packet events);
  - probes, and probe vs truth with a calibration strip;
  - video;
  - evidence (the reproduction check table plus all meta).
  v1.2 signals are read as recorder 872f753 defines them.
- Checks: the SSR render check covers every view and lens (plus the data toggle). With `REPLAY_DIR=~/work/rrp-data/viz/replays`
  it renders all 160 real replays alone and as A/B pairs (320 renders, every panel type that has data, 0 failures), plus a
  geometry check of all their geoms.
- Gap: the local 160 replays predate v1.2, so the velocity, grasp, energy, torque/force, packet_z and probe-truth panels have
  only been exercised on the synthetic fixture replays (which carry every v1.2 signal and pass `validate_replay`) until the
  re-recorded files are synced.
