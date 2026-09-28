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
