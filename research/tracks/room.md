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
- Cache: `viz/data/_cache/files.json` = (path, mtime_ns, size) -> sha1, and sha1 -> extracted products. Cold run ~8.5 s,
  warm full run ~1.5–3 s, `--live --only live` ~0.7 s (one ssh; the remote script is time-boxed to 3.5 s, ssh timeout 5 s).
  Peak RSS ~230 MB.
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

Resume: `cd ~/work/rrp-wt/roomdata && PYTHONPATH=src .venv/bin/python -m rrp.viz.export --out viz/data` (add `--live`).
