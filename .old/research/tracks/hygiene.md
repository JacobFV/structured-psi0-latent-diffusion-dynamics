# hygiene track (W2, docs/strategy.md) — 2026-09-26

State: completed (see the commits on `track/hygiene` merged to main). Resume file for W2.

## tests
- `tests/conftest.py`: markers `menagerie` (needs `.cache/assets/mujoco_menagerie`) and `packed_data`
  (needs `artifacts/packed/latent_pp_v3dart_s1_H16`) skip cleanly when the input is absent.
- Marked: `tests/unit/test_prev_action_col.py`, `tests/integration/test_adapt_branching.py` (menagerie);
  `tests/integration/test_binding_aug.py` (moved from `tests/`, packed_data).
- `addopts = "-ra"` (was `-q`; `pytest -q` then hid the summary line).

## untracked from git (2026-09-26; `git rm --cached` only, history not rewritten)
78 files, 391.4 MB: 30 `.npz` DAgger shards, 39 `.genctx.pkl` generator contexts, 3 `.pt` tracker weights, 6 `ops/legged/*.log`.
The demo page (`scripts/demo/build_page.py`, `refresh.sh`) reads only the `*.summary.json`/JSONL next to them, which stay tracked.
**Pulling this change deletes these files from other checkouts' working trees.** A copy with the same relative paths is at
`~/work/rrp-data/git-untracked-2026-09-26/` on the host (the peer store has the originals). Restore into a checkout with
`rsync -a ~/work/rrp-data/git-untracked-2026-09-26/ <checkout>/`. `scripts/contact_validate.sh` reads
`artifacts/trackers/<body>/actor.pt`, which now comes from the store, not git.

| dir | files | MB |
|---|---|---|
| `artifacts/runs/ladder_dagger_bc1` | 13 | 27.5 |
| `artifacts/runs/ladder_dagger_bc2` | 13 | 27.3 |
| `artifacts/runs/ladder_dagger_gdag1` | 13 | 122.4 |
| `artifacts/runs/ladder_dagger_gdag2` | 13 | 106.0 |
| `artifacts/runs/ladder_dagger_gdag3` | 13 | 99.2 |
| `artifacts/runs/ladder_dagger_gen1` | 4 | 8.3 |
| `artifacts/trackers/go2` | 1 | 0.2 |
| `artifacts/trackers/go2/rejected_iter1999` | 1 | 0.2 |
| `artifacts/trackers/hexapod6` | 1 | 0.2 |
| `ops/legged` | 6 | 0.0 |

## docs reconciled (checklist)
- [x] AGENTS.md is the single current contract (resources D-027/D-033/D-086: host ≤80% free CPU/memory, host GPU 3 leases,
      ≥100 GB disk free; peer 100%); CLAUDE.md `@`-imports it; `docs/handoff/AGENTS.md` carries a HISTORICAL banner
      (bytes below the banner still match `docs/handoff/SHA256SUMS`).
- [x] README: no correction-branch / private-remote / 20-Hz-for-all / old-path-only-GRPO claims; per-family clocks;
      arm ladder, legged and dual pipelines; entry points; links to strategy, training considerations, evidence matrix, naming.
- [x] STATUS: header 2026-09-26 18:40 PDT; current-state section with the D-094 workstream table; evidence summary kept;
      older sections under "history".
- [x] BRIEF: CURRENT section → docs/strategy.md; DEMO SPRINT and the big picture marked historical; RRP_PEER_REPO rule.
- [x] registry.jsonl (+42 rows, vocabulary states only) and artifacts/requirements.json (all 38 statuses; release gate passes).
- [x] research/naming.md.

## verification (2026-09-26)
- Worktree without assets: `pytest tests/unit -q` → 152 passed, 3 skipped (2 Menagerie, 1 transformers).
- Worktree with `.cache` symlinked to the main checkout's assets: 154 passed, 1 skipped (transformers);
  `tests/integration/test_binding_aug.py` with `artifacts/packed` linked: 2 passed; without: 2 skipped.
- Fresh `git clone` of origin/main (96f2e08) in a scratch dir: 152 passed, 3 skipped; 0 tracked `.npz/.pkl/.pt`.

## left for W4 (found, not changed)
- `docs/handoff/scripts/check_package.py` already reported 3 checksum mismatches before W2 (README.md, config/sources.seed.json,
  reference/original-design-v0.1.md); the "preserved verbatim" claim for docs/handoff is not strictly true.
- `docs/demo/raw/artifacts` duplicates ~176 blobs from `artifacts/` (24 MB); left tracked because the demo page reads it.
- `research/tracks/ladder/sprint_final/` and `armnosem/ladder_v1/<body>/` hold raw result files that belong in `artifacts/`.
- Small `.log` files remain tracked under `artifacts/runs/` (binding_diag, binding_v1_reeval, contact_v2/val, t1_edits_video;
  ~50 KB total): raw text output, not datasets.
- `research/tasks.json` (P01–P25) is stale since 2026-09-21 (only P02/P03 set).
- Naming inconsistencies: `research/naming.md` last section.
