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
