# track: rel-r21 (relation-factor fanout, unit R21: viz factor / attention inspection)

Branch `track/rel-r21`, worktree `~/work/rrp-wt/rel-r21`, from `origin/main` @ `c5e7d45` (D-144 foundation). Spec:
`docs/relations.md` sections 2-5 and 10 (row R21, W1 -- deps only on F). Host only (git / editing / unit suite;
CUDA hidden); no peer smoke needed (the row's acceptance is a recorder / exporter unit test plus frontend wiring, no
training or simulation beyond the unit suite's own tiny fixtures).

Continued from an interrupted prior attempt: the worktree already held uncommitted changes matching this row's
scope (`viz/record.py`, `viz/export/relations.py`, their tests, and the `viz/room/**` frontend) when this session
started; inspected with `git log` / `git status` / `git diff` first, verified every changed/untracked path is owned
by R21's row before continuing, then ran the change through to green rather than discarding it.

## scope (docs/relations.md 10, R21's owns + brief)

Owns `viz/record.py` (factor records), `viz/export.py` (the package `src/rrp/viz/export/`, specifically its
`relations.py` submodule), `viz/room/**`. Did not touch `relations/base.py` or `relations/ops.py` -- only imported
from them (`FactorSite.contributions`, `rrp.policies.relations.base.provenance`); no operator/interface change was
needed for this row.

- **`src/rrp/viz/record.py`**: `record_factor_maps(replay_id, entries, specs)` turns `FactorSite.contributions(rc,
  xq, xk)` calls (per-factor `[B,H,Q,K]` logit terms, `relations/ops.py` 3.5) at chosen recorded steps into one
  `rrp-viz/factormap/v1` document -- its own schema, not the `rrp-viz/replay/v1` one, since a per-factor per-head
  logit map is not a fixed-shape per-frame signal like the others in `FRAME_SIGNALS`. Only batch element 0 is kept
  (one recorded episode), values rounded to 4 dp like every other recorded signal (`rrp.viz.replay.r4`), and a
  factor absent from a step's `contributions` (e.g. `control: off`) stays absent -- never a zero-filled
  placeholder. `provenance` (the run's resolved `FactorSpec`s) is turned into `rrp.policies.relations.base.
  provenance` rows so the room can badge each map by its source without re-deriving it. `write_factor_map` /
  `read_factor_map` round-trip it to `<out_dir>/<id>.factormap.json.gz`, alongside the replay of the same id --
  verified by reading the Vite plugin's replay file walk (`rrp-api-plugin.ts`: strips only a trailing `.json` /
  `.json.gz`, so a file named `<id>.factormap.json.gz` is served at `/api/replay/<id>.factormap` with zero plugin
  changes, exactly as the frontend fetch (`replay.ts`) expects).
- **`src/rrp/viz/export/relations.py`**: `_competence_by_depth(records)` reduces the latest `ScheduleState` line
  (`<run>/schedule.jsonl`) to one row per factor (depth = current composition level `k_f`, share, and once R11
  fills in the scheduler's real signals -- today's F4 placeholder always leaves `signals` empty -- competence /
  plateau / interference / attributed_failures). A factor with no signals yet gets `competence: None`, never a
  fabricated number; `has_competence` flips true only once a schedule line actually carries `signals`. Wired into
  `_schedules()`'s existing per-run loop (adds `step`, `competence_by_depth`, `has_competence` to each row; no
  change to its file-reading logic).
- **`viz/room/src/lib/replay.ts`**: `FactorMap` / `FactorProvenance` / `FactorMapStep` types for the
  `rrp-viz/factormap/v1` document, and `nearestFactorStep(map, t)` (factor maps are sparse -- chosen steps, not
  every frame -- so the panel shows the recorded step nearest the scrub cursor, not an interpolation).
- **`viz/room/src/components/FactorHeat.tsx`** (new): `FactorMapPanel` -- a factor picker, an unmistakable source
  badge (`PRIVILEGED (<source>) · not deployable` / `ESTIMATED (probe|estimator:...)` / `PUBLIC (<source>)`, per
  AGENTS.md: source marked by more than colour alone) built from the map's own `provenance`, and `HeadGrids`: one
  small-multiple grid per attention head, diverging colour scaled to that map's own max |logit| (never a shared
  scale across factors, since raw logit magnitudes differ by factor).
- **`viz/room/src/views/RunHistory.tsx`**: fetches `<replay id>.factormap` alongside the replay via the existing
  generic `useReplay` hook (no new API surface), adds a "Relation-factor attention" pane when a map with >= 1
  recorded step comes back, with a `CAPS.factors` caption explaining rows/cols/colour/badge.
- **`viz/room/src/views/FactorsView.tsx`**: the "Runs with a factor set" panel now shows, per run, the latest
  schedule step and a competence-by-depth table (factor, depth, share, competence, plateau, interference), with an
  inline note when a run's schedule has no competence signals yet (R11 placeholder) instead of hiding the row.

## rebase conflict (origin/main moved during this session)

`git fetch && git rebase origin/main` picked up `3116892` (the lead's "aggressive deletion pass", concurrent with
this row, dropping raw-table panels room-wide: Results board, Knowledge/docs reader, Robustness/Physics/Psi0/
Matrix/Theatre views, and -- in `FactorsView.tsx` itself -- the "Registered factors" `DataTable` dump and the
"Presets" dump). That commit conflicted with this row's addition to the same file's "Runs with a factor set ·
schedules" section. Resolved by keeping the lead's room-wide simplification (no raw `DataTable` of the factor
registry or presets re-added -- `fs`/`prov`/`sources` stayed visible via the existing Registry heatmap and the
attention panel's own `badgeLabel`/`SourceBadge`, which is the actual unmistakable-badge implementation the row's
"privileged / estimated badges" criterion asks for, not a text column) while keeping this row's
`competence_by_depth` table (compact, decision-relevant, not a raw dump of `d.runs`) and dropping the superseded
raw `d.runs` `DataTable` alongside it, consistent with the lead's stated intent. `git diff` against `origin/main`'s
version of `FactorsView.tsx` confirms the only net changes are the `competence_by_depth` addition to the schedules
panel; the Registry and Candidate-catalog panels are untouched. Reran the full unit suite after the rebase (new
commits `3116892` landed) before pushing.

## acceptance (docs/relations.md 10's row)

"room shows per-factor logit maps per head for a recorded step" -- `FactorMapPanel` / `HeadGrids`, wired into
`RunHistory`'s pane list, gated on real recorded steps (`fmA.data.steps.length > 0`).
"the factor list with sources / controls, privileged / estimated badges" -- pre-existing `FactorsView` registry
table (foundation, field provenance public/estimated/privileged) plus this row's `badgeLabel` on the attention
panel itself.
"competence by composition depth (from R11 outputs when present)" -- `_competence_by_depth` + the `FactorsView`
table, `None`/`has_competence: false` until R11 lands, never fabricated.
"exporter unit test" -- `test_schedule_competence_by_depth_when_present` (two schedule lines, before/after R11
signals) plus the pre-existing `test_factors_doc_from_the_registry`.

## tests (red then green)

`tests/unit/test_viz_record.py` (+2 cases) and `tests/unit/test_viz_relations.py` (+1 case). Confirmed red first:
the uncommitted worktree state at session start already had these tests failing against a not-yet-`import`ed
`rrp.viz.record.record_factor_maps` / `_competence_by_depth` (module attribute errors); iterated the two source
files to green.

- `test_factor_maps_schema_and_provenance`: builds a REAL `FactorSite` (2 heads, tiny `pos3d` `sqdiff+diff/aug`
  factor registered via `register_factor`) on a CPU fixture (no simulation), calls `.contributions()`, and checks
  `record_factor_maps` reproduces its `[H,Q,K]` values (rounded to 4 dp) exactly, carries the factor's `source` /
  `privileged` provenance, and round-trips through `write_factor_map` / `read_factor_map` byte-for-byte (`gzip` +
  `json`).
- `test_factor_maps_batch_zero_and_missing_contributions_stay_absent`: an empty `specs=()` yields empty
  `provenance` (never guessed), and only the head dimension of batch element 0 survives.
- `test_schedule_competence_by_depth_when_present`: writes a two-line `schedule.jsonl` (first line no `signals`,
  second line with real `competence` / `plateau`); asserts `has_competence` is `False` then `True` and the row
  values match exactly, including every `None` field on the first read.

## commands run

```
cd ~/work/rrp-wt/rel-r21 && export PYTHONPATH=$PWD/src:$PWD
ln -s /home/brandonin/work/relational-robot-policy/.venv .venv      # same convention as sibling worktrees (refactor, roomdata)
.venv/bin/python -m pytest tests/unit -q          # full unit suite: 557 passed, 39 skipped, exit 0
.venv/bin/python -m pytest tests/unit/test_viz_record.py tests/unit/test_viz_relations.py -q \
    -k "factor_maps or competence_by_depth or factors_doc"   # this row's new/touched cases: 4 passed
```

State: **completed** (implementation + tests green; merge to main per docs/relations.md 10's merge rule).
