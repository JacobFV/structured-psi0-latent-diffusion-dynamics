# track: demo (sprint_demo, 2026-09-25 19:00 → 2026-09-26 05:00 PDT)

Owner: sprint_demo agent. Branch `track/demo`, worktree `~/work/rrp-wt/demo`, peer dir `/dev/shm/rrp-brandonin/wt/demo`.
Deliverable: `docs/demo/index.html` (one self-contained page, inline CSS, light/dark, phone width) + `docs/demo/video/*.mp4`,
all rebuilt from saved raw outputs. The lead publishes it; this track never publishes. This track also owns
`research/reports/evidence_matrix.md` for the sprint.

## how to rebuild (exact)
1. `scripts/demo/pull_raw.sh [extra peer paths under artifacts/runs ...]` — copies the small JSON/JSONL summaries the page
   cites from the peer store into `docs/demo/raw/artifacts/runs/<same path>` (no weights).
2. `python3 scripts/demo/build_page.py` — writes `docs/demo/index.html`. Every table number is read from a raw file; the page
   lists all raw files it read ("Sources"). Narrative numbers cite decision ids. Sprint updates go in `docs/demo/updates.html`
   (inserted verbatim under the bottom line).
3. Visual check (host, no listener): `google-chrome --headless=new --no-sandbox --window-size=390,5200 --screenshot=<png> file://$PWD/docs/demo/index.html`.

## videos (state: verified; encoded on the peer — the peer venv HAS imageio_ffmpeg 7.0.2, contrary to the sprint note)
Rendered on peer GPU EGL into `artifacts/runs/demo_video/` (NOT the shared artifacts/video INDEX), then copied to
`docs/demo/video/` and `artifacts/video/` (+ INDEX.md lines).
- scripted teacher: `bash scripts/demo/render_teacher.sh` (lease 1790388347_45aa63): panda_pg2 s3000002 success, parm6_tf3
  s3000003 success, paired binding scene panda_pg2 s3000001 with patient 0/1/2, all success. `scripts/render_episode.py`
  gained `--patient` for `--task pick_place_paired`.
- ladder R1 ORACLE (re-anchored every 8 ticks, prev 0): `bash scripts/demo/render_ladder.sh <rep> <tag> panda_pg2 <seeds>`
  (lease 1790388356_a3435d): jfdag1 (`ladder_rz_jointfix_dagger1/representation.pt`) s3000018 SUCCESS (matches the eval row, the only
  1/30), s3000003 failure-approach (eval row: lift; GPU vs CPU nondeterminism, stated in the caption); jointfix
  (`ladder_latent_sem_b1fix_anchor/representation.pt`) s3000008 failure-lift, s3000000 failure-approach.
- reused: dual-arm assign_left/right + handover teacher clips, oracle rebind causal-edit clip, learned flow v2@22k failure.
- R2 generated on a B-1-fixed bundle: NOT rendered yet — no flow on the jointly trained encoder exists (waiting for
  sprint_latent "SPRINT BEST ROUTE").

## log
- 19:05 worktree + peer dir; read BRIEF, D-029..D-049, evidence matrix, all track notes.
- 19:06 teacher + ladder renders done (see above).
- 19:20 page v1 built (44 raw files), evidence matrix updated with D-048/D-049 and the B-1-fixed BC state. BC source at
  update 12,000 (host). No SPRINT sections in ladder/acceptance/baselines notes yet.

## next
- Every ~90 min: check `SPRINT BEST ROUTE` (track/ladder ladder.md), `SPRINT SEMANTIC RESULTS` (acceptance.md),
  `SPRINT BC RESULT` (baselines.md) on origin/track/* and the worktrees; pull their raw files, add rows/clips, rebuild,
  commit, merge docs/demo + videos to main, tell the lead.
- R2 generated clip on the best bundle once a fixed flow exists.
- 19:30 refresh 1: sprint_bc result folded in (plain BC with the fix 25/30, 27/30 direct; 28/30, 25/30 codec; raw
  `artifacts/runs/baselines_bc_ladder/`). BC clips via `scripts/demo/render_bc.sh` (lease 1790388739_b6c0b1): re-renders
  differ from eval rows for 2 of 4 seeds (s3000013 panda eval fail/render success; s3000003 parm6 eval success/render
  fail-lift); captions say so. Bottom line, matrix and evidence_matrix.md headline updated. Merged to main.
- 19:40 held-out source-body BC (74/80 pooled, raw artifacts/runs/baselines_bc_ladder/heldout/) added; caption wrap fix.
- 19:50 lead request: build_page.py also writes `docs/demo/artifact.html` (no doctype/html/head/body; <title>, <style>,
  content, script; color-scheme:dark in both dark token blocks; relative video paths). Accuracy: R1 oracle route is
  CONFOUNDED (sprint_bc: shadow teacher's final phase lags in BC successes, computed on the page from raw
  final_teacher_phase); lede says the latent failure is not localized; "heads for the object" is an observation; next #1 =
  R2 vs BC. Same-scene triptychs (teacher | BC | oracle) for s3000008 and s3000018 via scripts/demo/render_triptych.sh +
  side_by_side.py (lease 1790388888_c98769); the jfdag1 "success" seed 3000018 FAILS (approach) in this re-render.
- 20:05 lead wording fix in §6 and evidence_matrix.md (failure in the packet route incl. possibly the generator). Lead republishes artifact.html per refresh.
