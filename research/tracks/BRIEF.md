# brief for parallel track agents (lead: main session)

## CURRENT (2026-09-26, D-094) — read this first
- What we do, in what order, and who owns which paths: `docs/strategy.md` (workstreams W1–W9, R0, gates, conflict rules).
  Rules: `AGENTS.md` (resources D-033/D-086: host ≤80% of free CPU/memory with host GPU, peer 100%, ≥100 GB host disk free).
  State: `STATUS.md`; evidence: `research/reports/evidence_matrix.md`; lineage codes of pre-schema runs: `.old/research/naming.md`.
- **Always `export RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/<track>` before `ops/bin/peer_sync.sh`.** Without it the
  script syncs (with `--delete`) into the lead's `/dev/shm/rrp-brandonin/repo`, where chains run (D-090 incident).
  Never sync into a dir whose jobs are still running (R0 runs from `wt/ladder`).
- The worktree/merge workflow below ("how to work") still applies. The host budget numbers in it are historical; ask the
  broker (`rrp ops status`) instead. The 2026-09-25/26 "big picture" and "DEMO SPRINT" sections were removed by D-145 (P5); the original text is in git history.

## how to work
- Repo: `~/work/relational-robot-policy` (the lead's checkout, on `main`; do not edit files there). Create your own worktree:
  `git -C ~/work/relational-robot-policy worktree add ~/work/rrp-wt/<track> -b track/<track> origin/main`, and work only in it.
- Host python: `~/work/relational-robot-policy/.venv/bin/python` with `PYTHONPATH=src` run from your worktree. Host data:
  `artifacts/datasets` and `artifacts/packed` in the main checkout are symlinks to `~/work/rrp-data`, which is being mirrored from the peer now.
  In your worktree, create the same two symlinks.
- Host jobs: `PYTHONPATH=src ~/work/relational-robot-policy/.venv/bin/python -m rrp.cli ops run --cpu X --mem Y [--gpu --gpu-mem G] --label <track>_x --max-seconds N -- cmd`.
  The host broker is shared (2026-09-25 numbers: 11 CPU, 33 GiB, 2 GPU leases across ALL agents; now 3 GPU leases, D-086). Take at most ~3 CPU / 10 GiB / 1 GPU lease
  unless the host is idle. If admission is refused, use the peer.
- Peer (the main compute): ALWAYS `export RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/<track>` first, then `ops/bin/peer_sync.sh push` from your worktree;
  then `ops/bin/peer_run.sh --gpu --gpu-mem 16G --cpu 4 --mem 24G --label <track>_x --max-seconds N [--detach] -- PY -m rrp.cli ...`.
  The peer GB10 is shared by ~6 agents plus the lead's chain: at most 2 concurrent GPU jobs per track, and each <=20 GiB GPU memory.
  CPU-only sim/eval jobs can use more (the peer has 20 cores). Name every output `artifacts/runs/<track>_...`.
  Never run `rrp ops stop` without `--lease <your id>`.
- For long jobs, use `--detach`, then poll the log (`/dev/shm/rrp-brandonin/repo/ops/logs/<lease>_<label>.log` on the peer). Do not sit idle:
  implement the next piece while jobs run.
- Tests: light (see CLAUDE.md). Before any long run, do a tiny smoke run of the real thing.
- Record notes and decisions in `research/tracks/<track>.md`: what you ran (exact command), where the raw outputs are, numbers, and state
  (planned / implementing / test_failed / verified / running / completed / failed_hypothesis / blocked_external). Keep it current;
  it is also your resume file.
- Commit often on `track/<track>`. When something is verified (code plus a smoke run, or a finished result), integrate into main:
  `git fetch origin && git rebase origin/main && git push origin HEAD:main` (retry on a race). Also push your branch.
  Do not edit STATUS.md, research/decisions.md or research/registry.jsonl; the lead folds in your track file.
  Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- The repo is PUBLIC: never commit weights (*.pt), datasets, tokens or third-party assets. Small JSON/JSONL results and
  small labelled videos (artifacts/video/, plus an INDEX.md line) are fine.
- Label sources honestly (scripted_teacher / privileged / learned:<ckpt> / random). Every reported number must come from a
  saved raw output.
- No paid APIs, sudo, network changes, public listeners, or physical robots.
- Finish with a concise report: what is done and verified, the numbers with artifact paths, what is still running
  (lease ids and output paths), and exact next steps.
