# brief for parallel track agents (2026-09-25, lead: main session)

Read first: `CLAUDE.md` (operating rules; resources D-033), `STATUS.md`, `research/corrections/controller-facing-semantic-latent.md`,
`research/reports/latent_slice1_progress.md`, and `research/decisions.md` D-029..D-033.

## the big picture
We are building and testing ONE architecture: system i (a flow model) generates a structured continuous latent packet
z[knots=4, assemblies, 64]. Semantic objectives supervise THAT packet. System 0 consumes the SAME packet together with
morphology, current proprioception/touch and phase, and emits native joint targets every 50 ms. Evidence of meaning must come
from probes and causal edits of the received packet, not from hidden states. The question the sealed protocol
(`configs/eval/latent_slice1.json`) asks: does semantic supervision on the packet improve new-body transfer and adaptation
(xarm7_pg2, xarm7_tf3, panda_tf3), compared with a capacity-matched no-semantic latent, the action-only codec and direct actions,
at <=25% p95 latency overhead? No learned policy has been shown competent in closed loop yet. Honest negative results are
results.

Known state:
- Stage A representations `artifacts/runs/latent_{sem,nosem}_v1`. Semantic z is decodable (D-031), but the
  **counterexample test FAILS** (D-032): the encoder reads the task object from the trajectory and ignores the supplied task binding.
- Stage B v2 flows (standardized target) are training in the lead's chain on the peer (`scripts/latent_chain_v2.sh`). Outputs:
  `artifacts/runs/flow_latent_{sem,nosem}_v2`, then `flow_latent_sem_v3`, each followed by eval_dev.jsonl, disturbance.jsonl and videos.
  Do not touch those directories or `/dev/shm/rrp-brandonin/repo`.

## how to work
- Repo: `~/work/relational-robot-policy` (the lead's checkout, on `main`; do not edit files there). Create your own worktree:
  `git -C ~/work/relational-robot-policy worktree add ~/work/rrp-wt/<track> -b track/<track> origin/main`, and work only in it.
- Host python: `~/work/relational-robot-policy/.venv/bin/python` with `PYTHONPATH=src` run from your worktree. Host data:
  `artifacts/datasets` and `artifacts/packed` in the main checkout are symlinks to `~/work/rrp-data`, which is being mirrored from the peer now.
  In your worktree, create the same two symlinks.
- Host jobs: `PYTHONPATH=src ~/work/relational-robot-policy/.venv/bin/python -m rrp.cli ops run --cpu X --mem Y [--gpu --gpu-mem G] --label <track>_x --max-seconds N -- cmd`.
  The host broker is shared (11 CPU, 33 GiB, 2 GPU leases in total across ALL agents). Take at most ~3 CPU / 10 GiB / 1 GPU lease
  unless the host is idle. If admission is refused, use the peer.
- Peer (the main compute): `export RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/<track>`; `scripts/peer_sync.sh push` from your worktree;
  then `scripts/peer_run.sh --gpu --gpu-mem 16G --cpu 4 --mem 24G --label <track>_x --max-seconds N [--detach] -- PY -m rrp.cli ...`.
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

## DEMO SPRINT (2026-09-25 19:00 → 2026-09-26 05:00 PDT) — read this section first; it overrides older track scopes
Goal: ready-to-demonstrate, HONEST artifacts in 10 h. Scope is frozen: no new variants beyond what is listed for your track.
The state is in research/decisions.md D-044..D-049 and research/reports/evidence_matrix.md. Key facts:
- Bug B-1 (D-044/D-045): train with `"zero_prev_action": true`. Anything trained without it is contaminated; never use it.
- No learned route is competent yet: oracle route (teacher-encoded packet → system 0) is at best 1/30 (D-047/D-048); the
  arm drives toward the cube rather than the packet's pregrasp waypoint (D-049). Re-anchoring the oracle expert every
  tick is INVALID (D-049); the default every 8 ticks is valid.
- In flight (do not duplicate): binding v4 chains (peer units rrp-binding-chain-v4-{sem,nosem}, dir wt/binding) →
  representations `artifacts/runs/binding_paired_{sem,nosem}_v4` → probes → counterfactuals → flows (latent_chain_v2) →
  dev eval → eval-binding; auto oracle-ladder evals (units rrp-ladder-wait-bindv4{sem,nosem}, tags bindv4sem/bindv4nosem);
  plain BC with fix on host (unit rrp-b1fix-baseline_direct_action, root artifacts/runs/latent_slice1_b1fix);
  codec baseline on peer (unit rrp-b1fix-codec, dir wt/lead).
Resources: the user wants ≥80% of the host and 100% of the peer GPU/memory in use continuously. The HOST GPU is mostly idle,
so use it (host budget ~15 CPU / 34 GiB incl. declared GPU memory; the BC job holds 16G GPU + 10G). Use peer CPU for simulation evals.
Every finished result gets: the raw JSON/JSONL path, a line in your research/tracks/<track>.md, and a short labelled video
(artifacts/video/<date>_<source>_<what>.mp4 + INDEX.md line; include a failure). Merge verified work to main often.
Labels: scripted_teacher (privileged) / oracle (teacher-encoded packet: DIAGNOSTIC, not deployable) / learned:<ckpt>.
Report to the lead by finishing with a summary. Do not stop early: when your list is done, pick the next highest-value
item for the demo within your scope, and say what you picked.
