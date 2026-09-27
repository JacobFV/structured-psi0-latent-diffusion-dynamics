# agent operating contract (current; applies to every agent and resumed session)

This file is the single source of the operating rules. `CLAUDE.md` imports it. `docs/handoff/AGENTS.md` is the
HISTORICAL contract from the original assignment: where the two differ (resources above all), this file wins; the
decisions cited here record the user's changes. What we are doing and in what order: `docs/strategy.md` (D-094).

## architecture (authoritative correction 2026-09-21, D-029)
Read `research/corrections/controller-facing-semantic-latent.md` first. System i generates a structured continuous latent
packet z[b, knots, assemblies, d]; the semantic objectives are supervised ON THAT PACKET; system 0 consumes THE SAME packet
with morphology + current proprio/local sensors + phase and emits native commands online. Acceptance evidence = probes and
causal edits of the received packet, not hidden-state probes. The old direct-action FlowPolicy and the action-only codec
are BASELINES ONLY. Integration branch: `main`.

## resources (enforced by the broker and watchdog)
- **Host** (shared with other users): aggregate ≤80% of currently FREE CPU cores and ≤80% of free memory, plus reserves
  (D-027, D-033); host GPU allowed, 3 GPU leases, each job applies the in-process CUDA cap to its declared GPU memory
  (D-027, D-054, D-086); disk: keep ≥100 GB free (D-086; was 300 GB, D-036); declare downloads/transfers with
  `ops run --disk` (D-036). The limit is an aggregate over all agents, never a per-job allowance. Host memory is also
  squeezed by other projects' processes: keep host jobs ≤24 GiB and expect the memory-PSI watchdog to shed them.
  Every non-trivial host process runs under
  `PYTHONPATH=src .venv/bin/python -m rrp.cli ops run --cpu X --mem Y [--gpu --gpu-mem G] [--disk D] --label L -- cmd`.
- **Peer** (`gb10-direct`): ALL of it (D-008, D-026, D-033). The broker is a registry only; several GPU jobs may share
  the GB10 (pack small jobs, watch memory). The watchdog acts only on emergencies. Keep it busy.
- **Host vs peer (D-115, user):** NO training, simulation evals, sweeps, data collection, rendering or other heavy compute on the host. Everything heavy runs on the peer. The host is for editing, git, unit tests, small analysis and orchestration only (its broker is capped at 2 CPU / 8 GiB / 0 GPU).
- **Peer code dirs:** the lead's checkout syncs to `/dev/shm/rrp-brandonin/repo` (running chains live there). Every other
  agent/worktree uses its OWN dir. ALWAYS export `RRP_PEER_REPO` before `scripts/peer_sync.sh` (D-090 incident):
  `export RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/<track>; scripts/peer_sync.sh push`, then
  `scripts/peer_run.sh --gpu --gpu-mem 12G --cpu 4 --mem 24G --label <track>_x --max-seconds N [--detach] -- PY -m rrp.cli ...`.
  Never sync into a dir whose jobs are running. Name outputs `artifacts/runs/<track>_...`; never overwrite another track's run.
  `scripts/peer_sync.sh push` now enforces this (D-096): it refuses without RRP_PEER_REPO, refuses the shared `repo` dir unless RRP_ALLOW_SHARED_REPO=1, and refuses a dir that running jobs use as their cwd.
- NEVER run `rrp ops stop` without `--lease <your lease id>` (2026-09-21 incident: an unscoped stop killed every peer job).
  Launch loops must check exit codes and be bounded (D-061).
- No paid compute/API calls, sudo/global upgrades, network reconfiguration, public listeners or physical robot commands.

## repository
- Public repo `github.com/JacobFV/structured-psi0-latent-diffusion-dynamics` (renamed from relational-robot-policy,
  D-030; public by design). Never commit tokens, weights (`*.pt`), datasets (`*.npz`, `*.pkl`, packed data) or
  third-party assets. Small JSON/JSONL results and small labelled videos are fine.
- Parallel agents: own worktree `~/work/rrp-wt/<track>` on `track/<track>`, notes in `research/tracks/<track>.md`
  (the lead folds them into `research/decisions.md`), merge verified work with
  `git fetch origin && git rebase origin/main && git push origin HEAD:main`. Workflow details: `research/tracks/BRIEF.md`.
- Read reference repositories; do not mutate their working trees, jobs or branch histories.

## research contract (from the handoff; still binding)
- Maintain `STATUS.md`, `research/registry.jsonl`, `research/decisions.md`, `ops/resource-ledger.jsonl`, `artifacts/requirements.json`.
- Progress states: planned, implementing, test_failed, verified, running, completed, failed_hypothesis, blocked_external,
  budget_exhausted. "Verified" requires recorded commands and artifacts.
- Task/event knowledge may be supplied. Future physical outcomes and simulator ground truth may not silently enter deployable observations.
- Stable instance/version provenance, role order/multiplicity, null identities and failure reasons are part of the runtime representation.
- Probes and attention maps are diagnostics. Demonstrate causal use through edits and rollouts.
- Same input information and fair acquisition accounting for competing methods. Report existing-controller transfer
  separately from new controller training.
- Train real models. Label sources unmistakably in UI and reports: scripted_teacher / privileged / oracle (diagnostic) /
  learned:<ckpt> / bc / random / mock.
- Make reversible routine decisions autonomously; ask only for genuinely unavailable permissions/credentials.
- At context/session limits, checkpoint state and write exact resume steps. Do not restart completed experiments.
- No unbounded search or fake success (`docs/handoff/docs/10_autonomy_and_recovery.md`).

## testing (user instruction 2026-09-21: light testing at the research stage)
- Prefer one quick smoke run of the real thing over new unit suites. Before any long run, a tiny real run.
- Write tests where a silent bug would invalidate results: loss/likelihood math, public vs privileged leakage,
  resource-safety enforcement, split/lineage leakage.
- `pytest tests/unit` must pass in a fresh checkout (asset/data-dependent tests skip via `tests/conftest.py` markers).
- Lighter testing is not permission for unverified claims: every reported number comes from a saved raw output.

## demo videos (user instruction 2026-09-21)
For every notable checkpoint render a short (≈10–30 s) video of representative episodes, including at least one failure;
label the controller source in the filename and caption; save under `artifacts/video/<date>_<source>_<what>.mp4` with a
line in `artifacts/video/INDEX.md` (checkpoint, robot, task, seed, outcome). Keep them small.
