# agent operating contract (current; applies to every agent and resumed session)

This file is the single source of the operating rules. `CLAUDE.md` imports it. The original assignment package
(handoff contract, master prompt, docs 00-10) is archived under `.old/docs/handoff/`; where it differs from this file
(resources above all) this file wins, and the decisions cited here record the user's changes. What we are doing and in
what order: `docs/strategy.md` (D-094).

## layout (D-145; `schema.toml`, docs/architecture.md section 13, `tests/unit/test_layout.py`)
Every tracked path must match `schema.toml`. `src/` code, `tests/`, `recipes/` (templates + thin per-track instances =
every run definition; `rrp run-dag`), `docs/` (five files), `research/` (decisions, registry, notes of the tracks in
`schema.toml [tracks]`), `artifacts/` (small raw evidence, append-only), `ops/`, `viz/` (the room), `.old/`. Anything
not in the schema moves under `.old/<same path>` with its group README line (`.old/README.md`); live code never
imports or reads `.old/`. A track that closes moves its note and recipes to `.old/` and leaves `[tracks]`; new runs are
`artifacts/runs/<track>/<lineage>/<stage>[-tag]_s<seed>/`. Do not add top-level areas, per-run config files or
one-off scripts (rendered `RunConfig` comes from a recipe).

## architecture (authoritative correction 2026-09-21, D-029)
Read `research/corrections/controller-facing-semantic-latent.md` first. System i generates a structured continuous latent
packet z[b, knots, assemblies, d]; the semantic objectives are supervised ON THAT PACKET; system 0 consumes THE SAME packet
with morphology + current proprio/local sensors + phase and emits native commands online. Acceptance evidence = probes and
causal edits of the received packet, not hidden-state probes. The old direct-action FlowPolicy and the action-only codec
are BASELINES ONLY. Integration branch: `main`.

## resources (enforced by the broker and watchdog)
- **Host** (shared with other users; D-147 caps below override the percentages): aggregate ≤80% of currently FREE CPU cores and ≤80% of free memory, plus reserves
  (D-027, D-033); host GPU allowed, 3 GPU leases, each job applies the in-process CUDA cap to its declared GPU memory
  (D-027, D-054, D-086); disk: keep ≥100 GB free (D-086; was 300 GB, D-036); declare downloads/transfers with
  `ops run --disk` (D-036). The limit is an aggregate over all agents, never a per-job allowance. Host memory is also
  squeezed by other projects' processes: keep host jobs ≤24 GiB and expect the memory-PSI watchdog to shed them.
  Every non-trivial host process runs under
  `PYTHONPATH=src .venv/bin/python -m rrp.cli ops run --cpu X --mem Y [--gpu --gpu-mem G] [--disk D] --label L -- cmd`.
- **Peer** (`gb10-direct`): ALL of it (D-008, D-026, D-033). The broker is a registry only; several GPU jobs may share
  the GB10 (pack small jobs, watch memory). The watchdog acts only on emergencies. Keep it busy.
- **One repository (D-140):** rrp holds every line of work, including the Ψ₀ line (`rrp.policies.psi0`, `rrp.envs.simple`; the former psi1z repo is archived, its decisions are appendix P of `research/decisions.md`). Name glossary (Ψ₀ upstream vs system 0 etc.): `docs/architecture.md` section 11.
- **No self-reproduction (D-140, owner):** do not re-run our own experiments to reproduce results. Unless there is a concrete reason for suspicion (a bug found, a changed code path, numbers that contradict each other), trust recorded results. Only small vibe-checks where absolutely necessary (a few episodes, minutes, not hours). This applies to refactors too: verify with golden/unit tests, not reruns.
- **Refactor stance (D-140, owner):** refactor aggressively; prefer deleting and merging over adding. Avoid file sprawl: no new module for what fits in an existing one, no compat shims or deprecated aliases kept around, no one-off scripts left behind.
- **Memory declarations (D-117):** declare ≥ 1.35 × measured peak memory (plus CUDA/unified bytes). The lease soft cap is 0.8 × declared, so "+20%" throttles the job.
- **After a host reboot (D-133):** the host watchdog is a transient unit; restart it with `RRP_NODE=host PYTHONPATH=src .venv/bin/python -m rrp.cli ops start-watchdog`, or host leases stay unadmitted. Host-side coordinators (run-dag units, chains, queues) must also be relaunched.
- **Host training allowed under hard caps (owner 2026-10-01, D-147 addendum; supersedes D-115 / D-127):** "you can also use this machine for training. just be careful to not crash it (mem+gpu mem+cpu usage safety)". The host broker (`ops/resources.local.json` `host.enforced`) admits at most 8 CPU, 32 GiB of declared memory for ALL host leases together INCLUDING GPU memory (the host is a GB10: unified memory), 1 GPU slot; the watchdog keeps a 24 GiB free-RAM reserve and sheds on PSI/OOM pressure (`rrp-watchdog-host.service` must be active; D-133). Host jobs declare memory >= 1.35 x the measured peak INCLUDING GPU memory; a run-dag node goes to the host with `placement: host` (a thin instance recipe, e.g. `recipes/humanoid/trackers_gap_ring_host.yaml`). Watch free RAM and GPU temperature after a launch and back off below the reserve or above ~85 C. The host has had freeze-ups/OOM before and other projects run on it. Coordinators on the host stay light (shell loops / run-dag). Agent concurrency is not capped (owner, 2026-09-30, D-143): run as many agents as are useful; fan-out implementation work in the established codebase goes to Sonnet agents/workflows (faster, cheaper), design/architecture to the lead model.
- **Host vs peer:** the peer stays the default for heavy compute (all of it is ours); the host takes training / evaluation only under the caps above. Unit tests, git and editing as before.
- **Peer code dirs:** the lead's checkout syncs to `/dev/shm/rrp-brandonin/repo` (running chains live there). Every other
  agent/worktree uses its OWN dir. ALWAYS export `RRP_PEER_REPO` before `ops/bin/peer_sync.sh` (D-090 incident):
  `export RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/<track>; ops/bin/peer_sync.sh push`, then
  `ops/bin/peer_run.sh --gpu --gpu-mem 12G --cpu 4 --mem 24G --label <track>_x --max-seconds N [--detach] -- PY -m rrp.cli ...`.
  Never sync into a dir whose jobs are running. Name outputs `artifacts/runs/<track>_...`; never overwrite another track's run.
  `ops/bin/peer_sync.sh push` now enforces this (D-096): it refuses without RRP_PEER_REPO, refuses the shared `repo` dir unless RRP_ALLOW_SHARED_REPO=1, and refuses a dir that running jobs use as their cwd.
- NEVER run `rrp ops stop` without `--lease <your lease id>` (2026-09-21 incident: an unscoped stop killed every peer job).
  Launch loops must check exit codes and be bounded (D-061).
- No paid compute/API calls, sudo/global upgrades, network reconfiguration or physical robot commands. Public listeners (e.g. the viz room on a LAN interface) are allowed (owner, D-132); keep them read-only.

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
- Implement the WHOLE declared scope; a first toy demo or small profile is a gate, not the deliverable. Use a dedicated
  repository and isolated environments; the resource preflight only measures, heavy jobs need tested enforcement (broker
  leases + watchdog).
- Budgets and evidence (from the handoff): one owner (the lead) for the broker, the integration branch, the run registry
  and final claims; every worker shares one budget and requests leases (agent count is not permission for more GPU owners
  or CPU threads). Each task ends with changed files, test commands and results, artifact paths, source version and
  remaining risks. Keep `STATUS.md` short and current; no secrets in `STATUS.md`, `ops/`, `research/` or `artifacts/`.
- Restart: read `STATUS.md` and broker state first, check whether an owned job still runs and attach instead of
  starting a duplicate (validate PID start time / cgroup ownership; a dropped SSH does not mean the remote job died).
  Checkpoints are written atomically (checksum + source/config/model/optimizer/RNG state + data cursor); keep the last
  known-good one until the new one verifies; register local and remote artifact paths explicitly; never overwrite newer
  remote results.
- Failure taxonomy: infrastructure (reproduce minimally, isolate the environment, fix with a regression check; no
  global upgrades or guessed CUDA flags); contract (unit, mask, identity, role order, stale cache, privileged leakage:
  stop contaminated runs, fix the boundary, label earlier results invalid); identifiability (add the legitimate
  observation or reduce the claim; privileged labels cannot repair indistinguishable inputs); optimization (check
  losses, gradients, normalizations, teacher coverage, small overfit; bounded variants; keep every failure);
  hypothesis (report it, keep the simpler working method, do not redefine success after seeing the test); external
  blocker (record the exact dependency and continue independent work; no paid cloud, no physical robots).
- Do not stop at a proposal, scaffold, first video or first negative result; do not rerun an unchanged failing job. Each
  new expensive run needs a specific hypothesis, cost profile and remaining-budget check; memory / disk / thermal
  emergencies override the experiment. At a ceiling, leave a complete resumable status and mark incomplete mandatory
  work; do not renew silently. No unbounded search or fake success.
- Final report of a delivery: where things live, exact launch command, model / controller scope, tests actually run,
  checkpoints actually produced, primary result with limitations, resource totals, remaining gaps, how to resume.
  Never claim readiness for real hardware.

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
