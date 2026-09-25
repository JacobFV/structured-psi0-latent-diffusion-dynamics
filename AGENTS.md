# agent operating notes (repo-level; the full contract is docs/handoff/AGENTS.md)

## AUTHORITATIVE CORRECTION (2026-09-21) — read research/corrections/controller-facing-semantic-latent.md first
System i generates a structured continuous latent packet z[b,knots,assemblies,d]; the semantic objectives are supervised
ON THAT PACKET; system 0 consumes THE SAME packet with morphology + current proprio/local sensors + phase and emits
native commands online. Acceptance evidence = probes of the received packet, not hidden-state probes. The old
direct-action FlowPolicy and the action-only codec are BASELINES ONLY (never present them as the corrected architecture).
Do not schedule old-path runs except as named baselines. Integration branch: `main` (the correction branch is merged).

## testing policy for this research stage (user instruction, 2026-09-21)
Be much lighter on tests. At this stage we care about RESULTS; implementations get hardened and
cleaned later, once they have proved their worth.
- Do not spend more effort on tests than on implementation. Prefer one quick smoke run of the real
  thing (a short training step, a few rollouts) over writing new unit-test suites.
- Write tests only where a silent bug would invalidate results: math that defines likelihoods/losses,
  public vs privileged leakage, resource-safety enforcement, split/lineage leakage.
- Do not re-run the whole suite after every edit; run the few tests touching what you changed.
- Never fake results, never hide failures, and keep teacher/privileged/learned labels honest.
  Lighter testing is not permission for unverified claims: every reported number must come from an
  actual run whose raw output is saved.

## demo videos (user instruction, 2026-09-21)
For every notable checkpoint (new learned policy, adaptation result, new body/task/teacher), render a
short video (≈10-30 s, peer GPU EGL render) of representative episodes — include at least one failure,
label the controller source (scripted_teacher / learned:<ckpt>) in the filename and caption, and save under
artifacts/video/<date>_<what>.mp4 with a line in artifacts/video/INDEX.md (checkpoint, robot, task, seed,
outcome). Keep them small so they can be committed.

## resources (D-033, 2026-09-25; enforced)
- Host: up to 80% of currently FREE CPU cores and memory (other users share it), plus reserves; host GPU allowed
  (up to 2 GPU leases, in-process memory cap + watchdog). Every non-trivial host process runs under
  `PYTHONPATH=src .venv/bin/python -m rrp.cli ops run --cpu X --mem Y [--gpu --gpu-mem G] --label L -- cmd`.
- Peer (`gb10-direct`): ALL of it (D-008/D-026); the broker is a registry only, so several GPU jobs may share the
  GB10 (pack small jobs; watch memory). Keep it busy.
- Peer code dirs: the lead's checkout syncs to /dev/shm/rrp-brandonin/repo (running chains live there; never push a
  different tree into it). Every other agent/worktree uses its OWN dir:
  `export RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/<track>; scripts/peer_sync.sh push;`
  `scripts/peer_run.sh --gpu --gpu-mem 12G --cpu 4 --mem 24G --label <track>_x --max-seconds N [--detach] -- PY -m rrp.cli ...`
  (artifacts/ and .cache/ in that dir are symlinks to the shared store; one shared broker via RRP_OPS_ROOT).
  Name outputs uniquely (artifacts/runs/<track>_...); never overwrite another track's run directory.
- Parallel agents: work on a branch in a git worktree; commit often; put decisions/notes in
  research/tracks/<track>.md (the lead folds them into research/decisions.md) and merge to `main` when verified.
- Commit as you go and push to the remote `origin` (github.com/JacobFV/structured-psi0-latent-diffusion-dynamics; renamed from relational-robot-policy 2026-09-25). The repo is PUBLIC by design (user intends to share it, D-030): write docs/commits for outside readers, and never commit tokens, weights, third-party assets or datasets.
- NEVER run `rrp ops stop` without `--lease <your lease id>`: other engineers' jobs share the project slice.
  (2026-09-21 incident: an unscoped stop killed every running peer job, including the lead's training.)
