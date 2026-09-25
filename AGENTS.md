# agent operating notes (repo-level; the full contract is docs/handoff/AGENTS.md)

## AUTHORITATIVE CORRECTION (2026-09-21) — read research/corrections/controller-facing-semantic-latent.md first
System i generates a structured continuous latent packet z[b,knots,assemblies,d]; the semantic objectives are supervised
ON THAT PACKET; system 0 consumes THE SAME packet with morphology + current proprio/local sensors + phase and emits
native commands online. Acceptance evidence = probes of the received packet, not hidden-state probes. The old
direct-action FlowPolicy and the action-only codec are BASELINES ONLY (never present them as the corrected architecture).
Do not schedule old-path runs except as named baselines. Work on branch correction/controller-facing-latent.

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

## resources (unchanged, enforced)
- Host: <=50% of currently free resources, no host GPU. Every non-trivial process runs under
  `PYTHONPATH=src python3 -m rrp.cli ops run --cpu X --mem Y --label L -- cmd`.
- Peer (`gb10-direct`): the user authorized using ALL of it (research/decisions.md D-008); one GPU
  job at a time via the broker's GPU slot. Workspace: /dev/shm/rrp-brandonin (sync with scripts/peer_sync.sh push).
- Commit as you go and push to the PRIVATE remote `origin` (github.com/JacobFV/relational-robot-policy); never make it public.
- NEVER run `rrp ops stop` without `--lease <your lease id>`: other engineers' jobs share the project slice.
  (2026-09-21 incident: an unscoped stop killed every running peer job, including the lead's training.)
