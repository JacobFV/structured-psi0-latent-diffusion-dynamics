# agent operating notes (repo-level; the full contract is docs/handoff/AGENTS.md)

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

## resources (unchanged, enforced)
- Host: <=50% of currently free resources, no host GPU. Every non-trivial process runs under
  `PYTHONPATH=src python3 -m rrp.cli ops run --cpu X --mem Y --label L -- cmd`.
- Peer (`gb10-direct`): the user authorized using ALL of it (research/decisions.md D-008); one GPU
  job at a time via the broker's GPU slot. Workspace: /dev/shm/rrp-brandonin (sync with scripts/peer_sync.sh push).
- Commit as you go (local commits only; no pushes).
