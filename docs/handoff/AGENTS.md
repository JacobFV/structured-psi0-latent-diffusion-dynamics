# agent operating contract

The master prompt is `AUTONOMOUS_AGENT_PROMPT.md`. This file is the short, persistent contract for all child agents and resumed sessions.

- Protect existing work. The host's limit is aggregate ≤50% of CURRENTLY FREE resources, plus reserves. Never interpret it as a per-job allowance. All child agents acquire leases from one broker.
- The handoff's preflight helper only measures. It does not install or enforce limits. Heavy jobs require tested enforcement and a live watchdog.
- Default host GPU work is off. Heavy training/rendering goes to the verified peer. Do not take over its preexisting work either.
- No paid compute/API calls, sudo/global upgrades, network reconfiguration, public listeners, remote publication, or physical robot commands.
- Use a dedicated repository and isolated environments. Read reference repositories; do not mutate their working trees, jobs, or branch histories.
- Implement the WHOLE declared scope. A first toy demo or small profile is a gate, not the final deliverable.
- Maintain `STATUS.md`, `research/registry.jsonl`, `research/decisions.md`, `ops/resource-ledger.jsonl`, and `artifacts/requirements.json`.
- Required progress states: planned, implementing, test_failed, verified, running, completed, failed_hypothesis, blocked_external, budget_exhausted. “Verified” requires recorded commands and artifacts.
- Task/event knowledge may be supplied. Future physical outcomes and simulator ground truth may not silently enter deployable observations.
- Stable instance/version provenance, role order/multiplicity, null identities, and failure reasons are part of the runtime representation, not afterthoughts.
- Probes and attention maps are diagnostics. Demonstrate causal use through edits and rollouts.
- Use the same input information and fair acquisition accounting for competing methods. Report existing-controller transfer separately from new controller training.
- Train a real model. Mark teacher, scripted, privileged, random, mock, and learned sources unmistakably in the UI and reports.
- Red/green tests for code; unit math tests for likelihoods; tiny physics rollouts before long runs; fresh audit before published conclusions.
- Make reversible routine decisions autonomously. Ask only for genuinely unavailable permissions/credentials; continue independent work when one track is blocked.
- At context/session limits, checkpoint state and write exact resume steps. Do not restart completed experiments merely because the conversation reset.
- Follow `docs/10_autonomy_and_recovery.md` for budget, retries, and closure. No unbounded search or fake success.
