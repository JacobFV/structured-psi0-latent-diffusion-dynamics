# relational robot policy — autonomous implementation and research handoff

prepared 2026-09-21 utc / 2026-09-21 san francisco. package version 1.0.

this package is the complete assignment for an autonomous coding/research agent running on the user's own computer. it is **not an implemented robot policy, a completed experiment, or an already-installed resource limiter**. the included helpers only inspect the machine and validate the handoff. the agent must build, test, and validate the actual runtime before training.

## start

extract this directory somewhere convenient. open your autonomous agent in this directory and give it `AUTONOMOUS_AGENT_PROMPT.md`, or paste the short launcher below. no ssh hostname needs to be filled in before the agent starts: it must discover the already-configured second gb10 safely and verify its identity. `ROBOT_PEER` may optionally specify a known ssh alias.

```text
Read AUTONOMOUS_AGENT_PROMPT.md and AGENTS.md in this directory. Treat this package as my complete implementation/research assignment and execute it end to end on my two GB10 computers. The host is shared: use no more than 50% of its currently free resources, aggregated across all your processes. Discover the existing SSH-accessible peer; prefer it for GPU work. Implement and verify resource enforcement before heavy work. Build the workbench, task runtime, variable-morphology policy, training and adaptation methods, then actually run the bounded research campaign and produce reproducible results. Do not stop after planning or scaffolding. Make routine decisions autonomously, preserve other work, and never invent measurements or call a failed hypothesis successful.
```

the included helper scripts require python 3.10 or newer. optional, read-only checks:

```bash
python3 scripts/check_package.py
python3 -m unittest discover -s tests -v
python3 scripts/preflight.py --role host --output host-preflight.json
```

`preflight.py` is NOT an enforcement layer and does not authorize a workload. running it does not start training, ssh, install packages, or change system settings. commands such as `rrp campaign run` in the plans are deliverables the autonomous agent will implement, not commands that already exist in this handoff.

## reading order

1. `AUTONOMOUS_AGENT_PROMPT.md`, `AGENTS.md`, `docs/00_scope_and_acceptance.md`.
2. `docs/03_resource_safety.md`, `plans/00_execution_order.md`.
3. architecture, interfaces, simulator, gui, learning, and experiment documents.
4. the four file-level implementation plans, as their dependencies become available.
5. `REQUIREMENTS_TRACEABILITY.md` and release gates before final delivery.

## contents

- `docs/`: technical contracts, experimental design, current source audit, safety and recovery.
- `plans/`: ordered, file-level implementation tasks with test examples and acceptance commands.
- `config/`: machine-readable resource policy, model profiles, campaign and robot catalogue.
- `contracts/`, `examples/`: task-graph schema and executable-semantics examples for the implementation.
- `templates/`: status, run manifest, and results ledger formats. empty results are not successes.
- `reference/original-design-v0.1.md`: supplied design preserved unchanged for provenance.
- `scripts/`, `tests/`: lightweight handoff utilities and their tests.
- `verification/`: what was checked when preparing this package, not robot-policy evidence.

## precedence

latest user instructions > resource/safety policy > master prompt and current scope > technical contracts > implementation plans > original v0.1 design. the original design's “not authorized / written for review” status describes its historical stage. the user's current request authorizes the local autonomous implementation and bounded local experiments described here; it does not authorize paid services, public publication, system-wide reconfiguration, or physical robot actuation.

research targets are deliberately falsifiable. a usable implementation plus an honestly negative research result is acceptable; an unfinished implementation described as complete is not. broad robot coverage includes separate asset, controller, teacher, and learned-policy statuses—never collapse these into “supported”.
