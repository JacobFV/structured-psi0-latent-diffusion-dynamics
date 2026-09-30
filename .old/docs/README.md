# .old/docs — retired documents (D-145 purge unit P8)

Old path `docs/...`. Nothing live reads this directory.

| path | what it was | cited by |
|---|---|---|
| `handoff/` | the original assignment package (2026-09): `AGENTS.md` (historical contract), `AUTONOMOUS_AGENT_PROMPT.md`, `docs/00..10` (scope, architecture, interfaces, resource safety, sim / robots, workbench, learning, research protocol, tests, sources, autonomy and recovery), `contracts/` (JSON schemas), `templates/`, `plans/`, `reference/`, `verification/`, `SHA256SUMS` of the original bytes | `research/decisions.md` (D-001..D-030); code docstrings ("archived handoff doc ..."). The still-binding rules are folded into the root `AGENTS.md` |
| `demo/` | the sprint demo page (2026-09-26): `index.html`, the private-artifact build `artifact.html`, `raw/` copies of the cited result files, `video/` | D-078, D-095; built by `.old/scripts/demo/build_page.py`; the raw results themselves are in `artifacts/runs/` |
| `architecture_s7-10.md` | sections 7-10 of `docs/architecture.md`: D-140 delete list, S0-S6 migration table, Psi0 / ComputerWorld hand-off checklist, old-to-new module path map (`rrp.controllers.*` -> `rrp.policies.*`, ...) | D-140; use section 10 to resolve a module name in an old note or run config |

Two fixtures the unit tests still need were COPIED out (the originals stay in `handoff/`): `contracts/task_graph.schema.json` and `examples/support_and_insert.task.json` are now `tests/data/`.

Moved in by D-145 unit P8 (`git log --follow -- .old/docs/<path>`).
