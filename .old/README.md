# .old — legacy material (D-145)

Everything here is RETIRED: configs, DAGs, scripts, notes and pages of closed tracks and of the pre-schema layout,
kept for provenance only. Rules (schema.toml, tests/unit/test_layout.py, docs/architecture.md section 13):

- A file keeps its old path under `.old/` (`configs/ladder/x.json` → `.old/configs/ladder/x.json`), so every path
  cited in `research/decisions.md`, reports or run provenance resolves by prefixing `.old/`.
- Live code never imports or reads anything here (no `.old/` string in `src/`, `tests/`, `recipes/`, `ops/bin/`).
  To reuse something, re-express it in the current schema (a recipe, a registry entry, a test fixture).
- Each group directory has a `README.md`: what the group was, which decisions cite it, the commit that moved it.
- Raw results stay in `artifacts/` (the evidence store); they are not legacy.

| group | what it was | index |
|---|---|---|
| `.old/configs/` | hand-written per-run JSON configs (ladder, latent, legged, adapt, data, model, vlm, t1_diag) | `.old/configs/README.md` |
| `.old/dags/` | per-lineage DAG copies superseded by `recipes/templates/` + instances | `.old/dags/README.md` |
| `.old/scripts/` | finished drivers, renderers and the demo page builder | `.old/scripts/README.md` |
| `.old/research/` | notes and working files of closed tracks, one-off analysis scripts, superseded reports, naming map | `.old/research/README.md` |
| `.old/docs/` | the original handoff package, the sprint demo page, the D-140 migration tables | `.old/docs/README.md` |
| `.old/tests/` | frozen pre-move scripts used by retired parity tests | `.old/tests/README.md` |
| `.old/ui/` | the loopback workbench React UI (retired, replaced by the viz room) | `.old/ui/README.md` |
| `.old/src/` | the loopback workbench service (`rrp.viz.workbench`) | `.old/src/README.md` |
| `.old/ops/` | initial host preflight record | `.old/ops/README.md` |
