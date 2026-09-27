# core track (W11: rrp as an installable core for psi1z, D-100)

Owner: core agent. Worktree `~/work/rrp-wt/core`, branch `track/core`. Started 2026-09-26. No peer use.
Scope: docs/strategy.md W11. Public API and extension guide: `docs/core_api.md`.

State: see the table at the end.

## what changed
- `pyproject.toml`: `requires-python >=3.11,<3.13`. Core deps are numpy and pydantic only; `jsonschema` moved to
  `dev` (only tests/unit/test_contracts.py uses it). Extras: sim (mujoco), ml (torch), service, viz (pillow,
  imageio, imageio-ffmpeg, matplotlib: lazily imported by video/plot code), vlm (transformers, huggingface_hub),
  dev, all. Lower bounds unchanged; nothing pinned tighter.
- Wheel: `src/rrp` + `tasks/*.json` as package data `rrp/_data/tasks` (hatch force-include; the only repo-level data
  the code reads that is not a run artifact). sdist limited to src/rrp, tasks, pyproject, README, docs/core_api.md
  (it was 146 MB with artifacts/ before).
- `rrp.core` (new, layer "core" = 10 in the layering test): lazy (PEP 562) re-export of the stable names,
  `CORE_API_VERSION = "1.0"`. `import rrp.core` needs numpy + pydantic only.
- Extension hooks:
  - `rrp.contracts.runconfig.register_family(name, flag_spec=, default_stage_flags=, legacy_flag_defaults=)`, plus
    entry points in the group `rrp.families` (loaded lazily on the first unknown family name). `RunConfig.family`
    is now a validated `str` (was `Literal["arm","dual","legged"]`); built-ins behave as before. `Pipeline(family)`,
    `register` (stages), `rrp run-dag` (plan) and `python -m rrp.pipelines stages` accept registered families.
  - `rrp.bodies.catalog.register_robot` + entry points `rrp.robots` (workbench robots; built-in keys protected).
- `rrp.contracts.system0.System0Base` / `System0Stats`: the packet acceptance protocol (receive / invalidate / stats)
  that `LatentSystem0` had inline. `LatentSystem0` now subclasses it; its methods are the same code (moved), so
  behaviour is unchanged (DualLatentSystem0 still overrides receive). `System0Stats` is re-exported from
  controllers.latent_realizer.
- `rrp.evaluation.statistics` (additive): `newcombe_diff` (from scripts/armnosem_compare.py `newcombe`, None on
  n=0), `boot_ci` (identical to latent_causal.boot_ci; test asserts equality), `boot_diff` (from armnosem_compare),
  `paired_bootstrap_ci`, `paired_permutation_test` (sign flip, exact for <=16 pairs, else MC (k+1)/(n+1)),
  `permutation_test` (two-sample MC), `mcnemar_exact`. The script copies are unchanged (scripts are frozen).
- `rrp.evaluation.edit_harness` (new, numpy; torch lazily): `EditCondition` (kinds control / replay / semantic /
  irrelevant_control / negative_control, declared prediction), `PacketSource` protocol, `restamp` (the
  `latent_causal.deliver` rule without a MuJoCo session: edits relabel source="debug" + sampling.intervention),
  `run_suite` (JSONL, resumable), `paired_effect` (paired bootstrap CI + sign-flip p), `matched_random`,
  `orthogonal_matched`, `probe_jacobian`, `probe_guided_edit` (generic form of latent_causal.probe_edits). The arm and
  legged suites are unchanged.
- Paths: `rrp.contracts.paths.rrp_home()` = `$RRP_HOME` > the source checkout rrp runs from (unchanged behaviour) >
  cwd (installed rrp). `data_path` falls back to packaged data. Used by: `contracts.provenance.repo_root`
  (+ `code_provenance()` of an installed rrp = the commit in `direct_url.json`, never the cwd's git state),
  `orchestration.runtime.REPO` (RRP_REPO / RRP_OPS_ROOT still override; ops_root still resolves to the main checkout
  on the host, so psi1z leases from the same broker), `envs.scenario.TASKS_DIR`, `bodies.importers.REPO`
  (menagerie), `evaluation.legged_catalog.REPO`, `data.legged_collect` teacher report, `RunIndex.load` (relative
  path without root -> rrp_home, was cwd), `Pipeline.run` default root (rrp_home, was cwd; every caller passes root).

## paths left (not core API; listed, not changed)
- `control/legged_tracker.py` `TRACKER_DIR = parents[3]/artifacts/trackers`: contact agent's file (track/contact).
- `service/app.py`, `service/policy_registry.py` `REPO = parents[3]` (workbench app; runs from the checkout).
- `training/baseline_campaign.py`: `configs/model/*.json` relative to cwd; CLI defaults `--protocol
  configs/eval/latent_slice1.json` (cli/train.py, cli/latent.py) relative to cwd.
- Run ids / native configs (`artifacts/runs/...`) are relative to the pipeline root by design (`Pipeline.run` chdirs
  into `root`; RunIndex.absolute). The built-in arm/legged stages run `scripts/*.py` / `scripts/*.sh` from the root,
  so they need an rrp checkout as root; an external family's stages bring their own code.
- `StageContext.env()` / `run_leased` prepend `<root>/src` to PYTHONPATH (harmless for an installed rrp: it is the
  consumer's src, or absent).
- `scripts/demo/build_page.py` line 1257 uses a 3.12-only f-string (backslash in an expression); the script is frozen
  (demo page must build unchanged) and is not part of the package. Everything in src/ and tests/ compiles under 3.11.

## verification
(commands and counts below; raw logs in ops/logs of the main checkout under the listed leases)
- Envs (uv, no sudo; torch from the PyTorch CPU index, aarch64): `~/work/ext/venvs/rrpcore311` (3.11.15, torch
  2.14.0+cpu, mujoco 3.14.0, numpy 2.4.6) and `~/work/ext/venvs/rrpcore312` (3.12.13, same versions, numpy 2.5.3);
  created under lease 1790487897_3377ed. Both have `rrp[ml,sim,dev] @ git+file://...@<sha>` installed.
- `uv build`: wheel 287 files / 0.6 MB with `rrp/_data/tasks/*.json`; sdist 0.5 MB. A fresh core-only 3.11 venv
  (`pip install "rrp @ git+file:///home/brandonin/work/relational-robot-policy@8bdbb55"`) pulls exactly numpy,
  pydantic (+ pydantic-core, annotated-types, typing-extensions, typing-inspection); from /tmp, `rrp.core` imports,
  `is_checkout()` False, task specs come from the wheel, `code_provenance()` = the installed commit, dirty False.
- `pytest tests/unit` (lease 1790489156_d606d3, same tree): main .venv 3.12: 318 passed / 4 skipped; rrpcore312:
  318 / 4; rrpcore311: 318 / 4 (before W11: 304 passed / 4 skipped on 3.12). Skips (identical in every env):
  test_object_qa (transformers not installed), test_checkpoint_load (no weights in the worktree), 2 x
  test_prev_action_col (no menagerie in the worktree). With `RRP_HOME=~/work/relational-robot-policy
  RRP_CHECKPOINT_ROOTS=~/work/relational-robot-policy/artifacts` those 3 pass under 3.11 (real checkpoints load).
  No test was skipped or weakened for 3.11.
- Found by the 3.11 run: `orchestration.yamlmini.load` used PyYAML when installed, which reads `lr: 5e-05` as a
  STRING (YAML 1.1), so `test_arm_dag_reproduces_legacy_configs` failed in any env with PyYAML (the venvs have it;
  the main .venv does not). `load` now always uses the built-in subset parser (the one every DAG was verified with).
- `tests/conftest.py`: the menagerie skip checks `$RRP_HOME/.cache/assets` when RRP_HOME is set (same path the code
  reads).
- Layering test green (`core` = layer 10). `python -m compileall` under 3.11: src/ and tests/ clean.

## state
| piece | state |
|---|---|
| pyproject / extras / wheel + package data | verified |
| rrp.core + docs/core_api.md | verified (test_core_api: every name imports; core names import with torch/mujoco blocked) |
| family / robot extension hooks (+ entry points) | verified (test with a dummy external family, a dummy entry-point dist and run-dag planning) |
| tests under 3.11 and 3.12 | verified (318 passed / 4 skipped each) |
| path assumptions | fixed in the core modules; remaining list above |
| psi1z pin bump | see below |

Resume: `cd ~/work/rrp-wt/core && git fetch origin && git rebase origin/main`; tests:
`~/work/ext/venvs/rrpcore311/bin/python -m pytest -q tests/unit` (and the 3.12 venv) under `rrp ops run`.
