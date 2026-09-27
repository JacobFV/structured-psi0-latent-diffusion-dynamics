# restructure track (W4: package restructure with shims, phase 2 of the audit)

Owner: restructure agent. Worktree `~/work/rrp-wt/restructure`, branch `track/restructure`. Started 2026-09-26.
Scope: docs/strategy.md W4; docs/repo_structure_audit.md "Target structure" and "Migration plan" phase 2.
Rule: moves only, no behaviour change (numerics, defaults, config semantics, on-disk formats unchanged).

State: **implementing** (phase table below).

## how the moves work
- `research/scripts/2026-09-26/w4_move.py old.mod=new.mod ...` does `git mv`, writes a shim at the old path and rewrites
  `rrp` import statements (only import statements, never string literals) in `src/` and `tests/`. `scripts/` and
  `configs/` are NOT rewritten: they keep using old paths through the shims.
- A shim makes the old path the SAME module object (`sys.modules[__name__] = importlib.import_module(new)`) and emits a
  DeprecationWarning. So private names, monkeypatching through the old path (e.g. `scripts/ladder_localize.py` patches
  `rrp.control.latent_realizer.realizer_node_feats`) and pickles that name the old path keep working.
  Shims of modules with a `__main__` block forward `python -m <old>` with `runpy.run_module(new, run_name="__main__")`.
- Files keep their depth (`src/rrp/<pkg>/<mod>.py`), so the `Path(__file__).parents[3]` repo-root lookups are unchanged.
- Consequence for other branches: a moved file's old path now holds the shim, so a branch that edits the old file gets
  a conflict on rebase. Port the edit to the new path (the shim's docstring names it) and keep the shim.

## layering (tests/unit/test_layering.py)
contracts 0 → physics 1 → bodies 2 → tasks 3 → envs 4 → features 5 → teachers, models 6 → controllers 6.5 → data 7 →
evaluation 8 → training 8.5 → pipelines 9 → orchestration, service, cli 10; research 11 (nothing imports it).
The test parses every real module (lazy imports included) and checks: no upward import (except `KNOWN`, which only
shrinks and fails when stale, and `ALLOWED`), an acyclic package graph, nothing imports `rrp.research`, new code does not
import shim paths, every shim matches `PLANNED` (the full old → new map), and every real module in a legacy package
(`control, sim, morphology, model, learning, policy, ops, cli_*`) is still PENDING in `PLANNED`.

Deviations from the audit's target text, forced by the layering rule (recorded here, not silently):
- `control/ik.py` → `bodies/ik.py` and `control/joint_targets.py` → `envs/joint_targets.py`: the sim sessions (envs)
  embed IK and the joint-target executor, and envs sit below controllers.
- Legged trackers (`legged_tracker`, `tracker_nets`) are planned for `envs/` for the same reason (LeggedSession embeds
  the tracker). P6.
- `service/` stays a top-level package at the cli layer (the workbench app); `policy/registry.py` moves into it
  (breaks policy↔service).
- `policy/runner.py`, `policy/latent_runner.py` (system i at deployment) → `controllers/`.
- The orchestrator↔workload contract (`apply_cap`, `CheckpointSignal`) → `contracts/workload.py`, because training
  code calls it and training sits below orchestration.
- evaluation is below training (training runs evaluation rollouts; evaluation only loads trained models, through
  loaders that move to controllers/data).
- `evaluation/latency.py` → `ops.runtime.make_broker` is ALLOWED (records active leases next to a latency measurement).

## on-disk formats
- Dataset episode pickles (`*.public.pkl.gz`) and ladder `*.genctx.pkl` store `rrp.data.features.PolicyInput`.
  `PolicyInput.__module__` stays pinned to `rrp.data.features`, so newly written pickles are byte-compatible with older
  checkouts; `rrp/data/features.py` must therefore stay as a permanent alias (not removed in audit phase 5).
- All 347 local `*.pt` files (main checkout artifacts, ~/work/rrp-data, all worktrees' artifacts) were scanned with
  pickletools: none references an rrp class (state dicts, tensors, numpy, plain containers only).
- Strings that are data are untouched, e.g. `"rrp.morphology.generators"` in RobotSpec lineage (feeds spec hashes).

## phases
| phase | content | state |
|---|---|---|
| P1 | skeleton packages; contracts (psi, workload), physics (snapshot), features (featurizer, multi, legged, derived) | merged 0882cce |
| P2 | bodies, envs, teachers, controllers, models | merged 4a9030e |
| P3 | data, training, evaluation (break evaluation↔training) | merged 3788f57 |
| P4 | CLI package without silent ImportError; ops → orchestration | verified |
| P5 | research diagnostics; dedup | planned |
| P6 | legged files excluded until W1 (contact) merges | blocked on W1 |

### P1 (contracts / physics / features)
- moves: control.psi_contracts → contracts.psi; sim.snapshot_contract → physics.snapshot; data.features →
  features.featurizer; data.features_multi → features.multi; control.legged_latent → features.legged;
  ops.gpu → contracts.workload (+ `CheckpointSignal` from ops.jobs, re-exported there).
- extracted unchanged (old names re-exported): `featurizer_for` (data.collect → features.featurizer);
  `local_sensors`, `active_operator`, `OPERATORS` (learning.packed → features.derived; system 0 needs local_sensors);
  `combined_hash` (features.multi → contracts.robot; the dual env needs it).
- tests: `tests/unit/test_layering.py`, `tests/unit/test_restructure_compat.py` (every shim is the same module object and
  warns; PolicyInput pickles under the old path; `python -m <old>` forwards).

### P2 (bodies / envs / teachers / controllers / models)
- morphology.{aloha,catalog,compiler,fixtures,generators,importers,surgery,variants} → bodies.*; control.ik → bodies.ik;
  morphology.legged_catalog → evaluation.legged_catalog (it evaluates trackers on the legged envs).
- sim.{native,dual,scenario,dual_scenarios,sensors,fixtures} → envs.*; control.joint_targets → envs.joint_targets.
- control.teachers → teachers.arm; control.dual_teachers → teachers.dual; control.legged_teachers → teachers.legged;
  control.dual_validate → teachers.dual_validate; control.functional_composition → teachers.functional_composition.
- control.latent_realizer → controllers.latent_realizer; policy.runner → controllers.policy_runner;
  policy.latent_runner → controllers.latent_runner; policy.registry → service.policy_registry.
- model.* → models.* (system2 waits for P5 → research); learning.checkpoint → models.checkpoint;
  learning.critics → models.critics.
- checks: unit 218 passed / 1 skipped; safe integration tests (adapt_branching, binding_aug, native_sim, service,
  teacher, ui_contract) 27 passed, same as origin/main; every `from rrp... import name` in scripts/, tests/, src/
  resolves (1465 names; scratch checker); demo page identical except the 2 build-timestamp lines.

### P3 (data / training / evaluation)
- learning.data → data.chunks; learning.packed → data.packed; learning.dual_latent → data.dual_latent.
- learning.{adapt, behavior, branching, expo, flow_sde, grpo, latent_grpo, latent_train, legged_bc, legged_dagger,
  legged_latent_train, replay_buffer, rollout, sft, swap_alignment, synthetic, vlm_train} → training.*;
  evaluation.{campaign, baseline_campaign, latent_campaign} → training.* (they train, then evaluate).
- evaluation no longer imports training (the audit's learning↔evaluation cycle). Moved unchanged, old names re-exported:
  `load_representation` (latent_train) and `load_rep`, `checkpoint_provenance`, `legged_flags`, `LEGGED_FLAG_KEYS`,
  legged `_dev` (legged_latent_train) → controllers.bundles; `LatentData` → data.latent; `batched_ticks`
  (latent_grpo) → controllers.latent_realizer; `LeggedBC`, `_mha`, `build`, `load_bc` (legged_bc) → models.legged_bc;
  the legged constants `H`, `MAX_J`, `KNOT_TICKS` → features.legged.
- `rrp.evaluation.checkpoint_audit` (new): loads checkpoints through the real loaders + W3 checkpoint_provenance.
  `tests/unit/test_checkpoint_load.py` runs it on 2 per kind under `RRP_CHECKPOINT_ROOTS` (default artifacts/; skips
  without weights). Full host audit: **347 files, 0 errors, 0 pickled rrp classes**
  (arm: 39 representations, 36 latent flows, 6 direct policies, 14 other; legged: 53 representations, 61 flows,
  11 refit realizers, 16 BC; 54 trackers, 39 resume states, 18 other), lease 1790474631_1a4221, raw
  `artifacts/runs/restructure_ckpt_audit/host_2026-09-26.json`. The peer store was not touched (see remaining steps).
- numerics parity (`research/scripts/2026-09-26/w4_parity.py`, old import paths, fixed seeds, 1 thread): the frozen
  jointfix sem flow + Stage A system 0 on panda_pg2 seed 3000100 (80 ticks) and the W3 smoke legged checkpoints on go2
  (2 s). Pre-W4 code (15984f7) and this branch give byte-identical outputs (floats compared by repr, arrays by sha256):
  `artifacts/runs/restructure_parity/p3_{base_15984f7,track_restructure}.json`.
- checks: unit 241 passed / 2 skipped; safe integration 27 passed; imports resolve (1509 names); pyflakes: no new
  undefined names; demo page identical except the 2 timestamp lines.

### P4 (CLI / orchestration)
- ops.{broker,budget,cgroup,child,discovery,jobs,runtime,telemetry,watchdog} → orchestration.* (ops.gpu went to
  contracts.workload in P1). The leased child is now spawned as `-m rrp.orchestration.child` (the old `-m rrp.ops.child`
  still works through the shim).
- `rrp/cli.py` → package `rrp/cli/` (`main.py` = root parser, doctor, ops; `__main__.py`; `__init__` re-exports
  main/build_parser/_parse_bytes). cli_ext → cli.ext, cli_ml → cli.data, cli_train → cli.train, cli_latent → cli.latent,
  cli_dual_latent → cli.dual_latent, cli_adapt → cli.adapt. `python -m rrp.cli` and the `rrp` console script unchanged.
- The `try/except ImportError` chain is gone: build_parser registers ext, data, train, latent, adapt explicitly (same
  order). The command modules are stdlib-only at import, so the full tree also registers on the bootstrap python.
- checks: the complete argparse tree (48 parsers, every `--help` text) is byte-identical to pre-W4 (scratch
  `clitree.py`); `tests/unit/test_cli.py` (bootstrap run with numpy/torch/mujoco/pydantic/fastapi blocked, 39 commands
  used by scripts parse, no swallowed ImportError, stdlib-only command modules); a real leased job through the new
  orchestration (lease 1790474978_b8c267, rc 0); test_host_gpu_exclusion same as pre-W4 (1 passed, 1 skipped);
  unit 264 passed / 1 skipped; safe integration 27 passed; imports resolve (1535 names); demo page unchanged.

## excluded until W1 merges (P6)
morphology/legged.py, morphology/contact.py, sim/legged.py, control/{legged_core, legged_vec, legged_tracker,
tracker_nets, tracker_training, tracker_validation, reward_schedule}.py. Check before each merge:
`git diff origin/main...origin/track/contact --stat` (and the local `track/contact` branch).

## resume steps
1. `cd ~/work/rrp-wt/restructure && git fetch origin && git status && git log --oneline -5`.
2. Tests: `PYTHONPATH=src ~/work/relational-robot-policy/.venv/bin/python -m pytest tests/unit -q`.
3. Continue with the first phase in the table that is not merged.
