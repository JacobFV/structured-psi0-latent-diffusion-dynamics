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
| P2 | bodies, envs, teachers, controllers, models | verified |
| P3 | data, training, evaluation (break evaluation↔training) | planned |
| P4 | CLI package without silent ImportError; ops → orchestration | planned |
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

## excluded until W1 merges (P6)
morphology/legged.py, morphology/contact.py, sim/legged.py, control/{legged_core, legged_vec, legged_tracker,
tracker_nets, tracker_training, tracker_validation, reward_schedule}.py. Check before each merge:
`git diff origin/main...origin/track/contact --stat` (and the local `track/contact` branch).

## resume steps
1. `cd ~/work/rrp-wt/restructure && git fetch origin && git status && git log --oneline -5`.
2. Tests: `PYTHONPATH=src ~/work/relational-robot-policy/.venv/bin/python -m pytest tests/unit -q`.
3. Continue with the first phase in the table that is not merged.
