# restructure track (W4: package restructure with shims, phase 2 of the audit)

Owner: restructure agent. Worktree `~/work/rrp-wt/restructure`, branch `track/restructure`. Started 2026-09-26.
Scope: docs/strategy.md W4; docs/repo_structure_audit.md "Target structure" and "Migration plan" phase 2.
Rule: moves only, no behaviour change (numerics, defaults, config semantics, on-disk formats unchanged).

State: **verified** for P1–P5 (merged to main); P6 **blocked_external** on the W1 contact track merging.

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
| P4 | CLI package without silent ImportError; ops → orchestration | merged aaf9201 |
| P5 | research diagnostics; dedup; peer_sync .rrp_revision | merged ff0d867 |
| P6 | legged files excluded until W1 (contact) merges | blocked_external (W1 still editing legged_core, tracker_training) |

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

### P5 (research diagnostics / dedup / peer_sync)
- → rrp.research (referenced only from research docs or one-off diagnostics; old paths are shims, `python -m` forwards):
  learning.legged_t1_diag, learning.qa_train, evaluation.system2_eval, evaluation.bc_semantic_edits,
  evaluation.latent_slice1_report, model.system2 (used only by system2_eval).
- peek scripts `scripts/ladder_{peek,locpeek,sumpeek,t0_check}.py` → `research/scripts/2026-09-26/` (git mv; referenced
  only from research/tracks/ladder.md and decisions.md; nothing runs them).
- dedup (provably identical; old names kept as aliases; tests in test_restructure_compat.py):
  - `wilson`: `rrp.evaluation.ladder.wilson` = the ladder convention (z=1.96, (0,1) at n=0) over
    `rrp.evaluation.statistics.wilson` (same expression, bitwise-equal on all k ≤ n ≤ 60).
  - `_seeds` (legged_collect, legged_latent_collect, legged_latent_eval) → `rrp.contracts.runs.parse_seed_spec`.
  - session featurizer cache: `ladder._featurizer`, `latent_semantic_edits._featurizer` (single-robot branch) and
    `LatentPolicy.featurizer` → `rrp.features.featurizer.cached_featurizer`.
- `scripts/peer_sync.sh push` now also writes `$R/.rrp_revision` = {git_sha, dirty (tracked changes, as W3), branch,
  synced_at, source}; `peer_sync.sh revision` prints it locally (tested; W3 `code_provenance` reads it). Not run
  against the peer (W4 did not touch the peer).
- checks: unit 274 passed / 1 skipped; safe integration 27 passed; numerics parity after the dedup still byte-identical
  to pre-W4 (`parity_p5`); argparse tree identical; imports resolve (1552 names incl. research/scripts); demo page
  unchanged.

### deferred to W5 (not provably identical, or a behaviour/format change)
- `_dev` helpers: `latent_train._dev` (apply_cap errors raise) vs legged `_dev` (now `controllers.bundles._dev`;
  apply_cap errors swallowed) vs `adapt._device` (returns (name, info), disables TF32). Different behaviour.
- oracle wrappers: `ladder.OraclePacketPolicy` (batched shadow teacher / BC lookahead), `latent_semantic_edits.OracleSource`
  (per-call context edits, teacher or BC demo), `legged_latent_eval.OracleShadow` (legged). Different interfaces.
- packet builders: `ladder.make_packet` = `build_packet(f, s, s.observe(), z, sampling={})` except that it keeps a
  non-contiguous z (`np.asarray` vs `np.ascontiguousarray`); `latent_semantic_edits.dual_packet` masks assemblies and uses
  dual handles. Merge in W5 with a layout-insensitive packet test.
- featurizer patching: `ladder.install_prev_action` (wraps step/snapshot/restore) vs the inline
  `PrevActionFeaturizer` install in `run_ladder` (recording is done by the loop): different.
- `policy_runner.LearnedPolicy.featurizer` also resets `_rrp_prev_action`: not merged into cached_featurizer.
- W3 follow-ups that change written strings: eval rows of arm ladder / dual still write free source strings
  (should use `source_label()`); `contracts.action.Source` Literal → enum.
- wilson copies in `scripts/legged_ladder_summary.py` and `scripts/demo/build_page.py` (scripts are frozen; the demo
  page must build unchanged); remaining one-off scripts (t1_diag_*.sh, chain scripts) → W5 run-dag.
- remove shims after the audit's phase-5 condition; keep `rrp/data/features.py` permanently (pickle path).

## module map (old → new; the full list is `PLANNED` in tests/unit/test_layering.py)
- contracts: control.psi_contracts → contracts.psi; ops.gpu (+ ops.jobs.CheckpointSignal) → contracts.workload;
  `combined_hash` → contracts.robot; `parse_seed_spec` (new, dedup) in contracts.runs.
- physics: sim.snapshot_contract → physics.snapshot; P6: morphology.contact → physics.contact.
- bodies: morphology.{aloha,catalog,compiler,fixtures,generators,importers,surgery,variants} → bodies.*; control.ik →
  bodies.ik; P6: morphology.legged → bodies.legged.
- envs: sim.{native,dual,scenario,dual_scenarios,sensors,fixtures} → envs.*; control.joint_targets → envs.joint_targets;
  P6: sim.legged, control.{legged_core,legged_vec,legged_tracker,tracker_nets} → envs.*.
- features: data.features → features.featurizer (+ featurizer_for, cached_featurizer); data.features_multi →
  features.multi; control.legged_latent → features.legged (+ H, MAX_J, KNOT_TICKS); features.derived (local_sensors,
  active_operator, OPERATORS from learning.packed).
- teachers: control.teachers → teachers.arm; control.dual_teachers → teachers.dual; control.legged_teachers →
  teachers.legged; control.dual_validate → teachers.dual_validate; control.functional_composition →
  teachers.functional_composition.
- models: model.{attention,backbone,batch,binding_aug,codec,flow,latent_batch,latent_probes,legged_latent,qa,
  semantic_latent} → models.*; learning.checkpoint → models.checkpoint; learning.critics → models.critics;
  models.legged_bc (LeggedBC/build/load_bc from learning.legged_bc).
- controllers: control.latent_realizer → controllers.latent_realizer (+ batched_ticks); policy.runner →
  controllers.policy_runner; policy.latent_runner → controllers.latent_runner; controllers.bundles (load_representation;
  legged load_rep, checkpoint_provenance, legged_flags, _dev).
- data: learning.data → data.chunks; learning.packed → data.packed; learning.dual_latent → data.dual_latent;
  data.latent (LatentData).
- evaluation: morphology.legged_catalog → evaluation.legged_catalog; evaluation.checkpoint_audit (new);
  P6: control.tracker_validation → evaluation.tracker_validation.
- training: learning.{adapt,behavior,branching,expo,flow_sde,grpo,latent_grpo,latent_train,legged_bc,legged_dagger,
  legged_latent_train,replay_buffer,rollout,sft,swap_alignment,synthetic,vlm_train} → training.*;
  evaluation.{campaign,baseline_campaign,latent_campaign} → training.*; P6: control.{reward_schedule,tracker_training}.
- pipelines: empty skeleton (W5).
- orchestration: ops.{broker,budget,cgroup,child,discovery,jobs,runtime,telemetry,watchdog} → orchestration.*.
- cli: cli.py → cli/main.py (package rrp.cli); cli_ext → cli.ext; cli_ml → cli.data; cli_train → cli.train;
  cli_latent → cli.latent; cli_dual_latent → cli.dual_latent; cli_adapt → cli.adapt.
- service: policy.registry → service.policy_registry.
- research: learning.legged_t1_diag, learning.qa_train, evaluation.system2_eval, evaluation.bc_semantic_edits,
  evaluation.latent_slice1_report, model.system2 → research.*.

## excluded until W1 merges (P6)
morphology/legged.py, morphology/contact.py, sim/legged.py, control/{legged_core, legged_vec, legged_tracker,
tracker_nets, tracker_training, tracker_validation, reward_schedule}.py. Check before each merge:
`git diff origin/main...origin/track/contact --stat` (and the local `track/contact` branch).

## P6 procedure (after W1 merges; do not start while track/contact edits these files)
1. `git fetch origin && git diff origin/main...origin/track/contact --stat` must not list the P6 files, and the lead must
   confirm W1 is merged/idle. Rebase this branch on origin/main.
2. In `research/scripts/2026-09-26/w4_move.py` empty the `EXCLUDED` set, then run
   `python research/scripts/2026-09-26/w4_move.py rrp.morphology.contact=rrp.physics.contact
   rrp.morphology.legged=rrp.bodies.legged rrp.sim.legged=rrp.envs.legged rrp.control.legged_core=rrp.envs.legged_core
   rrp.control.legged_vec=rrp.envs.legged_vec rrp.control.legged_tracker=rrp.envs.legged_tracker
   rrp.control.tracker_nets=rrp.envs.tracker_nets rrp.control.reward_schedule=rrp.training.reward_schedule
   rrp.control.tracker_training=rrp.training.tracker_training
   rrp.control.tracker_validation=rrp.evaluation.tracker_validation` (all targets match PLANNED).
   Check `from . import x` lines in the moved files (the tool now maps them; verify with the layering test).
3. Checks: `pytest tests/unit` (layering: the legacy packages then hold only shims; test_cli; test_restructure_compat
   runs `python -m <old> --help` for every moved argparse module, e.g. rrp.control.tracker_training); numerics parity
   `python research/scripts/2026-09-26/w4_parity.py OUT.json ~/work/rrp-wt/provenance/artifacts/runs/provenance_smoke
   ~/work/relational-robot-policy` (under `rrp ops run`; PYTHONPATH = this tree's src) must equal
   `artifacts/runs/restructure_parity/p3_base_15984f7.json` (its legged half runs LeggedSession with the go2 tracker and
   needs `artifacts/trackers/go2/actor.pt`); the checkpoint audit (trackers); a short tracker-training smoke
   (`python -m rrp.control.tracker_training --body go2` with a tiny iteration count under `rrp ops run`).
4. Decide the tracker home: `envs/` (planned; LeggedSession embeds the tracker) or `controllers/` with the tracker
   injected into LeggedSession (a W5 refactor, behaviour-neutral but not a pure move).

## remaining steps (for the lead)
- Peer checkpoint store (strategy W4 gate; W4 did not touch the peer): sync main into a FRESH dir and run the audit, e.g.
  `export RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/restructure; scripts/peer_sync.sh push; scripts/peer_run.sh --cpu 2
  --mem 8G --label restructure_ckpt_audit --max-seconds 3600 -- PY -m rrp.evaluation.checkpoint_audit
  --out artifacts/runs/restructure_ckpt_audit/peer.json /dev/shm/rrp-brandonin/repo/artifacts`
  (the host audit found no pickled rrp classes in 347 files, so import paths cannot break loading; this checks the rest).
- P6 above once W1 merges.
- Other branches rebasing onto main: edits to a moved file's OLD path conflict with its shim; port them to the new path.

## resume steps
1. `cd ~/work/rrp-wt/restructure && git fetch origin && git status && git log --oneline -5`.
2. Tests: `PYTHONPATH=src ~/work/relational-robot-policy/.venv/bin/python -m pytest tests/unit -q`.
3. Continue with the first phase in the table that is not merged.
