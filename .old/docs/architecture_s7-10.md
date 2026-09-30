# docs/architecture.md sections 7-10 (archived D-145 P8)

The D-140 refactor record: delete list (7), staged migration S0-S6 (8), the psi0 / ComputerWorld hand-off checklist (9) and the old-to-new module path map (10). Section numbers are kept because other notes cite them. To resolve a module name from an old note or run config, use the table in section 10.

## 7. delete list (status)

| what | status |
|---|---|
| W4 shim packages (`control/`, `learning/`, `model/`, `morphology/`, `ops/` alias, `policy/`, `sim/`, top-level `cli_*.py`, `data/features*.py`, evaluation re-export shims), `rrp/core.py` re-export module, every "moved; re-exported" alias | done (S1, S2, S2b); a test forbids shim modules |
| `src/rrp/research/` one-offs | done (S1) |
| `MjxLegged`, `WarpLegged` bake-off prototypes | done (S4; helpers kept as `envs.warp.model`) |
| duplicate oracles | done (S4: `policies.oracle`, one encoder; legged `policies.legged.OracleShadow`) |
| duplicate eval loops (`runner.evaluate`, `latent_eval.evaluate_latent`, `dual_latent_eval.evaluate_dual_latent`, `train.rollout.drive`, `latent_grpo.run_episodes`, semantic-edit episode loops, `legged_latent_eval.run_episode` special cases), `wilson` copy in the ladder, evaluation-side teacher wrappers | done in S5 (S5a runner, S5b arm/dual latent, S5c semantic edits, S5d GRPO/drive, S5e legged on the `legs` space + `tasks.spec.legged_judge`; every port golden-checked against its old loop; `wilson` copies in harness delegate to `harness.eval.statistics`). Open: `ShiftedGoalTeacher` / `TeacherSource` / `_DualTeacher` in `harness.eval.latent_semantic_edits` (semantic-edit experiment variants; policies/** owner), the two `wilson` copies in `viz.export` |
| `tests/unit/test_restructure_compat.py`, shim tables of `test_layering.py` | done (S1) |
| 175 finished-experiment scripts (chain drivers, analysis one-offs) | done (S1; `git show a951398^:scripts/<name>`). Kept: peer transport, asset fetch, UI type export, viz record specs, the ladder / legged-summary wrappers the pipelines call, renderers, demo builder, the paused W13 / armdiv drivers their RESUME steps name, `psi0_ext.sh`. The wrappers and renderers fold into the CLI after S5 |
| `dags/` | kept: every DAG is cited by a test, a track note's resume steps or the naming map (they are the lineage provenance run by `rrp run-dag`); comments name the new module paths (S6) |
| `configs/` | kept: every directory is cited by the naming map, a report or a DAG and is validated by test_runconfig; no config names a deleted module |
| docs: `core_api.md`, `repo_structure_audit.md`, `related_repos.md` (glossary now section 11), `intentions_backlog.md` (appendix of `experiments_roadmap.md`) | done (S2, psi0mig, S6) |

## 8. staged migration (each stage: `pytest tests/unit` green with exit code checked, then merged to main)

Verification is by golden tests, not reruns (D-140): `tests/unit/test_golden.py` records SHA-256 digests of the
outputs of fixed, seeded inputs before any move (arm featurizer on a procedural-arm pick_place reset; a teacher's first
commands and resulting qpos; a seeded-random latent planner packet z and realizer output; legged public context and
realizer output; dual multi-featurizer; packet serialization) and must stay byte-identical through every stage.

| stage | content | est. | status |
|---|---|---|---|
| S0 | this document; golden digests on the pre-refactor code | 2 h | done (c14ca55, 44e1f3d, 58d65c5) |
| S1 | delete shim packages and `research/`; rewrite importers in src/tests/scripts; layering test without shim tables; pickle remap in the data loader | 1.5 h | done (a951398) |
| S2 | re-layer into `core / ops / bodies / tasks / envs / policies / harness / viz` (codemod on module paths, new layering table); S2b: no re-export aliases | 3 h | done (cb0ea23, ba2ce79) |
| S3 | `envs.base` (Env, EnvSpec, capabilities, registry) implemented by the MuJoCo sessions and the Warp env; `policies.base` (Policy, Requirements, negotiate, registry); `tasks` registry; SIMPLE/ComputerWorld declared in the registries → **Ψ₀ and ComputerWorld agents can start here** | 3 h | done (this commit) |
| S4 | policy adapters (bc, latent arm/dual, legged latent/bc on the `legs` space, teachers, one oracle); `legs` control mode; prototypes deleted | 4 h | done (S4a 4c4d2bb, S4b this commit) |
| S5 | port the eval loops onto `harness.rollout` (S3) + hooks, arm/dual first, legged last; delete the duplicates; legged judge; `rrp eval` / `rrp matrix`; golden traces of tiny episodes (a few ticks, procedural bodies) | 5 h | done (S5a d623980, S5b aae0cf1, S5c 4fed1f6, S5d 10f3ff2, S5e: this commit) |
| S6 | scripts/dags/configs pruning, docs (backlog folded into the roadmap), README, STATUS, strategy (S6a); every `python -m <module>` entry point and script wrapper folded into `rrp` tool commands, legacy `rrp evaluate` folded into `rrp eval`, edit-suite teacher wrappers into policies.teachers, room Wilson copies onto harness statistics (S6b) | 2 h | done |

Running work is paused (D-140); unmerged track branches rebase onto the new paths using the move table that each
stage's commit message and section 10 record.

## 9. hand-off for the Ψ₀ and ComputerWorld agents (S3 is on main: start here)

Ψ₀ migration (psi1z → rrp; done, map in research/tracks/psi0.md):
- [x] `rrp/bodies/g1_simple.py`: the G1 + Dex3 36-d command morphology tables and `spec_hash()` (the G1 is simulated inside SIMPLE, so there is no compiled `RobotSpec`); key `g1_simple`.
- [x] `rrp/envs/simple/` (`make_env`): `SimpleEnv` drives a SIMPLE worker process (its own venv, Isaac Sim 5.1) over 127.0.0.1; the worker runs the unmodified upstream agent with an injected chunk client; `compat.py` holds the import hooks; action space `psi0` [36] at 50 Hz with capability `chunk_executor`; `chunk_request` / `psi0_state` channels; truth = palms, pelvis, objects, contacts (labels/judging only).
- [x] `rrp/policies/psi0/` (`make_direct`, `make_structured`, `make_replay`): policies over the upstream `Server` object (transforms, normalization, RTC); `nets.py`, `data.py` (feature cache, dataset, `LabelRecorder` hook), `train.py` (`rrp train psi0 ...`).
- [x] `rrp/tasks/spec.py`: `simple/<Task>` for the six benchmark tasks (`SIMPLE_TASKS`: released run, published rates, step-1 status), judge = SIMPLE `_success` via `env.truth()`.
- [x] P-001..P-023 folded as appendix P of `research/decisions.md`; notes in `research/tracks/psi0.md`; `docs/related_repos.md` deleted (glossary: section 11). psi1z archived read-only after owner approval.

ComputerWorld (track pointer, research/tracks/pointer.md):
- [x] optional extra `computerworld = ["computerworld==0.2.0"]` (PyPI abi3 wheel incl. aarch64; no Rust build).
- [x] `rrp/envs/computerworld.py` (`ComputerWorldEnv`, `make_env`), `tests/unit/test_computerworld.py` (mapping, depth
  modes, occlusion, null slots, button edges, negotiation; rollout/judge/determinism tests marked `computerworld`).
- [x] `rrp/bodies/fixtures.cw_pointer_spec` (family `pointer`).
- [x] `cw/calc_sum`, `cw/open_type`, `cw/drag_window`, `cw/fill_form` in `rrp/tasks/spec.py` (judges env-side: outcome +
  failure reason; no task graph yet) and `rrp/policies/teachers/computerworld.py` (`teacher:cw/*`, `scripted_teacher`,
  privileged).
- [x] `rrp matrix` (S5a) lists arm/legged/Ψ₀ policies declined on computerworld with reasons
  (artifacts/runs/s5_matrix/matrix.jsonl); [ ] a pointer BC trained on teacher data (accepted) is future work.

## 10. moved paths

S1 (a951398) deleted the W4 shim packages; their real targets are the "old" column below (e.g. `rrp.learning.latent_train`
was a shim of `rrp.training.latent_train`, now `rrp.harness.train.latent_train`; `rrp.sim.native` → `rrp.envs.native` →
`rrp.envs.mujoco.session`; `rrp.control.legged_tracker` → `rrp.envs.legged_tracker` → `rrp.envs.mujoco.legged_tracker`;
`rrp.morphology.*` → `rrp.bodies.*`; `rrp.ops.*` → `rrp.orchestration.*` → `rrp.ops.*`). `rrp.core` (the W11 re-export
module) is gone: import each name from its module. S2 moved packages as follows (a package row covers every module in it
unless a more specific row exists); module names inside packages are unchanged.

| old | new |
|---|---|
| `rrp.contracts` | `rrp.core` |
| `rrp.contracts.workload` | `rrp.ops.workload` |
| `rrp.controllers` | `rrp.policies` |
| `rrp.controllers.anchor_realizer` | `rrp.policies.system0_anchor` |
| `rrp.controllers.latent_realizer` | `rrp.policies.system0` |
| `rrp.controllers.latent_runner` | `rrp.policies.latent` |
| `rrp.controllers.policy_runner` | `rrp.policies.bc` |
| `rrp.data` | `rrp.harness.data` |
| `rrp.envs.dual` | `rrp.envs.mujoco.dual` |
| `rrp.envs.dual_scenarios` | `rrp.envs.mujoco.dual_scenarios` |
| `rrp.envs.fixtures` | `rrp.envs.mujoco.fixtures` |
| `rrp.envs.humanoid_scenes` | `rrp.envs.mujoco.humanoid_scenes` |
| `rrp.envs.joint_targets` | `rrp.envs.mujoco.joint_targets` |
| `rrp.envs.legged` | `rrp.envs.mujoco.legged` |
| `rrp.envs.legged_core` | `rrp.envs.mujoco.legged_core` |
| `rrp.envs.legged_scenes` | `rrp.envs.mujoco.legged_scenes` |
| `rrp.envs.legged_tracker` | `rrp.envs.mujoco.legged_tracker` |
| `rrp.envs.legged_vec` | `rrp.envs.mujoco.legged_vec` |
| `rrp.envs.mjx_legged` | `rrp.envs.warp.mjx_legged` |
| `rrp.envs.morph_obs` | `rrp.envs.mujoco.morph_obs` |
| `rrp.envs.motion_quality` | `rrp.envs.mujoco.motion_quality` |
| `rrp.envs.native` | `rrp.envs.mujoco.session` |
| `rrp.envs.perturb` | `rrp.envs.mujoco.perturb` |
| `rrp.envs.scenario` | `rrp.envs.mujoco.scenario` |
| `rrp.envs.sensors` | `rrp.envs.mujoco.sensors` |
| `rrp.envs.state_estimator` | `rrp.envs.mujoco.state_estimator` |
| `rrp.envs.tracker_nets` | `rrp.envs.mujoco.tracker_nets` |
| `rrp.envs.warp_legged` | `rrp.envs.warp.warp_legged` |
| `rrp.envs.warp_task_env` | `rrp.envs.warp.task_env` |
| `rrp.envs.warp_tracker_env` | `rrp.envs.warp.tracker_env` |
| `rrp.evaluation` | `rrp.harness.eval` |
| `rrp.features` | `rrp.policies.features` |
| `rrp.models` | `rrp.policies.nets` |
| `rrp.orchestration` | `rrp.ops` |
| `rrp.orchestration.dag` | `rrp.harness.dag` |
| `rrp.orchestration.yamlmini` | `rrp.harness.yamlmini` |
| `rrp.physics` | `rrp.bodies` |
| `rrp.physics.snapshot` | `rrp.envs.mujoco.snapshot` |
| `rrp.pipelines` | `rrp.harness.pipelines` |
| `rrp.service` | retired: `.old/src/rrp/viz/workbench/` |
| `rrp.teachers` | `rrp.policies.teachers` |
| `rrp.training` | `rrp.harness.train` |

`python -m <module>` entry points became `rrp <group> <tool>` commands in S6b (table in section 6; e.g. `python -m
rrp.training.warp_tracker_ppo` → `rrp train tracker-warp`, `python -m rrp.evaluation.legged_latent_eval` → `rrp suite
legged`, `scripts/ladder.py` → `rrp suite ladder`, `python -m rrp.pipelines run|stages` → `rrp stage run|list`,
`python -m rrp.viz.export` → `rrp viz export`); `python -m rrp.cli` (and the `rrp` console script) is unchanged. Pipeline families, stage names (`collect`, `pack`, `train_rep`, `train_flow`,
`dagger`, `refit`, `eval_r1`, `eval_r2`, `heldout`, `edits`, ...), DAG files, configs and `artifacts/runs/<track>/...`
output paths are unchanged, so armdiv's resume (`dags/arm_lineage_v7div*.yaml`, `dags/armdiv_bc_v7div*.yaml`, ledgers
under `artifacts/runs/armdiv/_dags/`) and W13's resume work as written after `ops/bin/peer_sync.sh push`; their
scripts (`armdiv_chain.sh`, `armdiv_pack.sh`, `humanoid_{steps_eval,steps_eval_grid,gap_smoke,gap_eval,tracker_gate,tracker_finalize}`,
`contact_waypoint_eval.py`, `render_contact_compare.py`) were kept for that and are rewritten to the new module paths.
Other deleted scripts: `git show a951398^:scripts/<name>`.

Legacy on-disk data: dataset/DAgger pickles written before D-140 reference `rrp.data.features.PolicyInput`;
`rrp.harness.data.collect.load_pickle` (used by `read_episode` and the generator-DAgger loader) remaps it. Checkpoints
store state dicts and plain containers (no rrp classes), so they load unchanged.
