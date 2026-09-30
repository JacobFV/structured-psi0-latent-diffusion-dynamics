# pre-training readiness plan (D-146) — round 2

Source: read-only audit of main 0bc74615 (94 findings → 31 issues D1–D31), re-mapped to the post-purge layout
(D-145: `recipes/`, `ops/bin/`, `.old/`). Decisions: D-146. Contracts: `docs/architecture.md` section 14,
`docs/relations.md` section 11. Owner scope: everything before training (incl. upper-body control and tasks, and
the rollout ports); armdiv restarts its latent lineages; dual-arm is parked.

## rules for every unit

- Worktree `~/work/rrp-wt/rdy-<id>` on `track/rdy-<id>` from origin/main; `export PYTHONPATH=$PWD/src:$PWD`.
- Host = git, editing, unit suite only (CUDA hidden; no training, no simulation beyond unit-test fixtures). A smoke
  that needs the peer goes through `ops/bin/peer_run.sh` and is ≤ 10 min.
- FIRST re-check the audit finding on current main (the audit predates the purge and the last relation sweeps); if
  it is already fixed, say so in the report and do only what remains.
- Edit only the files you own. A needed change in another unit's file is reported to the lead, not made.
- Light testing: one or two tests per unit on the production path. `tests/data/golden.json` unchanged unless the
  brief says an intended change (then record it under D-146).
- Merge: `flock ~/work/rrp-data/main-merge.lock bash -c 'git fetch origin && git rebase origin/main && <unit suite> &&
  git push origin HEAD:main'`; layout test and unit suite green; `rrp run-dag <recipe> --dry-run` for every recipe
  the unit touches. Commit trailer per AGENTS.md. Notes go into the unit's track note
  (`research/tracks/{humanoid,armdiv,psi0,pointer,relations}.md`), never a new file.
- No shims, no aliases, no new module where an existing one fits (new files are listed per unit).

## round 1 ledger (main 4af47836)

Merged (32): F0 F1 F2 F3 K1 K2 H5 A3 RP1 HT HJ HL A1 R1 P1 P2 C0 A2 RP2 U1 H7 H4 C1 P3 RP3 U2 HX C2 C3 U3 H6 RP4.
Not done: RP5 (blocked, re-sequenced below), X1, X2. Integration verdict: NOT ready — the gaps are the round-2
units below. Decisions on every round-1 lead question: `research/decisions.md`, D-146 addendum (round 2).

Extra rules for round 2:
- **Fresh-checkout green is a hard rule.** A test that needs untracked weights, Menagerie assets, warp or the peer
  skips with a reason naming what is missing (`pytest.skip("needs untracked weights <path>")`); never a failure.
  Tests slower than 20 s carry `@pytest.mark.slow`; the merge gate runs `pytest tests/unit -m "not slow"`, the final
  integration check runs everything.
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (AGENTS.md).
- A unit whose deliverable includes a recipe runs its dry-run and, where the table says "smoke", ONE ≤ 10 min peer
  smoke through `ops/bin/peer_run.sh` (own `RRP_PEER_REPO`), recorded in the track note. No training on the host.

## A. code units (all must merge before any training)

| id | unit | wave | deps | h | owns |
|---|---|---|---|---|---|
| G0 | fresh-checkout green + slow marker | 0 | – | 2 | `tests/conftest.py`, `pyproject.toml` (markers), `tests/unit/test_trackers.py`, skip guards / slow marks in `tests/unit/test_{humanoid_carry,pointer*,wholebody,morph_multi}.py` |
| RC | registry closure | 0 | – | 5 | `policies/relations/{base,ops,catalog}.py`, `policies/nets/{flow,batch}.py`, `tests/unit/test_relations_runtime.py` |
| FS | factor stamping everywhere + pipeline base cleanup | 0 | – | 6 | `policies/{bundles,latent,bc}.py`, `harness/pipelines/base.py`, `cli/dag.py`, `ops/bin/peer_sync.sh`, the `register(` call lines of `harness/pipelines/{psi0,pointer}.py` |
| SL | split-agnostic sealed guard + sealed constants | 0 | – | 5 | `core/sealed.py`, `bodies/armdiv.py`, `cli/{latent,tools}.py`, `harness/train/{latent_grpo,grpo_anchor}.py`, `tests/unit/test_sealed.py` |
| TK | humanoid task / teacher closure | 0 | – | 5 | `tasks/{humanoid,spec}.py`, `tasks/graphs/*.json`, `policies/teachers/{__init__,humanoid,legged_loco}.py`, `policies/base.py` (POLICIES keys only) |
| HS1 | sensors and env fixes | 0 | – | 10 | `envs/mujoco/{legged,legged_core,legged_tracker,humanoid_scenes,sensors}.py`, `envs/warp/task_env.py`, `envs/base.py`, `bodies/humanoid_gen.py`, `tests/unit/test_trackers.py` sensor cases (after G0) |
| HD1 | humanoid collect: upper targets, terrain labels, rollout | 0 | – | 12 | `harness/data/{legged_latent_collect,legged_collect}.py`, `policies/features/legged.py` |
| RG | relgen reaches the arm trainers | 0 | – | 10 | `harness/pipelines/relations.py`, `harness/data/mix.py`, `harness/train/{latent_train,behavior}.py`, `recipes/relations/**`, `recipes/templates/relations_factor.yaml` |
| AR | armdiv closure | 0 | – | 6 | `recipes/armdiv/**`, `recipes/templates/arm_*.yaml`, `recipes/README.md`, `harness/pipelines/arm.py`, `harness/eval/target_eval.py`, `research/tracks/armdiv.md` |
| DP | dual parking, finished | 0 | – | 4 | `harness/pipelines/dual.py`, `recipes/templates/dual_lineage.yaml`, `policies/teachers/{dual,dual_validate,functional_composition}.py`, `harness/eval/{dual_teacher_quality,dual_latent_eval}.py`, `tests/unit/test_layering.py` |
| RL | last private loops | 0 | – | 8 | `harness/data/collect.py`, `harness/eval/{ladder,ladder_cli,latent_eval,grasp_rig}.py` |
| HS2 | tracker training: upper curriculum, shared morph with upper, install command, guard | 1 | HS1, SL | 12 | `harness/train/{warp_tracker_ppo,tracker_training,tracker_recipes}.py`, `envs/warp/{tracker_env,model}.py`, `envs/mujoco/{legged_vec,morph_obs}.py` |
| HD2 | humanoid training path: task-agnostic data, upper supervision, relgen hook | 1 | HD1, RG | 12 | `harness/train/{legged_latent_train,legged_bc}.py`, `policies/nets/{legged_latent,legged_bc}.py` |
| HP | legged policy from checkpoint factors + upper check | 1 | FS | 5 | `policies/legged.py`, `harness/train/legged_dagger.py` |
| PS | Ψ₀ closure | 1 | FS | 5 | `harness/pipelines/psi0.py`, `policies/psi0/{train,__init__,data}.py`, `recipes/psi0/**`, `recipes/templates/psi0_step2.yaml`, `research/tracks/psi0.md` |
| PC | pointer closure | 1 | FS, SL, RC | 10 | `harness/pipelines/pointer.py`, `harness/train/pointer/**`, `policies/pointer/**`, `harness/data/relgen/ui.py`, `recipes/pointer/**`, `recipes/templates/pointer_lineage.yaml`, `research/tracks/pointer.md` |
| RP5 | recorder as rollout hooks | 1 | RL, HS1 | 8 | `viz/record.py`, `tests/unit/test_viz_record.py` |
| HA | adapting trainers + humanoid family stages | 2 | HD2, HP, HS2 | 14 | `harness/pipelines/humanoid.py`, `harness/eval/{humanoid_eval,legged_latent_eval}.py`, new `harness/train/humanoid_adapt.py` |
| HR | humanoid recipes for every task and every tracker run | 3 | HA, TK | 8 | `recipes/humanoid/**`, `recipes/templates/{humanoid_task,humanoid_transfer,tracker_gated,legged_lineage,legged_heldout}.yaml`, `research/tracks/humanoid.md` |
| X1 | residual dedupe + lints | 4 | all above | 7 | `harness/eval/hooks.py` → `harness/hooks.py` (and its importers), digest / Wilson helpers and callers, `harness/train/rollout.py` rename, `tests/unit/test_stale_paths.py` |
| X2 | docs truth pass | 5 | X1 | 4 | `STATUS.md`, `README.md`, `docs/*.md`, progress states in `research/tracks/*.md`, `research/relations_catalog.md` |

Total ≈ 158 h in 21 units. Critical path: HD1 → HD2 → HA → HR → X1 → X2 (≈ 57 h serial).

Waves: **0** G0 RC FS SL TK HS1 HD1 RG AR DP RL · **1** HS2 HD2 HP PS PC RP5 · **2** HA · **3** HR · **4** X1 · **5** X2.
Hand-overs: `tests/unit/test_trackers.py` G0 → HS1; `harness/pipelines/{psi0,pointer}.py` FS (one line each) → PS / PC;
`harness/train/latent_train.py` RG only; `nets/legged_latent.py` HD2 only; `harness/eval/hooks.py`: RL, RP5 and HA may
add hooks (rebase in merge order), X1 moves the file last.

### briefs (A)

- **G0.** Make the suite pass on a checkout with no untracked weights and no Menagerie assets: every test that
  loads `artifacts/trackers/**/actor.pt`, a checkpoint, warp or assets skips through one helper in `conftest.py`
  (`need_weights(path)`, `need_assets()`), with the missing path in the reason. `test_committed_registry_*` must
  compare against the tracked `meta.json` files only (no dependence on the shared store or `RRP_HOME`). Register
  `slow` in `pyproject.toml` and mark every test over 20 s (the pointer copy-head test among them); do NOT change
  `addopts` (the merge command passes `-m "not slow"`). Acceptance: `git worktree add` of main with no `.cache` →
  `pytest tests/unit -m "not slow"` exit 0; `-rs` shows each skip reason.
- **RC.** (1) Field operators honour `<field>.valid`: an invalid token contributes zero on both sides (`pape`,
  `diff`, `rel_rot`, `align`, `sim`, `unary`, `same`, `order`); test with a half-invalid set. (2) Catalog truth:
  delete `kin.mirror` (duplicate of `edge.mirror`), mark `time.same_track` `planned` (no family has history
  tokens), make `id.same_body` resolve by adding `entity_id` to the arm / dual `ctx>ctx` carries; name the scene
  parts `cw_viewport`, `cw_depth` in the `gen` of the `ui.*` entries. (3) An `@gt` EdgeSet hook for
  `ix.force_flow` (`relation_token_sets` attaches `"<site>#support-v1@gt"` from the support label when
  `labels=`). (4) `PolicyConfig.family` is set to `dual` by the dual loader path (report the one line if the file
  is not yours). Acceptance: `rrp factors coverage` lists every `implemented` factor in ≥ 1 family; goldens
  unchanged. The architect reviews this unit before merge.
- **FS.** Every checkpoint writer calls `stamp_versions` and every loader `require_factors` (`bundles.py`,
  `latent.py`, `bc.py`; report the psi0 / pointer / legged call sites to PS / PC / HP). An unstamped checkpoint is
  read as its family's legacy default preset (`require_factors(..., unstamped="default")`); a stamped mismatch
  raises unless `allow_mismatch`. In `pipelines/base.py`: `_factor_items` also reads `latent.encoder_factors` /
  `latent.realizer_factors`; compose the factor string with `stamp_versions`; delete the `register` alias
  (`register_stage` only; fix the two call lines in the psi0 / pointer pipeline modules); export
  `RRP_RUN_CONTEXT=<path to run_context.json>` so a grandchild process applies the same context
  (`apply_run_context` reads it when called without a config); `cli/dag.py` loads families at plan time (keep R1's
  line); `peer_sync.sh` syncs `ops/resources.local.json`. Tests: unstamped arm checkpoint loads as the `arm`
  preset; stamped mismatch raises; a spawned grandchild sees the kinfeat option.
- **SL.** `SealedSplit` becomes split-agnostic (`SealedSplit.load(name)` for `humanoid_v1`, `armdiv_v1`,
  `cworld_pointer_v2`, each hash-pinned); `cell_id` = (body, method, task, budget, adaptation, seed set) natively
  (drop the method-string encoding; report the call-site change to HA); `record_infrastructure_failure` gets the
  CLI `rrp suite sealed-log {list,infra-failure}`; one `is_sealed_target` in `rrp.bodies.armdiv` used by
  `cli/latent.py`, `latent_grpo.py`, `grpo_anchor.py` (its GRPO anchors must refuse armdiv sealed targets) —
  report the inline tuple in `ladder_cli.py` to RL. Red / green tests per split.
- **TK.** `max_steps`: `h_steps` 400, `h_gap` 300, and a value for every `h_*` task; one `ALL_MANIP_TEACHERS` list
  (delete the U2-only list); `AbsentLimb` becomes a `negotiate` reason ("body has no arm roles"), not an
  exception at build; retire the legacy `loco_pick` and `carry_tray_level` tasks and graphs to `.old/` (group
  README citing D-146); the parked dual tasks get `teacher=None`; every emitted failure reason is inside the
  task's declared vocabulary (rollout asserts it — report the one line to X1). Test: every TASKS entry has a
  resolvable teacher or an explicit None; the matrix shows berkeley × carry as n/a with the reason.
- **HS1.** (1) `range_ring`: a public horizontal range sensor (16 rays at torso height, 3 m, noise σ 2 cm, 2 %
  dropout, one-tick latency) in MuJoCo and Warp with one shared layout constant; actor obs of the gap recipes =
  proprio + `terrain_scan` + `range_ring`; capability + declared channel. (2) Per-step mechanical energy is
  integrated per physics substep inside the env and reported in `StepResult` (so hook-fed recordings keep exact
  `energy_j` / `cot`). (3) Remove the redundant `h_*` branch of `make_legged_env` (tasks build through
  `TaskSpec.build`). (4) `humanoid_gen.py`: the invalid `"manipulate"` capability (phum bodies with arms fail to
  compile). Tests: Warp / MuJoCo obs-dim parity for steps and gap actors; phum_0 compiles with arms; legs goldens
  unchanged.
- **HD1.** The collector records what training needs: teacher `upper` targets per tick (rows for upper joints,
  with a per-row `upper_valid`), `terrain`, `terrain_valid`, `foothold_cell`, `com_support`, `com_support_valid`
  (from `StateView` / the public scan), real kinematic parent ids and trunk-link ids in `LeggedMorph` (replacing
  HL's derivation — report the consumer line to HD2), one `--out` per task, and `TaskView` handles every
  registered `h_*` graph (widen the event / entity slots in ONE constant and report the dims to HD2). Port the
  private step loops of both collect files onto `harness.rollout` with a recorder hook; goldens of a 20-tick stub
  episode recorded before the port stay identical apart from the added keys. Test: a stub-tracker episode per
  `h_*` task writes all keys.
- **RG.** Shard rows must be forwardable: `relations_data` stores each sample's family collate input (the
  featurizer output on the snapshot observation) next to tokens and labels; unknown transform names raise
  (`relgen.transform_def`). `relation_batches` rows are forwarded by the trainer with the action loss masked and
  `estimates_loss` / readout losses on; wire the hook into the arm trainers (`latent_train.py` stage A and flow,
  `behavior.py`), with `batches.observe_estimates(step, metrics)` after each loss and `relgen` as a recipe input of
  `train_rep` / `train_flow` in the relations template. Remove the template's "trainer does not read the shards"
  caveat. Tests: 3 CPU steps of each arm trainer on a tiny pack + tiny shards where the factor loss of a mix row
  is non-zero and the action loss of that row is zero; `schedule.jsonl` written; replay reproduces batch
  composition. Smoke: `relations_geo` tiny profile on the peer.
- **AR.** `templates/arm_bc.yaml` must dry-run (placeholder convention `SET_IN_*` carried to the plan like the
  other templates); `recipes/README.md` and docstring examples name v8div; add `recipes/armdiv/
  arm_targets_v6ref.yaml` (the v6 reference cells for G4 contrast (a): same target recipe with `flow_ref` /
  `rep_ref` / `bc_ref` inputs and `pin_sha256` placeholders that refuse until filled); write the pin procedure in
  `research/tracks/armdiv.md` (which node produces each pending sha256, where it is recorded) and keep the G4
  entry DRAFT with its two open items listed for the lead; the v8div recipes depend on nothing unmerged.
  Acceptance: all armdiv recipes and all arm templates dry-run; `recipe.*` goldens for the new file.
- **DP.** Remove the dual `refit` stage and the template's `stageA` / `F0` training nodes; eval nodes take an
  external checkpoint input (`inputs.checkpoint`, run id); `pipelines/dual.py` uses `tasks_in('mujoco/dual')`;
  move `dual_validate`'s `run_one` / `main` / `summarize` into `harness/eval/dual_teacher_quality.py` (update the
  `cli/tools.py` target via its owner SL — report the line; update `test_layering`), delete
  `run_dual_teacher_episode` (`functional_composition` runs on `rollout`), and replace `PACKET_EDITS["swap_slots"]`
  by `packets.chunk_hook("swap_assembly", a=0, b=1)` (P3's locked-equal edit). Template dry-runs.
- **RL.** Port the remaining private loops: `harness/data/collect.py:269`, `ladder.py:490`; `_ladder_task` uses
  `TaskSpec.max_steps`; port `latent_eval.disturbance_test` onto `rollout` with `hooks.PrevAction` and delete
  `install_prev_action`; `ladder_cli.py` takes sealed targets from `rrp.bodies.armdiv.is_sealed_target`;
  `grasp_rig.py` runs on the bare-model bench env pattern of `tracker_validation._BenchEnv` (an Env, so `rollout`
  drives it). Goldens recorded before each port. `policies/oracle.py::ShadowTeacher.lookahead` stays (a look-ahead
  inside a discarded snapshot, in a layer that cannot import the harness): X1 allowlists exactly that call.
- **HS2.** (1) `upper_amp` curriculum in the `*_ub` recipes (0.1 → 0.4 rad over the first 30 % of iterations;
  payload 0 → full over the same ramp). (2) Shared morphology tracker with the upper block: obs format `morph_v2`
  (`morph_v1` + upper-body block), `MorphMultiEnv` supports `upper_body` (per-group dims checked, H7 tests
  extended). (3) `rrp train tracker-install <run> --body --version`: copies `actor.pt` to the store, writes
  `meta.json` with sha256, obs format, extra-obs kind and gate result (this is what replaces
  `SET_WHEN_REGISTERED`). (4) The tracker trainers call the sealed guard. Tests: fake-env 2 PPO updates for a
  `*_ub` and a `morph_v2` recipe; install round-trip on a tiny actor. Smoke: 6 iterations of `t1_steps_ub` on the
  peer.
- **HD2.** `LeggedData` is task-agnostic (events / context from `TaskView`, no `waypoints` key); `amask` and the
  targets cover the upper rows where `upper_valid`; the rep checkpoint meta carries `upper_trained: bool` and the
  action-group list; terrain / foothold / com labels feed `estimates_loss`; `legged_bc` uses `data.train_batch`;
  `relation_batches` hook as in RG; nets are built from the config's factor list with
  `resolve(family="legged")`. Tests: 3 CPU steps of rep, flow and BC on a tiny `h_reach` pack (upper loss
  non-zero) and a tiny `h_steps` pack (foothold estimate supervised); old waypoint packs still train.
- **HP.** `policies/legged.py` and `legged_dagger.py` build nets from the checkpoint (`load_legged_rep`, factors
  from the stamp, terrain keys fed at deploy); `upper=True` is refused unless the checkpoint says
  `upper_trained`; `control="wholebody"` adapter. Test on tiny random checkpoints (both refusals, one accepted).
- **PS.** Split `heldout` into the gate-only call (`--stage-a`, writes `packet_gate.json` with `gate.gap`,
  `gate.margin`) and the later model comparison; the structured node depends on the gate node; dedicated stage
  names (`labels`, `gate`) now that families load at plan time; `OursModel` loads with `N.load_structured` (factor
  hash checked); one ext-dir resolver (`envs.simple.compat`); recipes dry-run with the order features → labels →
  stage A → gate → head → eval. Smoke: `rrp data psi0-labels` for one episode on the peer.
- **PC.** Pipeline on `SPLIT_PATH_V2` (imported, not duplicated) with `env_kw={"strings": "procedural"}`,
  `--eng-version cw_pointer_eng.v2 --key-head`; `SealedSplit("cworld_pointer_v2")` replaces the local run-once
  log; pointer collect on `harness.rollout`; pointer trainers call `relation_batches`; `ui.drag_to` gt labels from
  the teacher's drag target in `relgen/ui.py`; checkpoints stamped / checked through `stamp_versions` /
  `require_factors`; recipes for collect (v2 format) → rep → flow / bc (≥ 3 seeds) → dev eval, sealed eval as a
  separate gated recipe. Smoke: `pointer_smoke` recipe on the peer (collect 4 episodes + 20 steps).
- **RP5.** The recorder becomes rollout hooks (`FrameCallback`, `CommandLog`, energy from `StepResult`), fed by
  `evaluate()`, `run_ladder`, the legged evals and the bench envs (`tracker_validation`, `grasp_rig`); delete
  every private `mj_step` / `.step` loop in `viz/record.py`; the recorded-equals-unrecorded ladder test and the
  exported schemas stay.
- **HA.** Humanoid family stages for the adapting cells: `adapt_refit` (system-0 refit from a target demo pack),
  `adapt_flow` (flow warm start), `adapt_bc` (BC SFT warm start), `adapt_ppo` (Level-1 tracker fine-tune and
  scratch, through the HS2 trainer) — each with the demo-budget accounting the transfer driver reads, sealed guard
  with SL's native cell ids, no `missing_run` cells left in a dry-run table; delete `check_waypoint_free`;
  `humanoid_eval` uses `tracker=` (no monkeypatch); `legged_latent_eval` evaluates `h_*` tasks. Tests: tiny CPU
  run of each adapt stage on random checkpoints; accounting test.
- **HR.** Recipes (all dry-run; the node lists go into `research/tracks/humanoid.md`, whose stale lines are
  fixed): tracker training + gate + `tracker-install` for every run of section B (scan trackers for `h_steps`,
  scan + range-ring for `h_gap`, `*_ub` wholebody, shared `morph_v2`), `transfer_h_*` for every registered
  humanoid task including `h_carry`, `h_loco_pick` and the held-out tasks (held-out: eval-only), with the adapting
  nodes; `legged_lineage` adds `preset:legged`; no `SET_WHEN_REGISTERED` left — tracker inputs are run ids of the
  tracker recipes.
- **X1.** Move `harness/eval/hooks.py` to `harness/hooks.py` (one hooks module beside `rollout.py`, `HOOKS` table
  there; delete the copies in `evaluate.py` and `data/collect.py`); one file-digest helper; one Wilson (documented
  n = 0 convention); rename `harness/train/rollout.py`; delete F0's `EXEMPT`; lint "no `.step(` / `mj_step(`
  outside `harness/rollout.py` and env modules" with the single oracle look-ahead allowlisted; rollout asserts
  failure reasons are in the task vocabulary.
- **X2.** Progress-state vocabulary everywhere; architecture 14.1 (hooks location, bench env), 14.3 (scan is in
  the yaw frame; `range_ring`), relations.md (shards live under `<run>/relgen`; per-family live-site table);
  roadmap lines A3 listed marked parked; STATUS regenerated from the ledger of this file.

## B. training runs that open the campaign (in order; NOT code units)

Each run's code deliverable (recipe that dry-runs + the smoke named above) belongs to the unit in the last column.
Nothing here starts before section A is merged and the integration check is green.

| # | run | gate / output | recipe owner |
|---|---|---|---|
| T0 | peer store check: pinned trackers present with the recorded sha256 (`t1:contact_v2` w8d install), v7div pack + BC 1701 present and hash-equal | pins verified; mismatches fixed by `tracker-install`, not by editing tests | HR, AR |
| T1 | terrain-scan trackers: `h_steps` (t1, g1, h1; `terrain_scan` actor), `h_gap` (t1, h1; + `range_ring`) | D-112 gate; `tracker-install`; legacy privileged steps actors stay labelled `privileged_teacher` | HR (HS2 trainer) |
| T2 | wholebody trackers `*_ub` (t1, g1, h1) with the amplitude / payload ramp | gate + teacher check: `h_carry` ≥ 10/12 both sides, `h_steps_carry` ≥ 8/10; else C1 / held-out carry are reported `blocked_external` and excluded before collection | HR |
| T3 | shared `morph_v2` tracker (legs + upper) on the source pool | gate on every source body; input of Level-1 adaptation | HR |
| T4 | humanoid collect for every `h_*` task with the installed trackers (teacher quality table per task) → pack | per-task teacher success ≥ 0.8, else the task is held back with a note | HR (HD1 collector) |
| T5 | humanoid lineages: rep → flow / BC per task group (`preset:legged` vs `legged-none`, ≥ 2 seeds), dev transfer tables | dev tables; then a written decision before any sealed cell | HR (HD2, HA) |
| T6 | armdiv: BC 1702, kinfeat BC 1701 → fill pins; v6 reference evals (or contrast (a) dropped if the v6 checkpoints are gone) → lead signs G4 → v8div lineages → G3 gate → sealed G4 | signed pre-registration; G3 gate | AR |
| T7 | Ψ₀: label recording → stage A (D-141 fix) → packet-use gate → structured head → eval | gate gap ≥ margin, else `failed_hypothesis` recorded | PS |
| T8 | pointer v2: re-collect (geometry + UI fields) → rep → flow / bc × 3 seeds → dev tables | dev tables; sealed only after a written decision | PC |
| T9 | relation-factor experiments: `relations_data` shards → arm lineages with factor sets + curriculum; interference table | per-factor competence by depth | RG |

T1–T5 are serial on the humanoid track; T6–T9 are independent tracks and share the peer by priority
(humanoid > armdiv > Ψ₀ > pointer > relations).
