# pre-training readiness plan (D-146)

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

## units

| id | unit (audit issue) | wave | deps | h | owns |
|---|---|---|---|---|---|
| F0 | hygiene sweep (D30) | 0 | – | 3 | stale module paths in docstrings / comments of every `src/`, `tests/`, `recipes/`, `ops/bin/` file NOT owned by another wave-0 unit; new `tests/unit/test_stale_paths.py`; stale commands in `research/tracks/*.md`, `STATUS.md` |
| F1 | relation runtime contract (D2, D5) | 0 | – | 16 | `policies/relations/{base,ops,catalog}.py`, `policies/nets/{batch,probes,checkpoint,flow}.py`, `harness/data/relgen/__init__.py`, `cli/tools.py`, `tests/unit/test_relations_runtime.py` |
| F2 | task / eval contract + `scene=` crash (D15, D16) | 0 | – | 10 | `tasks/spec.py`, `harness/eval/evaluate.py`, `harness/rollout.py`, `cli/harness.py`, `policies/teachers/__init__.py`, `envs/base.py`, `tests/unit/test_task_contract.py` |
| F3 | pipeline / recipe contract + kinfeat to children (D6, D7, D26, D5 manifest) | 0 | – | 14 | `core/{runconfig,provenance}.py`, `harness/dag.py`, `harness/pipelines/base.py`, `cli/dag.py`, `ops/bin/peer_sync.sh`, `policies/features/kinfeat.py`, `viz/export/relations.py`, `tests/unit/test_pipeline_contract.py` |
| K1 | Ψ₀ `StructuredHead.loss` TypeError (D17) | 0 | – | 2 | `policies/psi0/nets.py`, `tests/unit/test_psi0_train.py` |
| K2 | pointer `wpos3d` / `wcamuvd` KeyError (D21 crash part) | 0 | – | 2 | `policies/pointer.py`, `harness/train/pointer.py`, `tests/unit/test_pointer_train_smoke.py` |
| H5 | sealed split guard + sealed body adapters (D12) | 0 | – | 16 | new `core/sealed.py`, `bodies/{legged,humanoid_gen,importers}.py`, `harness/pipelines/legged.py`, `harness/train/{legged_bc,legged_dagger}.py`, `tests/unit/test_sealed.py` |
| A3 | dual-arm parking (D28) | 0 | – | 3 | `harness/pipelines/dual.py`, `policies/teachers/{dual_coord,dual_smooth}.py`, `harness/data/{collect_dual,dual_pairs,dual_quality}.py` (training-only parts), `recipes/templates/dual_lineage.yaml`, `.old/` group for the parked pieces |
| RP1 | `run_ladder` onto `rollout` (D29) | 0 | – | 10 | `harness/eval/{ladder,ladder_cli,hooks}.py`, `tests/unit/test_ladder_rollout.py` |
| RP5 | viz recorder onto rollout hooks (D29) | 0 | – | 8 | `viz/record.py`, its tests |
| HT | tracker registry, `rl_expert`, public terrain scan (D8, D10) | 1 | F2 | 18 | `envs/mujoco/{legged,legged_tracker,legged_core}.py`, `envs/warp/{task_env,tracker_env}.py`, `policies/teachers/humanoid.py`, `harness/train/{warp_tracker_ppo,tracker_training,tracker_recipes,humanoid_recipes}.py`, `tests/unit/test_trackers.py` |
| HJ | humanoid judge + task registration (D9) | 1 | F2 | 5 | new `tasks/humanoid.py`, `tests/unit/test_humanoid_tasks.py` |
| HL | legged relation sites (D3) | 1 | F1 | 12 | `policies/nets/{legged_latent,legged_bc}.py`, `harness/train/legged_latent_train.py`, catalog section `legged` |
| A1 | arm encoder / realizer factor plumbing; remove `SemanticReadout` (D4) | 1 | F1 | 8 | `policies/nets/semantic_latent.py`, `policies/system0.py`, `policies/bundles.py`, `harness/train/{latent_train,joint_adapt,baseline_campaign,behavior}.py` |
| R1 | relgen stage + trainer hook (D1) | 1 | F1, F3 | 16 | `harness/pipelines/relations.py`, `harness/data/mix.py`, `harness/data/relgen/{curriculum,geometry,contact,support,task,body,ui,transforms}.py`, `cli/{curriculum,tools}.py` (tools.py after F1), `recipes/relations/**`, `recipes/templates/relations_factor.yaml` |
| P1 | Ψ₀ factors + D-141 fix (D18, D19) | 1 | F1, K1 | 12 | `policies/psi0/{nets,train}.py`, `tests/unit/test_psi0_train.py` |
| P2 | Ψ₀ data driver + SIMPLE plumbing (D20) | 1 | F2, F3 | 10 | `policies/psi0/data.py`, `envs/simple/**`, `cli/{train,data,ext}.py`, `harness/pipelines/psi0.py`, `recipes/psi0/**`, `recipes/templates/psi0_step2.yaml` |
| C0 | pointer module split (D24) | 1 | K2 | 8 | `policies/pointer.py` → `policies/pointer/` package + `policies/nets/pointer.py`; `harness/train/pointer.py` → package |
| A2 | armdiv restart (v8div) + G4 protocol (D25, D27) | 1 | F3 | 12 | `harness/pipelines/arm.py`, `harness/eval/target_eval.py`, `bodies/armdiv.py`, `recipes/armdiv/**`, `recipes/templates/arm_{lineage,targets_bc,targets_latent,bc,collect}.yaml`, `recipes/presets/eval-armdiv_v1.json`, `tests/unit/test_recipes_armdiv.py`, `research/tracks/armdiv.md` |
| RP2 | arm eval loops onto `rollout` (D29) | 1 | F2, RP1 | 14 | `harness/eval/{latent_eval,latent_causal,latent_semantic_edits,edit_harness,latency,teacher_quality,video_arm,dual_latent_eval,dual_teacher_quality,video_dual}.py`, `policies/teachers/dual_validate.py` |
| U1 | upper-body control (`wholebody`) in both backends (D13) | 2 | HT | 14 | `envs/mujoco/{legged,legged_core,legged_vec}.py`, `envs/warp/{task_env,tracker_env,model}.py`, `harness/train/{tracker_recipes,humanoid_recipes}.py` (`*_ub` recipes), `tests/unit/test_wholebody.py` |
| H7 | MorphMultiEnv tests (D14) | 2 | HT | 4 | `tests/unit/test_morph_multi.py`, `tests/gpu/test_morph_multi.py` |
| H4 | humanoid collect (D11) | 2 | HT, HJ | 10 | `harness/data/legged_latent_collect.py`, `harness/data/legged_collect.py`, `policies/features/legged.py` |
| C1 | pointer training correctness (D21, D22) | 2 | C0, F1 | 14 | the pointer packages (collect / data / train / runtime modules) |
| P3 | Ψ₀ policy surface (D20) | 2 | P1 | 5 | `policies/psi0/__init__.py`, `policies/packets.py` (tensor-level edit registry) |
| RP3 | data / train loops onto `rollout` (D29) | 2 | A1, H5, RP1 | 14 | `harness/data/{collect,vlm_features}.py`, `harness/train/{sft,expo,grpo,latent_grpo,rollout,vlm_train,adapt}.py`, `policies/oracle.py`, `policies/teachers/arm.py` |
| U2 | upper-body teacher + tasks L0, L3, M1–M3 (D13, D9) | 3 | U1, HJ, H4 | 18 | `policies/teachers/{humanoid,legged_loco}.py`, `tasks/humanoid.py`, `envs/mujoco/humanoid_scenes.py`, `tasks/graphs/h_*.json`, `tests/unit/test_humanoid_manip.py` |
| HX | humanoid system 0 for `upper` + packet assemblies (D13) | 3 | U1, HL | 8 | `policies/legged.py`, `policies/nets/legged_latent.py` (realizer output groups) |
| C2 | pointer copy head + discrete key code + v2 split (D23) | 3 | C1 | 20 | pointer encoding / nets modules, `envs/computerworld.py` (string pools), `research/splits/cworld_pointer_v2.json` |
| C3 | pointer recipes + CW scene parts (D7, D1) | 3 | C1, R1, F3 | 8 | `harness/pipelines/pointer.py`, `recipes/pointer/**`, `recipes/templates/pointer_lineage.yaml`, `harness/data/relgen/ui.py` (CW parts), `research/tracks/pointer.md` |
| U3 | carry / loco-pick tasks C1, C2 + held-out tasks (D13, D9) | 4 | U2, HX | 14 | `tasks/humanoid.py`, `envs/mujoco/{humanoid_scenes,legged_scenes}.py`, `policies/teachers/humanoid.py`, `tasks/graphs/{carry_tray_level,loco_pick,h_*}.json` |
| H6 | transfer driver + humanoid family + recipes (D8, D7) | 4 | HL, HT, HJ, H4, H5, F3, U2 | 14 | new `harness/pipelines/humanoid.py`, `harness/eval/humanoid_eval.py` (→ the transfer driver), `recipes/humanoid/**`, `recipes/templates/humanoid_task.yaml`, `research/tracks/humanoid.md` |
| RP4 | legged / humanoid loops onto `rollout` (D29) | 5 | H6, HX | 10 | `harness/eval/{legged_latent_eval,tracker_validation,robustness,video_legged,deploy_eval}.py`, `harness/train/legged_dagger.py` (loop only) |
| X1 | residual dedupe (D31) | 5 | all above | 5 | one file-digest helper, one Wilson n = 0 convention, `harness/train/rollout.py` rename, callers |
| X2 | docs truth pass (D31) | 6 | X1 | 4 | `STATUS.md`, `README.md`, `docs/*.md`, `research/tracks/*.md` progress states |

Total ≈ 371 h in 35 units. Critical path: F2 → HT → U1 → U2 → U3 → H6 → RP4 → X1 → X2 (≈ 107 h serial);
F1 → HL → HX joins at U3 / H6.

## waves

- **Wave 0 (launch now; mutually disjoint):** F0, F1, F2, F3, K1, K2, H5, A3, RP1, RP5.
- **Wave 1 (as deps merge):** HT, HJ (F2); HL, A1 (F1); R1 (F1 + F3); P1 (F1 + K1); P2 (F2 + F3); C0 (K2);
  A2 (F3); RP2 (F2 + RP1).
- **Wave 2:** U1, H7 (HT); H4 (HT + HJ); C1 (C0 + F1); P3 (P1); RP3 (A1 + H5 + RP1).
- **Wave 3:** U2 (U1 + HJ + H4); HX (U1 + HL); C2 (C1); C3 (C1 + R1 + F3).
- **Wave 4:** U3 (U2 + HX); H6 (HL, HT, HJ, H4, H5, F3, U2).
- **Wave 5:** RP4 (H6 + HX); X1. **Wave 6:** X2.

File hand-overs between waves (the later unit owns the file only after the earlier merges):
`policies/psi0/nets.py` K1 → P1; pointer files K2 → C0 → C1 → C2; `envs/mujoco/legged*.py`, `envs/warp/*` HT → U1;
`tasks/humanoid.py` HJ → U2 → U3; `policies/teachers/humanoid.py` HT → U2 → U3; `nets/legged_latent.py` HL → HX;
`harness/train/legged_dagger.py` H5 → RP4; `harness/eval/hooks.py` RP1 → (RP2, RP3, RP4 add hooks in sequence);
`catalog.py` F1 → HL (own section only); `cli/tools.py` F1 → R1 → H6; `harness/train/legged_latent_train.py` is HL's (it adds H5's sealed-guard call); `recipes/templates/*` per owner above.

## briefs

- **F0.** Build the old → new module-path table from `docs/architecture.md` (history in `.old/docs/`) and rewrite
  stale `rrp.(data|training|evaluation|contracts|teachers|pipelines|controllers|models|features).` references in
  docstrings and comments; delete or fix dead references to removed scripts; remove pycache-only directories
  locally. Add `tests/unit/test_stale_paths.py`: greps the live trees for the removed prefixes with an allowlist of
  the recorded version-string constants (D-146 item 7: those are NOT renamed). Skip files owned by other wave-0
  units (their owners fix their own; the lint test lists them as temporarily exempt and X1 removes the exemption).
- **F1.** Implement relations.md section 11 exactly: `FamilyTokens` / `FAMILIES`, `resolve(..., family=, env_caps=,
  training=)` checks, the production token-set builder for arm / dual (ContextEncoder and the act set carry fields
  and, in training, labels), `estimates_loss`, `stamp_versions` / `require_factors` used by `save_checkpoint` /
  `load_checkpoint`, `relgen.load_families()`, `rrp factors coverage`. Goldens unchanged (default preset). Tests:
  every catalog factor a family claims resolves and runs a forward without KeyError; a toy batch where a
  probe-source field estimate and the `ix.support` pair estimate both decrease their loss over 50 CPU steps;
  unsupported factor → clear FactorError; coverage JSON asserted.
- **F2.** Implement architecture 14.1. TaskSpec gains `build`, `scene`, `hooks`, `teacher`, `max_steps`,
  `failure_reasons`; move the per-family string comparisons of `cli/harness.py` and the if / elif of
  `teachers/__init__.py` into TASKS / POLICIES entries; `make_env` / `evaluate` pass `scene` only when declared and
  negotiate a scene on a scene-less env as a reason; `--env-kw`; `rollout` negotiates every env; one `DUAL_TASKS`.
  `Env.failure_reason()` optional protocol method. Tests: parametrised build of every ENVS id through `evaluate()`
  with and without a scene (stub factories for simple / warp / computerworld); every TASKS entry has a resolvable
  teacher or an explicit None. Eval-loop goldens unchanged.
- **F3.** Implement architecture 14.2: `register_stage` open registry (families register in their own pipeline
  module), `apply_run_context(cfg)` at stage entry in parent and child (factor list → featurizer options; delete
  the `$RRP_KINFEAT` environment variable; subprocess test: a kinfeat lineage child loads its checkpoint, a
  mismatch still raises), adoption rule with `PIPELINE_VERSION`, catalog version and the `stale` state +
  `--adopt-stale`, ledger revision / tree hash / dirty (untracked counted; fix `peer_sync.sh`), manifest factor
  versions, room exporter reading `pipeline_manifest.json`. The `recipe.*` goldens change only if rendered configs
  change — they must not.
- **K1.** `StructuredHead.loss` calls the probe without `zmask` and `probe_loss` without `specs` (TypeError on the
  first step). Fix the call (use the probe's own specs and an all-true mask for the fixed G1 assemblies) and add
  one forward / backward test each for StageA, DirectHead and StructuredHead on random tensors. Nothing else.
- **K2.** `_relctx` indexes `b["wpos3d"]` / `b["wcamuvd"]` that `Demos.batch` never supplies. Make `Demos.batch`
  supply every key the net reads (from the stored widget table; zeros + invalid mask where the pack lacks them) and
  add a test running two steps of each trainer (rep, flow, bc) on a tiny synthetic pack. No refactor (C0 / C1 do it).
- **H5.** `core/sealed.py`: `SealedSplit.load()` (sha256 of `research/splits/humanoid_v1.json` pinned in code),
  `assert_train_allowed(bodies, seeds)`, `sealed_eval(cell)` with the run-once log
  `artifacts/runs/humanoid/sealed_log.jsonl`; call the guard in every legged / humanoid data and train stage you
  own. Write the missing adapters (n1, berkeley, toddlerbot variants, g1_hands, sealed phum region) with a
  load-and-stand test each (CPU MuJoCo fixture, a few ticks; skip-marked without Menagerie assets). Red / green
  tests for both refusals. A body that cannot be adapted is reported to the lead, not cut.
- **A3.** Park dual training: unregister dual `dagger_collect` and the dual BC-expert stage; move the W12
  coordination-teacher stubs and the dual training-only data helpers to `.old/` (group README citing D-146 item 5);
  keep `mujoco/dual`, its tasks, its evaluation teachers and the template's collect / eval nodes; dry-run the
  template. Mark the roadmap rows out of scope (report the lines to X2 if the file is not yours).
- **RP1.** Before touching the loop, add goldens of `run_ladder` on the tiny procedural-arm fixture (R0 teacher, R1
  oracle, R2 generated; a few ticks; rows digest) on the current code. Then re-implement `run_ladder` on
  `harness.rollout` + hooks (the monkeypatched `s.step` becomes `on_step` hooks; define `HOOKS: dict[str, factory]`
  in `hooks.py` as architecture 14.1 names it). Goldens byte-identical; `rrp suite ladder` unchanged.
- **RP5.** `viz/record.py` drives its own step loops; re-express the recorders as rollout hooks fed by
  `evaluate()`. Existing record tests and exported schemas unchanged.
- **HT.** Implement architecture 14.3: the tracker registry over `artifacts/trackers/<body>/<version>/meta.json`,
  `tracker=` on `make_legged_env` / `LeggedSession`, the public `terrain_scan` sensor in both backends (one cell
  layout constant shared by Warp and MuJoCo; noise / dropout / latency model; declared channel), actor obs =
  proprio + scan, critic keeps the exact scan, `rl_expert` policy with correct source labels, one recipe registry
  and one `_TURN`. Tests: two versions of one body load by id; Warp actor obs dim equals the MuJoCo adapter's;
  stub-actor MuJoCo smoke; fake-env trainer runs 2 PPO updates and checks meta keys. The D-126 physics goldens of
  `control="legs"` stay byte-identical.
- **HJ.** `tasks/humanoid.py`: judge mapping `env.failure_reason()` + truth to the vocabulary of 14.1; register
  `h_steps`, `h_gap` through `TaskSpec.build`; one synthetic test per failure reason.
- **HL.** Implement relations.md 11 (legged part): `legged-rel-v1` entries and presets in your catalog section;
  Context / LeggedEncoder / LeggedFlow build a RelCtx with joint, limb, foot and terrain-cell token sets and run
  FactorSites at `ctx>ctx`, `act>ctx`, `act>act`; the trainer resolves `factors` with `family="legged"`, uses
  `readout_loss` / `estimates_loss` per spec (delete the scalar reduction and the hand-written probe loss), stamps
  versions. Default `legged-none`: old checkpoints load, legged goldens unchanged. Test: toggling `leg.foothold`
  and `edge.kin_parent` changes attention and output on a tiny humanoid batch.
- **A1.** `TargetEncoder`, `make_realizer` and the run config take factor lists per site (encoder, realizer,
  policy); checkpoints store the resolved hash; assert that compared methods of one recipe share factor sets; remove
  `SemanticReadout` / `aux` (D-146 item 4: new lineage boundary) — the BC goldens that set `aux=False` stay
  identical; one `SFT_STEPS` / `SFT_LR`.
- **R1.** Make `relations_data` a registered stage of family `relations` (manifest with shard hashes and catalog
  version; snapshot collector through `StateView`); `relation_batches(cfg, out_dir)` per 14.2; an unknown label /
  transform raises; `rrp steer` reads / appends the run's `steer.jsonl`. Tests: 2-step CPU run where shard rows
  enter the loss and `schedule.jsonl` parses; replay reproduces batch composition; resume adopts by hash. Trainer
  call sites: report the one-line hook to A1 (arm) and HL (legged) owners.
- **P1.** `factors` on StageA, ContextTokens, StructuredHead, `load_stage_a` and the CLI, stored in the checkpoint
  (`stamp_versions`). Implement architecture 14.5 (constant-input mask in the checkpoint, state dropout +
  permuted-packet hinge in `StageA.loss`, the heldout gate that blocks the structured stage). Red / green test
  reproducing the ±1 torso-command shift on synthetic data. `load_stage_a` validates config.
- **P2.** Feature / label driver on `harness.rollout` with `LabelRecorder`; per-item seeded generators in
  `CachedDataset` (one epoch identical across worker counts); one ext-dir resolver; render profile as an env
  kwarg; the psi0 family stages registered (features → stage A → probes → heldout gate → head → eval) and the
  recipes dry-run.
- **C0.** Split the two pointer monoliths into packages with the public import path unchanged; nets in
  `policies/nets/pointer.py` (lazy torch import so the scripted route stays torch-free); one generic training loop;
  bound ids raise on collision; dt and screen size from the env spec. Frozen checkpoints give identical outputs.
- **A2.** Implement D-146 item 4: `recipes/armdiv/arm_lineage_v8div*.yaml` (inputs: the v7div pack and BC expert by
  run id + sha256), the `arm_targets_v8div_*` instances, `eval-armdiv_v1.json` with a pre-registration entry in
  `research/tracks/armdiv.md` (lead signs before training); sealed constants from `rrp.bodies.armdiv`; the heldout
  guard compares against the run's real train set; move the v7div lineage recipes to `.old/`; `recipe.*` goldens
  for the new recipes; dry-run all.
- **RP2 / RP3 / RP4.** For each owned file: record a golden of the private loop's rows on a tiny fixture first,
  then port the loop to `evaluate()` / `rollout()` + hooks (add hooks to `hooks.py` after the previous RP unit
  merged), keep the golden byte-identical, delete the private loop. Dual evaluation loops are ported (the env is
  registered); dual training loops were parked by A3. No loop may call `session.step` outside `rollout`
  afterwards (add the grep to `test_stale_paths.py` via its owner, X1).
- **U1.** Architecture 14.4 upper body: `control="wholebody"` adds the `upper` joint_position group (arms, waist,
  head; PD at 50 Hz) in MuJoCo and Warp; `control="legs"` byte-identical (goldens); tracker observation gets the
  upper-body joint state; `*_ub` tracker recipes with random upper-body targets and payload. Tests: group widths
  per humanoid body; a held upper pose tracks within tolerance for 1 s on the CPU fixture; legs goldens unchanged.
- **H7.** CPU fake-env test of MorphMultiEnv routing / concatenation and that every group agrees on obs / priv
  dims; GPU test (skipped without warp) stepping 10 ticks with independent per-group reset.
- **H4.** `--task` through the task registry and the teacher factory; EVENTS and `public_context` from the
  TaskSpec; manifest records task, teacher source and tracker sha; sealed guard called. Test: 20-tick episode per
  task on a stub tracker.
- **C1.** Collect stores pos3d, cam_uvd, zlayer, parent, focus rank and UI edges through ONE featurizer path shared
  with live rollout; `factors` threaded through nets and `_save` (hash checked on load); `--seed` separate from
  `--split-seed`; per-episode flow noise from (policy seed, env seed, packet index). Tests: train and inference
  batch key sets equal; identical z at batch 1 and batch 16; `ui.same_window` / `ui.label_for` change attention.
- **P3.** Tensor-level packet edits (`mean_packet`, `zero_slot`, `swap_assembly`) in one registry used by
  LatentStackPolicy and the Ψ₀ policy; provenance with source kind, stage-A hash and input spec; direct and
  structured arms receive identical inputs (test).
- **U2.** Scripted upper-body teacher (IK reach / grasp on the humanoid hands, public FK, `scripted_teacher`)
  composed with `rl_expert` legs; tasks L0 (stand / walk-to), L3 (turn in place), M1 reach, M2 squat-pick, M3
  place: TaskSpec + scene builder + task graph + judge reasons; a short scripted-teacher fixture test per task (CPU,
  a few seconds of sim inside the unit test budget, skip-marked if assets are missing).
- **HX.** Humanoid system 0 realizes the `upper` group from the packet (arm / hand assemblies join the packet's
  assembly set); legs-only checkpoints still load; test on a tiny random model.
- **C2.** Architecture 14.6: mixture key head with copy attention over instruction characters, `cw_pointer_eng.v2`
  7-bit key code (v1 decodable), procedural strings, `cworld_pointer_v2.json` declared before any demo. Tests:
  next char equals `instr[n_typed]` on unseen synthetic strings; encode / decode round-trip over KEY_VOCAB; split
  disjointness.
- **C3.** Pointer family stages registered; `recipes/pointer/` lineage (collect → rep → flow / bc → eval on dev and
  sealed with the split guard, ≥ 3 seeds); ComputerWorld scene parts for `compose`; roadmap rows with
  pre-registration notes (copy, discrete key, UI factors vs none, relgen curriculum).
- **U3.** Tasks C1 carry (tray / box while walking), C2 loco_pick, and the held-out `h_steps_carry`, `h_gap_cart`
  (declared held-out in the split; never in training lists — guard test); teacher compositions; judge reasons
  `dropped`, `hold_lost`.
- **H6.** `harness/pipelines/humanoid.py` (family stages of 14.4) and the config-driven transfer driver
  (`rrp eval humanoid-transfer`: sealed bodies × methods × demo budgets × seeds; Level 1 apart from Level 2; equal
  acquisition accounting; tables through `statistics.py`); recipes for every humanoid task; the former one-off
  eval logic in `humanoid_eval.py` is replaced (no monkeypatching). Test of demo-budget accounting; dry-run all
  humanoid recipes; rewrite the RESUME section of `research/tracks/humanoid.md`.
- **X1.** One file-digest helper; one documented Wilson n = 0 convention; rename `harness/train/rollout.py`;
  remove F0's temporary exemptions; add the "no `.step(` outside rollout" lint.
- **X2.** STATUS / README / docs / track notes: required progress-state vocabulary; "verified" only with a
  recorded command and artifact; per-family live-site table in `docs/relations.md`.
