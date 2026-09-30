# track d126arm (D-126 arm/training code items: roadmap #4, #5, #6, #7, #8, #9, #10, #35)

Owner: D-126 arm agent. Worktree `~/work/rrp-wt/d126arm`, branch `track/d126arm`. CODE ONLY (D-126): every item is an
additive, default-OFF option; no experiment was run. Host rule D-127: only git, editing and the unit suite ran on the
host; no smoke runs were made (peer smoke runs were skipped). State per item: `implemented` = code + unit tests
(fakes / tiny models, no simulation beyond two short fixture-arm teacher episodes in the unit suite); every item still
needs a peer smoke run of its first real node before a long run.

Default-off proof: existing configs and DAGs plan to identical config hashes (tests/unit/test_dag.py legacy equality,
tests/unit/test_d126_arm_dags.py::test_existing_arm_dags_still_plan_identically); IK with limit_margin 0 is the
historical solve (bit-equal, tested); v2 teacher calls are unchanged (no extra IK keyword); build_pick_place without a
spec builds the same model arrays (tested); collection job tuples, manifests and episode meta are unchanged when no
option is set; latent_grpo writes the same config.json; the ladder CLI's --help text is unchanged (parity test).

## per item
| # | what landed | flags / entry points | tests |
|---|---|---|---|
| 8 | limit-aware IK: joint-weighted DLS + null-space push out of a margin band + margin-aware solution choice (never loses a target the historical solve reached within 2 mm); teacher version bump `v2lim` = `scripted_teacher:pick_place_v2_minjerk_lim` (margin 0.05 of range) | `IKSolver.solve(limit_margin=)`; collect config `teacher_version: v2lim` [+ `ik_limit_margin: m`] (recorded in manifest flags `teacher_kw`, episode meta, teacher diag); v2 + margin is refused | test_d126_arm_data_options.py (3-link arm: margin >= 0.02 on +8 of 80 targets, no reach lost) |
| 5 | small-amplitude descent DART: in `descend`, executed = clean + (sigma/exec_noise) x the SAME held noise draw (noise RNG stream unchanged), through `DartProximityGuard` (5 mm, non-strict) | collect config `dart_descent_sigma: s` (needs `dart_safety: phase`); meta `dart.descent`, manifest `flags.dart_safety.descent_sigma` | every descent tick goes through the guard; v6dart behaviour unchanged without it |
| 35 | task-object variants: shape cube / cylinder / box_tall / box_flat, half size, mass, sliding friction (set after the grasp contact model, which would overwrite it); descriptor stays "<color> cube" unless descriptor_shape | `build_pick_place(object_spec=)`; collect `object_variation: {size: [lo,hi], mass: [..], friction: [..], shapes: [..], descriptor_shape}` (per-seed RNG [seed, 35]); `$RRP_OBJECT_VARIATION` (JSON) for eval harnesses; meta `object_spec` (object_spec_v1) | default arrays identical; friction survives grasp_v2; env var |
| 7 | overlapping-chunk blending: crossfade (w=(i+1)/(L+1) over L ticks) or ensemble (exp(-decay*age rank)); BC: blends the new chunk's rows with the earlier chunk rows covering the same ticks; system 0: realizes the previous packet(s) at the current state and their own phase; never across an invalidation | `LearnedPolicy(chunk_blend=, blend_ticks=, blend_decay=)`, `LatentSystem0.configure_blend()`, `LadderConfig.chunk_blend`, ladder CLI `--chunk-blend/--blend-ticks/--blend-decay` (hidden from --help), stage option `chunk_blend` on eval_r1/eval_r2/heldout/target_eval; summary `chunk_blend` record; metric = motion `chunk_vel_step_max` | test_chunk_blend.py (weights, rows, packets, batched == single tick) |
| 6 | pipeline stage `grpo`: method latent (latent_grpo with the DEPLOYED system 0 `representation`, budget in new control transitions, policy_it<N>.pt snapshots) or bc (training/adapt.py method grpo = matched-budget BC-GRPO); anchor / forgetting evals `rrp.training.grpo_anchor` (ladder R2 harness on the original training bodies, reference = pre-GRPO; regressed iff pooled drop > max_drop or a body drop > max_drop_robot; action flag|stop, stop restores the last non-regressed weights; anchor_report.json) | stage `grpo` options.method/anchor; LatentGRPORunConfig.representation/budget_env_steps/snapshot_evals/anchor; adapt cfg `anchor`; eval_r2 option `route: learned` (BC on the same sets) | anchor rule math, tracker stop/restore, stage wiring (latent, bc) |
| 9 | stages `target_eval` (sealed protocol scenes; ladder routes generated / learned; `--sealed-run` required for targets and held-out source bodies; smoke only on non-target bodies with dev seeds) and `target_adapt` (flow_sft / system0_refit with the new `refit_realizer` `episode_budget` / bc_sft; protocol budgets, seeds, update counts and lr; the pack must hold only the target) | `python -m rrp.evaluation.target_eval`; DAG template `dags/templates/arm_targets_latent.yaml` (per variant x seed x target: zero-shot, flow SFT b5/20/100, system-0 refit b5/20/100, their evals; source competence; target packs) | scene rules, stage argv, budget/seed refusals |
| 10 | arm `train_bc` stage (baseline_campaign.source_config recipe) and DAG template `dags/templates/arm_targets_bc.yaml` (BC seeds 1702/1703 x targets: zero-shot, SFT b5/20/100, evals, source competence) with identical packs, budgets, harness | stage train_bc; template | fairness test (same packs/budgets/targets as the latent template) |
| 4 | recipe overlays (fragments) `dags/overlays/arm_recipe/{stageA_beta_kl,bcdagger_schedule,refit_lengths}.yaml`; `load_dag` accepts `extends: [parent, fragment, ...]` | see its README.md | each fragment composed with arm_lineage_v2 changes only its declared keys; no shared outputs |
| - | stage option `grasp_contact` (sets $RRP_GRASP_CONTACT for the stage and its subprocesses; refuses a contradicting env) — needed by every v6 (grasp_v2.1) DAG; the templates set it | any stage `options.grasp_contact` | test |

CORE_API_VERSION 1.0 -> 1.1 (new stage names; additive).

## exact configs to run later (all on the peer; measure the first node's memory, declare >= 1.35 x peak, D-117)
Prerequisites: the v6 lineage set (#3) for flow/rep refs; the v6dart dataset/pack (exist); a child DAG per template.
- #6 GRPO: `dags/arm_grpo_v6.yaml`:
  ```yaml
  extends: templates/arm_grpo.yaml
  name: arm_grpo_v6
  vars:
    flow_ref: 'runs/<v6 track>/<lineage>-{variant}/flow_ft-gdag2h_s{seed}:policy.pt'
    rep_ref: 'runs/<v6 track>/<lineage>-{variant}/refit-gendag3_noqd_s{seed}:representation.pt'
    bc_ref: 'runs/armexpert_bcv6/baseline_direct_action:seed1701/source/policy.pt'
    bc_label: bcv6_direct1701
  ```
  `RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/<track> rrp run-dag dags/arm_grpo_v6.yaml --point variant=semfix,seed=1` first.
- #9: `extends: templates/arm_targets_latent.yaml` with flow_ref / rep_ref as above and
  `dataset: datasets/pick_place_primary_v6dart`. Sealed: run once, after the lineage set is final (protocol dev_rule).
- #10: `extends: templates/arm_targets_bc.yaml` with `dataset: datasets/pick_place_primary_v6dart`,
  `source_pack: packed/latent_pp_v6dart_s1_H16` (or reuse the v6 BC chain's seed 1702: set bc_ref, drop train_bc1702).
- #4: `extends: [arm_lineage_v6.yaml, overlays/arm_recipe/<fragment>.yaml]` (one DAG per fragment; nosem only).
- #7: add `chunk_blend: crossfade` (and `blend_ticks: 4`) to the options of an eval_r2 / target_eval node, or
  `scripts/ladder.py ... --chunk-blend crossfade --blend-ticks 4`; compare `motion.chunk_vel_step_max` and success
  against the same node without it (same seeds).
- #8: a collection with `teacher_version: v2lim` on the procedural bodies (the v6dart config with that one change),
  then `rrp.evaluation.gates.check_arm_dataset` with the parm* margin gated.
- #5: the v6dart config + `dart_descent_sigma: 0.02` (with `dart_safety: phase`), then the penetration gate on DART episodes.
- #35: collect config `object_variation: {size: [0.018, 0.028], mass: [0.02, 0.3], friction: [0.5, 1.2], shapes: [cube, cylinder, box_tall, box_flat]}`
  (a new dataset version), or evaluate with `RRP_OBJECT_VARIATION='{...}'`.

## known limits / open points
- Nothing here has been run on the simulator beyond the unit tests; resources in the templates are provisional.
- latent GRPO rollouts and feasibility use v1-teacher feasibility (as before); target_eval uses the ladder harness, so
  its numbers are comparable to lineage R2 rows, not to the older latent_slice1 `evaluate_latent` cells.
- system-0 few-shot adaptation reuses the Stage-A realizer objective on the target pack only (no DAgger buffers).
- pre-existing unit failure on main (not from this track): tests/unit/test_system2_harness.py needs the menagerie go2
  asset and has no skip marker (fails identically on origin/main).
