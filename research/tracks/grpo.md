# track: grpo — packet-policy GRPO for system i (corrected architecture)

Owner: grpo track agent. Branch `track/grpo`, worktree `~/work/rrp-wt/grpo`, peer dir `/dev/shm/rrp-brandonin/wt/grpo`.

## what it is (state: implementing -> see log)
RL fine-tuning of SYSTEM I only (the flow that emits the latent packet z[knots, assemblies, dz]); system 0 (LatentRealizer),
target encoder and packet probes are frozen and not in the optimizer; the flow's context encoder is frozen too by default
(`--trainable action_expert`). Code: `src/rrp/learning/latent_grpo.py`, CLI `rrp latent grpo`, test
`tests/unit/test_latent_grpo.py`. Generic GRPO machinery reused from the old direct-action track (`flow_sde.py`, `grpo.py`,
now with a `collate_fn` hook).

- Sampler: flow-SDE (`flow_sde.py`, derivation in `research/methods/grpo-derivation.md`) run in the flow's standardized
  latent space; the packet handed to system 0 is `denormalize(z_K)` (constant Jacobian: ratios unchanged). Default K=8,
  a=0.5, clamped first step, deterministic last step (the last-step std a*h/sqrt(1-h)=0.067 makes exact-sum ratios
  explode; same choice as adapt_grpo_exact_suffix_v4). Evaluation always uses the deployed ODE sampler.
- Likelihood: exact augmented-path log-likelihood summed over VALID packet coords (4 knots x real assemblies x 64);
  padded assembly slot contributes 0. Test checks ratio == 1 at identical weights, agreement with an independent
  torch.distributions computation, padding invariance, packet == denormalized path end, and the closed-form path KL.
- Objective: GRPO clipped surrogate (eta 0.2), group-relative advantages over G rollouts of the SAME seed (identical
  initial state), zero-variance groups excluded and counted, plus exact same-variance Gaussian path KL to the frozen
  reference flow (`--kl-coef`, default 0.05: labelled departure from Z-1, which uses none).
- Reward (training signal only, never an observation), label written in every row:
  `privileged_sim_success` [+ `--shaping-events w`: PUBLIC fraction of task-runtime events succeeded]
  [+ `--shaping-reach w`: PRIVILEGED 1 - min_t d(TCP,cube)/d_start] [+ `--shaping-dist w`: PRIVILEGED cube-zone distance].
- Curriculum (labelled): `--teacher-prefix-steps N` = the SCRIPTED TEACHER (privileged planner) controls the first N
  ticks, learned suffix after. Evals then run twice: with the prefix (suffix success, not deployable) and from reset.
- Accounting per iteration in train_log.jsonl and result.json: train_episodes, train_env_steps (learned ticks),
  train_teacher_prefix_steps, train_packets, rollout/update velocity evals, optimizer_updates, informative / zero-variance
  groups; eval episodes/steps counted separately (never used for updates).
- Guard: target bodies (xarm7_pg2, xarm7_tf3, panda_tf3) refuse to run without `--allow-target` (D-025).

## CLI for the campaign (after SFT on a target)
```
scripts/peer_run.sh --gpu --gpu-mem 12G --cpu 6 --mem 24G --label grpo_<target> --max-seconds 14400 --detach -- \
  PY -m rrp.cli latent grpo --checkpoint artifacts/runs/<sft_run>/policy.pt --robot <target> --allow-target \
    --out artifacts/runs/<campaign>_grpo_<target> --iters 20 --groups-per-iter 8 --group-size 8 \
    --lr 1e-6 --kl-coef 0.05 --eval-episodes 64 --eval-every 5 [--shaping-events 0.5] [--teacher-prefix-steps N]
```
Train seeds default 3,100,000+, eval seeds 3,000,000+ (feasible seeds only, disjoint, checked). For the sealed protocol set
`--train-seed-start/--eval-seed-start` to the protocol's adaptation/test ranges. Output: `policy.pt` (loadable by
`LatentPolicy.from_checkpoint`/`rrp latent evaluate`), `result.json` (evals, accounting, modules), `train_log.jsonl`,
`eval_episodes.jsonl`.

## log
- 2026-09-25 [verified] code + tests: `pytest tests/unit/test_latent_grpo.py` (likelihood/ratio/padding/KL; batched
  system-0 == per-session tick) and `tests/unit/test_grpo_density.py` pass. Smoke: `rrp latent grpo --checkpoint
  artifacts/runs/flow_latent_sem_v1/policy_interrupted.pt --robot panda_pg2 --out artifacts/runs/grpo_latent_smoke_v1
  --iters 1 --groups-per-iter 2 --group-size 4 --eval-episodes 8 --max-steps 80` (peer lease 1790363746_124da4) ran end to end.
- Reference (base) closed-loop on panda_pg2 dev seeds 3,000,000+ (deployed ODE sampler, 300 ticks), raw rows in
  `artifacts/runs/<run>/eval_episodes.jsonl` on the peer store:
  | base | from reset | teacher prefix 40 ticks (suffix success) |
  |---|---|---|
  | flow_latent_sem_v1 policy_interrupted (4.2k steps) | 0/64 (grpo_latent_ref_v1); 0/32, min TCP-cube 0.45 m of 0.54 (grpo_latent_diag_v1) | 0/32, grasp 1/32 |
  | flow_latent_sem_v2 snapshot step 22000 (copied to grpo_base_snapshots/) | 0/32, never grasps, min TCP-cube 0.43 m | 1/32, grasp 12/32 |
  Teacher timing on panda_pg2 (seeds 3,000,000-5): public grasp event at tick 42-47, success at ~100.
  => from reset GRPO has no success signal (no grasp, zero-variance success reward); with the labelled 40-tick teacher
  prefix there is grasp/suffix variance -> used as the dev curriculum.
- Diag (v1 base, prefix 40, reward success+0.5 events+0.5 reach, 1 iter, 4x8): first-pass |ratio-1| 7.6e-5, clip frac 0.18
  at lr 1e-6/2 epochs, path KL 0.010, 3/4 informative groups. Mechanics OK.
- RUNNING: lease 1790365845_5f45eb `grpo_v2s22k_p40`: base v2 step-22000 snapshot, prefix 40, reward
  success + 0.5 public events + 0.5 privileged cube-zone dist, 15 iters x 8 groups x 8, lr 1e-6, kl 0.05, eval 64 held-out
  dev seeds every 5 iters (both prefix-40 and from-reset). Out: artifacts/runs/grpo_latent_v2s22k_p40_v1.
- 2026-09-25 lead redirect (causal-semantics priorities): no new GRPO runs; the host lr=3e-6 run (lease 1790368812_e38d14)
  was stopped right after launch, before any iteration (`ops stop --lease ... --owned-only`); its output was deleted.
  GRPO resumes on a competent checkpoint later.

## rollout failure modes (input to the ladder track, item 3)
Source: `scripts/diag_rollout_trace.py` (committed), peer leases 1790369494_42befa (from reset, seeds 3,000,000-2)
and grpo_trace_p40 (40-tick teacher prefix, seeds 3,000,000-5); raw per-tick rows in
`artifacts/runs/grpo_trace_v2s22k/trace_{reset,p40}.jsonl` on the peer store. Checkpoint flow_latent_sem_v2 at step 22000
(snapshot, not final), system 0 = latent_sem_v1 realizer, deployed ODE sampler; all distances are PRIVILEGED sim truth
used only as diagnostics. Aggregates from the 64/32-episode evals above agree.
1. **Tracking is not the bottleneck.** Arm tracking error |target - measured q| on the learned path is 0.003-0.05 rad
   (the teacher's is 0.1 rad while moving, because it commands bigger steps). The tracker follows what system 0 commands.
2. **Approach failure from reset (100% of episodes).** The TCP moves (0.02-0.23 m/s) but does not go to the cube: TCP-cube
   distance goes 0.54 -> 0.49 -> 0.58 m over 150 ticks (teacher: 0.54 -> 0.002 m by tick 40). Commanded arm-target steps
   are 0.002-0.04 rad/tick, not directed (teacher 0.05-0.065 rad/tick, monotone). Median min TCP-cube distance over 300
   ticks is 0.43-0.44 m (64 episodes). No grasp in 0/192 reset episodes across v1 and v2-snapshot evals.
3. **Gripper never commits (averaged command).** Learned gripper commands sit at 0.013-0.043 (mean ~0.03) and change
   from tick to tick; the teacher's are bimodal (0.045 open / 0.0 closed). This looks like mode averaging of a binary
   decision in the packet or in system 0 (which of the two is the ladder's question: expert-encoded packet -> system 0).
4. **Grasp/transport failure after a teacher hand-off at the cube (40-tick prefix).** The learned suffix drifts off the
   cube (TCP-cube 0.005 -> 0.03-0.17 m within 120 ticks), sometimes pushes it (cube-zone 0.156 -> 0.119 m, cube never
   lifted). The public grasp event fires in 20-31% of episodes (partial closure registered as a grasp), but the cube stays
   at rest height (z 0.022 m; one 0.049 m blip), so transport never starts. Suffix success 0-2/64.
5. No packet rejections, no fallback holds, no stale packets in any run: the packet contract and runtime are not the cause.
Implication: localize command generation first (expert native -> tracker is fine; test expert-encoded packet -> system 0
next); the gripper channel and step amplitude are the specific signatures to check.
