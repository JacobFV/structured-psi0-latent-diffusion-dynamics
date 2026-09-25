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
