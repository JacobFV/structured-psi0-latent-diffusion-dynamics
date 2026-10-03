# track compute: compute-aware acceleration (D-147 addendum; precision / compile / seeds / eval)

State: **verified (wiring, default-preserving)**; the Enable phase (using bf16 / compile in a real run) has NOT happened. Unit `prec` of
this track. The `seeds` and `eval` units add `seeds_per_job` / `eval_backend` behaviour to the same block.

Design: `docs/architecture.md` 14.7. Code: `src/rrp/core/compute.py`, `RunConfig.compute` (`core/runconfig.py`), pipeline channel in
`harness/pipelines/base.py` (`apply_run_context`, `stage_versions`, manifest flags), trainers (every `harness/train/*`, `policies/psi0/train.py`),
tests `tests/unit/test_compute.py`, benchmark `rrp train bench-compute` (`harness/train/bench_compute.py`).

## use
```yaml
# a recipe / RunConfig: params-level `compute` (rendered into $RRP_RUN_CONTEXT; absent = today's behaviour)
compute: {precision: bf16, tf32: true, compile: reduce-overhead, cuda_graphs: true}
```
- `precision: null` (default) = each trainer's historical numerics (behavior fp32 + TF32 matmul; pointer and Psi0 bf16 autocast; others fp32).
  `bf16` = autocast for matmul / conv / attention + the fp32 islands. `fp32` = autocast off.
- `compile: reduce-overhead` is the CUDA-graph mode (needs `cuda_graphs: true`); `max-autotune` is graph-free unless `cuda_graphs: true`.
- Every trainer writes `compute.json` (requested block, effective precision / source / islands / TF32, per-callable compile status or
  fallback reason, torch version) next to its result (pointer: `<out>.compute.json`). Non-default blocks change `config_hash` and the stage pin.

## decisions / limits
- fp32 islands: `gaussian_nll`, both `readout_loss`, `masked_mse`, `estimates_loss`, the latent / legged / codec loss reductions, realizer and
  flow targets. The adapt.py likelihood ratios are untouched (own fp32 and TF32 off). Optimizer state is fp32 (autocast never casts parameters).
- Compile patches methods on the instance (no `_orig_mod.` in `state_dict`), first-call failure -> eager, recorded.
- Tracker PPO is not wired (see decisions D-147 addendum, unit prec). `target_adapt` does not exist. VLM inference autocasts in
  `policies/psi0/__init__.py` / `psi0/data.py` are left alone (feature extraction, not a trainer).
- Legacy Psi0 gated autocast on `dev == "cuda"` exactly; the helper uses `startswith("cuda")` (the pipeline passes "cuda").

## measurements (peer, tiny real batches)
See below; produced by `rrp train bench-compute` (median wall time of one optimizer step, cuda-synchronised, first 20 steps excluded as
warm-up / compile; settings interleaved per trainer; the peer GPU is SHARED with other tracks, so absolute numbers are noisy).

**State of the measurement: blocked_external (2026-10-02 16:30).** Both peer GPU slots were held for hours by other tracks (pointer
`ptrcp_nosem3_bc`, relations `rfgeo2_F0`) and a humanoid GPU node was queued (`t1_steps_p2fix`), so the humanoid-first handoff rule forbids
a third lease; the host GPU is paused. No GPU speedup number exists yet. Verified without a GPU (peer CPU lease, `CUDA_VISIBLE_DEVICES=`,
4 steps each, bench plumbing only, NOT a speed result): `arm_rep`, `arm_flow`, `arm_bc` (latent rep / latent flow / behavior BC) and
`pointer_rep` run to completion under `default` and `bf16`, and the stamp reports the effective settings (CPU: bf16 autocast is slower,
as expected). `legged_*` cases cannot run: the peer's legged packs (`datasets/legged_latent_v1/v2`) predate the current format
(`goal_columns` IndexError on `ctx`), and fresh humanoid shards are on the host. `psi0` needs the VLM / cached features and `joint_adapt`
a trained adapt pack, so neither was benchmarked; `target_adapt` does not exist; tracker PPO is deliberately not wired.

Resume (one GPU lease, <= 20 min, only when a slot is free and no humanoid GPU node is queued; declare >= 1.35x measured peak):
```sh
export RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/accel-prec RRP_PEER_PYTHONPATH=/home/brandonin/work/ext/cw-site
ops/bin/peer_sync.sh push
ops/bin/peer_run.sh --gpu --gpu-mem 12G --cpu 4 --mem 24G --label prec_bench --max-seconds 1200 -- \
    PY -m rrp.cli train bench-compute --out /dev/shm/rrp-brandonin/bench_prec.json     # default,fp32,bf16,bf16+compile x 6 trainers
```
Then fill the table here (median ms / step per trainer and setting, peak GB, compile status) and decide the Enable phase.

## warpeval: batched GPU evaluation with MuJoCo Warp

### design (hybrid, one code path)
`harness.rollout(..., backend="warp")` runs the SAME per-episode CPU `Session` objects (tasks, judges, hooks, detectors, controllers, trackers,
policies unchanged) and replaces only the physics integration of each control step by one batched MuJoCo Warp call over all running episodes.
`Session.step` is a generator that yields an `Integrate` request where it used to call `mujoco.mj_step` in a loop; the plain `step()` drives the
generator with CPU `mj_step` (identical arithmetic to before), the batched driver collects the requests of all episodes, advances substeps 1..K-1
on the GPU (CUDA graph), copies the state back and runs the LAST substep on CPU `mj_step` so the derived quantities (xpos, contacts, sensors,
actuator forces) have exactly the CPU semantics (computed before the last integration). Episodes are grouped by model structure (the arm's
distractor count changes the model); per-world differences in `body_pos / body_quat / qpos0` are batched model fields.
Unsupported combinations fall back to the CPU path per run, and the reason is RECORDED in the rows (`eval_backend_effective`, `eval_backend_fallback`):
perturbation hooks (`perturb.install_legged`), actuator-model modes, dual sessions, policies that cannot run `batch>1`.

### parity protocol (pre-registered 2026-10-02, BEFORE any Warp-vs-CPU measurement; not to be edited after measuring)
Fixture set (fixed seeds, no cherry-picking; fp64 CPU C MuJoCo is the reference):
- F1 arm: `mujoco/arm`, body `panda_pg2`, task `pick_place`, policy `teacher:pick_place` (scripted_teacher), scene `arm_scene(seed)` (distractors = seed % 3), seeds 0..11.
- F2 legged (host-runnable): `mujoco/legged`, body `hexapod6`, task `waypoint_contact`, `tracker_kind=cpg`, legs control, a scripted policy, seeds 0..7.
- F3 humanoid (peer; weights are untracked): `mujoco/legged`, body `g1`, tracker `g1:ub_v1`, tasks `h_walk` and `h_turn` under `teacher:<task>`, seeds 0..7 each.
Metrics and tolerances:
1. Single-step parity (teacher-forced, re-synchronised to the CPU state every control step, so no accumulation): the CPU episode is the reference;
   for every control step the Warp-hybrid advances from the CPU state with the CPU's own command. Statistic over all steps and episodes of the max over dofs
   of |qpos_warp - qpos_cpu| (rad for hinge/slide joints, m for free-joint position, quaternion components dimensionless):
   arm p99 <= 1e-3, max <= 1e-2; legged (F2, F3) p99 <= 2e-3, max <= 2e-2.
2. Closed-loop outcomes (HARD GATE): same policy, same seeds, `eval_backend=cpu` vs `warp`: `outcome` and `failure_reason` identical on EVERY fixture
   episode of the family. A family that fails this gate keeps `cpu` as its effective backend (recorded fallback, not an error).
3. Closed-loop trajectory divergence (reported with statistics; tolerance for "within declared tolerance"): per episode, over steps both reached:
   arm: max |Delta qpos| of the arm joints median <= 0.05 rad and p95 <= 0.15 rad; final pick-object position error median <= 0.02 m, p95 <= 0.05 m;
   legged: final base xy error median <= 0.30 m, p95 <= 1.0 m; episode end time difference <= 0.5 s.
   Closed-loop chaos is expected for contact-rich legged gaits; metric 3 is reported honestly even when it is exceeded, and a violation is
   listed as such. Gate 2 is the acceptance criterion; 1 and 3 are the declared tolerances.
Divergence statistics reported: n, mean, median, p95, p99, max per metric, per family, per backend precision (Warp fp32 vs CPU fp64).
Throughput (episodes/s) is measured on the peer: CPU workers vs Warp, at several N, with the GPU shared and busy (it is, ~90%).

### results 2026-10-02 (host, Warp on its CPU device; fp32 Warp substeps + CPU fp64 last substep)
These validate the numerics and the plumbing against the protocol above. They are NOT throughput numbers (a CPU-device Warp run is slower than CPU MuJoCo).
Raw output: `artifacts/runs/compute/warpeval/{F1,F2}_warpcpu.json` (untracked; numbers copied here).
- F1 arm (12 seeds): outcomes identical 12/12 (all `success`; the gate is therefore only exercised on the success path, no failure reason was compared).
  Single-step n=1262: median 1.8e-6, p95 1.5e-4, p99 9.97e-4, max 2.4e-3 (tolerance p99 1e-3, max 1e-2: PASS, p99 is at 99.7% of its tolerance).
  Closed loop: arm qpos max median 8.3e-5 / p95 4.0e-4 rad (tol 0.05 / 0.15), object error median 5.9e-5 / p95 1.5e-4 m (tol 0.02 / 0.05): PASS. End-time difference ~4e-13 s.
- F2 hexapod6 waypoint_contact (cpg tracker, 8 seeds): outcomes identical 8/8 (all `success`). Single-step n=7745: median 1.8e-7, p99 6.3e-7, max 1.1e-6
  (tolerance p99 2e-3, max 2e-2: PASS). Base xy error median 3.1e-6, p95 6.0e-6 m (tol 0.30 / 1.0); end-time difference max 2.4e-12 s: PASS.
  F2 note: the scripted teacher issues base_velocity commands through the cpg tracker (not raw leg control as the fixture text said).
- Wall time on the host (Warp CPU device, includes model build/compile): arm 12 episodes CPU 2.1 s vs Warp-cpu 113 s; hexapod 8 episodes CPU 1.7 s vs 29.5 s.
- F3 humanoid (g1, tracker g1:ub_v1, h_walk / h_turn): NOT RUN, state blocked_external (needs the peer, see below).
- GPU throughput (episodes/s at several N, CPU workers vs Warp): NOT MEASURED, state blocked_external. Peer `ops/bin/peer_run.sh` raised
  `AdmissionStopped: watchdog heartbeat missing or stale` on every attempt (even a 1 CPU / 1 GiB probe, 16:22 PDT); both GPU slots were held by other tracks,
  a humanoid GPU node was queued (humanoid-first rule), and the peer GPU was ~92% busy. Not worked around (no broker/watchdog changes by this unit).
  Resume: `RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/accel-warpeval RRP_PEER_PYTHONPATH=/home/brandonin/work/ext/pylibs/mjwarp ops/bin/peer_sync.sh push`, then
  `ops/bin/peer_run.sh --cpu 4 --mem 12G ... -- PY -m rrp.cli suite warp-parity --fixture F3w --device cpu --seeds 2`, then one GPU lease (<= 20 min, mem >= 1.35 x peak)
  with `--fixture F1|F2|F3w|F3t --device cuda:0 --bench 8,32,64`. The CUDA-graph capture path has never run on a GPU and is the main untested risk.
- Not done: batching the tracker MLP across envs (per-env CPU forwards remain).
- Ambient hook (after rebasing onto accel-prec): `rollout/evaluate(eval_backend=None)` read `core.compute.current().eval_backend`; an explicit argument or `--eval-backend` flag overrides it.
  `eval_backend` is stored in provenance and in run_matrix records only when not `cpu`.

## unit envs: tracker PPO throughput vs num_envs (peer GB10, one lease)

Question: does raising `nworld` (num_envs) of the not-yet-started tracker recipes (`*_steps_ub`, T3 shared morph, later trackers) raise samples/s?
Placed/running nodes, armdiv T6 phase B, T7 phase 2 and the goldens were not touched.

Method: the real trainer (`warp_tracker_ppo.py`, recipe `{t1,g1,h1}_steps_ub`) via `ops/bin/peer_run.sh`, one lease of about 11.4 min, `--iters 5`, iteration 0 dropped
(JIT/warmup), `--minibatches` scaled with `--nworld` so the minibatch stays 24576 samples (the per-sample PPO update is unchanged), nvidia-smi sampled every 2 s.
Raw output: `artifacts/runs/humanoid/accel-envs/bench-sweep_s1/` (`summary.json`, per-point `train_log.jsonl`, `.out`, `.time.txt`, `gpu_samples.log`).

| body | nworld | samples/s | competitor on GPU | GPU util % | peak GPU MiB | max temp C |
|---|---|---|---|---|---|---|
| t1 | 1024 | 12.6k | yes | 94 | 1623 | 78 |
| t1 | 2048 | 15.8k | yes | 94 | 2423 | 75 |
| t1 | 4096 | 39.4k | 20% of the time | 59 | 3921 | 68 |
| t1 | 8192 | 21.8k | yes | 96 | 7099 | 79 |
| t1 | 16384 | 17.1k | yes | 95 | 13211 | 77 |
| h1 | 2048 | 26.1k | yes | 94 | 2459 | 74 |
| h1 | 4096 | 73.9k | 75% of the time | 59 | 3983 | 77 |
| h1 | 8192 | 32.0k | yes | 87 | 7021 | 78 |
| h1 | 16384 | 32.0k | yes | 95 | 13131 | 73 |
| g1 | 4096 | 15.0k | yes | 95 | 7303 | 76 |
| g1 | 8192 | 15.7k | yes | 95 | 13839 | 78 |
| g1 | 16384 | 16.0k | yes | 95 | 26699 | 80 |

(Peak GPU MiB is the whole device incl. any competitor; per-world cost is about 0.8 MiB t1/h1 and 1.6 MiB g1, see `gpu_mib_per_world` in `summary.json`.)

Findings:
- Rollout (physics) is about 85-90% of an iteration; the update is small. Growing N only helps if the GPU has idle capacity.
- With another job on the GPU (the campaign's normal state: 2 slots) samples/s is flat from 4096 to 16384 (g1 15.0 / 15.7 / 16.0k; h1 32.0k at 8192 and 16384). More worlds only add memory.
- Thermals did not bite: max 80 C, no SM clock drop. Host RSS 2.5-3.1 GB per trainer.
- Decision: no num_envs change. `*_steps_ub` stay at 4096 worlds (4 minibatches), `shared_morph_ub` at 10240 (16 minibatches). Raising N is at best +10-20% in the contended regime, doubles GPU memory, changes the sample budget per iteration, and the peer just had a memory-PSI shed.
- Resource finding: g1 at 4096 peaks at 7303 MiB, so the old `gpu_mem: 8G` of `recipes/humanoid/trackers_wholebody_ub.yaml` was below D-117 (>= 1.35 x = 9.6 GiB). Now 10G; guarded by `test_steps_ub_declared_gpu_memory_covers_the_measured_peak`. Resources are outside the DAG `config_hash`, so already-placed nodes are not invalidated.

Caveats / not resolved:
- Only the t1 and h1 4096 points were mostly solo (2.3-2.5x faster than the contended ones; the GPU is already ~94-96% busy with one job at 4096). There is no solo 8192/16384 point, so the exact solo knee is unmeasured: one lease at a time, and the competitor reappeared. Resume: one lease, `peer_run.sh` with a waiter that starts only when `nvidia-smi` shows no other process, t1 and h1 at 4096/8192/16384.
- T3 shared morph (8 groups, 10240 worlds) was not benchmarked; extrapolating 0.8-1.8 MiB/world gives 8-18 GiB, which the declared 24G covers, but this is not verified.

## enable: equivalence protocol and enable rules (unit enable, 2026-10-02; PRE-REGISTERED before any paired measurement; not to be edited after measuring)

Scope: switch the `compute` block ON only for UNSTARTED campaign nodes, one recorded decision per family. Never touched: armdiv T6 (recipes/armdiv/*,
presets/eval-armdiv_v1.json), any node already placed/running/completed in a ledger, T7 phase 2, T8/T9 nodes already running, tracker PPO.

### paired equivalence check (`rrp train bench-compute --equiv`)
Per trainer case (real trainer, real store data, same seed, same data order): N = 300 optimizer steps under `default` (the reference; also run twice =
determinism floor, and once with seed+1 = seed-noise reference) and under the candidate setting (`bf16`; later `bf16+compile`). The loss of a step is the
sum of the scalars passed to `Tensor.backward` since the previous optimizer step.
1. Training-loss curve: block means over 25 steps; per block D = |L_cand - L_ref| / scale, scale = mean |L_ref block mean|. PASS iff all values are finite
   AND median(D) <= 0.05 AND max(D over the last 4 blocks) <= 0.10. The seed-noise D and the determinism D are reported next to it.
2. Dev metric (cheap, where the trainer finishes on its own and reports held-out `eval` numbers): every numeric leaf under the result's `eval`/`dev`
   that both runs report passes iff |cand - ref| <= max(0.25 |ref|, 0.01). Short-training dev numbers are noisy, so this is a sanity gate that the
   candidate has not broken a metric (it is not a quality claim); the seed-noise difference is reported alongside.
A trainer whose check cannot run (data missing, trainer error) is `not_checked`, never `passed`.

### enable rules
E1. A family is switched only if its equivalence check PASSED on its trainers. CPU-device passes (bf16 autocast on CPU) validate numerics only; GPU
    passes are required for `compile` / `cuda_graphs` (never enabled on CPU evidence) and for `eval_backend: warp` (Warp parity gate on CUDA).
E2. Comparability: never switch a subset of the arms of a contrast whose other arms ran (or are placed) under another setting. A contrast is switched
    only if EVERY node of every arm is unstarted; otherwise it keeps the default and the reason is recorded. Recipes that change get a D-147 addendum
    row listing exactly which runs use which compute settings.
E3. Nodes already in a ledger (any state but `planned`) are never changed (run-dag refuses a config-hash change for them anyway). Switching is by a
    per-node/base `compute:` overlay in the recipe of the unstarted family; goldens for the changed recipes are re-recorded.
E4. Everything stays one explicit, recorded option: the block hashes into the node's config, stamped in `compute.json`, pinned in `stage_versions`.
E5. `precision: bf16` carries a tf32 flag (true) for the fp32 islands' matmuls on CUDA; the fp32 islands themselves stay fp32 (unit prec).
