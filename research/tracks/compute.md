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
