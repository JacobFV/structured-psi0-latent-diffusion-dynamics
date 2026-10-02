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
