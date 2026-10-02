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
