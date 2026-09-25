# VLM system-II track (P16 backbone part: R04/R18/R19/R34): status and resume

Updated 2026-09-21, after the peer reboot. **State: implementing; paused on lead instruction.**
Nothing may be relaunched on the peer until the lead confirms the workspace has been rebuilt.
VLM-backed policy training waits for the controller-facing latent path (branch `correction/controller-facing-latent`).

## Verified before the reboot (commands recorded; the raw logs were on peer tmpfs and are lost)

**Peer venv packages.** Installed with uv under lease `vlm_pip`: transformers 4.57.1 (the psi0 pin), accelerate 1.15.0, safetensors 0.8.0, huggingface_hub 0.36.2, tokenizers 0.22.2.

**Downloads** (lease `vlm_download`; HF_HOME=/dev/shm/rrp-brandonin/cache/hf; 5.0 GB in total):
- psi0 System-II, `USC-PSI-Lab/psi-model@4c6f977…`, subfolder `psi0/pre.fast.2605160748.ckpt.ego390k`.
  - `model.safetensors` sha256 `b2ab7b35…06f0` matches the HF LFS oid.
- `Qwen/Qwen2.5-0.5B-Instruct@7ae5576…`, used as the frozen QA decoder.
- Sizes and licenses are in `research/sources.vlm.json`.
- The psi0 weights have no declared license: local research use only, never redistribute.

**Backbone actually loaded: psi0 System-II, not the fallback.**
- Loaded with `Qwen3VLForConditionalGeneration.from_pretrained(dir, bf16, sdpa)`, which is the psi0 training-path loader. The processor and tokenizer come from the same checkpoint dir.
- Hidden width 2048, 28 layers.
- Embedding rows 153,792 (FAST-extended). The tokenizer length is 151,669; the extra rows are the FAST/pad rows, which are never used by our prompts.

**Test `tests/gpu/test_vlm_backbone.py` PASSED on the peer GPU.** It covers:
- real MuJoCo EGL renders of the `front` camera at 256x256, with task text built from public descriptors;
- the actual adapter, giving tokens [B, 64 visual + 8 text, 2 taps (16, 28), 2048] in fp16;
- different scenes giving different features, and a blank image giving different features;
- the Resampler producing [B, 16, 256].

**Object-QA tests:**
- `test_gradients_reach_projector_readout_policy_not_decoder` PASSED. Gradient reaches the projector, the readout and the action-expert blocks; there are 0 decoder grads.
- The control test ran with a tiny random frozen decoder. Real accuracy was 0.81, blank 0.16, shuffled 0.13 (chance is 0.2).
- It failed only its original real>0.9 threshold. The threshold was changed to "real>0.6 and real−max(controls)>0.3", and the new version has not been re-run yet.

**Deterministic re-simulation for pixels.**
- Replay of `pick_place_parm5_pg2_s0` from the v3 dataset (`scripts/vlm_render_probe.py`) gave a max q0 deviation of **0.0** against the stored episode.
- It took 1.74 s per episode for 24 keyframes (every 4 steps).

**Feature cache.**
- The first attempt was stopped by the peer memory watchdog (`sustained_project_memory_psi`).
- The second attempt (lease 1790115631_2fe9de) reached at least 41 of 312 episodes (1023 frames). VLM time was about 2.5 s per episode after warm-up.
- That cache was lost in the reboot.
- Expected full size: 312 episodes × 24 frames × 72 × 2 × 2048 × 2 B ≈ 4.6 GB. Spec key `477a8493f2a47702`.

## Not done yet
- Full feature cache.
- VLM policy training and the matched no-image baseline (`configs/vlm/train_*.json`).
- Online closed-loop evaluation with real, shuffled and blank images, plus latency and memory measurements.
- QA training and evaluation (`configs/vlm/qa_v1.json`).
- No results exist for any of these; none are claimed.

## Code (committed)
- `src/rrp/model/backbone.py`: VLMBackbone, Resampler, Renderer, task_text.
- `src/rrp/data/vlm_features.py`: replay_render, the cache builder, FeatureStore.
- `src/rrp/model/qa.py`: ObjectQA, SlotReadout, question generation, policy_hidden.
- `src/rrp/learning/qa_train.py`
- `src/rrp/learning/vlm_train.py`: VLMFlowPolicy wrapper and the online-render evaluator. This must be ported to the latent packet z[b,knots,assemblies,d] before use.

## Resume (only after the lead's go-ahead)
```bash
scripts/peer_sync.sh push
L="ssh gb10-direct 'cd /dev/shm/rrp-brandonin/repo && export PATH=/dev/shm/rrp-brandonin/bin:\$PATH PYTHONPATH=src RRP_NODE=peer RRP_REPO=\$PWD && python3 -m rrp.cli ops run"
# 1. packages (CPU lease)
#    ... ops run --cpu 2 --mem 4G --label vlm_pip --env UV_CACHE_DIR=/dev/shm/rrp-brandonin/cache/uv -- \
#        /dev/shm/rrp-brandonin/bin/uv pip install --python /dev/shm/rrp-brandonin/venv/bin/python \
#        "transformers==4.57.1" accelerate safetensors "huggingface_hub>=0.34,<1.0"
# 2. weights (5.0 GB tmpfs, counts against peer memory)
#    ... ops run --cpu 2 --mem 8G --label vlm_download --env HF_HOME=/dev/shm/rrp-brandonin/cache/hf -- venv/bin/python -c \
#      'from huggingface_hub import snapshot_download as s; s("USC-PSI-Lab/psi-model", revision="4c6f9776fc5b18d87945254175e38bb74b9d7748", allow_patterns=["psi0/pre.fast.2605160748.ckpt.ego390k/*"]); s("Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775")'
# 3. tests: ... ops run --gpu --gpu-mem 10G --cpu 2 --mem 8G --label vlm_tests --env HF_HOME=... --env HF_HUB_OFFLINE=1 --env MUJOCO_GL=egl -- \
#      venv/bin/python -m pytest -q tests/gpu/test_vlm_backbone.py tests/unit/test_object_qa.py
# 4. cache (resumable: skips finished episodes): ... ops run --gpu --gpu-mem 10G --cpu 2 --mem 12G --label vlm_cache --max-seconds 7200 --detach \
#      --env HF_HOME=... --env HF_HUB_OFFLINE=1 --env MUJOCO_GL=egl -- venv/bin/python -m rrp.data.vlm_features --config configs/vlm/cache_pick_place_v1.json
# 5. after the latent-path migration: port vlm_train.py (image_tokens -> new policy), then train/evaluate, then qa_train.
```
Consider writing the 4.6 GB cache somewhere other than tmpfs, or reducing `per_robot`, given the new whole-project memory ceiling.
