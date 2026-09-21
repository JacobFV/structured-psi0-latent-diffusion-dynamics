# Primary-source audit (psi0, Z-1, EXPO-FT, menagerie, topoformer)

- Status: **verified** for the facts marked as read from primary sources below. Items marked **UNVERIFIED** were not confirmed.
- Retrieved: 2026-09-21 to 2026-09-21 (UTC).
- Pinned revisions: `research/sources.lock.json`.
- Method: read-only. We used `git ls-remote`, the GitHub REST API (read), raw.githubusercontent, arXiv abs/HTML pages, and the Hugging Face API JSON.
  - For the Psi0 checkpoints we read only the safetensors headers, via HTTP range requests on their first bytes.
  - We cloned no repositories, downloaded no weights, and modified no reference repos.
- **Differences from the handoff seed:**
  - topoformer `main` has moved from `e80ddacb…` to `1197afda5220ade22c6cc96824e2ad41b0811aed`. That commit is also the head of `campaign/extended-03`. The handoff's extended-03 SHA `a19ff5d8…` is an ancestor of it.
  - The Psi0 SONIC release-note blob matches (`988f77230eab5d8d7c48aeaed3d0b55914e8ec68`).
  - The extended-02 final-report blob matches (`4d92b5c39df46e3395d9533ebeb5f05fb9935379`).

Throughout, "Psi0 repo" means https://github.com/physical-superintelligence-lab/Psi0 at `4f3720d45e102b36d7c3e9465ab8062274170518`. Paths below are relative to that commit.

---

## a. psi0 System-II backbone

Sources:
- the paper, arXiv 2603.12263 (v1 is the only version): https://arxiv.org/html/2603.12263v1 §III-A;
- the Psi0 repo;
- the HF model repo `USC-PSI-Lab/psi-model` at HF sha `4c6f9776fc5b18d87945254175e38bb74b9d7748`.

### Model

- **Family:** Qwen3-VL. The base is **`Qwen/Qwen3-VL-2B-Instruct`**.
  - Declared at `src/psi/models/psi0.py:45` (`QWEN3VL_VARIANT`), `src/psi/config/model_psi0.py:117` and README.md:29.
  - The HF architecture is `Qwen3VLForConditionalGeneration` (`model_type qwen3_vl`).
  - The repo pins `transformers==4.57.1` (`pyproject.toml:69`).
- **Parameters:**
  - Psi0 System-II has **2,131,333,120** (625 tensors, from the safetensors header).
  - The base model has 2,127,532,032.
  - The difference of +3,801,088 comes from growing the vocabulary from 151,936 to **153,792**. That adds 2,048 FAST action-token bins (`src/psi/config/tokenizer.py:78`), padded to a multiple of 192 (`scripts/save_pretrain_qwen3vl_backbone.py:43-48`).
  - Split:

    | Component | Parameters |
    |---|---|
    | LM layers | 1.409B |
    | Tied embed_tokens | 315M (there is no separate lm_head) |
    | Vision blocks | 302M |
    | Deepstack mergers | 75.5M |
    | Merger | 25.2M |

- **Text tower:** hidden width **2048**, 28 layers, 16 query heads and 8 KV heads with head_dim 128, FFN 6144, tied embeddings, mRoPE sections [24, 20, 20].
- **Vision tower:** depth 24, width 1024, patch 16, spatial merge 2, temporal patch 2, output width 2048, deepstack layers [5, 11, 17].
- **Tokenizer and processor:** `Qwen2Tokenizer`, extended with the FAST tokens, and `Qwen3VLProcessor` with `Qwen2VLImageProcessorFast` (mean and std 0.5, patch 16, merge 2).
  - At inference, `Psi0Model.from_pretrained` loads the processor from the **base** HF id, not from the checkpoint (`psi0.py:1797`).
  - Prompts contain no action tokens, so this should be equivalent. **UNVERIFIED** token by token.

### Checkpoints, licenses, gating

**Licensing and gating**
- HF repo `USC-PSI-Lab/psi-model`:
  - `gated: false`, `private: false`.
  - **No license is declared.** There is no model card and no LICENSE file on HF.
- GitHub repo: the root `LICENSE` is Apache-2.0 ("Copyright 2026 PSI-lab, USC"). README.md:757-761 also says Apache-2.0.
- Whether that license covers the weights is **UNVERIFIED**. Treat the weights as "Apache-2.0 by repo statement, not declared on the HF repo".
- Base `Qwen/Qwen3-VL-2B-Instruct` (HF sha `89644892e4d8…`): Apache-2.0, not gated, one BF16 `model.safetensors` of 4,255,140,312 bytes.

**Checkpoints.** Each is a single unsharded `model.safetensors` with no index json.

| HF path | Role | Weights | Bytes |
|---|---|---|---|
| `psi0/pre.fast.2605160748.ckpt.ego390k` | Pre-trained VLM (EgoDex 390k). This is the VLM starting point for the SONIC post-train. | BF16, standard HF dir | 4,262,742,488 |
| `psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k` | Baseline VLM behind the 36-dim real checkpoints | BF16 | 4,262,742,488 |
| `psi0/pre.fast.egodex.2512241941.ckpt200k`, `pre.abl.*` | Ablation VLMs | BF16 | 4.26 GB each |
| `psi0/postpre.sonic1.0.unifolm.2609092156.40k` | SONIC post-train: VLM plus action expert | F32 VLM `model.safetensors` plus F32 `action_header.safetensors` (673.8M params) | 8,525,408,184 + 2,695,062,456 |
| `psi0/sonic-checkpoints/multi-task.psi-dream.2609092156` | SONIC fine-tune; the VLM was also tuned | Single F32 file with prefixes `vlm_model.*` and `action_header.*` | 11,220,488,216 |
| `psi0/postpre.1by1.pad36.2601131206.ckpt.he30k` | 36-dim action expert (HE post-train) | F32 `action_header` (497.7M params) | 1,990,811,480 |
| `psi0/real-checkpoints/task1..8` | 36-dim fine-tunes | VLM in BF16 and action expert in F32 | 6,253,648,840 each |
| `psi0/simple-checkpoints/*` (22 dirs) | Simulation fine-tunes | not inspected | about 6.25 GB each |

### What System-I consumes

**Common path for all checkpoints**
- The VLM runs with `output_hidden_states=True` (`psi0.py:1855-1862` in training, `:1970-1978` in `predict_action`). The output is a 29-tuple: the embedding output plus 28 layer outputs.
- **Every token of the sequence is passed on**: template tokens, image tokens, instruction text and the generation prompt. There is no pooling. Padding is masked through `vlm_attn_mask` (`psi0.py:1886, 1995-1997`).
- The prompt is the user content `[images…, instruction, (goal images)]` with `apply_chat_template(add_generation_prompt=True)` and `qwen-vl-utils==0.0.14` `process_vision_info(image_patch_size=16)` (`psi0.py:1930-1950`).
- A projection `obs_proj.views_proj: Linear(2048→1536)` feeds the action expert (`psi0.py:533`).

**Which hidden states, by checkpoint**
- **36-dim / older checkpoints:** only `hidden_states[-1]`, the last layer (`psi0.py:1835-1836, 1982-1983`).
- **SONIC checkpoints:** layerwise fusion with `vlm_layer_indices=[3,5,8,10,12,14,17,19,21,23,26,28]`, one VLM layer per action block (`psi0.py:1826-1837`; checkpoint `run_config.json`; `examples/psi0_for_sonic.md:32-35`).

**Where the state enters**
- Old checkpoints: as one extra context token (`psi0.py:740-755`).
- SONIC: as token 0 of the action stream, with a learned `state_null` for dropout (`psi0.py:1300-1323`).

### System-I (action expert)

| | 36-dim checkpoints | SONIC checkpoints |
|---|---|---|
| Class | `ActionTransformerModel` (`psi0.py:1090`) with `VLATransformerBlock` (`:896`). SD3-style MM-DiT with joint attention (`JointVLAAttnProcessor`, `:339`). | Same classes |
| Blocks | 6 | 12, all context_pre_only, with RMS qk-norm |
| Width / heads / FF | 1536 / 24×64 / 6144 | same |
| Parameters | 497.7M (F32) | 673.8M (F32) |
| Action dim × chunk | 36 × 30. The HE post-train expert uses Tp = 16. | 80 × 30 |
| State dim | 36 (32 real + 4 padding) | 45 |
| Time/text conditioning | `_TimeNetwork` | `CombinedTimestepTextProjEmbeddingsND` with frozen CLIP ViT-L/14 pooled text (768-d, shipped as `clip_pooled_cache.pt`) |
| Objective | Flow matching with `FlowMatchEulerDiscreteScheduler`, 1000 train timesteps (`psi0.py:1799-1818`) | same |
| Denoising steps | Config default 10 (`model_psi0.py:31`); the deploy server hard-codes **8** (`serve_psi0_amo.py:151,164`) | 8 (`serve_psi0_sonic.py:298,327`) |
| Control rate | 30 Hz (`serve_psi0_amo.py:47`); data at 30 FPS | 30 Hz; `PREDICT_HORIZON=30`, `MIN_EXEC_HORIZON=15` (`serve_psi0_sonic.py:64-69`); a chunk is 1.0 s |
| Real-time chunking (RTC) | Training-time (`max_delay=8`) | Test-time soft guidance (`guidance_alpha=0.9`) |

The paper describes the action expert as "~500M". That matches the 6-block expert. The SONIC expert is larger, at 674M.

### Loading System-II in PyTorch (repo code)

**Inference path:** `Psi0Model.from_pretrained(run_dir, ckpt_step, launch_config, device)` at `src/psi/models/psi0.py:1733-1824`.
1. `load_file(f"{run_dir}/checkpoints/ckpt_{step}/model.safetensors")`
2. Build `Qwen3VLForConditionalGeneration(AutoConfig.from_pretrained("Qwen/Qwen3-VL-2B-Instruct"))` in bf16, using `sdpa` if flash-attn is missing.
3. Strip the `vlm_model.` prefix.
4. Tie `lm_head.weight = model.language_model.embed_tokens.weight`.
5. `resize_token_embeddings(153792, pad_to_multiple_of=192, mean_resizing=True)`
6. `load_state_dict(strict=True)`
7. Load `action_header.*` into `ActionTransformerModel`.

**Training path:** `Qwen3VLForConditionalGeneration.from_pretrained(model_name_or_path, attn_implementation="flash_attention_2", dtype=…)` (`src/psi/trainers/finetune.py:273-283`; `posttrain.py:168-177`). The expert is loaded with `load_file(f"{ckpt_path}/action_header.safetensors")` (`finetune.py:333-337`).
- This path works directly on the HF-format dirs `psi0/pre.*` and `psi0/postpre.sonic1.0…`.

### Recommendation for aarch64 GB10, ~16 GB GPU budget

**Base setup (all options):** load the VLM only, in bf16, with `attn_implementation="sdpa"`.
- About 4.26 GB resident.
- Activations are small: a 270×480 frame gives about 130 visual tokens.
- flash-attn on aarch64 is **UNVERIFIED**, so do not depend on it.

Choose the checkpoint by the action expert you need features to match:

1. **Faithful to the released SONIC expert: `psi0/sonic-checkpoints/multi-task.psi-dream.2609092156`.**
   - Its VLM was fine-tuned (`tune_vlm: true`).
   - Do **not** `load_file` the whole 11.2 GB F32 file on unified memory.
   - Instead, stream it with `safetensors.safe_open(framework="pt")`:
     - read only the `vlm_model.*` keys and cast each to bf16;
     - replicate `psi0.py:1744-1776` (config from Qwen3-VL-2B, resize to 153,792, tie, `strict=True`);
     - take `hidden_states[i]` for i in [3,5,8,10,12,14,17,19,21,23,26,28].
   - The full 11.2 GB still has to be downloaded. Fetching only the VLM byte range by HTTP is possible but untested.
2. **Simplest HF-native option: `psi0/postpre.sonic1.0.unifolm.2609092156.40k`.**
   - Download only `model.safetensors` (8.53 GB F32) plus the small config and tokenizer files. Skip `action_header.safetensors`.
   - Then `from_pretrained(dir, dtype=torch.bfloat16, attn_implementation="sdpa")`.
   - These features match the post-train expert.
3. **Smallest real Psi0 System-II: `psi0/pre.fast.2605160748.ckpt.ego390k`** (4.26 GB BF16, HF-native).
   - This is before robot post-training. Its features do not match either SONIC expert, so label it "psi0 pre-trained VLM".
4. **For the 36-dim AMO experts: `psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k`** (4.26 GB).
   - The real-task runs used `--model.no-tune-vlm` from this directory.
   - Bitwise equality with the VLM inside `real-checkpoints/*` is **UNVERIFIED**.

**Labeled FALLBACK only (not psi0).** Licenses and sizes from the HF API:

| Model | License | Gated | Params / dtype | Size |
|---|---|---|---|---|
| `Qwen/Qwen3-VL-2B-Instruct` (the unadapted psi0 base) | Apache-2.0 | no | 2.128B BF16 | 4.26 GB |
| `Qwen/Qwen2-VL-2B-Instruct` | Apache-2.0 | no | 2.209B BF16 | 4.42 GB |
| `google/siglip-so400m-patch14-384` (image only) | Apache-2.0 | no | 0.878B F32 | 3.51 GB |
| `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | Apache-2.0 | no | 2.247B F32 | 8.99 GB |

Avoid these:
- `Qwen/Qwen2.5-VL-3B-Instruct`: `qwen-research` license.
- `google/paligemma2-3b-pt-224`: `gemma` license, manually gated.

---

## b. psi0 action contracts

`examples/psi0_for_sonic.md` has blob `988f77230eab5d8d7c48aeaed3d0b55914e8ec68` (4,845 bytes), which **matches the handoff**.
- Line 11: "64-dim body-arm token + 14 hand joints + 2 dof neck".
- The note gives no index ranges. The ranges below come from the source files cited.

### Original 36-dim action (AMO "prior command")

- Producer: `scripts/data/raw_to_lerobot.py:215-258`.
- Robot consumer: `real/deploy/psi-inference_rtc.py:291-315`.
- The paper §III confirms the set {q_hand, q_arm, torso_rpy, h_b, v_x, v_y, v_yaw, p_yaw}.

| Index | Group | Dims | Units / notes |
|---|---|---|---|
| 0:14 | q_hand (left 7, right 7) | 14 | Joint angles, radians assumed. H1 Inspire hands have 6 DOF per side and are padded elsewhere (`merge_sonic_v1.py:131-134`). |
| 14:28 | q_arm (`sol_q[15:29]`, left 7, right 7) | 14 | IK joint targets, radians assumed |
| 28:31 | torso roll, pitch, yaw | 3 | Radians assumed |
| 31 | base height h_b | 1 | Metres. Default 0.75; task1 stats 0.69–0.84. |
| 32 | v_x | 1 | Client snaps to {0, 0.6}. Units **UNVERIFIED**. |
| 33 | v_y | 1 | Client snaps to {0, ±0.5}. Units **UNVERIFIED**. |
| 34 | v_yaw | 1 | **UNVERIFIED** units |
| 35 | target yaw p_yaw | 1 | `torso_dyaw` is excluded (`raw_to_lerobot.py:258`) |

- **Rates:** 30 FPS data, 30 Hz control, chunk 30.
- **AMO:** maps [28:36] to 15 lower-body joints, for 43 DoF in total with the 28 upper joints.
- **State (36-dim checkpoints):** 32 real dims = hands 14, arms 14, torso_rpy 3, torso_height 1 (`raw_to_lerobot.py:203-212`), zero-padded to 36.
- **Normalization:** per-dimension min/max mapped to [-1, 1] ("bounds"). The bounds are embedded in each checkpoint's `run_config.json`.

### SONIC 80-dim action (released `multi-task.psi-dream` / `postpre.sonic1.0`)

Model output order:

| Index | Group | Dims | Notes |
|---|---|---|---|
| 0:64 | SONIC body/motion latent token | 64 | Unitless. Values lie on a 1/16 grid. The client quantizes to FSQ in [-0.625, 0.625] with step 0.0625 (`psi_rtc_sonic_client.py:27-38`, in submodule `GR00T-WholeBodyControl@d40376994ff094787591ab76a691c50b8e131786`). The dataset min/max reach -0.875, outside that clip; the mismatch is **unresolved**. |
| 64:78 | Hands: left thumb0/1/2, middle0/1, index0/1, then the same for the right hand | 14 | Dex3 joint angles (`merge_posttrain_sonic.py:73-86`) |
| 78:80 | Neck yaw, pitch | 2 | Units **UNVERIFIED** (range about ±1.2–1.7) |

**Sources for the layout**
- Label split `[64, 78]`: `transform_psi0_sonic.py:268-283` and `examples/psi0/openloop_g1_sonic_neck.py:120-126`.
- Raw dataset order: hands `[0:14]`, neck `[14:16]`, token `[16:80]`. It is repacked with `action[16:80] action[:14] action[14:16]` (`scripts/train/psi0/finetune-sonic-psi-dream-baseline.sh:232,241`).
  - This is inferred from the repack keys and `dataset_statistics.json`. It is not stated verbatim anywhere.

**Variants**
- Post-train (`postpre.sonic1.0`): the action is `body_token + hands` (78 dims), zero-padded to 80. The neck slots have degenerate statistics (`posttrain-psix-unifolm-g1-sonic1.0.sh:151-154,192-202`).
- The neckless runtime client publishes `token=a[:64]`, `left=a[64:71]`, `right=a[71:78]` (`psi_rtc_sonic_client.py:183-195`). The Dex1 bridge asserts a 78-dim action (`real/SONIC/psi0_vla_dex1_bridge.py:80-89`).
  - No 80-dim runtime client was found, so how the neck outputs reach the robot is **UNVERIFIED**.

**Normalization and rates**
- `SonicActionStateTransform` maps bounds to [-1, 1] with clipping; near-constant dimensions pass through (`transform_psi0_sonic.py:197-218, 291-326`).
- During training only: state noise with σ=0.05 and temporal jitter of ±10 frames.
- 30 Hz control, Tp = 30 (1 s), RTC minimum execution horizon 15, 8 flow steps.

### SONIC 45-dim state

| Index | Group | Dims |
|---|---|---|
| 0:12 | Legs: left hip pitch/roll/yaw, knee, ankle pitch/roll; then the right leg | 12 |
| 12:15 | Waist yaw, roll, pitch | 3 |
| 15:29 | Arms: left shoulder p/r/y, elbow, wrist r/p/y; then the right arm | 14 |
| 29:43 | Hands (same order as the action) | 14 |
| 43:45 | Neck yaw, pitch | 2 |

**Evidence**
- The preflight asserts slot 0 = `left_hip_pitch_joint`, slot 29 = `left_hand_thumb_0_joint`, and slots 39 and 41 are right middle0 and right index0 (`posttrain-psix-unifolm-g1-sonic1.0.sh:126-141`).
- The shipped statistics agree: knees at 3 and 9.
- The client builds a 43-dim state as `body_q(29) + left_hand(7) + right_hand(7)` (`psi_rtc_sonic_client.py:286-295`).

**Caveats**
- `scripts/data/reorder_state_columns.py` describes a different order (hand|arm|leg|waist|neck) for psix_sonic_v1. The shipped statistics contradict it for the released checkpoints.
- The within-block joint order is inferred from the G1 29-DoF convention.
- The neck state ranges ([-2.91, 1.03] and [-5.02, 0.72]) look suspicious; their units and wrapping are **UNVERIFIED**.
- Post-training used 43-dim state, zero-padded to 45.

**Adapter implication:** the 36-dim and 80-dim contracts are incompatible.
- The 36-dim action is explicit joint, torso, velocity and height commands for AMO.
- The 80-dim action is an opaque 64-d SONIC latent token plus hand and neck joints, which needs the SONIC whole-body controller.
- The state vectors differ in both dimension and order (32/36 vs 45).

---

## c. Z-1 flow-SDE GRPO

Sources:
- Z-1: arXiv 2606.31846, **v1 only** (https://arxiv.org/html/2606.31846v1), §3.1.2, §3.2–3.4 and App. C.3, C.4, D.1–D.4.
- πRL: arXiv 2510.25889v3, §3.2 and §4.2. Z-1 cites it as its flow-SDE source.
- Flow-GRPO: arXiv 2505.05470v5, §3, Eq. 1, 4, 5, 7–9 and App. A, B.3.
- RLinf code: https://github.com/RLinf/RLinf. Z-1 §4.1.2 says "Our GRPO implementation is based on the RLinf framework". The file was read at an earlier main commit (prefix `184ba07f0a`); main is now pinned at `807e5fdd…`.

### What Z-1 itself specifies

**Z-1 does not write out an SDE, a σ schedule, a noise level, the number of steps K, or which steps are stochastic.** It defers these: "we follow the flow-SDE formulation of πRL (Chen et al., 2026)" (§3.1.2 and D.3).

- **Transition (App. D.3):** $p_\theta(z_{k+1}\mid z_k,x)=\mathcal N(z_{k+1};\,m_\theta(z_k,x,t_k),\,\sigma_k^2 I)$.
  - The chain runs $z_0$ (noise) → $z_K$ (the action chunk).
  - The paper gives no formula for $m_\theta$ or $\sigma_k$.
- **Log-likelihood:** $\log\pi_\theta(a\mid x)=\sum_{k=0}^{K-1}\log p_\theta(z_{k+1}\mid z_k,x)+C$.
  - The ratio is $\rho=\exp(\log\pi_\theta-\log\pi_{\theta_{old}})$, computed per trainable action chunk.
  - The old log-probs, the sampled chains and the "selected denoising indices" are stored (D.4).
  - "Selected denoising indices" suggests RLinf's single-random-SDE-step mode, which conflicts with the D.3 sum over all K steps. **The paper does not resolve this.**
- **GRPO (§3.1.2, Eq. 1–4):**
  - Advantage: $A_i=(R_i-\mu_R)/(\sigma_R+\epsilon)$, where $\sigma_R$ is the population std (1/G). The value of ε is **not given**.
  - Loss: $\mathcal L=-\frac{1}{\sum_i|\mathcal B_i|}\sum_i\sum_{(x,a)\in\mathcal B_i}\min(\rho A_i,\ \mathrm{clip}(\rho,1-\eta,1+\eta)A_i)$, with η = 0.2 (η is the clip parameter here).
  - **No KL term, no reference policy, no critic.**
  - The trajectory advantage $A_i$ is applied to every trainable chunk of rollout i. There is no per-denoising-step credit.
- **Hyperparameters (App. C.3):**
  - G = 8 (the tree variant uses G = 4).
  - AdamW, lr 5e-6, wd 0.01, grad-clip 1.0.
  - Global batch 1024, micro-batch 16, 4 rollout epochs, 64 parallel environments.
  - "Reward filtering" with bounds 0.1 and 0.99. Undefined; presumably a filter on group success rate.
  - γ = 0.998 when reward decay is enabled.
- **Shared prefix (§3.2, D.1):**
  - A leader rollout executes action-chunk window [0, P). The simulator state is then cloned, and G suffixes run independently.
  - **Prefix chunks are masked out of $\mathcal B_i$.**
- **Tree variant (§3.3, citing TreeAdv, arXiv 2601.03703):**
  - $G=2^k$, with branch points $b_\ell=\lfloor P/2^{k-\ell}\rfloor$. Each cluster splits in two at each $b_\ell$.
  - Only the fully shared root is masked. Partially shared segments become trainable after their cluster separates.
  - Followers execute and store the leader's action, log-prob and inputs.
  - The branch-point schedule is fixed. In the ablation, P decays from 15 to 0 chunks.
- **Success-aware reward decay (§3.4):** $\tilde r=r\gamma^{d_i}$, where $d_i$ is the number of chunks until first success.
  - Groups where every rollout succeeds use $A_i=R_i-\min_jR_j$.
- **Base model:** π0.5 (PaliGemma plus a flow expert). By default the VLM is frozen and only the action expert is trained.
- **Not stated in Z-1:** chunk length H, K, noise level a, final-step treatment and time convention.

### πRL Flow-SDE, the formulation Z-1 defers to (arXiv 2510.25889v3 §4.2)

- **Convention:** $A^\tau=\tau A+(1-\tau)\epsilon$, with τ=0 as noise and τ=1 as data, $u=A-\epsilon$, and $\tau_k=k/K$. This is the same as ours.
- **Eq. 8:**
  $$dA^\tau=\Big[v^\tau+\frac{\sigma_\tau^2}{2\tau}\big(A^\tau+(1-\tau)v^\tau\big)\Big]d\tau+\sigma_\tau dw,\qquad \sigma_\tau=a\sqrt{\tfrac{\tau}{1-\tau}}$$
- **Eq. 9:** $\mu=A^\tau+[\cdot]\delta$, $\Sigma=\sigma_\tau^2\delta I$.
- **Inconsistency:** this score expression is Flow-GRPO's, written for the *opposite* convention. Read literally in πRL's own convention with δ>0, it is wrong. It is correct only in the openpi code convention (t=1 is noise), with a signed step Δt=−1/K. The RLinf code does exactly that.
- **Hybrid sampling (§4.2.3):** one randomly chosen denoising step per sample is SDE; all other steps are deterministic ODE steps.
- **Hyperparameters:** noise level a = 0.5, or 0.3 in some settings; 3–5 denoising steps for π0.5; H = 10.

### Flow-GRPO SDE (arXiv 2505.05470v5), the original construction

- **Convention (Eq. 1):** $x_0$ is data, $x_1$ is noise, $x_t=(1-t)x_0+tx_1$, and $v=x_1-x_0$. Sampling runs from t=1 to t=0.
- **Reverse SDE (Eq. 7):** $dx_t=(v_t-\frac{\sigma_t^2}{2}\nabla\log p_t)dt+\sigma_t dw$.
- **Score (App. A, Eq. 27):** $\nabla\log p_t(x)=-\frac{x}{t}-\frac{1-t}{t}v_t(x)$.
- **Eq. 8:**
  $$dx_t=\Big[v_t+\frac{\sigma_t^2}{2t}\big(x_t+(1-t)v_t\big)\Big]dt+\sigma_t dw$$
- **Eq. 9:**
  $$x_{t+\Delta t}=x_t+\Big[v_\theta+\frac{\sigma_t^2}{2t}(x_t+(1-t)v_\theta)\Big]\Delta t+\sigma_t\sqrt{|\Delta t|}\,\epsilon,\qquad \sigma_t=a\sqrt{\tfrac{t}{1-t}},\ \Delta t<0$$
- **GRPO (Eq. 5):**
  - A per-denoising-step ratio, averaged as $\frac1G\sum_i\frac1T\sum_t$.
  - A KL term against a reference policy with closed form $\frac{\Delta t}{2}\big(\frac{\sigma_t(1-t)}{2t}+\frac1{\sigma_t}\big)^2\|v_\theta-v_{ref}\|^2$.
  - Hyperparameters: β = 0.04 or 0.01, G = 24, a = 0.7, T = 10 for training.
- **Code:** clamps the t=1 singularity with `sigma_max=sigmas[1]`, and **averages** the log-prob over dimensions instead of summing it.

### RLinf implementation (openpi code convention, t=1 is noise)

- **Schedule and transition:**
  - `sigma = a*sqrt(t/(1-t))`, with t=1 clamped to `timesteps[1]`.
  - `mean = x0_pred*(1-(t-δ)) + x1_pred*((t-δ) - σ²δ/(2t))` and `std = σ√δ`.
  - This is algebraically equal to Flow-GRPO Eq. 9.
- **Default:** one SDE step index is sampled with `randint(0, K-1)`, shared by the whole batch; the other steps are ODE (std 0), and only that step's log-prob is used.
  - `joint_logprob=True` makes every step SDE, includes the N(0, I) prior log-prob, and averages the log-probs.
  - The last step is stochastic unless `ignore_last` is set.
- **`pi0_5.yaml`:** `noise_level 0.5`, `num_steps 5`, `num_action_chunks 10`.
- **Which settings Z-1 used is UNVERIFIED.**

### Mapping to OUR convention (τ=0 noise, τ=1 data; $z_\tau=(1-\tau)\varepsilon+\tau x$; $v=x-\varepsilon$; $dz/d\tau=v$)

**Change of variables:** $\tau=1-t$, $v=-v^{FG}$, $d\tau=-dt$. Generation then runs forward in τ, so no reverse-time SDE is needed.

**Posterior means and score:**
- $\hat\varepsilon=\mathbb E[\varepsilon\mid z_\tau]=z_\tau-\tau\hat v$
- $\hat x=z_\tau+(1-\tau)\hat v$
- Because $p(z\mid x)=\mathcal N(\tau x,(1-\tau)^2 I)$:
  $$s(z,\tau)=\nabla\log p_\tau(z)=-\frac{\hat\varepsilon}{1-\tau}=-\frac{z-\tau\hat v}{1-\tau}=-\frac{z-\tau\hat x}{(1-\tau)^2}.$$
- This agrees with Flow-GRPO Eq. 27 after the change of variables.

**Marginal-preserving forward SDE.** This follows from Fokker–Planck, adding $+\frac{g^2}{2}s$ to the drift:
$$dz=\Big[\hat v-\frac{g(\tau)^2}{2(1-\tau)}(z-\tau\hat v)\Big]d\tau+g(\tau)\,dW.$$

With the schedule mapped from Flow-GRPO, $g(\tau)=a\sqrt{(1-\tau)/\tau}$, this becomes
$$dz=\Big[(1+\tfrac{a^2}{2})\hat v-\tfrac{a^2}{2\tau}z\Big]d\tau+a\sqrt{\tfrac{1-\tau}{\tau}}\,dW,$$
which is Flow-GRPO Eq. 8 after the change of variables.

**Euler–Maruyama transition** with step $h=\tau_{k+1}-\tau_k>0$ and $g_k=g(\tau_k)$:
$$\mu_k=z_k+h\hat v_k-\frac{g_k^2h}{2(1-\tau_k)}(z_k-\tau_k\hat v_k),\qquad s_k=g_k\sqrt h .$$

Equivalent forms:
- $\mu_k=(\tau_k+h)\hat x+\big(1-\tau_k-h-\frac{a^2h}{2\tau_k}\big)\hat\varepsilon$. This matches the RLinf `x0_weight`/`x1_weight`.
- With the unclamped schedule, $s_k=a\sqrt{h(1-\tau_k)/\tau_k}$.

**Path log-likelihood**, summed over valid coordinates and over the stochastic steps only:
$$\log p(z_{k+1}\mid z_k)=-\tfrac12\sum_{\text{valid}}\Big[\big(\tfrac{z_{k+1}-\mu_k}{s_k}\big)^2+2\log s_k+\log 2\pi\Big].$$
- $\log s_k$ depends only on the schedule, so it cancels in the ratio.
- Upstream code averages instead of summing. That is a dimension-normalized surrogate and must be labeled as such if we use it.

**Endpoints:**
- **τ=0 (the first step):** $g\to\infty$.
  - Either clamp as RLinf and Flow-GRPO do, $g_0=a\sqrt{(1-\tau_0)/\tau_1}=a\sqrt K$ on a uniform grid, and use the **general** $\mu$ form (not the $a^2h/2\tau$ form). This gives $\mu_0=z_0+h\hat v-\frac{a^2}{2}z_0$ and $s_0=a$.
  - Or make step 0 deterministic and exclude it from the likelihood.
- **τ=1:** never evaluated, because Euler–Maruyama stops at $\tau\le1-h$.
  - The last step has $s=a/\sqrt{K(K-1)}>0$, and K=1 is degenerate.
  - Either keep that step stochastic (the RLinf default), or make it deterministic (`ignore_last`) and exclude it. Record which choice is made.
  - A deterministic step must never be given the Gaussian density.

**Sign checks (done by the audit agent):**
- Analytic: for a point mass at x=0, the variance ODE gives $\frac{d}{d\tau}\mathrm{Var}=-2(1-\tau)$, which matches $(1-\tau)^2$.
- Numeric: for $\mathcal N(2,0.5^2)$ with exact $\hat v$, a ∈ {0.5, 0.8} and K=200, the samples have std 0.500–0.501 and mean 2.00. Flipping the correction sign gives std 6.89.

**Carry these into `research/methods/grpo-derivation.md`, with tests.**

---

## d. EXPO-FT

Sources:
- Paper: arXiv 2605.25477 **v2** (17 Aug 2026; v1 25 May 2026), https://arxiv.org/html/2605.25477v2. Authors Dong, Hung, Gao, Sadigh, Finn; CC BY 4.0.
- Code: https://github.com/pd-perry/expo-ft @ `803381fc3b4c91a0c47904f1b688fc5e35904f50` (MIT). It uses the OpenPI fork `pd-perry/openpi@expo_ft` @ `46407a41…` (Apache-2.0).

**Original EXPO:** arXiv 2507.07986 v3, "EXPO: Stable Reinforcement Learning with Expressive Policies", Dong, Li, Sadigh and Finn, ICLR 2026. Code at https://github.com/pd-perry/EXPO @ `bbd57130…` (MIT).
- It uses a DDPM diffusion base policy trained by imitation, a small Gaussian edit policy that maximizes Q, and on-the-fly argmax-Q selection among base and edited samples.
- Its actions are single-step, with UTD 20, batch 256 and β between 0.05 and 0.7.
- It has an optional softmax "entropy backup".

### Ingredients

**Base policy**
- The π0.5 flow VLA. It starts from a task LoRA SFT checkpoint; if zero-shot success is low, it is first trained further on demos until it reaches about 40% (§4.2).
- It is updated online with its **own original flow-matching loss** (§4.2).
- The VLA image encoder is frozen.
- A target copy is kept with τ_π = 1e-3. In the code, candidates are sampled from the live parameters.
- **Found only in code:** `actor_success_only=True`, so the base-VLA imitation batch is drawn only from successful episodes, including demos and human interventions. The paper does not state this.
- Ablation: freezing the VLA does not reach 30/30 within the same budget.

**Edit policy**
- A tanh-squashed Gaussian. Its inputs are the critic's ResNet-50 image features, a 64-d proprio embedding and the base action chunk. The network is an MLP of 3×256.
- The edit is $\hat a=\beta\tanh(\cdot)\in[-\beta,\beta]$, with β = 0.05 or 0.2 per task (Table 4).
- **Found only in code:** `edit_action_xyzg` masks edits to the rotation dimensions.
- The edit acts only on the executed C-step chunk.

**Critic**
- An ensemble of 10 Q-networks with LayerNorm (REDQ-style).
- Critic encoder: its own trainable ResNet-50, with a 512-d image embedding and a 64-d proprio embedding. It is not shared with the VLA.
- TD target: the min over 2 randomly chosen target members, with soft update τ_Q = 5e-3.
- No entropy term in the target. Not distributional.
- The edit-policy loss uses the **mean** over all 10 online critics (code).

**Action selection (the OTF policy)**
- Sample N = 8 base chunks $a_i\sim\pi_{VLA}$, edit each, and pick deterministically:
  $$\tilde a^*=\arg\max_{a\in\cup_i\{a_i,\ a_i+\hat a_i\}}Q(s,a)$$
  over the 16 candidates.
- The same rule gives the next action in the TD backup.
- In code, candidates are scored with the target critic, taking the min over 2 subsampled members.

**Losses**
- Edit policy (Eq. 4):
  $$\mathcal L(\pi_{edit})=-\mathbb E\big[Q(s,a+\hat a)-\alpha\log\pi_{edit}(\hat a\mid s,a)\big]$$
- Critic (Eq. 5):
  $$\big(y-Q(s_t,a_{t:t+C})\big)^2,\qquad y=\sum_{i<C}\gamma^ir_{t+i}+\gamma^C m\min_{k\in K,|K|=2}Q'_k(s_{t+C},\tilde a^*)$$
  - Windows can start at any step. Windows that touch a timeout are marked invalid.
- Temperature: SAC-style learnable α with α₀ = 1. Target entropy is $-C\,d_a/2$, and the log-prob gets a $-d\log\beta$ correction (code).

**Other components**
- **Human-in-the-loop (new in FT):** SpaceMouse corrections overwrite part of a chunk, which is stored as executed. The intervention rate falls to 0 during training, and the method still works without interventions, only more slowly.
- **Replay:** demos initialize the single replay buffer. The default is `offline_ratio=0`, so there is **no** RLPD 50/50 sampling by default.
- **Schedule:**
  - UTD = 20, batch 64. In code, one update call runs 20 critic minibatches of 64, then one edit/α update and one base-VLA update.
  - The paper reports 1–6 updates per episode, about 1 per 40 environment steps.
  - This reading of "UTD 20" is the audit agent's reading of the code.
- **Other hyperparameters:**
  - Adam 3e-4 for the critic, edit policy and α; γ = 0.99.
  - Replan chunk C = 8, or 4 for Insert and Flower.
  - 8k–20k environment steps; 10–40 demos per task.
  - Observations: wrist and side views at 224², plus end-effector pose.
  - Actions: Cartesian and gripper velocities at 10 Hz. Rewards: sparse binary.
  - Compute: 2×H200.
  - The VLA learning rate is **UNVERIFIED**; it lives in the OpenPI fork config.

**What FT adds relative to EXPO:**
1. A VLA base policy (π0.5) with a frozen image encoder.
2. Chunk-level edits, critic and backups.
3. Human-in-the-loop corrections.
4. A separate ResNet critic encoder that the edit policy reuses.
5. An actor/learner server with configurable update timing.
6. Batch size 64.
7. Success-filtered base updates (in code).
8. Deterministic argmax selection, with no entropy backup.

**Portable to our interface:** the OTF argmax over base and edited chunks, a bounded tanh edit in our latent or normalized-action space, a chunked TD critic ensemble, and base updates by imitation.

**Label for our comparator:** "EXPO-FT-inspired" unless we use π0.5, human-in-the-loop corrections and the same hyperparameters.

### Related: unified latent space (arXiv 2601.15419 v1, Yan & Lee, CC BY 4.0)

**Method**
- **Stage 1: a latent space.**
  - Contrastive/triplet learning (ImitationNet lineage) of per-body-part 16-d latents for the left arm, right arm, trunk, left leg and right leg.
  - Human motion data is HumanML3D (SMPL, which has no hands).
  - Robot data is random joint samples pushed through forward kinematics from URDFs; no robot demonstrations are collected.
  - The encoder and decoder are shared, plus a per-robot embedding.
- **Stage 2: a goal-conditioned c-VAE.** It is trained on human motion only and predicts latent displacements.
- **Adding a new robot:** train only its embedding layers.

**Scope limits**
- Kinematic retargeting and end-effector reaching only. There is no dynamics, contact or torque modelling.
- Robots: ATLAS, H1, G1, JVRC, NAO, TIAGo++ and Kinova.
- **No code found (absence UNVERIFIED).**
- It motivates a shared latent, but it does not establish control semantics.

---

## e. Menagerie asset audit

**Repo:** https://github.com/google-deepmind/mujoco_menagerie @ `c96a32d28fb5da84da38c1da4d749e7a13212855`.

**Method**
- The tree came from the GitHub API (`git/trees/<sha>?recursive=1`, not truncated).
- Each `<dir>/LICENSE` was read from raw.githubusercontent at the pinned SHA and classified by its text.
- Actuator counts come from parsing the named MJCF's top-level `<actuator>` element. The count covers that file only, so included files are counted separately.
- Sizes are sums of git blob sizes under the directory.

**Findings**
- **All 45 candidate directories exist, and all 45 have a LICENSE file.** Every one is permissive:

  | License | Count |
  |---|---|
  | Apache-2.0 | 20 |
  | BSD-3-Clause | 14 |
  | MIT | 7 |
  | BSD-2-Clause | 3 (allegro, both robotiq) |
  | BSD-3-Clause-Clear | 1 (Stretch 2) |

- None is non-commercial or share-alike.
- BSD-3-Clause-Clear grants **no patent rights**.
- The Apache, BSD and MIT licenses still require keeping the copyright notices when redistributing.

| id | dir tree sha | LICENSE | top-level MJCF files | size MB | actuators (file:count) | notes |
|---|---|---|---|---|---|---|
| unitree_g1 | `57c00d310bfd` | BSD-3-Clause | g1.xml, g1_mjx.xml, g1_with_hands.xml, scene.xml, scene_mjx.xml, scene_with_hands.xml | 39.6 | g1_with_hands.xml:43; g1.xml:29 | floating base; g1.xml no hands; g1_with_hands.xml adds Dex3 hands; MJX variants |
| unitree_h1 | `4deafb41d0fd` | BSD-3-Clause | h1.xml, scene.xml | 17.0 | h1.xml:19 | floating base; no dexterous hands |
| pal_talos | `6ee47fcceea2` | Apache-2.0 | scene_motor.xml, scene_position.xml, talos.xml, talos_motor.xml, talos_position.xml | 10.0 | talos_position.xml:32; talos.xml:0 | floating base; talos.xml has no actuators - use talos_position.xml or talos_motor.xml |
| booster_t1 | `c0462e68226c` | Apache-2.0 | scene.xml, t1.xml | 10.6 | t1.xml:23 | floating base; no dexterous hands |
| toddlerbot_2xc | `cf152e3533e3` | MIT | scene.xml, scene_mjx.xml, scene_pos.xml, toddlerbot_2xc.xml, toddlerbot_2xc_mjx.xml, toddlerbot_2xc_pos.xml | 68.1 | toddlerbot_2xc.xml:30 | floating base; small humanoid, many passive/linkage joints; _pos variant |
| toddlerbot_2xm | `23b123674d49` | MIT | scene.xml, scene_mjx.xml, scene_pos.xml, toddlerbot_2xm.xml, toddlerbot_2xm_mjx.xml, toddlerbot_2xm_pos.xml | 68.1 | toddlerbot_2xm.xml:30 | floating base; near-duplicate lineage of 2xc |
| pndbotics_adam_lite | `fd6d43a296b0` | MIT | adam_lite.xml, scene.xml | 178.0 | adam_lite.xml:25 | floating base; largest asset |
| apptronik_apollo | `6241a82dfd76` | Apache-2.0 | apptronik_apollo.xml, scene.xml | 72.1 | apptronik_apollo.xml:32 | floating base; hand DOF not verified |
| berkeley_humanoid | `c29239060f51` | BSD-3-Clause | berkeley_humanoid.xml, scene.xml | 28.9 | berkeley_humanoid.xml:12 | floating base; legs only |
| fourier_n1 | `28ffaee512aa` | Apache-2.0 | n1.xml, scene.xml | 65.9 | n1.xml:23 | floating base |
| robotis_op3 | `c0b2a52ecc84` | Apache-2.0 | op3.xml, scene.xml | 48.5 | op3.xml:20 | floating base; small humanoid, no grippers |
| agility_cassie | `fcaa075b1d62` | MIT | cassie.xml, scene.xml | 7.8 | cassie.xml:10 | floating base biped; closed-chain passive joints; no arms |
| unitree_a1 | `94697b6b078c` | BSD-3-Clause | a1.xml, scene.xml | 21.4 | a1.xml:12 | floating base quadruped |
| unitree_go1 | `4457a836cc37` | BSD-3-Clause | go1.xml, scene.xml | 12.1 | go1.xml:12 | floating base quadruped |
| unitree_go2 | `98d14ab27a56` | BSD-3-Clause | go2.xml, go2_mjx.xml, scene.xml, scene_mjx.xml | 30.9 | go2.xml:12 | floating base quadruped; MJX variant |
| anybotics_anymal_b | `bc0d1e837a9e` | BSD-3-Clause | anymal_b.xml, scene.xml | 59.2 | anymal_b.xml:12 | floating base quadruped |
| anybotics_anymal_c | `eccaa604ebc8` | BSD-3-Clause | anymal_c.xml, anymal_c_mjx.xml, scene.xml, scene_mjx.xml | 18.4 | anymal_c.xml:12 | floating base quadruped; MJX variant |
| google_barkour_v0 | `4f5cb4048b28` | Apache-2.0 | barkour_v0.xml, barkour_v0_mjx.xml, scene.xml, scene_barkour.xml, scene_mjx.xml | 12.6 | barkour_v0.xml:12 | floating base quadruped; MJX variant |
| google_barkour_vb | `be2df4bf92d5` | Apache-2.0 | barkour_vb.xml, barkour_vb_mjx.xml, scene.xml, scene_hfield_mjx.xml, scene_mjx.xml | 6.2 | barkour_vb.xml:12 | floating base quadruped; MJX/hfield variants |
| boston_dynamics_spot | `5da1f3c532ab` | BSD-3-Clause | scene.xml, scene_arm.xml, spot.xml, spot_arm.xml | 55.2 | spot_arm.xml:19; spot.xml:12 | floating base; spot_arm.xml (arm+gripper) is a separate variant |
| franka_emika_panda | `3d2262eeb81e` | Apache-2.0 | hand.xml, mjx_hand.xml, mjx_panda.xml, mjx_panda_nohand.xml, mjx_scene.xml, mjx_single_cube.xml, panda.xml, panda_nohand.xml, scene.xml | 36.6 | panda.xml:8 | fixed base; 7 arm + 1 gripper; panda_nohand.xml; mjx variants |
| franka_fr3 | `b6cad1dd73c8` | Apache-2.0 | fr3.xml, scene.xml | 59.2 | fr3.xml:7 | fixed base; no gripper |
| universal_robots_ur5e | `2a3464301404` | BSD-3-Clause | scene.xml, ur5e.xml | 33.0 | ur5e.xml:6 | fixed base; no gripper (pair with robotiq_2f85) |
| unitree_z1 | `57dcad724b33` | BSD-3-Clause | scene.xml, z1.xml, z1_gripper.xml | 13.5 | z1_gripper.xml:7; z1.xml:6 | fixed base; z1_gripper.xml adds gripper |
| ufactory_lite6 | `0e9ddb13efb5` | BSD-3-Clause | lite6.xml, lite6_gripper_narrow.xml, lite6_gripper_wide.xml, scene.xml | 6.9 | lite6_gripper_narrow.xml:7; lite6.xml:6 | fixed base; gripper variants narrow/wide |
| rethink_robotics_sawyer | `4b0d742b4136` | Apache-2.0 | sawyer.xml, scene.xml | 32.0 | sawyer.xml:7 | fixed base; no gripper |
| ufactory_xarm7 | `16cfe120310f` | BSD-3-Clause | hand.xml, scene.xml, xarm7.xml, xarm7_nohand.xml | 6.3 | xarm7.xml:8 | fixed base; 7 arm + 1 gripper; xarm7_nohand.xml |
| trs_so_arm100 | `1214e62ef659` | Apache-2.0 | scene.xml, so_arm100.xml | 3.9 | so_arm100.xml:6 | fixed base; 5 arm + 1 jaw |
| robotstudio_so101 | `2db61569df87` | Apache-2.0 | scene.xml, scene_box.xml, so101.xml | 19.2 | so101.xml:6 | fixed base; 5 arm + 1 jaw |
| aloha | `402ce95ae465` | BSD-3-Clause | aloha.xml, filtered_cartesian_actuators.xml, joint_position_actuators.xml, keyframe_ctrl.xml, keyframe_no_act.xml, scene.xml | 19.9 | aloha.xml:0; joint_position_actuators.xml:14 | fixed dual arm; actuators come from included joint_position_actuators.xml |
| hello_robot_stretch | `d37c1fc83fc2` | BSD-3-Clause-Clear | scene.xml, stretch.xml | 77.3 | stretch.xml:8 | mobile (free joint + wheels) |
| hello_robot_stretch_3 | `e0d787abe0e1` | Apache-2.0 | scene.xml, stretch.xml | 80.1 | stretch.xml:10 | mobile (free joint + wheels); separate lineage from Stretch 2 |
| pal_tiago | `d16300392923` | Apache-2.0 | scene_motor.xml, scene_position.xml, scene_velocity.xml, tiago.xml, tiago_motor.xml, tiago_position.xml, tiago_velocity.xml | 5.7 | tiago_position.xml:14 | mobile (free joint + wheels); tiago.xml alone has no actuators - use *_position/velocity/motor.xml |
| pal_tiago_dual | `a1f7ee7ee244` | Apache-2.0 | scene_motor.xml, scene_position.xml, scene_velocity.xml, tiago_dual.xml, tiago_dual_motor.xml, tiago_dual_position.xml, tiago_dual_velocity.xml | 7.8 | tiago_dual_position.xml:25 | mobile dual arm; same variant scheme as tiago |
| stanford_tidybot | `c78b37aebf06` | MIT | base.xml, scene.xml, scene_base.xml, tidybot.xml | 11.2 | tidybot.xml:11 | mobile via planar slide joints + arm/gripper |
| google_robot | `dc7d7cc501ed` | Apache-2.0 | robot.xml, scene.xml | 7.3 | robot.xml:9 | no base joints in robot.xml (base fixed in model) |
| rainbow_robotics_rby1 | `128cade03001` | Apache-2.0 | rby1a_1.2.xml, rby1a_1.2_no_gripper.xml, rby1m_1.2.xml, rby1m_1.2_no_gripper.xml, rby1m_1.3.xml, scene_rby1a_1.2.xml, scene_rby1a_1.2_no_gripper.xml, scene_rby1m_1.2.xml, scene_rby1m_1.2_no_gripper.xml, scene_rby1m_1.3.xml | 127.1 | rby1a_1.2.xml:26; rby1m_1.3.xml:28 | floating-base wheeled dual arm; rby1a/rby1m 1.2/1.3 and no-gripper variants |
| wonik_allegro | `42564d556620` | BSD-2-Clause | left_hand.xml, right_hand.xml, scene_left.xml, scene_right.xml | 2.6 | right_hand.xml:16 | hand; left/right |
| shadow_hand | `8f85903630df` | Apache-2.0 | keyframes.xml, left_hand.xml, right_hand.xml, scene_left.xml, scene_right.xml | 4.8 | right_hand.xml:20 | hand (tendon-coupled); left/right |
| leap_hand | `efe8f001e555` | MIT | left_hand.xml, right_hand.xml, scene_left.xml, scene_right.xml | 15.6 | right_hand.xml:16 | hand; left/right |
| shadow_dexee | `efbb028eb957` | Apache-2.0 | scene.xml, shadow_dexee.xml | 10.9 | shadow_dexee.xml:12 | 3-finger hand |
| robotiq_2f85 | `3c3ba06388c1` | BSD-2-Clause | 2f85.xml, scene.xml | 4.3 | 2f85.xml:1 | parallel gripper, 4-bar linkage |
| robotiq_2f85_v4 | `d6c34a7e5192` | BSD-2-Clause | 2f85.xml, mjx_2f85.xml, scene.xml | 20.7 | 2f85.xml:1 | parallel gripper v4 (model name "Dual_wrist_camera") |
| umi_gripper | `c906d8c87d1a` | MIT | scene.xml, umi_gripper.xml | 2.2 | umi_gripper.xml:7 | gripper; 7 actuator elements parsed, not inspected further |
| sharpa_wave | `177b371e5eec` | Apache-2.0 | left_hand.xml, right_hand.xml, scene_left.xml, scene_right.xml | 15.8 | right_hand.xml:22 | dexterous hand; left/right |

Total for the 45 directories: **1454 MB** (sum of git blob sizes; includes meshes, textures and images).

**Lineage cautions:**
- These pairs are near-duplicates and must not be split across train and test as if independent:
  - toddlerbot_2xc / 2xm
  - barkour v0 / vb
  - anymal b / c
  - stretch / stretch_3
  - tiago / tiago_dual
  - robotiq_2f85 / v4
  - trs_so_arm100 / robotstudio_so101
  - franka panda / fr3
- The G1 hand and no-hand variants, and Spot with and without the arm, are variants of one body family.

**Manipulation eligibility, by actuators alone:**
- Humanoids with manipulators in stock MJCF: g1_with_hands, plus probably apollo and talos (their hand DOF is **UNVERIFIED**).
- h1, t1, op3 and berkeley_humanoid have no dexterous hands in stock form.
- Cassie and the quadrupeds are locomotion-only, except spot_arm.

### SIMPLE / Psi0 candidates

**SIMPLE** @ `6d10628794d9c7de4596b4f2afb2c054a637c2bc`, MIT (`license.md`; the GitHub API reports "other"):

| id | path | exists | Details |
|---|---|---|---|
| g1_dex3_simple | `src/simple/robots/g1.py` | yes (blob `9573a633…`) | `dof=31`, `hand_dof=7`; references MJCF `robots/g1/g1_29dof_with_dex3.xml`, USD and a cuRobo yml |
| g1_inspire_simple | `src/simple/robots/g1_inspire.py` | yes (blob `7f2fe55c…`) | `dof=41` (29 + 12 including mimic joints), `hand_dof=6`; MJCF `robots/g1_inspire/g1_29dof_with_inspire_hand.xml` |
| vega_1_simple | `src/simple/robots/vega.py` | yes (blob `c4498f91…`) | `dof=39`, `hand_dof=6`; MJCF `robots/vega_1/vega.xml` |

These are Python wrappers. **The MJCF and USD assets are not in git.**
- `resolve_data_path(auto_download=True)` fetches them from the HF **dataset** `USC-PSI-Lab/SIMPLE` (sha `1ce0fa3956706b408df2c7c0e26b0298aa7411fd`, not gated) as `robots_<name>.zip`:

  | File | Bytes |
  |---|---|
  | `robots_g1.zip` | 134.4 MB |
  | `robots_g1_inspire.zip` | 80.6 MB |
  | `robots_vega_1.zip` | 130.3 MB |
  | `robots_g1_sonic.zip` | 23.1 MB |

- **The dataset declares no license.** The asset license is therefore **UNVERIFIED**, and the MIT code license does not automatically cover upstream Unitree, Inspire or Vega meshes.

**Psi0** @ `4f3720d4…`, Apache-2.0 at the repo root:

| id | path | exists | Format | Size |
|---|---|---|---|---|
| h1_2_psi_assets | `real/assets/h1_2` | yes, 96 files | URDF (`h1_2.urdf`, `h1_2_simplified.urdf`) and MJCF (`h1_2.xml`, `scene.xml`); README says 51 DOF | 50.3 MB |
| inspire_hand_psi | `real/assets/inspire_hand` | yes, 86 files | URDF (`inspire_hand_left/right.urdf`) | 33.4 MB |
| dex3_psi | `real/assets/unitree_hand` | yes, 34 files | URDF (`unitree_dex3_left/right.urdf`) | 16.6 MB |

None of these three directories has its own LICENSE. They derive from upstream Unitree and Inspire descriptions, so treat them as "Apache-2.0 by repo, upstream provenance **UNVERIFIED**".

---

## f. topoformer (read-only)

**Revisions and files read**
- Read via the GitHub API at `campaign/extended-02` = `e80ddacbe731a1ae064d869962bd63720e5f86c0`. No clone was made, and `~/Documents/topoformer` was not touched.
- Blobs read:

  | File | Blob |
  |---|---|
  | `research/campaigns/extended-02/final-report.md` | `4d92b5c3…` (matches the handoff) |
  | `research/campaigns/extended-02/diagnostics/encoded-counterfactuals.md` | `a283c5de…` |
  | `src/topoformer/attention.py` | `40b38a09ff9fd7410a5609f8fb01fedc772dacd7` |
  | `research/stages/stage-01/design.md` | `cec933f8…` |
  | `research/campaigns/extended-01/composition/P01-protocol.md` | `f5fae866…` |
  | `research/campaigns/extended-01/composition/C04-claim-prerequisites-and-next.md` | `806ce975…` |

- The repo declares **no license** (GitHub API license = null). Reimplement the math with attribution; do not copy code.

### Typed structural bias math

`structural_attention(q,k,v,*,bias=None,strength=0.0,allowed=None)` (`attention.py` L46–100):
$$S_{bhij}=\frac{Q_{bhi}\cdot K_{bhj}}{\sqrt{d_k}}+\lambda\,B_{bhij},\qquad S\leftarrow S\text{ masked }(-\infty)\text{ where }\lnot\,\text{allowed}_{ij},\qquad W=\mathrm{softmax}_j(S),\qquad O=WV.$$

- Scores are computed in fp32 (fp64 if any input is fp64).
- If a query row has no allowed key, the function raises `ValueError`.
- The mask is applied after the bias, so a bias can never unmask a key.
- **Orientation:** rows are queries (i) and columns are keys (j). The softmax runs over keys. An edge i→j means "query i may preferentially **read** key j" (`design.md` L48–56).
  - Test: `test_attention.py` L22–31. Setting `bias[0,0,0,2]=1` raises only `W[0,0,0,2]`, and the other rows are unchanged.
- **Zero-bias equivalence:**
  - With `bias=None` or `strength=0`, the function is exactly standard SDPA. `test_attention.py` L9–19 tests this in fp64 for weights and outputs.
  - An explicit bias with λ=0 is equivalent trivially, but that case is untested.
  - Footgun: `strength` defaults to 0, so passing a bias without a strength does nothing.
- **Typing:** the kernel takes one generic bias tensor and one scalar λ.
  - The *design* contract is $\sum_r\lambda_{\ell,h,r}B_{b,r,i,j}$, a sum over relation types r with per-layer, per-head strengths.
  - The campaign code implements a per-(relation, head) learned strength (`campaign_attention.py` about L102, L130–140) for one relation per step.
  - `runtime_model.py` L209–214 soft-mixes the relations as $\sum_r p_rA_r$.
  - `graph_structure` (L103–120) has three modes: none, soft (A as a bias, the same for every head), and hard (a mask of A∪I).
  - There is no (query type, key type) → bias lookup table at this SHA.

### Provenance and failure-memory identifiability (extended-02 §2, encoded-counterfactuals)

We compared the encoded tensors of paired states under three feature versions:

| Paired states | m1 | m2 | m3 |
|---|---|---|---|
| Own current return vs a same-type foreign return | **bit-identical** | differ | differ |
| Own current return vs own stale return | **bit-identical** | differ | differ |
| Rejected for capacity vs rejected for funds | identical | identical | identical |
| "Rejected, then one other action" vs "never committed" | identical | identical | differs |

What this shows:
- **E19's collapse to 0.35 was an interface limit, not a learning failure.**
- **m2 (E20) reached 1.00 in 3 of 3 lineages.** Caveats:
  - m2 is close to an oracle given its naming;
  - evaluation is IID robustness, because the sealed worlds share the training generator.
- **m3 per-stage counters removed rejected-commit loops:** 126 → 0 failures over 22,272 episodes per arm (E22). Caveats:
  - the evidence comes mostly from one lineage;
  - a residual `choose_item` loop remains;
  - no feature version encodes the rejection *reason*.
- **Lesson for this project:** instance identity, version, provenance and failure reason must be **present and distinguishable in the encoded tensors**. Check this with paired counterfactual encodings before any training claim. Several failures were traced to missing inputs, and none of them shows that the model discovered a representation itself.

### Established vs not established

- **Not established:**
  - A structural-attention advantage. The extended-02 report says it is "Absent … (untested)". In C04, supplied exact-copy slightly beats the workspace at delay 16, and the copy and roles baselines are much faster.
  - Functional composition. E16 is sequential composition under a supplied, environment-enforced stage order, and semantic composition in RL01 and RL02 scored 0/512.
  - Multi-return binding and learned scheduling, which are untested.
- **Established, narrowly:**
  - successive halving as an efficient configuration search (E15);
  - transfer to new resource conditions and new stage orders;
  - repair of the return-binding shortcut: assign→select improved from 0.57 to 0.94 (E17);
  - the scoped C04 one-operation pipeline (P01 typed lowering, then the R04 accessor, then R05 consumption) with supplied operation codes, a supplied schedule and a single return.

### extended-03 (`1197afda…` = main)

- 8 commits ahead of `e80dd`. The handoff's `a19ff5d8` ("Register P1 design") is an ancestor.
- Added: `src/topoformer/campaign03_depworld.py`, P1 configs and state files.
- **Status: PAUSED** by the user because the GB10 is needed for robot-policy training.
- **Executed, CPU only:**
  - Stage A depworld world, encoders and references, with 173 tests passing.
  - A dev leverage gate: reuse .976 equals recompute .976 at 11% lower cost.
- **Proposed only:**
  - The P1 arms (reuse-boot, recompute-boot, no-applicability, no-attempt-memory), registered and frozen but not started.
  - No model has been trained.
  - The Stage B preflight audit was interrupted. Partial, untested work is on `campaign/e03-preflight` @ `caf1a785…`.
- Do not interfere with it.
