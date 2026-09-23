# flow-SDE GRPO: derivation, notation map, and departures (P18)

Status: implemented in `src/rrp/learning/{flow_sde,grpo,branching,rollout}.py`; math tests in
`tests/unit/test_grpo_density.py`, integration checks in `tests/integration/test_adapt_branching.py`.
Sources: Z-1 (arXiv 2606.31846v1 §3.1.2, §3.2, App. D.3–D.4), πRL (2510.25889v3 §4.2), Flow-GRPO
(2505.05470v5 Eq. 7–9, App. A), RLinf code. See `research/methods/source-audit.md` §c.

## 1. conventions

| | Flow-GRPO / openpi / RLinf | πRL (paper text) | **this project** |
|---|---|---|---|
| noise end | t = 1 | τ = 0 | **τ = 0** |
| data end | t = 0 | τ = 1 | **τ = 1** |
| interpolant | x_t = (1−t)x₀ + t x₁ | A^τ = τA + (1−τ)ε | **z_τ = (1−τ)ε + τx** |
| velocity target | v = x₁ − x₀ = ε − x | u = A − ε | **v = x − ε** |
| sampling direction | t: 1 → 0 (Δt < 0) | τ: 0 → 1 | **τ: 0 → 1 (h > 0)** |

Map: τ = 1 − t, v_ours = −v_FG, dτ = −dt. πRL's Eq. 8 reproduces Flow-GRPO's score term written
for the opposite convention; read literally with δ > 0 in πRL's own convention it has the wrong
sign (audit §c). We therefore re-derive in our convention instead of copying either.

## 2. derivation (our convention)

Posterior means given z = z_τ and the learned velocity v̂ ≈ E[x − ε | z]:
- z = (1−τ)ε̂ + τx̂ and v̂ = x̂ − ε̂ ⇒ **x̂ = z + (1−τ)v̂**, **ε̂ = z − τv̂**.

Score: p(z | x) = N(τx, (1−τ)²I) ⇒
**s(z, τ) = ∇log p_τ(z) = −E[(z − τx)/(1−τ)² | z] = −(z − τx̂)/(1−τ)² = −ε̂/(1−τ) = −(z − τv̂)/(1−τ)**.
(Agrees with Flow-GRPO App. A Eq. 27 under the map.)

Marginal-preserving SDE: for dz = f dτ + g dW the Fokker–Planck equation is
∂p/∂τ = −∇·(f p) + (g²/2)Δp. With f = v̂ + (g²/2)s and p s = ∇p the diffusion terms cancel,
leaving the probability-flow ODE's continuity equation ∂p/∂τ = −∇·(v̂ p). Hence

  **dz = [ v̂ − g(τ)²/(2(1−τ)) · (z − τv̂) ] dτ + g(τ) dW.**

Schedule (Flow-GRPO σ_t = a√(t/(1−t)) mapped by t = 1−τ): **g(τ) = a√((1−τ)/τ)**, giving
dz = [(1 + a²/2)v̂ − a²z/(2τ)]dτ + g dW (Flow-GRPO Eq. 8 after the map).

Euler–Maruyama on the uniform grid τ_k = k/K, h = 1/K, g_k = g(τ_k):

  **μ_θ,k = z_k + h v̂_k − g_k² h/(2(1−τ_k)) · (z_k − τ_k v̂_k),  s_k = g_k √h.**

This is `flow_sde.transition_mean`; it is the general form (valid for any g, including the clamp).
Code: `diffusion_schedule`, `sample_sde`, `path_log_prob`.

## 3. endpoints (implemented choice)

- **τ = 0** (step 0): g → ∞. Default `first_step="clamp"`: g₀ = a√((1−τ₀)/τ₁) = a√K (RLinf /
  Flow-GRPO clamp) with the general mean ⇒ μ₀ = z₀ + h v̂ − (a²/2) z₀, s₀ = a. Alternative
  `first_step="deterministic"`: ODE step, no density assigned.
- **τ = 1** is never evaluated (last step starts at τ = 1 − h, s = a h/√(1−h) > 0). Default
  `last_step="stochastic"` (RLinf default). Alternative `"deterministic"` (`ignore_last`): ODE step,
  excluded from the likelihood.
- Deterministic steps are **never** given a Gaussian density (`stochastic` mask in the recorded path).
- K ≥ 2 is enforced (K = 1 is degenerate); a ≤ 0 or non-finite is rejected.

Default run setting: K = 8 (the deployed sampler's NFE), a = 0.5 (πRL/RLinf `noise_level`), clamp
first step, stochastic last step ⇒ **all K steps stochastic** (RLinf `joint_logprob`-style full-chain
likelihood, which is what Z-1 App. D.3 writes; RLinf's default single-random-SDE-step mode is not used).

## 4. objective (augmented path likelihood, not an action density)

  log p_θ(path | o) = Σ_{k stochastic} log N(z_{k+1}; μ_θ,k(z_k, o, τ_k), s_k² I)
                    = −½ Σ_k Σ_valid [((z_{k+1} − μ_θ,k)/s_k)² + 2 log s_k + log 2π].

- Summed over **valid** coordinates (horizon × real action nodes × latent dims); padded nodes contribute
  exactly 0 (tested with NaN/garbage padding). No dimension normalization. (Flow-GRPO/RLinf code
  *averages* over dimensions — a dimension-normalized surrogate we do **not** use.)
- The N(0, I) prior of z₀ is omitted (parameter-free; cancels in the ratio). log s_k likewise cancels.
- This is the likelihood of the **augmented latent path** z₀…z_K, not the marginal density of the
  final latent/action, which is intractable. The executed action is decode(z_K) through the frozen codec
  (identity for the direct normalized-action policy) and the frozen joint controller.
- Ratio per chunk: ρ = exp(log p_θ − log p_θold), recomputed on the **same recorded path and the same
  observation** (the stored PolicyInput is re-collated). Tested: ρ = 1 at identical weights
  (TinyVel: |ρ−1| = 0 in float64; real FlowPolicy in float32: < 1e-4).

## 5. GRPO update (Z-1 Eq. 1–4)

- A_i = (R_i − mean R)/(std_pop R + ε), ε = 1e-6 (Z-1 does not state ε).
- **Zero-variance groups** (all equal returns) carry no mean-normalised signal: they are **excluded**
  and counted (`zero_variance_groups`, `excluded_chunks`). Option `all_equal="z1_min"` implements
  Z-1's A_i = R_i − min R for success-time-decayed rewards (off by default: our reward is undecayed).
- L = −1/|B| Σ min(ρA, clip(ρ, 1−η, 1+η)A), η = 0.2; the trajectory advantage is applied to every
  trainable chunk of that rollout (no per-denoising-step credit). The Z-1 normaliser Σ|B_i| is
  applied per minibatch (standard micro-batching approximation).
- **No KL, no reference policy** by default (faithful to Z-1). `kl_coef > 0` adds the exact
  same-variance Gaussian path KL Σ_k Σ_valid (μ_θ − μ_ref)²/(2s_k²) to a frozen initial copy; any run
  using it must be labelled as a departure.
- AdamW, weight decay 0.01, grad-clip 1.0, 4 rollout epochs (Z-1 C.3). Learning rate is set per run
  (Z-1's 5e-6 is for a 3B VLA; our small action expert uses 1e-5 by default, recorded in configs).
- Trainable set: action expert only (context encoder and auxiliary readout frozen, like Z-1's frozen
  VLM). Codec and controller are frozen and outside the graph. Changed modules + parameter counts
  are written to each run's `result.json["modules"]`.
- Staleness: every recorded path carries its behavior version (`<method>@<iteration>`); the learner
  rejects paths whose version differs from the current iteration (tested).

## 6. shared-prefix branching (Z-1 §3.2, event-defined)

- Departure: the branch point is the **public runtime event boundary** `grasp == succeeded`
  (not a fixed chunk window P). It is independent of final-test outcomes.
- Leader rollout → `Session.snapshot()` (physics incl. warm-start, controller interpolation state,
  task runtime, tracker/belief, command queue, env + sampler RNG) → G branch sessions `restore()` →
  independent SDE suffixes. The policy's previous-action feature is copied to each branch. At the branch
  point the leader's in-flight queue is dropped so each branch samples a fresh chunk (documented replan).
- Prefix chunks are excluded from all trainable sets; prefix transitions are counted **once** as new
  experience; the (G−1)·prefix transitions that sharing avoided are reported as
  `reused_prefix_transitions`. If the leader terminates before the boundary, the group has no gradient
  but its steps still count. Snapshot/restore wall time is logged.
- Determinism tested: two restores + identical commands ⇒ bit-identical physics trajectories.
- Tree branching (Z-1 §3.3) is not implemented.

## 7. rewards

Sparse terminal success from the **privileged simulator evaluator** (`privileged_sim_success`, allowed as
a reward in simulation training only; never a policy input). Optional documented shaping:
`shaping_grasp` × [public runtime grasp succeeded] (0 in all reported runs unless stated).

## 8. departures from upstream (summary)

1. Latent action space = our flow policy's per-node chunk (direct normalized joint targets, or frozen
   codec latents); chunk horizon 16, 8 executed rows.
2. Full-chain stochastic likelihood (all K steps), clamp at τ₀; not RLinf's single random SDE step.
3. Event-defined branch point; no tree variant.
4. Zero-variance groups excluded explicitly (Z-1's "reward filtering" is undefined).
5. Learning rate scaled for a small model; everything else per Z-1 C.3 unless a run config says otherwise.

This is **not** a faithful Z-1 reproduction (different model, task, branch rule and scale).

## 9. measured ratio sensitivity of the exact path likelihood (run finding, 2026-09-21)

With K = 8 fully stochastic steps and ~16×8 valid coordinates per step, the summed path log-ratio is
extremely sensitive to parameter changes: the smallest transition std is s_{K−1} = a h/√(1−h) ≈ 0.067
(1/s² ≈ 224). v3 runs at lr 3e-5 had a ratio clip fraction of 0.95–0.98 in every iteration although the
first-pass ratio was exactly 1 (max |ρ−1| ≤ 8e-5). The training-seed probe
(`scripts/dev/grpo_ratio_probe.py`, output `artifacts/runs/adapt_grpo_suffix_xarm7pg2_v3/ratio_probe.txt`)
measured the median |log ρ| after ONE Adam step: 12.6 (lr 3e-5), 2.2 (1e-5), 0.53 (3e-6), 0.16 (1e-6);
with a deterministic last step: 5.4 / 1.3 / 0.33 / 0.10. Consequences for v4:
- exact objective: lr 3e-7, 2 epochs, `last_step="deterministic"` (RLinf `ignore_last`), still summed;
- a **dimension-normalized surrogate** (`logprob_reduction="mean_dims"`: log-ratio divided by
  #valid coords × #stochastic steps, as in Flow-GRPO/RLinf code) is run as a separately named variant.
  It is not an exact likelihood ratio.
These choices come from training-side diagnostics only, never from evaluation outcomes.
