# EXPO-FT-inspired comparator: ingredients, port, departures (P19)

Label in all reports: **"EXPO-FT-inspired"** (not a reproduction). Code: `src/rrp/learning/{expo,critics,
replay_buffer}.py`; tests `tests/unit/test_replay.py`. Sources: EXPO-FT arXiv 2605.25477v2 + code
`pd-perry/expo-ft@803381f`; EXPO arXiv 2507.07986v3 (see `source-audit.md` §d).

## ingredients ported

| EXPO-FT ingredient | our port |
|---|---|
| base policy = π0.5 flow VLA, image encoder frozen, updated online with its own flow-matching loss | our FlowPolicy; context encoder + aux readout frozen; action expert updated with its own flow-matching loss (`FlowPolicy.loss`, codec frozen) |
| base imitation batch only from successful episodes (`actor_success_only`, code) | `base_success_only=True`: H-row windows of executed native commands from successful replay episodes, normalized with the public action space at each decision state |
| edit policy: tanh-Gaussian, â = β tanh(·) ∈ [−β, β], MLP 3×256 on critic features + base chunk | same (3×256 LayerNorm MLP); input = frozen observation embedding + flattened base chunk (C×n normalized) |
| edit acts on executed C-step chunk | C = execute_prefix = 8 rows × all action nodes |
| Q ensemble of 10, LayerNorm, min over 2 random target members, τ_Q = 5e-3, no entropy in target | same (`QEnsemble`, `min_of_random_pair`, `soft_update`) |
| edit loss −E[Q(s, a+â) − α log π(â)], mean over all online critics | same |
| SAC α, α₀ = 1, target entropy −C·d_a/2 | same; entropy measured on the unit-scale squashed variable (see departures) |
| OTF selection: N = 8 base samples + their edits, argmax Q over 16, scored by target critic min-of-2 | same (N = `n_base` = 8) |
| TD backup with the same OTF rule at s′, chunk-level n-step return | y = Σ γ^i r + γ^n (1−done) max_{cand} min2 Q′(s′, cand) |
| UTD 20, batch 64, Adam 3e-4, γ = 0.99, ~1 update call per 40 env steps | same defaults (`utd`, `batch`, `env_steps_per_update`) |
| replay seeded with demos, offline_ratio 0 | **no demos in replay** (see departures) |

## replay record

One record per decision point: base proposal, edit, executed (post-joint-bound-clip) normalized rows,
executed native rows, all N base candidates, frozen observation embedding, reward, n_steps, done vs
truncated, and versions (robot spec hash, controller version, graph/runtime version, base policy
version, edit version, codec version). A controller- or robot-version mismatch raises `ValueError`
(tested). Capacity-bounded with eviction counts.

## departures (all labelled)

1. **Observation encoder**: a frozen copy of the initial policy's context encoder (mean-pooled public
   token banks) replaces EXPO-FT's trainable ResNet-50 critic encoder. Embeddings are computed once per
   record (no image stream in this task).
2. **Next-state base candidates** in the TD backup reuse the N base samples drawn at s′ during the
   rollout (stale w.r.t. later base updates) instead of resampling the base policy per minibatch; the
   edits and critics are current. Saves N×K policy forwards per backup.
3. **Decision = SMDP option**: a chunk may be cut short by a task-runtime invalidation; the critic still
   scores the full submitted C-row chunk and the backup discounts by the actual n_steps.
4. **Timeouts** are truncations that bootstrap from the final-state embedding (EXPO-FT marks windows
   touching a timeout invalid); a truncated final record without a next decision reuses its own base
   candidates for the backup.
5. **Temperature**: entropy target −C·d/2 is defined on the unit-scale tanh variable; with the scale
   β the exact density differs by the constant −d log β, which cannot satisfy −C·d/2 for β = 0.05
   (the maximum entropy of a distribution on [−β, β]^d is d log 2β < −d/2). The constant does not affect
   the edit-policy gradient.
6. **No human interventions, no demos in replay** (fair matched-information comparison with GRPO, which
   also only uses its own online experience after the same starting checkpoint).
7. **Rewards**: sparse terminal success from the privileged simulator evaluator (sim training only).
8. **Evaluation** uses the OTF policy with the edit mean (deterministic) — EXPO-FT's evaluation-time
   sampling choice is not stated.

A frozen-base variant (`base_update=false`) is a separately named "frozen-base edit" baseline, not the
referenced method.

## compute accounting

Reported per run: new control transitions (the matched budget), rollout base-policy velocity evaluations
(B·N·K per decision batch — N× GRPO's actor cost), critic/edit/base optimizer steps, base update
forwards, update wall time vs rollout wall time.
