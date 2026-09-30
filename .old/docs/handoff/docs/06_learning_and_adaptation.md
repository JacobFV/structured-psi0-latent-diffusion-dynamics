# learning, sample-efficient adaptation, and algorithm verification

## phase 1: teachers, codec and behavioral baseline

Acquire validated teacher data with separate source and target-body splits. Fit normalization using source training data or public physical limits. Use a tiny CPU fixture to test batch collation and backpropagation before GPU work. Overfit a small deterministic demonstration batch as a debugging test; it is not generalization evidence.

Train the action codec with masked reconstruction and measured physical-effect error. Required controls: target latents decoded correctly; shuffled latents change actions; swapped bodies use correct decoder conditioning; held-out body reconstruction and teacher-latent rollout. A high latent similarity does not prove correct control. Freeze encoder/decoder for the initial policy experiment. Retain a no-codec direct normalized native-action target.

Train the shared action model first on unstructured serialized supplied task information and the same observations used by the structured model. Then add grouped auxiliaries. Compare matched parameter budgets or report actual differences; detach/freeze unused probe branches consistently. Reusing the same VLM feature cache is legitimate only for identical permissible images/processing; it does not eliminate the online cost of that VLM.

## auxiliary groups and targets

**entity/relation:** object identity across prompt/view/time, visible/gaze/focus/acted-on, held-by per manipulator, contact/support modes. Use simulator truth only for labels. Handle empty scenes, multiple simultaneous targets, occlusion, distractors and object re-entry.

**geometry/belief:** relative transforms and distances, projection/depth, contact-frame validity, slip, uncertainty and observability. Include known-FK baselines, sensor-noise/randomization and inconsistent-measurement tests. Do not confuse centroid distance with barycentric coordinates.

**event/role/effect:** operator classification, ordered role binding, prerequisite/overlap/resource edges, actual output receipts and desired physical effects. Hard runtime validity is supplied; learned decoding still requires tests. Effects include relative transforms, mode changes and continuous thresholds, not only verbs.

**compatible-swap alignment:** matched goals/worlds with different feasible hands/arms. Align role/task/physical-effect projections, not native joint coordinates or necessarily the full action latent. Preserve capability differences. Contrast valid alternative strategies against infeasible substitutions. Choose weights on development data, inspect gradient norms/conflict and report ablations.

**object QA:** prefix projection into a separate frozen text decoder. Test gradient reaches object encoder/projection but not decoder weights; empty/incorrect object substitutions degrade object-dependent answers. Save head/backbone revisions; unload or CPU-offload decoder when not training/querying it. A strong frozen language prior alone cannot satisfy grounding.

For at least one auxiliary group, read from system-i hidden/action representations rather than only the clean context encoder. Test that the loss produces nonzero gradients in the intended action-expert blocks and that interventions change corresponding decoded control. Current-state semantics may be supervised at sampled flow times; future-effect supervision uses a predicted clean representation and a declared time-dependent weight/mask. Do not treat decodability from a detached side branch as evidence that denoising was structured.

The supplied event graph supervises semantics and is a public input in v1. Later language-only acquisition is outside mandatory scope. Train queries to encode/decode the same canonical entity and event spaces, not disconnected classifiers that never affect the action model.

## inference and retention

Evaluate source-body competence before target adaptation. Save independent initialization checkpoints for each target/method/seed; do not sequentially adapt one model across held-out bodies and call each independent. Report retention on source tasks after adaptation. Keep perception frozen initially, unfreezing only when a registered development diagnostic supports it. Record the changed module list and parameter count.

Zero-shot means no target policy gradient updates or target demonstration normalization. A target-specific tracker or calibration is permitted only under the label “zero-shot high-level policy with existing/validated controller,” with its cost reported. True zero-integration deployment is a stronger separate claim.

Supervised target budgets: 0, 5, 20, 100 episodes; record episode lengths and control transitions. Use nested, fixed data subsets per target/seed for fairness. Optimizer updates and data reuse are explicit. A comparison to a new per-robot head begins at a nonzero adaptation budget; no nonexistent head is scored as a meaningful zero-shot baseline.

## flow-SDE GRPO: implement the actual likelihood, not a proxy

Verify the Z-1 paper's appendix and its cited flow-SDE construction in primary code/papers. Write `research/methods/grpo-derivation.md` mapping its notation and time direction to this project's τ=0 noise, τ=1 data convention. Test this map. Document all departures from upstream (latent action space, custom codec, event-defined branch points).

Optimize a stochastic generative path with recorded transitions; the deterministic codec/controller are fixed during a GRPO update. For a nondegenerate Gaussian transition with mean μθ, variance σ² and valid dimensionality d, the log-density is:

```text
log pθ(z_next | z_current, observation, τ)
  = -0.5 * sum_valid(((z_next - μθ) / σ)^2 + 2 log σ + log(2π))
```

Sum valid transition log-densities for the declared augmented-path objective. Record that this is a path likelihood, not an analytically computed marginal continuous-action density. Do not divide by node count and still call the result the exact likelihood ratio. Any dimension-normalized surrogate must be named and compared explicitly. Singular/deterministic endpoint steps cannot be assigned this Gaussian density without a justified treatment; use a documented schedule and mathematically valid endpoint handling.

Store old-policy log probabilities at rollout time, immutable behavior version, full sampled intermediate latent path, observation/cache provenance, valid masks, time grid and variance schedule. Recompute the new policy's likelihood on the SAME sampled path and observation. Ratio is one at identical weights; padding has no effect; invalid variances fail early. Clip the surrogate ratio only as prescribed, not probabilities or samples to hide overflow. Policy-gradient objectives are not flow-matching MSE objectives.

Group-relative advantages require actual independent suffix outcomes. Equal-return groups carry no ordinary mean-normalized signal; handle zero variance explicitly. Reward shaping or success-time ordering is documented separately. If retaining a KL penalty unlike the reference, label the change and compare it. Do not call an arbitrary PPO variant a faithful Z-1 reproduction.

Shared-prefix groups branch from a full continuation snapshot at a selected event/interaction boundary. The shared prefix is physically executed once and excluded from inappropriate suffix credit. Tree branching, when enabled, tracks which chunks are actually shared and the branch-specific reward aggregation. Keep prefix selection independent of final-test outcomes. Count effective groups, suffix diversity, duplicate transitions, policy forwards and snapshot overhead.

Tests: analytic 1D transition density; identity ratio; mean-shift direction; noise/time conversion; invalid/masked coordinate handling; gradient exclusion of frozen codec/controller and shared prefix; full-state deterministic branching; old-policy staleness rejection; zero-advantage groups; unit-scale synthetic control improvement. Then a short real simulation run must produce actual updates/returns before larger training.

## EXPO-FT-inspired comparator

Read the source paper and available implementation before implementing the edit policy, replay, critic/objective and base-policy updates. The current referenced approach is not just a permanently frozen base plus residual correction. Record its exact ingredients and what can be ported to our action interface; give the comparator an honest “inspired” label unless reproducing the original recipe.

The replay record distinguishes base proposal, edit, executed native action, sensor observation, robot/controller/task versions and termination. Keep base-policy updates and edit-policy conditioning consistent under replay, and account for both compute costs. Use the paper's justified objective rather than improvising an importance ratio on clipped actuator commands. Implement mathematical unit tests and a real short learning run. Compare at matched newly executed environment experience; show replay and optimizer compute separately.

## profile-driven scale and scheduling

Use the development profiles in the machine-readable config. A small model earns a longer run by passing shape, gradient, learning and latency gates. The actual VLM-backed path is mandatory even if most ablations use frozen/cached features. Scale the action expert only after measured benefit/cost. Serve a single GPU worker per peer lease; do not simultaneously allocate independent VLM, text-QA, learner and renderer maxima.

Prefer serialized or explicitly budgeted phases: generate batches; train codec; train policy; query QA; evaluate; adapt. Reuse safe frozen features and compact datasets. Asynchrony is allowed only with bounded queues and versioned policies; all stale-policy assumptions must be explicit.
