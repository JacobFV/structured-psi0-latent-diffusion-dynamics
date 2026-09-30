# registered research campaign and analysis

## hypotheses

h1: explicit event/role/effect structure reduces new-body adaptation experience relative to a matched shared morphology policy.
h2: compatible module swapping improves transfer to held-out attachments beyond ordinary parameter randomization.
h3: contact-relative anchors improve appropriate tasks under occlusion/noise, not merely fully observed reaching.
h4: cached typed context keeps the useful structural model on a favorable success/latency frontier.
h5: event-boundary rollout reuse helps online adaptation after accounting for all computation and interaction.

Each has supported/contradicted/inconclusive/not-tested states. “More anthropocentric” or “thinks better” is not a measured outcome. Novelty requires primary related-work comparison, not extrapolation from architecture aesthetics.

## campaign stages and budget discipline

A. engineering fixtures and source audit: deterministic CPU/unit tests, first teacher trace, real UI, telemetry and source pinning.
B. development: codec vs direct-action and structured vs matched unstructured policies on compact arm/hand tasks. Optimize only here; fixed maximum of three materially distinct recipe interventions per question before analysis.
C. primary confirmation: freeze the best justified baseline and structured method, three source-training seeds, one held-out arm family and one unseen compatible attachment combination, two tasks, three adaptation budgets plus zero-shot. 100 evaluation episodes per task/body/method/seed/checkpoint. Do not sum all adaptation checkpoints as independent samples.
D. breadth/stress: catalogue import and controller checks across requested families; actual learned-policy evaluations; graph size, topology, contact, uncertainty and runtime interventions. Expensive ablations are not crossed with every robot. Report one-seed stress results as exploratory.
E. RL comparison: a declared subset of up to four eligible body/task cells; no adaptation, SFT initialization, GRPO, shared-prefix GRPO and EXPO-FT-inspired comparison. Run a short truthful learning evaluation for every implemented method, even when a method does not merit expensive confirmation.
F. audit, analysis, reproducibility replay and final workbench/report packaging.

Allocate by measured profiles, not a fixed endless grid. Reserve 20% for confirmation and 10% for audits/reporting; check projected total cost before each release. Default limits are in `config/resources.json`. Budget failure is a declared experimental boundary. Finish independent engineering and documentation without silently claiming full experimental execution.

## split and eligibility controls

Split on body lineage, module lineage, task templates, scene/object instance, randomization seed and episode origin. Near-identical generated descendants must not leak across morphology-family holdouts. Freeze task/body eligibility using analytic capability + teacher/controller feasibility before testing learned methods. Include rejections/failures with denominators. Never remove a difficult cell after looking at policy performance.

Public inputs include known morphology/specs, configured sensors and the supplied plan. Exact simulator object poses, future completion, successful branches and target labels are not deployable inputs. Target-teacher training/checks and controller calibration are costs, even when excluded from the high-level model's learning. Report “existing-controller” and “new-controller” deployment separately.

Exact same recorded public information reaches comparison arms. A graph-as-input baseline tests whether benefits come from information rather than imposed routing. A zero-strength structural model recovers the matched attention model numerically. Report the real upstream ψ₀ only on its supported interfaces/tasks; do not manufacture an “upstream cross-body score” for an invalid action head.

## metrics and denominators

Primary: task success at each budget; new-body control transitions/demos to sustained 80% success; p95 policy latency overhead. Also report area under the adaptation curve over log(1+budget), source-task retention, completion time, event correctness, collisions/falls/constraint violations, contact-frame errors/calibration, controller tracking error, manual integration time and compute/disk/data costs.

Distinguish control transitions from smaller physics substeps, action chunks, sampler function evaluations, rendered observations, replay samples and optimizer presentations. Count a shared prefix once when actually executed; count all training reuse separately. Report per-method compute, not only sample efficiency. No multiplication of marginal probe accuracies to invent end-to-end reliability.

A failed episode, timeout, refusal, stale-action rejection, simulator crash caused by the method, and invalid command remain in the all-attempted denominator with categories. Externally caused machine outages have a separately predeclared handling rule and preserved records; do not quietly rerun only failures. If both methods fail to reach threshold, show censored curves rather than a meaningless ratio.

Per-cell intervals: Wilson intervals for episode success conditional on the trained checkpoint; paired episode differences on fixed conditions; bootstrap over independent training seeds/episode conditions with dependence acknowledged. Three seeds provide limited seed-distribution precision; do not overstate it. Adaptive checkpoint selection uses development only. Report raw counts, not just percentages.

Latency: warmup, synchronize correctly, measure full sensor->context->sampler->codec->controller time; cold and cached cases; actual batch, n/h/c widths and NFE; p50/p95/p99; deadline misses; allocation peaks. GUI timestamps are not CUDA measurements. Runtime comparisons are on the same machine/config/load envelope. Record external-load invalidation criteria before measuring.

## required discriminating interventions

- same observations, structure gates zero / true / reversed / rewired; graph serialization baseline.
- actor, patient and destination permutation; duplicate-looking objects; multiple role slots.
- wrong type, same-type foreign result, stale same-object result, old-but-valid result.
- rejected action -> unrelated action -> retry, preserving history and reason.
- contact anchor absent/valid/slipping/expired, with vision noise controlled.
- compatible hand swap, impossible hand substitution, changed controller interface.
- supplied stage-order generalization versus functional output-to-input composition.
- node-order permutations and larger/unseen topologies; missing sensors and null objects.
- intervention on task/event latent with all other context held fixed, and decode-to-control verification.

Before training a discrimination task, build matched counterfactuals and compare ACTUAL model-input tensors. If required states are identical, repair the observation/interface or label the problem unidentifiable. This is an especially important transfer from topoformer's extended-02 findings.

## reports and machine-readable outputs

Every run has `RunManifest` + per-episode JSONL + checkpoints/config/source hashes + resource receipts. The analysis program reads raw metrics and emits tables/figures deterministically. Do not manually type headline results. A lightweight independent audit recomputes success counts, checks split hashes, and reproduces one checkpoint/cell and one failed intervention.

Final report: question; source relations and novelty limits; exact implementation; public/privileged/supplied/learned decomposition; robot/controller coverage; data and compute; primary and negative results; ablations; adaptation/retention; latency; failure mechanisms; limitations; exact reproduction commands. Include a two-page collaborator brief and short narrated or captioned workbench demo. No paid or public publishing step is implicit.
