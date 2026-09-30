# latent_slice1 progress (corrected architecture, R38)

## stage A — representation (encoder E + system-0 realizer R + packet probes P), frozen after this stage
Evaluation rows are drawn from the TRAINING distribution (packed pick_place v3dart, source bodies) with a separate
RNG stream — i.e. encoded-target (oracle) quality, NOT held-out generalization.
Raw: artifacts/runs/latent_sem_v1/result.json, artifacts/runs/latent_nosem_v1/result.json

| metric | latent_sem_v1 | shuffled-z control | latent_nosem_v1 |
|---|---|---|---|
| system-0 realization MSE (normalized 1-step action) | 0.0129 | – | 0.0117 |
| zero-action MSE (reference) | 0.213 | – | 0.213 |
| packet probe held_by accuracy on positives | 0.954 | 0.145 | (probe untrained by design; post-hoc probe pending) |
| packet probe subtask accuracy | 1.00 | 0.649 | pending |
| packet probe focused_on accuracy | 1.00 | 0.879 | pending |
| packet probe visible accuracy | 0.983 | 0.811 | pending |
| relative position to TCP error (m) | 0.056 | 0.371 | pending |
| desired displacement error (m) | 0.011 | 0.016 | pending |

Pending: post-hoc measurement probes on frozen z (both variants) + metadata-only control; stage-B flows; free-sample
packet semantics; closed-loop success; disturbance; causal interventions; latency.

## post-hoc measurement probes (frozen, detached z; identical procedure for both; 6000 steps)
Training distribution, encoded targets (not generated packets). Raw: `artifacts/runs/latent_*_v1/probe_*.json`.

| metric | sem z | sem shuffled | nosem z | nosem shuffled | metadata only (no z) |
|---|---|---|---|---|---|
| held_by accuracy on positives | 0.956 | 0.151 | 0.888 | 0.239 | 0.000 |
| acting_on accuracy on positives | 0.871 | 0.220 | 0.622 | 0.230 | 0.000 |
| focused_on accuracy on positives | 1.000 | 0.840 | 0.981 | 0.869 | 0.842 |
| subtask accuracy | 1.000 | 0.645 | 0.968 | 0.650 | 0.784 |
| relative position error (m) | 0.054 | 0.365 | 0.258 | 0.267 | 0.268 |
| desired displacement error (m) | 0.012 | 0.016 | 0.015 | 0.015 | 0.014 |

Without semantic supervision the packet still carries holding/contact/subtask information, but almost no relative
geometry (0.258 m error vs 0.268 m with metadata alone).

## why stage-B semantic flow loss was higher (D-031)
Latent scale: second moment 5.24 (sem) vs 0.144 (nosem); the flow regressed raw z, so its do-nothing floor was
6.24 vs 1.14. v2 flows standardize the target. With standardization, the packet-semantic term still pushes the flow
above its do-nothing value early; v3 restricts semantic gradient to tau >= 0.6.
