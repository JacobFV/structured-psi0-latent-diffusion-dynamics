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
