# arm recipe ablation overlays (D-126 #4; D-099 fairness: can upstream recipe choices rescue nosem?)

Fragments, not DAGs: each changes ONE upstream recipe choice of the arm lineage DAG and renames the output lineage so
nothing collides with the parent's runs. Compose with the lineage DAG of the data set under test (list `extends`,
folded left: parent, then fragment, then the composing file):

```yaml
# dags/arm_lineage_v6_nosem_betakl.yaml
extends: [arm_lineage_v6.yaml, overlays/arm_recipe/stageA_beta_kl.yaml]
name: arm_lineage_v6_nosem_betakl
```
`rrp run-dag dags/arm_lineage_v6_nosem_betakl.yaml --dry-run` then shows only the changed configs with new hashes.
Every fragment sets `matrix.variant: [nosem]` (the question) and keeps the parent's seeds; add `semfix` to the matrix
for a same-recipe control. Values are vars (`abl_*`) that the composing file may override.

| fragment | changes | default ablation value (parent) |
|---|---|---|
| stageA_beta_kl.yaml | Stage A `latent.beta_kl` | 1e-4 (1e-3, LatentConfig default) |
| bcdagger_schedule.yaml | BC-DAgger rounds bc1/bc2/bc3: episodes per body, and the bc refits' `dagger_frac` | 48 episodes, 0.7 (24, 0.5) |
| refit_lengths.yaml | system-0 refit steps: bcdag1, bcdag2, gendag1/2/3 | 16000 each (4000, 16000, 8000) |

Tests: tests/unit/test_d126_arm_dags.py composes each fragment with dags/arm_lineage_v2.yaml and checks that exactly the
declared keys differ from the parent and that no output directory is shared.
