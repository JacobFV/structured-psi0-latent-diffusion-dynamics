Protocol `latent_slice1` (SEALED_BEFORE_RESULTS); raw outputs under `artifacts/runs/latent_slice1_b1fix`.
Rate = successes / attempted (infeasible scenes excluded), pooled over seeds; Wilson 95% CI. Eval scenes seed_start 2,000,000; source competence = held-out source bodies, 50 episodes per seed.

| method | target | budget | kind | successes/attempted | rate | Wilson 95% | per seed (k/n) | target demo eps | target transitions (per seed) | SFT updates | source updates (per seed) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_direct_action | source | 0 | source competence | 79/80 | 0.988 | [0.93, 1.00] | 1701:79/80 | 0 | 0 | 0 | 26304 |
| baseline_direct_action | xarm7_pg2 | 0 | zero-shot transfer | 0/100 | 0.000 | [0.00, 0.04] | 1701:0/100 | 0 | 0 | 0 | 26304 |
| baseline_direct_action | xarm7_tf3 | 0 | zero-shot transfer | 0/100 | 0.000 | [0.00, 0.04] | 1701:0/100 | 0 | 0 | 0 | 26304 |
| baseline_direct_action | panda_tf3 | 0 | zero-shot transfer | 78/100 | 0.780 | [0.69, 0.85] | 1701:78/100 | 0 | 0 | 0 | 26304 |
| baseline_action_only_codec | source | 0 | source competence | 77/80 | 0.963 | [0.90, 0.99] | 1701:77/80 | 0 | 0 | 0 | 26304 |
| baseline_action_only_codec | xarm7_pg2 | 0 | zero-shot transfer | 0/100 | 0.000 | [0.00, 0.04] | 1701:0/100 | 0 | 0 | 0 | 26304 |
| baseline_action_only_codec | xarm7_tf3 | 0 | zero-shot transfer | 0/100 | 0.000 | [0.00, 0.04] | 1701:0/100 | 0 | 0 | 0 | 26304 |
| baseline_action_only_codec | panda_tf3 | 0 | zero-shot transfer | 85/100 | 0.850 | [0.77, 0.91] | 1701:85/100 | 0 | 0 | 0 | 26304 |
