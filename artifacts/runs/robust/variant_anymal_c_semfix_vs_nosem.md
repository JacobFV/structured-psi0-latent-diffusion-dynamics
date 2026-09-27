# variant-level robustness: semfix vs nosem on anymal_c (training seeds [0, 1, 2])

Per eval seed: level mean = mean success over the perturbed single-factor levels (all_moderate excluded); drop = nominal - level mean. Per training seed: paired over the 20 eval seeds (seed-bootstrap CI, sign-flip p). Pooled: per-eval-seed differences averaged over training seeds. Variant level: the 3 training-seed means per variant, exact permutation over all 20 relabellings (min one-sided p = 0.05).

## privileged_success

| training seed | level mean a / b | diff [95% CI] p | drop a / b | diff drop [95% CI] p |
|---|---|---|---|---|
| 0 | 0.744 / 0.833 | -0.089 [-0.115, -0.064] p=4.5e-05 | +0.106 / +0.167 | -0.061 [-0.230, +0.083] p=0.445 |
| 1 | 0.770 / 0.845 | -0.076 [-0.108, -0.042] p=0.000555 | +0.130 / +0.105 | +0.026 [-0.152, +0.180] p=0.875 |
| 2 | 0.786 / 0.823 | -0.036 [-0.077, +0.006] p=0.128 | +0.114 / +0.077 | +0.036 [-0.161, +0.230] p=0.675 |
| pooled | - | -0.067 [-0.090, -0.045] p=7e-05 | - | +0.001 [-0.121, +0.110] p=1 |

variant level (3 vs 3): level mean a 0.744 / 0.770 / 0.786 vs b 0.833 / 0.845 / 0.823, diff -0.067, exact p one-sided (a lower) 0.05, (a higher) 1.00, two-sided 0.10; drop a 0.106 / 0.130 / 0.114 vs b 0.167 / 0.105 / 0.077, diff +0.001, one-sided (a lower) 0.60, (a higher) 0.50, two-sided 1.00

## public_success

| training seed | level mean a / b | diff [95% CI] p | drop a / b | diff drop [95% CI] p |
|---|---|---|---|---|
| 0 | 0.836 / 0.868 | -0.032 [-0.048, -0.017] p=0.00185 | +0.164 / +0.132 | +0.032 [+0.017, +0.048] p=0.00185 |
| 1 | 0.832 / 0.894 | -0.062 [-0.077, -0.047] p=1e-05 | +0.168 / +0.106 | +0.062 [+0.047, +0.079] p=1e-05 |
| 2 | 0.830 / 0.874 | -0.044 [-0.065, -0.023] p=0.00137 | +0.170 / +0.126 | +0.044 [+0.024, +0.064] p=0.00137 |
| pooled | - | -0.046 [-0.056, -0.036] p=1e-05 | - | +0.046 [+0.036, +0.056] p=1e-05 |

variant level (3 vs 3): level mean a 0.836 / 0.832 / 0.830 vs b 0.868 / 0.894 / 0.874, diff -0.046, exact p one-sided (a lower) 0.05, (a higher) 1.00, two-sided 0.10; drop a 0.164 / 0.168 / 0.170 vs b 0.132 / 0.106 / 0.126, diff +0.046, one-sided (a lower) 1.00, (a higher) 0.05, two-sided 0.10
