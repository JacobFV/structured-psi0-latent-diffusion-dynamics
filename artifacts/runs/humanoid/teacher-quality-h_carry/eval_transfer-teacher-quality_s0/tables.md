# humanoid_teacher_quality_h_carry (h_carry)

coverage: {'dev/done': 3}

## dev: Reference: scripted / privileged teacher (not a learned method) (budget unit: none; reference: none)

| body | method (source) | budget | k/n | rate | Wilson 95% | per training seed | failures |
|---|---|---|---|---|---|---|---|
| g1 | teacher (scripted_teacher) | - | 11/12 | 0.917 | [0.646, 0.985] | 0: 11/12 | {'fell': 1} |
| h1 | teacher (scripted_teacher) | - | 0/12 | 0.000 | [0.000, 0.242] | 0: 0/12 | {'dropped': 11, 'hold_lost': 1} |
| t1 | teacher (scripted_teacher) | - | 12/12 | 1.000 | [0.758, 1.000] | 0: 12/12 | - |
