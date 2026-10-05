# humanoid_teacher_quality_h_place (h_place)

coverage: {'dev/done': 3}

## dev: Reference: scripted / privileged teacher (not a learned method) (budget unit: none; reference: none)

| body | method (source) | budget | k/n | rate | Wilson 95% | per training seed | failures |
|---|---|---|---|---|---|---|---|
| g1 | teacher (scripted_teacher) | - | 20/20 | 1.000 | [0.839, 1.000] | 0: 20/20 | - |
| h1 | teacher (scripted_teacher) | - | 7/20 | 0.350 | [0.181, 0.567] | 0: 7/20 | {'dropped': 12, 'place_miss': 1} |
| t1 | teacher (scripted_teacher) | - | 20/20 | 1.000 | [0.839, 1.000] | 0: 20/20 | - |
