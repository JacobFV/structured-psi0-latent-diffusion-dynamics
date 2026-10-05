# humanoid_teacher_quality_h_loco_pick (h_loco_pick)

coverage: {'dev/done': 3}

## dev: Reference: scripted / privileged teacher (not a learned method) (budget unit: none; reference: none)

| body | method (source) | budget | k/n | rate | Wilson 95% | per training seed | failures |
|---|---|---|---|---|---|---|---|
| g1 | teacher (scripted_teacher) | - | 14/20 | 0.700 | [0.481, 0.855] | 0: 14/20 | {'fell': 1, 'no_grasp': 4, 'timeout': 1} |
| h1 | teacher (scripted_teacher) | - | 0/20 | 0.000 | [0.000, 0.161] | 0: 0/20 | {'dropped': 8, 'fell': 10, 'hold_lost': 2} |
| t1 | teacher (scripted_teacher) | - | 19/20 | 0.950 | [0.764, 0.991] | 0: 19/20 | {'fell': 1} |
