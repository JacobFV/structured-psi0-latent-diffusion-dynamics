# W6 gate backfill (report only)

| kind | subject | verdict | criteria (status: value) |
|---|---|---|---|
| tracker | anymal_c_installed | **pass** | slip_ratio_forward P: 0.0349; cot_forward P: 0.343; peak_foot_force_bw P: 2.61; joint_limit_margin P: 0.155; no_fall_rate P: 1; robust_in_training_range P: all conditions ok |
| tracker | g1_r1_installed | **fail** | slip_ratio_forward P: 0.0829; cot_forward P: 1.19; peak_foot_force_bw F: 4.61; joint_limit_margin P: 0.0349; no_fall_rate P: 1; robust_in_training_range P: all conditions ok |
| tracker | g1_src | **fail** | slip_ratio_forward F: 0.382; cot_forward F: 2.87; peak_foot_force_bw F: 9.75; joint_limit_margin F: -0.0885; no_fall_rate P: 1; robust_in_training_range P: all conditions ok |
| tracker | go2_installed | **pass** | slip_ratio_forward P: 0.0229; cot_forward P: 0.878; peak_foot_force_bw P: 2.24; joint_limit_margin P: 0.11; no_fall_rate P: 1; robust_in_training_range P: all conditions ok |
| tracker | t1_w8c | **fail** | slip_ratio_forward F: 0.152; cot_forward F: 2.36; peak_foot_force_bw F: 4.22; joint_limit_margin F: -0.0437; no_fall_rate P: 1; robust_in_training_range P: all conditions ok |
| tracker | t1_w8d_installed | **fail** | slip_ratio_forward P: 0.137; cot_forward F: 2.13; peak_foot_force_bw F: 4.46; joint_limit_margin F: -0.0533; no_fall_rate P: 1; robust_in_training_range P: all conditions ok |
| legged_dataset | w8_anymal_c_data | **pass** | slip_ok_fraction P: 1; falls_at_sigma0 P: 0 |
| arm_dataset | teacher_v1_grasp_v1 | **fail** | phase_switch_vel_step F: 0; joint_limit_margin F: 0.857; cmd_jerk_rms_vs_v2_teacher F: {'panda_pg2': 11.467, 'panda_tf3': 12.00; penetration L: 0.0025 |
| arm_dataset | teacher_v2_grasp_v1 | **fail** | phase_switch_vel_step P: 0.999; joint_limit_margin F: 0.871; cmd_jerk_rms_vs_v2_teacher P: {'panda_pg2': 0.999, 'panda_tf3': 1.001,; penetration L: 0.421 |
| arm_dataset | teacher_v2_grasp_v2 | **fail** | phase_switch_vel_step P: 0.999; joint_limit_margin F: 0.871; cmd_jerk_rms_vs_v2_teacher P: {'panda_pg2': 1.0, 'panda_tf3': 1.0, 'pa; penetration P: 0.995 |
