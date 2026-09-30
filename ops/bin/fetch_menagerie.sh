#!/usr/bin/env bash
# Pinned sparse fetch of audited menagerie directories (read-only reference assets).
set -euo pipefail
SHA=c96a32d28fb5da84da38c1da4d749e7a13212855
D=${RRP_MENAGERIE_DIR:-$(cd "$(dirname "$0")/../.." && pwd)/.cache/assets/mujoco_menagerie}
if [ ! -d "$D/.git" ]; then
  git clone --filter=blob:none --no-checkout --sparse https://github.com/google-deepmind/mujoco_menagerie "$D"
fi
cd "$D"
git sparse-checkout set unitree_g1 unitree_h1 pal_talos booster_t1 toddlerbot_2xc toddlerbot_2xm pndbotics_adam_lite apptronik_apollo berkeley_humanoid fourier_n1 robotis_op3 agility_cassie unitree_a1 unitree_go1 unitree_go2 anybotics_anymal_b anybotics_anymal_c google_barkour_v0 google_barkour_vb boston_dynamics_spot franka_emika_panda franka_fr3 universal_robots_ur5e unitree_z1 ufactory_lite6 rethink_robotics_sawyer ufactory_xarm7 trs_so_arm100 robotstudio_so101 aloha hello_robot_stretch hello_robot_stretch_3 pal_tiago pal_tiago_dual stanford_tidybot google_robot rainbow_robotics_rby1 wonik_allegro shadow_hand leap_hand shadow_dexee robotiq_2f85 robotiq_2f85_v4 umi_gripper sharpa_wave \
  kinova_gen3 kuka_iiwa_14 flexiv_rizon4 universal_robots_ur10e trossen_vx300s trossen_wx250s agilex_piper arx_l5 i2rt_yam
git checkout -q $SHA
git rev-parse HEAD
du -sh "$D"
