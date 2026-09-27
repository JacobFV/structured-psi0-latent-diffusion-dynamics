# track robust (W6): robustness sweeps + motion-quality gates

Owner: W6 robustness agent. Worktree `~/work/rrp-wt/robust`, branch `track/robust` (from origin/main b89ca70).
Peer code dir `/dev/shm/rrp-brandonin/wt/robust` (pushed once; peer CPU admission was full, so everything ran on the host).
Started 2026-09-27 ~09:40.

Question: where does each policy break as the physics departs from training, and is the semantic packet route more or
less robust than nosem and BC?

## what was built (state: verified; commits on track/robust, merged to main)
### `rrp.envs.perturb` (physics perturbations; evaluation-only, nothing enters an observation)
`PhysicsPerturbation` (frozen dataclass; `PhysicsPerturbation()` = nominal, every code path unchanged):
- model level, applied after the scenario is compiled (so the robot spec / morphology features the policy sees stay
  NOMINAL): `friction_scale` (all geom friction; under contact_v2 the floor has priority, so robot-floor mu = 0.9 x s),
  `mass_scale` (mass + inertia of every robot body), `com_offset_m` (legged: root/trunk; arm: last arm link),
  `kp_scale`/`kd_scale` (every position servo), arm `object_mass_scale` (cube) and `object_friction_scale` (cube AND the
  finger geoms, because MuJoCo takes the max of the pair: grasp mu = 1.5 x s). `mj_setConst` after mass/CoM changes.
- step level: `latency_ms` = legged: the existing actuator mode `v1lat` (`rrp.physics.actuator.ActuatorModel`, ideal
  joints + SOURCED torque/speed limits + fixed latency; nominal = ideal actuators, so the 0 ms level isolates the
  sourced speed envelope); arm: `ctrl_delay_v0` (FIFO delay of the robot ctrl by round(latency/dt) substeps; the arm has
  no actuator model). `push_impulse_Ns`: constant horizontal force J/0.1 s on the legged root (t = 4 s) or the last arm
  link (t = 3 s), direction drawn from the episode seed only (paired). `terrain_amp_m`: see below.
- hooks: `install_legged` replaces `LeggedSession._tracker_tick` by an instance-level copy with the same operations
  plus the hooks (used by every legged route: teacher tracker, system-0 adapter, BC adapter); `install_arm` wraps each
  controller's `apply_substep` (called right before every `mj_step` of `Session.step`).
### legged terrain flag (`rrp.bodies.legged.legged_world(..., terrain=)`, `build_waypoint_contact(..., terrain=)`)
`bumps_v1`: smoothed white noise (sigma 0.1 m), 0..amp m, heightfield named `floor` (so foot-contact / fall logic keep
working) over +-8 m, flat within 0.6 m of the spawn, plane `floor_outer` beyond. Default None = the flat plane, unchanged.
The pattern is drawn from the episode seed. Recorded in the model (`terrain_version` text) and in the scenario meta.
### `rrp.evaluation.motion_quality` (in EVERY eval row as `motion`; read-only recording)
- legged (`LeggedMotionRecorder`, attached in `legged_latent_eval.run_episode`): `slip_ratio` / `slip_cp_mps` (contact-point
  stance slip, loaded feet, over ticks with true base speed > 0.1 m/s), `cot` (sum |tau qdot| dt per substep / (m g path)),
  `joint_jerk_rms/peak` (measured policy joints, 50 Hz), `peak_contact_force_bw` (max foot touch sensor per substep / weight),
  `joint_limit_margin_min`.
- arm (`ArmMotionRecorder`, attached in `ladder.run_ladder`): `joint_jerk_rms/peak` (measured arm joints, 20 Hz),
  `chunk_vel_step_max` / `vel_step_any_max` (commanded joint-velocity step at chunk/packet boundaries +-1 tick / anywhere),
  `penetration_max_m` (cube-robot contacts, as teacher_quality), `joint_limit_margin_min`.
- Tests (`tests/unit/test_motion_quality.py`, known values): jerk of a cubic is exact; chunk-boundary velocity step 2 rad/s
  found at the boundary and not elsewhere; limit margin incl. unlimited joints; slip ratio / CoT; a 3 N s push on a 2 kg box
  gives 1.5 m/s; model scaling + nominal no-op; ctrl delay shifts by n substeps; terrain hfield is `floor` and bounded.
- Behaviour unchanged (checked against the pre-change code, same seeds): legged teacher x2, BC, semfix flow rows identical
  except the new `motion` field; arm route learned (BC) and generated rows identical.
### `rrp.evaluation.robustness` (harness; CLI `run` / `report` / `video`)
- `run --family legged|arm --robots R --route NAME=SPEC ... --seeds S [--factors] [--shard i/n] [--workers k]`.
  Routes: legged `teacher` | `bc:<ckpt>` | `flow:<ckpt>`; arm `teacher` | `learned:<ckpt>` | `generated:<flow>@<rep>`.
  Source labels come from the eval code (`scripted_teacher`, `bc:<ckpt>`, `learned:<ckpt>`, `learned(system-i flow)`).
- Grid (one factor at a time + `all_moderate` + nominal; same seeds at every level):
  legged friction 0.4/0.6/0.8/1.25/1.5, mass 0.8/0.9/1.1/1.2, com_x +-1/+-2 cm, com_y +1/+2 cm, kp 0.7/0.85/1.15/1.3,
  latency (v1lat) 0/10/20/30 ms, push dv 0.25/0.5/0.75/1/1.5 m/s (x total mass = N s), terrain 1/2/3/5/8 cm; all_moderate =
  friction 0.7, mass 1.1, com_x +1 cm, kp 0.85, 20 ms, push 0.5 m/s, terrain 2 cm.
  arm friction 0.4..1.5, mass 0.8..1.2, com_x +-1/+-2 cm, kp 0.7..1.3, ctrl delay 10/20/30 ms, push 2/5/10/20/40 N s,
  object mass x0.5/2/5/10/20, object friction x0.05/0.1/0.2/0.4/0.7; all_moderate = friction 0.8, mass 1.1, com +1 cm,
  kp 0.85, 20 ms, push 2 N s, object mass x2, object friction x0.6.
- Outputs `<out>/<route>/<robot>/<factor>=<level>.{jsonl,summary.json}` + `manifest.json` (seeds, checkpoint sha256);
  resumable (legged per episode, arm per shard: a batch shares the flow RNG, so a partial arm shard is redone whole).
- `report`: per level success k/n, Wilson 95% CI, falls, failure stages, task metrics, motion medians, paired lost/gained
  seeds vs nominal, BREAK-POINT = first level moving away from nominal (each side) with success > 20 points below nominal.
- `video --family --route --robot --condition --seed`: re-runs one sweep episode exactly and writes a captioned clip + INDEX line.
- The legged controller is loaded once per shard and its seed state reset per episode (rows verified identical to a
  fresh construction per seed; ~15 s/episode of checkpoint loading saved).

## findings on the way
- Arm R2 rows depend on the DIAGNOSTIC oracle comparison (`compare_oracle`, default True in scripts/ladder.py): its
  look-ahead rolls the real session forward and restores it, and the restore recomputes sensors with mj_forward (after
  mj_step, sensordata lags one substep), so the next public observation differs in the last bits and the chaotic rollout
  diverges (same 2 seeds: 114/171 steps, success/success with compare_oracle vs 114/300 success/timeout without). Not a
  privileged-information leak, but R2 episodes are not reproducible across that flag; statistics should be equivalent.
  The sweep runs every arm level with compare_oracle=False (step hooks are not part of a snapshot; `run_ladder` now refuses
  step-level perturbations when a look-ahead is active).
- Arm results depend on the batch composition (the flow/BC sampling RNG is shared by a 16-session batch): 2-seed runs do
  not reproduce the 30-seed run's episodes. The harness always uses the ladder's batching (16) on the same seed list.
- Host: the memory-PSI watchdog shed/refused jobs repeatedly (external swap pressure); a process at 3 GB RSS ran at ~4% CPU
  during one such episode. Launcher `scripts/robust_host_run.sh` retries (bounded) and resumes.

## SWEEP RESULTS (2026-09-27 10:00-11:19, host; state: completed)
Raw rows: `artifacts/runs/robust/{legged_anymal_c,arm}/<route>/<robot>/<condition>.jsonl` (local + peer store
`/dev/shm/rrp-brandonin/repo/artifacts/runs/robust/`); per-shard summaries, `manifest.json` (seeds, checkpoint sha256) and
`robustness_report.{md,json}` (full per-level tables with Wilson CIs, motion medians, paired lost/gained) are in git under
`artifacts/runs/robust/`. Commands: `scripts/robust_host_run.sh robust_leg$i 1 3G 12 -- --family legged --robots anymal_c
--seeds 10000-10019 --route semfix=flow:... --route nosem=flow:... --route bc=bc:... --route teacher=teacher --out
artifacts/runs/robust/legged_anymal_c --shard $i/4` (i = 0..3) and `... robust_arm$i 1 6G 20 -- --family arm --robots
panda_pg2,parm6_tf3 --seeds 3000000+20 --route frozen_sem=generated:<flow>@<rep> --route bc=learned:<ckpt> --route
teacher=teacher --out artifacts/runs/robust/arm --shard $i/2`; then `python -m rrp.evaluation.robustness report --out <dir>`.
Sources / checkpoints (sha256 prefix):
- legged (contact_v2, gait learned_tracker:anymal_c:iter2499:contact_v2 sha 2a16532b, ideal actuators at nominal):
  semfix = learned:legged8-anymal_c-semfix/train_flow_s0/policy.pt (cf69264b; Stage A 966bda92), nosem =
  learned:legged8-anymal_c-nosem/train_flow_s0/policy.pt (b49dd41f; Stage A 0e55feba), bc = bc:legged8-anymal_c-v2data/
  train_bc_s0/policy.pt (fe5ff485), teacher = scripted_teacher (privileged). Dev seeds 10000-10019 (W8's dev set).
- arm: frozen_sem = learned(system-i flow) ladder_flow_jointfix_gdag2h/policy.pt (d0d64918) -> system 0
  ladder_rz_jointfix_gendag3_noqd/representation.pt (f60cde41) (D-078 frozen route; Stage A ladder_latent_sem_b1fix_anchor
  49b2e2e3); bc = learned:direct1701_u12000 (5e6586bd; plain BC FlowPolicy); teacher = scripted_teacher(privileged) (v1
  PickPlaceTeacher, the ladder R0). First 20 feasible dev seeds from 3,000,000 (panda 3000000-19, parm6 3000003-27),
  batches of 16 as scripts/ladder.py; compare_oracle=False at every level (see findings).
Success = privileged end check (as W8 / ladder tables); legged public (task-runtime) success in parentheses.

### legged anymal_c (20 seeds per level; 33 single-factor levels + all_moderate)
| factor | level | bc | nosem | semfix | teacher |
|---|---|---|---|---|---|
| nominal | - | 20 (20) | 20 (20) | 17 (20) | 20 (20) |
| friction (mu = 0.9 x) | 0.4 / 0.6 / 1.5 | **0** f14 / 18 / 19 | **0** / 20 / 16 | **0** f1 / 19 / 15 | 20 / 20 / 20 |
| mass | 1.1 / 1.2 | **15** / **0** f3 | 20 / 20 | 18 (19) / 17 (16) | 19 / 20 |
| kp | 0.7 / 0.85 | **0** f9 / **0** f2 | **2** f17 / 20 | **0** f15 / 15 (14) | **14** f3 / 20 |
| latency v1lat | 0 / 10 / 20 / 30 ms | 20/19/20/17 | 19/19/20/18 | 19/18/16/17 | 20/19/20/20 |
| push dv (x 44.97 kg = N s) | 0.75 / 1.0 / 1.5 m/s | 17 / **11** f8 / **2** f18 | 18 / **9** f10 / **2** f17 | 13 (15) / **7** f13 / **2** f18 | 19 / 18 / **10** f9 |
| terrain bumps | 5 / 8 cm | **8** / **0** | 20 / **7** | 18 / **1** f9 | 20 / **13** |
| com_x, com_y +-2 cm | all | 16-20 | 18-20 | 15-17 (public 20) | 19-20 |
| all_moderate | - | **0** | **10** (13) f4 | **3** (1) f6 | 20 |
| pooled 660 perturbed | - | 488 (public 511) | 550 (573) | 491 (552) | 627 (636) |
Break-points (first level > 20 points below the route's own nominal): friction low 0.4 for all three learned routes
(teacher none: the tracker was trained on mu 0.4-1.25); mass high: bc 1.1, others none; kp low: bc 0.85, nosem/semfix 0.7
(semfix public 0.85), teacher 0.7; push: learned 1.0 m/s (semfix public 0.75), teacher 1.5; terrain: bc 5 cm, nosem/semfix
8 cm, teacher 8 cm; latency 0-30 ms (v1lat) and CoM +-2 cm: none for any route; all_moderate: every learned route breaks,
the teacher does not.
Paired over seeds (`robustness_report.md` "paired route comparison"; seeds are the unit, 33 levels pooled per seed,
sign-flip permutation p, seed-bootstrap CI), PUBLIC success (removes the D-105 end-check flicker artifact that costs
semfix 3/20 at nominal):
- semfix vs nosem: level mean 0.836 vs 0.868, diff -0.032 [-0.048, -0.017], p = 0.002 (privileged: -0.089, p < 1e-4).
- semfix vs bc: +0.062 [+0.042, +0.080], p < 1e-4; nosem vs bc: +0.094 [+0.077, +0.111], p < 1e-4.
- every learned route is far below the teacher (semfix -0.127, bc -0.189).
Where semfix loses to nosem: kp 0.85 (14 vs 20 public), mass 1.2 (16 vs 20), terrain 8 cm (1 vs 6), push 0.75 (15 vs 19),
all_moderate (1 vs 13). Nowhere is semfix clearly more robust than nosem.
Failure modes: low friction (mu 0.36) -> learned routes stall before waypoint a (drift_a, CoT 1.9-3.9 vs 0.37, slip ratio
0.12-1.0; BC falls 14/20) while the teacher's tracker walks (slip 0.08); weak servos / pushes / 8 cm terrain -> falls.
Pushes: the applied impulse is 98% of nominal (49 of 50 substeps inside the window; recorded per row).
Motion quality at nominal (medians): slip ratio 0.045-0.051, CoT 0.35-0.39, joint jerk RMS 1.8-2.2e3 rad/s^3 (50 Hz
measured joints), peak foot force 2.5-2.9 body weights, limit margin 0.28 for all four routes (the learned routes move
like the teacher at nominal); nosem has the highest jerk (2194) and peak force (2.86).

### arm (20 seeds per level; 35 single-factor levels + all_moderate)
| robot | route | nominal | pooled 700 perturbed | lost / gained vs nominal | break-points |
|---|---|---|---|---|---|
| panda_pg2 | bc | 17 | 548 (78%) | 80 / 33 | push 40 N s (10/20), object friction x0.05 (12/20) |
| panda_pg2 | frozen_sem | 9 | 256 (37%) | 122 / 63 | push 40 N s (3/20) |
| panda_pg2 | teacher | 20 | 698 | 2 / 0 | none |
| parm6_tf3 | bc | 19 | 633 (90%) | 49 / 17 | push 40 N s (8/20) |
| parm6_tf3 | frozen_sem | 12 | 462 (66%) | 74 / 116 | object mass x10 (5/20), x20 (3/20) |
| parm6_tf3 | teacher | 20 | 681 | 19 / 0 | push 40 N s (4/20, f10) |
Paired over seeds: frozen vs bc level mean -0.417 [-0.59, -0.21] (panda), -0.244 [-0.39, -0.11] (parm6), p <= 0.002;
drop relative to own nominal: +0.017 (p 0.86) and -0.106 (p 0.23): NO detectable robustness difference beyond the
frozen route's lower competence. The arm is insensitive to most factors in these ranges (friction 0.4-1.5, mass/CoM,
kp 0.7-1.3, 10-30 ms ctrl delay, object friction down to x0.05 = grasp mu 0.075): stiff position servos, and the grasp
does not depend on friction (penetration holding: max cube-finger penetration 5-7 mm panda / 17-20 mm three-finger parm6,
the known D-093 checklist item). Only large pushes (40 N s on the last link) and, for the frozen route on parm6, heavy
objects (x10 = 0.43 kg: 5/20 vs bc 20/20, failing at transport) break a learned route.
Arm motion quality at nominal (medians): joint jerk RMS panda bc 36.8 / frozen 23.4 / teacher 20.7 rad/s^3, parm6 87 /
80 / 59; chunk-boundary velocity step bc 3.3 / 2.2 rad/s, frozen 2.2 / 2.0 (always the episode maximum: the largest steps
are at chunk/packet seams, as D-102 found for BC); teacher (v1) 1.3 / 0.9 at phase switches; limit margin 0.03-0.09.
Caveat: the frozen route's success fluctuates +-3/20 between physically near-identical levels (parm6 gains 116 seeds vs
loses 74 relative to its nominal 12/20), so with n = 20 a 20-point break criterion is only reliable for large effects.

### answer
Legged (anymal_c, contact_v2): the semantic packet route (semfix) is NOT more robust than nosem; it is slightly less
robust (public success -3.2 points pooled over 33 perturbed levels, p = 0.002; all_moderate 1/20 vs 13/20), and both packet
routes are more robust than plain BC (+6 / +9 points, p < 1e-4; BC breaks at +10% mass, kp 0.85 and 5 cm terrain where the
packet routes do not). All learned routes share the same break-points on friction (mu 0.36), pushes (1 m/s) and weak
servos (kp 0.7), well inside the teacher's envelope; none breaks under 0-30 ms latency or +-2 cm CoM.
Arm: the frozen semantic route is less competent than BC at every level but not measurably less robust relative to its
own nominal; the arm grid mostly probes a regime where the simulated grasp is insensitive (penetration holding).
