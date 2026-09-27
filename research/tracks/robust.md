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

### checks
- compare_oracle neutrality (`artifacts/runs/robust/arm_compare_oracle_check.json`): frozen route nominal on the sweep's 20
  seeds with compare_oracle True vs False: parm6 13 vs 12, panda 10 vs 9 (5 discordant seeds each: chaotic divergence, same
  rate). The False run reproduces the sweep's nominal shards exactly (12, 9). The sweep's parm6 nominal 12/20 vs the
  recorded 24/30 dev (D-078) is seed-set / n = 20 noise, not the flag.
- Every video below re-ran its sweep episode and reproduced the recorded outcome.

### videos (artifacts/video/INDEX.md)
- `2026-09-27_robust_semfix_anymal_c_push_dv1_s10001_fell.mp4`: semfix at its push break-point (dv 1.0 m/s = 45 N s at
  t = 4 s) falls; nosem succeeds on the same seed (`..._nosem_..._success.mp4`, 5.7 MB, kept only in the peer store
  `artifacts/runs/robust/video/`).
- `2026-09-27_robust_frozen_sem_parm6_tf3_object_mass10_s3000003_failure-transport.mp4`: frozen arm route with a x10 cube
  drops it in transport (plain BC succeeds on this seed).

## state
| step | state | evidence |
|---|---|---|
| harness, perturbations, terrain flag, motion metrics + tests | verified | commits dbb95a0..; tests/unit 364 passed / 1 skipped (host lease 1790530264_01ccc2); nominal-identity checks above |
| legged anymal_c sweep (4 routes x 35 conditions x 20 seeds) | completed | artifacts/runs/robust/legged_anymal_c/ (report + summaries in git; rows local + peer store) |
| arm sweep (3 routes x 2 bodies x 37 conditions x 20 seeds) | completed | artifacts/runs/robust/arm/ |
| failure videos | completed | artifacts/video/INDEX.md (2 clips) |

## next (not done)
- Nominal-condition trackers/datasets "must pass the gates" (W6 scope item 3): gate thresholds are not yet defined; the
  motion metrics are now in every eval row, so a gate can be a pure function of the rows (proposal: legged slip ratio < 0.15,
  CoT < 2x teacher, no joint-limit violation (margin >= 0); arm chunk-boundary velocity step < 1.5 rad/s, penetration < 5 mm).
- Arm grid does not reach the grasp's physical limits because the simulated grasp holds by penetration; a friction-dependent
  grasp needs contact tuning (softer solref/solimp with lower penetration) before arm friction sweeps mean anything.
- n = 20 resolves only large effects; semfix vs nosem on anymal_c would need seeds 1/2 (W8 trained 3 seeds) to generalize.

## resume
`cd ~/work/rrp-wt/robust`; rerun/extend with `scripts/robust_peer_run.sh LABEL CPU MEM MAX_ATTEMPTS -- <run args>` (PEER ONLY since D-115; the host launcher was removed) (finished
shards are skipped; arm leases need --mem 6G: at 3-4G the lease's memory.high throttles the 3.7 GB worker to ~5% CPU);
rebuild tables with `python -m rrp.evaluation.robustness report --out artifacts/runs/robust/<dir>`. Videos render on the
peer (the host venv has no imageio/PIL): `RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/robust scripts/peer_run.sh --cpu 1 --mem 5G
--label robust_video -- env MUJOCO_GL=egl PY -m rrp.evaluation.robustness video --family ... --condition KEY --seed S`.

## VARIANT-LEVEL SEMFIX vs NOSEM (lead request after D-108; training seeds 0/1/2; state: completed)
Seeds 1/2 sweep: same grid, eval seeds 10000-10019, contact_v2, same host code. The host memory-PSI watchdog shed or refused
every attempt from 11:49 on (9 attempts per lease), so all four shards were moved to the PEER (CPU leases 1 CPU / 2.5 GiB,
measured peak 1.8-1.9 GB; leases 1790535604_8730ee, 1790535605_4aa83f, 1790536247_3174c4, 1790536247_a80e99, all rc 0).
Episodes finished on the host before the move were copied, not rerun (episode-level resume; per-seed controllers
are seeded, so the placement does not change outcomes). Checkpoints (sha256 prefix): semfix s1 bd0b6abe (Stage A 77c3b286),
s2 5aca8200 (22bd361a); nosem s1 7ada18a2 (a065d7b2), s2 f891f74c (5d15ad6a). Raw: `artifacts/runs/robust/legged_anymal_c_s12/`
(rows local + peer store); comparison: `python -m rrp.evaluation.robustness variant --robot anymal_c --a semfix --b nosem
--spec semfix:0=artifacts/runs/robust/legged_anymal_c/semfix --spec nosem:0=.../nosem --spec semfix:1=.../legged_anymal_c_s12/semfix_s1
... ` -> `artifacts/runs/robust/variant_anymal_c_semfix_vs_nosem.{md,json}`.
Per eval seed: level mean = mean success over the 33 perturbed single-factor levels; drop = nominal - level mean.

| training seed | nominal semfix / nosem (public) | level mean, public: semfix / nosem, diff [95% CI] p | privileged diff [95% CI] p | all_moderate public |
|---|---|---|---|---|
| 0 | 17 (20) / 20 (20) | 0.836 / 0.868, -0.032 [-0.048, -0.017] p=0.002 | -0.089 [-0.115, -0.064] p<1e-4 | 1 / 13 |
| 1 | 18 (20) / 19 (20) | 0.832 / 0.894, -0.062 [-0.077, -0.047] p<1e-4 | -0.076 [-0.108, -0.042] p=6e-4 | 6 / 14 |
| 2 | 18 (20) / 18 (20) | 0.830 / 0.874, -0.044 [-0.065, -0.023] p=0.001 | -0.036 [-0.077, +0.006] p=0.13 | 7 / 14 |
| pooled (per eval seed, averaged over training seeds) | - | -0.046 [-0.056, -0.036] p<1e-4 | -0.067 [-0.090, -0.045] p<1e-4 | 14 / 41 of 60 |
Variant level (3 vs 3 training-seed means, exact permutation over the 20 relabellings, as D-090): public level mean
semfix 0.836 / 0.832 / 0.830 vs nosem 0.868 / 0.894 / 0.874: every semfix seed below every nosem seed, one-sided p = 0.05
(the minimum possible), two-sided 0.10. Privileged: 0.744 / 0.770 / 0.786 vs 0.833 / 0.845 / 0.823, also fully separated,
p = 0.05 / 0.10. Drop relative to own nominal (public): semfix 0.164 / 0.168 / 0.170 vs nosem 0.132 / 0.106 / 0.126, fully
separated (p = 0.05); privileged drop does not separate (+0.001, p = 1.0) because semfix's privileged nominal is lowered by
the end-check flicker.
Break-points are the SAME for both variants on every seed (friction 0.4, kp 0.7, push 1.0 m/s, terrain 8 cm; exceptions
semfix s2 terrain 5 cm, nosem s2 push 0.75 m/s); none under latency 0-30 ms, mass +-20%, CoM +-2 cm. The difference is a
consistent few-point loss spread over levels, largest under combined moderate perturbations (all_moderate public 14/60 vs
41/60).
READING: at variant level (3 training seeds), semantic packet supervision (semfix) makes the anymal_c R2 route slightly LESS
robust to physics shifts than the capacity-matched nosem packet, by 4.6 points of public success pooled over 33 perturbed
levels (-6.7 privileged), replicated on every training seed (exact p = 0.05, the floor of a 3-vs-3 test). It does not move
any break-point. Together with D-105 (semfix has the stronger context-halt and goal-steering effects), the semantic packet
buys controllability at a small robustness cost on this body. Both packet routes remain more robust than BC (seed 0).

## PROPOSED GATES for trackers and datasets (for lead review; not enforced yet)
All quantities are the `motion` fields now in every eval row (rrp.evaluation.motion_quality) or the robustness report;
"nominal" = the deployed physics (contact_v2, ideal actuators, sourced limits). Numbers in brackets are what we measure now.

| applies to | metric | threshold | rationale / current values |
|---|---|---|---|
| legged tracker (validation forward trial) | stance slip ratio | < 0.15 | the D-093 contact gate, kept. [anymal_c 0.03, go2 0.02, t1 0.10-0.15] |
| legged tracker | cost of transport (forward, nominal) | quadruped <= 1.0, biped <= 2.0 | 2-3x the best accepted tracker per family; catches energy-wasting gaits without over-constraining. [anymal_c 0.35, go2 0.88, t1 1.4-1.8] |
| legged tracker | peak foot force (median over episodes) | <= 3.5 body weights | stomping / impact exploitation; accepted gaits sit at 2.5-2.9 in the waypoint task |
| legged tracker | joint-limit margin (min over the episode) | >= 0.02 of range | a gait that parks joints on their stops relies on the limit as a mechanical support. [0.28 nominal; learned routes reach -0.005 only under mu 0.36] |
| legged tracker | robustness inside the TRAINING randomization | no break-point at friction 0.6 / 1.25, mass 0.9 / 1.1, v1lat 0-20 ms, push dv 0.5 m/s; no-fall >= 0.9 at each | a tracker must be robust where it was trained to be; outside-range levels (mu 0.36, kp 0.7, dv 1.5) are reported, not gated. [anymal_c teacher route: passes] |
| legged dataset (teacher/tracker rollouts) | episode slip ratio / falls / limit margin | slip < 0.15 on >= 95% of episodes, 0 falls at DART sigma 0, margin >= 0 | data inherit the tracker gait; the per-episode check catches noise-induced exceptions (the D-105 stance flicker is an end-check artifact, not a gait defect, and is excluded) |
| arm teacher / dataset | commanded velocity step at phase switches | <= 0.5 rad/s | v2 teacher 0.18-0.28 passes, v1 0.9-1.4 fails (D-097): the step the policy must reproduce |
| arm teacher / dataset | commanded joint jerk RMS | <= 2x the v2 teacher of the same body | body-relative because link scales differ (v2: 36-55 rad/s^3 commanded) |
| arm teacher / dataset | max cube-finger penetration | <= 10 mm | grasps held by interpenetration are a simulator exploit (v1 three-finger 18-24 mm fails, v2 6-7 mm passes) |
| arm teacher / dataset | joint-limit margin | >= 0.02 | as legged. [nominal 0.03-0.09: parm6 is close] |
| learned policies (reported, not gated) | chunk-boundary velocity step, jerk, penetration, all robustness break-points | flag chunk step > 1.5 rad/s | a quality report for every eval, not an acceptance gate; BC 2.2-3.3, frozen route 2.0-2.2 rad/s (D-102 chunk-seam issue) |
Open point: the arm penetration gate currently FAILS the learned-route rollouts on parm6 (17-20 mm at nominal, same as the v1
teacher its data came from); applying it to v2-data policies only is consistent with D-097/D-102.

## GATES AS CODE (lead request after D-112; state: verified) + D-111 follow-up
### code
- `rrp.evaluation.gates` (thresholds in `GATES`, D-112 as adopted, penetration <= 3 mm gated for grasp_v2 only):
  `check_tracker(validation)`, `check_dataset(manifest, episodes)` -> `check_legged_dataset` / `check_arm_dataset`,
  `policy_flags(rows)` (reported, never gated). Report = {gate, verdict pass|fail|incomplete, criteria [{name, status
  pass|fail|not_evaluated|labelled, value, threshold, note}], failed}. Only "fail" fails a node; "incomplete" (a measurement
  missing) is recorded. Arm jerk reference = v2 teacher median joint_cmd_jerk_rms per body (grasp_v2, 300 seeds; embedded
  `ARM_TEACHER_V2_REFERENCE`, source sha 8083c464).
  Where a criterion needed interpretation: legged dataset slip uses episodes with a slip measurement (motion.slip_ratio);
  arm step / margin are ">= 95% of episodes", penetration ">= 99% of episodes <= 3 mm" (D-110 had 0 episodes > 3 mm).
- Measurements added (read-only; data byte-identical, verified old vs new code on 2 legged (sigma 0 / 0.2) and 2 arm
  (v1 noise 0, v2 DART 0.05) episodes: arrays, actions and inputs hash-equal):
  - `rrp.data.legged_latent_collect`: every episode meta has `motion` (LeggedMotionRecorder via install_legged).
  - `rrp.data.collect.collect_teacher_episode` (arm): `motion` with the CLEAN teacher command (the label, also in DART
    episodes): `phase_switch_vel_step_max`, `cmd_jerk_rms`, measured jerk, limit margin, penetration, grasp_contact_version.
    Equal to rrp.evaluation.teacher_quality on the same episode (3 panda seeds, v2, grasp_v2: step / jerk / penetration identical).
  - `rrp.evaluation.tracker_validation`: per-trial `peak_force_bw` (max per physics substep of the per-foot contact normal
    force / weight) and `joint_limit_margin_min`; `--robust` = forward trial under the tracker's OWN training randomization
    (floor mu 0.45 / 1.2, root mass x0.9 / x1.1, v1lat latency at the trained maximum (8 ms for contact_v2 randomization, 30 ms if
    trained with --actuator v1lat/v2), lateral kick 0.4 m/s (biped 0.2) at t = 4 s); output `w6_gate`; `--gate-dir`,
    `--gate-exit` (exit 86 on fail); `--freeze` records the W6 verdict next to the (unchanged) eligibility.
  - The recorders moved to `rrp.envs.motion_quality` (data collectors sit below evaluation in the layer order);
    `rrp.evaluation.motion_quality` re-exports.
- Pipelines: legged `collect` and arm `collect` compute the dataset gate from the collected metas / manifest, write
  `<out>/gate_report.json` and raise `GateFailed` on fail (option `gate: report` = record only); new legged stage
  `validate_tracker` (options body, actor, kind, seeds, robust) runs the validation + tracker gate. `python -m rrp.pipelines run`
  exits 86 on GateFailed; a stale gate report is removed at stage start.
- run-dag: a node that exits non-zero with a `gate_report.json` verdict fail is marked FAILED at once (no retries) with
  last_error "gate <name> failed: <criteria>"; dependants are blocked as usual.
- D-111 follow-up: the watchdog config `subtract_shmem` (on for the peer role in `rrp ops watchdog`): the startup term of the
  live memory limit becomes startup - meminfo Shmem (the RAM artifact store). MemAvailable already excludes Shmem, so the
  dynamic terms are unchanged. Fake-sample test: 108 GiB cap, 48 GiB Shmem -> 60 GiB; clamps at 0; unknown Shmem -> unchanged.
  Takes effect when the peer watchdog restarts (`rrp ops start-watchdog` on the peer; the lead's call, it is shared ops).
- Tests: tests/unit/test_gates.py (each criterion pass/fail/incomplete/labelled with synthetic inputs, teacher_quality rows,
  policy flags, apply_gate report-only mode, run-dag gated failure without retry), test_watchdog.py (+1).

### backfill (report only; `artifacts/runs/robust/gates/`, summary `SUMMARY.md`)
Trackers: `tracker_validation --contact v2 --seeds 5 --robust` on the actor files below (host lease 1790542233_95f525).
| tracker (sha256 prefix) | verdict | slip | CoT | peak force BW | limit margin | robust (in training range) |
|---|---|---|---|---|---|---|
| anymal_c installed (2a16532b) | PASS | 0.035 | 0.34 | 2.61 | 0.155 | ok |
| go2 installed (af3f06f4) | PASS | 0.023 | 0.88 | 2.24 | 0.110 | ok |
| t1 installed w8d (36e91467) | FAIL | 0.137 | **2.13** | **4.46** | **-0.053** | ok |
| t1 w8c (863d2469) | FAIL | **0.152** | **2.36** | **4.22** | **-0.044** | ok |
| g1 installed r1 (8b8a99cb; LEGACY limits, loaded with RRP_ACTUATOR_LIMITS=legacy_gains_v0) | FAIL | 0.083 | 1.19 | **4.61** | 0.035 | ok |
| g1 sourced-limit candidate contact_g1_src (f0a4daaa) | FAIL | **0.382** | **2.87** | **9.75** | **-0.089** | ok (fwd 0.5-0.7 everywhere) |
Dataset: W8 anymal_c `legged8-anymal_c-v2data/collect_s0` (600 episodes) replayed with the recorder (same seeds/sigmas/
teacher draws; `scripts/robust_backfill_w8data.py`, peer lease 1790544017_c48ba9): 600/600 episodes reproduce status, steps
and ticks with the declared tracker sha 2a16532b. Gate PASS: slip < 0.15 in 600/600 (median 0.058, max 0.133); 0 falls in
the 150 sigma-0 episodes. Raw `w8_anymal_c_data/replay.jsonl`.
Arm (extra, from existing teacher_quality rows, noise 0, 18 bodies x 300 seeds): v2 teacher grasp_v2 FAILS only on joint-limit
margin (87% of episodes >= 0.02; step 99.9%, jerk ratio 1.0, penetration 99.5% <= 3 mm); v2 grasp_v1 the same with penetration
labelled; v1 teacher fails step (0% <= 0.5 rad/s), jerk (9.5-14x) and margin.
Merge note: W8 (d83293a, 30f8416) had meanwhile added the same recorder to the legged collector and a report-only
`dataset_gate` metric plus its own bit-exact replay (anymal_c, go2 pass; t1 w8d fails slip). On rebase the collector keeps ONE
recorder (W8's placement; import moved to rrp.envs.motion_quality for the layer order), the collect metrics keep W8's
`dataset_gate` field and add the enforced `gate` (rrp.evaluation.gates). My anymal_c replay agrees with W8's (pass).
Unit suite after the rebase: 379 passed, 2 skipped (peer lease 1790544821_524322; the host refused admission).
### findings for the lead
1. Every humanoid tracker fails peak foot force (4.2-9.7 BW, validation trials at per-substep resolution). The 3.5 BW threshold
   was calibrated on quadruped waypoint rollouts; bipeds land on one foot. Suggest a biped threshold (e.g. <= 5 BW) or an
   impact-rate metric; the quadrupeds pass at 2.2-2.6.
2. The t1 trackers (w8c, installed w8d) and the g1 candidate drive joints PAST their ranges (margin -0.04 to -0.09: soft joint
   limits are penetrated), and their CoT exceeds the biped bound of 2.0. Robustness inside their training range is fine.
3. Arm joint-limit margin fails for the procedural arms parm5/5l/6/7 (13-36% of v2-teacher episodes touch a limit, margin ~0),
   not for panda/sawyer/ur5e/xarm7. Either the procedural joint ranges are too tight for the task or the threshold should be a
   "no limit contact" rule for these bodies; with the gate as adopted, v5dart data on parm bodies would FAIL the collect node
   (use `gate: report` until decided).

## D-112 FOLLOW-UPS (lead decisions on the four points + watchdog hysteresis; state: verified)
1. W8 t1: lead set `gate: report` (D-113).
2. Foot force = peak of the 20 ms moving-average per-foot contact normal force (tracker_validation `peak_force_bw`; the
   unfiltered per-step peak stays as `peak_force_raw_bw`; LeggedMotionRecorder adds `peak_contact_force_20ms_bw`).
   Thresholds quadruped <= 3.5, biped <= 3.0 BW. Backfill rerun (peer lease 1790545424_0d9ae6, `scripts/robust_backfill_trackers.sh`):
   | tracker | filtered peak BW (max trial) | raw | verdict (other failures) |
   |---|---|---|---|
   | anymal_c | 1.04 | 2.61 | PASS |
   | go2 | 1.54 (push) | 2.24 | PASS |
   | g1 r1 installed (legacy limits) | 2.16 | 4.61 | PASS (now) |
   | t1 w8d installed | 2.74 (turn_fast; stand 2.03) | 4.46 | FAIL: CoT 2.13, joint margin -0.053 |
   | t1 w8c | 2.75 (turn_fast; stand 2.28) | 4.22 | FAIL: slip 0.152, CoT 2.36, margin -0.044 |
   | g1_src candidate | **4.09** (turn_fast; turn 3.70) | 9.75 | FAIL: + slip 0.38, CoT 2.87, margin -0.089 |
   So the humanoid failures were impact transients except g1_src, which STOMPS in turns (3.7-4.1 BW filtered): a real signal
   for W1. Side observation: t1 loads one foot with 2.0-2.3 BW (filtered) even in the `stand` trial (weight shifting while
   standing).
3. Permanent joint-limit-margin hinge added to the gait_v2 reward (`RewardCfg.limit_margin` = -1.0, `limit_margin_penalty`,
   m0 = 0.02; details and the W1 note in research/tracks/contact.md); test in test_reward_schedule.py. Installed t1/g1 labelled.
4. Arm joint margin: enforced on menagerie arms, reported (`joint_limit_margin_procedural`, status labelled) on parm*.
   Re-gated teacher rows: v2 teacher PASSES under grasp_v2 (margin 99.9% on menagerie arms; parm* 74% labelled) and under grasp_v1
   (penetration labelled); v1 teacher fails (steps, jerk). W7 backlog item written in research/tracks/armexpert.md.
5. Watchdog hysteresis: when the project exceeds the live limit ONLY because of the RAM-store (Shmem) term, the watchdog stops
   admission at once and sheds only after the excess has persisted together with memory pressure (PSI full avg10 >= 10, or
   MemAvailable < 2x reserve) for 30 s (15 consecutive 2 s samples; `shmem_shed_grace_s`, `shmem_shed_psi`,
   `sample_interval_s` = the watchdog's --interval). An excess that exists without the store term sheds immediately, and the
   reserve emergency is unchanged. Test: test_watchdog.py::test_ram_store_excess_sheds_only_after_sustained_pressure (calm
   40 samples: never shed; PSI 43: shed at sample 15; a calm sample resets). Needs a peer watchdog restart to take effect.

## D-115 (user rule: no training or heavy compute on the host; peer only)
`scripts/robust_host_run.sh` is REMOVED; `scripts/robust_peer_run.sh LABEL CPU MEM MAX_ATTEMPTS -- <robustness run args>` places
every sweep lease on the peer (requires RRP_PEER_REPO; bounded retries; outputs in the shared peer store, fetch with rsync).
`scripts/robust_backfill_trackers.sh` refuses to run unless RRP_NODE=peer (set by scripts/peer_run.sh);
`scripts/robust_backfill_w8data.py` documents peer-only launch. Host use by W6 is limited to git, unit tests of pure code
and report building from saved rows. (The worktree's configs/resources.local.json carries the lead's D-115 host cap,
2 CPU / 0 GPU, uncommitted as the lead applied it.)
