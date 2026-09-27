# track: armexpert (W7 step 1: smooth scripted arm teacher)

Owner: arm-expert agent. Worktree `~/work/rrp-wt/armexpert`, branch `track/armexpert`, peer dir
`/dev/shm/rrp-brandonin/wt/armexpert`. Scope: docs/strategy.md W7 step 1 (smoothing) only; GRPO is later and needs lead
approval. No arm dataset was regenerated (lead decision, see "BC expert from v2" below).

State: **verified** (code + tests + old-vs-new report on identical seeds + videos). v1 stays the DEFAULT teacher everywhere.

## what was measured
`python -m rrp.evaluation.teacher_quality --bodies <18 bodies> --seeds 0-199,3000000-3000099 --versions v1|v2`
(one pick_place episode per body x seed, n_distractors = seed % 3, max 600 ticks = 30 s, privileged evaluator).
Seeds 0-199 are the v3 dataset seeds; 3,000,000-3,000,099 cover the ladder dev/fresh seeds. Bodies: the 13 source-train
bodies, the held-out source bodies parm5s_tf3 / parm5l_pg2, and the target-demo bodies panda_tf3, xarm7_pg2, xarm7_tf3.
The v1 feasibility check (which defines the ladder's "first 30 feasible seeds") is applied to BOTH versions, so the
seed sets are identical. Per episode: success + failure stage; commanded and measured joint velocity / acceleration /
jerk (finite differences at the 20 Hz control rate), measured and commanded (FK) TCP acceleration / jerk, the largest
joint-velocity step at phase switches (+-1 tick) and anywhere, joint-limit margin (fraction of range), time to done,
gripper timing (close command -> first held tick, lift before held, held lost before open), max object<->hand
penetration after the approach, object slip in the TCP frame while carried, IK residuals per phase.
All metrics are privileged diagnostics; nothing here feeds a policy.

## diagnosis of v1 (scripted_teacher:pick_place_v1_waypoint)
Raw: `artifacts/runs/armexpert_diag/v1_18bodies.jsonl.gz` (+ `.summary.json`), peer lease 1790475957_8cb8b6 (rc 0).
v1 succeeds on 4703/4811 feasible episodes. ALL 108 failures are three-finger (tf3) bodies; every parallel-jaw (pg2)
body is 100 %.

Motion (all bodies): v1 moves the TCP command at a constant 0.35 m/s toward each waypoint and stops dead on arrival, then
switches phase; the gripper command jumps open<->closed in one tick; the integral "ioff" correction adds step offsets.
Result: joint-velocity steps of ~1.0-1.4 rad/s within one 50 ms tick at phase switches (commanded acceleration peaks
20-30 rad/s^2), commanded joint jerk peaks 390-625 rad/s^3 (medians per body), measured joint jerk 60-1060 rad/s^3
(the light procedural arms ring the most: 820-1060). That is the "mechanical, jerky" look.

Failure causes (per body, v1):
| body | v1 success | cause (verified by per-tick traces: `research/scripts/2026-09-26/armexpert_{trace,contacts}.py`) |
|---|---|---|
| ur5e_tf3 | 262/300 | over-squeeze: closed target = geometric contact angle + 0.1 rad with kp 800 saturates the 8 N m finger torque (~145 N per fingertip on a 42 g cube); fingers sink in (median penetration 5 cm in failures vs 2 cm in successes), the three non-antipodal contacts squeeze the cube down the fingers; it dangles on one finger during transport and lands outside the zone (38 x "dropped") |
| xarm7_tf3 | 273/300 | same over-squeeze / slip (27); plus the fixed 0.7 s close timer lifts before the grasp has formed in 190/300 episodes (close -> held takes 0.85 s on this arm) |
| panda_tf3 | 280/300 | same (16 dropped, 4 placed outside the zone); lift before held in 204/300 |
| sawyer_tf3 | 289/300 | same over-squeeze / slip (11) |
| parm5_tf3, parm5l_tf3, parm5s_tf3, parm6_tf3 | 288/290, 189/191, 274/276, 208/210 | seeds 3000074 and 3000091 on every procedural tf3 arm: the fingers are opened to the joint limit (-0.6 rad, fingertip reach ~9 cm from the palm axis) and one lands on a distractor 9 cm from the cube; the descent is blocked 1.3 cm short and the FSM waits for 29 s (no timeout / recovery) |
| parm7_tf3 | 237/241 | the same distractor block (2) + seeds 3 and 171: the grasp yaw (closest to the current tool yaw) is out of reach (elbow at its joint limit, IK residual 2.1 cm) although the feasibility check (yaw 0, tol 1.2 cm) passed; stuck in descend |
| all pg2 bodies | 100 % | no failures; same jerky motion; grasp penetration ~7 mm (pg2 closes to 0 with its 40 N limit) |
Other observations: procedural arms reach joint-limit margin 0 (the elbow saturates for reaches near the base);
the demo page's "parm7_pg2 never completes the place" clip is a render-budget artifact (400 ticks), not a failure here
(238/238 at 600 ticks).

## v2 teacher (scripted_teacher:pick_place_v2_minjerk)
Code: `src/rrp/teachers/arm_smooth.py` (`SmoothPickPlaceTeacher`, registry `ARM_TEACHERS`, `make_arm_teacher`,
`teacher_source`). v1 is `DEFAULT_ARM_TEACHER` and nothing changed for existing callers.
- Motion: every plan is a list of segments whose joint-space delta trajectories are superposed; the next segment starts
  30 % before the previous one ends (blended corners). Free-space moves (to the hover pose, transport) are joint-space
  quintics to a seeded IK solution (a straight Cartesian line from ur5e's home passes the base singularity: 5 rad/s base
  spin); moves along the tool axis (descend, lift, lower, retreat, corrections) are straight Cartesian lines with a
  min-jerk time law, tracked by warm-started IK. Durations come from the limits: joint v 1.5 rad/s, a = min(6 rad/s^2,
  0.5 * tau_max / M_jj) (actuator force range x gear over the joint-space inertia diagonal at the start pose), TCP
  0.5 m/s / 2 m/s^2 free, 0.15 m/s for the final approach; a Cartesian segment is stretched until its IK path meets
  the joint limits (falls back to a joint quintic when IK tracking fails, e.g. at a joint limit).
- Grasp yaw: among the cube x gripper symmetric yaws (pg2 90 deg, tf3 30 deg), reachable (hover + grasp IK < 5 mm)
  first, then fingertip clearance from other objects >= 1.5 cm, joint margin >= 2 %, then the smallest wrist roll.
- Three-finger pre-shape: open to 0.55 rad short of contact instead of the joint limit (fingertip reach 6 cm vs 9 cm).
- Grasp: min-jerk close ramp; on contact (public touch sensors on >= 2 fingers) the target becomes the MEASURED finger
  position + a small squeeze (tf3 0.03 rad, pg2 3 mm); settle >= 0.25 s; lift only when touch >= 2 for 3 ticks and the
  gripper is blocked. No contact -> regrasp. Opening is a 0.35 s ramp.
- Residuals: short min-jerk correction moves (smooth replacement of v1's integral step) unless the IK itself cannot
  reach the goal; reach-limited grasps within 1.6 cm (or only too high, <= 2.2 cm) are taken, like v1's stall rule.
- Recovery instead of stalls: blocked approach, no contact, unconfirmed grasp, or the object leaving the hand
  (privileged object pose, > 4 cm from its grasp offset) -> retreat along the tool axis and re-approach with the next
  yaw candidate; give up after 3 attempts (explicit `gave_up` failure).
- Provenance: `collect_teacher_episode(..., teacher_version="v2")` / data config `"teacher_version": "v2"` record
  `source = scripted_teacher:pick_place_v2_minjerk` and `teacher_version` in the episode meta and the dataset
  provenance flags (v1-default collections are byte-for-byte unchanged in meta).
- Tests: `tests/unit/test_arm_teacher_versions.py` (min-jerk boundary conditions / peaks / duration limits, labels,
  a real v2 episode). Full unit suite on the peer: 277 passed, 2 skipped (layering test included).

## old vs new (identical bodies, seeds and harness)
Raw rows: v1 `artifacts/runs/armexpert_diag/v1_18bodies.jsonl.gz` (peer lease 1790475957_8cb8b6), v2
`artifacts/runs/armexpert_diag/v2final_18bodies.jsonl.gz` (final code, peer lease 1790477934_ddfde2, rc 0); per-body
summaries `*.summary.json`; this table `compare_v1_v2.json` (`research/scripts/2026-09-26/armexpert_compare.py`).
Medians over feasible episodes; jerk / acceleration are finite differences at the 20 Hz control rate (commanded =
the labels a BC learns; measured = what the video shows); "dv at switch" = largest joint-velocity step within +-1 tick of
a phase switch; pen = max object<->hand penetration after the approach; slip = object displacement in the TCP frame
while carried.

**Success 4703/4811 -> 4810/4811; no body drops (every tf3 body improves, every pg2 body stays at 100 %).
Commanded joint jerk peak drops ~10x (385-625 -> 36-55 rad/s^3), measured joint jerk 2.5-26x, measured TCP jerk
2-6x, velocity steps at phase switches 4-7x (0.9-1.4 -> 0.18-0.28 rad/s per tick); tf3 grasp penetration 18-24 ->
6-7 mm, slip 4-16 -> 2-4 mm; lift-before-held 394 -> 0. Cost: episodes ~1.5-1.9 s longer (median 4.8-8.2 -> 6.7-8.8 s;
p99 over successes 8.9 -> 9.5 s, max 16.6 -> 17.5 s; 1 success in ~4800 exceeds 15 s for either version).**
The only v2 failure (parm7_tf3 seed 3) is the grasp that no yaw can reach (v1 fails it too). Joint-limit margin is
unchanged (0 on procedural arms: the elbow saturates for near-base reaches; a kinematic property of those bodies).

| body | split | success v1 | success v2 | v1 failure stages | jerk cmd med v1 -> v2 (rad/s^3) | jerk meas med | TCP jerk meas med (m/s^3) | dv at switch med (rad/s per tick) | joint-limit margin min | time med (s) | grasp pen med (mm) | slip med (mm) | lift before held v1/v2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| panda_pg2 | train | 300/300 | 300/300 | - | 566 -> 46 | 141 -> 25 | 33.4 -> 9.5 | 1.26 -> 0.18 | 0.009 -> 0.013 | 6.1 -> 7.5 | 6 -> 5 | 3 -> 2 | 0/0 |
| panda_tf3 | target (demos only) | 280/300 | 300/300 | placed_outside_zone 4, dropped:retreat 16 | 572 -> 44 | 141 -> 24 | 35.9 -> 9.5 | 1.26 -> 0.18 | 0.009 -> 0.014 | 6.0 -> 7.4 | 23 -> 7 | 16 -> 4 | 204/0 |
| parm5_pg2 | train | 290/290 | 290/290 | - | 447 -> 47 | 880 -> 38 | 87.4 -> 14.1 | 1.04 -> 0.25 | -0.000 -> -0.000 | 4.9 -> 6.8 | 7 -> 2 | 4 -> 2 | 0/0 |
| parm5_tf3 | train | 288/290 | 290/290 | grasp_never_held:descend 2 | 453 -> 48 | 879 -> 40 | 88.6 -> 14.6 | 1.04 -> 0.25 | -0.000 -> -0.000 | 4.9 -> 6.7 | 18 -> 6 | 12 -> 2 | 0/0 |
| parm5l_pg2 | held-out source | 190/190 | 190/190 | - | 409 -> 41 | 878 -> 36 | 93.5 -> 14.5 | 0.96 -> 0.24 | -0.000 -> -0.000 | 5.0 -> 6.9 | 7 -> 2 | 3 -> 2 | 0/0 |
| parm5l_tf3 | train | 189/191 | 191/191 | grasp_never_held:descend 2 | 410 -> 44 | 875 -> 39 | 95.4 -> 14.8 | 0.96 -> 0.24 | -0.000 -> -0.000 | 5.0 -> 6.8 | 19 -> 6 | 12 -> 2 | 0/0 |
| parm5s_pg2 | train | 277/277 | 277/277 | - | 604 -> 53 | 1033 -> 40 | 87.0 -> 13.4 | 1.32 -> 0.27 | -0.000 -> 0.003 | 4.8 -> 6.9 | 7 -> 2 | 4 -> 3 | 0/0 |
| parm5s_tf3 | held-out source | 274/276 | 276/276 | grasp_never_held:descend 2 | 625 -> 55 | 1059 -> 41 | 86.7 -> 14.0 | 1.36 -> 0.28 | -0.000 -> 0.002 | 4.8 -> 6.7 | 20 -> 6 | 10 -> 2 | 0/0 |
| parm6_pg2 | train | 208/208 | 208/208 | - | 390 -> 43 | 872 -> 38 | 92.1 -> 14.5 | 0.91 -> 0.25 | -0.000 -> -0.000 | 5.0 -> 6.9 | 7 -> 2 | 3 -> 2 | 0/0 |
| parm6_tf3 | train | 208/210 | 210/210 | grasp_never_held:descend 2 | 385 -> 45 | 857 -> 39 | 92.6 -> 14.7 | 0.91 -> 0.25 | -0.000 -> -0.000 | 5.0 -> 6.7 | 18 -> 6 | 12 -> 2 | 0/0 |
| parm7_pg2 | train | 238/238 | 238/238 | - | 417 -> 45 | 821 -> 38 | 92.3 -> 16.0 | 0.98 -> 0.25 | -0.000 -> -0.000 | 5.0 -> 6.9 | 7 -> 2 | 3 -> 2 | 0/0 |
| parm7_tf3 | train | 237/241 | 240/241 | grasp_never_held:descend 4 | 419 -> 46 | 833 -> 39 | 91.1 -> 15.1 | 0.98 -> 0.25 | -0.000 -> -0.000 | 5.0 -> 6.8 | 18 -> 6 | 12 -> 2 | 0/0 |
| sawyer_pg2 | train | 300/300 | 300/300 | - | 549 -> 52 | 216 -> 27 | 35.5 -> 14.0 | 1.37 -> 0.21 | 0.041 -> 0.042 | 5.6 -> 7.4 | 7 -> 2 | 2 -> 2 | 0/0 |
| sawyer_tf3 | train | 289/300 | 300/300 | dropped:retreat 11 | 545 -> 53 | 214 -> 28 | 35.3 -> 12.9 | 1.35 -> 0.21 | 0.040 -> 0.050 | 5.6 -> 7.2 | 24 -> 6 | 4 -> 3 | 0/0 |
| ur5e_pg2 | train | 300/300 | 300/300 | - | 532 -> 37 | 63 -> 17 | 16.2 -> 7.3 | 1.32 -> 0.18 | 0.066 -> 0.068 | 8.2 -> 8.8 | 6 -> 2 | 5 -> 4 | 0/0 |
| ur5e_tf3 | train | 262/300 | 300/300 | dropped:retreat 38 | 534 -> 36 | 63 -> 16 | 16.2 -> 6.9 | 1.32 -> 0.18 | 0.066 -> 0.068 | 8.2 -> 8.7 | 18 -> 7 | 9 -> 3 | 0/0 |
| xarm7_pg2 | target (demos only) | 300/300 | 300/300 | - | 585 -> 49 | 135 -> 24 | 35.1 -> 10.7 | 1.24 -> 0.20 | 0.117 -> 0.122 | 6.0 -> 7.5 | 8 -> 2 | 4 -> 2 | 0/0 |
| xarm7_tf3 | target (demos only) | 273/300 | 300/300 | dropped:retreat 27 | 571 -> 50 | 149 -> 26 | 58.3 -> 10.8 | 1.25 -> 0.20 | 0.117 -> 0.121 | 6.0 -> 7.4 | 24 -> 6 | 12 -> 4 | 190/0 |

TOTAL success v1 4703/4811, v2 4810/4811


## videos (artifacts/video/, INDEX.md lines added; all scripted_teacher, privileged)
- panda_pg2 s3000001 (dev seed), both succeed: `2026-09-26_scripted_teacher_v1_panda_pg2_pick_place_s3000001_success.mp4`,
  `..._v2_panda_pg2_pick_place_s3000001_success.mp4`, side by side `..._v1_vs_v2_panda_pg2_pick_place_s3000001_success_vs_success.mp4`
- ur5e_tf3 s10: v1 FAILURE (cube slides down the over-squeezed fingers, lands outside the zone) vs v2 success:
  `2026-09-26_scripted_teacher_v1_ur5e_tf3_pick_place_s10_failure.mp4`, `..._v2_ur5e_tf3_pick_place_s10_success.mp4`,
  `..._v1_vs_v2_ur5e_tf3_pick_place_s10_failure_vs_success.mp4`
- parm6_tf3 s3000074: v1 FAILURE (splayed finger on a distractor, descent blocked) vs v2 success:
  `2026-09-26_scripted_teacher_v1_parm6_tf3_pick_place_s3000074_failure.mp4`, `..._v2_parm6_tf3_pick_place_s3000074_success.mp4`,
  `..._v1_vs_v2_parm6_tf3_pick_place_s3000074_failure_vs_success.mp4`
- parm7_tf3 s3: both FAIL (grasp out of reach at every yaw): `2026-09-26_scripted_teacher_v1_parm7_tf3_pick_place_s3_failure.mp4`,
  `2026-09-26_scripted_teacher_v2_parm7_tf3_pick_place_s3_failure.mp4` (v2 gives up after 3 attempts in 6 s instead of
  hovering for the whole episode)
Rendered by `scripts/render_teacher_versions.py` (same code path as the metrics; the end card prints the row's metrics),
peer GPU lease 1790477753_6a73bf (rc 0).

## BC expert from v2 (plan only; nothing regenerated)
How the ladder uses it (research/tracks/ladder.md): the DAgger labels for system 0 and the generator-DAgger targets
z* = E(chunk) come from the STATELESS BC expert `learned:direct1701_u12000` (direct-action BC, seed 1701, B-1 fixed,
trained on the pack `latent_pp_v3dart_s1_H16` = `pick_place_primary_v3dart`, 13 source-train bodies, DART noise,
H16, prev-action input 0), not from the scripted FSM (which is stale off its own trajectory). Training that expert from
v2 is feasible with the existing pipeline; proposed steps (each gated, none run):
1. Collect `pick_place_primary_v4dart` = the v3dart config (same items, seeds 0-299 per body, same DART noise levels,
   same distractor rule) + `"teacher_version": "v2"` (the only change). Cost: a v2 episode takes ~3-4x the CPU of v1
   (IK planning; 0.3-0.5 s vs 0.1 s wall), v3dart took 1231 s on 6 workers, so roughly 1-1.5 h on 6 peer CPUs.
   Gate: per-body teacher success on the clean (noise 0) episodes >= the v3dart manifest's; manifest provenance
   `scripted_teacher:pick_place_v2_minjerk`, flags.teacher_version; spot-check jerk with teacher_quality on the
   same seeds.
2. Pack with the same packer settings as latent_pp_v3dart_s1_H16 (H16, include_dart_failures), new pack name
   `latent_pp_v4dart_s1_H16`; do not overwrite v3.
3. Train direct-action BC with the identical config and seed 1701 (and 1702 to see seed noise), snapshots at 12k and
   final. Evaluate exactly like sprint_bc: ladder dev seeds (30 feasible from 3,000,000) on panda_pg2 / parm6_tf3,
   held-out parm5s_tf3 / parm5l_pg2 (sealed protocol seeds 2,000,000+), plus the teacher_quality motion metrics on
   the BC rollouts. Gate: success not worse than direct1701 beyond seed noise (Wilson intervals) AND lower commanded
   jerk / velocity steps in its chunks. Only then can it replace direct1701 as the DAgger labeller.
Expected issues to watch (why this is a gate, not a formality):
- v2 is time-parameterized: within a quintic the command depends on the time since the segment started, which a
  stateless BC must infer from joint velocity (the featurizer carries qd * 0.1) and from the object/gripper state. At
  segment starts and in the close/settle wait the observation is nearly static while the label changes; v1 had the
  same kind of hidden timers (0.7 s close, 0.6 s open) but fewer of them. Watch the grasp/settle phase in BC failures.
- v2 grasps with less squeeze and waits for confirmation; BC trained on it must learn the contact-conditioned gripper
  target (observable through the gripper width and touch channels) instead of "close fully".
- Episodes are ~20-25 % longer (median 6.5-9 s); the 300-tick evaluation budget (15 s) still covers the teacher's
  worst cases in the final run (see max time in the table), but BC/latent routes that are slower than the teacher
  have less slack.
- Recovery behaviour (regrasp after a blocked approach / lost object) is rare in clean data (<1 %) and appears as a
  few long episodes; it is useful coverage for DAgger but will be under-represented.
- A change of expert changes every downstream lineage (Stage A, flows, system-0 DAgger, generator DAgger): the
  existing sem / semfix / nosem comparisons must not mix v1- and v2-trained components. Recommend regenerating only as a
  complete new lineage set (all three variants, same seeds), after R0 / the seed-2 replication and the nosem ablation
  finish.

## commands
```bash
export RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/armexpert; scripts/peer_sync.sh push
B=panda_pg2,parm6_tf3,parm5_pg2,parm5_tf3,parm5l_tf3,parm5s_pg2,parm6_pg2,parm7_pg2,parm7_tf3,sawyer_pg2,sawyer_tf3,ur5e_pg2,ur5e_tf3,parm5s_tf3,parm5l_pg2,panda_tf3,xarm7_pg2,xarm7_tf3
scripts/peer_run.sh --cpu 6 --mem 12G --label armexpert_diag_v1 --max-seconds 2700 -- PY -m rrp.evaluation.teacher_quality \
  --bodies $B --seeds 0-199,3000000-3000099 --versions v1 --workers 6 --chunk 50 --out artifacts/runs/armexpert_diag/v1_18bodies.jsonl
scripts/peer_run.sh --cpu 4 --mem 10G --label armexpert_diag_v2_final --max-seconds 7000 -- env OMP_NUM_THREADS=1 PY -m rrp.evaluation.teacher_quality \
  --bodies $B --seeds 0-199,3000000-3000099 --versions v2 --workers 4 --chunk 25 --out artifacts/runs/armexpert_diag/v2final_18bodies.jsonl
python3 research/scripts/2026-09-26/armexpert_compare.py <v1 jsonl> <v2 jsonl> artifacts/runs/armexpert_diag/compare_v1_v2.json
scripts/peer_run.sh --gpu --gpu-mem 2G --cpu 1 --mem 4G --label armexpert_render --max-seconds 1700 -- PY scripts/render_teacher_versions.py \
  --robot ur5e_tf3 --seeds 10 --sbs --out /dev/shm/rrp-brandonin/armexpert_video      # likewise panda_pg2 3000001, parm6_tf3 3000074, parm7_tf3 3
```
Placement note: the lead asked for host placement at 19:50, but the host broker was full (12.9 / 12.98 CPU) and a 1.9-CPU
host attempt stalled under external memory pressure (swap 15/15 GB, memory PSI full 13 %); it was stopped with
`rrp ops stop --owned-only --lease 1790477714_81f5ec` and the run went to the peer (4 CPU).

## step 2 part 1 (D-097 approval): v4dart data, pack, v2 stateless BC expert — state: verified (completed)
Scope (lead, after D-097): collect `pick_place_primary_v4dart` = v3dart config + `"teacher_version": "v2"`; pack like
`latent_pp_v3dart_s1_H16`; train the stateless direct BC at seeds 1701/1702 and evaluate exactly like sprint_bc; report
BC output smoothness vs the v1 BC expert. Latent lineages NOT started (the lead launches them).

### v2 revised before its first data use (re-verified)
The first v4dart collection showed one CLEAN failure the D-097 run had not sampled (parm7_pg2 seed 291, seeds 200-299
were not in the verification set): the grasp pose is out of reach by 1 cm (elbow at its limit), the Cartesian descent
fell back to a joint-space arc that swept the fingers through the cube and knocked it away, and the retry picked an
unreachable yaw. Fixes (arm_smooth.py): (1) when the END point of a Cartesian segment is out of reach, track the
straight line to the closest reachable point (IK of the goal -> FK) instead of a joint arc; (2) retries cycle through the
reachable (or within-1.6-cm) yaws, repeating the best one, never an unreachable one; unreachable candidates are ranked by
IK error. That collection was discarded (its manifest is kept at `artifacts/runs/armexpert/v4dart/_superseded/`).
Re-verification with the fixed code, same 18 bodies x 300 seeds as D-097 (host, per-body leases because the host
memory-PSI watchdog shed longer jobs; `scripts/armexpert_verify_host.sh`): **v2 4811/4811** (v1 4703/4811); parm7_tf3
seed 3, the one D-097 failure, now succeeds; smoothness unchanged (table `artifacts/runs/armexpert_diag/compare_v1_v2b.json`,
rows `v2b_18bodies.jsonl.gz`). D-097's v2 numbers stay valid for the earlier code (commit c8cfd6a); the revised code is
commit 5de2b45. The version label stays `pick_place_v2_minjerk` (no data had been produced with the earlier code; every
manifest records the git sha).

### collection and pack (run-dag)
`dags/armexpert_v4dart.yaml` (collect -> pack), run with `rrp run-dag`. Placement history: host first (shed three times
by the host memory-PSI watchdog, then the host broker was fully leased by the contact track for 6 h), final run on the
PEER (collect 6 CPU, lease 1790483042_545738, 7 min; pack 2 CPU, lease 1790483469_14e3e7, 10 min; both rc 0, pipeline
manifests with config hashes). The pack output dir is a symlink to the peer disk (`~/rrp-peer-data/packed/
latent_pp_v4dart_s1_H16`, 15 GB; /dev/shm could not hold it); a copy is on the host at
`~/work/rrp-data/packed/latent_pp_v4dart_s1_H16`. Store aliases: `artifacts/packed/latent_pp_v4dart_s1_H16`,
`artifacts/datasets/pick_place_primary_v4dart` (peer).
What run-dag lacked: (a) BC training and the sprint_bc evaluations are not pipeline stages (train_policy/train_bc are
legacy-only kinds; no `learned`-route ladder eval or protocol source/b0 cells as stages), so they run through
`scripts/armexpert_bcv2_chain.sh` (`rrp campaign baseline-cell`, `scripts/ladder.py --route learned`); (b) no data
transfer between placements (cross-placement deps are refused; moving the partial host collection to the peer was manual);
(c) no way to put a stage output on a different filesystem than the derived `artifacts/runs/...` dir (the 15 GB pack
needed a pre-made symlink on the peer); (d) no wait-for-admission across memory-PSI admission stops (only capacity
refusals are waited for), hence the bounded wrapper `scripts/armexpert_dag_loop.sh`.
Manifest: `artifacts/runs/armexpert/v4dart/collect-v4dart_s1/manifest.json` (+ pipeline_manifest.json), provenance
source `scripted_teacher:pick_place_v2_minjerk`, flags.teacher_version, git 25817c1 (clean).

Gate (clean episodes per body >= v3dart): **passed on every body** (v4 clean successes 3897/3897 feasible vs v3
3820/3897; the tf3 bodies gain: ur5e_tf3 257 -> 300, sawyer_tf3 292 -> 300, panda_tf3 140 -> 150, xarm7_tf3 139 -> 150,
parm7_tf3 239 -> 243, parm5_tf3 287 -> 288).
DART episodes (noise 0.08 rad on the executed arm command, clean labels): successes drop 193 -> 63 of 3447. Probe
(`research/scripts/2026-09-26/armexpert_dart_probe.py`, ur5e_pg2 seeds 0-19): v1 6/20 (it grasps with its tolerant
stall rule and usually fails later, at retreat), v2 1/20 — v2's approach check measures the noisy TCP, never gets within
6 mm, retries and gives up (`approach_blocked` x3). So v4dart's DART rows concentrate on approach/descend states, while
v3dart's covered all phases (but 79 % of v3dart's rows were DART failures that stalled to the 600-tick limit: 546 ticks
per failure episode vs 250 in v4dart). Pack rows: v4 1,367,967 vs v3 2,242,823 (clean rows 506k vs 389k, DART-failure
rows 845k vs 1,776k). Consequence: 6 epochs = ~16k updates for v4 vs 26.3k for v3 (the recipe is epochs-based;
u12000 is compared at equal updates). If the BC shows weak recovery in late phases, a v2.1 that accepts the grasp after
the corrections when within 1.6 cm (as v1 does) would restore late-phase DART coverage.

### v2 stateless BC expert (learned:bcv2_direct1701 / bcv2_direct1702)
Recipe = sprint_bc's direct-action BC unchanged (`rrp campaign baseline-cell --method baseline_direct_action`,
policy-small-structured, packed stride 2, 6 epochs, batch 256, exact resume, zero_prev_action) except the source pack
(`--source-pack artifacts/packed/latent_pp_v4dart_s1_H16`, new CLI option; default = the sealed protocol's pack) and a
kept snapshot at 12,000 updates (`--snapshot-steps 12000`, new; the same weights sprint_bc copied from policy_last).
Both seeds: 16,068 updates (6 epochs of 685,748 stride-2 chunks; v1: 26,304), ~97 min each on the peer GPU (leases
1790484137_9d0046 for 1701, then 1702 sequentially: one peer GPU lease at a time; the host broker was full / the host
shed jobs). Checkpoints (peer store, not in git) `artifacts/runs/armexpert_bcv2/baseline_direct_action/seed<k>/source/`:
policy.pt sha256[:16] 56e6a6c6cbb8a8fb (1701), 7f5d2d398338881b (1702); policy_u12000.pt 6afcb8efdc277708 (1701),
b247c01babe85a55 (1702). Driver `scripts/armexpert_bcv2_chain.sh` (all steps rc-checked; one peer memory-reserve shed of
the 1702 xarm7_pg2 b0 cell was rerun).
Evaluation = sprint_bc's sets: ladder `learned` route (replan 8, NFE 8, prev-action 0, 300 ticks, privileged
evaluator) on panda_pg2 / parm6_tf3, first 30 feasible from 3,000,000 (dev) and 3,000,200 (fresh); held-out source
bodies with the protocol harness (50 seeds from 2,000,000, infeasible excluded) for u12000 (helper
`research/scripts/2026-09-26/armexpert_bc_heldout.py` = `evaluate_checkpoint`) and final (the protocol source cell);
budget-0 zero-shot cells (100 each) for final. v1 reference rows are sprint_bc's raw summaries (no v1 final run exists
on the fresh 3,000,200 set). Raw: `artifacts/runs/armexpert_bcv2_eval/<robot>/learned_bcv2_direct<seed>_<ck>_s<start>.*`,
`.../heldout/`, `artifacts/runs/armexpert_bcv2/baseline_direct_action/seed<seed>/cells/*.json`; table
`artifacts/runs/armexpert_bcv2_eval/compare_bc_v1_v2.json` (`research/scripts/2026-09-26/armexpert_bc_compare.py`).

| checkpoint | panda_pg2 dev | parm6_tf3 dev | panda_pg2 fresh 3,000,200 | parm6_tf3 fresh | held-out source (parm5s_tf3, parm5l_pg2) | b0 panda_tf3 / xarm7_pg2 / xarm7_tf3 | cmd_step_rad dev (panda / parm6) |
|---|---|---|---|---|---|---|---|
| v1 data: learned:direct1701 (sprint_bc) u12000 | 25/30 | 27/30 | 24/30 | 27/30 | parm5s_tf3 44/47 parm5l_pg2 30/33 | - | 0.085 / 0.048 |
| v1 data: learned:direct1701 (sprint_bc) final | 30/30 | 30/30 | - | - | 79/80 pooled | 78/100 / 0/100 / 0/100 | 0.097 / 0.047 |
| v2 data: learned:bcv2_direct1701 u12000 | 29/30 | 30/30 | 30/30 | 30/30 | parm5s_tf3 47/47 parm5l_pg2 33/33 | - | 0.066 / 0.029 |
| v2 data: learned:bcv2_direct1701 final | 30/30 | 30/30 | 30/30 | 30/30 | 80/80 pooled | 99/100 / 0/100 / 0/100 | 0.065 / 0.028 |
| v2 data: learned:bcv2_direct1702 u12000 | 26/30 | 30/30 | 29/30 | 30/30 | parm5s_tf3 47/47 parm5l_pg2 33/33 | - | 0.062 / 0.028 |
| v2 data: learned:bcv2_direct1702 final | 30/30 | 30/30 | 30/30 | 30/30 | 79/80 pooled | 99/100 / 0/100 / 0/100 | 0.064 / 0.028 |

Pooled over the ladder sets (panda + parm6, dev + fresh = 120 episodes): u12000 v1 103/120 (dev+fresh as available:
25+27+24+27) vs v2 119/120 (1701) and 115/120 (1702); final v1 60/60 (dev only) vs v2 120/120 and 120/120. Held-out
source: u12000 v1 74/80 vs v2 80/80, 80/80; final v1 79/80 vs v2 80/80, 79/80. Zero-shot panda_tf3 (new gripper
pairing) 78/100 -> 99/100 (both seeds); unseen xarm7 arms stay 0/100 (a new kinematic chain is not solved without
target data, as before). **The v2-data expert is at least as competent as the v1-data one on every set, clearly
better at the u12000 point used for DAgger labels, with fewer updates.** Caveat: v1 has one training seed.

Smoothness of the BC outputs (executed commands of closed-loop rollouts; `teacher_quality --policy`; bodies panda_pg2,
parm6_tf3, parm5s_tf3, parm5l_pg2, seeds 3,000,000-039, feasible only; the teachers on the same seeds for scale;
"max joint-velocity step" is per episode, the largest step between consecutive ticks, which for BC is at chunk
boundaries; BC time stops at the public success, the teachers run to the end of their retreat, so times are not
comparable). Raw `artifacts/runs/armexpert_bcv2_eval/motion/*.jsonl.gz`, table `motion/compare_motion.json`
(`research/scripts/2026-09-26/armexpert_bc_motion.py`):

| source | body | success | jerk cmd peak med | jerk cmd rms med | jerk meas peak med | TCP jerk meas med (m/s^3) | max joint-velocity step (rad/s per tick) med | time med (s) |
|---|---|---|---|---|---|---|---|---|
| learned:bcv2_direct1701_final | panda_pg2 | 40/40 | 354 | 66 | 53 | 26.6 | 0.59 | 6.9 |
| learned:bcv2_direct1701_final | parm5l_pg2 | 27/27 | 500 | 65 | 180 | 68.4 | 0.67 | 6.2 |
| learned:bcv2_direct1701_final | parm5s_tf3 | 36/36 | 588 | 67 | 211 | 70.3 | 0.85 | 6.0 |
| learned:bcv2_direct1701_final | parm6_tf3 | 28/28 | 333 | 47 | 128 | 63.0 | 0.56 | 6.0 |
| learned:bcv2_direct1701_u12000 | panda_pg2 | 40/40 | 564 | 75 | 77 | 29.8 | 0.81 | 7.0 |
| learned:bcv2_direct1701_u12000 | parm5l_pg2 | 27/27 | 655 | 71 | 242 | 71.6 | 0.90 | 6.1 |
| learned:bcv2_direct1701_u12000 | parm5s_tf3 | 36/36 | 638 | 69 | 245 | 73.8 | 1.00 | 6.0 |
| learned:bcv2_direct1701_u12000 | parm6_tf3 | 28/28 | 240 | 48 | 114 | 51.8 | 0.51 | 5.9 |
| learned:bcv2_direct1702_final | panda_pg2 | 40/40 | 341 | 64 | 51 | 28.2 | 0.57 | 6.9 |
| learned:bcv2_direct1702_final | parm5l_pg2 | 27/27 | 577 | 60 | 225 | 74.2 | 0.86 | 6.2 |
| learned:bcv2_direct1702_final | parm5s_tf3 | 36/36 | 920 | 88 | 309 | 84.8 | 1.32 | 6.0 |
| learned:bcv2_direct1702_final | parm6_tf3 | 28/28 | 382 | 47 | 137 | 58.9 | 0.60 | 6.0 |
| learned:bcv2_direct1702_u12000 | panda_pg2 | 33/40 | 698 | 77 | 92 | 39.7 | 0.97 | 7.0 |
| learned:bcv2_direct1702_u12000 | parm5l_pg2 | 26/27 | 468 | 59 | 199 | 66.3 | 0.68 | 6.2 |
| learned:bcv2_direct1702_u12000 | parm5s_tf3 | 36/36 | 835 | 86 | 312 | 97.2 | 1.14 | 6.0 |
| learned:bcv2_direct1702_u12000 | parm6_tf3 | 28/28 | 811 | 66 | 246 | 98.7 | 1.07 | 6.0 |
| learned:direct1701_final | panda_pg2 | 39/40 | 856 | 130 | 137 | 52.0 | 1.27 | 5.3 |
| learned:direct1701_final | parm5l_pg2 | 27/27 | 581 | 105 | 932 | 107.6 | 0.99 | 4.0 |
| learned:direct1701_final | parm5s_tf3 | 34/36 | 728 | 118 | 577 | 115.6 | 1.37 | 4.0 |
| learned:direct1701_final | parm6_tf3 | 28/28 | 515 | 87 | 437 | 105.3 | 0.90 | 4.0 |
| learned:direct1701_u12000 | panda_pg2 | 33/40 | 2074 | 253 | 257 | 101.3 | 2.76 | 5.4 |
| learned:direct1701_u12000 | parm5l_pg2 | 26/27 | 1340 | 187 | 672 | 199.9 | 1.82 | 4.3 |
| learned:direct1701_u12000 | parm5s_tf3 | 28/36 | 2351 | 269 | 909 | 312.3 | 3.10 | 4.6 |
| learned:direct1701_u12000 | parm6_tf3 | 27/28 | 1736 | 191 | 922 | 216.7 | 2.41 | 4.4 |
| scripted_teacher:pick_place_v1_waypoint | panda_pg2 | 40/40 | 560 | 83 | 141 | 33.5 | 1.27 | 6.0 |
| scripted_teacher:pick_place_v1_waypoint | parm5l_pg2 | 27/27 | 410 | 80 | 742 | 94.2 | 0.96 | 5.0 |
| scripted_teacher:pick_place_v1_waypoint | parm5s_tf3 | 36/36 | 618 | 109 | 957 | 86.9 | 1.47 | 4.8 |
| scripted_teacher:pick_place_v1_waypoint | parm6_tf3 | 28/28 | 382 | 71 | 757 | 93.6 | 0.92 | 5.0 |
| scripted_teacher:pick_place_v2_minjerk | panda_pg2 | 40/40 | 47 | 7 | 25 | 9.9 | 0.22 | 7.5 |
| scripted_teacher:pick_place_v2_minjerk | parm5l_pg2 | 27/27 | 40 | 8 | 38 | 15.0 | 0.24 | 6.9 |
| scripted_teacher:pick_place_v2_minjerk | parm5s_tf3 | 36/36 | 54 | 10 | 42 | 14.6 | 0.31 | 6.7 |
| scripted_teacher:pick_place_v2_minjerk | parm6_tf3 | 28/28 | 44 | 8 | 38 | 14.8 | 0.25 | 6.7 |

Reading: v2-data BC is smoother than v1-data BC at matched checkpoints: u12000 commanded jerk RMS 48-86 vs 187-269
rad/s^3 (2.5-4x), peak 240-838 vs 1340-2351, joint-velocity steps 0.5-1.1 vs 1.8-3.1 rad/s per tick, measured TCP jerk
30-99 vs 101-312 m/s^3. At the final checkpoints the RMS jerk is 1.3-2x lower (47-88 vs 87-130) and measured joint /
TCP jerk 1.5-4x lower, but peaks overlap (parm5s_tf3 seed 1702: 920 vs v1 728). Both BCs remain far rougher than the v2
teacher (RMS 7-10): the chunked executor replans every 8 ticks and each new chunk starts from a fresh sample, so the
largest velocity steps sit at chunk boundaries. Smoothing across chunk boundaries (temporal ensembling / blending the
overlap, or conditioning on the executing chunk) is the remaining lever; it is a policy-side change, not a data one.

Videos (learned, public observations): `artifacts/video/2026-09-27_learned_bcv2_direct1701_u12000_{panda_pg2_pick_place_s3000001,panda_pg2_pick_place_s3000023,parm6_tf3_pick_place_s3000001}_success.mp4`,
`2026-09-27_learned_bcv2_direct1701_final_panda_pg2_pick_place_s3000002_success.mp4`, FAILURE
`2026-09-27_learned_bcv2_direct1701_final_zeroshot_newbody_xarm7_pg2_pick_place_s2000000_failure.mp4` (unseen arm,
budget 0). Rendered on the peer GPU (leases 1790495783_0ee4f8, 1790495810_c774a0).

### recommendation (for the lead; nothing launched)
The v2-data BC expert passes the gate (not worse on any set, better at u12000, smoother outputs), so it can replace
learned:direct1701_u12000 as the DAgger labeller for the v2 lineage set: use `bcv2_direct1701` u12000 (same update
point as v1's expert) or final; 1702 is the seed-noise check (u12000 panda 26/30 dev vs 29/30 for 1701). Open items
before the lineage launch: (1) the DART-coverage difference (v4dart DART failures stop at approach); (2) the lineage
configs must point every pack reference at latent_pp_v4dart_s1_H16 and every BC expert reference at the v2 expert, no
mixing with v1 components.

## grasp contact v2 (D-108) — state: verified (arm data NOT regenerated; lead decides)
Problem (W6, D-108): arm grasps hold by interpenetration (cube-finger overlap 17-20 mm on parm6_tf3), so grasps were
insensitive to friction x0.05; the arm counterpart of legged skating (D-093).

Cause. MuJoCo's soft contact scales its stiffness with the constraint's effective mass; the cube weighs 42 g, and the
legacy pad/object contacts use the default solref 0.02 s / solimp (0.9, 0.95). A position-servo gripper closing to its
limit then sinks in: rig (below) with the actuator at its force limit: three-finger 10-13 mm during a held lift (it also
slips 10-14 mm during the lift and holds only 1.1x the cube's weight by friction), parallel jaw 1.3 mm. In the pick-place
teacher runs (v2 teacher, grasp_v1) the median post-approach penetration is 6-7 mm on every tf3 body (max 10 mm) and
1.6-4.8 mm on pg2. The tf3 hinge torque limit (8 N m, ~145 N at the fingertip) is also unrealistic.

Implementation (`src/rrp/physics/grasp_contact.py`, selectable, default v1 = byte-identical legacy build):
`RRP_GRASP_CONTACT=v2` (or `grasp_contact.apply(spec, "v2")`); every arm scenario builder compiles through
`rrp.envs.scenario.compile_scene` (pick_place, paired, reach; dual support_insert / handover / assign).
- finger pads: priority 1 (the pad's parameters define every pad-object contact, no max-mixing), friction
  [1.0, 0.004, 0.0001] (rubber-like pad, mu 1.0), condim 4 (torsional friction, 4 mm patch radius);
- objects: friction [0.9, 0.004, 0.0001], condim 4;
- pads and objects: solref [0.004, 1.0] (time constant = 2 physics steps, critically damped), solimp
  [0.99, 0.999, 0.0005, 0.5, 2]; margin = gap = 0;
- grip force (sourced): parallel jaw unchanged at 40 N per finger actuator (the measured squeeze, i.e. the normal force
  on each pad, is 20 N at the limit; Franka Hand: 70 N continuous / 140 N max grasping force [1]); three-finger hinge
  torque 8 -> 2.2 N m = 40 N at the 55 mm fingertip (Robotiq 3-Finger Adaptive Gripper: 30-70 N adjustable grip
  force [2]); measured sum of finger normal forces at the limit 43-48 N.
- provenance: the model carries a text element `grasp_contact_version`; `physics_provenance(model)` records
  `grasp_contact_version: "grasp_v2"` (W3 API; the key is omitted for legacy models so old records stay identical);
  teacher_quality rows carry `grasp_contact`.
Sources: [1] Franka Emika Panda datasheet / Franka Hand product manual (https://www.generationrobots.com/media/panda-franka-emika-datasheet.pdf,
https://download.franka.de/documents/220010_Product%20Manual_Franka%20Hand_1.2_EN.pdf); [2] Robotiq 3-Finger
specifications (https://qviro.com/product/robotiq/3-finger-robotiq/specifications,
https://assets.robotiq.com/website-assets/support_documents/document/3-Finger_PDF_20190221.pdf). Retrieved via web search 2026-09-27.

Tuning choice: solref time constant 0.004 vs candidates 0.006 / 0.008 on the bodies with the most held-contact flicker
(sawyer_pg2, xarm7_pg2, 60 seeds, partial runs before host sheds, `artifacts/runs/armexpert_grasp/tc/`): carry
penetration median 0.01-0.02 mm (0.004) vs 0.04 (0.006) vs 0.06 mm (0.008), max 0.7 / 1.3 / 2.5 mm; held fraction
during lift/transport 1.00 / 1.00 / 0.87-0.97; so 0.004 (the stiffest MuJoCo allows at dt 2 ms).

### rig: held lift + slip under load (`python -m rrp.evaluation.grasp_rig`, tests `tests/unit/test_grasp_contact.py`)
The real gripper modules on a vertical carriage; close to the force limit (the worst case a policy can command), lift
10 cm (min-jerk, 0.5 s), hold, then ramp the cube mass x1.08 every 0.25 s until it moves > 5 mm relative to the palm.
Coulomb prediction for the parallel jaw: m* = mu * sum(N) / g.
| version | gripper | friction x | penetration during lift (mm) | slip during lift (mm) | sum N (N) | slip mass (kg) | Coulomb m* (kg) | ratio |
|---|---|---|---|---|---|---|---|---|
| grasp_v1 | pg2 | 1 | 1.35 | 0.05 | 40.0 | 1.47 | 6.12 (mu 1.5) | 0.24 |
| grasp_v1 | pg2 | 0.05 | 1.34 | 0.05 | 40.0 | 0.315 | 0.306 | 1.03 |
| grasp_v1 | tf3 | 1 | 11.2 (yaw 0) / 12.9 (yaw 15) | 10.5 / 14.2 | 45.3 / 1.2 | 0.046 | - | slips at 1.1x the cube weight |
| grasp_v1 | tf3 | 0.05 | 0.4 | dropped (102 mm) | 0 | - | - | not held |
| **grasp_v2** | pg2 | 1 | **0.03** | 0.0 | 40.0 | 4.31 | 4.08 | **1.06** |
| **grasp_v2** | pg2 | 0.05 | **0.03** | 0.0 | 40.0 | 0.214 | 0.204 | **1.05** |
| **grasp_v2** | tf3 | 1 | **0.07-0.09** | 0.0-0.2 | 42.6-48.1 | 0.37 (2 fingers touch, yaw 0/30) / 6.8-10.1 (3 fingers, yaw 15/45) | - | - |
Reading: v2 removes interpenetration (< 0.1 mm) and the parallel jaw slips where Coulomb friction says it should
(1.05-1.06; v1's soft friction let it creep at 0.24 of the Coulomb load). The three-finger hand is not an antipodal
grasp: with three converging fingers around a 44 mm cube it holds by enveloping (caging) support once the cube settles a
few mm onto the fingertips, 160-240x the cube's weight when all three fingers touch; this is geometric support, not
interpenetration (0.07 mm). Raw: `artifacts/runs/armexpert_grasp/rig.jsonl`.

### object friction / mass sweep in the pick-place task (W6 semantics: x friction of cube AND pads; x cube mass)
v2 teacher, panda_pg2 and parm6_tf3, 30 dev seeds from 3,000,000 (22 feasible on parm6), grasp v1 vs v2
(`scripts/armexpert_gc_sweep.sh`, raw `artifacts/runs/armexpert_grasp/sweep/*.jsonl`, table `sweep/compare_sweep.json`):
| body | object friction x | cube mass x | grasp_v1 success | grasp_v2 success | grasp_v2 failures | slip median mm v1 -> v2 |
|---|---|---|---|---|---|---|
| panda_pg2 | 1 | 100 | 0/30 | 0/30 | {'gave_up': 30} | 32.0 -> 33.3 |
| panda_pg2 | 1 | 30 | 30/30 | 15/30 | {'gave_up': 15} | 3.7 -> 31.5 |
| panda_pg2 | 1 | 10 | 30/30 | 30/30 | - | 2.5 -> 0.7 |
| panda_pg2 | 1 | 1 | 30/30 | 30/30 | - | 2.5 -> 0.6 |
| panda_pg2 | 0.3 | 1 | 30/30 | 30/30 | - | 2.5 -> 3.0 |
| panda_pg2 | 0.1 | 1 | 30/30 | 30/30 | - | 2.5 -> 7.3 |
| panda_pg2 | 0.05 | 1 | 30/30 | 30/30 | - | 2.7 -> 25.3 |
| panda_pg2 | 0.02 | 1 | 30/30 | 0/30 | {'gave_up': 30} | 4.1 -> 34.0 |
| parm6_tf3 | 1 | 100 | 22/22 | 0/22 | {'gave_up': 22} | 5.8 -> 8.5 |
| parm6_tf3 | 1 | 30 | 22/22 | 22/22 | - | 3.0 -> 16.8 |
| parm6_tf3 | 1 | 10 | 22/22 | 22/22 | - | 2.3 -> 0.7 |
| parm6_tf3 | 1 | 1 | 22/22 | 22/22 | - | 1.9 -> 0.6 |
| parm6_tf3 | 0.3 | 1 | 22/22 | 22/22 | - | 2.2 -> 0.6 |
| parm6_tf3 | 0.1 | 1 | 22/22 | 22/22 | - | 2.7 -> 0.6 |
| parm6_tf3 | 0.05 | 1 | 22/22 | 22/22 | - | 2.6 -> 0.6 |
| parm6_tf3 | 0.02 | 1 | 22/22 | 22/22 | - | 30.3 -> 8.4 |
Reading: under grasp_v2 the parallel-jaw grasp behaves like Coulomb friction with the TEACHER's actual squeeze
(v2 teacher: 3 mm past contact x kp 4000 N/m = 12 N actuator force, sum N ~ 12 N): capacity mu * 12 N. It fails at
friction x0.02 (0.24 N < 0.42 N cube weight), slips but holds at x0.05 (0.6 N, 25 mm median slip), and splits 15/30 at
mass x30 (12.5 N weight vs 12 N capacity: exactly at the threshold). Under grasp_v1 the same grasp was indifferent
to friction down to x0.02 (the D-108 symptom). The three-finger grasp stays robust to friction under v2 by
enveloping support (see the rig), fails at mass x100 (4.3 kg vs 2.2 N m fingers), and slips 17 mm at x30.
Consequence for W6-style robustness: arm friction/mass sweeps are meaningful under grasp_v2 for parallel-jaw bodies;
for three-finger bodies friction matters little by design of the hand (caging), mass does.

### videos (closeup camera following the cube; artifacts/video/, INDEX.md lines)
- side by side, left grasp_v1, right grasp_v2, v2 teacher, same seed:
  `2026-09-27_scripted_teacher_v2_grasp_v1_vs_v2_parm6_tf3_pick_place_s3000003_closeup.mp4` (v1: a finger visibly
  sunk into the cube; v2: fingers on the surface), `..._ur5e_tf3_pick_place_s10_closeup.mp4`,
  `..._panda_pg2_pick_place_s3000001_closeup.mp4`
- FAILURE (physically expected): object and pad friction x0.02 on panda_pg2 s3000001: grasp_v1 still carries the cube,
  grasp_v2 lets it slip out, the teacher re-grasps three times and gives up:
  `2026-09-27_scripted_teacher_v2_grasp_v1_vs_v2_objfric0.02_panda_pg2_pick_place_s3000001_closeup.mp4`,
  `2026-09-27_scripted_teacher_v2_graspv2_objfric0.02_panda_pg2_pick_place_s3000001_failure.mp4`
Rendered on the peer GPU (leases 1790535979_c5492e, 1790536020_5a14a7, 1790536564_a44073; side-by-sides 1790536039_698590).

### v2 teacher under grasp_v1 vs grasp_v2: 18 bodies x 300 seeds (identical seeds and harness as D-097/D-102)
`RRP_GRASP_CONTACT=<v> python -m rrp.evaluation.teacher_quality --versions v2 --seeds 0-199,3000000-3000099 --resume`,
per-body host leases (`GC=<v> scripts/armexpert_verify_host.sh`; sheds by the host memory-PSI watchdog were resumed
chunk-wise). New metrics: penetration while the object is HELD during lift/transport (the D-108 gate), held fraction
during lift/transport. Raw `artifacts/runs/armexpert_grasp/final_grasp{v1,v2}_18bodies.jsonl.gz` (+ per-body
summaries), table `artifacts/runs/armexpert_grasp/compare_final_grasp_v1_v2.json`.
| body | success grasp_v1 -> grasp_v2 | held-lift penetration median / max (mm) v1 -> v2 | post-approach penetration max (mm) v1 -> v2 | slip while carried median / max (mm) v1 -> v2 | held fraction lift/transport min v1 -> v2 |
|---|---|---|---|---|---|
| panda_pg2 | 300/300 -> 300/300 | 0.85 / 1.5 -> 0.02 / 0.6 | 5.7 -> 1.6 | 2.3 / 7.0 -> 0.6 / 1.6 | 1.00 -> 1.00 |
| panda_tf3 | 300/300 -> 300/300 | 3.43 / 5.0 -> 0.09 / 0.1 | 9.6 -> 0.4 | 3.6 / 9.7 -> 0.9 / 2.0 | 1.00 -> 1.00 |
| parm5_pg2 | 290/290 -> 290/290 | 0.68 / 0.9 -> 0.02 / 0.4 | 5.7 -> 3.3 | 2.4 / 3.8 -> 0.4 / 0.7 | 1.00 -> 1.00 |
| parm5_tf3 | 290/290 -> 290/290 | 3.28 / 3.8 -> 0.09 / 0.1 | 7.6 -> 0.3 | 2.0 / 5.0 -> 0.6 / 0.9 | 1.00 -> 1.00 |
| parm5l_pg2 | 190/190 -> 190/190 | 0.68 / 1.0 -> 0.02 / 0.7 | 5.5 -> 2.9 | 2.3 / 3.7 -> 0.4 / 0.8 | 1.00 -> 1.00 |
| parm5l_tf3 | 191/191 -> 191/191 | 3.28 / 3.9 -> 0.09 / 0.1 | 9.9 -> 0.4 | 2.0 / 6.8 -> 0.6 / 1.0 | 1.00 -> 1.00 |
| parm5s_pg2 | 277/277 -> 277/277 | 0.68 / 0.8 -> 0.02 / 0.7 | 1.7 -> 3.3 | 2.6 / 4.4 -> 0.5 / 1.0 | 1.00 -> 0.97 |
| parm5s_tf3 | 276/276 -> 276/276 | 3.29 / 3.8 -> 0.09 / 0.1 | 6.2 -> 0.1 | 2.0 / 5.8 -> 0.6 / 0.9 | 1.00 -> 1.00 |
| parm6_pg2 | 208/208 -> 208/208 | 0.68 / 0.9 -> 0.02 / 0.6 | 4.1 -> 3.0 | 2.2 / 3.6 -> 0.4 / 0.7 | 1.00 -> 1.00 |
| parm6_tf3 | 210/210 -> 210/210 | 3.28 / 3.7 -> 0.09 / 0.1 | 9.5 -> 0.8 | 1.9 / 6.4 -> 0.6 / 0.9 | 1.00 -> 1.00 |
| parm7_pg2 | 238/238 -> 238/238 | 0.68 / 0.9 -> 0.02 / 0.8 | 4.7 -> 4.2 | 2.2 / 3.5 -> 0.4 / 0.7 | 1.00 -> 0.76 |
| parm7_tf3 | 241/241 -> 241/241 | 3.28 / 4.4 -> 0.09 / 0.1 | 10.3 -> 0.2 | 2.1 / 9.8 -> 0.6 / 2.7 | 0.93 -> 1.00 |
| sawyer_pg2 | 300/300 -> 300/300 | 0.69 / 0.9 -> 0.01 / 0.1 | 4.6 -> 2.1 | 1.9 / 4.2 -> 0.4 / 7.6 | 1.00 -> 1.00 |
| sawyer_tf3 | 300/300 -> 300/300 | 3.27 / 4.5 -> 0.09 / 0.5 | 10.4 -> 4.0 | 2.9 / 11.2 -> 0.6 / 17.8 | 0.87 -> 0.79 |
| ur5e_pg2 | 300/300 -> 300/300 | 0.68 / 1.0 -> 0.02 / 0.7 | 5.6 -> 3.2 | 3.5 / 6.5 -> 0.8 / 1.5 | 1.00 -> 0.94 |
| ur5e_tf3 | 300/300 -> 300/300 | 3.31 / 4.8 -> 0.09 / 0.3 | 9.9 -> 0.9 | 3.2 / 7.3 -> 0.7 / 1.6 | 1.00 -> 1.00 |
| xarm7_pg2 | 300/300 -> 300/300 | 0.68 / 1.0 -> 0.02 / 0.9 | 2.4 -> 3.0 | 2.5 / 4.0 -> 0.5 / 5.2 | 1.00 -> 0.96 |
| xarm7_tf3 | 300/300 -> 300/300 | 3.34 / 4.4 -> 0.09 / 0.3 | 10.3 -> 0.6 | 3.7 / 12.5 -> 0.7 / 4.6 | 0.91 -> 0.96 |
Gate: held-lift penetration < 3 mm: grasp_v2 max 0.92 mm over all 4811 episodes (p99 0.15 mm), 0 episodes above 3 mm;
grasp_v1 2407 / 4811 episodes above 3 mm (every tf3 body: median 3.3 mm with this gentle-squeeze teacher; W6 saw
17-20 mm with learned policies that close fully at 145 N). Success is unchanged (4811/4811 both), slip while carried
drops (tf3 median 2-4 -> 0.6-0.9 mm), timing unchanged. Remaining observations: pg2 "held" (>= 2 hand bodies in contact)
flickers more often under v2 during lift/transport on a few bodies (held-fraction minimum 0.76-0.97 on parm7_pg2,
ur5e_pg2, xarm7_pg2, parm5s_pg2) without drops or failures: a stiffer contact makes one pad lose contact for a tick
under arm vibration; sawyer_tf3 has one 17.8 mm slip episode (still placed).

Recommendation: adopt grasp_v2 for new arm results (D-108 rule: no new arm results on the old grasp physics). The
v2 teacher needs no change (4811/4811). Regenerating the arm data under grasp_v2 is the lead's call; it would change
every arm lineage's physics (record `grasp_contact_version` and do not mix with grasp_v1 data or checkpoints), and the
learned policies' closing commands (v1 labels close fully) will now meet a 40 N / 2.2 N m limit with stiff contact:
re-evaluate any existing checkpoint under grasp_v2 before comparing.
