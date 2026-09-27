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

## step 2 part 1 (D-097 approval): v4dart data, pack, v2 stateless BC expert — state: running
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
