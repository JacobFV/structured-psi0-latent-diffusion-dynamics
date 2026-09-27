# contact track: realistic foot-floor contact (contact_v2) and real walking gaits

Owner: contact track agent. Worktree `~/work/rrp-wt/contact`, branch `track/contact`. Started 2026-09-26.

## problem (confirmed)
The learned trackers skate. Under the new validation protocol v2, the forward-trial stance slip is measured at the
floor contact points (force-weighted, foot load > 2% of body weight). v1 trackers in v1 physics give slip ratios
(stance slip / body speed) of t1 0.86, h1 0.57, g1 0.42, go2 0.26, anymal_c 0.34 and hexapod6 0.85. Duty factors
of 0.85-1.0 with 0.02-0.04 s air time on the humanoids mean they shuffle, not step.
Raw output: `artifacts/runs/contact_v2/val/*_v1trk_physv1.json`.

## what was built (state: verified by code, unit tests and smoke runs)
- `src/rrp/morphology/contact.py`: versioned contact models. `v1` is the legacy default and byte-identical to the
  earlier setup. `v2` = `contact_v2`: elliptic cone, impratio 10, floor priority 2 (the floor defines every
  robot-floor contact), mu 0.9, condim 3, solref [0.008, 1], solimp [0.9, 0.99, 0.003, 0.5, 2], dt <= 0.002
  (10 substeps per 50 Hz tick). The design rationale is in the module docstring. Selection: `standalone_model(...,
  contact=)`, `legged_world(..., contact=)`, `build_waypoint_contact(..., contact=)`, or `$RRP_CONTACT_MODEL`
  (default v1). `meta["contact_model"]` and the scenario meta record the version. Trackers live at
  `artifacts/trackers/<body>/contact_v2/actor.pt` (v1 remains `.../<body>/actor.pt`). `load_tracker` picks the actor
  that matches the scene's contact model and raises `TrackerMismatch` on a mismatch. The tracker version string gets
  a `:contact_v2` suffix.
- Stiction: MuJoCo has a single Coulomb coefficient. Stiction is approximated by the elliptic cone plus impratio.
  noslip was measured and not adopted (see the table below). No kinetic-friction hack.
- Randomisation (v2 training, `ContactRandomizer`): mu U(0.4, 1.25) with torsional/rolling scaled to match;
  solref timeconst U(6, 12) ms and damping ratio U(0.8, 1.2), resampled per worker every 500 ticks; root mass
  x U(0.9, 1.1) and CoM shift +-2 cm per worker; PD/actuator latency U{0..4} substeps (0-8 ms) per episode.
- `LeggedBinding.stance()` gives true contact-point slip (velocity of the foot at each floor contact point,
  normal-force weighted), `foot_clearance()` gives foot-site height above standing.
- gait_v2 reward plus the lead's schedule (`RewardCfg.for_kind(kind, "gait_v2")`, `rrp/control/reward_schedule.py`):
  - PERMANENT: tracking, termination, orientation, height, lin_z/ang_xy, joint limits, alive, stand_still, and
    **stance slip** (linear contact-point speed; -1.0 bipeds, -0.5 others; it never decays).
  - PRIORS (decay to 10% floor, `w0 (0.1 + 0.9 (1 - alpha))`): air_time 1.0, clearance -2.0 (swing apex below
    target at touchdown, normalised; targets 0.08 m humanoid, 0.06 quadruped, 0.03 hexapod), contact_phase 1.0
    (bipeds), stand_contact 0.5.
  - NATURAL (ramp `w_min + alpha (w_max - w_min)`): torque -0.02 -> -0.1, action_rate -0.02 -> -0.08, jerk
    (second difference) -0.01 -> -0.04, power/CoT (mean |tau qdot| over the tick / (m g max(|cmd|, 0.25))) -0.02 -> -0.3,
    touchdown impact (force / mg - 1, clipped) 0 -> -0.2.
  - alpha is **performance-gated** (not wall time). Every 25 iterations after a 300-iteration warm-up: +0.1 when the window
    has track_rel_err <= 0.35, fall_rate <= 0.10 and slip_ratio <= 0.20; -0.1 when any exceeds 0.50 / 0.25 / 0.30.
    alpha and all effective weights are logged per iteration (`train_log.jsonl`: `alpha`, `weights`, `gate`) and in
    checkpoint/actor meta (`alpha`, `reward_weights`, `gate_history`). The actor never sees alpha; the critic
    gets it as one extra privileged input (the reward is non-stationary). When alpha first leaves 0, the priors-only
    policy is saved as `actor_alpha0.pt` for the alpha=0 vs final comparison.
- Validation protocol v2 (`tracker_validation --contact v1|v2`): adds `slip_cp_mps`, `slip_ratio`, `duty_factor`
  (per foot), `air_time_s`, `swing_apex_m`, `step_hz` per trial. The **contact gate** is the old gate AND forward slip
  ratio < 0.15 AND every foot steps (0.3 <= duty <= 0.9) AND swing apex >= 0.3 x target. The legacy `slip_mps`
  (foot body origin) is kept for comparability.
- Tests: `tests/unit/test_contact_model.py` (slope stick/slide, creep v1 vs v2, landing penetration, v1 unchanged,
  randomiser ranges), `tests/unit/test_reward_schedule.py` (prior/natural/permanent weights, gate advance/hold/back-off,
  window metrics). All pass, as does `tests/unit/test_legged.py`.

## physics test (raw: `artifacts/runs/contact_v2/slope_table.json`, `scripts/contact_slope_table.py`)
Gravity is tilted so that tan(slope) = r * mu (mu = 1 for both). Mean downhill speed is measured after 0.5 s.

| foot | model | r=0.5 | r=0.9 | r=1.2 (should slide) | max pen. |
|---|---|---|---|---|---|
| 30 kg box sole | v1 | 2.4 mm/s | 8.0 mm/s | 1.88 m/s | 1.2 mm |
| 30 kg box sole | v2 | 0.05 mm/s | 0.08 mm/s | 1.86 m/s | 2.4 mm |
| 30 kg box sole | v2 + noslip 5 | ~0 | ~0 | 1.86 m/s | 2.4 mm |
| 4-sphere stool | v1 | 2.6 mm/s | 12.5 mm/s | 1.88 m/s | 1.7 mm |
| 4-sphere stool | v2 | 0.05 mm/s | 0.10 mm/s | 1.82 m/s | 2.5 mm |

v2 cuts creep by about 100x. noslip removes the 0.1 mm/s residual at about 20% step cost, which is not needed. A 5 cm drop of the 30 kg
sole penetrates < 5 mm (unit test). v2 env step cost: t1 about 2.3x v1 (0.7 vs 0.31 ms per env-tick; elliptic cone plus
contact-force slip); go2 is faster (impratio 100 -> 10).

## reward references
- Energy -> gaits: Fu, Kumar, Malik, Pathak, "Minimizing Energy Consumption Leads to the Emergence of Gaits in Legged
  Robots", CoRL 2021 (arXiv 2111.01674). Verified via search on 2026-09-26.
- The values below were recalled from the public configs and NOT re-verified line by line. Treat them as orders of magnitude:
  legged_gym (Rudin et al. 2021) feet_air_time +1.0 with a 0.5 s threshold, action_rate -0.01, lin_vel_z -2, ang_vel_xy -0.05;
  unitree_rl_gym H1/G1: contact_no_vel -0.2 (squared stance-foot velocity), feet_swing_height -20 (target 0.08 m),
  phase-matched contact +0.18, action_rate -0.01; humanoid-gym (Gu et al. 2024): foot_slip -0.05 (on sqrt of speed),
  feet_clearance +1.0, feet_contact_number +1.2, feet_air_time +1.0, action_smoothness -0.002.
  Our v1 feet_slip (-0.05 x v^2) is about 10x weaker than unitree's contact_no_vel. gait_v2 uses a linear penalty, so small
  residual slips are also discouraged.

## baseline: v1 trackers, v1 physics vs v2 physics (zero-shot), protocol v2, 5 seeds
Raw: `artifacts/runs/contact_v2/val/<body>_v1trk_phys{v1,v2}.json`. Table: `python3 scripts/contact_table.py <files>`.
Zero-shot v2 physics alone lowers the slip ratio (t1 0.86 -> 0.58, go2 0.26 -> 0.08, anymal 0.34 -> 0.17), but the
humanoids still shuffle (duty 0.8-0.95, air 0.02-0.03 s) and their turn ratio collapses (t1 0.02, h1 0.04).
go2's v1 tracker passes the contact gate in v2 physics zero-shot.

## SUMMARY: v1 vs v2 (state 2026-09-27 ~00:50; raw: `artifacts/runs/contact_v2/val/`, table: `python3 scripts/contact_summary.py`)
v1 = the v1 tracker in contact_v1 physics (the currently deployed pair). v2 = the installed contact_v2 tracker in contact_v2 physics. Validation protocol v2,
5 seeds x 5 trials; slip ratio = forward-trial contact-point stance slip / body speed (target < 0.15).

| body | v2 tracker | no-fall v1 -> v2 | fwd ratio v1 -> v2 | turn ratio v1 -> v2 | **slip ratio** v1 -> v2 | duty v1 -> v2 | swing apex cm v1 -> v2 | CoT v1 -> v2 | legacy gate v1 / v2 | contact gate v2 |
|---|---|---|---|---|---|---|---|---|---|---|
| t1 | v2trk | 1.00 -> 1.00 | 0.74 -> 0.99 | 0.30 -> 0.02 | 0.86 -> **0.15** | 0.85-1.00 -> 0.72-0.82 | 1.4 -> 6.6 | 3.42 -> 1.39 | False / False | False |
| h1 | v2trk-r2 | 1.00 -> 1.00 | 0.69 -> 0.90 | 0.33 -> 0.03 | 0.57 -> **0.13** | 0.88-0.94 -> 0.80-0.86 | 1.1 -> 4.2 | 2.74 -> 3.03 | False / False | False |
| g1 | v2trk-r1 | 1.00 -> 1.00 | 0.91 -> 1.10 | 0.08 -> 0.02 | 0.42 -> **0.08** | 0.80-0.87 -> 0.76-0.78 | 4.0 -> 8.5 | 1.53 -> 1.19 | False / False | False |
| go2 | v2trk | 1.00 -> 1.00 | 1.05 -> 1.04 | 0.59 -> 0.75 | 0.26 -> **0.02** | 0.48-0.76 -> 0.44-0.55 | 3.0 -> 0.6 | 2.34 -> 0.65 | True / True | False |
| anymal_c | v2trk | 1.00 -> 1.00 | 0.98 -> 1.09 | 0.72 -> 0.80 | 0.34 -> **0.03** | 0.76-0.79 -> 0.46-0.65 | 3.0 -> 2.5 | 1.06 -> 0.34 | True / True | True |
| hexapod6 | v2trk-v2c-rejected | 1.00 -> 1.00 | 0.88 -> 1.05 | 0.93 -> 0.04 | 0.85 -> **0.17** | 0.00-0.87 -> 0.00-0.71 | 2.1 -> 2.9 | 7.45 -> 6.38 | True / False | False |

Installed contact_v2 trackers (gitignored weights; `artifacts/trackers/<body>/contact_v2/actor.pt`, selectable with contact="v2" /
$RRP_CONTACT_MODEL=v2; v1 stays the default): t1 iter5099 (sha 87e233c6c01f449b), h1 r2 iter5499 (92bff757bf334109), g1 r1 iter3999
(8b8a99cbc833f150), go2 iter3499 (48c632d75b2795c9), anymal_c v2c iter2499 (2a16532bbd07f7ab; alpha0 snapshot 3796df094f2ce85e).
g1 round 2 (`artifacts/runs/contact_g1_v2_r2`, iter5499, alpha stayed 0) validated as a close alternative: slip 0.08, CoT 0.90, fwd 0.96, apex 5.7 cm, turn 0.01 (`val/g1_v2trk-r2_physv2.json`); round 1 stays installed (better tracking and clearance).
hexapod6: no acceptable learned v2 tracker (3 rejected runs); the scripted CPG tripod stays (slip 0.20 in v2 physics).
Arc trial (walk + turn): yaw-rate ratio v1 -> v2: t1 0.71 -> 0.98, h1 0.13 -> 0.64, g1 0.49 -> 1.15, go2 1.05 -> 1.10, anymal_c 0.63 -> 1.05;
no falls in any stand/arc/push trial.

Findings:
1. Skating is fixed on every accepted body: slip ratio 0.26-0.86 -> 0.02-0.15. Humanoids now step (air 0.07-0.08 s vs 0.02-0.04;
   apex 4-9 cm vs 1-4 cm) and forward tracking improved. Only anymal_c passes the full contact gate.
2. The humanoids fail the gate only on TURN IN PLACE (turn ratio 0.02-0.03). v1 "turned" by twisting its feet on the floor (0.30/0.33 in v1
   physics; the v1 trackers also score 0.02-0.04 in v2 physics). In-place stepping turns were not learned. Arcs work (above).
3. Priors -> natural schedule (lead's claim), checked with alpha=0 vs final validations: go2 (slip 0.10 -> 0.02, CoT 3.56 -> 0.65; the swing
   flattened to 0.6 cm), anymal_c (0.10 -> 0.03, CoT 1.40 -> 0.34, apex 14.5 -> 2.5 cm), t1 (0.28 -> 0.15, CoT 2.79 -> 1.39, apex 7.9 -> 6.6 cm).
   None collapsed back into shuffling or skating. Energy minimisation lowers swing height, severely on go2. h1/g1 never passed their
   training-window gates, so they stayed at alpha=0.
4. Reward bugs found and fixed along the way (each is a recorded failure): gait_v2a touchdown-apex penalty -> stander basin; summed-over-feet
   slip/clearance -> legs held up or kicked out (non-bipeds; gait_v2c uses per-foot means); adaptive-lr ceiling 1e-2 -> warm-start
   collapse (`--max-lr`); sigma 0.25 too flat for small robots (range-scaled sigma).

Recommended next step for downstream data (the lead decides): regenerate legged datasets under contact_v2 for go2 and anymal_c now (anymal_c
passes everything; go2 passes everything except the mm swing clearance, which is fine on flat ground). For t1/h1/g1, the v2 trackers are
clearly better walkers than v1 (planted feet, real steps, arcs), but they cannot turn in place. Either restrict the teacher to arcs /
forward+turn commands (|vx| >= ~0.1 while turning), or first fix in-place turning (candidates: a turn-in-place curriculum with more
pure-turn samples plus a stepping-turn prior; a yaw-aligned foot placement reward), then regenerate. Keep hexapod6 on the CPG.
Next track item after this gate (strategy W1): actuator realism (armature, joint friction/damping, torque-speed limits) and 0-30 ms
latency randomisation (v2 currently randomises 0-8 ms).

## W1 follow-up (lead D-101, 2026-09-27)
### (1) go2 permanent clearance floor: GATE PASSED
New PERMANENT term `clearance_floor` (`--reward-set clearance_floor=W`): at each touchdown during a moving command,
W x clip((floor - apex)/floor, 0, 1)^2 per foot, where floor = 0.6 x swing_height (go2 3.6 cm, humanoids 4.8 cm). It is never scheduled.
- cf (W=-2): host lease 1790495753_12bd09, `artifacts/runs/contact_go2_cf`, warm from go2 contact_v2 iter3499, 1500 iters, lr ceiling 1e-3,
  schedule restarted (alpha back at 1.0 by ~700). Validation `val/go2_v2cf_physv2.json`: slip 0.03, apex 2.2 cm, CoT 0.71, turn 0.66,
  contact gate passes, but the apex is below the 3.6 cm floor.
- cf2 (W=-6): `artifacts/runs/contact_go2_cf2`, 800 iters from cf with alpha fixed at 1. **Installed** as `artifacts/trackers/go2/contact_v2/actor.pt`
  (sha af3f06f4e029e9f9; the previous no-floor tracker is at `~/work/rrp-data/contact-v1-actors/go2_contact_v2_iter3499_noflor.pt`).
  Validation `val/go2_v2cf2_physv2.json`: no falls, fwd 1.03, turn 0.79, **slip 0.02**, duty 0.40-0.60, air 0.25 s, **apex 3.8 cm**, **CoT 0.88**
  (v1 2.34; no-floor v2 0.65), legacy gate and contact gate pass. The floor costs about 35% CoT relative to the flat-swing policy.
- Videos (reviewed: the swing foot clears the floor visibly): `2026-09-27_contact_go2_forward_floor-vs-nofloor_iter1499_ok_ok.mp4`,
  `2026-09-27_contact_go2_{forward,turn}_v1-vs-v2floor_iter799_ok_ok.mp4`.
### (2) turn in place: running (t1, h1; g1 queued)
PERMANENT `yaw_slip` (-0.5 x sum over stance feet of |foot yaw rate|: no pivoting on a planted foot) and `turn_step` (+1 x agreement of
contacts with the alternating gait clock during pure-turn commands), plus clearance_floor -2. Curriculum: pure-turn probability 0.35;
yaw-rate range +-[0.3, 1] x scale x wz_max, scale starting at 0.4 and widened by 0.1 each 25-iteration window with turn ratio >= 0.6 and falls
<= 0.2 (`turn_scale` in train log and checkpoint). Humanoid alpha thresholds as in t1 r2. Runs (PEER, dir wt/contact3, warm from the
installed contact_v2 trackers): t1 lease 1790497102_81e726 `artifacts/runs/contact_t1_turn`, h1 lease 1790497102_12a17b
`artifacts/runs/contact_h1_turn`, 2500 iters each.
Attempt a (stopped at iter ~650-750; kept as `contact_{t1,h1}_turn_a` on the peer): the window turn ratio FELL (t1 0.76 -> 0.30, h1 0.77 -> 0.36).
Cause: the sigma 0.1 yaw kernel. Ignoring a 0.1-0.2 rad/s curriculum command costs only 0.2-0.7 per step, which is less than the new anti-pivot penalty.
Fix: separate `sigma_ang` (0.02: ignoring 0.2 rad/s now costs 1.7). Attempt b: peer dir wt/contact5, t1 lease 1790498747_1b719b,
h1 lease 1790498748_824e00, same outputs, `--reward-set clearance_floor=-2,yaw_slip=-0.5,turn_step=1,sigma_ang=0.02`.

Attempt c (after a and b): a deterministic check of the attempt-b checkpoint (`turndiag.py`) showed the policy stands with both feet down on
a pure yaw command (turn ratio 0.01 at 0.1/0.2/0.36 rad/s). The training-window turn ratio came from exploration noise and
carry-over from walking. It is an exploration problem, and the sharp exponential kernel gives almost no gradient far from the target. Added
permanent `turn_lin` (dense yaw progress during pure turns: x clip(w sign(c)/|c|, -0.5, 1.2)). Attempt c:
`--reward-set clearance_floor=-2,yaw_slip=-2,turn_step=2,turn_lin=1.5,sigma_ang=0.03`, curriculum start 0.5, init std 0.35. Peer dir
wt/contact6, t1 lease 1790499833_d772c7, h1 lease 1790499833_bfbb2f, `artifacts/runs/contact_{t1,h1}_turn` (a/b kept as `_turn_a`/`_turn_b`).

Attempt c result. t1 WORKS: the window turn ratio went 0.79 -> 1.0-1.28 and the curriculum widened to full range by iter ~520. Deterministic check at
iter 591: a stepping turn (duty 0.5-0.78, small foot yaw slip) that overshot about 2x. Full validation of the iter-1100 snapshot
(`val/t1_turnc-it1100_physv2.json`): **turn ratio 0.02 -> 1.35**, no falls, fwd 0.99, duty 0.50-0.53, apex 9.9 cm. But forward slip ratio
0.15 -> 0.27 and CoT 1.39 -> 3.25: alpha stayed 0 all run (the training-window track error of 0.56-0.6 never met the 0.5 threshold), so the priors were on
and the energy terms off. h1 did NOT work: window about 0.4, deterministic validation turn 0.10 (`val/h1_turnc-it1250_physv2.json`). Both stopped
(t1 at ~1150, h1 at ~1300; kept as `contact_{t1,h1}_turn_c`).
Phase 2 for t1: `artifacts/runs/contact_t1_turn2` (peer lease 1790502983_4e0ad0), init from the turn policy, alpha FIXED at 1 (the
accepted t1's setting), full turn range, same turn terms, 1200 iters, to recover slip and CoT while keeping the turn.
**t1 phase 2 result: TURN GATE PASSED.** Run `artifacts/runs/contact_t1_turn2` (1200 iters, alpha fixed at 1, full turn range). Installed as
`artifacts/trackers/t1/contact_v2/actor.pt` (sha 0d77322c0248e019). The previous no-turn tracker is kept at
`artifacts/runs/contact_t1_turn2/before.pt` = `~/work/rrp-data/contact-v1-actors/t1_contact_v2_iter5099_noturn.pt`.
Validation `val/t1_turn2_physv2.json` vs the previous t1 v2 (`val/t1_v2trk_physv2.json`): **turn in place 0.02 -> 1.15**, no falls in any trial,
**forward slip 0.15 -> 0.11**, duty 0.72-0.82 -> 0.69-0.72, air 0.08 -> 0.10 s, apex 6.6 -> 6.4 cm, contact gate FALSE -> **TRUE**.
Regressions (honest): forward ratio 0.99 -> 0.84 (v1 was 0.74), CoT 1.39 -> 1.77, arc yaw ratio 0.98 -> 0.66 and arc vx 1.00 -> 0.72.
Video (reviewed): the turn is a stepping turn, but with a wide, lunging stance; functional, not natural.
`artifacts/video/2026-09-27_contact_t1_turn_turn-before-vs-after_iter1199_ok_ok.mp4`, `..._t1_{turn,forward}_v1-vs-v2turn_iter1199_ok_ok.mp4`.
If W8 mostly needs arcs/forward+turn, the previous tracker is the better choice for that; both are kept.

g1 with the t1 recipe (`contact_g1_turn_c`, stopped at ~815): the same stall as h1 (window 0.26-0.40). Deterministic: both h1 and g1 STAND with
both feet down under pure yaw commands (turn 0.03/0.0, duty 1.0/1.0). They never start stepping from rest, although they turn well in arcs.
**Arc-to-in-place curriculum** (`--turn-vx0 V`): pure-turn commands start with forward speed V (0.2 m/s). Each window with turn ratio >= 0.6
shrinks it by V/4 until 0, and only then widens the yaw range. h1 on the HOST (peer load > 15), with automatic resume after watchdog sheds
(`scripts/contact_host_resumable.sh`, bounded to 8 launches): `artifacts/runs/contact_h1_turnarc`, 2500 iters.
**FAILED**: the curriculum reached vx 0.05 m/s by iter ~300, then stalled (window turn 0.4-0.5). Final validation (`val/h1_turnarc_physv2.json`):
turn 0.03, fwd 0.68 (was 0.90), slip 0.22, CoT 5.85. Deterministic: h1 barely lifts its feet from rest even at vx 0.2 (duty 0.9, both feet down 88%).
h1's low-speed gait is still near-shuffle, so there is no stepping gait to turn with. Recommendation for h1/g1: first train a real low-speed
stepping gait (for example a stepping-in-place command with a stronger clock/air-time term at |v| < 0.15), then the turn curriculum. Not attempted
further in this round; g1 turning was not rerun.

### (3) actuator realism: implemented; baseline measured; fine-tunes running
`src/rrp/physics/actuator.py` (actuator_v2; test `tests/unit/test_actuator.py`): armature max(model, 4e-4 x peak torque) x U(0.8, 1.2);
dof damping and frictionloss = model + U(0, 1%) of peak torque; linear torque-speed derating from 0.5 vmax to vmax (vmax ASSUMED:
humanoid 20, go2 30, anymal_c 12, procedural 8 rad/s; not in the MJCFs), enforced exactly per substep on the affine PD; 0-30 ms latency
per episode through a cross-tick pending-target queue. Train `--actuator v2` (randomised); validate `--actuator v2 --latency-ms L` (nominal).
Baseline (installed trackers, no fine-tune; `val/*_v2trk-act2-lat{0,15,30}_physv2.json`):
| body | no-fall 0 / 15 / 30 ms | fwd ratio 0 / 15 / 30 | slip ratio 0 / 15 / 30 | contact gate 0 / 15 / 30 |
|---|---|---|---|---|
| t1 | 1.00 / 1.00 / 1.00 | 0.80 / 0.78 / 0.74 | 0.08 / 0.08 / 0.09 | F / F / F (turn) |
| h1 | 0.80 / 1.00 / **0.24** | 0.84 / 0.57 / 0.00 | 0.16 / 0.07 / 0.24 | F / F / F |
| g1 | 1.00 / 1.00 / **0.40** | 0.99 / 0.96 / 0.00 | 0.07 / 0.11 / 0.16 | F / F / F |
| go2 | 1.00 / 1.00 / 1.00 | 0.98 / 0.90 / 0.85 | 0.02 / 0.02 / 0.02 | T / T / T |
| anymal_c | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 0.95 | 0.03 / 0.04 / 0.03 | T / T / T |
Short actuator_v2 fine-tunes (600 iters, `--actuator v2`, alpha fixed at 1). The host watchdog shed them, so they were rerun on the peer (wt/contact7).
Results (`val/{go2,anymal_c}_v2act2ft*-lat*`):
- go2 (with clearance_floor=-6): no-fall 1.00 at 0/15/30 ms; turn 0.74 -> 0.85/0.82/0.79; fwd 1.02/0.93/0.87 (before: 0.98/0.90/0.85);
  slip 0.03/0.04/0.02; CoT 0.88 -> 0.97. Contact gate passes at every latency. The change is neutral to slightly positive: go2 was already robust.
- anymal_c WITHOUT a clearance floor: the swing flattened to 1.1 cm and the contact gate FAILED, so it was not installed (`..._act2_nofloor`). Rerun with
  clearance_floor=-6 (`artifacts/runs/contact_anymal_c_act2`): no-fall 1.00, contact gate passes at 0/15/30 ms, apex 4.8-6.2 cm, slip 0.05, but
  fwd 0.87/0.85/0.79 (installed: 1.00/1.00/0.95) and CoT 0.56 (installed 0.36-0.39). Not installed: the installed anymal_c is already robust to
  actuator_v2 at 30 ms and tracks better.
- Decision: actuator_v1 stays the deployed physics. The go2/anymal_c actuator fine-tunes are recorded as candidates for when actuator_v2 becomes the default.
- Bug found: `--alpha-schedule fixed:<a>` still ran the alpha gate. Fixed. Audit: the go2 cf2, go2 act2 and anymal_c act2 runs stayed at alpha 1.0
  throughout (unaffected). t1 phase 2 had drifted to 0.5, was stopped (`contact_t1_turn2_alphabug`) and was restarted (peer lease 1790504263_a43726).
- g1 actuator fine-tune (`artifacts/runs/contact_g1_act2`, 800 iters, alpha fixed at 0 as trained, clearance_floor=-2): `val/g1_v2act2ft-lat*`.
  **The 30 ms collapse is fixed**: no-fall 0.40 -> 1.00, fwd 0.00 -> 0.72, slip 0.16 -> 0.08. Cost: fwd at 0/15 ms 0.99/0.96 -> 0.79/0.76. Turn still 0.01.
- t1 turn-trained tracker under actuator_v2 (`val/t1_turn2-act2-lat*`): no falls and turn 0.66/0.99/1.01, but fwd 0.55/0.37/0.37 (the pre-turn t1
  held 0.80/0.78/0.74). The turn-trained t1 is FRAGILE to realistic actuators. Running: t1 fine-tune with actuator_v2 plus the turn terms (host,
  `artifacts/runs/contact_t1_act2`, 1000 iters, alpha 1).
- t1 fine-tune with actuator_v2 plus the turn terms (`artifacts/runs/contact_t1_act2`, 1000 iters): fwd under actuator_v2 recovers to 0.87/0.76/0.67 at
  0/15/30 ms (turn-trained: 0.55/0.37/0.37), but **turn in place is lost (0.00)** and slip is 0.14-0.16. Within this budget, turning and
  actuator robustness conflict for t1. Not installed. The installed t1 (turn-trained) stays: it passes the gate under the deployed actuator_v1 physics.
- h1 actuator fine-tune (`artifacts/runs/contact_h1_act2`, 800 iters): no-fall 0.80/1.00/0.24 -> 0.92/0.92/0.24, fwd 0.84/0.57/0.00 -> 1.12/0.48/0.24.
  **The 30 ms collapse is NOT fixed**; not installed. h1 needs a longer actuator_v2 run, or a gait that is not near-shuffle (see (2)).
- The validations of the t1/h1 fine-tunes ran on the peer (host shed them twice at memory PSI 32-33).

#### (3) robustness summary: no-fall at 30 ms latency (actuator_v2), installed tracker -> best fine-tune
anymal_c 1.00 (already robust; fine-tune not needed) | go2 1.00 -> 1.00 (turn +0.05-0.11) | g1 **0.40 -> 1.00** (fwd 0 -> 0.72) |
t1 (turn-trained) 1.00, but fwd 0.37 -> 0.67 and turning lost in the fine-tune | h1 **0.24 -> 0.24 (not fixed)**. Pre-turn t1: 1.00, fwd 0.74.

## W1 round 3 (lead D-103, 2026-09-27)
### (1) sourced actuator limits: DONE
Every accepted body has manufacturer per-joint limits (`<limit effort velocity>`) in its official URDF. They are now used in
`rrp.physics.actuator.SOURCED` (torque = min(adapter, source); speed = URDF velocity used as the zero-torque speed of the envelope, which is our
modelling choice). Sources (sha256 first 16 hex of the files read on 2026-09-27):
| body | source | hip / knee / ankle torque (N m) | speed (rad/s) |
|---|---|---|---|
| t1 | BoosterRobotics/booster_gym resources/T1/T1_serial.urdf (027a5333ce4ed0a1); same torques as menagerie t1.xml | 45 pitch, 30 roll/yaw / 60 / 20 pitch, 15 roll | 12.5, 10.9 / 11.7 / 18.8, 12.4 |
| h1 | unitreerobotics/unitree_ros robots/h1_description/urdf/h1.urdf (ebd495cba7887406) | 200 / 300 / 40 | 23 / 14 / 9 |
| g1 | unitreerobotics/unitree_ros robots/g1_description/g1_29dof.urdf (e1dc89366bf96aa3) | 88 / 139 / 35 | 32 / 20 / 30 |
| go2 | unitreerobotics/unitree_ros robots/go2_description/urdf/go2_description.urdf (7d19fe48e2e689ee) | 23.7 hip, thigh / 45.43 calf | 30.1 / 15.7 |
| anymal_c | ANYbotics/anymal_c_simple_description urdf/anymal.urdf (3902c3957ac82176) | 80 all | 7.5 all |
| procedural | none | model | ESTIMATE 8 rad/s (labelled `limit_source: estimate`) |
Findings: (a) **our t1 adapter torques are 2-3x the manufacturer's**: the legacy gains table set hip 60, knee 130, ankle 50 N m; Booster (and the menagerie
t1.xml) say hip 45/30, knee 60, ankle 20/15. The ideal actuator_v1 physics every t1 tracker was trained in therefore overstates t1's strength.
(b) My earlier speed estimates were too high for anymal_c (12 vs 7.5) and h1 ankles (20 vs 9). (c) g1 hip roll: the menagerie MJCF allows 139 N m, the
URDF 88; the conservative min is used. New actuator mode `v1lat` = ideal joints + sourced torque/speed limits + 0-30 ms latency, as the lead asked.
### (2) h1/g1 slow stepping gait then turning: running (h1)
PERMANENT `stance_cap` (-1): during moving or turning commands, each foot in contact longer than 0.75 of a gait period costs
clip((t - cap)/period, 0, 1), which sets a minimum swing frequency. Plus `--slow-frac 0.3` (30% of walking commands rescaled to 0.05-0.2 m/s), the turn
terms and curriculum, clearance_floor -2. h1: host (resumable), `artifacts/runs/contact_h1_step`, 3000 iters from the installed h1. g1 follows.
### (3) t1 turning + latency under v1lat: running
`artifacts/runs/contact_t1_lat` (host, resumable), from the installed turn-trained t1, `--actuator v1lat` (sourced limits, 0-30 ms latency), turn
terms, alpha 1, 1500 iters. Baseline of all installed trackers under v1lat at 0/30 ms: queued on the peer (when load < 15).

## runs
### gait_v2a: failed_hypothesis (stopped at iter 500-1400)
t1/h1 (host) and g1/go2 (peer) with a clearance PENALTY at touchdown ((target - apex)/target)^2 x -2. After 1200-1400 iters
(t1/h1) and about 800 (go2), every run was a stander: gate-window track_rel_err 1.02-1.15 (a zero-velocity policy scores
about 1.0). The t1 checkpoint at iter 1200 lifts a foot on the clock and tips over (validation: falls in 9/10 episodes).
Diagnosis (scratchpad rdiag: the v1 walker vs a zero-action stander on the same forward command in v2 physics, return per step):
the gait_v2a optimum is still walking (go2 1.83 vs 1.24; t1 3.13 vs 2.98, a thin margin). But a touchdown-only
penalty is avoided entirely by never lifting a foot, and early foot flicker pays it, which makes a stander basin.
Logs: `artifacts/runs/contact_{t1,h1}_v2a/` (host), `artifacts/runs/contact_{g1,go2}_v2a/` (peer shared store).
### gait_v2b (current)
The clearance term is now a positive humanoid-gym-style swing-height reward: sum over feet in a bounded swing
(off the floor for < 0.6 period, moving command) of min(h / target, 1); w0 = 1.0 bipeds, 0.5 others; still a decaying prior.
Margins: go2 walker 2.42 vs stander 1.27, t1 3.59 vs 2.99.
Two t1 runs on the host test warm start (A/B):
- `contact_t1_v2_scratch`: lease 1790474219_eaad24, from scratch, 4000 iters x 4x48.
- `contact_t1_v2_warm`: lease 1790474219_cd8a6d, `--init-actor artifacts/trackers/t1/actor.pt --init-std 0.3` (actor and obs
  normaliser from the v1 tracker, sha 185df8519a8ca280; fresh critic); labelled `init_from` in meta.
A/B result at iter ~520: scratch was still a stander (fall 0.38-0.69, track_rel_err 1.05-1.08, slip 0.8-1.0), while warm had
fall 0.12-0.21, track_rel_err 0.43-0.46, slip ratio 0.64 -> 0.53 and falling. The scratch run was stopped at iter ~560 to free CPU
(log kept: `artifacts/runs/contact_t1_v2_scratch/`). **Decision: all v2 trackers are warm-started from their v1 tracker**
(label: fine-tuned from contact_v1 tracker, `init_from` in meta). A from-scratch v2 humanoid is an open item and needs
far more samples than this setup gives (legged_gym-style runs use about 4096 envs).
Warm runs:
- t1: host lease 1790474219_cd8a6d, `artifacts/runs/contact_t1_v2_warm` (4000 iters).
- h1: peer lease 1790475226_5b8177, `artifacts/runs/contact_h1_v2_warm` (4000; 6 h cap, `--resume` if cut).
- g1: peer lease 1790475227_f4019d, `artifacts/runs/contact_g1_v2_warm` (4000; same).
- go2: host lease 1790475352_0f5f7a, `artifacts/runs/contact_go2_v2_warm` (3500).
- queued: anymal_c (2500), hexapod6 (2000), warm.
v1 actors for the peer runs: `artifacts/runs/contact_v1_actors/<body>/actor.pt` in the shared peer store (same sha as local).

## results (contact_v2 trackers, validation protocol v2 in contact_v2 physics, 5 seeds x 5 trials)
### go2: completed (verified)
Run `artifacts/runs/contact_go2_v2_warm` (host lease 1790475352_0f5f7a, 3500 iters, warm from v1 iter1199). Installed at
`artifacts/trackers/go2/contact_v2/actor.pt` (gitignored; sha256 48c632d75b2795c9...). alpha=0 snapshot:
`artifacts/runs/contact_go2_v2_warm/actor_alpha0.pt` (sha 6839f2e84238eec7..., iter 324).
Schedule: alpha advanced from iter 324, backed off twice at alpha 0.4 (fall-rate spikes 0.26/0.32), and reached 1.0 at iter 699.
Training-window CoT went 5.3 -> 1.7 (iter 699) -> 0.93 (iter 3499); slip ratio 0.17 -> 0.08.
Validation (`artifacts/runs/contact_v2/val/go2_*`): v2 final: slip ratio 0.02 (v1: 0.26), no falls, fwd 1.04, turn 0.75,
duty 0.44-0.55, air 0.18 s, CoT 0.65 (v1: 2.34). Legacy gate passes. **The contact gate fails only on clearance**: mean swing apex
is 6 mm (target 0.3 x 6 cm). The alpha=0 policy lifted 10 cm, ran in a low crouch, and had CoT 3.56 and slip 0.10.
So the priors-to-natural shift kept planted stance feet and a stepping trot (no return to shuffling or skating), removed the crouch and cut
CoT 5x, but it flattened the swing to mm clearance. That is fine on flat ground and fragile on rough terrain. Candidate fix (not run):
make a minimum clearance a permanent term like slip, or raise the clearance floor from 10% to about 50%.
Videos (reviewed): `artifacts/video/2026-09-26_contact_go2_{forward,turn}_v1-vs-v2_iter3499_ok_ok.mp4`,
`artifacts/video/2026-09-26_contact_go2_forward_alpha0-vs-final_iter3499_ok_ok.mp4`.

### t1: completed (verified; gate not passed)
Run `artifacts/runs/contact_t1_v2_warm` (host lease 1790474219_cd8a6d, warm from v1 iter2999). It was SHED by the host memory watchdog
(system memory PSI from another project) at iter ~3605 of 4000. A resume was shed again within seconds (lease 1790481580_97c00d),
so the final tracker is the iter-3599 export: `artifacts/trackers/t1/contact_v2/actor.pt` (sha256 b3b006f357d62d50...).
alpha stayed 0 for the whole run: the training-window gate (track <= 0.35, fall <= 0.10, slip <= 0.20) was never met (windows
at about 0.43 / 0.15-0.25 / 0.27-0.36). There is no alpha0 snapshot because alpha never moved.
Validation (`artifacts/runs/contact_v2/val/t1_*`): slip ratio 0.86 (v1/v1 physics) -> 0.20; swing apex 1.4 -> 7.2 cm; air time
0.02 -> 0.08 s; duty 0.85-1.0 -> 0.67-0.73; fwd ratio 0.74 -> 0.91; no falls; CoT 3.42 -> 2.96. **Gate fails**: turn-in-place
ratio 0.02 (v1 turned 0.30 by skating in v1 physics; in v2 physics the v1 tracker also turns only 0.02), and slip is 0.20 > 0.15.
Videos (reviewed: v2 visibly lifts and places feet, v1 glides): `artifacts/video/2026-09-26_contact_t1_{forward,turn}_v1-vs-v2_iter3599_ok_ok.mp4`.
Round 2 (queued with admission retries): `artifacts/runs/contact_t1_v2_r2`, resumed from the iter-3599 checkpoint to 5100 iters with
humanoid gate thresholds (advance track 0.5 / fall 0.25 / slip 0.3; back off 0.65 / 0.4 / 0.45), so the schedule is exercised on a humanoid.

### t1 round 2: completed (verified; the contact gate misses by slip 0.151 vs 0.15 and turn 0.02)
`artifacts/runs/contact_t1_v2_r2` (host lease 1790482385_a87fcf): resumed from the iter-3599 checkpoint to 5100 with humanoid gate
thresholds. alpha first advanced at iter 3824 (alpha0 snapshot `actor_alpha0.pt`, sha 6cfc2121b2b2c6e5..., iter 3824) and reached 1.0 by
about 4800. Training-window CoT went 2.8 -> 2.0 and slip 0.27 -> 0.23. The final is installed at `artifacts/trackers/t1/contact_v2/actor.pt` (sha
87e233c6c01f449b..., iter 5099). The iter-3599 tracker is kept at `~/work/rrp-data/contact-v1-actors/t1_contact_v2_iter3599.pt`.
Validation (`val/t1_v2alpha0_physv2.json` vs `val/t1_v2trk_physv2.json`): alpha=0 -> final gives slip ratio 0.28 -> **0.151**, CoT 2.79 -> **1.39**,
apex 7.9 -> 6.6 cm (above the 2.4 cm floor), duty 0.66-0.75 -> 0.72-0.82, air 0.08 -> 0.08 s, fwd 1.11 -> 0.99, no falls. **On t1 the lead's
claim holds: as the priors fade, the stance feet stay planted, swings stay clear and CoT halves; it does not collapse into shuffling.** Turn
in place is still 0.02, so the gate fails. Videos (reviewed): `2026-09-26_contact_t1_{forward,turn}_v1-vs-v2_iter5099_ok_ok.mp4`,
`2026-09-26_contact_t1_forward_alpha0-vs-final_iter5099_ok_ok.mp4`.
### anymal_c run 1: REJECTED (failure example)
`artifacts/runs/contact_anymal_c_v2_warm_r1` (2500 iters, lr ceiling 1e-2, KL spikes to 9.5, alpha stayed 0). Validation: no falls, fwd 0.85,
turn 1.05, but slip 0.29 (the v1 tracker zero-shot in v2 physics gives 0.17), CoT 2.85 (1.03), swing apex 32 cm. The video shows the shanks kicked high
and backwards in swing, an unnatural gait. Moved to `artifacts/trackers/anymal_c/contact_v2_rejected_iter2499/` (not loadable as contact_v2).
Video with a FAILURE label in INDEX: `2026-09-26_contact_anymal_c_{forward,turn}_v1-vs-v2_iter2499_ok_ok.mp4`. Rerun r2 (lease
1790485178_e1b7ca, `artifacts/runs/contact_anymal_c_v2_r2`) uses `--max-lr 1e-3` and the range-scaled tracking sigma 0.16.
### h1 round 1: completed (alpha 0; gate fails)
`artifacts/runs/contact_h1_v2_warm` (peer, 4000 iters, sha fca86e8f89996326...; pulled to the host run dir). Validation
`val/h1_v2trk-r1_physv2.json`: slip ratio 0.57 (v1) -> 0.18, fwd 0.69 -> 1.14, apex 1.1 -> 4.2 cm, air 0.03 -> 0.08 s, duty 0.63-0.80, CoT 2.74 -> 2.84,
but no-fall 0.92 and turn 0.18 (v1 0.33). Round 2 (humanoid thresholds, lr ceiling 1e-3): peer lease 1790488049_a8d76d, dir wt/contact2,
`artifacts/runs/contact_h1_v2_r2` to 5500. At iter 4486 alpha is still 0 (training-window track_rel_err 0.61 > 0.5).
### hexapod6 run 3 and anymal_c round 2: REJECTED; the reward flaw found (gait_v2c)
hexapod6 run 3 (`contact_hexapod6_v2_warm`, sigma 0.0225): fwd 1.05, turn 1.07, slip 0.46, duty 0.09-0.33, CoT 13.5. Legs are held off the ground.
anymal_c round 2 (`contact_anymal_c_v2_r2`): slip 0.16, fwd 0.98, turn 1.05, but shanks kicked out (apex 21.5 cm, duty 0.31-0.43, CoT 1.98).
Both are rejected (`trackers/<b>/contact_v2_rejected_*`, FAILURE lines in INDEX). The v1 hexapod6 tracker also holds its front legs up (duty min 0.00).
**Diagnosis:** the stance-slip penalty and the swing-height reward were SUMS over feet, so keeping fewer feet on the ground reduced
the penalty and raised the reward. Bipeds are protected by the contact-phase clock; go2 happened not to exploit it.
**gait_v2c (non-bipeds only):** slip = 2 x mean over feet in contact (x2 keeps a trot's scale), clearance = mean over swinging feet. Runs on
the PEER (load 6; host limit fell to 9 CPU): anymal_c lease 1790489199_66f489, hexapod6 lease 1790489200_781fbf, dir wt/contact3,
`artifacts/runs/contact_{anymal_c,hexapod6}_v2c`.
### hexapod6 run 2: failed (stander) -> run 3
With the lr cap, the run was stable but converged to standing (track_rel_err 0.88-0.96 at iter 1000): the sigma 0.25 tracking kernel is too
flat for a 0.18 m/s command. gait_v2 now uses sigma = min(0.25, (0.5 vx_max)^2) for non-bipeds (hexapod6 0.0225, anymal_c 0.16, go2 0.25
unchanged). The v1 walker then beats a stander 1.84 vs 1.30 per step. Run 3: lease 1790484707_890cde, `artifacts/runs/contact_hexapod6_v2_warm`.
Kept: `..._v2_warm_collapsed` (run 1), `..._v2_warm_stander` (run 2).
### hexapod6 run 1: failed (training instability, not physics)
`artifacts/runs/contact_hexapod6_v2_warm_collapsed`: the policy collapsed at iter ~185-200. Per-iteration KL went 25 -> 340 -> 1341 and the fall
rate 0 -> 0.9, with episodes of 30-110 ticks, and it never recovered. Physics was checked: the v1 hexapod actor plus N(0, 0.3) exploration noise in the
v2 training env (DR on) has no falls, no BADQACC warnings and |qvel| <= 18, the same as v1. Cause: the adaptive-lr rule multiplies lr by
1.5 per minibatch while KL < desired/2, up to a hard-coded 1e-2. A converged warm-started policy has tiny KL, so the lr reaches 1e-2 within an
iteration (every warm run logged max lr 0.01; go2/anymal_c had KL spikes of 7.9/9.5 and recovered). Fix: `--max-lr` (default 1e-2
unchanged); hexapod6 restarted with `--max-lr 1e-3`. go2/t1/anymal_c/h1/g1 ran with the 1e-2 ceiling (recorded in their meta args).

## compute incidents
- Host memory-PSI watchdog (another project holds 3 x 20 GB processes): shed t1 (iter 3605), the t1 resume, the go2 finalize (renders
  redone), anymal_c and hexapod6 (twice, within about 20 iterations). New launches go through `scripts/contact_launch_retry.sh`
  (bounded: 120 s x TRIES, exit 3 when never admitted).
- Peer at load 23-58 for most of the evening; h1/g1 warm runs continue there at 3-5 s/iter under the 6 h cap.

## resume steps
1. `tail -1 artifacts/runs/contact_<body>_v2/train_log.jsonl` (alpha, gate). The ops log is in the main checkout
   `ops/logs/<lease>_contact_<body>_v2.log`. `--resume` continues from checkpoint.pt, including the gate state.
2. When done, copy `actor.pt` to `artifacts/trackers/<body>/contact_v2/actor.pt` (gitignored), then run
   `scripts/contact_validate.sh <list>` with lines `<body> v2 v2 v2trk` and `<body> runs/.../actor_alpha0.pt v2 v2a0`.
3. Videos: `scripts/render_contact_compare.py --body <b>`. Host: GPU lease, and imageio/pillow from an isolated
   `--target` dir (`uv pip install --target <scratch>/pylib imageio imageio-ffmpeg pillow`); the host venv has neither.
