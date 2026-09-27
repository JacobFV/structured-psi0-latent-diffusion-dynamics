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
