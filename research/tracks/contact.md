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
The peer was at load 55 (other users), so no peer runs; g1/go2/h1/anymal_c/hexapod6 are queued behind the A/B result.

## resume steps
1. `tail -1 artifacts/runs/contact_<body>_v2/train_log.jsonl` (alpha, gate). The ops log is in the main checkout
   `ops/logs/<lease>_contact_<body>_v2.log`. `--resume` continues from checkpoint.pt, including the gate state.
2. When done, copy `actor.pt` to `artifacts/trackers/<body>/contact_v2/actor.pt` (gitignored), then run
   `scripts/contact_validate.sh <list>` with lines `<body> v2 v2 v2trk` and `<body> runs/.../actor_alpha0.pt v2 v2a0`.
3. Videos: `scripts/render_contact_compare.py --body <b>`. Host: GPU lease, and imageio/pillow from an isolated
   `--target` dir (`uv pip install --target <scratch>/pylib imageio imageio-ffmpeg pillow`); the host venv has neither.
