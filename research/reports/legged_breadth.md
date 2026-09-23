# Legged / humanoid breadth track (P09 generators, P12 legged controllers, P08 psi0 contracts)

Status date: 2026-09-21. Per-body machine-readable ladder: `artifacts/assets/legged_catalog.json`
(rebuilt by `python -m rrp.morphology.legged_catalog`). Every number below comes from a saved run
(validation JSON under `artifacts/trackers/<body>/`, teacher reports under
`artifacts/assets/legged_teacher/`, training logs `artifacts/trackers/<body>/train_log.jsonl`).

## What was built

| piece | file |
|---|---|
| procedural n-leg generator (hexapod 6x3, mammal quadruped 4x3, sprawl 4/6/8-leg variants, long-leg hexapod), positive inertias from densities, foot spheres + touch sites, bounded PD position servos, IMU (framequat/gyro/acc) | `src/rrp/morphology/legged.py` |
| menagerie legged importers (go2, anymal_c, a1, h1, g1, t1, op3, talos_position, cassie) with a declared actuator adapter (motor -> bounded joint PD servo keeping source torque limits; per-asset gain table), IMU + foot-touch sensors added, linear joint equalities re-oriented when an actuated joint is the dependent side (talos grippers) | `src/rrp/morphology/legged.py` |
| catalogue builder / status ladder | `src/rrp/morphology/legged_catalog.py` |
| tracker core (public obs, privileged critic extras, reward, vectorised env) | `src/rrp/control/legged_core.py`, `legged_vec.py` |
| PPO (asymmetric actor-critic, own implementation) + resumable checkpoints | `src/rrp/control/tracker_training.py`, `tracker_nets.py`, `scripts/legged_supervise.sh` |
| deployable trackers (learned actor; scripted CPG baseline) | `src/rrp/control/legged_tracker.py` |
| independent tracker validation + eligibility freeze | `src/rrp/control/tracker_validation.py` |
| `waypoint_contact` task graph (walk_to A -> walk_to B -> halt with full-stance + low-speed completion; `upright` invariant) | `tasks/waypoint_contact.json` |
| scenario builder (registered as `BUILDERS["waypoint_contact"]` on import of `rrp.sim.legged`, existing builders untouched) + `LeggedSession(Session)` | `src/rrp/sim/legged.py` |
| scripted waypoint teacher (privileged true base pose + true waypoint positions, label `scripted_teacher`) | `src/rrp/control/legged_teachers.py` |
| legged collector (public/private split like `rrp.data.collect`, failures kept, resumable) | `src/rrp/data/legged_collect.py` |
| psi0 36-dim (AMO) and SONIC 80-dim/45-state contracts | `src/rrp/control/psi_contracts.py` |
| tests (only where silent bugs would invalidate results) | `tests/unit/test_legged.py` (IMU == root state; public tracker obs invariant to base position / linear velocity / yaw), `tests/unit/test_psi_contracts.py` |

Interfaces. Policy-facing command group `base_velocity = [vx, vy, wz]` (body frame; m/s, m/s, rad/s)
at 10 Hz, bounds = the body's command ranges (+5 % tolerance, else rejected). The tracker runs at 50 Hz
and outputs joint position targets for the body's `legs` actuators; `upper` actuators (arms/waist/head
on humanoids) are servoed to the default pose. Tracker actor inputs are only deployable signals: IMU gyro,
IMU gravity direction (roll/pitch), command, joint encoders, previous action, gait clock. The critic
additionally sees base linear velocity, height, true foot contacts, friction scale and a push flag
(training only). LeggedSession public sensors: encoders, IMU, foot touch, a declared noisy localization
sensor (x, y, yaw; sigma 2 cm / 0.02 rad), an overhead-camera detector for waypoint markers. Public
predicates: `distance_m(body, waypoint)` (localization vs tracked marker), `upright`, `stance_fraction`
(touch), `base_speed` (1 s localization baseline). Privileged truth versions are separate.

Tracker gate (frozen protocol, `tracker_validation/v1`, seeds 1000-1004, before any learned
high-level policy): 5 scripts (stand, forward 0.6 vmax, turn-in-place 0.6 wzmax, arc, forward + lateral
push); eligible iff no-fall >= 90 %, forward ratio in [0.5, 1.5], turn ratio in [0.4, 1.6], no fall while
standing. Teacher gate: >= 80 % privileged success on `waypoint_contact` seeds 0-19 with the frozen tracker.

## Results per body

Ladder: source_verified -> imported -> physics_validated -> controller_validated -> teacher_validated.
"physics_validated" = positive masses/inertias, finite 3 s rollout under default-pose PD hold, no deep
initial penetration (standing under a static hold is recorded, not required: humanoids fall without
balance control, which is expected).

| body | kind / source | status | controller (frozen) | tracker gate: fwd ratio / turn ratio / no-fall | waypoint teacher |
|---|---|---|---|---|---|
| hexapod6 | hexapod, procedural (synthetic) | teacher_validated | learned PPO (iter 774) eligible; CPG eligible | learned 0.88 / 0.93 / 1.0; CPG 0.96 / 0.75 / 1.0 | CPG 20/20; learned 0/20 (see below) |
| pquad4 | quadruped (mammal layout), procedural | teacher_validated | CPG | 0.90 / 0.57 / 1.0 | CPG 20/20 |
| sprawl4 / sprawl8 / hexapod6_long | 4/8/6-leg diagnostic variants, procedural | teacher_validated | CPG | 0.65/0.69, 0.96/0.60, 0.98/0.74; no falls | CPG 20/20 each |
| go2 | quadruped, menagerie unitree_go2 | teacher_validated | learned PPO iter 1199 | 1.05 / 0.59 / 1.0 | 20/20 |
| anymal_c | quadruped, menagerie anybotics_anymal_c | teacher_validated | learned PPO iter 1274 | 0.98 / 0.72 / 1.0 | 20/20 |
| g1 | humanoid, menagerie unitree_g1 (29-DoF, no hands; 12 leg joints policy-controlled, waist+arms held) | physics_validated (+ limited qualification) | learned PPO iter 3599 FAILS gate on turn-in-place | 0.91 / 0.075 / 1.0 (arc yaw ratio 0.49) | default teacher 16/20 (3 falls); arc_only teacher variant 19/20, 0 falls |
| t1, h1 | humanoid, menagerie booster_t1 / unitree_h1 | physics_validated; tracker training in progress at report time (see addendum) | - | - | - |
| op3, talos | humanoid, menagerie robotis_op3 / pal_talos (talos_position.xml) | physics_validated | not attempted (CPU budget) | - | - |
| a1 | quadruped, menagerie unitree_a1 (same unitree_quadruped family key as go2) | physics_validated | not attempted | - | - |
| cassie | biped, menagerie agility_cassie (closed-chain, 0.5 ms timestep) | physics_validated | not attempted | - | - |

Selected validation details (5 seeds each, `validation_learned.json`):
- go2 learned: forward 0.63 m/s at 0.6 cmd, CoT 2.3, slip 0.30 m/s; turn 0.36 rad/s at 0.6; survives 0.3 m/s lateral kicks.
- anymal_c learned: forward 0.47 m/s at 0.48, CoT 1.07; turn 0.35 at 0.48.
- hexapod learned vs CPG: similar tracking, but the learned gait is far less efficient (CoT 7.4 vs 1.6, slip 0.16 vs 0.07 m/s).
- g1: stands and walks forward 0.44 m/s at 0.48 cmd without falls (also under 0.15 m/s kicks); arcs at about half the commanded yaw rate; does not turn in place.

## Failures and how they were handled (preserved, not hidden)

1. **go2 checkpoint regression.** The iter-1999 go2 actor falls in all 5 turn-in-place episodes (turn
   ratio 0.0, no-fall 0.8) -> rejected, kept in `artifacts/trackers/go2/rejected_iter1999/`. The iter-1199
   actor passes and is the frozen one. Checkpoint choice used only the independent tracker protocol, never
   downstream policy results.
2. **Learned hexapod cannot complete `halt`.** It walks both waypoints (40/40 walk_to events) but idles
   on a tripod (stance_fraction 0.33), so the full-stance halt event fails (0/20). A 300-iteration
   fine-tune with an added zero-command stance-contact reward did not fix it (0/6) and slowed forward
   tracking to 0.61; it was not adopted. The CPG baseline (20/20) is the hexapod teacher controller; the
   learned tracker remains eligible for locomotion-only commands.
3. **G1 humanoid.** v1 reward (tracking kernel sigma 0.25, alive 1.0) converged to a stable non-walking
   stander (forward ratio 0.04). v2 (sigma 0.1, more tracking weight, less alive bonus; resumed from v1)
   learned forward walking but no yaw. v3 (turn weight 2.0 + in-place-turn commands; resumed) learned
   arc turning but not in-place turning. Under the frozen gate G1 is NOT controller_validated.
   A limited qualification is recorded (`artifacts/trackers/g1/limited_qualification.json`):
   whole-body floating-base forward/arc walker, eligible only with the declared `arc_only` teacher variant
   (min forward speed while turning), 19/20 teacher success. Any use must carry this limitation label.
4. Operational: peer thermal/memory-pressure sheds killed several training segments (all resumed from
   checkpoints via `scripts/legged_supervise.sh`); a `peer_sync.sh push --delete` wiped an early in-repo
   run directory (runs now live outside the synced repo at `/dev/shm/rrp-brandonin/legged_runs`). I did
   not run `rrp ops stop`; I only stopped my own units with `systemctl --user stop rrp-job-<my lease>`.

## psi0 action contracts

`rrp.control.psi_contracts` encodes exactly the audited layouts (source-audit section b):
original 36-dim AMO action (q_hand 0:14, q_arm 14:28, torso_rpy 28:31, base_height 31, v_x 32, v_y 33,
v_yaw 34, p_yaw 35; state 32 real dims zero-padded to 36) and SONIC 80-dim action (token 0:64, Dex3 hands
64:78, neck 78:80) with the 45-dim state (legs 0:12, waist 12:15, arms 15:29, hands 29:43, neck 43:45).
UNVERIFIED units stay flagged. `apply()` checks the body first, then exact width: the neckless 78-dim
variant is rejected by the 80-dim contract, cross-contract widths are rejected, and no body is validated
yet (menagerie g1.xml has no Dex3 hands; h1.xml has no Inspire hands), so applying either contract to any
body - including G1 -> another body - raises `ContractBodyError`. Tests: `tests/unit/test_psi_contracts.py`.

## Breadth-gate accounting (honest)

- Generated hexapod: done (CPG + learned trackers, teacher-validated).
- Two quadruped families: go2 (unitree) + anymal_c (anybotics) both controller- and teacher-validated.
  pquad4 is synthetic and does not count as a commercial family.
- Three humanoid families: imported and physics-validated: h1, g1, t1, op3, talos (5 families).
  Controller-validated: none under the frozen gate; g1 has a limited (forward/arc) qualification.
  **The humanoid part of the breadth gate is not met yet.**
- Learned high-level policy evaluation on waypoint_contact: not done in this track (owned by the
  learning track); datasets are ready (below).

## Datasets (teacher demonstrations, `scripted_teacher`, privileged)

`artifacts/datasets/legged_waypoint_contact/<body>/` (gitignored; manifest per body): 20 episodes each for
hexapod6, pquad4, sprawl4, sprawl8, hexapod6_long (CPG), go2, anymal_c (learned trackers), g1 (default
teacher, learned tracker, includes 3 falls + 1 failure). `.../legged_waypoint_contact_arc_only/g1/`:
20 episodes with the arc_only teacher. `.../legged_waypoint_contact_learnedtracker/hexapod6/`: the 20
failed-halt episodes with the learned hexapod tracker. Public records: encoders, IMU, touch, localization,
detector slots, public predicates, task view; action = base_velocity. Private: true base pose/velocity,
contacts, truth predicates, teacher phase.

## Costs

GPU: 0 device-hours (all training on peer CPU cores; PPO updates on CPU).
Training wall time (sum of iteration times from train logs): hexapod6 0.33 h (3-7 cores), go2 0.92 h
(5 cores), anymal_c 0.58 h (5 cores), g1 v1+v2+v3 2.07 h (2-5 cores); samples 4.5 M / 12.4 M / 8.0 M /
~39 M respectively. Estimated upper bound ~= 18 CPU core-hours for training. The resource ledger
(`scripts/legged_costs.py` over host + peer ledgers) records 1.22 core-h, because units stopped by the
watchdog/`systemctl stop` do not write ledger rows; treat the train-log estimate as the real figure.
Validation/teacher/catalog jobs on the host: < 0.1 core-h.

## Resume / next steps

- T1/H1 trackers: see addendum; supervisors `scripts/legged_supervise.sh {t1,h1} ...` resume from
  `/dev/shm/rrp-brandonin/legged_runs/<body>/checkpoint.pt`.
- G1 in-place turning: likely needs a turning curriculum from the start or a yaw-rate-conditioned gait clock.
- op3 / talos / cassie / a1 trackers not attempted.
