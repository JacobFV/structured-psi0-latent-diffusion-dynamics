# track: humanoid (W13, D-138) — humanoid transfer program

Owner request (2026-09-28): "focus more on humanoids. we need to test lots of humanoid transfer" and "the humanoid tasks need
to be more complex". Worktree `~/work/rrp-wt/humanoid`, branch `track/humanoid`, peer code dir
`/dev/shm/rrp-brandonin/wt/humanoid` (always `export RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/humanoid`).
Limits: host = git/editing/unit tests only (D-115); ≤ 2 concurrent GPU leases for this track; memory declared ≥ 1.35 × the
peak measured PER BODY (D-117); never live-modify a lease; `rrp ops stop` only with `--lease`/`--owned-only`.

## 0. starting point (verified 2026-09-28 from decisions and peer logs)
- Trackers under sourced limits + contact_v2: **none passes the D-112 tracker gate.** t1 w8d (installed) fails CoT 2.13 and
  joint margin −0.053 (D-114) and the lab forward check 0.72; its dataset fails the slip gate (86.2% < 95%, D-113).
  g1 r1 passes the gate but was trained with hip roll 139 N·m vs sourced 88 (D-107) and refuses to load under sourced limits;
  g1_src stomps (4.09 BW) and overspins. h1 parked: shuffles, never lifts the feet in turns, falls under actuator latency.
- Learned humanoid routes: t1 BC 0/30, latent R2 30–33/90 with 57–60 falls (D-124). The routes imitate the tracker's joint
  targets; a realistically limited t1 gait is not learnable by imitation of teacher+tracker targets alone.
- Humanoid task: `waypoint_contact` (walk to A, walk to B, halt) only. **Humanoid transfer evidence: none.**
- Tracker PPO today: CPU C-MuJoCo, ~2.5 k samples/s at 3–4 workers (g1_src 800 it × 6144 samples in 1,887 s; t1 w8d 2,497 s;
  h1 r2 5,500 it in 3,842 s of resumed wall). ≈ 650–800 samples/s per worker ⇒ ≈ 10–12 k samples/s with 16 workers on an idle peer.
- Code ready, never run: `--ref-gait clock`, `--yaw-progress-cap`, `--limit-margin[-agg]`, recipes in
  `rrp.harness.train.tracker_recipes`, `recipes/humanoid/d126_tracker_*.yaml`; MJX prototype (dropped with the env refactor; contact_v2 accepted with
  `no_self_collision`; 512 envs ≈ 512 ticks/s on a CONTENDED GPU, i.e. not yet useful).

## 1. body pool (a): menagerie humanoids + a procedural family

Menagerie pinned at `c96a32d28fb5da84da38c1da4d749e7a13212855` (`ops/bin/fetch_menagerie.sh`; assets never committed).
Licenses read from each directory's LICENSE at that commit. MJCF sha256 prefixes recorded so the sealed set cannot drift.

| body | dir / file | sha256[:16] | license | mass kg | root z m | DoF (nu) | arms | role |
|---|---|---|---|---|---|---|---|---|
| t1 | booster_t1/t1.xml | 78af3f7910e44138 | Apache-2.0 | 31.6 | 0.67 | 23 | yes | **train** |
| g1 | unitree_g1/g1.xml | 3c2616550a31f33e | BSD-3-Clause (Unitree) | 33.3 | 0.79 | 29 | yes (no hands) | **train** |
| h1 | unitree_h1/h1.xml | 4d15bc381ee81137 | BSD-3-Clause (Unitree) | 51.4 | 1.06 | 19 | yes | **train** |
| op3 | robotis_op3/op3.xml | 02e07fe80bb0352a | Apache-2.0 | 3.1 | 0.30 | 20 | short | **train** (small-scale anchor) |
| apollo | apptronik_apollo/apptronik_apollo.xml | 28513f17a37f1cfa | Apache-2.0 | 80.9 | 1.02 | 32 | yes | **train** |
| adam_lite | pndbotics_adam_lite/adam_lite.xml | a8725b28db6a1c74 | MIT | 58.2 | 0.93 | 25 | yes | **train** |
| talos | pal_talos/talos_position.xml | ce2d3af3f1c80e98 | Apache-2.0 | 94.0 | 1.02 | 32 | yes + grippers | **train** (heavy anchor; mesh collisions, 45 mesh geoms: primitive feet required) |
| **n1** | fourier_n1/n1.xml | 50a15602f6dfb4db | Apache-2.0 | 39.7 | 0.70 | 23 | yes | **SEALED S2** new family, adult mid-size |
| **berkeley** | berkeley_humanoid/berkeley_humanoid.xml | e5b9396bf23dea75 | BSD-3-Clause (Hybrid Robotics) | 16.1 | 0.52 | 12 | **no** | **SEALED S3** new family, legs only |
| **toddlerbot_2xc** | toddlerbot_2xc/toddlerbot_2xc.xml | 2c89e8fc7161f432 | MIT | 3.5 | 0.31 | 30 | yes | **SEALED S4** new family, tiny |
| toddlerbot_2xm | toddlerbot_2xm/toddlerbot_2xm.xml | 0599ae55c3562408 | MIT | 3.8 | 0.31 | 30 | yes | sealed with its family (second S4 sample; never trained) |
| **g1_hands** | unitree_g1/g1_with_hands.xml | 2d41a3c6783fbf9e | BSD-3-Clause | – | 0.79 | – | yes + Dex3 hands | **SEALED S1 "near"** (new end-effectors on a known body; manipulation tasks only; analog of D-135 panda_tf3) |

Not humanoid / excluded: cassie (biped without torso/arms; optional extra training body, never a target).
Torque limits: t1/g1/h1 sourced_v1 (D-107). The others use menagerie `forcerange`/`ctrlrange` as published by the model
authors; recorded per body as `limits_source: menagerie_author` (not manufacturer-audited), a labelled limitation.
Adapters for n1, berkeley, apollo, adam_lite, toddlerbot are new code (P1); op3/talos adapters exist but were never trained.

**Procedural humanoid family `phum` (new, `rrp.bodies.humanoid_gen`, P1c)**: a generator of MJCF humanoids from a parameter
vector with primitive geoms only (capsule limbs, box feet, box/capsule torso), sourced-style torque limits scaled by
mass × leg length (a declared scaling law, labelled `limits_source: procedural_scaling`):
- total height 0.35–1.8 m (log-uniform), mass from height^2.3 with ±25% noise, thigh/shin ratio 0.8–1.25,
  torso/leg ratio 0.6–1.1, hip width 0.18–0.32 × height, foot length 0.12–0.2 × height, foot shape {box, capsule-pair};
- leg DoF {5 (no hip yaw), 6, 7 (toe)}, ankle {pitch-only, pitch+roll}; arms {none, 3-DoF, 4-DoF, 7-DoF}, waist {0,1,3};
- link masses and CoM offsets ±15%, joint ranges from a template ± 10%.
- **Training region** = the box above minus the SEALED procedural region **phum_sealed**: thigh/shin ratio > 1.15 AND leg
  DoF 7 (toe joint) AND height > 1.5 m (extrapolation on three axes jointly); generator seeds ≥ 9,000,000 are sealed,
  training uses seeds < 1,000,000 and rejects any sample inside the sealed region (unit-tested).

Stable identities: every body carries `body_id` = `menagerie/<dir>@<sha12>` or `phum/<param-hash>@gen<version>`; role
multiplicity and null bindings (no arms ⇒ `hand_*` roles bound to null with reason `absent_limb`) are part of the packet
context, not afterthoughts.

## 2. tasks (b): staged, easy → hard (success metric; failure reasons; privileged-only fields marked)
All scenes built by `rrp.envs.mujoco.humanoid_scenes` (P2), task graphs built in code (no `tasks/h_*.json` files any more), contact_v2, sourced limits, scaled to
body height where geometry matters (step height, gap width, object size = f(leg length)). Every episode records `fell`
(base height < 0.55 × nominal or tilt > 0.7 rad), `timeout`, and the task-specific reasons below; failure reasons are
computed by the privileged evaluator and never enter deployable observations.

| stage | task id | what | success | failure reasons |
|---|---|---|---|---|
| L0 | `waypoint_contact` (exists) | walk A → B, halt | reach both within 0.25 m, halt ≥ 1 s | fell, timeout, halt_drift |
| L1 | `h_gap_sidestep` | turn and sidestep through a gap between two walls (width 1.2–1.6 × shoulder width), then face a heading | base past the far wall line, heading error < 0.3 rad, no torso–wall contact > 0.2 s | fell, wall_collision, wrong_heading, timeout |
| L2 | `h_steps` | climb up/down 2–4 steps of 0.10–0.30 × leg length on bumps_v1 rough ground (≤ 4 cm) | top/bottom platform reached, no fall | fell, trip (toe contact with riser > 0.3 s), missed_step, timeout |
| L3 | `h_stepping_stones` | cross 6–10 stones (reuses `foothold_steps`: dx, dy, dyaw targets, foot order with null = any foot) | every stone touched in order by the required foot, no floor contact between stones (floor = "water") | fell, wrong_foot, missed_foothold, water_contact, timeout |
| M1 | `h_carry_box` | box (0.5–3 kg, 20–40% body width) starts between the hands at chest height (pre-grasped by weld release at t=0.5 s); walk 3–5 m to a waypoint and place it on a table | box on table within 0.2 m of target, box tilt < 20° en route, no drop | dropped, box_tilt, fell, place_miss, timeout |
| M2 | `h_squat_pick` | walk to a box on the floor, squat, bimanual pick (palm/forearm pressure; no hands required), stand, hold 2 s | box ≥ 0.3 × height above floor for 2 s, standing | fell, no_grasp, dropped, not_upright, timeout |
| M3 | `h_push_cart` | push a wheeled cart (slide joints, friction 0.1–0.4 × weight) 3 m along a lane; variant: push a hinged door open ≥ 60° and walk through | cart displacement ≥ 3 m within lane / door angle ≥ 60° and base through | fell, lost_contact (> 1 s), off_lane, door_not_open, timeout |
| C1 | `h_sit_stand` | walk to a stool (height 0.4–0.55 × leg length), turn, sit (pelvis on seat ≥ 1 s), stand up | seated ≥ 1 s, then upright | fell, missed_seat, not_upright, timeout |
| C2 | `h_wall_support` | contact sequence: place one hand on a wall, then step across a 0.25–0.4 × leg-length gap in the floor while the hand stays on the wall | hand contact spans the gap step, both feet beyond the gap | fell, hand_off_wall, wrong_sequence, gap_fall, timeout |

Arms-free bodies (berkeley, phum with arms=none) are evaluated on L0–L3 and C1 only; M*/C2 bind the hand roles to null and are
reported as `not_applicable` (never as failures, never averaged in).
**Held-out tasks (sealed, never trained; evaluated once):** `h_steps_carry` (M1 on a 2-step staircase), `h_gap_cart` (M3
through an L1 gap), and parameter extrapolation L2 step height 0.30–0.40 × leg length (training ≤ 0.30).
Task/event knowledge (the task graph, role bindings, targets) may be supplied; future outcomes, object masses/friction and
simulator ground truth may not enter deployable observations (public context = encoder FK + localisation, as `FootholdSession`).

## 3. teachers (c)
- **Privileged RL experts** (PPO, label `privileged_teacher:rl_expert:<sha>`), one per task stage, **morphology-conditioned**
  over the training pool + phum training region (inputs: proprioception, morphology graph features, privileged object/terrain
  state: heightfield samples, object pose/velocity/mass, foothold targets, contact flags). Output: whole-body joint targets
  (legs + arms + waist). Warm-started from the shared tracker (P1) and trained with a task reward + the gait_v2 permanent terms
  (clearance floor, joint-limit-margin hinge `max`, slip, foot-force) so every expert rollout can pass the D-112 motion gates.
- **Scripted** components only where scripting is known to work: the waypoint/heading command layer (existing scripted
  teacher over the tracker, label `scripted_teacher`) for L0/L1. Hand-scripted whole-body squat/carry/sit trajectories are NOT
  used (the D-124 lesson: imitating a brittle scripted target stream on a humanoid falls).
- The expert is stateless given the privileged state, so it is also the **DAgger labeller** for every learned route (fixes
  the D-124 failure mode: learned routes now get labels at the states they visit, including near-falls).
- Datasets: per (body, task) 300–600 episodes at DART σ ∈ {0, 0.05, 0.1} on the expert's joint targets; dataset gate =
  D-112 legged gates (slip < 0.15 on ≥ 95% episodes, 0 falls at σ 0) + expert success ≥ 0.9 at σ 0 per body + task-specific
  checks (box penetration ≤ 3 mm under grasp contact v2.1 settings, no wall interpenetration).

## 4. transfer matrix (d) — PRE-REGISTERED here, before any training (D-138)
Training pool = {t1, g1, h1, op3, apollo, adam_lite, talos} ∪ phum training region; tasks L0–L3, M1–M3, C1–C2 minus the
held-out tasks. Sealed bodies = S1 g1_hands (manipulation tasks), S2 n1, S3 berkeley (legs-only tasks), S4 toddlerbot_2xc
(+ 2xm), phum_sealed (seeds ≥ 9,000,000, 20 sampled bodies). Sealed evaluation scenes = seeds 2,000,000+ (same convention as
D-135); dev seeds 3,000,000+ may be used for non-sealed diagnostics on NON-sealed bodies only.

Level 1 — **existing-controller transfer** (reported separately, AGENTS): the morphology-conditioned shared tracker (P1c)
and the task experts, zero-shot on sealed bodies; and PPO fine-tune budgets 1e6 / 1e7 samples vs a per-body tracker trained
from scratch with the same samples.
Level 2 — **new controller training** (the core claim), all with the same inputs (public context, proprioception, morphology,
task graph) and the same acquisition (same demos, same DAgger rounds with the same expert, same updates):
| method | label | adapts |
|---|---|---|
| latent semfix (packet semantics supervised) | learned:latent_semfix | zero-shot; system-0 refit; **joint flow + system-0 (D-136 split, gen_frac 0.5)** |
| latent nosem (capacity-matched) | learned:latent_nosem | same three |
| whole-policy BC | bc:direct | zero-shot; **whole-policy BC SFT** |
Budgets 5 / 20 / 100 target demos (expert on the sealed body, privileged teacher — allowed as adaptation data, labelled);
update counts matched to BC SFT 150 / 300 / 600 (D-136). Seeds 2 training seeds × 100 sealed scenes per cell, each cell run
ONCE; metric success k/200 with Wilson 95%; method − BC SFT with Newcombe 95%. Held-out tasks: zero-shot only, on training
bodies and on S2. Causal checks (packet edits: halt, goal shift, hand-binding swap to null) on training bodies and S2, since
probes are diagnostics only.

## 5. phases, gates and wall-clock (e) — single peer GPU (GB10), ~20 CPU, shared with other tracks
Throughput assumptions: CPU PPO ≈ 11 k samples/s at 16 workers (measured per-worker × 16; the peer is shared, so ×0.6 when
busy). GPU sim (MuJoCo Warp / MJX) is UNMEASURED on an idle GPU; P1a measures it. Estimates are given for both engines.

| phase | work | gate | GPU-sim path | CPU-only path |
|---|---|---|---|---|
| P0 | this plan, D-138, sealed set | committed before any training | 2 h | 2 h |
| P1a | engine bake-off: C-MuJoCo CPU vs MuJoCo Warp vs MJX on t1 + g1 (contact_v2, primitive feet, no self-collision declared), 1,024–8,192 envs; parity vs C over open-loop PD replays | parity (base height, contact timing within tolerance over 4 s) and ≥ 5× CPU samples/s ⇒ GPU trainer, else CPU | 3–4 h | – |
| P1b | per-body trackers from scratch (clock reference gait, sourced limits, W8 command mix, limit-margin max) on t1, g1, h1, op3, apollo, adam_lite (talos last): ~2e8 samples each | D-112 tracker gate + lab gate (no-fall 1.0, fwd ≥ 0.8, turn ≥ 0.5, slip < 0.15) on **≥ 4 humanoids**; tiny rollouts; videos incl. failures | ~1 h/body, 1 day with debugging | ~5 h/body ⇒ 2–3 days |
| P1c | `phum` generator + morphology-conditioned shared tracker (1–2e9 samples) = first transfer test (zero-shot to sealed bodies, Level 1) | passes the tracker gate on ≥ 4 pool bodies; sealed zero-shot reported once | 1–2 days | 5–8 days (not recommended) |
| P2 | scenes L1–L3, M1–M3, C1–C2 + privileged experts (≈ 5e8–1e9 samples/stage) + datasets | expert ≥ 0.9 success per body and dataset gates | 3–4 days (8 tasks) | ~8 days for 3 tasks (reduced scope) |
| P3 | collection, latent semfix/nosem × 2 seeds + BC × 2 on the pool; DAgger with the expert; sealed transfer evals once | all cells complete; fresh audit | 3–4 days | 3–4 days (training is GPU either way) |
| total | | | **≈ 9–13 days** | ≈ 3 weeks with 3 tasks |
Peer disk (lead, 2026-09-28): ~84 GB free, armdiv needs ~25 GB; humanoid packs/rollouts stay small (float16 packs, ≤ 300
episodes per body×task first), finished artifacts are archived to the host `~/work/rrp-data` and removed from the peer.
Stop rules (docs/10_autonomy_and_recovery.md): a per-body tracker gets at most 3 recipe attempts (≤ 6e8 samples) before it
is marked `failed_hypothesis` and replaced by the next pool body; a task expert at most 2 reward designs before
`blocked`/`failed_hypothesis`; no sealed cell is re-run except after a recorded infrastructure failure (partial rows deleted).

## 6. relationship to psi1z (Ψ₀ / SIMPLE)
SIMPLE's G1 tasks (TabletopGraspMP, BendPickMP, XMovePick, HandoverTeleop) are humanoid (whole-body) manipulation on a G1 with
Dex3 hands rendered in Isaac. Our M2 `h_squat_pick` is the MuJoCo, multi-body analog of BendPickMP, M1 carry of XMovePick,
and sealed S1 `g1_hands` is the nearest rrp body to psi1z's `g1_simple` family. This track uses no SIMPLE/Ψ₀ assets or code.
Results here (does a semantic packet help a humanoid realizer transfer?) inform psi1z's structured-head design; psi1z results
(Ψ₀ + structure on real teleop data) are not evidence for this track's sealed bodies.

## log
- 2026-09-28 P0: plan + sealed set committed (this file, D-138). Next: P1a engine bake-off, P1b first recipes.

## P1a RESULT (2026-09-29 00:00): MuJoCo Warp adopted as the humanoid training simulator (gate partly missed, recorded)
Engine: mujoco_warp 3.14.0 + warp-lang 1.17.0 in an isolated target dir on the peer (`~/work/ext/pylibs/mjwarp`, 468 MB; not an
rrp dependency; `PYTHONPATH=src:~/work/ext/pylibs/mjwarp`). Code: `src/rrp/envs/warp/` (bake-off; then model/tracker_env/task_env), raw JSON
`artifacts/runs/humanoid_p1a_warp_{t1,g1,h1}.json` (peer store). Model = the contact_v2 tracker world with the declared
adaptation `no_self_collision` (robot collides with ground only), identical on both sides.
- Parity (open-loop PD, same model, float32 vs float64): max |dqpos| 4.5e-6 (t1), 2.2e-6 (g1), 2.6e-6 (h1) over 50 ticks (1 s);
  over 200 ticks both engines fall identically (final heights equal to 1e-3 m). PASS.
- Throughput, env-ticks/s (1 tick = 10 substeps), idle GPU: t1 62.7k/57.7k/55.3k at 1,024/4,096/8,192 worlds; g1 52.0k/58.0k/58.0k;
  h1 106.8k/126.4k/119.5k (saturates at ~1k worlds). C MuJoCo one thread: t1 1.5k, g1 1.5k, h1 4.3k.
  Fewer solver iterations do not help (10 iters/20 ls: g1 43k, with line-search warnings; 4 iters: C side unstable).
- GPU PPO end to end (`rrp.harness.train.warp_tracker_ppo`, t1, 4,096 worlds, horizon 24): 2.3 s/iter = 42k samples/s vs the CPU
  trainer's ~11k samples/s at 16 workers (measured 650-800/worker) => 3.8x. The pre-declared gate asked >= 5x: MISSED on
  training throughput (5.2x only on raw simulation). Decision (recorded, not silent): adopt the GPU trainer anyway, because it
  leaves the shared 20 peer CPUs to the other tracks and still cuts per-body wall-clock ~4x. Side effect: two concurrent GPU
  runs hold the GB10 at ~95% and the SoC at ~95 C, which trips the broker's `cpu_hot` admission stop for everyone; keep W13 to
  <= 2 GPU training leases and schedule short checks between runs.
- Observation parity of the GPU env (`humanoid_warp_obs_parity` (throwaway diagnostic, not kept)) vs `LeggedBinding.public_obs`: <= 3e-7 on t1, g1,
  op3, apollo, adam_lite. Zero-action survival agrees with C MuJoCo (t1 ~1.2 s both; op3 stands both; apollo/adam_lite fall at
  ~1.4-1.6 s both).

## P1b/P1c code (landed on main)
- `rrp.envs.warp_tracker_env.WarpTrackerEnv`: GPU twin of LeggedEnv (same public obs/actions, gait_v2 reward terms, contact_v2
  randomisation per world at reset, latency 0-4 substeps); host-sync-free; K-variant batching of same-topology bodies (MORPH_FIELDS,
  phum variant parity <= 9e-6 vs C per variant). `MorphMultiEnv`: many groups behind one morph_v1 interface.
- `rrp.envs.morph_obs` (morph_v1): 14 canonical leg slots with sign canonicalisation + static morphology context (obs 156);
  LearnedTracker deploys it in C MuJoCo (version string `learned_tracker:shared_morph_v1[:transfer]:<body>`); parity 1.5e-7.
- Adapters: apollo (collision group 3 enabled; author PD replaced by the declared auto-gain rule), adam_lite, n1 (SEALED, code
  only). Declared rule for bodies without author PD: kp = clip(effort, 10, 300) N m/rad, kd = 0.025 kp.
- `rrp.bodies.humanoid_gen` (phum): 72 topologies in the declared space, sealed region ~1% of sealed seeds
  (first S5 seeds 9000029, 9000302, 9000414).
- P2 start: `rrp.envs.humanoid_scenes` (h_steps), `rrp.envs.warp_task_env.WarpStepsEnv` (privileged height scan expert env).

## Peer code dirs
Long runs use `/dev/shm/rrp-brandonin/wt/humanoid` (never re-synced while they run); checks/dev use `.../wt/humanoid_dev`.

## P1b log (per-body GPU trackers; C-MuJoCo gate numbers only count)
- 00:20 all peer leases shed by the watchdog (`disk_below_reserve`: /dev/shm 10 GB free, other tracks' run dirs); lead archived
  ~15 GB; t1/h1 r1 resumed from their iter-349/649 checkpoints (exact resume of weights/optimizer; env RNG restarts).
- h1 r1 INTERIM (actor at ~iter 450, `humanoid_p1b_h1_r1/gate_interim`, C MuJoCo, full self-collision, 10 seeds): no-fall 1.0,
  forward 0.89, slip 0.05, CoT 0.47, joint margin 0.068, robust-in-range ok; FAILS peak foot force 3.43 BW (> 3.0) and the
  pure turn runs at 2.1x the command (turn_lin rewarded over-rotation; the sharp yaw kernel gave ~0); stand drifts -0.06 m/s,
  so the waypoint teacher's halts never complete (waypoint 0/20, 0 falls, all timeouts). -> r2 fixes: yaw cap 1.0 + overshoot
  penalty, stand_vel -3, natural terms (impact/power) on.
- h1 r1 stopped at iter 690 (bounded decision: fine-tune rather than finish the flawed objective).
- h1 r2 FAILED (iter 282: window fall 0.95, tracking error 1.86): trainer bug, the warm-start observation normaliser was
  overwritten by the first batch (count 1e-4). Fixed (count 1e6, as tracker_training). r2b = fixed + alpha 0.5: running.
- t1 r1 (running, iter ~600): window fall 4%, tracking error 0.53, turn ratio 1.8 (same over-rotation), slip 0.21.
- **Sim-to-sim finding (the main P1b lesson so far):** policies trained with `no_self_collision` use leg poses where the legs
  pass through each other. In C MuJoCo with full self-collision: h1 r2b push no-fall 0.5 and waypoint 15/20 fell; t1 r1
  (iter 1049) forward no-fall 0.2, forward ratio 0.0, peak force 5.6 BW, waypoint 20/20 fell. With left-right leg contacts
  enabled on the GPU, the t1 r1 actor falls within ~1 s in every episode. New declared adaptation `leg_cross_collision`
  (left-leg vs right-leg contacts on, other self-contacts off; same GPU cost under contention: t1 13.7k vs 13.0k ticks/s)
  is now the GPU training default. t1 r1/r2 are void; t1 v2 restarts from scratch; h1 r3 fine-tunes r1 iter 649 under it.
- Shared tracker v1 (no_self_collision) was paused at iter ~10 after profiling: 10 groups x 512 worlds ran at 6.6k samples/s
  (per-group Python/torch overhead ~25 ms/tick dominates at 512 worlds). The recipe now uses 8 larger groups; relaunch after the
  per-body trackers, under leg_cross_collision.
- **h1 r3 (fine-tune of r1 iter 649 under leg_cross_collision, 600 iters) in C MuJoCo, full self-collision, 10 seeds**
  (`artifacts/runs/humanoid_p1b_h1_r3/gate/{val.json,gate/gate_report.json,lab_gate.json}`): no-fall 1.0 on every trial incl.
  the 0.15 m/s push and all in-range robustness conditions; forward 1.08, turn 0.96, slip 0.018, CoT 0.51, joint margin 0.032.
  Lab gate PASS. D-112 gate FAIL on one criterion: peak foot force 3.72 BW (limit 3.0). Waypoint (scripted_teacher) 0/20
  success, 0 falls: walk_to_a and walk_to_b succeed, the final `halt` fails because under a zero command h1 keeps stepping and
  drifts ~0.06 m/s with yaw (`humanoid_waypoint_diag` (throwaway diagnostic, not kept)). Videos (peer store, `artifacts/video/INDEX.md`):
  `2026-09-29_contact_h1_{forward,turn}_*-vs-h1-gpu-r3_ok_ok.mp4` (both panels show r3: no installed h1 contact_v2 actor on the
  peer, labelled by the version string), t1 r1 failure clip `2026-09-29_contact_t1_forward_forward_installed-vs-t1-gpu-r1_ok_fell.mp4`.
  -> h1 r4 (running): ~25% zero commands, stand_contact 2, stand_still -1, impact -0.5.
- h1 r4 (r3 + standing terms + impact -0.5, 25% zero commands) in C MuJoCo: no-fall 1.0, fwd 1.06, turn 1.06, slip 0.005,
  robust ok; FAILS peak force 4.08 BW and joint margin -0.005; arc yaw ~0 (0.004 vs 0.24) and it still steps in place under a
  zero command (stand trial duty 0.5 at 1.25 Hz), so halts fail (waypoint 0/20, 0 falls). Reward weights alone did not stop
  clock-driven stepping. -> structural fix **clock gate** (`--clock-gate`, actor meta `clock_gate`): the gait-clock inputs
  are zeroed while the command is ~0, identically in the GPU env, `LeggedBinding.public_obs`, morph_v1 and LearnedTracker
  (parity 1.8e-7 incl. gated worlds). h1 r5 = r3 + clock gate + arc turn progress (yaw_lin_all <= 0.5 rad/s) + impact -2 +
  joint-limit hinge -4: running. g1 v3 stopped at iter 55 to give h1 r5 the slot; g1/op3/apollo/adam_lite/t1 get clock-gated
  v4 recipes.
- **h1 r5 (clock gate) in C MuJoCo:** no-fall 1.0 on every trial and robustness condition, fwd 1.04, turn 0.98, arc yaw
  0.236 of 0.24, slip 0.015, CoT 0.65, **waypoint 13/20 success, 0 falls** (the clock gate fixed the halts: stand duty 0.94).
  D-112 still FAILS on peak force 3.37 BW (walking trials 3.1-3.4, limit 3.0) and joint margin 0.0017 (turn_fast only).
  Stop-rule note: h1 has used 5 recipe attempts (r1, r2b, r3, r4, r5; r2 was a trainer bug) against the declared 3, but
  ~3.6e8 of the declared 6e8-sample budget; r6 (impact -4, limit hinge -8, 400 iters) is the last h1 attempt.
- **t1 v2** (from scratch under leg_cross_collision, 1500 iters, no clock gate) in C MuJoCo: no-fall 1.0 on every trial and
  robustness condition, fwd 0.93, turn 1.01, slip 0.071, **CoT 0.45** (installed CPU-trained t1 w8d: 2.13), joint margin 0.0175
  (limit 0.02), peak force 3.80 BW, waypoint 0/20 with 0 falls (halt defect). Lab gate PASS; D-112 FAIL (force, margin).
  Video: `2026-09-29_contact_t1_{forward,turn}_*-vs-t1-gpu-v2_*.mp4` (left: installed w8d). -> t1 v2ft (clock gate, impact -4,
  limit hinge -8) queued.
- Pattern across h1 and t1: every GPU tracker now passes no-fall / tracking / slip / CoT / in-range robustness in C MuJoCo, and
  the two remaining D-112 failures are peak foot force (3.3-4.1 BW vs 3.0) and the worst-tick joint margin (0.002-0.018 vs
  0.02). g1 v4 (from scratch, clock gate) is still learning to stand (iter 344).
- **h1 r6 = final h1 tracker (stop rule reached)**, C MuJoCo, full self-collision: **waypoint 20/20 success, 0 falls**;
  no-fall 1.0 on all trials incl. push and all in-range robustness conditions; fwd 1.11, turn 0.82, slip 0.012, CoT 0.71.
  Lab gate PASS. D-112 FAIL on peak foot force 3.74 BW (limit 3.0) and joint margin -0.0014 (limit 0.02) -- label every use
  "h1 gpu r6: D-112 fails peak force 3.74 BW, joint margin -0.001". Heavier impact/limit weights (r5 -> r6) did not reduce
  either, so the next attempts (t1, g1) add two structural mechanisms: `target_margin` (joint targets clipped 3% inside the
  range, applied identically in deployment) and `land_vel` (training-only touchdown foot-speed penalty).
- **t1 v2ft** (t1 v2 + clock gate, impact -4, limit hinge -8) in C MuJoCo: **waypoint 19/20, 0 falls**; no-fall 1.0 everywhere,
  fwd 1.11, turn 1.05, slip 0.072, CoT 0.54; D-112 FAIL on peak force 3.28 BW and joint margin 0.016. Next: t1 v2ft2 with
  target_margin 0.03 + land_vel.

## P2 log
- `h_steps` GPU expert env bug found and fixed: changing box geom sizes at run time (per-world step heights via batched
  geom_size/pos/aabb/rbound) breaks mujoco_warp contacts -- with a flush h = 0 staircase, h1 r6 (which walks indefinitely on the
  flat GPU env and crosses the same h = 0 staircase in C MuJoCo, 2/3 success + 1 lateral miss) tipped over at x ~ 1 m in 100% of
  episodes; the same model compiled at h = 0 did not (`humanoid_steps_diag` (throwaway diagnostic, not kept)). Steps are now fixed-size boxes on mocap
  bodies moved per world (level 0: no falls in 500 ticks). The first h1 steps run (v2, geom-resizing) was void and stopped.
- h1 steps expert v2 (warm start h1 r6, privileged 11x3 height scan + h_frac, scripted heading command, step-height
  curriculum 0-0.30 L): running.
- **g1 v4** (from scratch, leg_cross_collision + clock gate, 1500 iters) in C MuJoCo (left video panel: CPU-trained g1_src
  evaluated with RRP_ALLOW_LIMITS_MISMATCH, labelled): no-fall 1.0 on every trial and robustness condition, fwd 0.90, turn 0.98,
  slip 0.016, CoT 0.47, **peak force 2.13 BW (passes)**; D-112 fails ONLY joint margin 0.012 (limit 0.02). Waypoint 17/20,
  3 falls. -> g1 v4ft (target_margin + land_vel) queued.
- **t1 v2ft2** (+ target_margin 0.03, land_vel -2): waypoint 19/20, 0 falls; no-fall 1.0 everywhere, fwd 1.09, turn 1.18,
  slip 0.073, CoT 0.59, **joint margin 0.022 (passes now)**; D-112 fails ONLY peak force 3.63 BW. Recorded deviation from the
  3-attempt stop rule (t1 is at ~2.7e8 of 6e8 samples): one more attempt, v2ft3 = + per-tick force cap (2.5 BW, -2).
- **h1 steps expert v2** (1500 iters, curriculum reached level 0.4 = step heights up to 0.12 L; training success ~0.45-0.50
  per window) in C MuJoCo, full self-collision, 20 seeds per height (`artifacts/runs/humanoid_p2_h1_steps_v2/c_grid/`):
  expert (privileged_teacher:rl_expert + scripted heading command) h = 0.10 L: **20/20 success, 0 falls**; h = 0.15-0.30 L:
  0/20, all timeouts, it stops before the first riser (inputs outside its curriculum range; it never falls). Blind h1 r6
  tracker (no scan): 0/20, **20/20 falls** at 0.10 L and 0.15 L. P2 gate for h_steps (expert >= 0.9 over 0.10-0.30 L) not met
  yet -> resume the curriculum (checkpoints now store the level; resume with --level0 0.4).
  Videos (peer `artifacts/video`, INDEX.md): `2026-09-29_humanoid_steps_h1_h0.10_s7300_expert-v2_success.mp4`,
  `..._h0.20_s7301_expert-v2_timeout.mp4` (stops before the stairs), `..._h0.10_s7300_blind-r6_fell.mp4` (blind tracker trips).
- **g1 v4ft** (target_margin 0.03 + land_vel): no-fall 1.0 on validation trials, fwd 0.96, turn 0.86, slip 0.016, CoT 0.51,
  peak force 2.12 BW (pass); FAILS joint margin 0.018 (PD overshoot past the 3% target band); waypoint 15/20 with 5 falls
  (v4: 17/20, 3 falls). -> g1 v4ft2 (target_margin 0.05) = last g1 attempt.
- **t1 v2ft3** (+ per-tick force cap): **waypoint 20/20, 0 falls**; no-fall 1.0 everywhere, fwd 1.08, turn 0.95, slip 0.054,
  CoT 0.53, **peak force 2.72 BW (passes)**; D-112 fails only joint margin 0.0094 (regressed from 0.022). -> t1 v2ft4
  (+ target_margin 0.05), recorded stop-rule deviation (sample budget ~3.3e8 of 6e8).
- **g1 v4ft2** (target_margin 0.05): WORSE: waypoint 0/20 with 20 falls, force 3.04 BW, margin 0.016, turn 0.69. g1 stops
  here (stop rule); best g1 = **v4** (only joint margin 0.012 fails; waypoint 17/20, 3 falls).
- Per-joint margin diagnosis (`humanoid_margin_diag` (throwaway diagnostic, not kept), C MuJoCo, validation command set): the worst joints are the
  KNEES at the straight-leg limit for g1 (0.018 with a 5% target band) and right HIP YAW for t1 v2ft3 (0.0085 with a 3% band):
  the PD servo overshoots the (clipped) target at stance impact. Target clipping alone cannot bound it.
- Status asked of the lead (2026-09-29): a D-113-style labelled exception for the margin/force criteria, or more iterations
  (knee-specific band / stance knee-flex term). t1 v2ft4 (5% band + force cap) is the last per-body attempt.
- **t1 v2ft4**: waypoint 20/20, 0 falls, fwd 0.87, turn 0.92, slip 0.096, CoT 0.71, force 2.35 BW; fails only margin 0.016.
  Selected by the D-139 pre-stated rule (tie on criteria, larger margin). Pool trackers for P1c-P3 (D-139 labels):
  **h1 r6** (force 3.74 BW, margin -0.001; flagged), **t1 v2ft4** (margin 0.016), **g1 v4** (margin 0.012; waypoint 17/20).
- Shared morph_v2 (6 menagerie x 1024 + 2 phum topologies x 32 bodies x 2048 worlds; clock gate, target band, force cap):
  launched 2026-09-29 (lease 1790702624_8e8209).
- h1 steps v2 resumed to 4000 iters stayed at level 0.4 (window success 0.38-0.50): the 20 s episode is too short for the
  staircase course (x_end + 0.3 L ~ 6.5 m at 0.48 m/s = 13.5 s on flat ground; slower on steps), so timeouts count as failures
  and the 0.7 level-up threshold is never reached. Next segment: 30 s episodes, level-up 0.6 (recorded before running).

## RESUME (state at the repo-refactor wind-down, owner decision via lead 2026-09-29 ~12:00; commands rewritten as recipes, D-145 P4a)
State: **no W13 lease running, no host loop running, nothing queued.** Do not re-run anything to reproduce results.
Done (all numbers in this file / D-138 / D-139):
- P0 plan + sealed split (D-138); P1a Warp adopted; P1b pool trackers under the D-139 exception: h1 r6, t1 v2ft4, g1 v4.
- P2 h_steps expert (h1) and h_gap_sidestep code (GPU env, C scenario, teacher, eval; never trained).
Stopped / final state of the last two runs:
- shared morph_v2 tracker: STOPPED by me at iter 415 of 3000 (lease 1790702624_8e8209). Last checkpoint = iter 399
  (`artifacts/runs/humanoid_p1c_shared_v2/checkpoint.pt` + `actor.pt`); still learning to stand (window fall 0.94); no gate,
  no sealed evaluation. Resume: `--recipe shared_morph_v2 --out artifacts/runs/humanoid_p1c_shared_v2 --resume`.
- h1 steps expert v2: FINISHED its segment (iter 6000, 30 s episodes). Curriculum level 0.6 (step heights up to 0.18 L),
  last window success 0.33; NOT evaluated in C MuJoCo after the resume (last C grid = the iter-1500 actor: 20/20 at 0.10 L,
  0/20 at >= 0.15 L). Checkpoint `artifacts/runs/humanoid_p2_h1_steps_v2/checkpoint.pt` (stores level 0.6).
Not started: D-139 g1 knee side attempt, gap experts, sealed transfer (no berkeley/toddlerbot adapters yet), P3 (planned as recipes, item 6 below; not run).
Artifacts: all `artifacts/runs/humanoid_*` (93 MB incl. every tracker actor/checkpoint/gate JSON) and the 31 W13 videos
(`artifacts/video/2026-09-29_{contact_h1,contact_t1,contact_g1,humanoid}_*`) are copied, sha256-verified, to
`~/work/rrp-data/peer-archive/{runs,video}` (ARCHIVE_LOG.txt); peer copies kept (small). Weights are never committed.
Peer code dirs `/dev/shm/rrp-brandonin/wt/humanoid*` (~45 MB each) can be deleted after the refactor. The isolated Warp
install `~/work/ext/pylibs/mjwarp` (468 MB, peer disk) is needed to resume GPU training.
Refactor notes: the GPU stack is `rrp.envs.warp.{model,tracker_env,task_env}`, `rrp.envs.mujoco.{morph_obs,humanoid_scenes}`,
`rrp.harness.train.{warp_tracker_ppo,tracker_recipes}`, `rrp.bodies.humanoid_gen`, `rrp.policies.teachers.humanoid`; deployment-side
options live in actor meta (clock_gate, target_margin, obs_format morph_v1, extra_obs_dim) and are honoured by
`rrp.envs.mujoco.legged_tracker.LearnedTracker` — keep them when moving to the common env/policy abstractions.

### Resume steps as recipes (D-145 P4a; the old `scripts/humanoid_*` drivers are in `.old/scripts/`)
`rrp run-dag recipes/humanoid/<name>.yaml [--dry-run]` on the peer (`RRP_PEER_REPO` = a fresh `wt/humanoid_runN` per launch;
<= 2 GPU leases; never sync a code dir with running jobs). Weights are not in git: restore the run directories named below from
`~/work/rrp-data/peer-archive/runs` into `artifacts/runs/` on the peer first. Stages: `train_tracker` (option `engine: warp`,
`resume_from`), `eval_tracker` (tasks `steps`, `gap`, `gap_smoke`, `waypoint`; tools `rrp suite {humanoid-steps, humanoid-gap,
humanoid-gap-smoke, contact-waypoint}`), `validate_tracker` (the D-112 gate). Template: `recipes/templates/humanoid_task.yaml`.
Resources in the recipes are ESTIMATES: measure and redeclare >= 1.35 x peak (D-117).
1. **h1 steps expert, C grid** -- `recipes/humanoid/h1_steps_v2.yaml`: nodes `eval` (expert actor
   `humanoid_p2_h1_steps_v2/actor.pt`, h_frac 0.10-0.30, 20 seeds; P2 gate: success >= 0.9 at every step height) and
   `eval_blind` (r6 tracker at 0.10 / 0.15). Add a training segment by dropping `train: null` (header of the file).
2. **Gap experts** -- `h1_gap_v1.yaml`, then `t1_gap_v1.yaml`: nodes `smoke` -> `train` (`{h1,t1}_gap_gpu_v1`, stop rule <= 2 reward
   designs, <= 3e8 samples per body) -> `eval` (C, `--level 1.0`).
3. **Shared morph_v2** -- `shared_morph_v2.yaml`: `train` resumes the run stopped at iter 415/3000 (`resume_from`), then the C gate
   (`eval@<body>`, tracker gate) on each of the 6 pool bodies. The SEALED zero-shot (n1 / berkeley / toddlerbot / phum sealed seeds
   `sealed_region_seeds(20)`, once) is Level-1 existing-controller transfer; it is not a recipe yet because the berkeley and
   toddlerbot adapters still have to be written.
4. **Pool tracker gate** -- `tracker_gate_pool.yaml` (h1 r6, t1 v2ft4, g1 v4): validation + D-112 gate + waypoint +
   `lab_gate.json` (replaces `humanoid_tracker_gate.sh`). The comparison videos of `humanoid_tracker_finalize.sh` are `rrp video legged`
   now; the side-by-side installed-vs-new renderer went to `.old/scripts/render_contact_compare.py`.
5. **Not a recipe (needs code first):** the D-139 side attempt (g1 knee-specific target band + stance knee-flex term; add a
   `g1_*` entry to `rrp.harness.train.tracker_recipes` (`WARP_RECIPES`), then an instance of `humanoid_task.yaml`).
6. **Transfer matrix (P3; H6)** -- one recipe per humanoid task, `recipes/humanoid/transfer_<task>.yaml` for h_steps, h_gap, h_walk, h_turn,
   h_reach, h_squat_pick, h_place (template `recipes/templates/humanoid_transfer.yaml`, family `humanoid`,
   `rrp.harness.pipelines.humanoid`: collect -> pack -> train_rep -> train_flow / train_bc -> eval_transfer -> sealed_eval).
   The driver is `rrp eval humanoid-transfer` (bodies x methods x demo budgets 5 / 20 / 100 x 2 training seeds; Level 1 apart from Level 2;
   one acquisition record per (level, task, body, budget): demos, teacher TICKS, env samples, updates 150 / 300 / 600; tables through
   `statistics.py`: Wilson k/200, Newcombe vs BC SFT). Order: `collect_src` / `collect_tgt` / `collect_sealed` (teacher demos, sealed bodies
   only on adaptation seeds) -> `pack` / `pack_sealed` (nested first-N packs) -> per system (semfix, nosem, bc) x seed the trainers ->
   `eval@<system>.s<seed>` on the non-sealed bodies (dev scenes) -> `sealed@...` (needs `vars.sealed: true`; every cell runs ONCE, logged in
   `artifacts/runs/humanoid/sealed_log.jsonl`, and a cell whose run is missing never enters `sealed_eval`). `eval_ref` / `sealed_ref` are the
   variant-less cells (Level 1 existing controllers, the scripted-teacher reference). Pooled tables:
   `rrp eval humanoid-transfer --config <point out>/transfer_config.json --tables-only`.
   What is NOT in the DAG and is reported as `missing_run` / `unaccounted` (never faked): the adapting trainers (system-0 refit from a
   demo pack, joint flow warm start, BC SFT warm start: `train_flow` / `legged_bc` have no warm start) and the Level-1 PPO fine-tune / scratch
   runs. The humanoid task shards carry no `waypoints`, which `LeggedData` requires, so `train_rep` / `train_flow` / `train_bc` refuse them with
   a StageError until the loader is task-agnostic. h_steps / h_gap name `SET_WHEN_REGISTERED` trackers (no contact_v2 tracker of those tasks is
   registered), so their collect nodes fail at once. Only t1, g1, h1 have registered trackers: source and dev pool = those three; sealed
   bodies n1 / berkeley / toddlerbot_2xc (legs tasks), g1_hands / n1 (manipulation tasks); phum sealed is not planned.
   Not planned: the carry / loco-pick tasks (U3 not merged). Training is paused: nothing here has been run.
Also here: the four never-run D-126 CPU tracker recipes `recipes/humanoid/d126_tracker_*.yaml` (superseded in practice by the GPU
recipes of `tracker_recipes.py`).

Dry-run node lists (`rrp run-dag <recipe> --dry-run`, 2026-09-30):
```
$ rrp run-dag recipes/humanoid/h1_steps_v2.yaml --dry-run
DAG humanoid_h1_steps_v2_eval: 2 nodes (source recipes/humanoid/h1_steps_v2.yaml)
- eval [planned ] eval_tracker-expert @peer cpu=4 mem=8G retries=0
- eval_blind [planned ] eval_tracker-blind @peer cpu=4 mem=8G retries=0
```
```
$ rrp run-dag recipes/humanoid/h1_gap_v1.yaml --dry-run
DAG humanoid_h1_gap_v1: 3 nodes (source recipes/humanoid/h1_gap_v1.yaml)
- smoke [planned ] eval_tracker-smoke @peer cpu=2 mem=8G gpu=4G retries=0
- train [planned ] train_tracker @peer cpu=4 mem=24G gpu=12G retries=2
- eval [planned ] eval_tracker @peer cpu=2 mem=6G retries=0
```
```
$ rrp run-dag recipes/humanoid/t1_gap_v1.yaml --dry-run
DAG humanoid_t1_gap_v1: 3 nodes (source recipes/humanoid/t1_gap_v1.yaml)
- smoke [planned ] eval_tracker-smoke @peer cpu=2 mem=8G gpu=4G retries=0
- train [planned ] train_tracker @peer cpu=4 mem=24G gpu=12G retries=2
- eval [planned ] eval_tracker @peer cpu=2 mem=6G retries=0
```
```
$ rrp run-dag recipes/humanoid/shared_morph_v2.yaml --dry-run
DAG humanoid_shared_morph_v2: 7 nodes (source recipes/humanoid/shared_morph_v2.yaml)
- train [planned ] train_tracker @peer cpu=6 mem=40G gpu=24G retries=2
- eval@t1 [planned ] validate_tracker @peer cpu=1 mem=4G retries=0
- eval@g1 [planned ] validate_tracker @peer cpu=1 mem=4G retries=0
- eval@h1 [planned ] validate_tracker @peer cpu=1 mem=4G retries=0
- eval@op3 [planned ] validate_tracker @peer cpu=1 mem=4G retries=0
- eval@apollo [planned ] validate_tracker @peer cpu=1 mem=4G retries=0
- eval@adam_lite [planned ] validate_tracker @peer cpu=1 mem=4G retries=0
```
```
$ rrp run-dag recipes/humanoid/tracker_gate_pool.yaml --dry-run
DAG humanoid_tracker_gate_pool: 6 nodes (source recipes/humanoid/tracker_gate_pool.yaml)
- validate@h1_r6 [planned ] validate_tracker @peer cpu=1 mem=4G retries=0
- waypoint@h1_r6 [planned ] eval_tracker @peer cpu=1 mem=4G retries=0
- validate@t1_v2ft4 [planned ] validate_tracker @peer cpu=1 mem=4G retries=0
- waypoint@t1_v2ft4 [planned ] eval_tracker @peer cpu=1 mem=4G retries=0
- validate@g1_v4 [planned ] validate_tracker @peer cpu=1 mem=4G retries=0
- waypoint@g1_v4 [planned ] eval_tracker @peer cpu=1 mem=4G retries=0
```
```
$ rrp run-dag recipes/humanoid/transfer_h_walk.yaml --dry-run          # the other six differ in bodies / methods only
DAG humanoid_transfer_h_walk: 29 nodes
- collect_src, collect_tgt, collect_sealed  [collect @peer cpu=6 mem=16G]
- pack, pack_sealed                         [pack @peer cpu=1 mem=2G]
- eval_ref, sealed_ref                      [eval_transfer / sealed_eval @peer cpu=4 mem=8G]
- rep@{semfix,nosem}.s{0,1}                 [train_rep @peer cpu=2 mem=3G gpu=4G]
- flow@{semfix,nosem}.s{0,1}                [train_flow @peer cpu=2 mem=3G gpu=4G]
- bc@bc.s{0,1}                              [train_bc @peer cpu=2 mem=3G gpu=4G]
- eval@<system>.s<seed>, sealed@<system>.s<seed>   [eval_transfer / sealed_eval @peer cpu=4 mem=8G]   (6 + 6 nodes)
```

## Sealed split guard and sealed-body adapters (H5, audit D12, D-146 item 2)

- `rrp.core.sealed.SealedSplit` (hash-pinned to `research/splits/humanoid_v1.json`) refuses: any training / data collection on a
  sealed body except on target-adaptation seeds; evaluation / development seeds in any training; a dataset whose shard manifests
  hold such seeds; a second run of a sealed cell (body x method x training seed, 100 evaluation scenes) unless the log
  `artifacts/runs/humanoid/sealed_log.jsonl` holds a recorded infrastructure failure with a reason. Wired into the legged
  pipeline stages and trainers; validate_tracker / edits refuse sealed bodies. The direct `rrp train tracker-*` CLI is not guarded yet.
- Adapters (`rrp.bodies.legged`): berkeley, toddlerbot_2xc / 2xm, g1_hands added; n1 pitch actuators declared. All five sealed target
  groups build and stand 1 s under PD on CPU MuJoCo (`tests/unit/test_sealed.py`, load-and-stand only; no policy, no sealed
  evaluation was run). No body is cut. Declared, not manufacturer, values: toddlerbot torque limits (mass x leg-length scaling law,
  `limits_source: declared_scaling`), berkeley gain_scale 4.0 (kp = 4 x effort; found in the stand test), g1_hands hand gains,
  toddlerbot_2xc dropped self-collision pairs (asset pairs interpenetrate at home).

## HT (D-146, 2026-09-30): tracker registry, public terrain scan, `rl_expert`
- `rrp.envs.mujoco.legged_tracker.TRACKERS[(body, version)]` (built from `artifacts/trackers/<body>/<version>/meta.json`; v1 = `contact_v1`);
  spec `"<body>:<version>"`, `make_legged_env(..., tracker=spec)` / `LeggedSession(tracker=spec)` replace monkeypatching `load_tracker`. A
  meta `sha256` pin is checked against the actor file (t1 and anymal_c contact_v2 are pinned). `TRACKER_DIR` now resolves through `rrp_home()`
  (it pointed at `src/artifacts` before).
- `terrain_scan_v1` is a public sensor: 11x7 yaw-frame elevation cells (0.1 m, x -0.2..0.8, y -0.3..0.3), noise 1 cm, 2 % dropout (reads 0.0),
  one tick of latency, ground only (floor + scene ground geoms; walls are not scanned). Warp (analytic ground) and MuJoCo (`mj_ray`) share the
  constants. The steps expert's ACTOR now consumes it (`extra_obs` = "terrain_scan", `terrain_scan` spec in the actor meta); its critic keeps the
  exact scan. Actors of the earlier privileged 11x3 scan (pre-D-146) are labelled `privileged_teacher` by `rl_expert`. The gap expert stays blind
  to walls (critic-only gap terms), so a wall-avoiding gap actor is a separate, declared sensor.
- `rl_expert:<body>:<version>` (POLICIES): registered actor under the task's scripted command layer; label `learned:rl_expert:<sha12>`
  (or `privileged_teacher:...`); `requires.privileged` because the command layer reads the base pose truth.
- One recipe registry: `tracker_recipes.{CPU_RECIPES, WARP_RECIPES, RECIPES, recipe_record}` (`humanoid_recipes.py` is gone; recorded recipe shas unchanged).

## H4 (D-146, 2026-09-30): legged collectors take `--task`
- `legged-collect` and `legged-latent-collect` take `--task` (default `waypoint_contact`), `--tracker-id <body>:<version>` and `--max-steps`
  (default `TaskSpec.max_steps`, else 1300). Env = `make_env("mujoco/legged", task=...)` (the task's `build`), teacher = policy registry key
  `TaskSpec.teacher`, status/failure vocabulary from the task judge; a task without a teacher or without `mujoco/legged` is refused.
- Events and target slots of `public_context` come from the task graph (`policies.features.legged.TaskView`, `EVENT_SLOTS=3`, `TARGET_SLOTS=2`,
  unused slots zero with flag 0; `done` is slot 3). waypoint_contact keeps the 22-dim layout bit for bit. A graph with more slots raises.
- Episode/shard meta and the manifest record task, teacher (name, version, source, privileged), tracker source/sha256; `guard_sealed` runs before
  anything is written. One output directory holds one task (`assert_one_task`). Test: `tests/unit/test_legged_collect_tasks.py`.
- Open: h_steps / h_gap have `TaskSpec.max_steps=None` (set in `tasks/humanoid.py`, U2); M2/M3/C1 graphs with >3 events or >2 targets need a
  wider `EVENT_SLOTS`/`GLOBAL_DIM` (touches HL/HX nets).

## U1 (D-146, 2026-09-30): upper-body control `control="wholebody"`
- `LeggedSession(control="wholebody")` (MuJoCo) = `legs` plus the `upper` joint_position group (the body's held joints: arms, waist, head), both 50 Hz,
  PD through the same actuators. A command carries either or both groups; a group left out keeps its last target (default stance after reset);
  `upper` is clipped to the held actuator ctrlrange; `legs`-only commands are bit-identical to `control="legs"` (test). Held-joint counts: t1 11,
  g1 17, h1 9, op3 8, apollo 20, adam_lite 13, n1 11, talos 20, phum_0 9. `perturb.install_legged` replaces the tick and holds `upper` at the default
  pose, so wholebody + that hook raises RuntimeError instead of silently ignoring the target.
- Tracker observation: an actor whose meta says `upper_obs` reads the upper joint state `(q - q0, qdot * 0.05)` appended AFTER the terrain block
  (`LeggedBinding.upper_obs`, `LearnedTracker`; dims checked; morph_v1 refuses it), so `init_shared` warm starts zero-pad the new columns.
- Warp (`WarpTrackerEnv(upper_body=True)`, trainer `--upper-body --upper-amp --upper-speed --payload-frac`): the upper joints follow a RANDOM
  slew-limited target trajectory (goal every 1-3 s, 20 % rest), a random payload in [0, 8 % of the robot mass] is added on the two hand-side bodies
  (mass plus a 0.1 m point-load inertia), the actor sees the upper state, the critic also sees payload fraction and target offset. Meta: `upper_obs`,
  `upper_body` (source `random`). Recipes `{t1,g1,h1,op3,apollo,adam_lite}_clock_gpu_ub` (t1/g1/h1 warm-start the latest gait actors, the others v4 from scratch).
  Peer smoke (t1, 256 worlds, 6 iters): env, trainer, actor export and `LearnedTracker` load work; the random arm targets cause more early falls than
  a payload alone (16/256 vs 3/256 episodes ended by tick 130 with the untrained warm start), the expected training signal. No training was run.
- Known gap: the procedural `phum_*` bodies with arms (phum_0/2/3/4/7) fail scene compile (`capabilities: 'manipulate'` is not an AssemblySpec
  literal, `humanoid_gen.py`); the CPU `LeggedEnv`/`tracker-cpu` trainer is not extended (humanoid training is on Warp).

## HL (D-144 / relations.md section 11, 2026-09-30): legged relation sites
- Catalog `legged-rel-v1` (nine edges: same_node, kin_parent, kin_child, same_assembly, mirror, node_in_assembly, limb_adjacent, foot_of,
  over_cell) and presets `legged-none` (default: no parameters, state-dict keys and outputs identical, so old checkpoints and legged goldens are
  unchanged) and `legged` (nine edges + `leg.foothold` + `leg.com_support`, for NEW lineages). `leg.com_support` is a gauss readout (mu, logvar), out 2 (was 1).
- Nets (`policies/nets/{legged_latent,legged_bc}.py`): with a relational list the public context grows from [glob, joints] to
  [glob, joints, limbs, feet, terrain cells]; Context / LeggedEncoder / LeggedFlow / LeggedBC run FactorSites at `ctx>ctx` (every context
  layer), `act>ctx`, `act>act` (every packet block; BC: the joint rows). Graph from PUBLIC tensors only (`legged_graph`); kin_parent / kin_child come
  from the per-assembly depth chain and mirror / limb_adjacent from kind + side + position, because parent ids are not in shards or `LeggedMorph`
  (HX / U1 / H4 own adding them). `foothold_next` and `com_support` labels are training-only and dropped in deploy (`set_deploy`).
- Trainer (`legged_latent_train.py`): `factors` resolved with `family="legged"`; `readout_loss` / `estimates_loss` per spec replace the scalar
  reduction and the hand-written probe loss; checkpoints carry `factors` and `versions["factors"]`; `load_legged_rep` rebuilds the nets from the
  checkpoint (unstamped = `legged-none`, refused if the config asks for relational factors); sealed guard in every stage (H5). fit_probe now uses
  unit-weight specs and each spec's `lv_min`.
- Acceptance: `tests/unit/test_legged_relations.py` (toggling `leg.foothold` and `edge.kin_parent` off changes attention and output on a tiny humanoid
  batch, encoder and BC; legged-none identity; estimates loss; deploy refusal; three trainer runs). No training or simulation was run.
- Open (not HL-owned): shards need `terrain`, `foothold_cell`, `com_support(_valid)` from collect (H4), else those labels are masked;
  `bundles.load_rep`, `policies/legged.py` and `legged_dagger.py` build nets without factors (use `load_legged_rep` / `build_legged_rep`, feed the
  terrain keys at deploy); `legged_bc.bc_batch` should use `data.train_batch` for the foothold label; the lineage recipe adds `preset:legged`.

## HX (D-146, 2026-09-30): humanoid system 0 realizes the `upper` group
- The packet already carried the arm / body assemblies (`LeggedMorph`: nf legs + body + one assembly per arm side, held joints as nodes) and the
  realizer already emitted a value for every actuated node; what was missing was the group split and the command. `nets.legged_latent`:
  `REALIZER_GROUPS = ("legs", "upper")`, `group_masks(b)` (the public node flag `node_static[:, IS_POLICY_COL]`), `LeggedRealizer.groups(...)`.
  No parameter or state-dict key was added: legs-only checkpoints load unchanged and the upper rows share the one output head.
- `policies/legged.py`: `LatentLeggedController(upper=True)` (a declaration that the realizer's upper rows were trained; never inferred) makes
  `System0Adapter` emit `upper = clip(q0_held + action_scale * a_upper, held_lo, held_hi)` next to `legs` (fallback: default stance / measured
  hold), `LeggedLatentPolicy` requires groups {legs, upper} and `control="wholebody"`; without it the policy is unchanged (legs command only;
  it also runs on a wholebody env and leaves `upper` at the default stance). `upper=True` with oracle / BC-expert packets is refused.
- Acceptance: `tests/unit/test_legged_upper.py` (tiny random models; t1 tests need the Menagerie assets): group partition, own/body-assembly routing and
  arm-knot causality on the realizer, packet with arm assemblies + both command groups on t1 wholebody, an edited packet changes the upper command.
- Open (not HX-owned): the trainer (`legged_latent_train.py`, `amask` = policy joints only) and shards (`a` holds policy rows only) do not yet
  supervise the upper rows; until collect (U2 teacher upper targets) and the trainer add them, `upper=True` is only valid for a realizer trained that way.
  Nothing stamps "upper trained" in a checkpoint yet.

## U2 (D-146, 2026-09-30): scripted upper-body teacher and tasks L0 h_walk, L3 h_turn, M1 h_reach, M2 h_squat_pick, M3 h_place
- Tasks (`tasks/humanoid.py`, graphs `tasks/graphs/h_*.json`, scenes `envs/mujoco/humanoid_scenes.py:build_h_manip` + `HumanoidSession`): mujoco/legged only
  (Warp has no upper-body scenes), `control="wholebody"` by default, 50 Hz `max_steps = 50 x seconds`. Judge vocabulary = `HUMANOID_REASONS + MANIP_REASONS`
  (`drift` ends at once; `no_grasp`, `not_upright`, `place_miss` are read at the end). Public predicates come from forward kinematics of the public body model
  (`hand_distance_m`, `height_above_m`, `bearing_error_rad`, `drift_m`, `stand_frac`); `truth_predicate` twins read the simulator; `failure_reason()` is privileged.
  The truth `base_speed` twin uses the estimator's 1 s baseline (the instantaneous velocity spiked to 0.16 m/s at 10 Hz ticks while the public estimate passed:
  the judge then raised on a public success the truth denied).
- Teachers (`policies/teachers/humanoid.py`, `MANIP_TEACHERS`, POLICIES keys `teacher:h_*`, source `scripted_teacher`, PRIVILEGED: true base pose, joint state, scene map):
  `legs` = the registered rl_expert tracker (`t1:contact_v2`) on a base velocity command for h_walk / h_turn; `upper` = damped-least-squares IK on the public body
  model (`UpperIK`). h_reach / h_squat_pick / h_place plant the feet: the tracker sags 0.83 of the stand height and fights a squat, so their legs are a static pose
  (`SquatPlanner`, CoM over the soles) plus an ankle-pitch PID on the CoM x error (the ankle servos are weak: kp 50, 20 Nm). The row label says `planned_com` for
  those and the tracker version + sha for the others. No `*_ub` wholebody tracker exists yet, so nothing here is a learned whole-body result.
- h_turn: the tracker's turning steps walk the base 0.17 m per rad, so the teacher adds a body-frame velocity command (P on the true position, 0.15 m/s cap,
  off once aligned so the base comes to rest); `drift_max` is 0.5 m (0.33 m failed 5 of 20 seeds unassisted, a stronger hold stalls the turn).
- Acceptance (host, real t1 actor, `evaluate("teacher:<task>", "mujoco/legged", ...)`, batch 1, seeds 0-19): 20/20 public AND privileged success for each of the five tasks
  (sim 2-15 s each, about 5x real time on the host CPU). `tests/unit/test_humanoid_manip.py`: registration / vocabulary / POLICIES, judge reasons, stub-actor wiring
  per task, reach-arm side edit, one real-actor episode per task (skips without the tracker or the Menagerie assets). No peer smoke was run.
- Not owned but touched: `policies/base.py` (five POLICIES keys).
- Open: the collect path that records the teacher's `upper` targets into shards (HX note above); a `*_ub` tracker so h_walk / h_turn legs and the squat share one learned controller.
- 2026-09-30: readiness RP4 (D-146, audit D29): re-checked on main fa98416a. Already on `harness.rollout` since D-140 S5e / RP1: `legged_latent_eval.run_episode` (also behind `video_legged`, the legged parts of `robustness` and `legged_dagger collect`), `deploy_eval` (options are hooks / wrappers, no loop). The one private loop left in the owned files was `tracker_validation.run_episode` (own `mj_step` loop on the bare standalone model): it is now a rollout episode of a `_BenchEnv` (Env on the bare model; the body tracker is its own controller, command = `base_velocity`), a `_ScriptedCommand` policy (source scripted_teacher) and the hooks `_Kick` (push) and `_Meter` (energy, 20 ms peak force, tracking, slip, gait, margin). Golden `loop.legged.tracker_validation` (`tests/unit/test_legged_rollout.py`; hexapod6, CPG, ideal + v1lat actuator, push, forced early fall, trajectory) was recorded on the old loop first and is byte-identical; the row key order is unchanged, so recorded validation JSONs and `gates.check_tracker` are unaffected. Same test greps the six owned files for session / `mj_step` steps outside an Env. Not owned, for X1: `viz/record.py::run_tracker_val` still carries its own `mj_step` loop (RP5 file; could reuse `_BenchEnv`); `legged_dagger.Recorder` is a per-tick callback the policy adapter invokes (`ctl.recorder`), not a loop. No peer smoke was run.
