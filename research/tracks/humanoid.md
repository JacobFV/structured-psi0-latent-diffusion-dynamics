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
  `rrp.training.tracker_recipes`, `dags/d126_tracker_*.yaml`; MJX prototype (`envs/mjx_legged.py`, contact_v2 accepted with
  `no_self_collision`; 512 envs ≈ 512 ticks/s on a CONTENDED GPU, i.e. not yet useful).

## 1. body pool (a): menagerie humanoids + a procedural family

Menagerie pinned at `c96a32d28fb5da84da38c1da4d749e7a13212855` (`scripts/fetch_menagerie.sh`; assets never committed).
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
All scenes built by `rrp.envs.humanoid_scenes` (P2), task graphs in `tasks/h_*.json`, contact_v2, sourced limits, scaled to
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
rrp dependency; `PYTHONPATH=src:~/work/ext/pylibs/mjwarp`). Code: `src/rrp/envs/warp_legged.py` (bake-off), raw JSON
`artifacts/runs/humanoid_p1a_warp_{t1,g1,h1}.json` (peer store). Model = the contact_v2 tracker world with the declared
adaptation `no_self_collision` (robot collides with ground only), identical on both sides.
- Parity (open-loop PD, same model, float32 vs float64): max |dqpos| 4.5e-6 (t1), 2.2e-6 (g1), 2.6e-6 (h1) over 50 ticks (1 s);
  over 200 ticks both engines fall identically (final heights equal to 1e-3 m). PASS.
- Throughput, env-ticks/s (1 tick = 10 substeps), idle GPU: t1 62.7k/57.7k/55.3k at 1,024/4,096/8,192 worlds; g1 52.0k/58.0k/58.0k;
  h1 106.8k/126.4k/119.5k (saturates at ~1k worlds). C MuJoCo one thread: t1 1.5k, g1 1.5k, h1 4.3k.
  Fewer solver iterations do not help (10 iters/20 ls: g1 43k, with line-search warnings; 4 iters: C side unstable).
- GPU PPO end to end (`rrp.training.warp_tracker_ppo`, t1, 4,096 worlds, horizon 24): 2.3 s/iter = 42k samples/s vs the CPU
  trainer's ~11k samples/s at 16 workers (measured 650-800/worker) => 3.8x. The pre-declared gate asked >= 5x: MISSED on
  training throughput (5.2x only on raw simulation). Decision (recorded, not silent): adopt the GPU trainer anyway, because it
  leaves the shared 20 peer CPUs to the other tracks and still cuts per-body wall-clock ~4x. Side effect: two concurrent GPU
  runs hold the GB10 at ~95% and the SoC at ~95 C, which trips the broker's `cpu_hot` admission stop for everyone; keep W13 to
  <= 2 GPU training leases and schedule short checks between runs.
- Observation parity of the GPU env (`scripts/humanoid_warp_obs_parity.py`) vs `LeggedBinding.public_obs`: <= 3e-7 on t1, g1,
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
