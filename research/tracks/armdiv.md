# armdiv: training-arm diversity for new-arm transfer (D-137)
State: **planned** (2026-09-28). Owner decision after D-135/D-136: "yes, add more training-arm diversity".
Branch `track/armdiv`, worktree `~/work/rrp-wt/armdiv`, peer code dir `/dev/shm/rrp-brandonin/wt/armdiv` (never the
shared repo). At most ONE concurrent peer GPU lease (humanoids have priority); CPU leases for simulation are separate
and declared at >= 1.35 x measured peak. No host compute beyond unit tests, tiny smokes and orchestration.

## Motivation (from research/tracks/armdiag.md)
- On the sealed new arm xarm7 the latent route (flow system i -> packet -> system 0 realizer) adapts only with joint
  flow + system-0 adaptation, and stays below BC SFT at equal data/updates (semfix 595 vs 842 of 1,200; nosem 151).
- The source realizer drives xarm7 poorly even from ideal packets (1-step TCP cosine 0.59 vs 0.96 on source). With 13
  source bodies (about 7 distinct arm kinematics) it interpolates between known arms instead of learning a
  Jacobian/IK-like map.
- Feature weaknesses (not bugs): the static joint-axis feature is the LOCAL joint axis (it carries no chain geometry:
  z for every menagerie hinge, y/z for procedural arms); the chain is visible only through dynamic public-FK features;
  the world-axis column `d.xaxis` is not rotated into the base frame while jp/jr/anchor are (harmless while mount yaw
  is 0, inconsistent for yawed mounts).

Hypothesis H1 (primary): training on many more distinct arm kinematics makes the realizer (and the flow's
body-specific packet directions) generalize, so the latent route transfers better to NEW arms (zero-shot and at
small adaptation budgets) than the v6 13-body lineages, and closes the gap to BC SFT.
H2 (secondary, ablation): informative kinematic static features (flag `kinfeat`) help new-arm transfer further.
Both can fail; a null or negative result will be recorded as failed_hypothesis, not re-tuned on sealed scenes.

## Design
### 1. Expanded training pool `armdiv_pool_v1` (all training keys fixed in `research/splits/armdiv_pool_v1.json` before data)
- Keep the 13 v6 source keys unchanged (same keys, same seeds).
- Procedural generator v2 (`rrp.bodies.generators_v2`, lineage `procedural_arm_family/v2`, `synthetic: true`), a pure
  function of (generator seed, params):
  - DoF 5-8; per-joint axis from {x, y, z} plus optional fixed link twist (non-axis-aligned joint axes in the parent
    frame, e.g. 30/45/60 deg);
  - link lengths with total-reach constraint (0.55-0.95 m), lateral shoulder/elbow offsets (UR-like), wrist geometry
    classes (spherical 3-axis, offset wrist, 2-axis wrist);
  - mounting inside the module: pedestal height 0-0.15 m (mount yaw stays 0 in the scene in pool v1: the scene/teacher
    yaw path is untested; recorded as an explicit limit);
  - joint ranges, link radius/density, kp/damping/effort scaled with link mass;
  - home pose from IK to a fixed tool-down ready pose (not the axis-pattern heuristic);
  - the legacy `procedural_arm`/`parm*` builders are untouched (their keys are bit-identical).
- Candidate procedural arms are drawn from generator seeds 0-999 (training range). Admission screen (teacher only, no
  policy anywhere): kinematic reach of the task workspace, then scripted teacher v2 + grasp_v2.1 on 20 source-range
  scenes per (arm, gripper): keep if feasible >= 0.75 and teacher success >= 0.9 of feasible. Target: ~24 admitted
  procedural arms (x pg2 and tf3), chosen as the first admitted seeds in order (no cherry-picking).
- Additional menagerie arms (pinned SHA c96a32d, same sparse fetch mechanism, additive): candidates ur10e, flexiv_rizon4,
  trossen vx300s / wx250s, agilex piper, arx_l5, i2rt yam, each with a small adapter (attachment site, gripper
  stripped, position actuators) and the same admission screen. Excluded from training: ufactory family (xarm7, lite6),
  franka near-duplicates (fr3, fr3_v2) and every (arm, module) combination that is a target; the D-135 targets
  xarm7_pg2 / xarm7_tf3 / panda_tf3 stay excluded.
- Scale: new keys get 150 clean + 150 DART episodes each (v6: 300 + 300 per key). Pack rows ~2-2.5 M (v6: 1.0 M). Peer
  disk is at 96% (84 GB free): one pack only (~25 GB), old packs untouched.

### 2. NEW sealed targets (declared in `research/splits/armdiv_v1.json` and committed BEFORE any training)
- Primary, new families: `kinova_gen3_pg2`, `kuka_iiwa14_tf3` (families kinova / kuka never in training). Pre-declared
  fallback order if one fails the teacher admission screen (target-demo seeds only): kinova_gen3, kuka_iiwa_14,
  flexiv_rizon4 (the first two admitted become targets; rizon4 enters training only if not needed as a target, decided
  before data collection).
- Secondary, in-distribution new procedural arms: two arms drawn from the v2 generator at SEALED generator seeds
  (900000+, first two admitted), one with pg2, one with tf3. They test "new arm from the training family".
- Tertiary, REUSED (clearly labelled "xarm7 reused test; scenes looked at in D-135 and D-136"): xarm7_pg2, xarm7_tf3.
- Seeds: target demos 1,000,000+ (150 per target), sealed eval scenes 2,000,000+ (100 per cell), dev 3,000,000+.
  Sealed scenes of new targets are not touched until the pre-registration commit (gate G4).

### 3. Flags (ablatable, default OFF = bit-identical legacy features)
- `kinfeat` (featurizer): (a) static joint axis in the BASE frame at the home pose (instead of the local axis);
  (b) parent-joint -> joint offset vector in the base frame at home; (c) the dynamic world-axis column rotated into the
  base frame (Rb @ xaxis). Same feature width, so model shapes are unchanged; recorded in flags/provenance.
- Red/green unit tests: legacy output unchanged with the flag off; with it on, invariance to mount yaw and to the
  local-frame convention of an equivalent chain.

### 4. Recipe (v6, unchanged except the pool)
- Data `v7div`: teacher v2, grasp_v2.1, phase-gated DART 0.08, W6 dataset gate; pack H16 stride 1.
- BC experts `bcv7div` 1701/1702 on the v7div pack (the v6 BC recipe): both the DAgger labeller / stateless-R1 expert of
  the latent lineages AND the BC baseline (same inputs, same data).
- Latent lineages `arm7div-{semfix,nosem}` x seeds 1, 2: the full `arm_lineage_v6` recipe (Stage A, F0, Fft, bc/gen
  DAgger rounds, refits, Fgdag1/2h), DAG `dags/arm_lineage_v7div.yaml` extending v6 with the new body lists.
- Ablation `arm7div-semfix-kinfeat` x seeds 1, 2 plus BC `bcv7div_kinfeat` 1701 (same input information for both).
- Comparison baseline for H1 (no new training): the existing v6 lineages and v6 BC experts evaluated on the new
  targets with the same protocol.

### 5. Evaluation
- In-distribution (dev seeds 3,000,000+): R2 deployable route on 4 original source bodies (parm6_tf3, panda_pg2 + v6's
  held-out parm5s_tf3, parm5l_pg2) and 4 new training arms; BC expert on the same. Gate: v7div semfix >= v6 semfix - 0.05
  on the original bodies (diversity must not destroy source competence), BC expert >= 0.85 on new training arms.
- Sealed (once, after pre-registration): per target x {v6, v7div, v7div-kinfeat} x {semfix, nosem} x seeds:
  zero-shot; joint_adapt (D-136 method: split, gen_frac 0.5, 150/300/600 updates at budgets 5/20/100); BC zero-shot and
  BC SFT at the same budgets/updates/demos. Metric: privileged success k/100 per cell, pooled k/200 over seeds, Wilson
  95%; primary contrasts with Newcombe 95%: (a) v7div − v6 latent (H1), (b) v7div latent − v7div BC SFT, (c) kinfeat −
  no-kinfeat (H2). Reading rules fixed in the pre-registration.
- Reported separately: existing-controller transfer (zero-shot) vs new controller training (adaptation).
- Short labelled demo videos (success and failure) at each notable checkpoint; sources labelled
  (scripted_teacher / learned BC / learned latent / privileged evaluator).

## Gates (each reported to the lead)
- G0 code (host, unit tests only): generator v2 + registry, menagerie adapters, `kinfeat` flag, pool/split files,
  DAGs, dry-run. Red/green tests.
- G1 screens + data (peer CPU): admission screens (procedural candidates, menagerie adds, sealed targets on target-demo
  seeds only); declare pool and targets (commit); collect v7div + dataset gate; pack. Tiny smoke first (2 arms x 4 episodes).
- G2 BC experts (peer GPU, 1 lease): 200-step smoke, then 1701, 1702 (+ kinfeat 1701); in-distribution eval.
- G3 latent lineages (peer GPU 1 lease + CPU collections): tiny whole-DAG smoke (few steps per node), then semfix s1
  -> in-distribution check -> nosem s1, semfix s2, nosem s2, then kinfeat s1, s2.
- G4 pre-registration commit (methods, cells, reading rules), then the sealed run once; results + audit.

## Wall-clock estimate at 1 peer GPU lease (from measured v6 durations)
| step | basis | estimate |
|---|---|---|
| G0 code | host editing/tests | 0.5 day of agent time |
| G1 screens + collect + pack | v6 collect 6.5 min / 8,250 eps at 6 workers; pack 1 M rows ~1 h | 3-5 h (incl. screens) |
| G2 BC 1701 + 1702 (+ kinfeat) | v6 BC ~3.5 h per seed incl. queue | 7-11 h |
| G3 6 lineages | v6: 4 lineages in 8.3 h at 2 GPU leases (~4 GPU-h each incl. contention) | 25-30 h |
| G4 sealed | D-136: 72 nodes; here ~3x cells (3 lineage families, 5-6 targets), GPU parts short | 8-12 h |
| total | assumes admission not starved by humanoid work | **~45-60 h wall-clock (2-2.5 days)**; smoke gates add ~3 h |
Cheapest early signal: G2 BC in-distribution + the first semfix lineage at ~15-20 h.

## Risks / limits
- Procedural arms may be kinematically unlike real arms; the menagerie targets are the real test.
- Teacher admission may reject many generator draws (z1 was feasible on 4/20); pool size is then smaller (reported).
- Pack memory/disk (2-2.5 M rows); if the peer disk falls below 40 GB free, reduce new-key episodes to 100 + 100.
- Mount yaw stays 0 (not tested for yaw); the kinfeat flag fixes the feature inconsistency but no yawed mount is trained.
- 2 seeds per cell as in D-135/D-136.

## Log
- 2026-09-28: plan written (D-137). Next: G0.
