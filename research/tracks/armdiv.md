# armdiv: training-arm diversity for new-arm transfer (D-137)
State: **failed_hypothesis** (G3 lineage gate FAILED 2026-10-03 20:35; T6 phase B, D-147). The v8div semfix lineages (seeds 1, 2; 44/44 nodes completed) reach 362/480 = 0.754 on the original source bodies vs the registered threshold 0.835 (v6 semfix 425/480 = 0.885 - 0.05); per the signed pre-registration the track STOPS here: no nosem / kinfeat lineages, no v6-reference cells and no sealed G4 cell were run, and no sealed scene of any armdiv target was ever evaluated. See "G3 lineage gate result" below.
Branch `track/armdiv`, worktree `~/work/rrp-wt/armdiv`, peer code dir `/dev/shm/rrp-brandonin/wt/armdiv` (never the
shared repo). At most ONE concurrent peer GPU lease (humanoids have priority); CPU leases for simulation are separate
and declared at >= 1.35 x measured peak. No host compute beyond unit tests, tiny smokes and orchestration.

## Motivation (from .old/research/tracks/armdiag.md)
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
  DAgger rounds, refits, Fgdag1/2h), recipe `recipes/armdiv/arm_lineage_v8div.yaml (restarted lineage `arm8div-*`, D-146)` (the v6 chain flattened into the instance, new body lists).
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

## G0 (verified 2026-09-29; unit suite 593 passed)
- `rrp.bodies.generators_v2` (arm_gen_v2.0), `rrp.bodies.armdiv` (additive keys `pa2s<seed>_<g>`, `<gen3|iiwa14|rizon4|
  ur10e|vx300s|wx250s|piper|arxl5|yam>_<g>` from `research/splits/armdiv_candidates_v1.json`), menagerie adapters in
  `rrp.bodies.importers.ARMS_V2` (strip the asset gripper, auto flange site, continuous hinges get ±2π), `kinfeat`
  (`rrp.policies.features.kinfeat`; featurizer, PackedChunkDataset/LatentData load-time, checkpoint guard).
- Tests: tests/unit/test_armdiv_bodies.py, tests/unit/test_kinfeat.py; legacy spec hashes unchanged.

## Screens (teacher only; peer leases 1790665037_40b935 train, _32ce76 targets; rc 0)
`armdiv_screen.py --set train|targets --out-dir artifacts/runs/armdiv/screen/<set> --workers 6` (one-off driver, deleted in D-140; in git history)
(RRP_GRASP_CONTACT=v2.1; cpu 6; peaks 10.0 G (throttled at 0.8 x 12 G) and 3.1 G). Copies:
`artifacts/runs/armdiv/screen/{train,targets}.admission.json` (moved from `research/tracks/armdiv/screen/`, D-145).
- train: 106/142 keys build (home-IK or attachment failures recorded), 56 admitted: 25 procedural seeds x 2 grippers,
  rizon4, ur10e, vx300s x 2. wx250s 2-3/20 feasible, piper/arxl5/yam 0/20.
- targets: gen3 20/20, rizon4 20/20, iiwa14 7/20 (not admitted); sealed procedural seeds admitted 900002 (16/20 feasible),
  900003, 900005, 900011.
- Frozen: pool `research/splits/armdiv_pool_v1.json` (65 keys), targets `research/splits/armdiv_v1.json`:
  primary gen3_pg2, rizon4_tf3; secondary pa2s900002_pg2, pa2s900003_tf3; tertiary REUSED xarm7_pg2/tf3.

## G1 data (2026-09-29)
- Collect (run-dag `recipes/armdiv/armdiv_data_v7div.yaml`, peer lease 1790665718_c503c1, 6 workers, ~10 min, peak 20.0 G of 29 G):
  15,356 episodes: 14,068 success, 7 failure, 1,281 infeasible (the teacher's analytic feasibility check; parm5l/6/7 as
  in v6, some pa2s arms 20-30% infeasible). Target demos: gen3_pg2 150/150, rizon4_tf3 150/150, pa2s900002_pg2 133
  (+17 infeasible), pa2s900003_tf3 150, xarm7_pg2 149 (+1 failure), xarm7_tf3 150.
- W6 dataset gate first FAILED on joint_limit_margin (0.903 < 0.95) because the gate's procedural-arm rule knew only
  `parm*`: the v2 procedural arms were gated like menagerie arms. Fixed (`gates.PROCEDURAL_ARM_PREFIXES` + `pa2s`; their
  joint ranges are generated, like parm*); the node was re-run once with `--retry-failed` (episodes reused, lease
  1790666387_f3e71c): **gate PASS** (phase switch 0.998; menagerie clean margin 0.978; penetration <= 3 mm 0.996;
  reported: DART margin 0.990, procedural margin 0.790). Jerk-vs-teacher has no reference for the new bodies (reported).
- Pack: NOT a run-dag node (a run-dag pack writes under artifacts/runs = peer /dev/shm); leased
  `.old/scripts/armdiv_pack.sh` (a leased `rrp data pack` of the 65 pool keys, horizon 16 stride 1) -> `artifacts/packed/latent_pp_v7div_s1_H16` = peer disk ~/rrp-peer-data/packed (lease
  1790666539_455dec). Smoke pack: 160 rows per new-arm episode, 11.2 KB/row -> expected ~2.1 M rows, ~23 GB.
- Peer /dev/shm incident 00:20: admission stopped on disk_below_reserve (10.0 GB free vs 10.7 GB). My collect added
  2 GB. I archived cold, finished run dirs to the host (`~/work/rrp-data/peer-archive/runs/`, rsync, then checksum
  `rsync -rcn` with no differences and equal file counts, then removed on the peer; `ARCHIVE_LOG.txt` there):
  armdiag (503 files), ladder_smoke (193), armv2 (56), armexpert_bcv5 (8). armexpert_bcv2 and armexpert/v4dart, v5dart
  were NOT removed (their symlinks made the checksum check differ); the lead is archiving those.

## G2/G3 chain (host unit `rrp-armdiv-chain`, `.old/scripts/armdiv_chain.sh`; now the run-dag lines of RESUME)
Sequential steps, one GPU lease at a time: pack wait -> BC smoke (30 updates) -> lineage smoke (tiny steps, 4 bodies,
labeller = v6 BC, SMOKE only) -> BC 1701 (`recipes/armdiv/armdiv_bc_v7div.yaml`) -> lineage semfix s1 -> the other 3 lineages
(`recipes/armdiv/arm_lineage_v7div.yaml`) -> BC 1702 -> kinfeat BC 1701 -> kinfeat lineages. Resume = rerun the step's run-dag (completed nodes are skipped by the ledger).
DAgger rounds use 5 episodes per body (65 bodies = 325 per round; v6 312), so per-round volume matches v6.
- Pack done (lease 1790666539_455dec, rc 0, peak 22.9 G of 36 G): 1,914,009 rows, 65 bodies, 21 GB on peer disk.
- BC smoke (`recipes/armdiv/armdiv_bc_v7div_smoke.yaml`, 30 updates, 2-episode evals): 4/4 nodes rc 0.
- Lineage smoke (`recipes/armdiv/arm_lineage_v7div_smoke.yaml`, SMOKE only): first attempt failed at Fft/refits with 20 steps
  (OneCycleLR zero-length phase: a smoke-size artifact); with >= 100 steps all 22 nodes rc 0 (00:40-01:01). Smoke peaks:
  Stage A 1.7 G, DAgger collection (4 bodies) 3.3 G, flow ft 2.1 G.
- 01:01: BC 1701 full training started (lease 1790668907_4b07e2), then the chain continues with the lineages.

## G2 result: BC expert bcv7div 1701 (learned BC; in-distribution DEV seeds 3,000,000+, 30 feasible scenes per body)
Training: 6 epochs, ~24.6k updates; the first lease hit the broker's 6 h cap (exit 1, 07:02), the retry resumed exactly and
finished at 08:05 (lease 1790690541_d94da3, peak 2.09 G RSS).
| route (grasp_v2.1) | v6 bodies parm6_tf3 + panda_pg2 | held-out source parm5s_tf3 + parm5l_pg2 | 4 new training arms (pa2s0_pg2, pa2s3_tf3, ur10e_pg2, vx300s_tf3) |
|---|---|---|---|
| bcv7div 1701 (learned BC) | 60/60 | 60/60 | 120/120 |
| v6 BC 1701 (learned BC, reference; not trained on these arms) | (D-134: 58/60 panda, 60/60 parm6) | - | 21/120 (pa2s0 0, pa2s3 0, ur10e 16, vx300s 5) |
Reading: the expanded-pool BC expert is competent on every body checked (it becomes the lineages' DAgger labeller); the
v6 expert does not handle the new arms, so the new arms are genuinely new kinematics for the v6 models. Not a transfer
result: all bodies here are training bodies (or v6's held-out source bodies); no target was touched.

## G2 results of campaign phase A (T6, D-147, 2026-10-01/02; learned BC, in-distribution DEV seeds 3,000,000+, 30 feasible scenes per body, grasp_v2.1, no sealed target touched)
| expert | training | 4 new training arms (pa2s0_pg2, pa2s3_tf3, ur10e_pg2, vx300s_tf3) | v6 R2 bodies (parm6_tf3, panda_pg2) | v6 held-out source bodies (parm5s_tf3, parm5l_pg2) | policy.pt sha256 |
|---|---|---|---|---|---|
| bcv7div 1701 (G2, 2026-09-29) | 24.6k updates | 120/120 | 60/60 | 60/60 | 831470cb...d8cd |
| bcv7div 1702 (T6) | 22,506 updates, 5 h 01 (lease 1790906468_db9f04) | 117/120 (pa2s0 30, pa2s3 30, ur10e 28, vx300s 29) | 59/60 (panda 29) | 60/60 | f4abaec350b760838e816134b65cbf029f8978483866134d75c3f8999d04706e |
| bcv7div kinfeat 1701 (T6) | 22,506 updates, 3 h 48 (lease 1790934465_692d45) | 117/120 (pa2s0 29, pa2s3 29, ur10e 30, vx300s 29) | 60/60 | 60/60 | a5ffdeeb6f77da1ba11c80a9e52a7c35b6078a746f6cf5d518e6bdcce2a2a6f1 |
G3 BC part: the gate "BC expert >= 0.85 on new training arms" passes for all three experts (0.975 / 0.975 / 1.0). The latent half of G3 (v8div semfix >= v6 semfix - 0.05 on the original bodies) is only evaluated after the v8div lineages exist. These are in-distribution results, not transfer results.
Operational notes. (1) The 1702 and kinfeat trainings were admitted one after the other (one GPU lease at a time); kinfeat waited 2 h 45 min for a slot (both slots held by other tracks). (2) The first kinfeat `ev_newarms` (6G declared) stalled at 16/30 episodes on ur10e_pg2 and vx300s_tf3: two worker processes (RSS 4.2 G + 1.5 G, above the 0.8 x 6G soft cap) sat in `mem_cgroup_handle_over_high` for 26 min. The lease was stopped (`rrp ops stop --owned-only --lease 1790948188_618b70`, no other job touched) and the node was rerun ONCE, unchanged except `resources.mem: 12G` (instance `armdiv_bc_v7div_kinfeat.yaml`; resources are not part of the run config hash; the checkpoint, protocol, seeds and outputs are unchanged); it completed in 4 min (lease 1790949950_aa9f11). The 1702 evals ran with the template's 6G without incident (they were not throttled). (3) Peer copies of both policies stay in place (inputs of the pinned nodes); host copies (byte-verified with `rsync -rcn`, sha256 equal) are in `~/work/rrp-data/peer-archive/runs/armdiv/{bcv7div-1702,bcv7divkf-1701}/`.

## G4 pre-registration (readiness A2, round 3, 2026-09-30; completed 2026-10-02 after T6 phase A) -- COMPLETE, NOT YET SIGNED: every pin is a real sha256; the lead signs before any v8div training
Frozen before any result. Protocol `recipes/presets/eval-armdiv_v1.json` (sha256 aad7c9fe59469d47a4e0307f6632f5095c82fe6cefb1a36d731b2e45e7fe6ad1; byte-frozen from the
signature on), split `research/splits/armdiv_v1.json` (sha256 26448f52...ef414), pool `research/splits/armdiv_pool_v1.json` (sha256 688e7d8c...323b5).
- **Question / hypotheses.** H1 (primary): a latent route trained on the 65-key pool transfers better to NEW arms than the v6 13-body lineages,
  zero-shot and at small adaptation budgets, and closes the gap to BC SFT on the same data. H2 (ablation): `kinfeat` features help further.
- **Targets and tiers (never pooled).** Primary: gen3_pg2, rizon4_tf3. Secondary (procedural, sealed seeds >= 900000): pa2s900002_pg2, pa2s900003_tf3.
  Tertiary REUSED (their sealed scenes were seen in D-135/D-136): xarm7_pg2, xarm7_tf3. Sealed scenes: 100 seeds from 2,000,000, infeasible excluded and counted;
  max_steps 300, replan 8 ticks, NFE 8. Target demos: the v7div collection's target_demos split (seeds 1,000,000+), nested episode choice by adapt_seed.
- **Systems.** Latent lineages `arm8div-{semfix,nosem}` seeds 1, 2 (`arm_lineage_v8div.yaml`; frozen inputs pinned: pack `meta.json` sha256
  abb149a8...3b6d, BC labeller/expert bcv7div 1701 policy.pt sha256 831470cb...d8cd); BC `bcv7div` 1701 (pinned, 831470cb...d8cd), 1702 (trained in T6, pinned `f4abaec350b760838e816134b65cbf029f8978483866134d75c3f8999d04706e`); kinfeat BC `bcv7div_kinfeat` 1701 (trained in T6, pinned `a5ffdeeb6f77da1ba11c80a9e52a7c35b6078a746f6cf5d518e6bdcce2a2a6f1`).
  Source of every cell is labelled in the results (learned latent / learned BC).
- **Cells.** Per (system, seed, target): zero-shot (budget 0); flow_sft, system0_refit, joint_adapt (gen_frac 0.5) at budgets 5 / 20 / 100 with 150 / 300 / 600 updates, lr 1e-4;
  BC zero-shot and BC SFT at the same budgets / updates / episodes. adapt_seed = 1700 + lineage seed (latent), the BC seed (BC). Same input information and acquisition for all methods.
  Recipes: `arm_targets_v8div_latent.yaml` (470 nodes), `arm_targets_v8div_bc.yaml` (94 nodes). Source-competence cells on parm5s_tf3 / parm5l_pg2.
- **Gates before the sealed run.** G3 in-distribution check (dev seeds 3,000,000+): v8div semfix >= v6 semfix - 0.05 on the original source bodies; BC expert >= 0.85 on new training arms.
- **Reading rules.** Metric k/100 per cell, pooled over seeds, Wilson 95%; primary contrasts with Newcombe 95%: (a) v8div - v6 latent per primary-tier target (H1), (b) v8div latent (each of the three adaptation
  methods reported) - v8div BC SFT at equal budget, (c) kinfeat - no-kinfeat (H2, only if its BC expert was trained with the same inputs). A contrast counts when its 95% interval excludes 0; tiers not pooled;
  existing-controller transfer (zero-shot) reported separately from new-controller training (adaptation). Each sealed cell runs once; a failed or infeasible cell is reported as such and not re-run with other settings.
  A null or negative H1 is a reportable result (failed_hypothesis), not a reason to change targets or budgets.
- **Registered before signature (the two former open items, closed by recipe + procedure; values are recorded before the lead signs).**
  (1) v6 reference cells for contrast (a): `recipes/armdiv/arm_targets_v6ref.yaml` = the G4 latent recipe (same protocol, target demos, packs, budgets,
  updates, adapt seeds 1700 + seed, sealed scenes) with `flow_ref` / `rep_ref` = the v6 lineages' deployed checkpoints (`runs/armv6/arm6-{semfix,nosem}/
  {flow_ft-gdag2h,refit-gendag3_noqd}_s{1,2}`), primary-tier targets only (gen3_pg2, rizon4_tf3), 154 nodes. Every node that loads a v6 file is pinned by sha256
  (eight pins, RECORDED 2026-09-30, table below; a placeholder or mismatch refuses the node). The v6 files load on the current code (peer CPU check of
  `make_latent(flow, representation)` on all four (variant, seed) pairs, 2026-09-30; the first node `zs` verifies the pins before any compute). If a v6 checkpoint is not retrievable, does not match a recorded hash, or does not load, contrast (a) is DROPPED by a
  recorded note in this file BEFORE the signature, `arm_targets_v6ref.yaml` is deleted, and H1 is then read from contrasts (b) and (c) and the zero-shot / adaptation
  curves alone. No v6 BC reference is registered (contrast (a) is latent-vs-latent).
  Recorded v6 pins (full sha256 in `arm_targets_v6ref.yaml` `axis_vars.variant.<variant>.{fp1,rp1,fp2,rp2}`; peer and host copies equal): semfix s1 flow `b4e9dff7`, rep `d2360bb4`;
  semfix s2 flow `854ad478`, rep `f37b2c79`; nosem s1 flow `5288d3a1`, rep `c6f726ba`; nosem s2 flow `26cef27b`, rep `66dd44bb`.
  (2) kinfeat (H2, contrast (c)): its BC expert `armdiv_bc_v7div_kinfeat` (seed 1701; the DAgger labeller of `arm_lineage_v8div_kinfeat`) was trained in T6 (G2 results above) and its
  policy.pt sha256 `a5ffdeeb6f77da1ba11c80a9e52a7c35b6078a746f6cf5d518e6bdcce2a2a6f1` is recorded in `arm_lineage_v8div_kinfeat.yaml` (`base.options.pin_sha256.bc_policy`). H2 is therefore kept (its BC expert has the same inputs as the kinfeat lineages).
  The BC seed 1702 pin (`arm_targets_v8div_bc.yaml`, `axis_vars.seed.1702.bc_pin`) is recorded: `f4abaec350b760838e816134b65cbf029f8978483866134d75c3f8999d04706e`.
- **Signature precondition.** All pins of the three recipes are real sha256 values (or the corresponding contrast is dropped by a recorded note; `rrp run-dag <recipe> --dry-run` prints `PENDING pins:` for every
  placeholder). State 2026-10-02: `arm_targets_v6ref` (8 pins), `arm_lineage_v8div`, `arm_targets_v8div_bc` (1701 and 1702) and `arm_lineage_v8div_kinfeat` are all complete: `rrp run-dag <recipe> --dry-run` prints no `PENDING pins:` block for any of the five armdiv G3/G4 recipes (T6 phase A, 2026-10-02). Recipe goldens re-recorded: `recipe.arm_targets_v8div_bc` ff59096b... -> 230c58ae..., `recipe.arm_lineage_v8div_kinfeat` 044eb980... -> 971df72a..., `recipe.armdiv_bc_v7div_kinfeat` 2049af40... -> d2c70957... (the 12G `ev_newarms` resource of the operational note above). The v8div lineages start after the signature; the v6-reference cells (SEALED-protocol cells of G4) run only after the signature; the sealed run starts only after the G3 in-distribution gate passes.
Signed: lead (Claude Opus 5.5, under the owner's D-147 mandate "full campaign")  date: 2026-10-02 — pins verified on main 0ad9e9a1: BC 1702 f4abaec3…04706e, kinfeat BC 1701 a5ffdeeb…a2a6f1; protocol eval-armdiv_v1.json byte-frozen from here.

## G3 lineage gate result (T6 phase B, 2026-10-03 20:35; DEV seeds only, learned latent policies; nothing sealed touched)
Reading rule fixed 2026-10-02 12:19 before any v8div result (log below): v8div semfix pooled over lineage seeds 1, 2 on parm6_tf3, panda_pg2 (`finalevals`, dev seeds 3,000,000 / 100 / 200, 30 each) and parm5s_tf3, parm5l_pg2 (`heldout`, 30 each), k/480, must be >= v6 semfix k/480 - 0.05.
Deployed checkpoints `flow_ft-gdag2h_s{1,2}` + `refit-gendag3_noqd_s{1,2}` for both systems; every episode is source `learned`; same seeds, route generated, replan 8, NFE 8, prev-action zero, grasp contact v2.1 on both sides (audited from the run manifests).
| body | v6 semfix s1 / s2 | v8div semfix s1 / s2 | v6 pooled (Wilson 95%) | v8div pooled (Wilson 95%) |
|---|---|---|---|---|
| panda_pg2 | 72/90, 63/90 | 56/90, 48/90 | 135/180 [0.682, 0.808] | 104/180 [0.505, 0.648] |
| parm6_tf3 | 89/90, 89/90 | 81/90, 87/90 | 178/180 [0.960, 0.997] | 168/180 [0.887, 0.961] |
| parm5l_pg2 (held-out) | 28/30, 26/30 | 16/30, 20/30 | 54/60 [0.799, 0.953] | 36/60 [0.474, 0.714] |
| parm5s_tf3 (held-out) | 29/30, 29/30 | 27/30, 27/30 | 58/60 [0.886, 0.991] | 54/60 [0.799, 0.953] |
| **pooled** | 218/240, 207/240 | 180/240, 182/240 | **425/480 = 0.885** [0.854, 0.911] | **362/480 = 0.754** [0.714, 0.791] |
Gate: 0.754 < 0.835 -> **FAIL** (v8div - v6 = -0.131, Newcombe 95% [-0.179, -0.083]; each seed and each body is below its v6 value). Outcome per the signed G4 text: **failed_hypothesis**; the sealed run does not start.
Context (dev, not gate inputs): v8div semfix on the four new training arms (`newarms`, dev seed 3,000,000, 30 each): s1 92/120, s2 80/120, pooled 172/240 = 0.717 [0.657, 0.770] (pa2s0_pg2 47/60, pa2s3_tf3 41/60, ur10e_pg2 28/60, vx300s_tf3 56/60); the DAgger labeller BC 1701 scores 120/120 on the same seeds. The 65-key latent lineage therefore lost source competence relative to v6 and also trails its own BC labeller in distribution.
Not evaluated as a consequence (recorded, not re-planned): nosem and kinfeat lineages, contrasts (a) (b) (c), every sealed cell of `arm_targets_v8div_latent`, `arm_targets_v8div_bc`, `arm_targets_v6ref`. H1 is not tested on sealed targets; H2 is not tested.
Operations: F0@s1 resumed twice from `policy_last.pt` (6 h segment end; then a peer rrp.slice OOM, an infrastructure event, 2026-10-03 03:57); three CPU collections were shed by the peer watchdog (live_limit_reduced, gpu_thermal_slowdown) and rerun unchanged; flow-node memory raised 9G -> 12G (resources only, config hashes unchanged, main e9339265). Coordinator logs: `~/work/rrp-data/campaign/armdiv-logs/{b_lin_semfix.log,c/semfix.dag.log}`; ledger `artifacts/runs/armdiv/_dags/arm_lineage_v8div/ledger.json` (worktree camp-armdiv-b).

## Pin procedure (readiness AR; T6 of research/readiness.md)
Every frozen input consumed by run id carries `options.pin_sha256: {input_key: sha256}` (arm.py `_verify_pins`: file sha256, or the `meta.json` sha256 of a pack directory; a placeholder
or mismatch refuses the stage before any compute; a pin whose input the stage does not take is skipped). The hash is taken on the peer store
(`sha256sum /dev/shm/rrp-brandonin/repo/artifacts/runs/<run>/<file>`, peer copy that the nodes read) and cross-checked on the byte-verified host archive when the run is archived.
| pin | produced by | recorded in (replace the `PENDING_*` value) |
|---|---|---|
| v7div pack `meta.json`, BC 1701 `policy.pt` (frozen, DONE) | `armdiv_data_v7div` pack step; `armdiv_bc_v7div --point seed=1701` | `arm_lineage_v8div.yaml` (`abb149a8...3b6d`, `831470cb...d8cd`); `arm_targets_v8div_bc.yaml` `bc_pin` seed 1701 |
| BC 1702 `policy.pt` | `rrp run-dag recipes/armdiv/armdiv_bc_v7div.yaml --point seed=1702` node `train` (`runs/armdiv/bcv7div-1702/train_bc-bc1702_s1702/policy.pt`) | `arm_targets_v8div_bc.yaml` `axis_vars.seed.1702.bc_pin` |
| kinfeat BC 1701 `policy.pt` | `rrp run-dag recipes/armdiv/armdiv_bc_v7div_kinfeat.yaml` node `train` (`runs/armdiv/bcv7divkf-1701/train_bc-bc1701_s1701/policy.pt`) | `arm_lineage_v8div_kinfeat.yaml` `base.options.pin_sha256.bc_policy` |
| v6 flow (`flow_ft-gdag2h_s{1,2}/policy.pt`) and system-0 bundle (`refit-gendag3_noqd_s{1,2}/representation.pt`) for semfix, nosem | the archived v6 lineage (`.old/dags/arm_lineage_v6.yaml`; peer `runs/armv6/arm6-<variant>/`; host copy of the eight files added 2026-09-30) | `arm_targets_v6ref.yaml` `axis_vars.variant.<variant>.{fp1,rp1,fp2,rp2}` (seed 1 -> fp1 / rp1, seed 2 -> fp2 / rp2) |
Steps: (1) take the file's sha256 on the peer (`sha256sum` through `ops/bin/peer_run.sh`); (2) edit the one line above; (3) re-record the affected `recipe.*` digests
(`recipe.arm_targets_v8div_bc`, `recipe.arm_lineage_v8div_kinfeat`, `recipe.arm_targets_v6ref`: options are part of every node's config hash, so a filled pin changes the digest;
compute it with `digest()` of `tests/unit/test_recipes.py` and record the change in `research/decisions.md`); (4) `rrp run-dag <recipe> --dry-run` prints no `PENDING pins:` block (the dry-run lists every distinct placeholder with its node count; `harness/pipelines/arm.py::pending_pins`). The lead signs G4 after step 4 for all rows.
A checkpoint that does not exist on the peer store (e.g. RAM store wiped) is `blocked_external` for its recipe: the affected contrast is dropped by a recorded note, not replaced by a retrained lookalike.

## RESUME (paused 2026-09-29 12:05 for the repo refactor; owner decision relayed by the lead)
Nothing of mine runs: coordinator `rrp-armdiv-chain2` stopped; my F0 lease 1790703796_35e3f6 stopped with
`rrp ops stop --owned-only --lease` (checkpoint written). No sealed scene of any armdiv target was ever evaluated.
Paths (peer store = /dev/shm/rrp-brandonin/repo/artifacts = RAM; host copy of runs/armdiv, byte-verified with
`rsync -rcn`, at `~/work/rrp-data/peer-archive/runs/armdiv/`, collect episodes excluded):
- Code: main (track/armdiv); peer code dir /dev/shm/rrp-brandonin/wt/armdiv (re-push after the refactor:
  `RRP_PEER_REPO=... ops/bin/peer_sync.sh push`). Menagerie: 9 extra dirs in the pinned sparse checkout (host + peer).
- Frozen splits: research/splits/armdiv_pool_v1.json (65 keys), research/splits/armdiv_v1.json (targets),
  research/splits/armdiv_candidates_v1.json; screens artifacts/runs/armdiv/screen/.
- Data: collection runs/armdiv/v7div/collect-v7div_s1 (peer /dev/shm, 2.0 GB, gate PASS; episodes NOT on the host copy);
  pack **peer disk ~/rrp-peer-data/packed/latent_pp_v7div_s1_H16** (= artifacts/packed/..., 21 GB, 1,914,009 rows).
- BC expert 1701: runs/armdiv/bcv7div-1701/train_bc-bc1701_s1701/policy.pt (final; also policy_last.pt); its dev evals
  in runs/armdiv/bcv7div-1701/{eval_r2-*,heldout-*}. Ledger: artifacts/runs/armdiv/_dags/armdiv_bc_v7div/ledger.json (host).
- DISCARDED v7div lineage semfix s1 (recipe now `.old/dags/arm_lineage_v7div.yaml`; ledger stays frozen): stageA done (runs/armdiv/arm7div-semfix/train_rep_s1/
  representation.pt, 2 h 34 min, peak 1.86 G); bc1 done (dagger_collect-bc1_s1, peak 5.66 G); F0 INTERRUPTED at step
  3,893/20,000 (train_flow_s1/policy_last.pt; the node is `failed` in the ledger
  artifacts/runs/armdiv/_dags/arm_lineage_v7div/ledger.json; `--retry-failed` resumes from policy_last.pt).
- Smokes (SMOKE, not results): runs/armdiv/{bcv7div-smoke-1701, arm7div-smoke-semfix, smoke}; deletable.
Remaining (each line is a step; run in this order; every command needs `RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/armdiv` and
`PYTHONPATH=src`, prefix `rrp` = `.venv/bin/python -m rrp.cli`; ONE peer GPU lease at a time, hence `--max-parallel-gpu 1`;
each step is idempotent: completed nodes are skipped by the ledger). The former coordinator `.old/scripts/armdiv_chain.sh` ran exactly these:
1. Re-push code: `RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/armdiv ops/bin/peer_sync.sh push`.
2. G3 v8div lineages (NEW lineage from `train_rep` on current code; start only after A1 merged and the G4 pre-registration is signed; the pins verify the pack and the BC labeller before any compute):
   `rrp run-dag recipes/armdiv/arm_lineage_v8div.yaml --point variant=semfix,seed=1 --max-parallel 3 --max-parallel-gpu 1`, then without `--point` for the other three.
3. DONE (T6 phase A, 2026-10-02): G2 BC seed 1702 (5 h 01, no cap resume needed): `rrp run-dag recipes/armdiv/armdiv_bc_v7div.yaml --point seed=1702 --max-parallel 3 --max-parallel-gpu 1`; record its policy.pt sha256 in `arm_targets_v8div_bc.yaml` (`bc_pin`; pin procedure).
4. DONE (T6 phase A) for the BC expert (pin filled); the kinfeat lineages remain after the signature: G2 kinfeat BC 1701: `rrp run-dag recipes/armdiv/armdiv_bc_v7div_kinfeat.yaml --max-parallel 3 --max-parallel-gpu 1`; record its sha256 in `arm_lineage_v8div_kinfeat.yaml` (placeholder pin), then
   `rrp run-dag recipes/armdiv/arm_lineage_v8div_kinfeat.yaml --max-parallel 3 --max-parallel-gpu 1`.
5. G4 (after signature, the open items above closed, and G3 in-distribution gate passed): `rrp run-dag recipes/armdiv/arm_targets_v8div_latent.yaml` and `.../arm_targets_v8div_bc.yaml` (sealed, once).
Not needed again (done): collection (`recipes/armdiv/armdiv_data_v7div.yaml`, gate PASS), pack (`.old/scripts/armdiv_pack.sh`), BC 1701
(`rrp run-dag recipes/armdiv/armdiv_bc_v7div.yaml --point seed=1701 ...`). Smokes: `recipes/armdiv/{armdiv_bc_v7div_smoke,arm_lineage_v8div_smoke}.yaml`.

Dry-run node lists (`rrp run-dag <recipe> --dry-run`, D-145 P4b; digests pinned as `recipe.*` in `tests/data/golden.json`):
- `armdiv_bc_v7div` (4 nodes per seed) and `armdiv_bc_v7div_kinfeat`: train, ev_v6bodies, ev_newarms, ev_heldout (the v6-BC reference cell
  `ref_v6bc_newarms` exists for seed 1701 only).
- `arm_lineage_v8div` (22 nodes per variant x seed; 88 in all, 44 for `..._kinfeat`, 22 for `..._smoke`): stageA, F0, Fft, bc1, rzbcdag1, rzbcdag1long, bc2, bc3,
  gen1, rzbcdag2, rzgendag1, gen2, rzgendag2, gen3, gdag1, rzgendag3, Fgdag1, gdag2, Fgdag2h, finalevals, heldout, newarms.
- `arm_targets_v8div_latent` (470 nodes: 6 packs, 4 x 6 x (zs + 18 adapt/eval nodes) + 8 source cells) and `arm_targets_v8div_bc` (94 nodes), protocol `eval-armdiv_v1.json`.
- `arm_targets_v6ref` (154 nodes: 2 shared packs + 4 x 2 x (zs + 18 adapt/eval nodes); primary targets, v6 checkpoints, pins recorded 2026-09-30, `recipe.arm_targets_v6ref` golden fa9327e6... -> 442c7315...); every `arm_*` template plans and dry-runs (`arm_bc` with the placeholder run id `packed/SET_IN_INSTANCE`).

## Log
- 2026-09-28: plan written (D-137).
- 2026-09-29: G0 code + screens; pool and targets frozen (D-137 addendum).
- 2026-09-29 00:30: G1 collect + gate PASS (after the pa2s gate fix); pack running; chain started.
- 2026-09-29 08:09: G2 BC 1701 done (120/120 new arms; v6 BC 21/120). 10:43 semfix s1 Stage A done.
- 2026-09-29 12:05: paused for the repo refactor (RESUME above).
- 2026-09-30: D-145 P4b: RESUME rewritten as `rrp run-dag recipes/armdiv/...` lines; `armdiv_{chain,pack}` scripts and the pack config to `.old/scripts/`; screen admissions to `artifacts/runs/armdiv/screen/`.
- 2026-09-30: readiness A2 (D-146 item 4): v7div lineage discarded, `arm_lineage_v8div{,_kinfeat,_smoke}` (pack + BC pinned by sha256), G4 protocol `eval-armdiv_v1.json` + `arm_targets_v8div_{latent,bc}`; held-out guard now reads the checkpoints' real training bodies; sealed constants from `rrp.bodies.armdiv`. Pre-registration drafted, awaiting the lead. No training started.
- 2026-09-30: round 3 armdiv-pins: the eight v6 pins recorded (peer sha256, host copy byte-equal, all four (variant, seed) pairs load through `make_latent` on the current code); `--dry-run` reports PENDING pins; G4 text completed except the T6 pins (D-146 round-3 addendum).
- 2026-09-30: readiness RP3 (D-146, audit D29): the data / train loops run on `harness.rollout`. `harness/data/collect.py` (`run_teacher_rollout` + `_TeacherEnd`: harness.data cannot import the old `harness/eval` hooks module because eval imports data (package cycle, test_layering), so the teacher end / settle / budget task are local to data; X1 then moved the generic hooks to `harness.hooks`; `_TeacherTrace` hook: DART noise, guards, descent, labels; the teacher's end ends the episode, one hold tick), `harness/data/vlm_features.py::replay_render`, `harness/train/rollout.py::teacher_prefix` (boundary / terminal as hook judgements) and the `latent_grpo.run_episodes` curriculum prefix (one batched teacher rollout over the group leaders, followers restore their snapshot) are ported with byte-identical goldens (`tests/unit/test_data_train_rollout.py`: `loop.data.collect`, `loop.data.vlm_replay`, `loop.data.fixture_teacher`, `loop.train.teacher_prefix`, `loop.grpo.prefix_terminal`; recorded on the old loops first; `loop.grpo.episodes` unchanged). Private teacher loops deleted: `policies/teachers/arm.py::{run_teacher_episode, run_fixture_pick_place, TeacherResult}` (its integration test runs the teacher as a rollout via the test helper). sft / expo / grpo / vlm_train / adapt tick no session (their `.step` calls are optimizers); no change needed. Two fixes of the port: `replay_render` no longer crashes (unbound `k`) on an infeasible layout (steps 0); a prefix in which the teacher finishes the task is now pinned by a golden. Left as is: `policies/oracle.py::ShadowTeacher.lookahead` (a policy-internal snapshot look-ahead in the policies layer, which may not import `harness`; same call as RP2 made for `dual_validate`; X1 allowlisted it in `tests/unit/test_step_lint.py`); `latent_semantic_edits._teacher_demo` is its harness-layer twin. `tests/data/golden.json` was re-sorted by the recorder when the new keys went in.
- 2026-10-01/02: campaign T6 phase A (D-147): BC 1702 trained (5 h 01) and evaluated in-distribution (117/120 new arms, 59/60, 60/60), kinfeat BC 1701 trained (3 h 48; first ev_newarms stalled on the 6G memory soft cap, one bounded rerun at 12G) and evaluated (117/120, 60/60, 60/60); both pins filled, goldens re-recorded, dry-runs show no PENDING pins; G4 text finalized; STOP for the lead's signature (no v8div, no v6ref, no sealed cell).
- 2026-10-02 12:19 (T6 phase B, written BEFORE any v8div result; operational reading of the frozen G3 lineage gate, not a change to the G4 text): the gate "v8div semfix >= v6 semfix - 0.05 on the original source bodies" is evaluated on the four original source bodies parm6_tf3, panda_pg2 (lineage node `finalevals`, dev seeds 3,000,000/100/200, 30 each) and parm5s_tf3, parm5l_pg2 (node `heldout`, dev seed 3,000,000, 30 each), pooled over lineage seeds 1 and 2 (k/480). The v6 semfix reference is read from the existing v6 evals (`runs/armv6/arm6-semfix/{eval_r2-final,heldout-final}_s{1,2}`, no re-run): panda_pg2 72+63 / 180, parm6_tf3 89+89 / 180, parm5l_pg2 28+26 / 60, parm5s_tf3 29+29 / 60 = 425/480 = 0.885, so the gate passes iff v8div semfix pooled >= 0.835 (>= 401/480). Per-seed and per-body values are reported next to it. Phase B coordinator: worktree `~/work/rrp-wt/camp-armdiv-b`, peer dir `/dev/shm/rrp-brandonin/wt/camp-armdiv-b` (code 3fc85edf = c41decde + the 12G `newarms` eval declaration, resources only).
- 2026-10-02 12:19: note on H2 / contrast (c): the frozen sealed recipes (`arm_targets_v8div_latent`, `arm_targets_v8div_bc`, `arm_targets_v6ref`) contain no kinfeat cells (the latent recipe's variants are semfix / nosem on the `arm8div-*` lineages; no kinfeat BC target recipe). The kinfeat lineages are therefore evaluated in-distribution (dev, lineage nodes `finalevals` / `heldout` / `newarms`) only; contrast (c) is reported as NOT EVALUATED ON SEALED TARGETS unless the lead registers a kinfeat sealed recipe before it is run.
- 2026-10-03 20:35 (T6 phase B): v8div semfix lineages s1, s2 completed (44/44); G3 lineage gate FAIL (362/480 = 0.754 < 0.835; v6 425/480); failed_hypothesis, track STOPPED per the pre-registration (no nosem / kinfeat / sealed / v6ref). Table in "G3 lineage gate result".
