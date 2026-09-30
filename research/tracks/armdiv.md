# armdiv: training-arm diversity for new-arm transfer (D-137)
State: **paused for the repo refactor** (2026-09-29 12:05; owner wind-down). G0, G1 and G2 (BC 1701) done; G3 started (semfix s1 Stage A + bc1 done, F0 stopped at step 3,893 with a checkpoint); G4 not started (no sealed run).
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

## G0 (verified 2026-09-29; unit suite 593 passed)
- `rrp.bodies.generators_v2` (arm_gen_v2.0), `rrp.bodies.armdiv` (additive keys `pa2s<seed>_<g>`, `<gen3|iiwa14|rizon4|
  ur10e|vx300s|wx250s|piper|arxl5|yam>_<g>` from `research/splits/armdiv_candidates_v1.json`), menagerie adapters in
  `rrp.bodies.importers.ARMS_V2` (strip the asset gripper, auto flange site, continuous hinges get ±2π), `kinfeat`
  (`rrp.features.kinfeat`; featurizer, PackedChunkDataset/LatentData load-time, checkpoint guard).
- Tests: tests/unit/test_armdiv_bodies.py, tests/unit/test_kinfeat.py; legacy spec hashes unchanged.

## Screens (teacher only; peer leases 1790665037_40b935 train, _32ce76 targets; rc 0)
`PY scripts/armdiv_screen.py --set train|targets --out-dir artifacts/runs/armdiv/screen/<set> --workers 6`
(RRP_GRASP_CONTACT=v2.1; cpu 6; peaks 10.0 G (throttled at 0.8 x 12 G) and 3.1 G). Copies:
`research/tracks/armdiv/screen/{train,targets}.admission.json`.
- train: 106/142 keys build (home-IK or attachment failures recorded), 56 admitted: 25 procedural seeds x 2 grippers,
  rizon4, ur10e, vx300s x 2. wx250s 2-3/20 feasible, piper/arxl5/yam 0/20.
- targets: gen3 20/20, rizon4 20/20, iiwa14 7/20 (not admitted); sealed procedural seeds admitted 900002 (16/20 feasible),
  900003, 900005, 900011.
- Frozen: pool `research/splits/armdiv_pool_v1.json` (65 keys), targets `research/splits/armdiv_v1.json`:
  primary gen3_pg2, rizon4_tf3; secondary pa2s900002_pg2, pa2s900003_tf3; tertiary REUSED xarm7_pg2/tf3.

## G1 data (2026-09-29)
- Collect (run-dag `dags/armdiv_data_v7div.yaml`, peer lease 1790665718_c503c1, 6 workers, ~10 min, peak 20.0 G of 29 G):
  15,356 episodes: 14,068 success, 7 failure, 1,281 infeasible (the teacher's analytic feasibility check; parm5l/6/7 as
  in v6, some pa2s arms 20-30% infeasible). Target demos: gen3_pg2 150/150, rizon4_tf3 150/150, pa2s900002_pg2 133
  (+17 infeasible), pa2s900003_tf3 150, xarm7_pg2 149 (+1 failure), xarm7_tf3 150.
- W6 dataset gate first FAILED on joint_limit_margin (0.903 < 0.95) because the gate's procedural-arm rule knew only
  `parm*`: the v2 procedural arms were gated like menagerie arms. Fixed (`gates.PROCEDURAL_ARM_PREFIXES` + `pa2s`; their
  joint ranges are generated, like parm*); the node was re-run once with `--retry-failed` (episodes reused, lease
  1790666387_f3e71c): **gate PASS** (phase switch 0.998; menagerie clean margin 0.978; penetration <= 3 mm 0.996;
  reported: DART margin 0.990, procedural margin 0.790). Jerk-vs-teacher has no reference for the new bodies (reported).
- Pack: NOT a run-dag node (a run-dag pack writes under artifacts/runs = peer /dev/shm); leased
  `scripts/armdiv_pack.sh` -> `artifacts/packed/latent_pp_v7div_s1_H16` = peer disk ~/rrp-peer-data/packed (lease
  1790666539_455dec). Smoke pack: 160 rows per new-arm episode, 11.2 KB/row -> expected ~2.1 M rows, ~23 GB.
- Peer /dev/shm incident 00:20: admission stopped on disk_below_reserve (10.0 GB free vs 10.7 GB). My collect added
  2 GB. I archived cold, finished run dirs to the host (`~/work/rrp-data/peer-archive/runs/`, rsync, then checksum
  `rsync -rcn` with no differences and equal file counts, then removed on the peer; `ARCHIVE_LOG.txt` there):
  armdiag (503 files), ladder_smoke (193), armv2 (56), armexpert_bcv5 (8). armexpert_bcv2 and armexpert/v4dart, v5dart
  were NOT removed (their symlinks made the checksum check differ); the lead is archiving those.

## G2/G3 chain (host unit `rrp-armdiv-chain`, `scripts/armdiv_chain.sh`)
Sequential steps, one GPU lease at a time: pack wait -> BC smoke (30 updates) -> lineage smoke (tiny steps, 4 bodies,
labeller = v6 BC, SMOKE only) -> BC 1701 (`dags/armdiv_bc_v7div.yaml`) -> lineage semfix s1 -> the other 3 lineages
(`dags/arm_lineage_v7div.yaml`) -> BC 1702 -> kinfeat BC 1701 -> kinfeat lineages. Resume = rerun the script.
DAgger rounds use 5 episodes per body (65 bodies = 325 per round; v6 312), so per-round volume matches v6.
- Pack done (lease 1790666539_455dec, rc 0, peak 22.9 G of 36 G): 1,914,009 rows, 65 bodies, 21 GB on peer disk.
- BC smoke (`dags/armdiv_bc_v7div_smoke.yaml`, 30 updates, 2-episode evals): 4/4 nodes rc 0.
- Lineage smoke (`dags/arm_lineage_v7div_smoke.yaml`, SMOKE only): first attempt failed at Fft/refits with 20 steps
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

## RESUME (paused 2026-09-29 12:05 for the repo refactor; owner decision relayed by the lead)
Nothing of mine runs: coordinator `rrp-armdiv-chain2` stopped; my F0 lease 1790703796_35e3f6 stopped with
`rrp ops stop --owned-only --lease` (checkpoint written). No sealed scene of any armdiv target was ever evaluated.
Paths (peer store = /dev/shm/rrp-brandonin/repo/artifacts = RAM; host copy of runs/armdiv, byte-verified with
`rsync -rcn`, at `~/work/rrp-data/peer-archive/runs/armdiv/`, collect episodes excluded):
- Code: main (track/armdiv); peer code dir /dev/shm/rrp-brandonin/wt/armdiv (re-push after the refactor:
  `RRP_PEER_REPO=... ops/bin/peer_sync.sh push`). Menagerie: 9 extra dirs in the pinned sparse checkout (host + peer).
- Frozen splits: research/splits/armdiv_pool_v1.json (65 keys), research/splits/armdiv_v1.json (targets),
  research/splits/armdiv_candidates_v1.json; screens research/tracks/armdiv/screen/.
- Data: collection runs/armdiv/v7div/collect-v7div_s1 (peer /dev/shm, 2.0 GB, gate PASS; episodes NOT on the host copy);
  pack **peer disk ~/rrp-peer-data/packed/latent_pp_v7div_s1_H16** (= artifacts/packed/..., 21 GB, 1,914,009 rows).
- BC expert 1701: runs/armdiv/bcv7div-1701/train_bc-bc1701_s1701/policy.pt (final; also policy_last.pt); its dev evals
  in runs/armdiv/bcv7div-1701/{eval_r2-*,heldout-*}. Ledger: artifacts/runs/armdiv/_dags/armdiv_bc_v7div/ledger.json (host).
- Lineage semfix s1 (dags/arm_lineage_v7div.yaml): stageA done (runs/armdiv/arm7div-semfix/train_rep_s1/
  representation.pt, 2 h 34 min, peak 1.86 G); bc1 done (dagger_collect-bc1_s1, peak 5.66 G); F0 INTERRUPTED at step
  3,893/20,000 (train_flow_s1/policy_last.pt; the node is `failed` in the ledger
  artifacts/runs/armdiv/_dags/arm_lineage_v7div/ledger.json; `--retry-failed` resumes from policy_last.pt).
- Smokes (SMOKE, not results): runs/armdiv/{bcv7div-smoke-1701, arm7div-smoke-semfix, smoke}; deletable.
Remaining:
- G2: BC 1702 (`dags/armdiv_bc_v7div.yaml --point seed=1702`, ~7 h GPU incl. one 6 h-cap resume), kinfeat BC 1701
  (`dags/armdiv_bc_v7div_kinfeat.yaml`).
- G3: finish semfix s1 (F0 resume, then Fft ... Fgdag2h, evals), then semfix s2, nosem s1/s2 (~5-7 GPU-h each;
  Stage A alone took 2.6 h here), then kinfeat semfix s1/s2 (`dags/arm_lineage_v7div_kinfeat.yaml`).
- G4: write the pre-registration (protocol config listing the armdiv targets, methods zero-shot / joint_adapt split
  gen_frac 0.5 / BC SFT at budgets 5/20/100, reading rules) and commit it BEFORE any sealed run; target demo packs from
  the v7div collection (target_demos split); the v6 lineages/BC evaluated on the same new targets as the no-diversity
  comparison. Not started.
- Resume command (after re-pushing code and checking the refactor kept these stage names/paths):
  `systemd-run --user --unit rrp-armdiv-chain3 --setenv=RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/armdiv
  --setenv="STEPS=lin_sf1 lin_rest bc1702 bckf lin_kf" --working-directory=$HOME/work/rrp-wt/armdiv bash scripts/armdiv_chain.sh`
  (add `--retry-failed` to the lin_sf1 run-dag call once, for the interrupted F0).

## Log
- 2026-09-28: plan written (D-137).
- 2026-09-29: G0 code + screens; pool and targets frozen (D-137 addendum).
- 2026-09-29 00:30: G1 collect + gate PASS (after the pa2s gate fix); pack running; chain started.
- 2026-09-29 08:09: G2 BC 1701 done (120/120 new arms; v6 BC 21/120). 10:43 semfix s1 Stage A done.
- 2026-09-29 12:05: paused for the repo refactor (RESUME above).
