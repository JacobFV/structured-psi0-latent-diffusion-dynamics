# track legged_vlm — legged/humanoid breadth and VLM system II on the latent-packet path

## LEGGED FIXED-SEM RERUN (legged_fixsem agent, 2026-09-26 08:15 →; worktree ~/work/rrp-wt/legged_vlm, peer dir wt/legged_vlm)
Question (D-085/D-087): the go2/hexapod6 sem results (R2 success equal, D-070/D-076; task-context steering equal, D-071/D-079;
sem had stronger probe-direction z handles on hexapod6 and the only working goal-readout handle, D-069/D-079) were measured with the
defective semantic recipe (unbounded probe NLL + shared clip → system 0 starved, packet collapsed). Rerun sem with the bounded NLL
(`latent.probe_lv_min: -4`), everything else identical (same data, seed 0, steps, flow recipe), and repeat the same evals.
Labels: learned:legged_fixsem_flow_sem_<body>_lv4/<ckpt> (DEPLOYABLE R2). nosem and original sem are the existing runs.

### State
| step | state | evidence |
|---|---|---|
| configs `configs/legged_fixsem/{rep,flow}_sem_{go2,hexapod6}_lv4.json` (diff vs original: probe_lv_min −4, name, note) | verified | generated from the saved original `config.json`s |
| Stage A + flow, go2 then hexapod6 (`scripts/legged_fixsem_train.sh`) | running | host GPU lease 1790435710_97b4ff (3 CPU / 9G / gpu-mem 4G) → `artifacts/runs/legged_fixsem_{rep,flow}_sem_{go2,hexapod6}_lv4` |
| evals per body (`scripts/legged_fixsem_eval.sh BODY`): gen gate, R2 snap_s4000 + final (dev 10000–10029), z-edit suite + task-context suite on R2 snap_s4000 (dev 10000–10019), mirror effects | planned | peer CPU |

Resume: if the training lease died, rerun `bash scripts/legged_fixsem_train.sh` under a host GPU lease (it skips finished stages; the
trainers resume from rep_last/flow_last). Then copy each finished `legged_fixsem_{rep,flow}_sem_<body>_lv4` dir to the peer store
(`/dev/shm/rrp-brandonin/repo/artifacts/runs/`), `scripts/peer_sync.sh push` with `RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/legged_vlm`,
and run the eval script under a peer CPU lease; copy `artifacts/runs/legged_{ladder,edits,gate}/<body>/*fixsem*` back.

## T1 DIAGNOSIS (t1_diag agent, 2026-09-26 05:00 →; worktree ~/work/rrp-wt/legged_vlm, peer dir wt/legged_vlm)
Questions (D-084, D-080): (1) why the SEMANTIC packet falls on the t1 deployable route (sem 38/120 vs nosem 107/120, 4 training
seeds, same recipe/data); (2) why nosem fails the stateless oracle route R1 (the robot barely moves) but succeeds on R2.
Method: closed-loop interventions on the matched dev seeds 10000–10029, and offline measurements on MATCHED recorded states
(the same states scored by both variants). Labels: learned:<ckpt> (deployable), ORACLE DIAGNOSTIC (packet = E(stateless BC chunk)),
PRIVILEGED ORACLE (packet = E(shadow scripted-teacher chunk), no generator; diagnostic only). Every number below is from the raw file named.

### Answer in one paragraph
The sem failure is a TRAINING-DYNAMICS defect of the semantic recipe, not a generator-error or packet-semantics effect. In Stage A the
semantic probe loss is a Gaussian NLL whose log-variance may shrink to −8, so it keeps decreasing (−13 at the end) and its gradient
dominates. The whole E+R+P gradient is clipped to norm 1, and the sem gradient norm is 390–580 (median; nosem 0.4–1.1), so every sem
update is scaled by ~0.002–0.004 (nosem 0.70–0.92), in all 4 training seeds and also on go2/hexapod6. System 0 (R) gets gradient only
from the realization loss, so the sem system 0 trains at an effective learning rate ~250× lower: it is undertrained. The same pressure
shrinks the sem posterior σ 10× (0.04–0.06 vs 0.54) and halves the packet's effective dimensionality (participation ratio 4.4–4.7 vs
10–11.5), squeezing balance state (pitch rate) out of the packet. Causal tests on training seed 0: (a) the sem system 0 falls even
with PRIVILEGED teacher-encoded packets and no generator (10/30 vs nosem 25/30), so the generator is not the cause; (b) retraining only
the sem system 0 with the realization loss on the frozen sem encoder gives a dose-response 3 → 15 (4k steps) → 24/30 (12k steps) on R2
(teacher packets 10 → 14 → 23/30), nosem stays 30/30; the teacher-packet failure replicates on training seeds 1 and 3 (sem 9 and 15/30 vs nosem 28 and 29/30); (c) the one-line fix of the recipe (bounded semantic NLL: probe log-variance
floor −4 instead of −8; everything else identical; a no-op for nosem, whose semantic weight is 0) gives sem R2 **29/30** (nosem 28/30)
and 25/30 with teacher packets, while the probes stay as good (goal err 0.0175 vs 0.0173, displacement 0.039 vs 0.039, subtask 0.997,
contact-swing accuracy 0.67 vs 0.19). The fix replicates on training seeds 1 and 3 (R2 25/30 and 27/30; pooled over seeds 0, 1, 3: fixed sem 81/90 vs original sem 21/90 vs nosem 83/90). The nosem R1 failure is not
an encoder collapse: with teacher-encoded packets the same nosem system 0 walks (25/30); at the states the R1-nosem loop visits, the
stateless BC expert's own chunks ask for slow motion, and both encoders read that faithfully.

### Closed-loop results (t1, dev seeds 10000–10029, 30 episodes each; training seed 0 unless stated)
| route | sem | nosem | raw (`artifacts/runs/legged_ladder/t1/`) |
|---|---|---|---|
| R2 deployable, original recipe (D-084) | 3/30 (27 fell) | 28/30 | `r2_*_t1_v2_orig_policy.jsonl` |
| PRIVILEGED teacher-encoded packets E(shadow teacher chunk) → original system 0 (no generator) | **10/30 (20 fell)** | **25/30** | `r1t_t1diag_{sem,nosem}_v2_orig.jsonl` |
| R2, flow retrained WITHOUT the packet-semantic loss (sem rep, `packet_semantic_weight` 0) | 13/30 (16 fell) | (nosem flow already has 0) | `r2_t1diag_sem_flow_w0.jsonl` |
| R2, system 0 refit 4k steps, realization only, frozen encoder (control) | 15/30 | 30/30 | `r2_t1diag_{sem,nosem}_rzctl.jsonl` |
| R2, same refit with 50% generated packets (generator-aware system 0) | 15/30 | 29/30 | `r2_t1diag_{sem,nosem}_rzgenz.jsonl` |
| R2, system 0 refit 12k steps, realization only | **24/30** | 30/30 | `r2_t1diag_{sem,nosem}_rzctl12k.jsonl` |
| teacher-encoded packets → refit system 0 (4k / 12k) | 14/30 / 23/30 | – | `r1t_t1diag_sem_rzctl{,12k}.jsonl` |
| **R2, FIX: Stage A with bounded semantic NLL (lv floor −4) + its flow (w_sem 0.5 as before)** | **29/30 (1 fell)** | 28/30 (original; fix is a no-op) | `r2_t1diag_sem_lv4.jsonl` |
| teacher-encoded packets → FIX system 0 | 25/30 (3 fell) | 25/30 | `r1t_t1diag_sem_lv4.jsonl` |
| FIX replication, training seed 1: R2 / teacher packets | **25/30** / 28/30 | 27/30 (D-084) / 28/30 | `r2_t1diag_sem_lv4_s1.jsonl`, `r1t_t1diag_sem_lv4_s1.jsonl`, `r1t_t1diag_nosem_v2s1_orig.jsonl` |
| FIX replication, training seed 3: R2 / teacher packets | **27/30** / 21/30 | 28/30 (D-084) / 29/30 | `r2_t1diag_sem_lv4_s3.jsonl`, `r1t_t1diag_sem_lv4_s3.jsonl`, `r1t_t1diag_nosem_v2s3_orig.jsonl` |
| original sem, teacher packets, seeds 1 / 3 | 9/30 (21 fell) / 15/30 (14 fell) | 28/30 / 29/30 | `r1t_t1diag_{sem,nosem}_v2s{1,3}_orig.jsonl` |

**Pooled over training seeds 0, 1, 3 (dev seeds 10000–10029 each).** R2 deployable: original sem 21/90 (3+8+10), FIXED sem **81/90** (29+25+27), nosem 83/90 (28+27+28).
Teacher-encoded packets: original sem 34/90 (10+9+15), fixed sem 74/90 (25+28+21), nosem 82/90 (25+28+29). The fix's Stage-A update scale
is 0.16 / 0.16 / 0.15 (median grad norm 11 / 10 / 11) and its held-out realization MSE 0.0176 / 0.0141 / 0.0164 (original sem seed 0: 0.024;
nosem 0.014). Seed 2 was not rerun (budget); the fix was trained once per seed with the original seed's config otherwise unchanged.
Reading: generator-aware system-0 training adds nothing over the equal-length control (15 vs 15), so system-0 sensitivity to generator
error is not the lever. Removing the Stage-B semantic loss helps (3 → 13) but leaves 16 falls. What moves sem to nosem level is giving
system 0 the training it was denied (dose-response) or not starving it in the first place (the bounded-NLL fix).

### Mechanism evidence (offline, matched states)
Buffers (`scripts/t1_diag_collect.sh`, peer lease 1790424402_920e35, rc=0; seeds 22000–22015, disjoint from dev; received packet and
shadow-teacher chunk recorded at packet ticks): `artifacts/runs/t1_diag/buf/{bc, r1_{sem,nosem}_v2, r2_{sem,nosem}_{v2,v2s1,v2s2,v2s3}}`
(peer store and host). The recorded R2 routes reproduce D-084 on new seeds: sem 5, 2, 7, 8 /16 vs nosem 16, 15, 15, 14 /16.
- **Clip scale** (`artifacts/runs/t1_diag/clip_scale.txt`, from the Stage-A train logs): median grad norm sem 389 / 414 / 460 / 582, nosem
  0.4 / 0.4 / 0.5 / 1.1 (seeds 0–3); mean update scale min(1, 1/gn) sem 0.0038 / 0.0037 / 0.0034 / 0.0020, nosem 0.92 / 0.91 / 0.90 / 0.70
  (seed 3's log covers only 26 of 60 intervals). go2 sem 0.0036 vs nosem 0.94; hexapod6 0.0042 vs 1.00. Fix (lv floor −4): median gn 11.1,
  scale 0.16. Final training realization loss: sem 0.018 / 0.017 / 0.027 / 0.027 vs nosem 0.008 / 0.007 / 0.012 / 0.016; fix 0.014.
- **Latent geometry** (`diag_<ts>.json` → geometry, held-out teacher rows): sem posterior σ 0.054 / 0.059 / 0.041 / 0.040 vs nosem 0.55;
  participation ratio of μ 4.7 / 4.6 / 4.4 / 4.5 vs 11.5 / 10.3 / 10.0 / 10.6; KL/entry 4.1 / 4.2 / 3.9 / 3.6 vs 0.76–0.81. nosem is NOT
  near-collapsed: 396–398 of 640 entries have KL > 0.1 and it spreads variance over twice as many directions; its KL is low because its
  posterior is wide. Fix: σ 0.106, PR 6.8, KL 2.8 (`geom_lv4.json`).
- **Balance channel** (`balance_{v2,v2s1,lv4}.json`, ridge with λ chosen on an inner split, R² on held-out episodes, z = E(BC chunk)):
  on the R2-nosem states the nosem packet encodes gravity-x/y R² 0.86/0.95 (s0), 0.83/0.88 (s1) and pitch rate 0.71/0.69; the sem packet
  0.48/0.76, 0.56/0.80 and pitch rate −0.01/−0.09 (none). BC states: same pattern. Fix: pitch rate 0.42 (R2-nosem states) / 0.34 (BC).
  System 0 also receives the IMU directly, so this is a correlate, not by itself a cause.
- **One-step system-0 error does NOT separate the variants** (`diag_<ts>.json` → sets.*.err_ratio; ratio to hold-still, BC-chunk labels,
  ticks 0–19 after a packet). E.g. seed 0 on R2-sem states: oracle 0.051 / 0.050, generated 0.048 / 0.048 (sem / nosem); on R2-nosem
  states 0.023 / 0.024 and 0.023 / 0.023; the same for seeds 1–3 and in the first 2 s (`early_<ts>.json`). Against the shadow-TEACHER
  first action (`readout_<ts>.json`, err_vs_teacher_j0, teacher packets) sem is worse on BC states in all 4 seeds (0.078–0.088 vs
  0.061–0.069) and on R2 states in seeds 0–2 (e.g. seed 0: 0.232 vs 0.189 on R2-sem states), but not in seed 3 (0.122 vs 0.127). This is why the offline gate passed while the closed
  loop failed: the deficit is closed-loop.
- **System-0 sensitivity to generator error** (the D-080 candidate): refuted. Generated packets are 150–475 posterior σ² from E(BC chunk)
  for sem vs 4–15 for nosem (relative gap 0.22–0.37 vs 0.34–0.62), but the action change per unit packet error is smaller for sem
  (0.020–0.026 vs 0.046–0.060), and the resulting error with generated packets equals the oracle-packet error in both variants. Packet
  boundary jumps are LARGER for nosem (0.014–0.035 vs 0.006–0.010 of hold-still; `temporal_<ts>.json`); sample spread is equal (0.002–0.007).
- **Balance feedback of system 0** (`feedback_<ts>.json`; action change for a 0.05 rad extra tilt in the IMU, packet fixed, as a gain on the
  stateless BC expert's own change): sem responds more strongly to pitch (1.38–1.74 vs 1.19–1.43) and less to joint velocity
  (qd×1.2: 0.19–0.25 vs 0.26–0.47), across seeds (seed 2, the smallest gap, is the closest). The fixed model keeps the high pitch gain
  (1.44–1.55) but moves the velocity gain to nosem level (0.28–0.34). So a high tilt gain alone does not cause falls; weaker velocity
  damping is a candidate correlate, not a tested cause.
- **Fall onset** (`diag_<ts>.json` → falls; tilt > 0.35 rad): sem R2 falls start early (mean onset 2.5 / 3.8 / 7.8 / 7.1 s; dev rows: 21 of
  27 seed-0 falls before 4.5 s), in walk_to_a, with BOTH feet in stance at onset (40 of 41 falls in seeds 0–3) and the robot
  tipping forward/sideways (pitch+ 15, roll+ 21, roll− 5). Onset is spread over the packet (ticks-in-packet histogram flat), so no
  knot/replan boundary is implicated, and no single leg (no swing foot at onset). The received-packet one-step error rises only in the
  last second before onset (0.12–0.17 of hold-still vs 0.02 in non-fall episodes), i.e. after the state has already left the
  distribution. In the first 2 s the sem robot accelerates harder than nosem or BC (1–2 s forward speed 0.44 / 0.50 m/s for seeds 0/1 vs
  nosem 0.35 / 0.42, BC 0.37; computed from the buffers' poses).

### Q2: why nosem fails R1 but walks on R2
- Same nosem system 0: E(stateless BC chunk) packets 0/30 (D-079); PRIVILEGED teacher-encoded packets **25/30** (`r1t_t1diag_nosem_v2_orig`).
  So the system 0 can walk; the R1 packet SOURCE is the problem.
- At the states the R1-nosem loop visits (`readout_v2.json`, r1_nosem): the requested forward displacement read from the packet is
  0.19 for E(BC chunk), 0.40 for E(teacher chunk) and 0.32 for a flow sample (nosem post-hoc probe; the sem encoder reads the same:
  0.18 / 0.37 / 0.30); realized ≈ 0.00. The BC chunks there move less than the teacher's (RMS vs hold-still 0.56 vs 0.70). So the
  hypothesis "the low-KL nosem encoder maps BC chunks to stand" is REFUTED: both encoders read the BC chunk faithfully, and the BC expert
  itself asks for slow motion at those states. R1 is a closed loop between the stateless BC expert (queried every 0.4 s, then held for the
  whole packet) and system 0. It can settle into a slow regime; the flow, trained on teacher chunks, requests more speed at the same states.
  Which variant falls into this regime is seed-dependent (D-082: seed 1 is the reverse, sem 0 / nosem 12). R1 is not an upper bound for R2
  on t1, and it is not evidence about the packet.

### Consequences
- The t1 result "semantic supervision hurts" (D-084) is a result about the RECIPE (unbounded Gaussian NLL + joint clipping), not about
  semantic content. With the bounded NLL, the sem packet matches nosem (seed 0: 29/30 vs 28/30) with its probes intact.
  With the bounded NLL the sem packet matches nosem on the deployable route over 3 training seeds (81/90 vs 83/90). So on t1 the
  evidence is now: no semantic advantage, and no intrinsic disadvantage. D-084 should be re-read as a recipe artifact.
- The same starvation is present in the go2 and hexapod6 sem Stage-A runs (clip scale 0.004); those bodies tolerated it (go2 R1 sem 25 vs
  nosem 30 is consistent with a weaker sem system 0). Any sem-vs-nosem comparison trained with this recipe is confounded by optimizer
  starvation of system 0. The arm pipeline should be checked for the same pattern (not checked here).
- Recommended recipe change for all legged sem runs: bound the probe NLL (lv floor) or clip gradients per objective. Re-run sem-vs-nosem
  comparisons after that.

### Commands and leases (all one-shot, rc checked)
- buffers: `scripts/t1_diag_collect.sh` (peer 1790424402_920e35, rc=0).
- offline: `python -m rrp.learning.legged_t1_diag --ts <ts>` (host 1790425274_e1569e: seeds 0–1, then shed by the host watchdog;
  seeds 2–3 + readout/temporal/early in host 1790425919_94fca1, rc=0); `… feedback|balance|geom <ts> <out>` (host 1790428424_8baa0c,
  1790428386_cc232e, 1790429083_bb8674, 1790429274_f5b87c).
- refits: `scripts/t1_diag_refit.sh` (host GPU 1790424619_63e81c; 12k: 1790426510_275aa4) → `artifacts/runs/t1diag_rz_{sem,nosem}_{ctl,genz,ctl12k}`.
- flow w/o semantic loss: peer GPU 1790424650_5d90c4 → `artifacts/runs/t1diag_flow_sem_w0`.
- fix: `scripts/t1_diag_lv4_chain.sh` (peer GPU 1790425883_41b926) → `artifacts/runs/t1diag_{rep,flow}_sem_lv4`; seeds 1, 3:
  `scripts/t1_diag_lv4_seeds.sh` (peer GPU 1790429049_c48a2e, rc=0) → `t1diag_{rep,flow}_sem_lv4_s{1,3}`; evals 1790431753_6bcda7,
  1790434263_edace8; teacher-packet baselines seeds 1/3: 1790431849_32aabe (all rc=0).
- closed-loop evals: `scripts/t1_diag_evals.sh {oracle|refit|refit_r1t|flow_w0|lv4|lv4s}` (peer CPU leases 1790424669_b9c86e,
  1790425909_364c58, 1790426505_ad95e5, 1790426521_18fcf8, 1790425883_7be876, 1790428371_c9daec, 1790428386_41f922).
  Note: r1t rows record the source as `privileged_oracle_packet:<latent space>`; the refit system 0 used is in the file name and the command.
- clips (labelled, INDEX.md): `artifacts/video/2026-09-26_t1diag_*` — seed-0 R2 sem fall (10001, 2.6 s) vs R2 nosem success; teacher-packet
  sem fall vs nosem success; fixed sem (bounded NLL) success on the same seed.
- code: `src/rrp/learning/legged_t1_diag.py`; `legged_dagger.py` (Recorder `--diag`, generator-aware refit `gen_flow`);
  `model/legged_latent.py` + `legged_latent_train.py` (`latent.probe_lv_min`, default −8 = old behaviour).

## RESEARCH RESTART (2026-09-25 22:15 →; legged agent; supersedes the wind-down state below for the legged part)
Question: does the corrected architecture (system i → packet z[4 knots, legs+body(+arms), 32] → system 0 at 50 Hz)
work on legged/humanoid bodies, and do the packet semantics (per-leg contact per knot, goal waypoint, base
displacement, subtask, fall) causally control behaviour? Arm lessons applied up front (B-1, D-050, D-052..D-056).
Compute: host GPU (leases below), host CPU for closed-loop sims. Data copied to host `~/work/rrp-data/datasets/legged_latent_v{1,2}` (lease 1790399700_94cca7, --disk 2G).

### IN FLIGHT / RESUME (updated 2026-09-26 03:25)
t1 second training seed (seed 1) of the sem/nosem pair tests whether the t1 R2 gap (sem 11–13 vs nosem 26–27) is seed variance:
- Stage A seed 1 is done: `artifacts/runs/legged_rep_{sem,nosem}_t1_v2s1` (KL sem 4.29 / nosem 0.77; realization 0.022 / 0.013, the same as seed 0).
- Flows seed 1 resume on the host GPU from flow_last.pt (the watchdog shed them once under external memory pressure). Rerun with `bash scripts/legged_flow_pair.sh t1_v2s1` under an ops lease; it resumes automatically.
- DAgger-1 buffers for seed 1: peer lease 1790418228_0124ec → `artifacts/runs/legged_buf/dag1_{sem,nosem}_t1s1` (peer store). Then copy them to the host and run `bash scripts/legged_refit_pair.sh t1s1 dag1` (host GPU; configs `configs/legged_dagger/rz_*_t1s1_dag1.json`).
- Then copy the realizers and flows to the peer and run `bash scripts/legged_t1s1_eval.sh` (R1 DAgger-1; R2 with the final flow using the original and the DAgger-1 system 0). Append the result to FINAL.
Host-side jobs are shed whenever external memory PSI > ~25 (the tensorcode processes). The peer is the reliable place for CPU evals.

### LEGGED RESEARCH RESULT FINAL (frozen 2026-09-26 ~03:00 PDT; legged agent). The demo builds from this table.
All numbers are closed-loop successes on the matched dev seeds 10000-10029 (30 episodes; waypoint_contact = walk to a, walk to b, halt; privileged evaluator), taken from the raw rows under `artifacts/runs/legged_ladder/<body>/`. Labels: scripted_teacher (privileged); BC = learned plain behaviour cloning (no packet; the positive control); R1 = ORACLE DIAGNOSTIC (packet = E(stateless BC chunk at the current state) → learned system 0; not deployable); R2 = DEPLOYABLE latent route (learned system-i flow samples the packet from public context every 0.4 s → learned system 0 at 50 Hz). sem = packet-semantic supervision; nosem = capacity-matched, no semantic loss.

| body | teacher | BC | R1 sem / nosem | R2 sem / nosem (deployable) | raw rows |
|---|---|---|---|---|---|
| go2 (quadruped, learned tracker) | 30/30 | 30/30 | 25 / 30 | 4k: 29 / 30; 8k: 30 / 30; **final: 30 / 30** | `go2/{teacher_bc5k,bc_bc20k,r1_*_go2_v2,r2_*_go2_v2_{snap_s4000,snap_s8000,policy}}.jsonl` |
| hexapod6 (6 legs, CPG tracker) | 30/30 | 30/30 | 30 / 30 | 4k: 30 / 30; **final: 30 / 30** | `hexapod6/{teacher_v1,bc_v1,r1_*,r2_*_hexapod6_v2_{snap_s4000,policy}}.jsonl` |
| t1 (humanoid, learned tracker) | 30/30 | 24/30 | original system 0: 12 / 0; DAgger-1: 18 / 0; DAgger-2: 12 / 0 | original system 0 (8k): 6 / 24; original system 0 (final): 3 / 28; DAgger-1 (8k): 7 / 26; **DAgger-1 (final): 11 / 27**; DAgger-2 (final): 13 / 26 | `t1/{teacher_v1,bc_v1,r1_*_t1_{v2,dag1,dag2},r2_*_t1_{v2_snap_s8000,dag1_snap_s8000,dag1_policy,dag2_policy}}.jsonl` |
| g1 (humanoid, "limited qualification" tracker) | 25/30 (arc_only teacher) | **2/30** (replan 5), 7/30 (replan 2), 4/30 (replan 3) | not run | not run | `g1/{teacher_v1,bc_v1,bc_replan*_dev30}.jsonl` |
The positive control fails on g1, so no latent-route claim is made for g1 (Stage A was trained, `artifacts/runs/legged_rep_*_g1_v2`, but not evaluated in closed loop).

**Conclusions.**
1. The corrected architecture works on legged bodies. The DEPLOYABLE route matches plain BC on go2 (30/30) and hexapod6 (30/30), and on the t1 humanoid the nosem route (26–27/30) matches or exceeds BC (24/30). It is the first competent deployable latent-packet route in the project (the arm route is still not competent, D-056).
2. Why legs work and the arm did not: feature parity was exact from the start (no B-1 analogue), qd dropout removed the velocity shortcut (qd zeroed costs ≤ 1.5 pp of hold-still error), and system-0 accuracy on BC-visited states is 1–3% of hold-still on all three bodies (arm: 42–74%, D-052). How much system 0 relies on the packet depends on the body: z shuffled gives 12–13% of hold-still on go2 but 41–43% on hexapod6. The hexapod's system 0 cannot walk without the packet.
3. sem vs capacity-matched nosem, TASK SUCCESS: equal on go2 and hexapod6. On t1 they disagree by route. On R1 (oracle packets) sem is better (12–18 vs 0). On the deployable R2 route nosem is much better (26–27 vs 6–13). That is identical treatment, the same seeds, and two DAgger rounds; one training seed per variant. A second training seed of the t1 pair is training (host lease 1790416099_b4991b). The R1 gap is NOT from per-step realization accuracy. On matched states the two system 0s have equal error (cross gate: 0.043 vs 0.042 of hold-still on R1-sem states), and with generated packets they are also equal (1.5% vs 1.3% on BC states). Why nosem fails R1 but succeeds on R2: the nosem R1 humanoid barely moves (mean speed 0.06 m/s vs 0.26 on R2; 21–26/30 never reach waypoint a). Its generator produces packets far from E(BC chunk) (relative gap 0.40 on BC states, 0.64 on R1-nosem states; sem 0.23–0.26). **Hypothesis (untested):** the low-KL nosem encoder (KL 0.75 vs sem 4.1) maps BC's chunks at learner states into its "stand" region. The flow was trained on E(teacher chunks) and samples moving packets. So R1 is not an upper bound for R2 on t1, and the oracle-packet diagnostic can mislead when the encoder's posterior is nearly collapsed. Why sem falls on t1 R2 (17–24/30 falls): not diagnosed.
4. Causal packet semantics (deployable route, 20 seeds, paired, matched-norm random controls; tables above): turn and halt are causal handles on go2 in BOTH packets. On hexapod6 the SEMANTIC packet is more steerable: its halt stops the robot (−0.19 m vs nosem +0.03 m), and its turn edit is 1.5–3x stronger. The goal readout is a weak causal handle only in the sem packet (go2 +6–7 cm, hexapod6 +1.1 cm toward the mirror; nosem at chance). Per-leg contact per knot is neither decodable (swing accuracy 0.19–0.42) nor steerable on any body. Task-context edits (system i regenerates the packet from a mirrored active waypoint) steer both packets equally (go2 +0.36 / +0.44 m, hexapod6 +0.17 / +0.17 m; irrelevant edit ≈ 0). A task-view `halt` does not stop either (out of distribution). t1 edits are running (peer lease 1790416178_246312 → `artifacts/runs/legged_edits/t1/r2*_dag1_policy/`).
5. Net semantic verdict for legged: no task-success advantage on go2/hexapod6, and a replicated DISADVANTAGE on the t1 humanoid's deployable route (2 seeds; see the seed table below). Mixed causal-control evidence: the semantic packet exposes some more reliable probe-defined handles (hexapod halt/turn; goal readout), while the no-semantic packet is equally or more steerable through the task context and on go2 turns. There is one t1 reversal in each direction, from a single training seed. The acceptance claim "semantic supervision adds causal control" is PARTIALLY supported (probe-edit handles on hexapod6), not established.

**t1 humanoid causal edits on R2 (final flows + DAgger-1 system 0s; 20 seeds; peer lease 1790416178_246312; `artifacts/runs/legged_edits/t1/r2{,ctx}_{sem,nosem}_dag1_policy/`).** nosem (no falls in the unedited runs, 0/20): turn +0.6 → Δyaw +0.26 [0.22, 0.30]; turn −0.6 → −0.31 [−0.38, −0.26]; halt → Δforward −0.34 m [−0.40, −0.27]. Random |dz| 4–12 gives |Δyaw| ≤ 0.03 and Δforward ≥ −0.04. The goal-readout edit is at chance (+0.03 m, the same as random). Task context: mirroring the active waypoint → +0.40 m [0.28, 0.51] toward the mirror (yaw +0.97); irrelevant inactive mirror −0.01; task-view halt: no stop. sem: 10/20 unedited runs already fall inside the 5 s window, so its edit effects are confounded by falls. Directionally: turn +0.20 / −0.17, halt −0.31 m, task-context mirror +0.25 m [0.17, 0.33], inactive +0.01, goal readout +0.02 (n.s.). On the humanoid's deployable route, the nosem packet is the controllable one; the sem route is limited by falls.

**t1 seed replication (appended 2026-09-26 ~04:15; second training seed of the sem/nosem pair, seed 1, identical recipe and evaluation; raw `artifacts/runs/legged_ladder/t1/*_t1s1_*.jsonl`, `r2_*_t1_v2_orig_policy.jsonl`):**
| t1, dev seeds 10000-10029 | seed 0 sem / nosem | seed 1 sem / nosem |
|---|---|---|
| R2 final flow, original system 0 | 3 / 28 | 8 / 27 |
| R2 final flow, DAgger-1 system 0 | 11 / 27 | 16 / 26 |
| R1 (oracle), DAgger-1 system 0 | 18 / 0 | **0 / 12** |
**Reading:** on the humanoid's DEPLOYABLE route, nosem beats sem in both training seeds (26–28 vs 3–16/30; sem fails by falling). The R1 oracle gap flips sign between seeds (18/0 → 0/12), so the earlier "first sem-over-nosem gap" was seed variance of the oracle diagnostic, not a semantic effect. **Seeds 2 and 3 (appended ~05:25; same recipe; R2 with the final flow and the original system 0; `artifacts/runs/legged_ladder/t1/r2_*_t1_v2s{2,3}_orig_policy.jsonl`):** seed 2 sem 17/30 vs nosem 24/30; seed 3 sem 10/30 vs nosem 28/30. **Across 4 training seeds (R2, original system 0): sem 3, 8, 17, 10 (pooled 38/120 = 0.32) vs nosem 28, 27, 24, 28 (pooled 107/120 = 0.89).** nosem wins in every seed, and sem's failures are falls. On the t1 humanoid, packet-semantic supervision HURTS the deployable route. This is robust to training seed. The mechanism is not diagnosed. Candidates: the sem packet carries more information (KL 4.1–4.3 vs 0.75–0.77), so generator error perturbs a balance-critical system 0 more; or the semantic loss shapes z in ways that don't matter for control.

Clips (INDEX.md lines, all labelled): go2 BC success; go2 R1 nosem success / sem fall / sem halt + turn edits; go2 R2 nosem success, sem success, sem fall (s10017), nosem turn edit, sem halt edit, sem task-context mirror-active vs irrelevant mirror-inactive; hexapod6 BC; t1 BC.

### LEGGED RESEARCH RESULT (working notes, superseded by FINAL above; updated 2026-09-26 00:3x; every number from the raw files named)
**Headline (go2 quadruped, matched dev seeds 10000-10029, privileged evaluator):**
| route | source label | success | raw |
|---|---|---|---|
| R0 teacher | scripted_teacher (privileged) | 30/30 | `artifacts/runs/legged_ladder/go2/teacher_bc5k.jsonl` |
| BC positive control (no packet) | learned:legged_bc_go2_v1/policy.pt | 30/30 | `bc_bc20k.jsonl` |
| R1 stateless oracle E(BC chunk) → system 0 | ORACLE DIAGNOSTIC | nosem 30/30, sem 25/30 (5 fell) | `r1_{nosem,sem}_go2_v2.jsonl` |
| **R2 deployable: flow → packet → system 0** | learned:legged_flow_{nosem,sem}_go2_v2/snap_s4000.pt | **nosem 30/30, sem 29/30** (1 fell) | `r2_*_go2_v2_snap_s4000.jsonl` |
| R2, flow snap_s8000 | learned:…/snap_s8000.pt | nosem 30/30, sem 30/30 | `r2_*_go2_v2_snap_s8000.jsonl` |
| R2, final flow (12k) | learned:legged_flow_{nosem,sem}_go2_v2/policy.pt | **nosem 30/30, sem 30/30** | `r2_*_go2_v2_policy.jsonl` |
This is the first competent deployable latent-packet route in the project. On the legged body, the corrected architecture matches plain BC. The arm lessons were applied from the start (feature parity exact; qd dropout; stateless BC expert). Why legs are easier than the arm (hypothesis, untested): the gait is periodic, so system 0 predicts most of the target from proprioception + oscillator phase, and the packet has to carry only the low-dimensional residual (speed, turn, stop). The arm needs precise object-relative geometry.

**Causal packet edits (go2, 20 dev seeds, every packet edited from t=2 s, effect in t=2–5 s, paired vs unedited; mean [95% bootstrap CI]; sem edits use its jointly trained probe, nosem its post-hoc measurement probe; controls = random z directions of matched norm, plus z=0).** Deployable route R2 (snap_s4000):
| edit (target read by the probe) | sem | nosem | matched random |dz| 4–12 |
|---|---|---|---|
| turn +0.6 rad (Δyaw) | +0.22 [0.14, 0.30] | +0.37 [0.27, 0.48] | −0.01..+0.04 (CIs include 0) |
| turn −0.6 rad (Δyaw) | −0.17 [−0.23, −0.13] | −0.23 [−0.29, −0.19] | |
| halt (Δforward, m) | −1.39 [−1.48, −1.29] | −1.19 [−1.28, −1.09] | −0.02..−0.13 |
| goal mirrored laterally (lateral displacement TOWARD the mirrored side, m; sign-normalized by the true goal side) | +0.074 [0.042, 0.110] (|dz| 6.2) | +0.025 [0.011, 0.043] (|dz| 2.3) | +0.018 / +0.002 at |dz| 8 |
| leg-0 stance / swing (Δ leg-0 contact fraction) | −0.00 / −0.02 [−0.03, −0.01] | +0.00 / +0.01 (wrong sign) | −0.01..+0.05 |
R1 gives the same pattern (turn sem +0.18/−0.15, nosem +0.30/−0.15; halt −1.24 / −1.23; goal mirror toward-mirror sem +0.064 [0.040, 0.090] vs random +0.013, nosem +0.013 ≈ random +0.019; contact ±0.02). (An earlier unsigned lateral metric had called the goal edit null. The sign-normalized metric, `scripts/legged_mirror_effect.py` → `mirror_effects.json`, replaces it.) Raw: `artifacts/runs/legged_edits/go2/{r1,r2}_{sem,nosem}*/effects.json` + `*.jsonl`.
**Task-context edits on R2 (system i generates the packet from an EDITED public task context; same seeds/window; `artifacts/runs/legged_edits/go2/r2ctx_{sem,nosem}_snap_s4000/mirror_effects.json`):** mirroring the ACTIVE waypoint's lateral estimate moves the robot toward the mirrored side: sem +0.36 m [0.22, 0.50] (yaw +0.61 rad), nosem +0.44 m [0.30, 0.59] (yaw +0.51). Mirroring both waypoints gives sem +0.34, nosem +0.41. The irrelevant control (mirroring only the INACTIVE waypoint) gives +0.04 [−0.02, 0.11] and +0.05 [−0.01, 0.11]. Setting the task view to `halt` does NOT stop the robot (forward +0.14 m sem, +0.46 m nosem). In the data, halt only ever happens when the robot is standing at waypoint b, so a halt request far from b is out of distribution for system i. So meaning flows task → packet → behaviour for the goal direction in both packets, with no semantic advantage.
**Reading:** the packet causally controls turning and stopping on the deployable route, well beyond matched-norm random edits. Semantic supervision is NOT needed for this: the no-semantic packet is at least as steerable, through a probe fit afterwards. The one semantic-specific handle is the goal readout. Editing z so the sem probe reads a mirrored goal moves the robot 6–7 cm toward the mirror side in 3 s (3–5x the random control). The same edit through the nosem post-hoc probe is at chance. This effect is small. Per-leg contact per knot is barely decodable (probe swing accuracy 0.41 / 0.38) and not steerable. So for sem vs capacity-matched nosem on legged: equal task success, and no semantic advantage in causal control. This is the same conclusion as the arm (D-059).
Clips: `artifacts/video/2026-09-26_legged_learned-R2_*` and `2026-09-25_legged_*` (INDEX.md lines).
Generator gap at BC-visited states (|z_gen − z_orc| / |z_orc|): 4k 0.20 / 0.28, 8k 0.16 / 0.25, final 0.15 / 0.24 (sem / nosem). System-0 error with generated packets is 1.2% of hold-still at every snapshot (`artifacts/runs/legged_gate/go2_*_gen_*.json`).
**Second body, hexapod6 (6 legs, CPG tracker), deployable route R2 (flow snap_s4000; the flows were moved from the peer to the host GPU at step 2500, exact resume): sem 30/30, nosem 30/30** (BC 30/30, teacher 30/30, R1 30/30 and 30/30). Raw `artifacts/runs/legged_ladder/hexapod6/r2_*_snap_s4000.jsonl`. On the hexapod, system 0 depends on the packet far more than on go2 (z shuffled: 41–43% of hold-still vs 12–13% on go2).
**hexapod6 causal edits on R2 (snap_s4000, 20 seeds, same protocol; `artifacts/runs/legged_edits/hexapod6/`).** The hexapod walks slowly (about 0.5 m in the 3 s window), so the effects are smaller in absolute terms:
| edit | sem | nosem | matched random |dz| 4–12 |
|---|---|---|---|
| turn +0.6 (Δyaw rad) | +0.106 [0.099, 0.114] | +0.034 [0.031, 0.038] | sem +0.006..+0.019, nosem ≈0 |
| turn −0.6 | −0.113 [−0.121, −0.105] | −0.072 [−0.077, −0.067] | |
| halt (Δforward m) | **−0.19 [−0.21, −0.18]** | **+0.03 [0.02, 0.04] (no stop)** | −0.001..−0.007 |
| goal mirror (toward-mirror lateral m) | +0.011 [0.007, 0.015] | +0.001 (chance) | +0.001..+0.002 |
| leg-0 stance/swing (Δ contact) | 0.00 / −0.00 | −0.00 / +0.01 | |
| z = 0 | fwd −0.47, leg0 contact +0.40 | fwd −0.56, +0.33 | |
| task ctx: mirror active waypoint (toward-mirror lateral m) | +0.165 [0.125, 0.210] | +0.168 [0.114, 0.225] | inactive: +0.001 / −0.005 |
| task ctx: halt | no stop (+0.02) | no stop (+0.01) | |
**On the hexapod, the semantic packet is the more steerable one.** Its probe-defined halt stops the robot (nosem: no effect), and its turn handle is 1.5–3x stronger. Task-context steering (goal mirror via system i) is equal for the two. Task success is equal (30/30).
**Other bodies (BC positive control, dev seeds 10000-10029):** hexapod6 BC 30/30 (teacher 30/30); t1 humanoid BC 24/30 (teacher 30/30). **g1 humanoid: the positive control FAILS**: BC 2/30 at replan 5 (28 falls) vs the arc-only scripted teacher 25/30 (5 halt not completed). A pilot on 10 seeds gave replan 2 → 6/10 and replan 1 → 1/10; a 30-seed check of replan 2/3 is running. The g1 latent route is not meaningful until g1 BC is competent (the g1 tracker is only "limited qualification"). Raw `artifacts/runs/legged_ladder/{hexapod6,t1,g1}/`. hexapod6 and t1 latent routes: Stage A done / training on the peer (see the plan table).

### plan / state
| step | state | evidence |
|---|---|---|
| 0. feature parity (train rows vs deployed featurizers, 3 re-simulated go2 episodes incl. DART) | verified | `artifacts/runs/legged_parity_go2/parity.json`: re-sim == stored (max abs 0); system-0 inputs q/qd/imu/touch/osc + morphology identical (0.0); system-i ctx identical at step-start ticks (0.0; mid-step ticks reuse the step-start ctx in training, and deployment only reads ctx at step-aligned replans). No teacher command or previous action enters any learned input (the tracker's last_a stays inside the frozen tracker). |
| 1. BC positive control (flow, same public inputs, 40-tick chunk, replan 5 ticks, NFE 8), go2 | **verified: competent** | 20k steps (lease 1790400275_8dce13), held-out first-tick MSE 0.024 vs hold-still 1.11 (`artifacts/runs/legged_bc_go2_v1/result.json`). Closed loop, go2 dev seeds 10000-10029: **BC 30/30** [0.886, 1.0] (`artifacts/runs/legged_ladder/go2/bc_bc20k.jsonl`, peer lease 1790403263_cf4e40); scripted_teacher 30/30 (`teacher_bc5k.jsonl`, same seeds); BC@5k 26/30 (drift_a 2, drift_b 2; `bc_bc5k.jsonl`). The data, teacher, tracker and eval support a learned controller on go2. |
| 2. Stage A go2 sem + capacity-matched nosem, qd dropout 0.5, resumable | completed | 12k steps (lease 1790400276_a2a139). Held-out teacher episodes, realization MSE (action units; zero-action 1.41): sem 0.019, nosem 0.0125. Joint probe (sem): contact 0.76 (swing 0.42), goal err 0.017, disp xy 0.034 / yaw 0.018, subtask 1.0, fall 1.0. KL sem 3.7 vs nosem 0.58 (nosem z carries much less). Post-hoc probes: running. `artifacts/runs/legged_rep_{sem,nosem}_go2_v2/result.json` |
| 3. offline gate on BC-visited states (R vs hold-still, qd-zero, z-zero/shuffle step gain) | **verified: passes** | BC-visited buffer: 24 BC episodes (seeds 20000-20023, 24/24 success, labels = stateless BC chunk at every tick; `artifacts/runs/legged_buf/bc_go2`). System-0 error as a fraction of hold-still (gate ≤ 0.20), BC-visited / teacher held-out: sem 0.011 / 0.017, nosem 0.011 / 0.012; first tick of a packet (j=0) 0.012 / 0.011. Shortcut measures (BC-visited): qd zeroed 0.014 (sem) / 0.014 (nosem), so no velocity shortcut after qd dropout; z zeroed 0.185 / 0.145, z shuffled across states 0.121 / 0.128, so the packet cuts the error ~11x. Caveat: most of the legged target is predictable from proprioception + osc (step gain with a shuffled z is still 0.93; with z zeroed 0.77–0.78), because the gait is periodic. The packet carries the residual (speed, turn, stop), which is exactly what the task needs. `artifacts/runs/legged_gate/go2_{sem,nosem}_go2_v2.json` |
| 4a. R1 stateless oracle (packet = E(BC chunk at the current state), replan 0.4 s) → system 0, go2 dev seeds 10000-10029 | **completed** | **nosem 30/30** [0.886, 1.0]; **sem 25/30** [0.664, 0.927] (5 fell). With qd zeroed at system 0 (diagnostic): nosem 30/30, sem 10/30 (15 fell, 5 drift). ORACLE DIAGNOSTIC (not deployable: the packet encodes BC's chunk). This is the first competent latent-packet realization on any body in this project. Unlike the arm (D-052), the legged system 0 is not the bottleneck. The sem system 0 is less robust (falls) and relies more on qd. `artifacts/runs/legged_ladder/go2/r1*_go2_v2.jsonl` (peer lease 1790403763_2ff2c5) |
| 4. ladder on matched dev seeds 10000-10029: R0 teacher / BC / R1 stateless oracle E(BC chunk) / R2 generated; failure stages fell/stall/drift_a/drift_b/halt | implementing | `src/rrp/evaluation/legged_latent_eval.py` (--bc, --oracle-bc, --rep, --realizer, --zero-qd) |
| 4b. causal packet edits on the R1 route, go2 (20 dev seeds 10000-10019; every packet from t=2 s is edited; window t=2–5 s; paired effect vs the unedited run; bootstrap 95% CI). Edits are gradient steps on z against a probe: sem uses its jointly trained probe, nosem uses its post-hoc measurement probe (fit on frozen z). Controls: random directions of matched norm, plus z=0 | **completed** | see the table in LEGGED RESEARCH RESULT; raw `artifacts/runs/legged_edits/go2/r1_{sem,nosem}/*.jsonl`, `effects.json` (peer leases 1790404196_da402d, 1790405348_b9b25f) |
| 4c. post-hoc probes on frozen z (held-out teacher episodes) | completed | sem / nosem / metadata-only: displacement xy err 0.036 / 0.043 / 0.47; yaw 0.018 / 0.027 / 0.12; goal 0.017 / 0.128 / 0.39; subtask 1.0 / 0.87 / 0.36; contact acc 0.758 / 0.756 / 0.746 (swing 0.41 / 0.38 / 0.0). Per-leg contact is barely decodable in either packet (stance is the majority class). `artifacts/runs/legged_rep_{sem,nosem}_go2_v2/probe_posthoc.json` |
| 5. BC-expert DAgger refit of system 0 | implementing (not needed for nosem: R1 is already 30/30) | `legged_dagger.py collect/refit` |
| 6a. system-i flows sem (packet-semantic loss 0.5) / nosem, go2, 12k steps | running (snap_s4000 evaluated) | host lease 1790404339_79d01e, `artifacts/runs/legged_flow_{sem,nosem}_go2_v2` (flow_last.pt every 500, snaps every 4k) |
| 6b. **R2 DEPLOYABLE route** (flow samples z from public context every 0.4 s → system 0 at 50 Hz; no oracle, no BC at run time), go2 dev seeds 10000-10029, flow snap_s4000 | **completed: competent** | **nosem 30/30** [0.886, 1.0]; **sem 29/30** [0.833, 0.994] (1 fell). BC 30/30 and teacher 30/30 on the same seeds; mean episode time 12.2 s (nosem) / 13.3 s (sem) vs BC 12.0 s and teacher 9.9 s. Generator gap at BC-visited states: \|z_gen − z_orc\| / \|z_orc\| = 0.20 (sem) / 0.28 (nosem); system-0 error with generated packets is 1.3% of hold-still for both (oracle packets: 1.1%). Source label learned:legged_flow_{sem,nosem}_go2_v2/snap_s4000.pt. Raw `artifacts/runs/legged_ladder/go2/r2_*_snap_s4000.jsonl`, `artifacts/runs/legged_gate/go2_*_gen_snap_s4000.json` (peer lease 1790406416_51e790) |
| 6c. causal edits on R2 (sem vs nosem) | running | peer lease 1790406715_50e7ac → `artifacts/runs/legged_edits/go2/r2_{sem,nosem}_snap_s4000/` |
| 7a. BC positive control, other bodies (dev seeds 10000-10029; same recipe, 20k steps) | completed (hexapod6, t1); g1 training | hexapod6 (CPG tracker): BC **30/30**, teacher 30/30 (`artifacts/runs/legged_ladder/hexapod6/*_v1.jsonl`). **t1 humanoid**: BC **24/30** [0.63, 0.91] (4 fell, 2 halt not completed); teacher 30/30 (`artifacts/runs/legged_ladder/t1/*_v1.jsonl`, peer lease 1790405735_79f597). Held-out first-tick MSE vs hold-still: hexapod6 0.0003 / 0.047, t1 0.024 / 0.87. |
| 7b. hexapod6 Stage A (peer GPU, lease 1790407365_10709f) + gate + R1 | **completed** | Held-out realization MSE sem 0.0002 / nosem 0.0001 (zero-action 0.086, CPG tracker). Gate on BC-visited states (24 BC episodes, 24/24; `artifacts/runs/legged_buf/bc_hexapod6` on the peer): system-0 error / hold-still sem 0.015, nosem 0.019. qd zeroed 0.016 / 0.020. **z zeroed 2.8 / 1.3 and z shuffled 0.41 / 0.43, so unlike go2, the hexapod's system 0 depends heavily on the packet.** R1 (ORACLE DIAGNOSTIC) dev 10000-10029: **sem 30/30, nosem 30/30**; with qd zeroed 30/30 and 30/30 (peer lease 1790410814_0ccadf). Raw `artifacts/runs/legged_ladder/hexapod6/r1*_hexapod6_v2.jsonl`, `artifacts/runs/legged_gate/hexapod6_*`. Flows + post-hoc probes: running (peer lease 1790410813_c5776a). |
| 7c. t1 humanoid Stage A (peer GPU, lease 1790407746_9a7c70) + gate + R1 | completed; the R1 route is NOT competent | Gate on BC-visited states (24 BC episodes, 20/24 success; labels = stateless BC): system-0 error / hold-still sem 0.025, nosem 0.029 (passes ≤ 0.2); qd zeroed 0.039 / 0.044; z shuffled 0.07 / 0.14; z zeroed 0.33 / 0.36. **R1 (ORACLE DIAGNOSTIC) dev 10000-10029: sem 12/30 (14 fell, 4 halt), nosem 0/30 (29 fell)**, vs BC 24/30 and teacher 30/30. With qd zeroed: 1/30 and 0/30. So the humanoid's system 0 passes the offline gate but compounds error in closed loop and falls, which is the arm pattern (D-052). The semantic packet does better here (12 vs 0). **Is the sem > nosem gap a balance difference in system 0?** Cross gate: each system 0 is scored on the states visited by EACH R1 route (pre-refit, stateless-BC labels, `artifacts/runs/legged_gate/t1_{sem,nosem}_on_r1{sem,nosem}_states.json`). On R1-sem states: sem 0.043, nosem 0.042 of hold-still. On R1-nosem states: sem 0.216, nosem 0.163. On matched states the nosem system 0 is NOT less accurate (it is slightly better), so per-step realization accuracy does not explain the gap. The difference is in where each closed loop goes. The nosem route reaches harder states (zero-action norm 2.3 vs 1.16 on its own states). Post-hoc probes show that both packets carry the displacement (xy err 0.040 sem / 0.050 nosem; yaw 0.028 / 0.039; goal 0.021 / 0.163; metadata-only 0.28 / 0.11 / 0.40). KL: sem 4.1, nosem 0.75.
**BC-expert DAgger round 1** (32 R1-route episodes per variant, seeds 21000-21031, labels = stateless BC chunk; plus the 24 BC episodes; system-0 refit 4k steps on the frozen encoder, qd dropout 0.5, IDENTICAL for sem and nosem; host lease 1790413947_24170a): **R1 sem 18/30** [0.42, 0.75] (7 fell, 5 halt), **nosem 0/30** (21 drift_a: never reaches waypoint a, 9 fell). Raw `artifacts/runs/legged_ladder/t1/r1_{sem,nosem}_t1_dag1.jsonl`. After DAgger, the nosem humanoid no longer mostly falls, but it does not navigate. Causal turn/halt edits on these routes, and t1 R2 (flows snap_s8000 + the DAgger system 0s), are running. Raw `artifacts/runs/legged_ladder/t1/r1*_t1_v2.jsonl`, `artifacts/runs/legged_gate/t1_*`. |
| 7d. g1 | blocked at the positive control | BC 2/30 (replan 5), 7/30 (replan 2), 4/30 (replan 3) vs teacher (arc_only) 25/30: `artifacts/runs/legged_ladder/g1/`. A g1 Stage A pair is training on the host GPU anyway (lease 1790408341_6bb36b; the host GPU was otherwise idle), but no latent-route claim will be made while BC fails. |


Worktree `~/work/rrp-wt/legged_vlm`, branch `track/legged_vlm`, peer dir `/dev/shm/rrp-brandonin/wt/legged_vlm`.

## inventory (2026-09-25)
- Legged code from the pre-correction track exists (report `research/reports/legged_breadth.md`, D-023).
- Frozen tracker actors that SURVIVED (git-tracked): `artifacts/trackers/go2/actor.pt` (learned PPO iter 1199),
  `artifacts/trackers/hexapod6/actor.pt` (learned iter 774; fails full-stance halt). CPG trackers need no weights
  (pquad4, hexapod6, hexapod6_long, sprawl4, sprawl8).
- NOT in the live checkouts: anymal_c, g1, t1, h1 actors and the old legged teacher datasets. FOUND in the host
  archive `~/.archive/relational-robot-policy-2026-09-25-original/artifacts/trackers/*/actor.pt` (iters 1274 / 3599 /
  2999 / 2999, matching legged_breadth.md; go2 actor sha identical to the tracked one). Copied (not committed) to the
  peer store `artifacts/trackers/<body>/actor.pt`. The VLM weights/feature caches were lost (re-downloaded below).

## log
- 12:3x t1 tracker retrain with the recorded v3 recipe (same args as `artifacts/trackers/t1/meta.json`, seed 1),
  lease 1790363836_501e49, out `/dev/shm/rrp-brandonin/legged_runs_lv/t1` (state: running). This is a RE-RUN of a
  lost checkpoint; it is gated again by `tracker_validation` before any use. STOPPED at iter ~400 once the original
  t1 actor was found in the archive (not needed).
- Tick-level collector `rrp.data.legged_latent_collect` (50 Hz native labels = clean tracker joint targets, DART
  execution noise sigma in {0,0.1,0.2,0.3} action units, randomized teacher gains). Smoke 4/4 go2 episodes.
  Full collection: lease 1790364032_2c1515, bodies go2(learned) pquad4 hexapod6 sprawl4 sprawl8 hexapod6_long (CPG),
  seeds 0-599 each -> `artifacts/datasets/legged_latent_v1/<body>/s*.npz` (peer disk). state: completed.
- INCIDENT (host, ~12:49): copying the 4.0 GB psi0 snapshot to host disk (~/work/rrp-data/hf) pushed host free disk
  below the watchdog reserve (393.7 GB; host disk was already within ~4 GB of it because of the peer->host data
  mirror). The host watchdog shed my own download lease AND another track's job `binding_v1_reeval`
  (lease 1790363904_b7ef3b, ran 31 min, stopped_by=disk_below_reserve). I deleted the host copy immediately
  (free disk back above reserve). All VLM work now runs on the peer only (weights on peer disk
  ~/rrp-peer-data/cache/hf). The binding track needs to relaunch that job.
- psi0 System-II weights: peer HF_HOME=/home/brandonin/rrp-peer-data/cache/hf, USC-PSI-Lab/psi-model@4c6f977,
  subfolder psi0/pre.fast.2605160748.ckpt.ego390k; model.safetensors sha256 b2ab7b35...06f0 (matches the HF LFS oid;
  verified with sha256sum). Packages (transformers 4.57.1 etc.) in an isolated --target dir
  /home/brandonin/rrp-peer-data/vlm_pkgs (the shared peer venv is NOT modified); use PYTHONPATH=src:<that dir>.
- xet download stalled at 75 kB; HF_HUB_DISABLE_XET=1 downloaded 4.0 GB in 7.5 min.

## STATE AT WIND-DOWN (2026-09-25 ~14:00, lead instruction: stop, the critical path has priority)

Overall: (A) legged = implementing (pipeline built and smoke-verified end to end; NO trained model finished; no
closed-loop result). (B) VLM system II = implementing (weights + code ready; no result). Nothing here is evidence
for or against the latent-packet architecture yet.

### what exists and was verified
| item | state | evidence |
|---|---|---|
| legged packet contract: assemblies = one `leg` per foot + `body` (+ `arm` per side for humanoids, held), opaque handles, knots (0.1,0.3,0.5,0.7) s, replan 0.4 s, system 0 at the 50 Hz native tracker rate, osc-v1 recurrent state | code | `src/rrp/control/legged_latent.py` |
| tick-level teacher data (source `scripted_teacher` driving the frozen body tracker; label = clean native joint targets; DART execution noise) | completed | `artifacts/runs/legged_vlm_dataset_summary.json` (committed); data on peer disk `artifacts/datasets/legged_latent_v{1,2}` (not committed) |
| models E / R (system 0) / P (packet probe: per-leg contact at each knot; body goal, displacement, subtask, fall) / system-i flow | code + smoke | `src/rrp/model/legged_latent.py`, `src/rrp/learning/legged_latent_train.py` |
| leakage guards (system 0 invariant to task context; probe reads only z; metadata-only control has no z) | test passes (3/3, host) | `tests/unit/test_legged_latent.py` |
| closed-loop eval: system i -> `LatentActionChunk` (check_packet) -> system 0 inside LeggedSession; closed-loop packet probes vs truth; packet edits (mirror_goal, halt, probe_yaw:+-X gradient edit, freeze, zero); privileged oracle-packet diagnostic (E on a shadow teacher rollout); labelled videos | code + smoke | `src/rrp/evaluation/legged_latent_eval.py` |
| end-to-end smoke (go2 only, 40 episodes, 300 rep steps + 300 flow steps): runs; both episodes FELL (expected at 300 steps; not a result) | smoke only | peer `/dev/shm/rrp-brandonin/lv_smoke2/` (tmpfs) |
| scripted_teacher reference on dev seeds 10000-10019 with the frozen trackers: 20/20 success on go2, pquad4, hexapod6(CPG), sprawl4, sprawl8, hexapod6_long | completed | `artifacts/runs/legged_vlm_teacher_ref/eval_dev.{jsonl,summary.json}` |
| system II: psi0 System-II weights downloaded + hash-verified on peer disk; isolated package dir; instruction families (color order; distance order, needs the image), declared static camera, answer scoring + probe readout, closed-loop A/B/C over the public task binding (oracle / default / system2) | code only (grounding smoke was stopped before its first batch finished) | `src/rrp/model/system2.py`, `src/rrp/evaluation/system2_eval.py`, `artifacts/runs/legged_vlm_system2_smoke/example_front_camera.png` |

Dataset (episodes / 50 Hz ticks / teacher success; failures at DART sigma >= 0.2 are caused by the injected
execution noise, the teacher is 100 % at sigma 0): go2 600 / 323k / 598; pquad4 600 / 984k / 279 (+16 falls);
hexapod6 600 / 1.07M / 258; hexapod6_long 600 / 1.00M / 328; sprawl4 600 / 1.13M / 405; sprawl8 600 / 1.08M / 263;
t1 600 / 452k / 562 (sigma set 0-0.15); g1 (arc_only teacher) 600 / 1.35M / 552. anymal_c: not collected (stopped).

### what ran and was stopped (no final artifacts)
- Stage A `legged_vlm_rep_{sem,nosem}_v1` (6 bodies, 12k steps planned): STOPPED by lead request at step 9200 / 9400
  (lease 1790365668_333a10). `train_rep` writes no intermediate checkpoint, so nothing is resumable; only the train
  logs remain (`artifacts/runs/legged_vlm_rep_*_v1/train_log.jsonl`). Last TRAINING-BATCH values (not held-out, not
  evidence): realization MSE (action units; zero-action ~1.2) sem 0.0016, nosem 0.0008; KL sem 3.39 vs nosem 0.097.
  The nosem posterior is close to collapse (z carries little), consistent with system 0 predicting the next target
  mostly from current proprioception; whether the packet is causally used can only be shown by the closed-loop
  packet edits, which were not run.
- t1 tracker re-train (lease 1790363836_501e49): stopped at iter ~400 because the original actor was found.
- anymal_c collection: stopped (no shard written). g1/t1 shards complete.
- system II CPU grounding smoke (lease 1790367341_6ba1c2): stopped by me (peer memory pressure).

### not done
Trained representation / flows; held-out probe numbers; post-hoc and metadata-only probes; closed-loop success of the
learned policy; oracle-packet diagnostic; causal packet edits; videos of learned policies; humanoid (t1/g1) latent
training; any system II accuracy or closed-loop result. No claims are made for any of these.

### resume (exact; each is a separate expensive run: start only when the lead releases GPU time)
```bash
cd ~/work/rrp-wt/legged_vlm && export RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/legged_vlm && scripts/peer_sync.sh push
# 1. stage A (sem + nosem in parallel, then post-hoc probes + metadata-only control); ~75 min on an uncontended GB10
scripts/peer_run.sh --gpu --gpu-mem 20G --cpu 4 --mem 40G --label legged_vlm_rep --max-seconds 18000 --detach -- bash scripts/legged_latent_rep.sh
# 2. stage B (system-i flows, 15k steps each)
scripts/peer_run.sh --gpu --gpu-mem 16G --cpu 4 --mem 32G --label legged_vlm_flow --max-seconds 14400 --detach -- bash scripts/legged_latent_flow.sh v1
# 3. closed-loop evals on CPU (6 bodies x 20 dev seeds; packet edits on go2/pquad4)
scripts/peer_run.sh --cpu 6 --mem 16G --label legged_vlm_eval_sem --max-seconds 14400 --detach -- bash scripts/legged_latent_evals.sh legged_vlm_flow_sem_v1 6
scripts/peer_run.sh --cpu 6 --mem 16G --label legged_vlm_eval_nosem --max-seconds 14400 --detach -- bash scripts/legged_latent_evals.sh legged_vlm_flow_nosem_v1 6 /dev/shm/rrp-brandonin/repo/artifacts/runs/legged_vlm_rep_nosem_v1/probe_posthoc.pt
#    oracle-packet diagnostic (system 0 alone) and videos (GPU lease for EGL):
scripts/peer_run.sh --gpu --gpu-mem 4G --cpu 2 --mem 8G --label legged_vlm_video --max-seconds 3600 -- PY -m rrp.evaluation.legged_latent_eval --flow artifacts/runs/legged_vlm_flow_sem_v1/policy.pt --bodies go2,hexapod6 --seeds 10000-10003 --out artifacts/runs/legged_vlm_flow_sem_v1/video_eval.jsonl --video-dir artifacts/video --video-n 2
scripts/peer_run.sh --cpu 2 --mem 8G --label legged_vlm_oracle --max-seconds 7200 -- PY -m rrp.evaluation.legged_latent_eval --flow artifacts/runs/legged_vlm_flow_sem_v1/policy.pt --oracle --bodies go2,pquad4 --seeds 10000-10009 --out artifacts/runs/legged_vlm_flow_sem_v1/oracle.jsonl
# 4. anymal_c data (then a v2 representation over v1 + t1/g1/anymal_c: new configs needed)
scripts/peer_run.sh --cpu 4 --mem 8G --label legged_vlm_collect_h --max-seconds 10800 --detach -- env PY=/dev/shm/rrp-brandonin/venv/bin/python SIGMAS=0,0.05,0.1,0.15 bash scripts/legged_latent_collect.sh /dev/shm/rrp-brandonin/repo/artifacts/datasets/legged_latent_v2 4 "anymal_c:auto" 0 599 100
# 5. system II (psi0) grounding, then closed loop with the trained flow (GPU)
scripts/peer_run.sh --gpu --gpu-mem 16G --cpu 3 --mem 24G --label legged_vlm_s2 --max-seconds 7200 -- env PYTHONPATH=src:/home/brandonin/rrp-peer-data/vlm_pkgs HF_HOME=/home/brandonin/rrp-peer-data/cache/hf HF_HUB_OFFLINE=1 PY -m rrp.evaluation.system2_eval ground --small --body go2 --out artifacts/runs/legged_vlm_system2_smoke
#    (drop --small for the full 160+80 seed run; then `closed_loop --flow artifacts/runs/legged_vlm_flow_sem_v1/policy.pt`)
```
Known caveats before resuming: (1) add periodic checkpoints/resume to `train_rep` / `train_flow` (they save only at
the end); (2) the tracker actors for anymal_c/g1/h1/t1 on the peer came from the host archive (not in git); after a
peer reboot copy them again from `~/.archive/relational-robot-policy-2026-09-25-original/artifacts/trackers/`;
(3) `scripts/peer_sync.sh` previously deleted a worktree's `artifacts` symlink on push (exclude `artifacts/` does not
match a symlink); fixed here by anchoring the excludes (`/artifacts`, `/.cache`). Use absolute output paths anyway.
