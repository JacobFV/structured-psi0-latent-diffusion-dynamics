# track legged_vlm — legged/humanoid breadth and VLM system II on the latent-packet path

## RESEARCH RESTART (2026-09-25 22:15 →; legged agent; supersedes the wind-down state below for the legged part)
Question: does the corrected architecture (system i → packet z[4 knots, legs+body(+arms), 32] → system 0 at 50 Hz)
work on legged/humanoid bodies, and do the packet semantics (per-leg contact per knot, goal waypoint, base
displacement, subtask, fall) causally control behaviour? Arm lessons applied up front (B-1, D-050, D-052..D-056).
Compute: host GPU (leases below), host CPU for closed-loop sims. Data copied to host `~/work/rrp-data/datasets/legged_latent_v{1,2}` (lease 1790399700_94cca7, --disk 2G).

### LEGGED RESEARCH RESULT (for the demo agent; updated 2026-09-26 00:3x; every number from the raw files named)
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
