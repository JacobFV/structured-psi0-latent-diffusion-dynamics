# track legged_vlm — legged/humanoid breadth and VLM system II on the latent-packet path

## RESEARCH RESTART (2026-09-25 22:15 →; legged agent; supersedes the wind-down state below for the legged part)
Question: does the corrected architecture (system i → packet z[4 knots, legs+body(+arms), 32] → system 0 at 50 Hz)
work on legged/humanoid bodies, and do the packet semantics (per-leg contact per knot, goal waypoint, base
displacement, subtask, fall) causally control behaviour? Arm lessons applied up front (B-1, D-050, D-052..D-056).
Compute: host GPU (leases below), host CPU for closed-loop sims. Data copied to host `~/work/rrp-data/datasets/legged_latent_v{1,2}` (lease 1790399700_94cca7, --disk 2G).

### plan / state
| step | state | evidence |
|---|---|---|
| 0. feature parity (train rows vs deployed featurizers, 3 re-simulated go2 episodes incl. DART) | verified | `artifacts/runs/legged_parity_go2/parity.json`: re-sim == stored (max abs 0); system-0 inputs q/qd/imu/touch/osc + morphology identical (0.0); system-i ctx identical at step-start ticks (0.0; mid-step ticks reuse the step-start ctx in training, and deployment only reads ctx at step-aligned replans). No teacher command or previous action enters any learned input (the tracker's last_a stays inside the frozen tracker). |
| 1. BC positive control (flow, same public inputs, 40-tick chunk, replan 5 ticks, NFE 8), go2 | **verified: competent** | 20k steps (lease 1790400275_8dce13), held-out first-tick MSE 0.024 vs hold-still 1.11 (`artifacts/runs/legged_bc_go2_v1/result.json`). Closed loop, go2 dev seeds 10000-10029: **BC 30/30** [0.886, 1.0] (`artifacts/runs/legged_ladder/go2/bc_bc20k.jsonl`, peer lease 1790403263_cf4e40); scripted_teacher 30/30 (`teacher_bc5k.jsonl`, same seeds); BC@5k 26/30 (drift_a 2, drift_b 2; `bc_bc5k.jsonl`). The data, teacher, tracker and eval support a learned controller on go2. |
| 2. Stage A go2 sem + capacity-matched nosem, qd dropout 0.5, resumable | completed | 12k steps (lease 1790400276_a2a139). Held-out teacher episodes, realization MSE (action units; zero-action 1.41): sem 0.019, nosem 0.0125. Joint probe (sem): contact 0.76 (swing 0.42), goal err 0.017, disp xy 0.034 / yaw 0.018, subtask 1.0, fall 1.0. KL sem 3.7 vs nosem 0.58 (nosem z carries much less). Post-hoc probes: running. `artifacts/runs/legged_rep_{sem,nosem}_go2_v2/result.json` |
| 3. offline gate on BC-visited states (R vs hold-still, qd-zero, z-zero/shuffle step gain) | implementing | `src/rrp/learning/legged_dagger.py gate` |
| 4. ladder on matched dev seeds 10000-10029: R0 teacher / BC / R1 stateless oracle E(BC chunk) / R2 generated; failure stages fell/stall/drift_a/drift_b/halt | implementing | `src/rrp/evaluation/legged_latent_eval.py` (--bc, --oracle-bc, --rep, --realizer, --zero-qd) |
| 5. BC-expert DAgger refit of system 0 | implementing | `legged_dagger.py collect/refit` |
| 6. flows sem/nosem, R2, packet edits (mirror goal, halt, turn, per-leg contact) with irrelevant-edit controls | planned | |
| 7. hexapod6, then t1, g1 | planned | |


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
