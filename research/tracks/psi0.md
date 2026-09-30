# track psi0: the Ψ₀ line (W10) inside rrp

Owner: psi0mig agent (D-140 migration). Branch `track/psi0mig`, worktree `~/work/rrp-wt/psi0mig`. Before D-140 this work
lived in the separate repo psi1z (github.com/JacobFV/psi1z, last commit 6f5e2b3, archived); its decisions are the
P-appendix of `research/decisions.md`, its notes are folded verbatim at the end of this file. The early W10 scratch branch
`track/psi0` (before psi1z existed) was merged long ago and holds nothing of its own.

Question (D-098): does adding our structure (morphology-relation tokens, object entity token, contact/binding packet
supervision, packet z[5, 6, 64] realized by a system 0) improve a pretrained humanoid VLA? Ψ₀ direct vs Ψ₀ + structure,
fine-tuned on the same data with the same compute, in Ψ₀'s SIMPLE benchmark.

## state at the D-140 wind-down (2026-09-29)

- Step 0 (feasibility): SIMPLE on Isaac Sim 5.1 / aarch64 with import-time compat hooks and path-traced rendering (P-002, P-007).
- Step 1 (reproduction of released checkpoints, L0): 3/6 reproduce: TabletopGraspMP 10/10, BendPickMP 10/10,
  HandoverTeleop 7/10; XMovePickTeleop 0/10, LocomotionPickBetweenTablesTeleop 0/5, XMoveBendPickTeleop 3/6 do not
  (P-010, P-012, P-015, P-016; table below).
- **Step 2, TabletopGraspMP L0, 10 eval configs x 2 repeats, same seeds/render (pt4_iso55) for every arm. Completed
  2026-09-28 on the peer but never written up in psi1z; recorded here from the raw outputs (no rerun):**

  | arm | source | successes | Wilson 95% | median episode | raw (peer) |
  |---|---|---|---|---|---|
  | released Ψ₀ ckpt_40000 (upstream server, RTC as released) | `Ψ₀:g1wholebodytabletopgrasp-v0…2603181503/ckpt_40000` (upstream weights) | 20/20 | [0.84, 1.00] | 105 s | `~/work/ext/runs/psi1z/cl/step2_tabletop_released/` |
  | Ψ₀ direct (our matched fine-tune, 8000 x 32, no RTC) | `learned:~/work/ext/runs/psi1z/train/tabletop_direct_s0/final.pt` | 19/20 | [0.76, 0.99] | 53 s | `…/cl/step2_tabletop_direct/` |
  | Ψ₀ + structure (stage A v2e + structured head, no RTC) | `learned:~/work/ext/runs/psi1z/train/tabletop_structured_v2e/final.pt` | **0/20** | [0.00, 0.16] | 227 s | `…/cl/step2_tabletop_structured/` |

  Direct − structured: +0.95, Newcombe 95% [0.70, 0.99]; paired by config/repeat, McNemar exact p = 3.8e-6 (19 vs 0
  discordant). Released vs direct: McNemar p = 1.0. The structured arm fails every episode by timeout (800 steps);
  max task reward 0.26–0.44; both hands touch the target in most episodes (first-contact steps in
  `simple_eval/episodes.jsonl`). Not diagnosed (wind-down). Training: direct 5573 GPU-s, structured 8282 GPU-s (plus
  stage A 1157 GPU-s, reported separately); held-out flow losses are over different targets and not comparable.
  Diagnostic probes on E(demonstrated chunk) (`train/probes_E_tabletop_v2e.json`), probe vs metadata-only control:
  hand-target distance MAE 1.2 vs 2.5 cm, contact 78.0 vs 65.5%, lift 93.9 vs 93.2%, target position 2.0 vs 2.5 cm,
  active hand 67.4 vs 73.1% (control better: binding is not decodable above the base rate in this task's packet).
  Not produced: probes on system-i generated z, the held-out open-loop comparison for this task, the P-018 packet edits
  (moot while the structured arm does not succeed).
- Step 2 on BendPickMP / HandoverTeleop: not started (Handover stage A and structured were trained for the older v1
  stage A before P-009; XMovePick arms deleted, P-011).

Next steps after the migration (not scheduled; the owner sets directions after D-140): diagnose the structured arm's
0/20 before any edit test (offline first: open-loop held-out L1 per action group on the cached features vs direct, the
oracle route R(E(a)); then 2–3 logged episodes, as a vibe-check, not a rerun of the table).

## migration map (psi1z 6f5e2b3 → rrp)

Upstream Ψ₀ (`~/work/ext/psi0` @ 4f3720d) and SIMPLE (`third_party/SIMPLE` @ 803db7e) stay unmodified clones outside the
repo; weights, datasets, venvs and runs stay under `~/work/ext`. psi1z never needed a source patch to upstream (its
`third_party_patches/psi0/` was an empty, untracked directory): every fix is an import-time hook, now in
`rrp/envs/simple/compat.py`.

| psi1z | rrp | notes |
|---|---|---|
| `body_g1.py` | `rrp/bodies/g1_simple.py` | 36-d command layout, dim/assembly/relation tables, `SIMPLE_QPOS_INDEX`; body key `g1_simple` |
| `simple_compat.py`, `_simple_autocompat.py`, `simple_run.py` | `rrp/envs/simple/compat.py` | one module: hooks, render profiles, task-uid fix; the `.pth` line and `python -m rrp.envs.simple.compat <module> …` runner live in it |
| `eval_loop.py`, `closed_loop.py`, the contact/palm logging of `replay_labels.py` | `rrp/envs/simple/` (`SimpleEnv` + `worker.py`) | the SIMPLE process runs env + the UNMODIFIED upstream agent (`psi0` MP/AMO or `psi0_decoupled_wbc`) with an injected action client; the harness drives reset/step over a 127.0.0.1 RPC; per-step palms/pelvis/objects/contacts/contact points are `env.truth()` (labels only) |
| `serve_psi0.py`, `serve_ours.py` | `rrp/policies/psi0/__init__.py` (`psi0_direct`, `psi0_structured`, `psi0_replay`) | in-process: the upstream `Server` object supplies image transforms, state/action normalization and RTC; our heads replace `Psi0Model.from_pretrained` exactly as `serve_ours` did; no HTTP server, no port |
| `structured.py`, `system_i.py` | `rrp/policies/psi0/nets.py` | E / R (system 0) / P, StageA, DirectHead, StructuredHead, ContextTokens; byte-identical math (golden test) |
| `serve_ours --edit probe:*/random:*` | harness packet hook (`psi0_probe_edit`) | `--edit entity:<name>` = policy option `entity_override` (input channel, not a packet edit) |
| `features.py`, `data.py`, `compat_psi.py` | `rrp/policies/psi0/data.py` | frozen-VLM feature cache, memmap, `CachedDataset`, replay labels, lerobot PyAV patch |
| `train.py`, `fit_probes.py`, `openloop_cached.py` | `rrp/policies/psi0/train.py` | `rrp train psi0 --arm stageA|direct|structured`, `… probes`, `… heldout` |
| `replay_labels.py` | policy `psi0_replay` (recorded rows, source `replay:<task>`) on `simple` (`split="train"`, `render=False`, `sim_mode="mujoco"`) + `LabelRecorder` hook in `rrp/policies/psi0/data.py` (a Hook; imports nothing from harness) | one contact/palm logger (the worker's truth) instead of two; the P-005 diagnostic options (`--vx-override`, `--step-delay`, `--save-frames`) are dropped |
| `prov.py` | deleted | `rrp.core.provenance` and `rrp.harness.eval.statistics` directly; upstream revisions in `SimpleEnv.spec.provenance` |
| `openloop.py` | deleted | port of the upstream notebook for the released model (done, notes below); `load_launch_config` moves to `policies/psi0` |
| `render_calib.py`, `render_sensitivity.py`, `isaac_smoke.py`, `debug_isaac_env.py`, `diag/{cl_audit,factor_test,render_walk,rtc_openloop,vx_oracle}.py` | deleted | finished diagnostics of P-002/P-005/P-007/P-009/P-012; `git show 6f5e2b3:src/psi1z/<name>` in psi1z |
| `tests/test_core_invariants.py`, `tests/test_roadmap_24_26.py` | `tests/unit/test_psi0.py` | + golden digests of the nets; skips cleanly without torch |
| `scripts/install_{simple_env,simple_deps,psi_env}.sh`, `upgrade_torch_psi.sh`, `build_cyclonedds.sh`, `download_*.sh`, `psienv.sh` | `ops/bin/psi0_ext.sh` | one script: `simple-env`, `psi-env`, `cyclonedds`, `fetch-base`, `fetch-ckpt RUN…`, `fetch-data TASK…` |
| other scripts (`peer_queue`, `queue_host`, `stageA_v2*`, `step2_*`, `cl_parallel`, `*_retry`, `resume_after_lease`, `sync_to_peer`, `server_loadtest`, `prepush_check`, `factor_tests`, `features_task`, `rrp_ops`) | deleted | finished queues/one-offs; commands for reruns are `rrp train psi0 …` / `rrp eval …` |
| SIMPLE tasks | `rrp/tasks`: `simple/<Task>` TaskSpecs (6 tasks) | judge = SIMPLE `_success` from `env.truth()` (success_privileged; there is no public success estimator); the task table (released run, published L0/L1/L2, step-1 status) moves from `envs/simple.TASKS` into the task registry at S3 (tasks sit below envs) |
| viz exporter `rrp/viz/export/psi0.py` | reads the P-appendix and this file | no psi1z checkout or crosswalk table any more |
| pyproject | extra `rrp[psi0]` | psi-side Python deps; Isaac Sim + SIMPLE stay in their own venv (`ops/bin/psi0_ext.sh simple-env`), which imports only `rrp.envs.simple.{compat,worker}` |

Runtime layout: the harness and the Ψ₀ policies run in the psi venv (`~/work/ext/venvs/psi`, Python 3.11, torch
2.14+cu130, `rrp[psi0]` editable); `SimpleEnv` spawns `python -m rrp.envs.simple.worker` in the SIMPLE venv
(`~/work/ext/venvs/simple`, Isaac Sim 5.1, torch 2.7) with the aarch64 preloads, connected over
`multiprocessing.connection` on 127.0.0.1 with a random auth key.

Recorded numbers in this file and the P-appendix come from the psi1z code path (upstream HTTP server + SIMPLE agent in
one process). The rrp path uses the same upstream agent, transforms and weights but is verified by unit/golden tests and
a vibe-check only (D-140: no reproduction runs).

## migration status (2026-09-29)
Merged on main after S3 (see git log "psi0mig"). Registered: env `simple` (`rrp.envs.simple:make_env`), policies
`psi0_direct` / `psi0_structured` (+ `make_replay`, not in the registry table), tasks `simple/<Task>` (6). Verified by unit
tests only (D-140): nets byte-identical to psi1z (goldens), worker RPC and upstream-agent queue semantics against a test
double, a full `harness.rollout` of `psi0_replay` x `simple` (fake worker) x `simple/<Task>` with the `LabelRecorder` hook,
negotiation both ways. Not yet run against real Isaac/SIMPLE or real checkpoints from the rrp path.
Open: `psi0_replay` in the POLICIES table (policies/base.py, S4
owner), structured-arm diagnosis (D-141), psi1z archival (owner approval).

## D-141 offline diagnosis of the structured arm's 0/20 (2026-09-29; no closed-loop runs)
Raw: `artifacts/runs/psi0mig_diag/*.json` (the exact scripts are stored next to them as provenance). All on recorded data or
cached held-out features (TabletopGraspMP val episodes 10, 16, 26, 36, 90, 91, 93, 94; 252 frames); peer leases, ≤ 7.6 GB.
1. Recorded closed-loop queries (step2_tabletop_*/ep*_r0.npz): the first query has IDENTICAL inputs across arms (same scene,
   60 stand steps). There the structured arm commands waist pitch ≈ 0.09–0.12 rad in all 10 configs (direct ≈ 0.015; demo
   mean −0.001, p95 0.018) and keeps ≈ 0.09 for the whole episode; right shoulder pitch overshoots (−1.07 vs direct −0.72);
   no right-hand contact.
2. Offline, the structured route is ACCURATE: held-out chunk L1 (rows 0–23, rad) arm 0.0028 / hand 0.0042 with system-i
   generated z, vs oracle R(E(a)) 0.0025 / 0.0035 and direct 0.043 / 0.045 (fp32 = bf16). But the generated z is unrelated
   to its target (normalized MSE 1.04 per assembly, chance level).
3. System 0 barely reads the packet: R with z = dataset mean, a random z, or another frame's E(a) loses only ~10% (arm
   0.0158 vs 0.0144 normalized); another frame's STATE destroys it (0.43). R is 3.4x better than a ridge state→chunk
   baseline: it is a proprioception-only policy (packet bypass; the P-020 locality test did not measure this).
4. The state it relies on is out of distribution in closed loop: in all 12,226 training frames the last-commanded torso
   state is constant (pitch −0.15 rad, roll/yaw 0, height 0.75), so the normalizer maps torso pitch to −1 always; the
   upstream MP agent starts from [0, 0, 0, 0.75] and feeds back its own commands, which normalize to +1 (clipped).
5. Confirmation: on the logged first-query states, R's waist-pitch command is 0.091 with that one dim at its closed-loop
   value and −0.001 at the training value; the logged structured policy commanded 0.095. The direct arm (same state)
   commanded 0.005: the Ψ₀ transformer tolerates the dim, the structured system 0 does not.
Conclusion: an integration/design bug, not evidence about structure: (a) system 0 takes its plan from state instead of the
packet, and (b) the closed-loop torso-command state is a value never seen in training (same class as P-005's XMovePick
torso-state finding). Proposed (not run; needs a lead decision):
- vibe-check without retraining (2–3 episodes, peer): feed every state dim that is CONSTANT in the training data at its
  training value (a rule that uses training statistics only, applied to both arms), structured arm on TabletopGraspMP.
- real fix (retraining, budgeted): mask training-constant state dims in DimEncoder/Realizer and force packet use in
  stage A (state noise/dropout for R, or restrict R's state to its own assembly's proprio), then check with the item-3
  test (R(z_mean) must be clearly worse than R(E(a))) before any closed loop.

## resume (P4c, D-145: recipes replace the psi1z scripts)
The paused work is "fix the structured arm, then rerun step 2". Every command is a recipe now:
`recipes/templates/psi0_step2.yaml`, instances `recipes/psi0/psi0_tabletop_step2.yaml` (public data; released 20/20,
direct 19/20, structured 0/20) and `psi0_bendpick_step2.yaml` (not started); family `psi0`, stages in
`src/rrp/harness/pipelines/psi0.py` (each wraps `rrp data psi0-features`, `rrp train psi0 [probes|heldout]`, `rrp eval
--env simple`). Peer, GPU lease limit 1, psi venv as peer python (`ops/bin/psi0_ext.sh psi-env`):
`RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/psi0 rrp run-dag recipes/psi0/psi0_tabletop_step2.yaml --dry-run`, then the same
without `--dry-run`; `--point seed=0`. Dry-run nodes (P2 order): global `feat`, global `labels` (`collect` with
`options.data: labels` = `rrp data psi0-labels`); at s0 `stage_a`, `probes`, `gate` (`heldout` on stage A only), `direct`,
`structured` (input `gate`), `heldout`, `probes_gen`, `eval_released`, `eval_direct`, `eval_structured`. Labels are recorded
by the `labels` node (`psi0_replay` x `simple` through `harness.rollout` + `LabelRecorder`); `vars.labels_dir` is gone. The
recipe reads the released run + data from `~/work/ext/psi_home` (`ops/bin/psi0_ext.sh fetch-ckpt|fetch-data`). Old checkpoints, features
and closed-loop outputs stay in `~/work/ext/runs/psi1z/{train,features,replay_labels,cl}/` and are still usable as
inputs (`@`-refs accept any `artifacts/<store>/<name>` path; copy or link them there).
The D-141 code fix is in (D-146 P1, architecture 14.5): `rrp train psi0 --arm stageA` fits the constant-input mask on the
feature cache (state dims with std < 1e-4 are zeroed in E, R and the structured head, mask stored in the checkpoint), drops
R's state while training (per dim 0.3, whole 0.1) and adds the permuted-packet hinge (margin 0.05; `--p-state-dim`,
`--p-state-all`, `--w-perm`, `--perm-margin`); checkpoints are written by `save_checkpoint` with `config["stage_a"]` (widths
+ factor list) and `versions["factors"]`, and `load_stage_a` validates them (a pre-P1 stage A is refused: retrain). `--factors
'<json list>'` sets the factor list (stage A: E/R/probe; structured: context tokens). Gate: `rrp train psi0 heldout --feat-dir
.. --val-episodes .. --stage-a <stage_a.pt> --out gate.json` (no `--run-dir` = gate only) reports err(R(z_mean)) - err(R(E(a)))
and writes `<stage-a dir>/packet_gate.json`; `--arm structured` refuses to start unless that file passes for the same
stage-A file (`--gate` overrides the path). So the recipe's `heldout` gate runs BEFORE `structured` (stage wiring: P2).
Only then `eval_structured`.
Stale psi1z watcher loops on the peer (bash `until ... sleep` loops, pids 2873754, 3156994, 3389109 on 2026-09-29) hold no
lease and can be killed.

---

# psi1z research/notes.md (folded verbatim, 2026-09-26..28)


Part of rrp workstream W10. Project-level plan, evidence and rules: https://github.com/JacobFV/structured-psi0-latent-diffusion-dynamics (see docs/related_repos.md, docs/strategy.md). Lead decisions D-xxx map to P-xxx in research/decisions.md.

Owner: psi0 agent. Repo `~/work/psi1z` (LOCAL ONLY, main). Third-party code, weights, data and envs live under
`~/work/ext/` (never in this repo); upstream clones are pristine, local patches are patch files in
`third_party_patches/psi0/`. Jobs run through the rrp broker (`rrp ops run`, from an rrp checkout).

### layout outside the repo (host)
| path | what |
|---|---|
| `~/work/ext/psi0` | upstream Ψ₀ clone @ 4f3720d (Apache-2.0), submodule `third_party/SIMPLE` @ 803db7e (MIT) with nested submodules gear_sonic, decoupled_wbc, openpi-client, AMO, unitree_sdk2_python (https, LFS skipped) |
| `~/work/ext/venvs/simple` | py3.11 env: torch 2.7.0+cu128 (aarch64), isaacsim 5.1.0 (`ops/bin/psi0_ext.sh simple-env`) |
| `~/work/ext/venvs/psi` | py3.11 env: Ψ₀ deps; upstream `src/` on sys.path via `.pth` (`ops/bin/psi0_ext.sh psi-env`) |
| `~/work/ext/psi_home` | PSI_HOME: `cache/checkpoints/psi0/...`, `data/simple/<task>`, `data/simple-eval/<task>/dr-level-{0,1,2}` |

### step 0 findings (feasibility)
- Isaac Sim: SIMPLE pins `isaacsim[all,extscache]==4.5.0`, python 3.10. pypi.nvidia.com has NO aarch64 wheel for 4.5
  (only x86_64); the first aarch64 wheels are isaacsim 5.1.0 (cp311), then 6.x (cp312). So the documented SIMPLE stack
  cannot be installed on the GB10 as pinned; we try 5.1.0 with a py3.11 env and a compatibility patch for the removed
  `omni.isaac.*` namespaces if needed.
- SIMPLE sim modes: `mujoco_isaac` (MuJoCo physics, Isaac rendering; the benchmark setting), `mujoco` (MuJoCo physics and
  MuJoCo offscreen cameras; `MujocoSimulator.render`). The latter is the fallback, with a visual-domain caveat (the
  policies were trained on Isaac-rendered images).
- Lower-body controller in the SIMPLE teleop eval (`eval_decoupled_wbc.py`, agent `psi0_decoupled_wbc`): the GR00T
  decoupled WBC (ONNX Walk/Balance policies in `third_party/decoupled_wbc`), configured through gear_sonic's
  `SimLoopConfig`; Ψ₀ commands upper-body joints (28 arm/hand + 3 waist) + base height + navigate (vx, vy, turn flag,
  target yaw) = 36 dims, chunk 30 at 50 Hz.
- Eval sets: `simple-eval/<task>/dr-level-{0,1,2}` have 10 episodes each (initial scene configs), matching the
  published "x/10" numbers.
- Published Ψ₀ SIMPLE numbers (SIMPLE README, 10 trials per level, L0|L1|L2): XMovePickTeleop 10|10|6, BendPickMP
  10|10|10, HandoverTeleop 7|7|10, LocomotionPickBetweenTablesTeleop 7|5|6, TabletopGraspMP 10|10|8,
  XMoveBendPickTeleop 10|9|9.
- Released SIMPLE-finetuned checkpoints (HF `USC-PSI-Lab/psi-model/psi0/simple-checkpoints`, 6.25 GB each, ckpt_40000,
  frozen VLM, action expert fine-tuned 40k steps at global batch 64–256):
  | benchmark task | checkpoint run | train data id | base |
  |---|---|---|---|
  | XMovePickTeleop | g1wholebodyxmovepick-v0 ...2604022205 | G1WholebodyXMovePick-v0 | public ego200k.he30k + postpre he30k |
  | BendPickMP | g1wholebodybendpick-v0 ...2603151312 | G1WholebodyBendPick-v0 | NOT public (hfm.pre.fast.mixed ckpt30k) |
  | HandoverTeleop | g1wholebodyhandover-v0 ...2604071507 | G1WholebodyHandover-v0 | public |
  | LocomotionPickBetweenTablesTeleop | g1wholebodylocomotionpickbetweentablesteleop-v0 ...2604081126 | same | public |
  | TabletopGraspMP | g1wholebodytabletopgrasp-v0 ...2603181503 | G1WholebodyTabletopGrasp-v0 | public |
  | XMoveBendPickTeleop | g1wholebodyxmovebendpickteleop-v0 ...2604100422 | same | public |
- Sizes: base VLM 4.26 GB, action header 1.99 GB, each fine-tuned ckpt 6.25 GB (all 6: 37.5 GB), SIMPLE train data
  34–203 MB per task, eval sets <20 MB total, SIMPLE assets (`USC-PSI-Lab/SIMPLE`) assets.zip 2.8 GB.

- Host memory (2026-09-26 23:20–23:45): the host swap is full (16 GB, external stopped `tensorcode` processes hold
  ~60 GB RSS). Any of our jobs above ~10 GB resident drives system memory PSI (full avg10) past the watchdog's shed
  threshold (25, 3 samples) within 60–90 s although MemAvailable reads ~30–40 GB; a plain 6.3 GB file read does not
  (1.3 GB/s, PSI 0). Shed: the Isaac Sim install (twice), the Ψ₀ open-loop smoke (4 attempts, trace
  `~/work/ext/runs/psi1z/psi_trace3.txt`). The brief allows the peer when the host cannot run a job: the stack is
  mirrored to the peer with `scripts/sync_to_peer.sh` (same user/home paths, so venvs stay valid) and jobs run through
  `ops/bin/peer_run.sh` with `RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/psi0` (rrp code only; psi1z lives in ~/work/psi1z
  on the peer).

- Peer stack (2026-09-27): Isaac Sim 5.1.0 installed (17 GB venv). Headless render of a trivial scene verified
  (`~/work/ext/runs/psi1z/isaac_smoke.png`, RTX active on the GB10, 60 steps 2.0 s). Required fixes, all outside the
  upstream tree (psi1z.simple_compat / closed_loop env):
  1. `LD_PRELOAD=libgomp.so.1` (Isaac 5.1 aarch64 startup check);
  2. `LD_PRELOAD` of Isaac's `kit/libcarb.so` (else "cannot allocate memory in static TLS block" when torch loads first);
  3. torch 2.7.0+cu128 cannot JIT for sm_121 (nvrtc "invalid value for --gpu-architecture"): psi env upgraded to
     torch 2.14.0+cu130 / torchvision 0.29 (`ops/bin/psi0_ext.sh psi-env`); torchvision 0.29 lacks `VideoReader`, so
     lerobot's pyav path is replaced by an exact-timestamp PyAV decoder (psi1z.compat_psi);
  4. cuRobo (motion planning / IK only) and envlogger (recording only) are stubbed: any use raises;
  5. cyclonedds has no aarch64 wheel: built from source into ~/work/ext/cyclonedds (`ops/bin/psi0_ext.sh cyclonedds`);
  6. Isaac 5.1's `add_reference_to_stage` drops the reference when the metrics assembler reports divergent units
     (kit command path): we author it directly when missing (post-import patch of SIMPLE's `isaacsim_stage`);
  7. Isaac 5.1 still ships the deprecated `omni.isaac.*` extensions, so no module aliasing is needed (opt-in only).
  Remaining visual difference: the RTX NRD denoiser shader fails to compile on this build
  ("rtx.denoising.plugin Failed to compile compute shader: rtx/nrd/PackForNRD.cs.hlsl") and the first rendered frame
  is much darker than the released data (`scratchpad` comparison); under investigation — closed-loop numbers from this
  renderer carry a rendering-domain caveat versus the published x86 Isaac 4.5 numbers.
- Open-loop (Ψ₀ released XMovePick ckpt_40000, 10 train episodes, stride 4, 563 frames, NFE 10;
  `~/work/ext/runs/psi1z/openloop_xmovepick_ep10_s4.json`, peer): L1 hand 0.019 rad, arm 0.054 rad, waist rp 0.015,
  waist yaw 0.021, vx 0.016 m/s, turn flag 0.016; inference 0.70 s median per chunk. Oracle-scale reference "hold the
  current recorded command" (uses the recorded row 0): hand 0.035, arm 0.030, vx 0.073. The model beats it on hands and
  locomotion, not on arm joints (the reference gets the current command for free).
- Replay labels (`psi1z.replay_labels`, MuJoCo-only, `privileged:sim_replay`, labels only): recorded rows replayed
  through the same WBC reproduce the recorded arm joints to ~1e-3 rad (MAE) and lift the target 5–6 cm (the teleop
  recordings stop right after the lift begins, so the 7 cm success threshold is not reached in replay).
- MuJoCo-only closed-loop fallback works end to end (1 episode, 146 s, 0/1; MuJoCo renders an untextured scene
  — strong visual shift for a policy trained on Isaac images).

- Rendering calibration (`psi1z.render_calib`, `~/work/ext/runs/psi1z/render_calib*_xmp{0,5}.json|png`): vs the
  released eval frame 0 of the same scene config, RTX real-time renders mean brightness 8 (reference 64; MSE 4100 on
  160x90 thumbnails). Real-time toggles (NRD off, sampled lighting, auto-exposure, indirect diffuse) reach at most
  26 / MSE 2100. Path tracing + OptiX denoiser renders the room correctly but brighter (mean 89, MSE ~900); with film
  ISO 65 (chosen from {100, 65, 50} on 2 L0 eval frames) MSE is 204 / 103 and mean 67 / 62 vs 64 / 65.
  DEFAULT PROFILE for all our Isaac evals: `pt4_iso65` (psi1z.simple_compat.RENDER_PROFILES). This is a deviation from
  the upstream renderer and is reported with every closed-loop number. Also: Isaac renders only once per 50 Hz control
  step instead of once per physics substep (the observed frame is identical; 4x less rendering).
- Matched fine-tune, XMovePickTeleop (91 train / 8 held-out episodes, same split for both arms, cached frozen VLM
  features `~/work/ext/runs/psi1z/features/G1WholebodyXMovePickTeleop-v0`, no image augmentation, 8000 steps x batch
  32, AdamW 1e-4 cosine, warmup 300, only transformer blocks from the post-trained header):
  - direct (`~/work/ext/runs/psi1z/train/xmovepick_direct_s0`): 3370 GPU-s (host), train flow 0.013, held-out 0.148.
  - stage A (`.../xmovepick_stageA_s0`, 15k steps x 256, 1229 GPU-s): held-out rec MSE 0.020 (normalized), probes on
    E(z): hand-target distance MAE 2.1 cm, contact acc 97.7% (positive recall 96.5%), lift acc 99.7%, target position
    error 2.8 cm, base displacement error 0.9 cm; active hand is ALWAYS the right hand in this task's data (binding
    label uninformative here). No metadata-only control probe yet.
  - structured (`.../xmovepick_structured_s0`, 3658 GPU-s incl. one shed+resume at step 4000): held-out flow 0.41.
  - Held-out open-loop, cached features, NFE 10, 8 episodes (`.../openloop_heldout_xmovepick.json`), L1 hand / arm /
    waist rp / waist yaw / vx: released Ψ₀ header 0.013 / 0.055 / 0.015 / 0.019 / 0.014 (NOT held out for it: it was
    trained on all of its data); direct 0.054 / 0.053 / 0.017 / 0.035 / 0.036; structured 0.045 / 0.026 / 0.011 /
    0.020 / 0.136. The structured arm's base velocity is much worse (diagnosis pending: flow vs system 0).
- Gate-0 closed loop (Ψ₀ released ckpt_40000, XMovePickTeleop L0, render pt4_iso65, upstream agent + RTC server,
  `~/work/ext/runs/psi1z/cl/psi0rel_xmovepick_L0`): running, 10 configs x 2 repeats; ~12-13 min per episode on the
  shared peer GPU. NOTE the eval env's instruction comes from the eval config's language DR ("bend to pick up the
  cracker box") while the training data says "move forward to pick up the cracker box" — same as upstream eval.
  TimeLimit(800) counts the ~271 stabilization steps, so the policy acts ~530 steps (upstream identical).

- GATE-0 CLOSED LOOP, first result (Ψ₀ released ckpt_40000, XMovePickTeleop L0, eval-config instruction
  "bend to pick up the cracker box" as upstream): 0/10 (10 configs x 1; Wilson 95% [0, 0.28]); stopped before the
  second repeat to run diagnostics (lease 1790500519_536c77, `~/work/ext/runs/psi1z/cl/psi0rel_xmovepick_L0/part0/
  simple_eval/episodes.jsonl`, videos `.../video/ep*_r0/head_stereo_left.mp4`). Median episode 12 min (709 s policy +
  stabilization) on the shared peer GPU. Published: 10/10.
  Failure mode: the robot walks too little (ep6: pelvis +0.11 m in 10.6 s; the teleop demos walk ~0.26 m in ~4.5 s),
  so the hand stops ~0.29 m short; 3/10 episodes touch the box (max lift 1.8 cm).
  Diagnostics:
  * not the WBC timing: replaying recorded commands with +1.2 s wall-clock per step moves the pelvis 0.265 / 0.253 m
    vs 0.263 / 0.253 m in real time (`~/work/ext/runs/psi1z/replay_timing_{0,1.2}`);
  * not the first-frame rendering: released Ψ₀'s first predicted chunk on OUR render vs the upstream render of the same
    scene differs by 0.0004–0.007 (vx m/s) / 0.002–0.006 (arm rad), about the flow-sampling noise floor
    (`~/work/ext/runs/psi1z/render_sensitivity_xmp.json`);
  * candidate: instruction mismatch. The eval config's language DR gives "bend to pick up ..." (a different SIMPLE
    task's wording) while all 99 training episodes say "move forward to pick up the cracker box". Running 10 episodes
    with the training instruction now (`.../cl/psi0rel_xmovepick_L0_traininstr`, lease 1790512148_4b3525).
  * also noted: the upstream agent sends a frozen torso command [0, 0, 0, 0.74] as state; training states carry the
    real previous torso command (pitch 0.07–0.20 rad).

- Instruction variant (training wording "move forward to pick up the cracker box"): 0/5 (max lift 0.18/0.11/0/0/0.11;
  stopped). Torso-feedback variant (last executed waist rpy + height fed back as state; deviation): running, ep0 fail
  (lift 0.16).
- Diagnostic probes, Handover (`~/work/ext/runs/psi1z/train/probes_E_handover.json`, held-out 8 episodes, probe on
  E(z) vs metadata-only control): contact acc 97.8% vs 60.8%, active hand 91.9% vs 55.0%, hand-target distance MAE
  1.9 vs 5.5 cm, target position 4.2 vs 14.6 cm, lift 99.2% vs 62.8%.
- Handover (100 eps, stride-2 features 22.8k frames): stage A done (held-out rec 0.005, active hand acc 91.8%),
  structured done (3792 GPU-s, held-out flow 0.46), direct resumed after a host shed (queue_host.sh).

### reproduction (P-005), 2026-09-27
- (a) cadence (`psi1z.diag.cl_audit`, `~/work/ext/runs/psi1z/cl_audit_torsofb.json`): 24 rows executed per query, every
  query; 50 Hz both in training data (postprocess: 60-frame head cut, downsample 1) and eval. The closed-loop policy
  asks for about the demo's integrated forward distance (0.32 m vs demo mean 0.33 m, onset row ~58 vs frame 60), but
  as GRADED values (0.1-0.32 m/s) in fragments, where the demos command a binary 0 / 0.5 m/s. Replaying demos through
  the same WBC with their vx replaced (`replay_labels --vx-override`, MuJoCo): closed-loop vx profile -> pelvis 0.09 /
  0.12 / 0.12 m; constant 0.2 m/s same integral -> 0.24 / 0.22 / 0.17; constant 0.5 -> 0.26 / 0.25 / 0.20 (≈ original
  0.26 / 0.25). The WBC walks far less for the graded profile: this explains the stop-short failure. The switch to the
  walk policy is at |cmd| >= 0.05 (not the cause). Wall-clock pacing is not the cause (earlier test).
- Chunked open loop on TRAINING episodes, server-like (24-row execution; `diag.rtc_openloop`,
  `~/work/ext/runs/psi1z/rtc_openloop_xmp.json`, 6 episodes): outputs stay binary in all variants (graded rows 2.6-3.3%
  vs demo 2.9%); RTC (train mode, d=6) cuts the commanded distance to 0.21 m vs demo 0.31 m (no RTC: 0.30); frozen
  torso state / "bend" instruction change little (0.20 / 0.21). => graded vx in closed loop comes from the closed-loop
  input distribution; RTC is an additional ~30% distance loss even in distribution.
- (b) observation: hand/arm state at the first query in-distribution (mean |z| 0.33 / 0.63 vs training frame 0);
  torso-command state [0,0,0,0.74] is far out of distribution (|z| 53; training frame 0 ≈ [-0.008, 0.066, 0.001, 0.74]
  with tiny spread); the torso-feedback variant fixes it from the 2nd query on (mean pitch 0.083 vs training 0.12).
  Image: RGB order correct (first image channel means [84, 66, 56] vs training [81, 69, 63]); 360x640 -> resize/crop
  180x320 by the same server transform as training; camera sensor configs identical in training and eval configs.
- (d) rendering on TRAINING frames: replaying training episodes 0/20 through Isaac (`replay_render_xmp`) gives
  frames geometrically aligned with the training video; ISO 65 is ~10% brighter (mean 76-79 vs 69-71); the table
  material of TRAINING configs is not reproduced by SIMPLE's reset (eval configs are). Reset-time comparisons against
  training frame 0 are invalid (frame 0 is 1.2 s into the recording). Pending: ISO choice from replay frames.
- Peer memory emergency 07:50 shed all jobs (all agents). Peer disk: freed 18 GB (3 re-downloadable released ckpts).
- (c) RELEASED TabletopGraspMP ckpt (AMO / MP eval path, upstream `eval.py` + `psi0` agent, ISO 55): 4/4 successes
  before the 08:28 peer memory emergency killed the run (`~/work/ext/runs/psi1z/cl/psi0rel_tabletop_L0/simple_eval/
  eval_stats.txt`). The stack reproduces on a manipulation task; XMovePick (teleop / decoupled-WBC path) has a
  task/path-specific failure. AMO needed: weights from data/robots/g1 (sha256 = LFS oid) and TorchScript GPU fusers off
  (torch 2.7 cannot nvrtc-compile for sm_121; the trace has CUDA baked in, so CPU is impossible).
- Torso-feedback variant (XMovePick): 0/4 (max lift 0.16 / 0 / 0 / 0.19); per-episode audit: commanded distance
  0.28-0.40 m, rows > 0.25 m/s 4.5-8.7% (demo ~16%), vx max 0.40-0.47, pelvis 0.10-0.16 m. Stopped (no benefit).
- (c) TabletopGraspMP, released ckpt, AMO path, ISO 55: 9/9 successes over configs 0-8 (`cl/psi0rel_tabletop_L0` 0-3
  + `cl/psi0rel_tabletop_L0_ep4-9` 4-8; config 9 running). Published L0: 10/10. The stack reproduces.
- (d) re-run at ISO 55 (upstream settings, full query logging, `cl/psi0rel_xmovepick_L0_iso55`): XMovePick 0/5.
- XMovePick no-RTC (ISO 55): 1 episode, fail, peak vx 0.26, pelvis +0.014 m (stopped for the logged re-run).
- Factor test (`diag.factor_test`, `~/work/ext/runs/psi1z/factor_test_ep{0,1}.json`): re-predicting the SAME logged
  closed-loop inputs without RTC, the model asks to walk (0.34-0.49 m/s at queries 1-5 in ep 1) where the executed,
  RTC-served chunks were exactly 0: RTC 'train' mode (first 6 rows frozen to the previous chunk's tail) is sticky —
  a standing prefix keeps the robot standing. Without RTC the per-query walk decision is unstable (e.g. 1 of 3 samples
  walks). Swapping in training images at matched steps restores the demos' walking at queries 1-3 in ep 0 but not in
  ep 1 (inconclusive, 3 samples).
- Controller check: replayed leg-joint trajectories through the eval's decoupled WBC match the recorded
  `observation.leg_joints` (MAE 0.006-0.011 rad, same amplitude and gait frequency 0.2-0.29 Hz) -> same lower-body
  controller in data and eval (the "G1 SONIC" training label is not a different controller).
- Running: Isaac renders of 5 training episodes at walking-phase steps 24-72 (`replay_render_walk`), then
  `diag.render_walk` = P(walk) from our render vs the training frame at identical pose and state.
- Budget since P-005 (≈06:40): ~5.5 h wall, ~5.6 peer GPU-h (of 24 h / 20 GPU-h).
- STEP 1 (P-010) running: TabletopGraspMP L0 10/10 (Wilson [0.72, 1.0]; published 10/10) = REPRODUCED
  (`cl/psi0rel_tabletop_L0` configs 0-3 + `cl/psi0rel_tabletop_L0_ep4-9` 4-9). Queue (`~/work/ext/runs/psi1z/
  peer_queue1.txt`, scripts/peer_queue.sh keeps <= 2 psi1z peer GPU leases): XMovePick RTC test_time 5 eps ->
  HandoverTeleop L0 10 -> BendPickMP L0 10; LocomotionPickBetweenTables and XMoveBendPick ckpts downloading.
- STEP 2 prep on TabletopGraspMP (public data: 100 episodes, 12.2k frames): MP-path replay labels
  (`replay_labels --mp`, aligned after the MP agent's 60 standing steps: arm MAE 0.0007 rad, hand 0.003, replayed
  demos succeed); host pipeline `scripts/step2_host.sh` (stage A v2e -> direct -> structured -> probes) waits for
  features (queued) and labels (syncing). The tabletop instruction names no object, so the entity-swap edit does not
  apply; edits will be probe-guided target shifts and hand binding, with random-direction controls.
- Structured-arm vx (offline): the oracle route R(E(actions)) has the same vx error (0.127 m/s) -> not a system-i
  problem. vx in data is binary 0 / 0.5 m/s on 17% of frames. v1 packet KL was only ~6 nats with z-noise 0.1; the
  per-loss grad-norm logging was broken (norms computed on unlogged steps; fixed). v2a (w_kl 1e-4, z-noise 0, base_cmd
  packet probe = knot vx/vyaw LABELS): KL 55, probe decodes knot vx (MAE 0.010 normalized) but R(E) vx still 0.123 ->
  hypothesis: the vx information sits in tokens system 0's base dims do not read. v2c (base_cmd probe restricted to
  base/torso tokens): oracle vx 0.124 -> hypothesis REFUTED. v2b (reconstruction weight 4 on vx/vyaw): oracle vx
  0.050 m/s (from 0.127) -> the realizer under-fits the sparse binary vx under uniform per-dim MSE. Per-loss grad norms
  on the shared encoder (v2b, end): rec 0.014, sem 0.14, kl 0.013 (x w_kl) -> D-085 pattern; v2d = v2b + w_sem 0.1
  (shed once by the host disk reserve; host disk freed 31 GB of finished runs' optimizer states and a smoke run):
  oracle vx 0.050 (= v2b). v2e (weight 16): oracle vx 0.015 m/s (direct arm end-to-end: 0.036), held-out rec 0.017,
  arm L1 0.022, probes unchanged -> FIX ADOPTED (stage A v2e). Structured arm retraining on v2e queued on the host
  (not run in closed loop before reproduction, P-005).

### state (2026-09-27 ~07:00 PDT)
- step 0: Isaac Sim 5.1 on aarch64 verified (headless render + full SIMPLE env) with 7 compat fixes; open-loop verified;
  closed-loop reproduction of released Ψ₀ FAILS on this stack (0/10 L0 XMovePickTeleop, published 10/10); diagnostics
  above; not the WBC timing, not the first-frame rendering, not the instruction wording (0/5).
- step 1: remaining 5 released checkpoints downloaded on the peer (`scripts/download_step1.sh`); not evaluated.
  XMoveBendPickTeleop has no public training data (step 2 impossible for it).
- step 2: XMovePick direct/stage A/structured trained; Handover stage A/structured trained, direct running; no
  closed-loop evaluation of our arms yet (peer GPU lease limit 1, occupied by the gate-0 diagnostics).

### prep status (2026-09-29, before S3)
Drafted on `track/psi0mig` as NEW files only (no existing `src/rrp` file touched): `bodies/g1_simple.py`,
`policies/psi0/{__init__,nets}.py`, `envs/simple/{__init__,compat,worker}.py`, `harness/data/psi0.py`,
`harness/train/psi0.py`, `ops/bin/psi0_ext.sh`, extra `psi0` in pyproject, `tests/unit/test_psi0.py` (nets byte-identical
to psi1z; worker plumbing with a test double). After S3: adapt to the real `Env`/`Policy`/`TaskSpec`/`Act`, register
`psi0_direct`/`psi0_structured`/`psi0_replay` and `simple` + `simple/<Task>`, wire `rrp train psi0`, point the viz
exporter at the P-appendix + this note, delete docs/related_repos.md (glossary → architecture.md), README/STATUS/
AGENTS/strategy pointers, D-entry for the step-2 result (labelled: structured route likely has an integration bug).

## resume (historical; superseded by the recipes named at the top of this note)
psi1z queues (`scripts/queue_host.sh`, `cl_parallel.sh`, `peer_retry.sh`) are deleted; their reruns are
`rrp run-dag recipes/psi0/psi0_tabletop_step2.yaml` (idempotent: finished nodes are skipped; `rrp train psi0 --resume`
continues from `last.pt`). Packet edits (`--edit entity:<name> | probe:target_dx:<m> | probe:hand:<0|1> | random:<norm>`)
are the policy option `entity_override` and the harness packet hook `psi0_probe_edit`, not a recipe node yet.
- host disk (other users) fell below the 100 GB reserve twice; freed my XMovePick ckpt copy, Handover v1 arms and cached features (all re-creatable).
- INCIDENT 13:08-13:42: two of my concurrent eval jobs used the same policy-server port 22085 (cl_parallel default).
  The Handover eval's queries failed after the XMovePick-RTC server exited ("Server is not up"); zero Handover queries
  were answered (0 logged), and the XMovePick run's episodes finished before Handover's first query, so no result is
  contaminated. Handover's own server was stuck loading under peer memory pressure. Fixed: closed_loop picks a free
  port (--port 0 default; cl_parallel passes 0). Handover L0 relaunched.

### step 1 per-task table (P-010, final 2026-09-27 18:40; L0, ISO 55 path tracing, released ckpt_40000)
| task | path | ours | Wilson 95% | published L0 | status | step 2 eligible |
|---|---|---|---|---|---|---|
| TabletopGraspMP | AMO / upstream eval.py | 10/10 | [0.72, 1.00] | 10/10 | reproduced | yes (public data) — running |
| BendPickMP | AMO / upstream eval.py | 10/10 | [0.72, 1.00] | 10/10 | reproduced | yes (public data) |
| HandoverTeleop | WBC / psi1z.eval_loop | 7/10 | [0.40, 0.89] | 7/10 | reproduced | yes (public data) |
| XMoveBendPickTeleop | WBC / eval_loop | 3/6 (stopped) | [0.19, 0.81] | 10/10 | not reproduced (best reachable 7/10 excludes 1.0) | no |
| LocomotionPickBetweenTablesTeleop | WBC / eval_loop | 0/5 (stopped, P-015) | [0, 0.43] | 7/10 | not reproduced | no |
| XMovePickTeleop | WBC / eval_loop | 0/10 (+ variants 0/5, 0/5, 0/5, 0/4) | [0, 0.28] | 10/10 | not reproduced (P-012) | no |
Raw: peer `~/work/ext/runs/psi1z/cl/psi0rel_<task>_L0*/` (eval_stats.txt for eval.py, part0/simple_eval/episodes.jsonl
for eval_loop). Fixes needed along the way: AMO weights path + TorchScript fusers off (MP path); task uid alignment
(XMoveBendPick, upstream bug); free port per job (port-collision incident). Reproduction compute since P-005: ≈9-12
peer GPU-h (lease-log estimate).

### state 2026-09-27 ~18:50 (step-1 gate report)
- Step 1 complete (table above). Step 2 on TabletopGraspMP: stage A v2e done; direct arm at step 3899/8000 (checkpointed);
  training resumes on the peer via `scripts/peer_retry.sh 240 ... step2_train_peer.sh` (log
  ~/work/ext/runs/psi1z/step2_train_launch.log) once peer admission reopens (stopped by the watchdog at 18:45).
- Then: `scripts/step2_eval.sh {released,direct,structured} G1WholebodyTabletopGraspMP-v0 tabletop <run> 2` (declare
  --cpu 2 --mem 17G --gpu --gpu-mem 18G, ≤ 2 peer GPU leases), then packet edits (probe:target_dx, probe:hand, random
  controls) on the structured arm.
- Next step-2 tasks with public data: BendPickMP, HandoverTeleop.

### step 2 — TabletopGraspMP (running)
- released Ψ₀ ckpt_40000 (upstream server, RTC as released), psi1z.eval_loop MP mode, 10 eval configs x 2 repeats,
  ISO 55: 20/20, Wilson [0.84, 1.00], median episode 106 s (`cl/step2_tabletop_released/summary.json`).
- direct / structured arms: training on the peer (direct done at 8000 steps; structured next in the same lease);
  their 20-episode evals start after W7's priority job is admitted (P-019).

### 2026-09-30 readiness P2 (audit D20): data driver and SIMPLE plumbing
- `CachedDataset`: the realization tick j is drawn from `default_rng([seed, episode, frame, visit])`, `visit` = fetches of
  that item so far (a shared-memory counter: every DataLoader worker bumps the same one, persistent workers included). One
  epoch is identical for `num_workers` 0 / 2 / 3 (test), later epochs draw fresh ticks, `reset_epochs()` replays from 0. The old
  shared `rng` gave each worker a copy and made the tick depend on the worker count.
- One ext-dir resolver: `rrp.envs.simple.compat.ext_dir()` (`RRP_PSI0_EXT`, default `~/work/ext`, read at call time; stdlib only
  so the SIMPLE-venv hook can use it), with `psi_home()` and `simple_root()` derived from it. `SimpleEnv`, the worker and
  `policies/psi0/data.py` use it; no import-time `EXT` / `SIMPLE_ROOT` constants remain.
- Render profile is an env kwarg: `make_env("simple", ..., render_profile=)` / `--env-kw render_profile=pt4_iso65`, validated
  before any process starts, carried to the worker as `RRP_SIMPLE_RENDER` (still validated there).
- `rrp data psi0-labels --task T --out D --episodes A:B` = `psi0_replay` (recorded rows) on `simple` (`split=train`,
  `render=False`, `sim_mode=mujoco`) through `evaluate()` with the `LabelRecorder` hook; rows carry
  `label_source=privileged:sim_replay` (labels only). The stage is `collect` with `options.data: labels`, source
  `privileged_teacher:sim_replay`.
- Gate (architecture 14.5): `heldout` without a `structured` input is the pre-head gate and needs `heldout.json["gate"] =
  {gap, margin}`; gap < margin (or no gate block) fails the node, and `train_flow` re-reads the same file from its `gate` input
  before it starts (GateFailed, `gate_report.json`). The trainer-side writer of the block belongs to P1 (`train.heldout`).
- Tests: `tests/unit/test_psi0_data.py`; `test_pointer_psi0_stages.py` follows the new recipe shape. Peer smoke not run (no
  training or simulation on the host; the label driver is covered against the fake-worker double).
