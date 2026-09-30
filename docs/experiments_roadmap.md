# experiments roadmap: open questions we intend to test

Owner request (2026-09-27, D-126). Everything here is still UNCERTAIN: a question we are interested in but have not answered.
Each item says what it would settle, what it depends on, a rough peer cost, and its code status. Code status is tracked
separately from experiment status: per D-126, everything that is purely a code problem is implemented now behind ablation
flags, so experiments can launch as soon as compute allows. Related docs: docs/strategy.md (workstreams, allocation),
the appendix below (stated-but-unimplemented items, D-123), research/reports/evidence_matrix.md (what IS established).

Status legend (experiment): running · queued · planned · open (needs design) · done → decision.
Code: ✅ exists · 🔧 implementing (D-126) · ❌ not a code problem / needs research design.

## A. Arm: semantic packet under realistic physics
| # | question | depends on | cost | experiment | code |
|---|---|---|---|---|---|
| 1 | Do the arm semantic results (D-095) survive grasp_v2? Re-eval of existing routes | – | ~free | done → D-127 (yes; semfix 311/480 vs nosem 40/480; frozen sem s1 collapses) | ✅ |
| 2 | Is v6dart data good enough: v6 BC expert ≥ v2 expert (D-121 condition)? | – | ~3 h | done → D-121 addendum (yes: 58/60 panda, 60/60 parm6 under grasp_v2.1) | ✅ |
| 3 | Do semfix vs nosem arm results hold on realistic physics + teacher v2 + v6dart? | 2 | 24–30 h | done → D-134 (semfix +0.23, all cells; nosem competent; binding seed-dependent) | ✅ |
| 4 | Can upstream recipe choices (Stage A β, BC-DAgger schedule, refit lengths) rescue nosem? (D-099 fairness) | 3 | ~1 day | planned | ✅ overlays (dags/overlays/arm_recipe) |
| 5 | Does small-amplitude descent-phase DART restore near-grasp recovery without penetration? (D-121 fallback) | 2 | ~0.5 day | conditional | ✅ `dart_descent_sigma` |
| 6 | Can GRPO (success reward, anchor/forgetting evals) lift the latent route past the scripted ceiling vs BC at equal budget? | 3 | 6–10 h | planned | ✅ grpo stage + anchors (dags/templates/arm_grpo.yaml) |
| 7 | Does overlapping-chunk blending remove chunk-boundary steps without losing success (BC and system 0)? | – | 0.5–1 day | planned | ✅ `chunk_blend` |
| 8 | Does limit-aware IK remove procedural-arm joint-limit contact so the gate can be enforced? | – | ~1 day + data | planned | ✅ `ik_limit_margin` / teacher v2lim |

## B. Cross-body transfer (the untested core claim)
| # | question | depends on | cost | experiment | code |
|---|---|---|---|---|---|
| 9 | Does the latent route transfer to the sealed target bodies (xarm7_pg2/tf3, panda_tf3): zero-shot and with small adaptation budgets? | 3 | 1–2 days | done → D-135 (no on new arm: latent refit ≤38/200 vs BC SFT up to 192/200; new gripper on panda transfers, still ≤ BC) | ✅ DAG template arm_targets_latent |
| 10 | Fair baselines on those bodies: BC seeds 1702/1703, SFT budgets on the same data | – | ~1 day | done → D-135 | ✅ DAG template arm_targets_bc |
| 11 | Legged held-out body transfer (heldout + refit stages) | – | ~1 day | planned | ✅ code (D-126 legged): `dags/templates/legged_v2_heldout.yaml` (validate_tracker-gated; zero-shot `heldout` + equal-budget refit vs BC) |
| 12 | Do anchor-relative packets transfer better than base-frame ones? (W12 H2) | 20 | in W12 | planned | ✅ (W12 A) |

## C. Legged and humanoid
| # | question | depends on | cost | experiment | code |
|---|---|---|---|---|---|
| 13 | (W13, D-138: P1b) Can we train humanoid trackers under sourced limits that pass the gates (t1 lab gate, h1 clock-driven gait, g1 without stomp/overspin), and does the halt effect hold on t1 then? | – | 2–4 days | planned | ✅ code: `--ref-gait clock`, `--yaw-progress-cap`, `--yaw-overshoot`, `--limit-margin[-agg]`; recipes `rrp.harness.train.tracker_recipes` + `dags/d126_tracker_*.yaml` (not run) |
| 14 | Do trackers and latent routes survive actuator dynamics + latency as default (sourced speeds)? | 13 | 1–2 days | planned | ✅ code: `actuator_mode` ideal/v1lat/v2 (default ideal) through training, validation, pipeline (`options.actuator_mode`), eval; estimated speeds flagged |
| 15 | Does a terrain curriculum (and MJX for throughput) move the 8 cm terrain break-point? | – | 2–4 days | planned | ✅ code: `--terrain-curriculum gated`; MJX prototype `envs/warp/model.py` (peer smoke: parity ok, no throughput win yet) |
| 16 | Is the semantic robustness cost (D-112: −4.6 points) general beyond anymal_c (go2, later t1)? | – | ~1 day/body | planned | ✅ |
| 17 | Why does semfix steer 2× more on anymal_c but not go2? Does it replicate? | – | ~1 day | open | ✅ |

## D. Feature-centric coordination (W12)
| # | question | depends on | cost | experiment | code |
|---|---|---|---|---|---|
| 18 | Grasp fix for pegs/bars and dual teacher v3 (smooth, contact-confirmed, limit-aware, phase-gated DART), then data | – | 1–2 days | queued (W7 grasp part) | ✅ teacher v3 `policies/teachers/dual_smooth.py` (default off: `teacher_version: v3`, ablations `teacher_options`), phase-gated DART `noise_phase_gate`, quality record `record_quality`, gate `gates.check_dual_dataset`; peer smoke: support_insert v3 5/5 vs v2 1/5 (research/tracks/w12.md §10). Grasp model: W7 |
| 19 | H1 go/no-go: base-frame vs anchor-relative supervision vs anchor-input system 0 (seed 0) | 18, 3 | 9–24 GPU-h | planned | ✅ |
| 20 | Full W12 matrix (+event-aligned knots, anchor-input control, 3 seeds) with anchor-shift and contact-sequence edits | 19 | 2–6 days | planned | ✅ |
| 21 | Legged stance drift with the new contact metrics on existing W8 checkpoints | – | ~0.5 day | planned | ✅ |
| 22 | Coordination tasks: pivot against a surface, carry a tray level, legged foothold stepping | – | design + data | open | arm tasks ✅ stubs: `src/rrp/tasks/graphs/pivot_against_surface.json`, `src/rrp/tasks/graphs/carry_tray_level.json`, scenes `envs/dual_scenarios.build_pivot/build_carry_tray`, labelled unvalidated teacher stubs `policies/teachers/dual_coord.py` (W12); legged foothold ✅ `envs/mujoco/legged_scenes.py`, `src/rrp/tasks/graphs/foothold_steps.json`; see research/tracks/d126_legged.md) |

## E. Ψ₀ line (psi1z, W10)
| # | question | depends on | cost | experiment | code |
|---|---|---|---|---|---|
| 23 | TabletopGrasp: released vs Ψ₀ direct fine-tune vs Ψ₀ + our structure; hand-binding and goal edits | – | ~1 day | running | ✅ |
| 24 | BendPickMP, HandoverTeleop step 2, with a grasp-region affordance head | 23 | 2–3 days | planned | ✅ head (psi1z P-022, default off; needs re-replay labels) |
| 25 | Can a better render (materials/lighting) reproduce the walking tasks (XMovePick, LocoPick, XMoveBendPick)? | – | open-ended | open | ❌ research |
| 26 | Higher DR levels (1–2) on reproduced tasks | 23 | ~1 day/task | planned | ✅ eval flags (psi1z P-023) |

## F. Representation and deployment credibility
| # | question | depends on | cost | experiment | code |
|---|---|---|---|---|---|
| 27 | Do routes still work with estimated (not truth+noise) base velocity? | – | 1–2 days | planned | ✅ `--base-state-source estimator` (bse-1) |
| 28 | Automated per-group privileged-information audit (ablate input groups, measure causal use) | – | ~1 day | planned | ✅ `rrp.harness.eval.privileged_audit` (paudit-1) |
| 29 | Does an OOD packet detector + fallback catch the packet edits that make t1 fall? | – | 1–2 days | planned | ✅ `--packet-ood`, `rrp.harness.train.packet_ood_fit` (pood-1) |
| 30 | Safety layer (clamp, rate limit, safe stop, fall recovery): cost to success, benefit under perturbation | – | 1–2 days | planned | ✅ `--safety` (safety-1) |
| 31 | VLM system II: language → packet, real evaluation | – | 3+ days | open | ✅ harness only: `rrp.harness.eval.system2` (s2h-1), `--system2` |
| 32 | Minutes-long runs (drift) and latency on current routes | – | ~0.5 day | planned | ✅ `--eval-mode long`, `--measure-latency`; `rrp latent latency --representation` |

Section F code (D-126, 2026-09-28): commands and flags in research/tracks/d126_deploy.md "exact configs". All options
default off; defaults golden-identical to pre-D-126 rows.

## G. New task families
| # | question | depends on | cost | experiment | code |
|---|---|---|---|---|---|
| 33 | First dual-arm learned models (M=2 "across bodies") | 18 | 2–3 days | planned | ✅ stages collect/pack/train_rep/probes/train_flow/flow_ft/refit/eval_r2/heldout/edits (`harness/pipelines/dual.py`), DAG `dags/templates/dual_lineage.yaml` (zero_prev_action true); dual DAgger ❌ (no dual label source) |
| 34 | Loco-manipulation (walk to a table, then pick); humanoid versions M1–M3 in W13 (D-138) | 13 | 3+ days | open | ✅ task + scene (spot_arm) + STUB teacher; needs a spot tracker |
| 35 | Richer arm objects (sizes, masses, friction, shapes) | – | 1–2 days + data | planned | ✅ `object_spec` / `object_variation` |

## H. Humanoid transfer (W13, D-138; plan research/tracks/humanoid.md, sealed split research/splits/humanoid_v1.json)
| # | question | depends on | cost | experiment | code |
|---|---|---|---|---|---|
| 36 | Does a GPU simulator (MuJoCo Warp / MJX) reproduce contact_v2 humanoid rollouts and give ≥ 5× CPU PPO throughput? | – | 3–4 h | planned (P1a) | 🔧 |
| 37 | Does a morphology-conditioned shared tracker trained on the pool + `phum` walk sealed humanoids zero-shot, and how fast does it adapt vs per-body PPO? (existing-controller transfer) | 13, 36 | 1–2 days | planned (P1c) | 🔧 |
| 38 | Can privileged RL experts solve the staged tasks (L1–L3, M1–M3, C1–C2) on ≥ 4 pool humanoids and yield gate-passing datasets? | 13 | 3–4 days | planned (P2) | 🔧 |
| 39 | Does the semantic packet help new-humanoid transfer (sealed S1–S5) vs nosem and BC at matched data/updates (zero-shot, refit, joint adaptation, BC SFT)? | 38 | 3–4 days | planned (P3) | 🔧 |
| 40 | Held-out task compositions (steps+carry, gap+cart) zero-shot | 38 | in P3 | planned | 🔧 |

## I. ComputerWorld pointer (track pointer, D-142; research/tracks/pointer.md "pointer policy", split research/splits/cworld_pointer_v1.json)
| # | question | depends on | cost | experiment | code |
|---|---|---|---|---|---|
| 41 | Can an engineered (scripted) pointer system 0 realize packets well enough to solve all four cw/* tasks from teacher-oracle packets? | – | ~free | done → D-142 (400/400, teacher-identical step counts) | ✅ `policies/pointer.py` |
| 42 | Does a learned pointer latent route (system i flow → learned system 0) solve the cw/* tasks on sealed seeds, vs pointer BC with the same inputs and demos, and vs a learned system i driving the engineered system 0? | 41 | ~4 GPU-h | done → D-142 (sealed_id: BC 0.995, latent semfix 0.930, nosem 0.920, eng 0.463) | ✅ `rrp train pointer` |
| 43 | Does semantic packet supervision (semfix) make the UI probes (target widget, pointer-relative target, button/key phase) decodable and causally usable (probe-guided retargeting) vs nosem? | 42 | ~1 GPU-h | done → D-142 (probes sharper with semfix; semfix edits steer system 0, nosem ≈ random; no success difference) | ✅ `rrp train pointer probe/edit` |
| 44 | Held-out variants (unseen calc pairs, words, names): does typing copy characters from the instruction? | 42 | in 42 | done → D-142 (no: 0/100 unseen words/names for every learned method; unseen calc pairs 45–50/50) | ✅ |

## ordering once current runs finish (lead)
1–3 (running/queued) → 9–10 (core claim) → 23 (queued) → 13 (unblocks humanoid) → 18–19 (W12 go/no-go). The rest fits
around peer capacity, which is about one major experiment per 1–2 days on the single peer GPU.
When an item is answered, record the decision number here and move the result into the evidence matrix.

## appendix: stated-but-unimplemented intentions (backlog, D-123; merged here in D-140)

Audit of 2026-09-27 (read-only; rrp origin/main ed537d2, psi1z 8f7e297), recorded as D-123. It lists every intention stated in
docs, decisions or track notes that is only partly implemented or not implemented in code, verified against the code. Keep it
current: when an item lands, move it to "checked and implemented" with the commit; when an item is dropped, mark it superseded
and cite the decision.

I found 44 intentions that are stated in the docs but only partly built or not built. Checked against rrp origin/main `ed537d2` and psi1z main `8f7e297`. Nothing was modified, and the audit worktree has been removed.

Two things to know first:
- `docs/robot_training_considerations.md` and `STATUS.md` are themselves out of date. Several items marked ❌ there now exist in code; those are listed under "Checked and implemented" at the end.
- The only real TODO or not-implemented markers in code are the skeleton in `harness/pipelines/dual.py` and the `NotImplementedError` stubs in `core/system0.py`. The latter are abstract base methods, not gaps.

### backlog: Physics / sim
| item | source | status | evidence | size | blocks |
|---|---|---|---|---|---|
| Actuator dynamics and latency on by default | W1; D-103(2), D-107(4) | **code done 2026-09-28 (D-126 legged)**: `actuator_mode` switch, default still ideal; procedural-body speeds flagged `estimate` | `bodies/actuator.py` (armature, friction, torque–speed, 0–30 ms) is opt-in, by decision. Max joint speeds are still estimates, and trackers don't hold their gaits under it (h1 no-fall 0.72) | M–L | actuator-realism claim |
| MJX GPU simulation | considerations §4.7, §5.5 | **prototype 2026-09-28 (D-126)**: `envs/warp/model.py`, peer venv ~/work/ext/venvs/mjx; parity ok, not faster yet | no mjx or jax anywhere in `src/` | L | terrain curriculum at scale |
| Terrain curriculum in tracker training | §4.2 | **code done 2026-09-28 (D-126)**: `--terrain-curriculum gated` (bumps_v1; not run) | terrain is eval-only: `envs/mujoco/perturb.py` `terrain_amp_m` heightfield. `harness/train/tracker_training.py` has only a turn curriculum | M | rough-terrain claims (break-point at 8 cm) |
| Link-length, encoder-offset and motor-strength randomization in training | §4.2 | partial | mass, CoM and gains exist only as eval perturbations (`envs/mujoco/perturb.py`). No link-length or joint-offset randomization anywhere | M | — |
| IMU bias/drift, encoder quantization, tick jitter or missed-tick injection | §4.3, §4.4 | missing | none found; `harness/eval/latency.py` only counts deadline misses | S–M | — |

### backlog: Robots / bodies
| item | source | status | evidence | size | blocks |
|---|---|---|---|---|---|
| h1 tracker that can turn in place | D-103(3); contact.md "PARKED" | **recipe ready 2026-09-28 (D-126)**: `h1_clock_scratch` (not run) | `artifacts/trackers/h1/contact_v2` walks but has turn ratio 0.02. The planned phase-clocked gait was never started | L | h1 in W8; W1 gate |
| g1 tracker with sourced limits that doesn't stomp | D-114 addendum | **recipe ready 2026-09-28 (D-126)**: `g1_yawcap_ft` (not run) | g1_src fails the gates (slip 0.38, 4.09 BW); no run after that | M–L | g1 in W8 |
| t1 tracker that passes the lab gate under the waypoint command mix | D-107(2), D-113 | **recipe ready 2026-09-28 (D-126)**: `t1_turn_latency_ft` (not run) | w8d has forward 0.72 and joint margin −0.053. Learned t1 routes fall (BC 0/30) | L | W8 t1 claim |
| Retune the joint-limit-margin reward | D-114 addendum | **exposed 2026-09-28 (D-126)**: `--limit-margin`, `--limit-margin-agg max` (mean dilutes one joint to 1/n); recipes use max | `RewardCfg.limit_margin=-1.0` is in `envs/mujoco/legged_core.py`, but no tracker has been trained with it | S | next tracker run |
| Latent route on the sealed target bodies | §4.8; W9 | code implemented (D-126 arm; experiment pending) | target demos (xarm7_*, panda_tf3) are collected in `dags/armexpert_v6dart.yaml`, and the arm/legged `heldout` stages exist. No DAG node evaluates or adapts the latent route on those bodies | M | central cross-body claim |
| Legged held-out body, DAgger and refit in the v2 DAGs | W8/W9 | **template 2026-09-28 (D-126)**: `dags/templates/legged_v2_heldout.yaml` | `legged_v2_*.yaml` goes collect→…→eval_r2/edits; the `heldout` and `refit` stages are never used | S | legged held-out claim |

### backlog: Data / teachers
| item | source | status | evidence | size | blocks |
|---|---|---|---|---|---|
| Limit-aware IK | D-114(3) | code implemented (D-126 arm; experiment pending) | no margin or barrier term in `bodies/ik.py` or `policies/teachers/arm_smooth.py` (only a comment in `harness/eval/gates.py:232`) | M | enforcing the parm* joint-margin gate |
| Arm object variety (size, mass, friction, shape) | §4.2 | code implemented (D-126 arm; experiment pending) | distractors exist, and a `cube_size` parameter exists but is fixed at the default (`envs/mujoco/scenario.py`). Mass and friction are eval perturbations only; no shape variety | M | — |
| Loco-manipulation task | §4.9; W9 | **task + scene 2026-09-28 (D-126)**: `src/rrp/tasks/graphs/loco_pick.json`, spot_arm scene, STUB teacher | none in rrp (only `src/rrp/tasks/graphs/waypoint_contact.json`). psi1z drops walking-heavy tasks by D-120 | L | W9 |
| Grasp-accept tolerance under DART | armexpert.md backlog | partial | only the 1.6 cm reachability tolerance (`arm_smooth.py:312,408`) | S | — |

### backlog: Models / packet / representation
| item | source | status | evidence | size | blocks |
|---|---|---|---|---|---|
| OOD packet detection with a fallback controller | §4.11 | **code done 2026-09-28 (D-126: `policies/packet_ood.py`, `--packet-ood off\|monitor\|enforce`, fit/score tool; research/tracks/d126_deploy.md); experiment #29 pending** — was partial | `core/system0.py` checks only version and staleness, with a `hold_measured` fallback. No distributional check | M | safe deployment claim |
| Safety layer (clamp, rate limit), safe stop, fall recovery | §4.11 | **code done 2026-09-28 (D-126: `policies/safety.py`, `--safety off\|monitor\|enforce`; research/tracks/d126_deploy.md); experiment #30 pending** — was missing | only `np.clip(±6)` in `policies/bc.py:94`. No `safe_stop` or recovery | M / L | hardware framing |
| State estimation | §4.3 | **code done 2026-09-28 (D-126: `envs/mujoco/state_estimator.py` bse-1 fills `base_vel_estimate`; `--base-state-source estimator`; research/tracks/d126_deploy.md); experiment #27 pending** — was partial | Tracker actors use IMU + encoders only; true base velocity goes only to the critic (`envs/mujoco/legged_core.py` `priv`). System i's speed comes from a "declared localization" that is truth + Gaussian noise (`envs/mujoco/legged.py _sense`). `base_vel_estimate` (`core/observation.py:26`) is never filled | M–L | deployable-observation credibility |
| Privileged-information audit per input group | §4.3 | **code done 2026-09-28 (D-126: `harness/eval/privileged_audit.py` ablation runner + static/schema/tripwire checks; research/tracks/d126_deploy.md); experiment #28 pending** — was partial | `core/channels.py` rejects privileged keys on transport. No automated per-group causal or ablation audit | M | fairness and leak claims |
| VLM system II | legged_vlm.md | **harness done 2026-09-28 (D-126: `harness/eval/system2.py` 98a9beb, `--system2` in the legged eval); no model/result** — was partial | `research/system2_eval.py` (ground, closed_loop) sits in `research/`, smoke test only, no results | L | System II claim |
| Dual-arm learned models | dualarm.md next 1–2 | missing | the `swap_slots` edit exists (`harness/eval/dual_latent_eval.py:234`), but no dual lineage has ever been trained | L | M=2 "across bodies" |
| Canonical source labels everywhere | provenance.md; audit §3 | **done 2026-09-28 (D-126 sl-1 switch 5aaea4d, default off; legged rows too)** — was partial | `core/action.py:10` is still a Literal with teacher / scripted_teacher / privileged_teacher. Ladder rows write a bare `"oracle"` (`harness/eval/ladder.py:649`) | S–M | labelling contract |

### backlog: Training / RL
| item | source | status | evidence | size | blocks |
|---|---|---|---|---|---|
| Arm GRPO with anchor evals | W7; §4.6 | code implemented (D-126 arm; experiment pending) | `harness/train/latent_grpo.py` and `grpo.py` have no anchor or forgetting eval. There is no grpo pipeline stage or DAG, and the only run was on a 0/64 base | M–L | W7 gate |
| Chunk-boundary blending (BC and system 0) | D-102(4) | code implemented (D-126 arm; experiment pending) | blending exists only inside the teacher (`arm_smooth.py`); `motion_quality` only flags steps above 1.5 rad/s | M | learned-policy smoothness |

### backlog: Evaluation / metrics / gates
| item | source | status | evidence | size | blocks |
|---|---|---|---|---|---|
| `validate_tracker` stage used in DAGs | D-114 | **done for future runs 2026-09-27** (commit feec007: `dags/templates/legged_v2_gated.yaml` + sha check + tests/unit/test_dag_templates.py; running/completed DAGs untouched) — was partial | stage at `harness/pipelines/legged.py:200`; no `dags/*.yaml` references it | S | W6 gate enforcement |
| Relax the anymal_c privileged end check | D-105 | partial | `legged_latent_eval.py:527` still uses `privileged_success()`; public success is only reported alongside | S | anymal_c R2 numbers |
| BC seeds 1702/1703 and SFT budgets (four-way comparison) | baselines.md | code implemented (D-126 arm; experiment pending) | campaign code exists; the cells were never run | M | fair baselines on target bodies |
| Long-duration drift runs | §4.10 | **code done 2026-09-28 (D-126: `--eval-mode long`, `long_run` drift metrics; research/tracks/d126_deploy.md)** — was missing | none found | S–M | — |
| Latency re-measured on current routes | acceptance.md next 4 | **code done 2026-09-28 (D-126: legged `--measure-latency`; arm `rrp latent latency --representation` for refit routes); measurement pending** — was missing | `harness/eval/latency.py` has only been run on old routes | S | — |

### backlog: Orchestration / ops / infra
| item | source | status | evidence | size | blocks |
|---|---|---|---|---|---|
| `scripts/ladder.py` main moved into `rrp.harness.eval` | W5; pipeline.md step 3 | **done 2026-09-28 (D-126 ade3628: `harness/eval/ladder_cli.py`, script = wrapper, parity test)** — was missing | the 149-line `main()` is still called as a subprocess by `harness/pipelines/arm.py` | S–M | in-process arm stages |
| Legged summary/effects scripts moved into the library | pipeline.md step 3 | **done 2026-09-28 (D-126 ade3628: `harness/eval/legged_summaries.py`, scripts = wrappers, parity test)** — was missing | `harness/pipelines/legged.py:347` runs `scripts/legged_ladder_summary.py` | S | — |
| Dual pipeline stages | W5; `harness/pipelines/dual.py` TODO | **done 2026-09-28 (D-126 #33, W12)** except dual DAgger (refuses: no dual label source) — was: missing (skeleton) | only `train_rep` and `train_flow`. Pack logic is still in `cli/dual_latent.cmd_pack`; there is no dual DAgger | M (L with DAgger) | dual models; "one pipeline" |
| Chain-script retirement | W5 gate; audit phase 4 | missing | 13 `*_chain.sh` remain; there are 107 `.sh` scripts in total (89 at audit time) | S–M | phase 5 |
| `rrp run-dag` host↔peer artifact transfer | pipeline.md step 4 | missing | `harness/dag.py:266-270` places jobs but has no transfer between nodes. Moot while D-115 keeps heavy work on the peer | M | mixed-node DAGs |
| Robot thermal and duty-cycle limits | §4.11 | missing | the only "thermal" hits are the host watchdog | M | — |

### backlog: Repo structure / docs
| item | source | status | evidence | size | blocks |
|---|---|---|---|---|---|
| Registry and requirements kept current | AGENTS.md | **done 2026-09-27** (commit a2f0248: D-095..D-123 entries, running entries closed, R39–R42) — was stale | `research/registry.jsonl` and `artifacts/requirements.json` were last committed 09-26 18:37; the newest entry is "W9 planned"; D-095..D-121 are absent | S | contract compliance |
| `STATUS.md` kept current | AGENTS.md | **done 2026-09-27** (commit 9d16009: W1–W12 table, evidence through D-123, running work) — was stale | header says "Updated 2026-09-26 18:40"; it shows W3 implementing and W4–W8 planned; decisions cited only up to D-094 | S | resume correctness |
| Demo page republished | demo.md | **rebuilt in the repo 2026-09-27** (commit ddc3468: 'since D-095' section); NOT published — the owner decides — was missing | last `docs/demo` refresh was 09-26 19:21; nothing from D-101..D-121 is on it | S–M | outside communication |
| `robustness_checklist` file | your list | missing, and never stated | no such file and no reference to it anywhere; the closest thing is the considerations doc | S | — |
| Considerations checklist kept current | its own header rule | **done 2026-09-27** (commit 616a90d: every §4 status re-checked, dated revision line) — was stale | still shows ❌ for robustness sweeps, latency and physics-version provenance, which are now implemented | S | — |
| One-off scripts moved to `research/scripts/<date>/` | audit; restructure.md | partial | 9 `t1_diag_*.sh`, the `diag_*.py` scripts and `binding_diag_learnability.py` are still in `scripts/` | S | — |
| Shim removal (phase 5) | audit phase 5 | pending (waiting on its precondition) | control/model/sim/morphology/ops/learning/policy (about 1000 lines) and the `cli_*.py` files are pure shims | S | — |
| Decide where the tracker lives (envs vs controllers) | restructure.md step 4 | missing | still in `envs/mujoco/legged_tracker.py` | S | — |

### backlog: psi1z
| item | source | status | evidence | size | blocks |
|---|---|---|---|---|---|
| rrp pin kept current | README, "bump the pin" | **done 2026-09-27** (psi1z P-020: pinned to rrp 68a6657) — was stale | `pyproject.toml` pins `b7dc677`, which is 207 commits behind rrp origin/main | S | shared fixes (gates, provenance) |
| Uses the rrp core (system 0, provenance, statistics, run-dag, broker) | README "dependencies" | partial | psi1z imports only `rrp.core.probe_guided_edit` and `rrp.policies.nets` attention/flow/gnll. Queueing is shell (`scripts/peer_queue.sh`); no rrp provenance or statistics | M | cross-repo consistency |
| "Affordance supervision" and "object entities" | README (the structure being tested) | partial | morphology-relation tokens, contact and binding (active hand) are in `structured.py`. Entity token positions are in `harness/data.py`. No affordance head or loss was found | M | the W10 claim as worded |
| Step 2 on BendPickMP and HandoverTeleop | D-120; notes "Next" | missing | only TabletopGraspMP has been started | M (compute) | W10 generality |
| Step 2 evals and packet edits on TabletopGraspMP | P-018/P-019 | partial | released 20/20; direct and structured evals are queued behind W7 | S | W10 result |

### backlog: Top 10 by value and by how much they block
1. **Latent route on the sealed target bodies (arm, plus legged `heldout`).** The target demos already exist; only DAG nodes are missing. This is the untested core cross-body claim.
2. **Fair baselines on the target bodies** (BC seeds 1702/1703, SFT budgets). Without them, item 1 has no fair comparison.
3. **Humanoid trackers** (a t1 that passes the lab gate, h1, g1). These block W8 humanoid results and the W1 gate.
4. **State estimation plus an automated per-group privileged audit.** Deployable observations currently rely on truth + noise.
5. **Actuator dynamics and latency as the default,** with sourced joint speeds, which needs trackers that hold their gaits under it.
6. **Dual pipeline stages and a first dual lineage.** "Across bodies" currently excludes M=2, and it blocks "one pipeline".
7. **GRPO with anchor evals as a pipeline stage.** This is the W7 gate and needs a competent checkpoint first.
8. **Chunk-boundary blending and limit-aware IK.** These make the D-112 policy and arm joint-margin gates enforceable.
9. **Terrain curriculum (and MJX to afford it).** Needed for rough-terrain credibility.
10. **Records and gate wiring:** registry, requirements, STATUS, demo republish, `validate_tracker` in DAGs, the psi1z pin bump. All small, and the AGENTS contract requires them.

Next tier: OOD packet fallback and a safety layer, loco-manipulation, `ladder.py` → library plus chain-script retirement.

### backlog: Checked and implemented
- D-126 arm code items (track d126arm; default-off options, experiments not run; research/tracks/d126_arm.md):
  limit-aware IK `IKSolver.solve(limit_margin=)` + teacher `v2lim` = `pick_place_v2_minjerk_lim` (collect `ik_limit_margin`);
  descent-phase DART `dart_descent_sigma` (with `dart_safety: phase`, proximity-guarded); task-object variants
  `build_pick_place(object_spec=)` / collect `object_variation` / `$RRP_OBJECT_VARIATION`; chunk blending `chunk_blend`
  (`rrp.policies.chunk_blend`, ladder `--chunk-blend`, stage option); arm stages `grpo` (latent + BC-GRPO, anchor /
  forgetting evals `rrp.harness.train.grpo_anchor`), `target_eval` (`rrp.harness.eval.target_eval`, sealed protocol),
  `target_adapt` (flow SFT, system-0 few-shot refit `episode_budget`, BC SFT), arm `train_bc`; stage option
  `grasp_contact`; DAG templates `dags/templates/arm_grpo.yaml`, `arm_targets_latent.yaml`, `arm_targets_bc.yaml`;
  recipe overlays `dags/overlays/arm_recipe/*` (list `extends`).
- Records refresh (D-123 item 10, records agent, 2026-09-27): STATUS 9d16009, registry + requirements a2f0248, considerations checklist 616a90d, validate_tracker DAG template feec007, demo page rebuilt (not published) ddc3468. The psi1z pin bump was done by the psi0 agent (psi1z P-020: pinned to rrp 68a6657).
- Contact v2, the reward schedule with permanent standing and `MIN_STOP_SHARE`, and the slip gate.
- Sourced actuator limits as the default (`bodies/actuator.py`); grasp contact v2/v2.1.
- The robustness harness (`harness/eval/robustness.py`, `envs/mujoco/perturb.py`, including arm pushes and terrain); motion quality in every eval row.
- Legged seed 1–2 sweeps (D-112).
- Gates in code (`harness/eval/gates.py`), including the legged dataset gate. The t1 data fails it under the exception recorded in D-113, and v1 data is labelled rather than gated, by design.
- Filtered foot force.
- Provenance and physics version; legged weight fingerprints (`policies/bundles.py`); the `Source` enum (`core/provenance.py:122`).
- Pydantic `RunConfig` with overlays and `extends`; an import-layering test; no silent `except ImportError` left in `cli/`.
- The arm pipeline with its parity DAG; the W8 legged DAGs.
- The v2 minimum-jerk teacher.
- The watchdog's memory.high throttle reporting and its anon+shmem+kernel memory count.
- Superseded: broker `/dev/shm` subtraction (D-116 reverted it); XMovePick follow-ups (closed by P-012).
