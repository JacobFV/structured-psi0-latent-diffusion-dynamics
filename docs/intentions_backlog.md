# stated-but-unimplemented intentions (backlog)

Audit of 2026-09-27 (read-only; rrp origin/main ed537d2, psi1z 8f7e297), recorded as D-123. It lists every intention stated in
docs, decisions or track notes that is only partly implemented or not implemented in code, verified against the code. Keep it
current: when an item lands, move it to "checked and implemented" with the commit; when an item is dropped, mark it superseded
and cite the decision.

I found 44 intentions that are stated in the docs but only partly built or not built. Checked against rrp origin/main `ed537d2` and psi1z main `8f7e297`. Nothing was modified, and the audit worktree has been removed.

Two things to know first:
- `docs/robot_training_considerations.md` and `STATUS.md` are themselves out of date. Several items marked ❌ there now exist in code; those are listed under "Checked and implemented" at the end.
- The only real TODO or not-implemented markers in code are the skeleton in `pipelines/dual.py` and the `NotImplementedError` stubs in `contracts/system0.py`. The latter are abstract base methods, not gaps.

## Physics / sim
| item | source | status | evidence | size | blocks |
|---|---|---|---|---|---|
| Actuator dynamics and latency on by default | W1; D-103(2), D-107(4) | partial | `physics/actuator.py` (armature, friction, torque–speed, 0–30 ms) is opt-in, by decision. Max joint speeds are still estimates, and trackers don't hold their gaits under it (h1 no-fall 0.72) | M–L | actuator-realism claim |
| MJX GPU simulation | considerations §4.7, §5.5 | missing | no mjx or jax anywhere in `src/` | L | terrain curriculum at scale |
| Terrain curriculum in tracker training | §4.2 | missing | terrain is eval-only: `envs/perturb.py` `terrain_amp_m` heightfield. `training/tracker_training.py` has only a turn curriculum | M | rough-terrain claims (break-point at 8 cm) |
| Link-length, encoder-offset and motor-strength randomization in training | §4.2 | partial | mass, CoM and gains exist only as eval perturbations (`envs/perturb.py`). No link-length or joint-offset randomization anywhere | M | — |
| IMU bias/drift, encoder quantization, tick jitter or missed-tick injection | §4.3, §4.4 | missing | none found; `evaluation/latency.py` only counts deadline misses | S–M | — |

## Robots / bodies
| item | source | status | evidence | size | blocks |
|---|---|---|---|---|---|
| h1 tracker that can turn in place | D-103(3); contact.md "PARKED" | missing | `artifacts/trackers/h1/contact_v2` walks but has turn ratio 0.02. The planned phase-clocked gait was never started | L | h1 in W8; W1 gate |
| g1 tracker with sourced limits that doesn't stomp | D-114 addendum | partial | g1_src fails the gates (slip 0.38, 4.09 BW); no run after that | M–L | g1 in W8 |
| t1 tracker that passes the lab gate under the waypoint command mix | D-107(2), D-113 | partial | w8d has forward 0.72 and joint margin −0.053. Learned t1 routes fall (BC 0/30) | L | W8 t1 claim |
| Retune the joint-limit-margin reward | D-114 addendum | partial | `RewardCfg.limit_margin=-1.0` is in `envs/legged_core.py`, but no tracker has been trained with it | S | next tracker run |
| Latent route on the sealed target bodies | §4.8; W9 | code implemented (D-126 arm; experiment pending) | target demos (xarm7_*, panda_tf3) are collected in `dags/armexpert_v6dart.yaml`, and the arm/legged `heldout` stages exist. No DAG node evaluates or adapts the latent route on those bodies | M | central cross-body claim |
| Legged held-out body, DAgger and refit in the v2 DAGs | W8/W9 | partial | `legged_v2_*.yaml` goes collect→…→eval_r2/edits; the `heldout` and `refit` stages are never used | S | legged held-out claim |

## Data / teachers
| item | source | status | evidence | size | blocks |
|---|---|---|---|---|---|
| Limit-aware IK | D-114(3) | code implemented (D-126 arm; experiment pending) | no margin or barrier term in `bodies/ik.py` or `teachers/arm_smooth.py` (only a comment in `evaluation/gates.py:232`) | M | enforcing the parm* joint-margin gate |
| Arm object variety (size, mass, friction, shape) | §4.2 | code implemented (D-126 arm; experiment pending) | distractors exist, and a `cube_size` parameter exists but is fixed at the default (`envs/scenario.py`). Mass and friction are eval perturbations only; no shape variety | M | — |
| Loco-manipulation task | §4.9; W9 | missing | none in rrp (only `tasks/waypoint_contact.json`). psi1z drops walking-heavy tasks by D-120 | L | W9 |
| Grasp-accept tolerance under DART | armexpert.md backlog | partial | only the 1.6 cm reachability tolerance (`arm_smooth.py:312,408`) | S | — |

## Models / packet / representation
| item | source | status | evidence | size | blocks |
|---|---|---|---|---|---|
| OOD packet detection with a fallback controller | §4.11 | **code done 2026-09-28 (D-126: `controllers/packet_ood.py`, `--packet-ood off\|monitor\|enforce`, fit/score tool; research/tracks/d126_deploy.md); experiment #29 pending** — was partial | `contracts/system0.py` checks only version and staleness, with a `hold_measured` fallback. No distributional check | M | safe deployment claim |
| Safety layer (clamp, rate limit), safe stop, fall recovery | §4.11 | **code done 2026-09-28 (D-126: `controllers/safety.py`, `--safety off\|monitor\|enforce`; research/tracks/d126_deploy.md); experiment #30 pending** — was missing | only `np.clip(±6)` in `controllers/policy_runner.py:94`. No `safe_stop` or recovery | M / L | hardware framing |
| State estimation | §4.3 | **code done 2026-09-28 (D-126: `envs/state_estimator.py` bse-1 fills `base_vel_estimate`; `--base-state-source estimator`; research/tracks/d126_deploy.md); experiment #27 pending** — was partial | Tracker actors use IMU + encoders only; true base velocity goes only to the critic (`envs/legged_core.py` `priv`). System i's speed comes from a "declared localization" that is truth + Gaussian noise (`envs/legged.py _sense`). `base_vel_estimate` (`contracts/observation.py:26`) is never filled | M–L | deployable-observation credibility |
| Privileged-information audit per input group | §4.3 | **code done 2026-09-28 (D-126: `evaluation/privileged_audit.py` ablation runner + static/schema/tripwire checks; research/tracks/d126_deploy.md); experiment #28 pending** — was partial | `contracts/channels.py` rejects privileged keys on transport. No automated per-group causal or ablation audit | M | fairness and leak claims |
| VLM system II | legged_vlm.md | **harness done 2026-09-28 (D-126: `evaluation/system2.py` 98a9beb, `--system2` in the legged eval); no model/result** — was partial | `research/system2_eval.py` (ground, closed_loop) sits in `research/`, smoke test only, no results | L | System II claim |
| Dual-arm learned models | dualarm.md next 1–2 | missing | the `swap_slots` edit exists (`evaluation/dual_latent_eval.py:234`), but no dual lineage has ever been trained | L | M=2 "across bodies" |
| Canonical source labels everywhere | provenance.md; audit §3 | **done 2026-09-28 (D-126 sl-1 switch 5aaea4d, default off; legged rows too)** — was partial | `contracts/action.py:10` is still a Literal with teacher / scripted_teacher / privileged_teacher. Ladder rows write a bare `"oracle"` (`evaluation/ladder.py:649`) | S–M | labelling contract |

## Training / RL
| item | source | status | evidence | size | blocks |
|---|---|---|---|---|---|
| Arm GRPO with anchor evals | W7; §4.6 | code implemented (D-126 arm; experiment pending) | `training/latent_grpo.py` and `grpo.py` have no anchor or forgetting eval. There is no grpo pipeline stage or DAG, and the only run was on a 0/64 base | M–L | W7 gate |
| Chunk-boundary blending (BC and system 0) | D-102(4) | code implemented (D-126 arm; experiment pending) | blending exists only inside the teacher (`arm_smooth.py`); `motion_quality` only flags steps above 1.5 rad/s | M | learned-policy smoothness |

## Evaluation / metrics / gates
| item | source | status | evidence | size | blocks |
|---|---|---|---|---|---|
| `validate_tracker` stage used in DAGs | D-114 | **done for future runs 2026-09-27** (commit feec007: `dags/templates/legged_v2_gated.yaml` + sha check + tests/unit/test_dag_templates.py; running/completed DAGs untouched) — was partial | stage at `pipelines/legged.py:200`; no `dags/*.yaml` references it | S | W6 gate enforcement |
| Relax the anymal_c privileged end check | D-105 | partial | `legged_latent_eval.py:527` still uses `privileged_success()`; public success is only reported alongside | S | anymal_c R2 numbers |
| BC seeds 1702/1703 and SFT budgets (four-way comparison) | baselines.md | code implemented (D-126 arm; experiment pending) | campaign code exists; the cells were never run | M | fair baselines on target bodies |
| Long-duration drift runs | §4.10 | **code done 2026-09-28 (D-126: `--eval-mode long`, `long_run` drift metrics; research/tracks/d126_deploy.md)** — was missing | none found | S–M | — |
| Latency re-measured on current routes | acceptance.md next 4 | **code done 2026-09-28 (D-126: legged `--measure-latency`; arm `rrp latent latency --representation` for refit routes); measurement pending** — was missing | `evaluation/latency.py` has only been run on old routes | S | — |

## Orchestration / ops / infra
| item | source | status | evidence | size | blocks |
|---|---|---|---|---|---|
| `scripts/ladder.py` main moved into `rrp.evaluation` | W5; pipeline.md step 3 | **done 2026-09-28 (D-126 ade3628: `evaluation/ladder_cli.py`, script = wrapper, parity test)** — was missing | the 149-line `main()` is still called as a subprocess by `pipelines/arm.py` | S–M | in-process arm stages |
| Legged summary/effects scripts moved into the library | pipeline.md step 3 | **done 2026-09-28 (D-126 ade3628: `evaluation/legged_summaries.py`, scripts = wrappers, parity test)** — was missing | `pipelines/legged.py:347` runs `scripts/legged_ladder_summary.py` | S | — |
| Dual pipeline stages | W5; `pipelines/dual.py` TODO | missing (skeleton) | only `train_rep` and `train_flow`. Pack logic is still in `cli/dual_latent.cmd_pack`; there is no dual DAgger | M (L with DAgger) | dual models; "one pipeline" |
| Chain-script retirement | W5 gate; audit phase 4 | missing | 13 `*_chain.sh` remain; there are 107 `.sh` scripts in total (89 at audit time) | S–M | phase 5 |
| `rrp run-dag` host↔peer artifact transfer | pipeline.md step 4 | missing | `orchestration/dag.py:266-270` places jobs but has no transfer between nodes. Moot while D-115 keeps heavy work on the peer | M | mixed-node DAGs |
| Robot thermal and duty-cycle limits | §4.11 | missing | the only "thermal" hits are the host watchdog | M | — |

## Repo structure / docs
| item | source | status | evidence | size | blocks |
|---|---|---|---|---|---|
| Registry and requirements kept current | AGENTS.md | **done 2026-09-27** (commit a2f0248: D-095..D-123 entries, running entries closed, R39–R42) — was stale | `research/registry.jsonl` and `artifacts/requirements.json` were last committed 09-26 18:37; the newest entry is "W9 planned"; D-095..D-121 are absent | S | contract compliance |
| `STATUS.md` kept current | AGENTS.md | **done 2026-09-27** (commit 9d16009: W1–W12 table, evidence through D-123, running work) — was stale | header says "Updated 2026-09-26 18:40"; it shows W3 implementing and W4–W8 planned; decisions cited only up to D-094 | S | resume correctness |
| Demo page republished | demo.md | **rebuilt in the repo 2026-09-27** (commit ddc3468: 'since D-095' section); NOT published — the owner decides — was missing | last `docs/demo` refresh was 09-26 19:21; nothing from D-101..D-121 is on it | S–M | outside communication |
| `robustness_checklist` file | your list | missing, and never stated | no such file and no reference to it anywhere; the closest thing is the considerations doc | S | — |
| Considerations checklist kept current | its own header rule | **done 2026-09-27** (commit 616a90d: every §4 status re-checked, dated revision line) — was stale | still shows ❌ for robustness sweeps, latency and physics-version provenance, which are now implemented | S | — |
| One-off scripts moved to `research/scripts/<date>/` | audit; restructure.md | partial | 9 `t1_diag_*.sh`, the `diag_*.py` scripts and `binding_diag_learnability.py` are still in `scripts/` | S | — |
| Shim removal (phase 5) | audit phase 5 | pending (waiting on its precondition) | control/model/sim/morphology/ops/learning/policy (about 1000 lines) and the `cli_*.py` files are pure shims | S | — |
| Decide where the tracker lives (envs vs controllers) | restructure.md step 4 | missing | still in `envs/legged_tracker.py` | S | — |

## psi1z
| item | source | status | evidence | size | blocks |
|---|---|---|---|---|---|
| rrp pin kept current | README, "bump the pin" | **done 2026-09-27** (psi1z P-020: pinned to rrp 68a6657) — was stale | `pyproject.toml` pins `b7dc677`, which is 207 commits behind rrp origin/main | S | shared fixes (gates, provenance) |
| Uses the rrp core (system 0, provenance, statistics, run-dag, broker) | README "dependencies" | partial | psi1z imports only `rrp.core.probe_guided_edit` and `rrp.models` attention/flow/gnll. Queueing is shell (`scripts/peer_queue.sh`); no rrp provenance or statistics | M | cross-repo consistency |
| "Affordance supervision" and "object entities" | README (the structure being tested) | partial | morphology-relation tokens, contact and binding (active hand) are in `structured.py`. Entity token positions are in `data.py`. No affordance head or loss was found | M | the W10 claim as worded |
| Step 2 on BendPickMP and HandoverTeleop | D-120; notes "Next" | missing | only TabletopGraspMP has been started | M (compute) | W10 generality |
| Step 2 evals and packet edits on TabletopGraspMP | P-018/P-019 | partial | released 20/20; direct and structured evals are queued behind W7 | S | W10 result |

## Top 10 by value and by how much they block
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

## Checked and implemented
- D-126 arm code items (track d126arm; default-off options, experiments not run; research/tracks/d126_arm.md):
  limit-aware IK `IKSolver.solve(limit_margin=)` + teacher `v2lim` = `pick_place_v2_minjerk_lim` (collect `ik_limit_margin`);
  descent-phase DART `dart_descent_sigma` (with `dart_safety: phase`, proximity-guarded); task-object variants
  `build_pick_place(object_spec=)` / collect `object_variation` / `$RRP_OBJECT_VARIATION`; chunk blending `chunk_blend`
  (`rrp.controllers.chunk_blend`, ladder `--chunk-blend`, stage option); arm stages `grpo` (latent + BC-GRPO, anchor /
  forgetting evals `rrp.training.grpo_anchor`), `target_eval` (`rrp.evaluation.target_eval`, sealed protocol),
  `target_adapt` (flow SFT, system-0 few-shot refit `episode_budget`, BC SFT), arm `train_bc`; stage option
  `grasp_contact`; DAG templates `dags/templates/arm_grpo.yaml`, `arm_targets_latent.yaml`, `arm_targets_bc.yaml`;
  recipe overlays `dags/overlays/arm_recipe/*` (list `extends`).
- Records refresh (D-123 item 10, records agent, 2026-09-27): STATUS 9d16009, registry + requirements a2f0248, considerations checklist 616a90d, validate_tracker DAG template feec007, demo page rebuilt (not published) ddc3468. The psi1z pin bump was done by the psi0 agent (psi1z P-020: pinned to rrp 68a6657).
- Contact v2, the reward schedule with permanent standing and `MIN_STOP_SHARE`, and the slip gate.
- Sourced actuator limits as the default (`physics/actuator.py`); grasp contact v2/v2.1.
- The robustness harness (`evaluation/robustness.py`, `envs/perturb.py`, including arm pushes and terrain); motion quality in every eval row.
- Legged seed 1–2 sweeps (D-112).
- Gates in code (`evaluation/gates.py`), including the legged dataset gate. The t1 data fails it under the exception recorded in D-113, and v1 data is labelled rather than gated, by design.
- Filtered foot force.
- Provenance and physics version; legged weight fingerprints (`controllers/bundles.py`); the `Source` enum (`contracts/provenance.py:122`).
- Pydantic `RunConfig` with overlays and `extends`; an import-layering test; no silent `except ImportError` left in `cli/`.
- The arm pipeline with its parity DAG; the W8 legged DAGs.
- The v2 minimum-jerk teacher.
- The watchdog's memory.high throttle reporting and its anon+shmem+kernel memory count.
- Superseded: broker `/dev/shm` subtraction (D-116 reverted it); XMovePick follow-ups (closed by P-012).