# evidence matrix (current; updated 2026-09-25 14:30 PDT)

The single place that states what has and has not been shown. Every cell is backed by a saved raw output (paths
given) or says **not shown**. Column meanings:
- **implementation**: code exists and a smoke run of the real pipeline passed.
- **oracle-target behaviour**: packets made by the frozen encoder from the TEACHER's demonstrated actions (an oracle
  diagnostic that uses future actions; not deployable).
- **generated-packet behaviour**: packets sampled by system i from public observations (the deployable route).
- **semantic interventions**: VALID edits of the packet (binding, manipulator, requested effect) with irrelevant-edit
  controls. Zero/shuffle sensitivity and probe accuracy do not count as semantic control.
- **held-out transfer**: target bodies of the sealed protocol (xarm7_pg2, xarm7_tf3, panda_tf3).

Headline: **no learned policy is competent in closed loop yet, and causal packet semantics are not shown.** Bug B-1 (D-044): system 0 was trained to copy the previous teacher command, which is zero at deployment, so every learned closed-loop number below is dominated by that bug until the realizer is refit.
The priority is a competent source controller and causal packet semantics
(research/corrections/2026-09-25-causal-semantics-priorities.md).

| component | implementation | oracle-target behaviour | generated-packet behaviour | semantic interventions | held-out transfer |
|---|---|---|---|---|---|
| single-arm pick_place, corrected path (latent_sem) | verified: stage A, stage B (standardized, resumable), system 0 runtime, eval, disturbance, render (D-031) | Probes on encoded z, training distribution: held_by_pos 0.956, rel_pos 0.054 m, subtask 1.00; shuffled z 0.151 / 0.365 m / 0.645; metadata-only 0.000 / 0.268 m / 0.784 (`artifacts/runs/latent_*_v1/probe_*.json`). System-0 1-step teacher-forced MSE 0.0129 vs zero-action 0.213. **Closed-loop oracle route FAILS: 0/8 on panda_pg2, never grasps**, while the scripted teacher succeeds 12/12 on the same seeds (`artifacts/runs/acceptance_semantic_{teacher,sem_v1rep}/`). Root cause B-1 (D-044). With B-1 fixed, system-0 refits on the frozen v1 encoder (refit, DAgger 1/2, anchored) are still 0/30 on panda_pg2 and parm6_tf3 (D-046): closed-loop compounding. A joint Stage-A retrain with the fix (D-047) is still 0/30 on the oracle route, but it now reaches the cube (0.3–4 cm) and fails at grasp/lift: system 0 does not follow a 'hover at pregrasp' packet precisely. Earlier symptom: at step 0 the packet norm is 98.6 (20–73 later) and system 0 outputs ≈−0.1 where the teacher commands −1.0; the arm barely moves and never recovers. Ladder track is localizing it. | Closed loop panda_pg2, dev seeds: flow v1 interrupted **0/64**; flow sem_v2 snapshot @22k **0/32**, never grasps, min TCP-cube 0.43 m (`artifacts/runs/grpo_latent_ref_*`). With a labelled 40-tick teacher prefix: 1/32 success, 12/32 grasps (not deployable). Full sem_v2 / nosem_v2 / sem_v3 dev evals: running (lead chain, host). | Counterexample (object rebinding) **FAILS**: 0/14 focus changes (D-032). The test had limitations (asymmetric graph edit, slot prior: metadata-only probe already gets focus 0.927); binding track is repairing it (symmetric rebinding, permutations, balanced bindings). Embodiment swap pg2↔tf3 on encoded z: focus agreement 0.98, subtask 1.00. Semantic edits (`rrp latent semantic-edits`): scripted teacher 12/12 for rebind_obj, goal_shift and both irrelevant controls, so the edits are physically achievable. Oracle route: rebinding moves the arm 8.4 [6.8, 9.7] cm from control vs 1.8/0.9 cm for irrelevant edits, but toward the ORIGINAL cube (−1.3 cm): packet dependence, not semantic control. Causal edits on the v1-interrupted generator: the packet matters (zero packet moves TCP 5.85 cm in 0.4 s vs 2.92 cm control), but directed edits are noise-level (rel +x: +0.03 cm) → **no semantic control shown**. | **not shown** (sealed; no competent source controller) |
| latent_nosem (capacity-matched control) | verified | Probes: held_by_pos 0.888, rel_pos 0.258 m (≈ metadata only) | nosem_v2 training | counterexample 0/14 (same limitations) | not shown |
| baseline_direct_action / action_only_codec | verified runner (`rrp campaign baseline-cell`, exact resume; D-035) | n/a | seed-1701 sources training on peer; **no results** | n/a | budget-0 cells queued after sources; SFT budgets on hold |
| packet-policy GRPO | verified code + likelihood tests (`tests/unit/test_latent_grpo.py`) | n/a | dev (panda_pg2, base sem_v2 @22k): 0/64 from reset before and after 15 iterations; with the labelled 40-tick teacher prefix 0/64 → 1/64 (no improvement; D-042). On hold until a source policy is competent | n/a | not started |
| dual-arm (support_insert, handover), M=2 | verified pack/train smoke on 2 assemblies | teacher reference (scripted_teacher): support_insert 20/20, 19/20, 20/20, held-out source pair 9/17; handover 20/20 ×3, 16/16 | **not shown** (training deferred, D-043) | paired arm-assignment tasks ready: 2,203 valid identical-scene pairs; scripted teacher performs both assignments (D-043); learned: not shown | not shown |
| legged / humanoid | data collection + representation code | teacher data only | **not shown** (winding down: peripheral) | not shown | not shown |
| VLM system II | smoke (Qwen3-VL weights hash-verified, isolated packages) | n/a | **not shown** | not shown | not shown |
| latency | verified (`rrp latent latency --direct-config`) | — | p95 1.12× direct-action path (threshold 1.25×), 0/80 system-0 deadline misses. Caveats: direct path timed with random weights (same compute), peer loaded (interleaved pairs); rerun on final flows in a quiet window | — | — |
| compatibility / safety | verified: bundle fingerprint in compatibility IDs with a mismatch-rejection test (D-038); packet admission, staleness, spec checks | — | — | — | — |

## in flight on the critical path
1. ladder track: R0 expert → tracker, R1 oracle packet → system 0, R2 generated packet → system 0, on matched scenes (item 3).
2. binding track: repaired counterexample, then paired physically consistent tasks through the whole pipeline, plus goal/predicted/observed-effect separation (items 1, 2, 5).
3. acceptance track: valid semantic interventions plus irrelevant-edit controls, once a competent checkpoint exists (item 4).
4. lead chain (host GPU): flow sem_v2 → nosem_v2 → sem_v3, each with a 20-episode dev eval on 4 source bodies, a disturbance test and videos.
