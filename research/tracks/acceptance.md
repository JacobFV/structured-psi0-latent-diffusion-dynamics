# track: acceptance (causal edits, composition, latency on the corrected latent path)

Branch `track/acceptance`, worktree `~/work/rrp-wt/acceptance`, peer dir `/dev/shm/rrp-brandonin/wt/acceptance`.
Spec: research/corrections/controller-facing-semantic-latent.md; list: research/reports/latent_slice1_progress.md "Pending".
Counterexample / embodiment swap are done elsewhere (D-032) and not repeated here.

## commands (reusable on any flow checkpoint; dev rule enforced: seeds >= 3,000,000, no target bodies)
- `rrp latent causal --checkpoint <flow policy.pt> [--robots panda_pg2,parm6_pg2,ur5e_pg2] [--episodes 16]
   [--episode-seeds 8] [--protocol window|episode|both] [--delta-cm 5] --out artifacts/runs/acceptance_causal_<tag>`
- `rrp latent composition ...` (same args; composition conditions only)
- `rrp latent latency --checkpoint <flow> --direct-config configs/model/policy-small-structured.json [--reps 100] --out ...`
- `scripts/render_causal_edit.py --checkpoint <flow> --robot panda_pg2 --seed 3000001 --condition <c> --out <dir>`
Code: src/rrp/evaluation/latent_causal.py, cli_latent.py (register_causal), latency.py (direct_config + interleaved
pairs), policy/latent_runner.py (`packets(..., noise_keys=)` per-session flow noise so control/edit runs are paired).

## design
Only the packet handed to system 0 changes; system i and system 0 are frozen. The post-hoc measurement probe
(`<rep>/probe_posthoc.pt`, same procedure for sem and nosem) is used ONLY to define probe-guided edit directions.
- window protocol (exactly paired): along an unedited closed-loop rollout, at ticks 8,32,...,128, snapshot; per condition
  restore, deliver the edited packet, run 8 ticks (one replan period, no system-i call); effect = TCP(final) - TCP(control).
  `control_replay` re-runs the unedited packet (must be exactly 0; it is).
- episode protocol: full closed loop (240 ticks), every received packet transformed; flow noise keyed by (seed, call).
- conditions: rel+-x / rel+y / rel+xy (probe-guided delta on z so probe rel_pos(task object, m0) moves by the requested
  5 cm with other readouts anchored), sum_xy (delta_x + delta_y without re-optimization), rand (random direction, norm of
  delta_x), cf+-x (system i's OWN packet for a counterfactual state with the object + its tracker belief moved 5 cm:
  on-manifold edit), zero, shuffle (other seed's packet, same body/tick), focus_swap (packet from the counterfactual in
  which the task object and distractor0 exchange places), chain_A_then_B (episode: unedited until tick 48, then
  focus_swap) = two objects in sequence.
- statistics: per-seed mean over decision points, bootstrap 95% CI over seeds (robot x seed units).

## log
- 2026-09-25 smoke (sem_v1 interrupted, panda_pg2, 3 seeds): pipeline verified; control_replay shift exactly 0.
  Early signal: zero/shuffle packets move the TCP far more than probe-guided or counterfactual 5 cm edits.
- 2026-09-25 latency on sem_v1 interrupted under heavy shared peer load (11 active leases): NOT a clean measurement.
  Raw: artifacts/runs/acceptance_latency/sem_v1_interrupted.json (peer store).
- 2026-09-25 window protocol, sem_v1 INTERRUPTED checkpoint (dev reference, not a result on the final flow; 3 source bodies x
  16 dev seeds, 42 feasible, 6 decision points each; raw artifacts/runs/acceptance_causal_sem_v1i/window_rows.jsonl):
  control_replay 0.00 cm (exact pairing). zero packet: TCP shift 5.85 [5.13, 6.70] cm in 0.4 s (control moves 2.92 cm);
  shuffle 2.81 [2.25, 3.51] cm => system 0 depends strongly on the packet. But semantic geometry edits do not steer:
  rel+x along edit +0.03 [0.00, 0.06] cm (gain 0.006 per requested m), rel-x +0.02 (wrong sign), rel+y -0.06 cm (wrong sign),
  rand 0.01 cm along x; cf+x/cf-x (system i's own packet with the object moved 5 cm; probe reads 4.4 cm) -0.00/-0.01 cm,
  antisymmetry -0.01 [-0.03, 0.01] cm; focus_swap toward distractor -0.04 [-0.12, 0.02] cm. Policy never grasped in 128 ticks.
  Composition: sum_xy vs sum of single-edit effects cosine 0.97, residual 0.08 (linear at this tiny amplitude, but the single
  effects themselves are ~1-3 mm noise-level perturbations, not directed); rel+xy jointly optimized: cosine 0.57.

## 2026-09-25 re-scope (lead; research/corrections/2026-09-25-causal-semantics-priorities.md item 4)
Zero/shuffle sensitivity is packet DEPENDENCE, not semantic control. The v1-interrupted reference runs were STOPPED on
the lead's request (scoped stops, leases 1790364195_73c2da and 1790366954_af0767) to free the peer GPU:
- sem_v1i: window protocol COMPLETE (numbers above; raw artifacts/runs/acceptance_causal_sem_v1i/window_rows.jsonl +
  window_summary.json, committed). Episode protocol PARTIAL, rows not saved; the log lines (0 successes in every condition
  on panda_pg2 8 seeds and parm6_pg2 4 feasible seeds) are in artifacts/runs/acceptance_causal_sem_v1i/episode_partial_log.txt.
- nosem_v1i (CPU, window only): stopped at panda_pg2 seed 12/16; no rows saved (rows are written at the end). Not a result.

New suite `rrp latent semantic-edits --route teacher|oracle|generated` (src/rrp/evaluation/latent_semantic_edits.py):
VALID semantic edits of the context the packet is generated for, physical scene unchanged, system 0 frozen:
rebind_obj (task rebound to distractor0 in the public belief; oracle demo = teacher on distractor0), goal_shift (target
zone belief displaced 12 cm; oracle demo places there), and IRRELEVANT controls: irrelevant_distractor (unbound
distractor belief moved 10 cm) and orthogonal_matched (random z edit orthogonal to all probe-readout gradients, norm =
||z_goal_shift - z_control|| per packet). Measures: which object is lifted, where objects end, new-goal success.
Routes: teacher (reference rung: expert native commands for the edited task, proves each edit is achievable), oracle
(ORACLE DIAGNOSTIC: E(context, teacher demo) -> system 0; source=target_encoder_oracle), generated (system i's packet).
Manipulator assignment: pending the dualarm track's paired tasks.
Scenes: pick_place with n_distractors = max(1, seed % 3) so rebind/irrelevant edits always have a distractor.

## results of the semantic-edit suite (panda_pg2 source body, dev seeds 3,000,000+)
State: suite **verified** (code + teacher rung + oracle rung); semantic control **not demonstrable yet**: no competent
packet route exists (the oracle route is not competent either; see below). Rerun on a competent checkpoint when the
ladder track names one.

| route (source label) | seeds | control followed | rebind_obj followed | goal_shift followed | irrelevant_distractor unchanged | orthogonal_matched unchanged |
|---|---|---|---|---|---|---|
| teacher (scripted_teacher, reference rung) | 12 | 12/12 cube in zone | 12/12 distractor0 lifted + in zone, cube untouched | 12/12 cube at shifted goal, 0/12 in old zone | 12/12 same as control | n/a |
| oracle (target_encoder_oracle, E(latent_sem_v1) + teacher demo; DIAGNOSTIC) | 8 | 0/8 (nothing grasped) | 0/8 | 0/8 | 0/8 success (same as control) | 0/8 |

Oracle graded signals (paired with control, bootstrap 95% CI over 8 seeds; raw
artifacts/runs/acceptance_semantic_sem_v1rep/semantic_rows_oracle.jsonl, summary semantic_summary_oracle.json):
mean TCP deviation from control: rebind_obj 8.4 [6.8, 9.7] cm, goal_shift 1.9 [1.6, 2.3] cm, irrelevant_distractor
1.8 [1.5, 2.1] cm, orthogonal_matched 0.9 [0.7, 1.0] cm. So the rebind edit changes behaviour far more than the
irrelevant edits, but NOT in the predicted direction: approach preference for distractor0 vs control is
-1.3 [-2.1, -0.7] cm (the TCP ended relatively closer to the cube). The goal edit cannot show anything before a grasp.
Orthogonal control: the probe-relevant span has rank 42 of 256 z dims; the matched norm is ~4% of ||z||.
Commands:
  `rrp latent semantic-edits --route teacher --representation artifacts/runs/latent_sem_v1/representation.pt --episodes 12 --max-steps 500 --out artifacts/runs/acceptance_semantic_teacher`
  `rrp latent semantic-edits --route oracle --representation artifacts/runs/latent_sem_v1/representation.pt --episodes 8 --out artifacts/runs/acceptance_semantic_sem_v1rep`
  (peer CPU leases 1790370424_4f7427, 1790369814_3235a2)

Why the oracle route fails (diagnostic for the ladder track; scratch script, peer CPU): on the stored training episode
pick_place_panda_pg2_s0, runtime featurization equals the stored inputs exactly (all banks, nodes, relations, q0), and
E -> R reproduces the teacher's 1-step command at t = 10 and 30 (e.g. [-0.02 -0.24 0.25 ...] vs [-0.02 -0.28 0.26 ...]),
but NOT at t = 0: there |z| = 98.6 (vs 20-73 later) and R outputs about -0.1 on every joint instead of the teacher's
wrist command -1.0. Online, system 0 therefore barely moves at the start; the shadow expert's tcp_cmd runs ahead, the
demonstrations become large (|a| grows from 1.2 to 3.8), and system 0 keeps under-producing them
(|a_s - a_t| from 0.9 to 3.7). A 12-tick expert warm start does not fix it (tcp-cube 0.34 m -> 0.22 m after 200 ticks).

## latency (keep; contaminated by shared load)
`rrp latent latency --checkpoint artifacts/runs/flow_latent_sem_v1/policy_interrupted.pt --direct-config configs/model/policy-small-structured.json --out artifacts/runs/acceptance_latency/sem_v1_interrupted.json`
The trained direct-action checkpoints (dev3/dev4) were lost in the 2026-09-21 reboot. Compute cost does not depend on the
weights, so the SAME direct-action architecture is timed with random init (labelled "timing only"). Measured on the peer
GB10 while 11 other leases were active, so it is NOT the protocol's quiet condition (NFE 4 p95 101.7 ms was slower than
NFE 8 p95 61.7 ms). The fairest number is the interleaved pairs: latent obs->first native command p95 175.5 ms vs direct
obs->chunk p95 156.5 ms at NFE 8, **p95 overhead ratio 1.12 (threshold 1.25)**, p50 ratio 1.12. Per replan period
including the 7 further system-0 ticks the latent path costs 2.24x the direct chunk (system-0 tick p95 25.0 ms, 0/80
misses of the 50 ms deadline). Rerun on sem_v2/v3 when the GPU is quiet.

## resume / next
1. When the ladder track names a competent route/checkpoint: `rrp latent semantic-edits --route generated --checkpoint
   <flow> --episodes 12 --out artifacts/runs/acceptance_semantic_<tag>` (and `--route oracle --representation <rep>` for
   the matching oracle), then `rrp latent causal ...` and `rrp latent composition ...` on the same checkpoint.
2. With the binding track's reps (binding_latent_{sem,nosem}_v2): oracle rung first (CPU is enough; E and R are small).
3. Manipulator assignment: needs the dualarm track's paired tasks (same scene, other arm assigned); add a condition.
4. Latency on sem_v2/v3 in a quiet GPU window with `--reps 100`.
