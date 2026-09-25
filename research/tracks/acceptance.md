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
