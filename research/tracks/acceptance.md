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

## sprint_bc coordination note (2026-09-25 ~19:15, from the baselines track; append-only)
- Plain BC with the B-1 fix is COMPETENT on the ladder scenes at mid-training (learned:direct1701_u12000: panda_pg2 25/30,
  parm6_tf3 27/30; details in research/tracks/baselines.md "SPRINT BC RESULT").
- BC route for the semantic-edit suite: `python -m rrp.evaluation.bc_semantic_edits --policy <ckpt> --label <tag>
  [--scene pick_place|paired] --robots panda_pg2 --episodes N --out <dir>` (src/rrp/evaluation/bc_semantic_edits.py).
  It reuses THIS track's module (context_edit, goal_offset, followed, summarize_semantic; and _scene, paired_edit_keys,
  approach_metrics, robot_contacts once your paired-scene version lands on main), so the numbers are directly comparable.
  BC has no packet: the edit is applied to the public context BC observes at each chunk (snapshot/edit/chunk/restore).
  Running now on pick_place scenes (u12000, 24 seeds, panda_pg2 + parm6_tf3; out artifacts/runs/baselines_bcsem_u12000/).
  Please ping here when the paired/approach-level version is on main; I will rerun BC with `--scene paired`.

## sprint (2026-09-25 19:00 -> 09-26 04:00): semantic edits at the level the routes reach (agent sprint_semantic)
State: implementing -> verified (teacher rung); oracle/learned runs in progress.

Code (merged to main 7381aba, de4955a):
- `rrp latent semantic-edits --scene paired`: binding-paired scenes (`pick_place_paired`, identical physics for every
  assignment). Episode key = 10*scene_seed + patient, patient alternates with the seed; rebind target = distractor0
  (first other cube in physical order). The VALID rebind edit changes only the public task context: the `cube` entity's
  descriptor becomes the other cube's color and the public entity->slot binding follows (`rebind_descriptor`); tracker
  beliefs, physics and runtime statuses are untouched. Conditions: control, rebind_obj, goal_shift (zone belief 12 cm),
  irrelevant_distractor (unbound belief 10 cm), orthogonal_matched (probe-orthogonal z, norm = ||z_rebind - z_control||
  per packet), control_replay (generated route only: other flow noise; oracle/teacher are deterministic).
- Approach-level metrics (privileged measurement): min TCP distance to old (cube) / new (distractor0) object, which
  object the TCP first comes within 6 cm of, first robot-object contact, cosine of the initial xy motion (first 4 cm)
  toward each object, and for goal edits the end position of the cube (only if moved >= 2 cm) relative to old/new goal.
  Summaries: counts with Wilson CIs, paired-with-control bootstrap CIs, and contrasts valid-edit minus control-edit
  (difference of paired differences per scene).
- `rrp latent arm-edits`: dual-arm assign_pick_place dev pairs (3 robot pairs, seeds 3,000,000+; assigned arm = left
  for even seeds, right for odd). swap_arm = VALID context edit (actor of take/place rebound = the other variant's task
  graph, `rebind_actor`); swap_slots = control packet with the two assembly slots exchanged; orthogonal_matched (norm =
  ||z_swap_arm - z_control||); control_replay. Metrics per arm: min distance to the bar, first arm to touch the bar,
  TCP path share. DualLatentSystem0 now applies the anchored-realizer column (was missing: v4 bundles are anchored) and
  DualLatentPolicy takes noise keys.
- `scripts/render_causal_edit.py --suite semantic|arm`: side-by-side video from the SAME runner as the measurement.
- `scripts/sem_merge.py`: merge sharded rows -> one summary.

### teacher rung (scripted_teacher, privileged; proves each edit is achievable and the metrics read it)
`rrp latent semantic-edits --route teacher --scene paired --representation <any> --episodes 30 --seed-start 3100000 --max-steps 500 --conditions control,rebind_obj,goal_shift,irrelevant_distractor --out artifacts/runs/acceptance_sprint_sem_teacher_paired`
(peer lease 1790388785_36db43). 30 scenes, panda_pg2: control approached/touched the assigned cube 30/30; rebind_obj
approached the new cube first 30/30, first touch new 29/30, lifted+placed new 30/30; goal_shift cube ends at the new goal
30/30 (end-position preference +11.2 cm [11.0, 11.4]); irrelevant_distractor identical to control 30/30.
Arm: `rrp latent arm-edits --route teacher --episodes 8 --max-steps 700 --conditions control,swap_arm --out artifacts/runs/acceptance_sprint_arm_teacher`
(lease 1790388786_c5d0ce): 3 pairs x 8 seeds = 24; swap_arm: the edited-to arm touches the bar first 24/24 (control 0/24),
bar-distance preference +0.35 m. (privileged_success is scored against the REAL task, so it is 0 for the edited arm.)

### oracle rung, B-1-fixed bundle (ORACLE DIAGNOSTIC: E(ladder_latent_sem_b1fix_anchor) + scripted_teacher demo -> frozen system 0)
6 peer CPU shards (leases 1790388786_981f44 .. 1790388789_de0d14), each
`rrp latent semantic-edits --route oracle --scene paired --representation artifacts/runs/ladder_latent_sem_b1fix_anchor/representation.pt --episodes 5 --seed-start 31000{00,05,..,25} --max-steps 300 --out artifacts/runs/acceptance_sprint_sem_b1fix_oracle/shard<i>`,
merged by `scripts/sem_merge.py` -> `artifacts/runs/acceptance_sprint_sem_b1fix_oracle/semantic_summary_oracle.json`.
30 paired scenes, panda_pg2, 0 task successes in any condition (the route is not competent), but the approach changes:
| condition | first approach old/new/none | first touch old/new/none | min-dist pref. new vs control (m) | init-dir pref. vs control |
|---|---|---|---|---|
| control | 15/4/11 | 13/5/12 | - | - |
| rebind_obj (valid) | 3/12/15 | 2/19/9 | +0.163 [+0.123, +0.211] | +0.81 [+0.49, +1.17] |
| irrelevant_distractor | 16/3/11 | 13/5/12 | +0.007 [-0.004, +0.020] | +0.00 [-0.00, +0.00] |
| orthogonal_matched (norm = rebind) | 22/4/4 | 22/8/0 | +0.019 [-0.002, +0.042] | +0.26 [+0.04, +0.55] |
| goal_shift (valid) | 13/6/11 | 14/6/10 | -0.003 [-0.018, +0.010] | -0.00 |
Contrasts (per scene, rebind effect minus control-edit effect): min-dist +0.156 m [0.115, 0.205] vs irrelevant,
+0.144 [0.110, 0.183] vs orthogonal. Goal edit: the cube is moved in only ~12/30 (pushed, never carried), end-position
preference for the new goal vs irrelevant +0.004 m [-0.019, 0.025]: no measurable goal effect (nothing is transported).
CAVEAT (lead, D-049): the oracle packet encodes the teacher's demonstration toward the new object and the shadow
teacher is not a valid expert off-trajectory, so this is WEAK evidence: it shows that system 0 follows the packet's
content, not that anything reads the supplied binding. The binding test is the generated route (v4 flows) and the BC
reference below.

### BC reference controller (added at the lead's request, 19:15)
`--route bc`: the competent direct-action BC `learned:artifacts/runs/baselines_bc_ckpts/direct1701_u12000.pt`
(sha256 5e6586bd6000412c.., track baselines: 25/30 panda_pg2), NOT the latent path; context-conditioned (sees the task
graph); the chunk for an edited condition is computed from the edited public context and executed in the real scene;
flow noise keyed per (seed, call); control_replay = other noise.
- Paired scenes (smoke, 1 scene): BC is NOT competent there (hovers over the target zone; the paired scenes have
  permuted slot order and new colors, which BC never saw). So BC is tested on the canonical pick_place scenes with
  `rebind_desc` = the same descriptor/binding-only edit (task entity "red cube" -> distractor0's descriptor).
- Running: 4 peer shards `acceptance_sprint_sem_bc_pp/shard{0..3}` (32 seeds from 3,000,000; control, rebind_desc,
  rebind_obj (belief swap), goal_shift, irrelevant_distractor, control_replay; 400 steps) + teacher rung on the same
  scenes `acceptance_sprint_sem_teacher_pp`.
