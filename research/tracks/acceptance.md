# track: acceptance (causal edits, composition, latency on the corrected latent path)

## SPRINT SEMANTIC RESULTS (2026-09-25/26 demo sprint; agent sprint_semantic; details in "sprint" below)
Question: do VALID edits of the task semantics in the context the packet is generated for change behaviour causally,
beyond irrelevant edits of matched size? Measured at the level each route reaches (approach / first touch / end
position), panda_pg2, dev seeds; 95% bootstrap CIs over scenes; all numbers from the raw rows named below.
| route (source label) | scenes | rebind: first touch NEW (edit / control) | rebind effect beyond irrelevant edit, min-dist pref. (m) | goal edit: cube at new goal / end-pos. effect (m) | arm swap: edited-to arm touches bar first |
|---|---|---|---|---|---|
| scripted_teacher (privileged reference) paired | 30 | 29/30 / 0/30 | +0.295 | 30/30 / +0.222 | 24/24 (control 0/24) |
| ORACLE DIAGNOSTIC E(ladder_latent_sem_b1fix_anchor)+teacher demo, paired | 30 | 19/30 / 5/30 | +0.156 [+0.115, +0.205] | 0/30 / +0.004 [-0.019, +0.025] | n/a (single-arm bundle) |
| learned:direct1701_u12000 (BC reference, NOT latent), canonical scenes, rebind_desc | 32 | 0/32 / 0/32 | +0.006 [+0.001, +0.013] | 24/32 / +0.167 [+0.141, +0.190] | n/a |
| ORACLE DIAGNOSTIC E(binding_paired_sem_v4)+teacher demo, paired | 30 | 8/30 / 5/30 | +0.037 [-0.001, +0.077] | no transport / none | 6/18 (control 4/18); vs orthogonal +0.001 m [-0.072, +0.070] |
| ORACLE DIAGNOSTIC E(binding_paired_nosem_v4)+teacher demo, paired | 30 | 4/30 / 7/30 | +0.052 [+0.020, +0.082] | no transport / none | 2/18 (control 4/18); vs orthogonal +0.006 [-0.025, +0.037] |
| SPRINT BEST ROUTE: ORACLE DIAGNOSTIC E(ladder_rz_jointfix_bcdag2)+stateless BC demo -> jfbcdag2, canonical, rebind_desc | 48 | 4/48 first approach new (control 0/48); old cube abandoned 38/48 | +0.163 [+0.130, +0.200] (by abandoning, not redirecting) | **16/48 at new goal** (control 1/48, orthogonal 0/48) / +0.104 [+0.081, +0.127] | n/a |
| **DEPLOYABLE generated route**: learned:ladder_flow_jointfix/snap_final_s20000 -> system 0 ladder_rz_jointfix_gendag1_noqd, **parm6_tf3**, canonical, rebind_desc | 41 | **28/41** / 3/41 (first approach new 31/41 vs 0/41) | +0.211 [+0.175, +0.248] (beyond orthogonal +0.205 [+0.170, +0.240]; beyond noise replay +0.216 [+0.179, +0.253]) | **15/41 at new goal** (control 0/41, irrelevant 1/41, orthogonal 0/40, replay 0/40) / +0.104 [+0.077, +0.131] | n/a |
| **DEPLOYABLE, POOLED 80 seeds** (original 41 + lead extension 39, seeds 3,000,000-3,000,119; D-075) | 80 | **56/80** [0.59, 0.79] / 6/80 (first approach new 60/80 vs 0/80; old cube lifted 0/80 vs 58/80 unedited) | +0.231 [+0.202, +0.261] (beyond orthogonal +0.223 [+0.194, +0.252]; beyond replay +0.236 [+0.207, +0.265]) | **23/80** [0.20, 0.39] at new goal (controls: 1/80, 1/80, 1/79) / +0.090 [+0.072, +0.108] (beyond orthogonal +0.083 [+0.064, +0.102]; beyond replay +0.089 [+0.071, +0.108]) | n/a |
| (extension seeds alone, 3,000,060-3,000,119) | 39 | 28/39 / 3/39 | +0.252 [+0.206, +0.297] | 8/39 [0.11, 0.36] at new goal (controls <= 1/39) / +0.076 [+0.053, +0.099] | n/a |
**DEPLOYABLE-ROUTE RESULT (01:00, arm counterpart of legged D-071):** on the deployable latent route (system i flow
-> packet -> system 0, no oracle, no BC anywhere at runtime), the packet generated from an EDITED public context
causally redirects behaviour on parm6_tf3 (41 feasible dev seeds, noise-keyed pairs):
- goal_shift: the cube ends at the NEW goal in 15/41 episodes vs 0-1/41 under all three matched controls; end-position
  preference beyond the irrelevant edit +10.4 cm [7.7, 13.1], beyond the orthogonal edit +9.4 cm [6.5, 12.1], beyond a
  flow-noise replay +9.4 cm [6.5, 12.2]. Old-zone successes drop 12/41 -> 0/41.
- rebind_desc (only the task entity's descriptor + public binding changed): the arm first approaches the NEW cube in
  31/41 (control 0/41) and first touches it in 28/41 (control 3/41); min-distance preference beyond the irrelevant edit
  +21.1 cm [17.5, 24.8]. The new cube is lifted in only 5/41 and placed in the zone 1/41, so this is redirection of the
  approach, not task completion.
- Controls do not move behaviour: irrelevant edit +0.0 [-2.2, +2.3] cm goal effect, orthogonal +0.4 [-1.6, +2.5],
  noise replay +0.4 [-2.2, +2.9]; control successes 12/41, 11/41, 15/40, 10/40.
POOLED (02:10, with the lead's 39-seed extension; D-075): the effect replicates on new seeds. The goal effect is
SMALLER on the extension seeds: 8/39 at the new goal (vs 15/41 originally), end-position effect beyond the irrelevant
edit +7.6 cm [5.3, 9.9] (vs +10.4 [7.7, 13.1]). The CIs overlap, so this is most likely sampling variation plus the
lower control competence there (the extension has fewer transports: cube lifted 23/39 under goal_shift). Pooled: goal
23/80 [0.20, 0.39] at the new goal vs 1/80 for each control, effect +9.0 cm [7.2, 10.8]. Rebind: first touch new 56/80
vs 6/80, original cube never lifted (0/80 vs 58/80 unedited), min-distance effect +23.1 cm [20.2, 26.1]; the new cube
is lifted in only 7/80. Raw extension rows: `artifacts/runs/acceptance_sprint_sem_gen_jf_parm6_ext/shard{0..5}`
(lead leases lead_semgen_ext_s0..5); summaries `semantic_summary_generated_{ext,pooled}.json` (scripts/sem_merge.py).
Honest reading: this is the first deployable-route evidence of PACKET-LEVEL CAUSAL CONTROL by task semantics (goal and
binding) beyond matched controls. It is NOT a semantic-supervision claim: this bundle has no nosem counterpart yet, and
a nosem flow could route the same information. The rebinding may act through the binding pointer or through the public
predicate estimates that follow the binding (e.g. distance(gripper, cube)); both are part of the valid edited context.
Contrast: the plain BC controller given the same edited context ignores the rebinding (0/32). One body (parm6_tf3);
panda_pg2 is not competent on this route (D-063: 0/30).
Raw: `artifacts/runs/acceptance_sprint_sem_gen_jf_parm6/shard{0..5}/semantic_rows_generated.jsonl`, summary
`semantic_summary_generated.json` (host lease 1790402031_9d384e, shard 0 cut at 34 rows by the host watchdog, rc -10,
seed 3000005 partial; peer leases 1790402041_2fa744 .. 1790402043_e55765).

Reading of the other rows: the metrics and edits are valid (teacher ~100%). A valid GOAL edit redirects placement
through every competent route: the deployable one above, the best oracle latent route (16/48 vs 0/48) and BC (24/32).
The supplied BINDING is ignored by BC (0/32) and by the oracle routes whose demo does not follow it. The deployable
flow does follow it at the approach level (above). v4 sem vs nosem (oracle rung) shows no semantic advantage: both
system 0s barely reach objects. The arm-assignment edit has no detectable effect on v4.

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
- 19:25 host routing (lead request): the host broker refuses every admission (`AdmissionStopped: disk_below_reserve:321604132864`,
  the watchdog shed state; even `--cpu 1 --disk 10M`). I do not bypass the broker, so the shards stay on the peer until
  the host reserve is fixed (lead: lower the disk reserve or free disk space).

### BC reference result (canonical pick_place, 32 seeds 3,000,000+, panda_pg2, 400 steps) -- completed
Raw: `artifacts/runs/acceptance_sprint_sem_bc_pp/shard{0..3}/semantic_rows_bc.jsonl`, summary
`semantic_summary_bc.json` (peer leases 1790389070_139486, _155de8, 1790389071_9e7868, _77ba5c). Teacher rung on the
same scenes: `acceptance_sprint_sem_teacher_pp` (rebind_desc: approach + touch new 32/32; goal_shift 32/32).
| condition (source learned:direct1701_u12000, BC reference) | cube in zone | first approach new | new cube lifted / in zone | cube at new goal | min-dist pref. vs control (m) |
|---|---|---|---|---|---|
| control | 28/32 | 0/32 | 0 / 0 | 0 | - |
| rebind_desc (VALID binding edit: descriptor + public binding only) | 25/32 | **0/32** | 0 / 0 | 0 | +0.006 [+0.002, +0.014] |
| rebind_obj (belief edit: tracks of cube and distractor0 exchanged) | 0/32 | 32/32 | 29 / 11 | 0 | +0.301 [+0.273, +0.330] |
| goal_shift (VALID: zone belief moved 12 cm) | 1/32 | 0/32 | 0 / 0 | **24/32** | end-pos. pref. +0.173 [+0.147, +0.194] |
| irrelevant_distractor | 27/32 | 0/32 | 0 / 0 | 0 | +0.001 [-0.000, +0.002] |
| control_replay (other flow noise) | 26/32 | 0/32 | 0 / 0 | 0 | +0.009 [+0.003, +0.015] |
Reading: the competent BC controller follows the goal edit and the belief swap (where the bound slot's object IS),
but IGNORES a valid rebinding of the task entity: with the descriptor and public binding pointed at the other cube it
still picks the original cube 25/32 (effect vs irrelevant edit +0.6 cm [0.1, 1.3]). The training data never varied the
binding (the task object is always the canonical slot/descriptor), so BC learned "slot 0", not "the bound entity".
This is the reference the latent binding result (v4 sem/nosem, trained on binding-varying paired data) is compared with.
Paired scenes: BC is not competent at all (permuted slots/colours; running on host shards, record only).

### 22:05 runs launched (v4 bundles, SPRINT BEST ROUTE)
(The v4 representations were ready at 20:26; my watcher missed it, so these started at 22:02.)
- v4 sem / nosem oracle, paired semantic suite, host leases 1790398887_{5f8ac6,a0cd67,ec2c75} (sem) and
  {d92e93,1c09ed,0ec73a} (nosem), 1 CPU each: `rrp latent semantic-edits --route oracle --scene paired --representation artifacts/runs/binding_paired_{sem,nosem}_v4/representation.pt --episodes 10 --seed-start 31000{00,10,20} --max-steps 300 --out artifacts/runs/acceptance_sprint_sem_v4{sem,nosem}_oracle/shard<i>`
  (orthogonal control uses the jointly trained probe for both; no probe_posthoc exists for v4).
- v4 sem / nosem oracle, arm edits, peer leases 1790398893_0e8623 .. 1790398896_a8d96a (one per robot pair):
  `rrp latent arm-edits --route oracle --representation .../binding_paired_{v}_v4/representation.pt --pairs <pair> --episodes 6 --max-steps 400 --conditions control,swap_arm,swap_slots,orthogonal_matched --out artifacts/runs/acceptance_sprint_arm_v4{v}_oracle/<pair>`
- SPRINT BEST ROUTE (ladder, 21:58): R1 stateless oracle = E(chunk of BC direct1701_u12000) -> system 0 of
  `ladder_rz_jointfix_bcdag2`. New `--oracle-expert bc`: the BC chunk is computed for the EDITED context. Canonical
  pick_place, 30 seeds, 5 peer shards (leases 1790399024_a80e4c .. 1790399027_0107c0) ->
  `artifacts/runs/acceptance_sprint_sem_best_orcbc/shard<i>`. Prediction from the BC result: goal edits propagate,
  the rebinding does not (the demo source ignores it). orthogonal_matched here is norm-matched to goal_shift.

### v4 sem vs nosem, oracle rung, paired scenes -- completed 22:04 (host; raw `artifacts/runs/acceptance_sprint_sem_v4{sem,nosem}_oracle/`)
ORACLE DIAGNOSTIC (E(binding_paired_{sem,nosem}_v4) + scripted_teacher demo -> v4 system 0), 30 paired scenes, panda_pg2.
Both v4 system-0s barely reach anything: in the control condition the TCP comes within 6 cm of no object in 22/30 (sem)
and 21/30 (nosem) scenes; 0 lifts, 0 successes anywhere. (Consistent with the ladder finding at 20:45: v4 system 0
copies the joint velocity and largely ignores the packet.)
| bundle | first touch NEW: control / rebind | rebind min-dist effect beyond irrelevant (m) | beyond orthogonal (m) | init-dir effect beyond irrelevant | goal edit |
|---|---|---|---|---|---|
| v4 sem | 5/30 -> 8/30 | +0.037 [-0.001, +0.077] | +0.002 [-0.037, +0.042] | +0.12 [+0.05, +0.20] | no transport; no effect |
| v4 nosem | 7/30 -> 4/30 | +0.052 [+0.020, +0.082] | +0.040 [+0.010, +0.071] | +0.40 [+0.13, +0.70] | no transport; no effect |
The key contrast, sem vs nosem, shows NO semantic advantage at the behaviour level on the oracle route. Both show a
small rebind-directed bias in the initial motion. For sem it cannot be separated from a matched-norm probe-orthogonal
edit, which moves the sem arm a lot: mean TCP deviation 0.39 m vs 0.24 m for the rebind, so the sem system 0 is very
sensitive to off-probe z directions. The nosem bias exceeds both controls. Weak evidence either way: the route is not competent.

### v4 sem vs nosem, arm edits (dual-arm assign dev pairs), oracle rung -- completed (peer; raw `artifacts/runs/acceptance_sprint_arm_v4{sem,nosem}_oracle/<pair>/arm_rows_oracle.jsonl`, merged `arm_summary_oracle.json`)
3 robot pairs x 6 seeds = 18 per condition, 400 steps, ORACLE DIAGNOSTIC E(v4)+teacher demo -> v4 dual system 0.
| bundle | edited-to arm touches bar first: control / swap_arm / swap_slots / orthogonal | no arm touches the bar (control) | swap_arm bar-pref. vs control (m) | swap_arm beyond orthogonal (m) |
|---|---|---|---|---|
| v4 sem | 4 / 6 / 5 / 3 of 18 | 10/18 | -0.031 [-0.087, +0.030] | +0.001 [-0.072, +0.070] |
| v4 nosem | 4 / 2 / 3 / 2 of 18 | 14/18 | +0.002 [-0.039, +0.040] | +0.006 [-0.025, +0.037] |
No detectable arm-assignment effect for either bundle: which arm approaches the bar does not follow the actor
rebinding or the slot swap beyond the matched orthogonal control. (Teacher reference: 24/24.) Again limited by the
route: the arms reach the bar in only 4-8 of 18 control episodes.
- BC on paired scenes (host leases 1790389569_{49db16,0bf376}; raw `artifacts/runs/acceptance_sprint_sem_bc_paired/`),
  12 scenes: 2/12 successes in control (out of distribution). Rebinding changes nothing: first touch new 3/12 vs 4/12 in
  control, min-dist effect +0.003 m [-0.002, +0.007], TCP deviation from control 1.4 cm (vs 11.8 cm for a noise replay).
  So the BC controller is insensitive to the binding on paired scenes too.
- 22:40 INCIDENT (mine): the launch loop for best-route shards 8/9 retried `ops run --detach` with stderr hidden and
  grepped stdout for "lease_id". The lease JSON never matched, so the loop relaunched shard 8 every 30 s (~20 copies,
  ~24 GiB host memory), which blocked the legged track. The lead killed the loop and scoped-stopped the copies. I stopped the
  last copy, moved the polluted `shard8/` out of the results (not used), and relaunched shard 8 once into `shard8b`
  (lease 1790400179_8f8ec1), checking the exit code. Shard 9 was refused (host memory cap) and is not retried
  automatically. Rule for myself: launch once, check rc, no unbounded retry loops.

### SPRINT BEST ROUTE (ladder 21:58): stateless BC-expert oracle -> system 0 jfbcdag2, canonical pick_place -- 48 seeds done
ORACLE DIAGNOSTIC: packet = E(ladder_rz_jointfix_bcdag2)(edited public context, chunk that BC direct1701_u12000 emits
for the EDITED context at the current state) -> frozen system 0 jfbcdag2. Host leases 1790399070_{fac25e,80c73e,574d0a,
e711d4,4fbdbb} + 1790399462_{050d36,66944c,10d591}; raw `artifacts/runs/acceptance_sprint_sem_best_orcbc/shard{0..7}/`,
summary `semantic_summary_oracle.json` (seeds 3,000,000-3,000,047; shard8b running for 6 more).
| condition | cube in zone (success) | cube at NEW goal | first approach old / new / none | effect beyond irrelevant edit |
|---|---|---|---|---|
| control | 27/48 | 1/48 | 48 / 0 / 0 | - |
| goal_shift (VALID) | 1/48 | **16/48** | 48 / 0 / 0 | end-pos. pref. +0.104 m [+0.081, +0.127]; beyond orthogonal +0.074 [+0.050, +0.097] |
| rebind_desc (VALID) | 2/48 | 0 | 6 / 4 / 38 | min-dist pref. +0.163 m [+0.130, +0.200], but by ABANDONING the old cube, not by approaching the new one (new first 4/48 vs 0/48) |
| irrelevant_distractor | 27/48 | 0 | 48 / 0 / 0 | +0.004 [+0.001, +0.009] |
| orthogonal_matched (norm = goal edit) | 2/48 | 0 | 44 / 0 / 4 | success collapses 27 -> 2 (system 0 is fragile to off-probe z) |
| control_replay (other BC noise) | 23/48 | 0 | 48 / 0 / 0 | +0.001 [-0.008, +0.010] |
Reading: through the best (oracle) latent route, a valid GOAL edit in the packet's context causally redirects placement
(16/48 at the new goal vs 0/48 under matched-size controls). A valid REBINDING disrupts the approach but does not
redirect it. Its demo source, BC, ignores the binding (above), so the packet mixes a new binding with a demo toward the
old cube. Caveat: a probe-orthogonal edit of the same norm also destroys success (2/48), so the goal effect is specific in
direction but system 0 is not robust to packet perturbations.
- Best-route videos (seed 3000009, where control succeeds and the goal edit places at the new goal):
  `artifacts/video/2026-09-25_semantic_edit_{goal_shift,rebind_desc,orthogonal_matched}_oracle_bcexpert_ladder_rz_jointfix_bcdag2_panda_pg2_k3000009.mp4`
  (the rebind_desc and orthogonal videos show FAILURES: the old cube is abandoned or dropped).

### generated (DEPLOYABLE) route, lead request 22:45: learned:ladder_flow_jointfix/snap_final_s20000 -> system 0 ladder_rz_jointfix_gendag1_noqd, parm6_tf3
`rrp latent semantic-edits --route generated --checkpoint artifacts/runs/ladder_flow_jointfix/snap_final_s20000.pt --representation artifacts/runs/ladder_rz_jointfix_gendag1_noqd/representation.pt --robots parm6_tf3 --episodes 10 --seed-start 30000{0,1,..,5}0 --max-steps 400 --conditions control,goal_shift,rebind_desc,irrelevant_distractor,orthogonal_matched,control_replay --out artifacts/runs/acceptance_sprint_sem_gen_jf_parm6/shard<i>`
(new `--representation` override for the generated route = the system-0 bundle, same latent space; as in
rrp.evaluation.ladder). 60 seeds, of which parm6_tf3 finds ~half feasible. Shard 0 on host (lease 1790402031_9d384e,
1.5 GB, measured RSS 0.8 GB); shards 1-5 on the peer (host memory cap full; leases 1790402041_2fa744 .. 1790402043_e55765).
Flow noise keyed per (seed, call); control_replay = other noise. Scenes use n_distractors = max(1, seed % 3) (the
ladder uses seed % 3), so the control rate is not the same as D-063's 9/30.
- 01:05 same deployable-route suite on panda_pg2 (route 0/30 there, D-063; approach-level effects are still
  measurable): host leases 1790412063_{b69980,23c4cb,047183,2805b6,7df3b6,246cd0}, 48 seeds, 1.5G each, one launch each
  -> `artifacts/runs/acceptance_sprint_sem_gen_jf_panda/shard<i>`. Videos of parm6_tf3 seed 3000052 (control succeeds,
  goal edit places at the new goal, rebind touches + lifts the new cube) are rendering on the peer.
