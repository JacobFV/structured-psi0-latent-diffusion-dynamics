# dual-arm tasks: support_insert and handover (scenarios, estimators, teachers, data)

Track owner: dual-arm engineer. Everything here comes from the **scripted privileged teacher**
(`scripted_teacher`). No learned-policy result is reported. Raw outputs are under
`artifacts/assets/dual_teacher_validation/` and `artifacts/assets/functional_composition/`.

## scenarios (`src/rrp/sim/dual_scenarios.py`)

* Two manipulators face the table. They are either **two separately mounted robots** (left at
  (0, +0.28), right at (0, -0.28), both facing +x) or **one dual-arm body** (menagerie ALOHA,
  mounted at the table centre) whose two gripper assemblies are bound to the roles `left` and
  `right`. The role bindings are public; the entity-to-sim-body map is privileged.
* `support_insert`: a free-floating 220 x 220 x 45 mm fixture block (1.3 kg) with a blind
  square hole and an upright cylindrical peg. Randomized per seed: fixture x, y and yaw
  (+-0.3 rad), hole offset inside the block, support-pad offset, and peg position. The hole
  position therefore changes every seed.
  **Tolerance:** peg diameter 24 mm, hole 28 x 28 mm, i.e. **4 mm diametral clearance
  (2 mm radial along the hole axes, more along the diagonals)**, hole depth 39 mm. This is
  inside the requested 3-5 mm band. Physical success (privileged evaluator): the peg bottom is
  at least 20 mm below the hole top, the lateral offset is smaller than the hole half-width,
  and the fixture tilt is under 0.1 rad.
* `handover`: a 180 x 22 x 40 mm bar lies on the left side with its long axis roughly along y
  (+-0.35 rad), and a target zone is on the right. Grasp points are +-65 mm from the bar
  centre.

## task graphs

* `tasks/support_and_insert.json`: the supplied graph, unchanged.
* `tasks/handover.json` (new): `take(left,bar)` -> `offer(left,bar; maintain_hold; invariant
  held_by(bar,left); produces anchor while active)` || `receive(right,bar; source=left;
  reference=offer#0.anchor; requires_active offer)` -> `release(left)`, which completes when
  held_by(bar,left)=false AND held_by(bar,right)=true -> `place(right,bar,target)`. Both
  held_by states are true at the same time during the overlap, and `offer` and `receive` run
  concurrently on different exclusive resources.

## runtime fixes needed to execute the supplied graph (minimal, backwards compatible)

1. `validated_receipt` predicates other than `event_succeeded` evaluated UNKNOWN. That left
   `align` (precondition `anchor_valid`) and `insert` (precondition `receipt_valid`) stuck in
   `precondition_unknown` forever. `anchor_valid`, `receipt_valid` and `frame_valid` now
   evaluate the latest receipt of the bound output; a missing receipt counts as known-invalid.
2. Retry rebinding (`rebind_output`) happened only when an output had role-binding consumers.
   Outputs consumed only through guard conditions or `frame_binding` (such as
   `align#k.aligned` and `locate#k.hole_frame`) kept pointing at the failed attempt. The
   rebind edit now also moves `frame_binding`.
3. Maintained outputs whose value was unchanged were re-issued as a new receipt version on
   every tick, because the stored value had `covariance_diag` removed and the comparison did
   not account for that. The comparison now matches the stored form.
4. `StepResult.commands` records the executed groups of every robot (`command` is still
   robot 0 only).

The existing tests `tests/integration/test_native_sim.py`, `tests/unit/test_task_runtime.py`
and `tests/unit/test_interventions.py` pass (31 tests).

## public estimators (`src/rrp/sim/dual.py`, `DualSession`)

These use only detector measurements, FK of measured joints, touch/width sensors, issued
gripper commands and declared task geometry:

| predicate | estimator |
|---|---|
| frame_estimate_valid(hole) | static-feature average of >= 12 noisy detections (sigma 4 mm each); restarts if the feature jumps |
| held_by(obj, arm) | touch on >= 2 pads, closed command, object track within 10 cm of the FK TCP; 0.25 s release hysteresis |
| reachable(arm, obj) | tracked object inside 0.9 x declared reach of that arm's base |
| lateral_error_m / axis_error_rad / insertion_depth_m / inside (peg, hole) | peg bottom from FK TCP + in-hand offset belief (running mean over ~12 detections in the TCP frame) + declared peg length, measured in the frame of the receipt bound by `frame_binding` |
| supported(fixture) | some manipulator touching and pressing within the fixture footprint while holding nothing |
| contact_anchor output | TCP at touch-down; invalid after 0.3 s without contact or after more than 12 mm of slip |

Leakage guard: `tests/unit/test_dual_public_estimators.py`. Corrupting the privileged layout
and the entity map leaves every public estimate unchanged, and the hole receipt lies within
6 mm of the true hole without being equal to it.

Known public-estimator limitation: vision gives the peg position, not its orientation. If a
three-finger hand lets the peg tilt, the public estimate can report an insertion that did not
happen. The per-pair `agree` count below measures this.

## teachers (`src/rrp/control/dual_teachers.py`, label `scripted_teacher`, privileged)

The TCP-waypoint FSMs per arm use DLS IK, a rate-limited TCP command and integral correction.
Both teachers first move to a staging pose on each arm's own side, and they are gated only on
**public runtime statuses and receipts**. support_insert: the left arm descends slowly onto a
support pad, stops on touch and presses 6 mm with the integral term off, then holds until
`insert` succeeds. The right arm stages, grasps the peg (three-finger hands grasp 40 mm below
the top, parallel hands 20 mm), lifts, waits until `align` is active, makes a high transit and
a vertical approach to the **bound locate receipt**, aligns 12 mm above the hole, inserts to
32 mm when `insert` is active, releases and retreats. handover: the giver takes the bar at one
end, presents it at (0.40, 0, 0.20) and holds until `release` is active. The receiver grasps
the other end while `offer` is active and waits for `release` to succeed before placing.
Three-finger hands rotate 90 degrees for bar grasps so that no finger lands on the bar axis.

Teacher v2 (D-019): the commanded tool orientation is slewed from the current tool pose to the
nearest symmetric equivalent of the target at <= 1.5 rad/s, and joint targets are slewed at
<= 3 rad/s. Demonstrations therefore contain no wrist snaps or flips.

Before execution, an IK feasibility check over the key waypoints records infeasible layouts
as `infeasible:<arm>.<waypoint>`. Those layouts are not attempted.

## teacher validation (30 seeds per pair, seeds 0-29)

**support_insert, teacher v2** (`support_insert_30seeds_v3.jsonl`; pair = left support __ right insertion). The success rate is over feasible layouts; `agree` counts episodes where public runtime success equals privileged physical success.

| pair | split role | n | success | failure | infeasible | success/feasible | agree | top non-success reasons |
|---|---|---|---|---|---|---|---|---|
| aloha | target: dual-arm body | 30 | 30 | 0 | 0 | 1.00 | 30/30 |  |
| panda_pg2__ur5e_pg2 | source (eligible) | 30 | 30 | 0 | 0 | 1.00 | 30/30 |  |
| parm5_pg2__parm5_pg2 | source (eligible) | 30 | 27 | 0 | 3 | 1.00 | 30/30 | infeasible:right.grasp x3 |
| parm5_tf3__parm7_pg2 | source (eligible) | 30 | 21 | 2 | 7 | 0.91 | 30/30 | ended_in_phase:L:l_hold|R:r_descend x1; ended_in_phase:L:l_hold|R:r_lift x1; infeasible:right.grasp x7 |
| parm5l_pg2__parm5s_tf3 | source held-out eval | 30 | 14 | 15 | 1 | 0.48 | 16/30 | ended_in_phase:L:l_done|R:r_done x7; ended_in_phase:L:l_hold|R:r_align x5; ended_in_phase:L:l_hold|R:r_approach x1 |
| parm6_pg2__panda_tf3 | target: panda+tf3 insertion arm (teacher-weak) | 30 | 8 | 22 | 0 | 0.27 | 23/30 | ended_in_phase:L:l_done|R:r_done x1; ended_in_phase:L:l_hold|R:r_align x3; ended_in_phase:L:l_hold|R:r_approach x7 |
| parm6_pg2__parm7_tf3 | source (eligible) | 30 | 15 | 7 | 8 | 0.68 | 29/30 | ended_in_phase:L:l_done|R:r_done x1; ended_in_phase:L:l_hold|R:r_approach x3; ended_in_phase:L:l_hold|R:r_descend x1 |
| parm6_tf3__parm6_pg2 | source (eligible) | 30 | 19 | 0 | 11 | 1.00 | 30/30 | infeasible:right.grasp x11 |
| parm7_pg2__parm5_tf3 | source (eligible) | 30 | 16 | 11 | 3 | 0.59 | 23/30 | ended_in_phase:L:l_done|R:r_done x6; ended_in_phase:L:l_hold|R:r_approach x2; ended_in_phase:L:l_hold|R:r_descend x1 |
| sawyer_pg2__panda_pg2 | source (INELIGIBLE) | 30 | 1 | 29 | 0 | 0.03 | 30/30 | ended_in_phase:L:l_hold|R:r_abort x1; ended_in_phase:L:l_hold|R:r_align x23; ended_in_phase:L:l_hold|R:r_approach x5 |
| sawyer_tf3__ur5e_pg2 | source (eligible) | 30 | 22 | 8 | 0 | 0.73 | 30/30 | ended_in_phase:L:l_hold|R:r_abort x1; ended_in_phase:L:l_hold|R:r_align x7 |
| ur5e_pg2__panda_tf3 | target: panda+tf3 insertion arm (teacher-weak) | 30 | 3 | 27 | 0 | 0.10 | 25/30 | ended_in_phase:L:l_done|R:r_done x3; ended_in_phase:L:l_hold|R:r_align x2; ended_in_phase:L:l_hold|R:r_approach x5 |
| ur5e_pg2__sawyer_pg2 | source (eligible) | 30 | 27 | 3 | 0 | 0.90 | 30/30 | ended_in_phase:L:l_hold|R:r_transit x3 |
| ur5e_tf3__parm6_pg2 | source (INELIGIBLE) | 30 | 0 | 19 | 11 | 0.00 | 30/30 | ended_in_phase:L:l_hold|R:r_wait x19; infeasible:right.grasp x11 |
| xarm7_pg2__xarm7_pg2 | target: held-out family | 30 | 30 | 0 | 0 | 1.00 | 30/30 |  |
| xarm7_tf3__xarm7_pg2 | target: held-out family | 30 | 30 | 0 | 0 | 1.00 | 30/30 |  |

**handover, teacher v2** (`handover_30seeds_v3.jsonl`; pair = left giver __ right receiver).

| pair | n | success | failure | infeasible | success/feasible | agree | top non-success reasons |
|---|---|---|---|---|---|---|---|
| aloha | 30 | 30 | 0 | 0 | 1.00 | 30/30 |  |
| panda_pg2__ur5e_pg2 | 30 | 30 | 0 | 0 | 1.00 | 30/30 |  |
| parm5_pg2__parm5_pg2 | 30 | 30 | 0 | 0 | 1.00 | 30/30 |  |
| parm5_pg2__parm6_tf3 | 30 | 30 | 0 | 0 | 1.00 | 30/30 |  |
| parm6_pg2__parm7_pg2 | 30 | 24 | 1 | 5 | 0.96 | 30/30 | ended_in_phase:L:l_descend|R:r_wait x1; infeasible:left.grasp x5 |
| parm7_tf3__parm5_pg2 | 30 | 27 | 0 | 3 | 1.00 | 30/30 | infeasible:left.grasp x3 |
| sawyer_pg2__panda_tf3 | 30 | 0 | 30 | 0 | 0.00 | 30/30 | ended_in_phase:L:l_hold|R:r_pre x30 |
| ur5e_pg2__sawyer_pg2 | 30 | 30 | 0 | 0 | 1.00 | 30/30 |  |
| xarm7_pg2__xarm7_pg2 | 30 | 30 | 0 | 0 | 1.00 | 30/30 |  |

Teacher v1 results are superseded and kept for provenance: `support_insert_30seeds_final.jsonl`
and `handover_30seeds{,_v2}.jsonl`. Teacher v2 (D-019) slews the tool orientation from the current
pose and limits joint rate. That removed the wrist snaps and flips, and it changed several rates:
ALOHA support_insert went from 22/30 to 30/30, ALOHA handover from 0/30 to 30/30,
parm6_pg2__parm7_tf3 from 10/22 to 15/22, and sawyer_tf3__ur5e_pg2 from 0/30 to 22/30.
Earlier diagnoses written for v1 (for example "ALOHA servo sag") were wrong about the cause.

Remaining teacher failures:
* A sawyer support arm next to a panda insertion arm fails (1/30): the insertion arm stalls in
  `r_align`/`r_approach`, consistent with the two bodies interfering.
* ur5e+tf3 as the support arm never yields a valid anchor, so `align` is never activated
  (0/19 feasible).
* panda+tf3 as the insertion arm drops or tilts the peg (3/30 and 8/30).
* sawyer giver + panda_tf3 receiver never reaches the receive pose in handover (0/30).
All of these are recorded, not hidden. The public/privileged disagreements come from the tilted
peg case (three-finger insertion arms).

## ALOHA (menagerie `aloha`, one dual-arm body)

Imported by `src/rrp/morphology/aloha.py` as one Module with four assemblies (left/right arm
and gripper) and four controller groups. Declared surgery is listed in `meta['surgery']`:
fingertip-midpoint TCP sites, box touch sites and touch sensors on all four finger links,
finger-joint width sensors, keyframes removed, and the finger equality leader swapped to the
actuated finger (the coupling is symmetric with identical physics; without the swap the
compiler rejected the model because an actuator drove a mimic joint). The assets were synced
to the peer (`.cache/assets/mujoco_menagerie/aloha`).
Status, with the stages of docs/04 used literally: imported and physics-validated (the body
compiles in the scenario; the arms hold home; the fingers, touch and width sensors report).
**Teacher-validated for both tasks with teacher v2: support_insert 30/30 and handover 30/30.**
Support posture: fully closed ALOHA fingers touch each other, which reads as contact on the
finger touch sensors and would create a false support anchor, so the support posture closes to
12 mm. The low servo gains (waist kp 43) make joint tracking lag larger than on the other
arms. That is visible in the action magnitudes (max normalized one-step action 1.23) but no
longer causes failures.

## datasets

**Current: `artifacts/datasets/support_insert_primary_v2`** (teacher v2, 1.5 GB). Generated on the
host under a broker lease, because the peer was saturated by other tracks for hours
(admission refused on CPU and memory). A copy was pushed to the peer at
`/dev/shm/rrp-brandonin/repo/artifacts/datasets/support_insert_primary_v2`. Config:
`configs/data/support_insert_primary_v2.json`. Split: `research/splits/primary_v1_support_insert.json`
(frozen before any data; eligibility re-resolved for teacher v2 in D-019). Command:
`python -m rrp.data.collect_dual --config configs/data/support_insert_primary_v2.json`.
The format matches `rrp.data.collect`: `*.public.pkl.gz` holds MultiFeaturizer inputs
(feat-multi-v1+feat-v2: node/morph dim 44, scene dim 27), flat namespaced actions (`r<i>:<group>`),
q0, public runtime statuses and the flat action space. `*.private.pkl.gz` holds per-manipulator
privileged labels, teacher phases, true insertion geometry and the privileged layout. Episode
`status` comes from the privileged evaluator; `public_runtime_success` is stored separately.
Failed and infeasible attempts are kept. The data loads unchanged with
`rrp.learning.data.load_episodes` / `ChunkDataset`
(`artifacts/receipts/dual/support_insert_v2_loader_check.txt`). Split/lineage check:
`artifacts/receipts/dual/support_insert_v2_split_lineage_check.txt` found no ufactory, panda+tf3,
fr3, trossen or held-out-source lineage in the source_train episodes.

| pair | split | episodes | success | failure | infeasible | other | mean steps (success) |
|---|---|---|---|---|---|---|---|
| parm5l_pg2__parm5s_tf3 | source_heldout_eval | 60 | 33 | 24 | 3 | 0 | 783 |
| panda_pg2__ur5e_pg2 | source_train | 150 | 150 | 0 | 0 | 0 | 265 |
| parm5_pg2__parm5_pg2 | source_train | 150 | 135 | 6 | 9 | 0 | 192 |
| parm5_tf3__parm7_pg2 | source_train | 150 | 104 | 6 | 40 | 0 | 201 |
| parm6_pg2__parm7_tf3 | source_train | 150 | 74 | 32 | 44 | 0 | 201 |
| parm6_tf3__parm6_pg2 | source_train | 150 | 91 | 7 | 52 | 0 | 195 |
| parm7_pg2__parm5_tf3 | source_train | 150 | 71 | 64 | 15 | 0 | 302 |
| sawyer_tf3__ur5e_pg2 | source_train | 150 | 103 | 47 | 0 | 0 | 265 |
| ur5e_pg2__sawyer_pg2 | source_train | 150 | 141 | 9 | 0 | 0 | 228 |
| xarm7_pg2__xarm7_pg2 | target:held_out_arm_family | 60 | 60 | 0 | 0 | 0 | 208 |
| xarm7_tf3__xarm7_pg2 | target:held_out_arm_family | 60 | 60 | 0 | 0 | 0 | 208 |
| parm6_pg2__panda_tf3 | target:held_out_attachment_combination_on_insertion_arm | 60 | 14 | 46 | 0 | 0 | 863 |
| ur5e_pg2__panda_tf3 | target:held_out_attachment_combination_on_insertion_arm | 60 | 9 | 51 | 0 | 0 | 951 |
| aloha | target:held_out_dual_arm_body | 60 | 60 | 0 | 0 | 0 | 276 |

total 1560 {'failure': 292, 'infeasible': 163, 'success': 1105} manifest_hash 246f1a54d1d96f32

Seeds: source 0-149 per pair (0-29 overlap the teacher-validation scenes; demonstrations only),
targets 1,000,000+, source held-out development 3,000,000+. Episodes run up to 1200 control
steps at 20 Hz. The long mean lengths of the teacher-weak targets reflect failures that ran to
the step cap.

Superseded: `artifacts/datasets/support_insert_primary_v1` (teacher v1 with wrist snaps; 1260
episodes, 910 successes, manifest_hash 2181bf6728ab7a6d) is kept on the host for provenance
only and has been removed from the peer. Do not mix it with v2.
