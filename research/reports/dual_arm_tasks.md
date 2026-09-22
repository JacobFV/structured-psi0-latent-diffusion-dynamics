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

Before execution, an IK feasibility check over the key waypoints records infeasible layouts
as `infeasible:<arm>.<waypoint>`. Those layouts are not attempted.

## teacher validation (30 seeds per pair, seeds 0-29)

SI_TABLE_PLACEHOLDER

HANDOVER_TABLE_PLACEHOLDER

## ALOHA (menagerie `aloha`, one dual-arm body)

Imported by `src/rrp/morphology/aloha.py` as one Module with four assemblies (left/right arm
and gripper) and four controller groups. Declared surgery is listed in `meta['surgery']`:
fingertip-midpoint TCP sites, box touch sites and touch sensors on all four finger links,
finger-joint width sensors, keyframes removed, and the finger equality leader swapped to the
actuated finger (the coupling is symmetric with identical physics; without the swap the
compiler rejected the model because an actuator drove a mimic joint). The assets were synced
to the peer (`.cache/assets/mujoco_menagerie/aloha`).
ALOHA_PLACEHOLDER

## datasets

DATASET_PLACEHOLDER
