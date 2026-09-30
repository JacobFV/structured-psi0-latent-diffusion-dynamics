# functional composition at the runtime level (support_insert)

Status: **verified** for the scripted privileged teacher (`scripted_teacher`, teacher v2 per D-019).
The teacher-v1 run is archived under `artifacts/assets/functional_composition/archive_teacher_v1/`; its
conclusions are the same.
No learned policy is involved; this shows that the task runtime + teacher implement
data-dependent composition, not only a sequence. Raw outputs:
`artifacts/assets/functional_composition/parm5_pair.json`,
`artifacts/assets/functional_composition/panda_ur5e_pair.json`.
Command: `python -m rrp.control.functional_composition --pair <pair> --seeds 0:12 --out <json>`
(host lease, 1 core).

## what is composed

`tasks/support_and_insert.json`: `locate` (camera observes the hole) produces
`hole_frame : frame_estimate`; `support` (left arm) maintains `anchor : contact_anchor` while
active; `align` binds `reference = locate#0.hole_frame`, `support = support#0.anchor` and has
`frame_binding = locate#0.hole_frame`; `insert` uses the same frame binding while `support`
stays active (`requires_active`).

Runtime facts (all public):
* the hole frame is produced by a declared static-feature estimator
  (`static_feature_average/v1`: mean of >= 12 noisy detector measurements; sigma 4 mm per
  detection), never from simulator truth (`tests/unit/test_dual_public_estimators.py`);
* the hole-relative estimates (`lateral_error_m`, `axis_error_rad`, `insertion_depth_m`) are
  evaluated in the frame of the receipt bound to the consuming event (`frame_binding`), not in
  "the latest frame of that type";
* the teacher places the peg using that same bound receipt. Its only privileged inputs are the
  peg pose (for grasping) and the true in-hand offset.

The hole position is randomized per seed (fixture pose x/y/yaw plus the hole offset inside the
fixture), so the upstream output is needed.

## conditions (same seed, same scene)

| condition | what changes | parm5_pg2 + parm5_pg2 (12 seeds, 10 feasible) |
|---|---|---|
| receipt | nothing: the normal run | 10/10 physical successes, 10/10 public runtime successes |
| shifted | the locate estimator OUTPUT is offset by (+20, -15, 0) mm before it becomes a receipt | 0/10 physical, 0/10 public |
| fixed_order | same supplied stage order and runtime gating, but the hole frame is a fixed nominal constant | 0/10 physical, 0/10 public (align times out) |
| stale_receipt | locate receipt invalidated right after publication | 0/10; align never activates: `binding_unavailable` |

2 of 12 seeds were rejected before execution by the teacher's IK feasibility check
(`infeasible:right.grasp`) in every condition. They count as attempted but not feasible.

**Paired command shift.** For every feasible seed, the final commanded align TCP goal moved
by exactly the injected receipt offset. `|Δgoal - Δreceipt| <= 0.2 mm` on 10/10 seeds; the
residual comes from the slightly different in-hand pose between the two runs. The insert
command follows the same bound frame. Downstream commands are therefore a function of the
upstream event output, not of the stage order.

**Fixed order is not composition.** With the supplied order and all runtime gates in place, a
constant hole frame never satisfies `align` completion: its lateral error is measured against
the bound receipt, so the runtime refuses to start `insert` and the attempt times out and
retries. The failure reason is kept in the rejection history.

**Provenance.** With the receipt invalidated, `align` stays `pending(binding_unavailable)`.
The runtime does not fall back to some other frame.

## second pair (menagerie panda_pg2 + ur5e_pg2)

| condition | panda_pg2 + ur5e_pg2 (10 seeds, all feasible) |
|---|---|
| receipt | 10/10 physical, 10/10 public |
| shifted | **0/10 physical, but 8/10 public "successes"** (public insertion-depth false positives, see below) |
| fixed_order | 0/10 physical, 0/10 public (align times out and retries) |
| stale_receipt | 0/10; align `pending(binding_unavailable)` |

Paired command shift: the final align goal moved by the injected receipt offset within 0.2 mm on
9/10 seeds. Seed 7 had a 38 mm residual; its last recorded align goal belongs to a different
point of the approach, and this is not investigated further.

On this stronger arm pair, the peg jams on the fixture top at the wrong location and slips in
the grasp faster than the public in-hand belief (12-detection window) can follow. The public
`insertion_depth_m` then crosses 20 mm, the runtime marks `insert` as succeeded and the teacher
releases, while the privileged evaluator records a failure. This is the public-estimator
limitation recorded in D-018: dataset labels use the privileged evaluator, and
`public_runtime_success` is stored separately. The proposed fix, not implemented, is the
conservative minimum of the depth from direct detection and the FK-based depth.

## estimator issue found during this experiment (fixed)

In the first version of the public in-hand pose belief, the running mean covered up to 50
detections. In the `shifted` condition the peg jammed on the fixture surface and slipped in
the grasp. The stale in-hand belief then made the public `insertion_depth_m` report a
successful insertion that had not physically happened. That run's output was overwritten and
is not used for any number here. The window was reduced to 12 detections
(`INHAND_WINDOW` in `src/rrp/sim/dual.py`). With the shorter window, public and privileged outcomes
agree in every condition for the parm5 pair. They still disagree for the panda pair (above), so
the window change reduces the problem but does not remove it. The disagreement rate between public and
privileged success is also tracked per pair in the teacher validation
(`artifacts/assets/dual_teacher_validation/*.summary.json`, field `agree`).

## limits

* Teacher-level demonstration only. Whether a *learned* policy uses the frame causally needs
  the same receipt edits on policy rollouts. Probes and attention maps do not count as that
  evidence.
* The hole axis is declared (fixture resting on the table); only the position is estimated.
