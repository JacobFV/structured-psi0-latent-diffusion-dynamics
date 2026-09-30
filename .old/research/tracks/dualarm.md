# track: dualarm — dual-arm tasks on the corrected latent-packet architecture

Owner: dualarm track agent. Branch `track/dualarm`, worktree `~/work/rrp-wt/dualarm`, peer dir
`/dev/shm/rrp-brandonin/wt/dualarm`. Scope: migrate support_insert and handover from the old direct-action path
to the latent packet z[knots=4, assemblies=2, 64] (one assembly per arm; role order + multiplicity are part of the
representation), train sem/nosem representations and flows identically, closed-loop dev evaluation with packet
probes, labelled videos.

## state (2026-09-25, re-scoped by the lead: research/corrections/2026-09-25-causal-semantics-priorities.md, D-037)
- RE-SCOPED to item 2 for manipulators: paired dual-arm tasks with an identical initial scene and a different
  assigned manipulator, valid scripted_teacher demonstrations for both, packed for the binding pipeline, plus a
  pair index for the acceptance track's manipulator-assignment interventions.
- dual-arm latent code path (multi-assembly packet, packing, per-slot probes, closed-loop dual evaluator,
  slot-swap intervention, labelled render): verified by smoke runs, merged to main.
- sem/nosem dual representation+flow chain: STOPPED at re-scope (was ~300/8000 steps of Stage A at ~2 s/step on
  the shared GPU; ~16 h to finish; not close). No dual-arm learned result exists. Resume command if wanted later:
  `scripts/peer_run.sh --gpu --gpu-mem 16G --cpu 6 --mem 32G --label dualarm_chain --max-seconds 21600 --detach -- bash scripts/dualarm_chain.sh`
  after re-packing (`rrp latent pack-dual --config configs/latent/pack-dualarm_latent_v1.json`, ~30 min; the 14 GB
  pack was deleted from the peer to free shared disk).

## design (what the packet means for two arms)
- Packet slot order = order of grasping-assembly tokens in the morph bank = robot order for two-body pairs. The
  dual scenarios mount `left` on r0 and `right` on r1, so slot 0 = left role, slot 1 = right role. A single body
  with two grippers (ALOHA) uses its declared assembly order. Missing assemblies are explicit null slots
  (mask False, handle `asm:0000000000000000:null/<m>`).
- Each slot has an opaque AssemblyHandle `asm:<robot spec hash>:<tcp link>` + robot_index (stable instance
  provenance); packet robot_spec_hash = combined multi-robot hash; graph/runtime versions and observation id as
  for single-arm packets.
- System 0 (`LatentRealizer`) routes every action node ONLY to its own assembly's slot (`node_asm`), and each node
  receives its own assembly's touch/width sensors (per-node local [B,N,4]). Node->slot map is public: relation
  node_in_assembly + kinematic descendants (`rrp.learning.dual_latent.node_assembly_index`); for multi-gripper
  bodies the declared per-node gripper map (`MultiFeaturizer._node_grippers`).
- Probe queries held_by/acting_on/rel_pos/subtask are asked for EVERY slot (was: manipulator 0 only). Privileged
  per-manipulator labels are permuted into slot order through the public role bindings. Per-slot subtask label
  (public) = operator of the first active event whose actor nodes belong to that slot (node_actor_of relations).
  Operators appended: maintain_hold, release (probe n_operators 14 via config `probe`, old checkpoints unchanged).
- Slot-swap control: the same received packet with slots exchanged, scored against the unswapped labels.

## data
- handover had no dataset. Split addendum `research/splits/primary_v1_handover.json` (frozen before any handover
  data/policy result): 6 source-train pairs (teacher v2 success/feasible 0.96-1.00), held-out source pair
  parm5l_pg2__parm5s_tf3, targets xarm7_pg2__xarm7_pg2 / sawyer_pg2__panda_tf3 / aloha (not used here).
- DART for dual teachers: continuous 0.02-0.08 rad arm noise makes the precise dual teachers fail almost always
  (handover 0.02 rad: 1/4; support_insert 0.05: 0/5 on parm5 pair, host smoke). Burst noise (0.04-0.05 rad for 5
  of every 25 control steps) lets the teacher recover (handover 4/5 at 0.05). Used: 0.04 rad, bursts 5/25.
- `configs/data/handover_primary_v1dart.json`: 6 pairs x 120 seeds x {clean, DART}; handover episodes end 10 steps
  after public runtime success (the teacher otherwise idles to 1200 steps).
- `configs/data/support_insert_dart_v1.json`: DART companion of support_insert_primary_v2 (8 source pairs x seeds
  0-99).
- Pack: `configs/latent/pack-dualarm_latent_v1.json` -> `artifacts/packed/dualarm_latent_v1_H16` (successes only,
  incl. DART successes; limits morph 24 / task 32 / N 16 / R 256; multi_m 2).

## log
- 12:20 data jobs (peer CPU, leases 1790364027_cce6a2 / 1790364028_32d9d9):
  `PY -m rrp.data.collect_dual --config configs/data/handover_primary_v1dart.json --workers 6` -> 1215 success /
  133 failure / 92 infeasible (585 s). `... support_insert_dart_v1.json` -> 94 success / 586 failure / 120 infeasible
  (824 s): burst DART at 0.04 rad still breaks the 2 mm-clearance insertion most of the time; only DART successes
  are packed.
- 12:35 host smoke (verified): pack-dual on 2 pairs x 3 seeds -> train-representation 40 steps -> train-flow 40
  steps -> fit-probes 30 steps -> evaluate-dual (support_insert and handover, parm5 pair, 2 episodes x 30 steps):
  packets accepted (0 rejections), per-slot probe metrics and slot-swap control produced, runtime advanced
  (locate succeeded). Scratch outputs only.
- 12:36 real pack running on peer (lease 1790364984_3fe26b):
  `PY -m rrp.cli latent pack-dual --config configs/latent/pack-dualarm_latent_v1.json`.
- 12:37 scripted_teacher reference on the dev scenes (seeds 3000000-3000019; pairs panda_pg2__ur5e_pg2,
  parm5_pg2__parm5_pg2, ur5e_pg2__sawyer_pg2, held-out parm5l_pg2__parm5s_tf3), both tasks:
  `PY -m rrp.cli latent teacher-ref-dual --task <t> --pairs ... --out artifacts/runs/dualarm_teacher_ref/<t>.jsonl`.
- 12:57 teacher reference (scripted_teacher, privileged; dev seeds 3000000-19, max 800 steps;
  raw `artifacts/runs/dualarm_teacher_ref/{support_insert,handover}.jsonl`), success/feasible:
  support_insert panda_pg2__ur5e_pg2 20/20, parm5_pg2__parm5_pg2 19/20, ur5e_pg2__sawyer_pg2 20/20,
  parm5l_pg2__parm5s_tf3 (held-out source) 9/17 (3 infeasible); handover 20/20, 20/20, 20/20, 16/16 (4 infeasible).
- 13:05 pack v1 found truncation (handover receipts accumulate in the interact bank: 4 -> 24+ tokens per episode,
  more under DART), repacked with limits interact 64 / R 1024 / P 128 -> 524,805 rows (si clean 200,151; si DART
  27,917; handover 296,737); 1,199 handover rows (0.4%) still exceed 64 interact tokens and are truncated (tail =
  sensor tokens lost). Collate now drops pointers into truncated tokens (caused a CUDA gather assert).
- 13:40 peer GPU is shared by ~10 processes (load ~41 on 20 cores): Stage A runs ~2 s/step regardless of data
  prefetch (added forked-worker prefetch, `prefetch: true`). Budget cut to 8k steps per stage (was 20k), probes
  3k steps. Chain (lease 1790369312_a8c248, `bash scripts/dualarm_chain.sh`, resumable, 6 h lease cap: relaunch
  the same command to continue).
- 13:55 re-scope received; chain lease 1790369312_a8c248 stopped (owned-only).

## manipulator-assignment pairs (D-037 item 2)
- Family `assign_pick_place`: tasks `assign_left` / `assign_right` (`rrp.sim.dual_scenarios.build_assign`,
  `assign_task(arm)`). One layout per seed (bar + target zone in the shared central workspace; layout RNG
  independent of the assignment). Task graph = handover's take/place events with the actor rebound to the assigned
  manipulator; the other manipulator is declared and bound to nothing.
- Teacher `AssignedPickPlaceTeacher` (scripted_teacher, privileged bar pose / in-hand offset): reads the acting arm
  from the PUBLIC task graph; the other arm stages on its own side and holds.
- Host validation (5 seeds x 3 pairs x 2 assignments, 600 steps): 30/30 privileged successes. Scene identity
  checked (initial physics-state hash equal for both variants) and the privileged held_by label shows only the
  assigned arm holds the bar (seeds 0, 7 on parm5_pg2__parm6_tf3).
- Split addendum `research/splits/primary_v1_assign_pick_place.json`; data config
  `configs/data/assign_pick_place_v1.json`: 6 source pairs x seeds 0-199 x {left,right} x {clean, DART 0.04 burst}
  + held-out source pair and 3 dev pairs on dev seeds (clean, 20 each).
- 14:00 generation on the peer (lease 1790369470_ef0bce, CPU 8, --disk 6G):
  `PY -m rrp.data.collect_dual --config configs/data/assign_pick_place_v1.json --workers 8`.
- Pair index: `rrp latent pair-index-dual --dataset <ds> --packed <pack> --out research/pairs/assign_pick_place_v1.json`
  (per pair: both episode ids, statuses, scene fingerprint + identity check, held-steps per manipulator,
  packed ep_idx). Pack: `rrp latent pack-dual --config configs/latent/pack-assign_pick_place_v1.json` (meta
  `episode_index`). Mini end-to-end check on host (2 seeds): 4 pairs, scene_identical 4/4, holder matches 4/4;
  per-slot subtask labels follow the assignment (slot 0 grasp/place only in assign_left, slot 1 only in
  assign_right).
- 14:10 generation done: 4665 success / 293 failure / 2 infeasible (599 s). Raw: peer
  `artifacts/datasets/assign_pick_place_v1` (manifest per episode: task, robot_key, seed, exec_noise, status).
- 14:15 pack (lease 1790370081_8fc3e6, --disk 20G): `artifacts/packed/assign_pick_place_v1_H16` on the peer,
  751,325 rows from 4,507 source_train success episodes (clean + DART), 0 truncated rows, 11 GB, meta
  `episode_index`. Loads in `LatentData` and runs `representation_step` with multi-assembly labels (peer check).
- 14:20 pair index (lease 1790370968_7a4f42): `research/pairs/assign_pick_place_v1.json`.
  2,480 pairs; scene_identical 2480/2480 (initial physics-state hash); both variants privileged-success 2207;
  valid pairs (both successes AND only the assigned arm ever held the bar) 2203:
  source_train clean 1196/1200, source_train DART 929/1200, development_eval (dev seeds, 3 source pairs) 60/60,
  held-out source pair parm5l_pg2__parm5s_tf3 18/20. 5 DART "successes" pushed the bar into the zone without a
  grasp: flagged `valid_demonstration=false`; their packed ep_idx are listed in
  `summary.invalid_success_packed_ep_idx` (2748, 2788, 3445, 3512, 3926) — filter them when training on pairs.
- Videos (scripted_teacher, same scene, different assigned arm): `artifacts/video/2026-09-25_dual_scripted_teacher_assign_{left,right}_panda_pg2__ur5e_pg2_s3000001_success.mp4`
  (+ handover teacher video). Frames checked: panda (left) grasps in assign_left, UR5e (right) in assign_right.

## how other tracks use the pairs
- binding track (whole-pipeline binding diversity): train on `artifacts/packed/assign_pick_place_v1_H16`
  (same LatentData/representation_step path; multi_m=2 labels held_m/contact_m/rel_tcp_m/subtask_m in packet
  slot order; realizer routes nodes to their own slot via node_asm; set `probe: {n_operators: 14}` in the rep
  config). Paired rows: `episode_index` + pair index give the two ep_idx of each pair.
- acceptance track (manipulator-assignment intervention): scene of a pair = `make_session("assign_left"|"assign_right",
  robot_pair, seed)`; the valid edit is rebinding the actor of take/place (`rrp.sim.dual_scenarios.assign_task`);
  expected behaviour = the other variant's demonstration. Dev-scene pairs (split development_eval, seeds 3000000+)
  are not in the pack. Closed-loop evaluator for the dual packet: `rrp latent evaluate-dual --task assign_left ...`
  (+ `--packet-edit swap_slots`, slot-addressed probes, slot-swap probe control).
- 14:30 compatibility with the binding track (merged on main): host smoke of `train-representation` with
  `latent.binding_cf 0.5` + `probe.n_operators 14` on a 2-seed assign pack, then `fit-probes`: runs end to end
  (multi-assembly labels + counterfactual-binding augmentation). Scratch outputs only.

## next steps (for whoever resumes)
1. binding track: Stage A/B on `artifacts/packed/assign_pick_place_v1_H16` (drop the 5 invalid ep_idx), so
   manipulator assignment varies through the whole pipeline (item 2).
2. acceptance track: manipulator-assignment intervention on a competent checkpoint using the dev-scene pairs
   (`research/pairs/assign_pick_place_v1.json`, split development_eval) with `rrp latent evaluate-dual`.
3. Optional (not critical path): the stopped support_insert/handover sem/nosem chain (`scripts/dualarm_chain.sh`).
