# armdiag: why the latent route does not adapt to xarm7 (D-135 follow-up)
State: **completed** (2026-09-28). Everything here is a NON-SEALED DIAGNOSTIC: xarm7 closed-loop runs use DEV seeds
3,000,000–3,000,029 only (all 30 feasible; teacher 30/30). Offline analyses use only the target DEMO pack (the adaptation
data, seeds 1,000,000+). No sealed eval scene (2,000,000+) was touched. None of these numbers replaces a D-135 cell.
Peer GPU use: < 0.5 GPU-h (plus ~6 CPU lease-hours of simulation).

## Root cause
Adapting to a new arm needs BOTH modules. Each of the two single-module adaptations in the sealed protocol leaves one of
them broken:
1. **System 0 (R_src) cannot realize xarm7 motion, even from perfect packets.** With the encoder's own packet
   (oracle E-mean) on xarm7 demo states, R_src's 1-step TCP displacement has cosine 0.59 to the demo's, and 4.2 cm error
   on a 2.4 cm step. On the source bodies it is 0.96 and 0.47 cm. Flow SFT cannot fix this: dev flow-SFT b100 → R_src
   gives 0/30 (min TCP–cube 0.37 m).
2. **After a system-0 refit, the UNADAPTED flow's packets are off-distribution for the refit realizer in closed loop.**
   Offline, flow_src's z on xarm7 is 6x further from E's target than on the source bodies (relative MSE 0.048 vs
   0.0074; flow SFT brings it to 0.012–0.016). Its semantic probes still read correctly (rel_pos 5.6 mm, subtask 100%),
   so the semantic probes do not show the error: it sits in the body-specific, non-semantic directions of z. In closed
   loop, flow_src → R_refit b100 gives 0/30, 0/30, 2/30 and 12/30 (four lineage×target cells). This matches the sealed
   pattern: the arm reaches the cube area (~5 cm) and then fails at approach.
3. **Adapting both, with checkpoints the sealed protocol already produced** (flow SFT b + system-0 refit b, both from
   the same frozen encoder), makes the latent route work on dev seeds, at or near BC SFT:

| dev 30 eps, grasp_v2.1 | refit only b100 (flow_src→R_rz) | flow SFT + refit b20 | flow SFT + refit b100 | BC SFT b100 |
|---|---|---|---|---|
| semfix s1 xarm7_pg2 | 0 | 16 | 16 | 18 |
| semfix s2 xarm7_pg2 | 2 | 13 | 17 | 27 |
| semfix s1 xarm7_tf3 | 0 | 20 | 23 | 30 |
| semfix s2 xarm7_tf3 | 12 | 23 | 26 | 28 |
| nosem s1 xarm7_pg2 | 0 | 8 | 10 | 18 (BC 1701) |
Pooled semfix: joint b100 82/120 vs refit-only 14/120 vs BC SFT b100 103/120. semfix s1 xarm7_pg2 extra cells:
zero-shot 0/30, flow SFT only 0/30, joint b5 5/30.

Caveat on budget: "joint" pairs two 600-update adaptations (1,200 updates in all), while BC SFT uses 600. A matched probe
(300 flow SFT + 300 refit updates, same 100 episodes; `scripts/armdiag_joint_adapt.py`) gives **9/30** on semfix s1
xarm7_pg2, against BC SFT's 18/30 there. So once the bottleneck is removed, the latent route still adapts more slowly
than BC per update, but the gap is about 2x, not the 0-vs-140 gap in D-135.

## Other checks
- **(1) Is morphology reaching system 0? No bug found.** On xarm7 the node count is 8 (7 hinges + 1 gripper slide,
  gripper last, flag correct), and the ordering follows the command groups. Every node-feature column lies inside the
  range of the 13 source bodies (`offline/semfix-s1-xarm7_pg2.json` target_nodes vs source_nodes), and the
  anchor/drop_qd flags match the deployed bundle. The flow and the bundle share an identical encoder. One weakness, not
  a bug: the static joint axis is the LOCAL axis, (0,0,1) for every hinge on every body, so the static features barely
  describe the kinematic chain. The chain is visible only through the dynamic public-FK features (anchor, world axis,
  Jacobian columns, lever). R_src did not learn a body-general Jacobian-based mapping from them: with 13 source bodies
  (about 7 distinct arm kinematics), it interpolates rather than doing IK (EE cosine 0.59 on xarm7). Also, the
  world-frame axis column is not rotated into the base frame, while jp/jr are. This is harmless while mounts have
  yaw 0 (true here) but is a latent inconsistency for yawed mounts.
- **(2) Packet vs realizer.** The packet's semantic content is right on xarm7 from every source (probe rel_pos error
  5–8 mm, the same as on source). The error is in the realizer (item 1), and in the flow's body-specific z directions
  once the realizer is refit (item 2).
- **(3) Refit learning curve and capacity.** The refit is not underfitting. Teacher-forced realize MSE is 3.9e-4 at
  update 100, 1.4e-4 at 600 (source Stage A: 6.8e-4), and 3.3e-4 on held-out demo episodes (mild overfit). A 5,000-update
  refit (b100) lowers offline error further (oracle EE error 0.26 cm) but does not help the unadapted flow: flow_src →
  R_rz5k gives 1/30, while flow_sft100 → R_rz5k gives 17/30 (vs 16/30 with rz100). More realizer capacity or steps does
  not help; generator–realizer consistency does.
- Open: the teacher-oracle route (E(teacher look-ahead) → R) fails on xarm7 even with R_rz100 (0/30, 22 approach). So
  the teacher-oracle packet is not an upper bound for this realizer. This was not investigated further; the stateless
  BC-oracle (R1 orcbc) is the variant that was validated on source.

No code bug was found, so there is no flagged fix. Sealed cells must NOT be re-run. A joint-adaptation method needs a
new sealed-protocol decision by the lead.

## Recommended next experiment
Add a **joint flow+system-0 adaptation** method to a NEW sealed protocol decision, with total updates matched to BC SFT.
Cells: 150/300/600 updates split evenly, or trained simultaneously with one optimizer. Include one budget-unmatched arm
(600 + 600, what the existing checkpoints already are) to report both. Run it on xarm7_pg2/tf3 × semfix/nosem × 2 seeds.
Before sealing, test on dev seeds whether training the realizer on flow-generated packets (gen-DAgger style, as the
source gendag refits did) closes the remaining per-update gap to BC.

## Commands (peer dirs wt/armdiag, wt/armdiag2; raw outputs in peer store `artifacts/runs/armdiag/`)
- Offline: `scripts/peer_run.sh --gpu --gpu-mem 6G --cpu 3 --mem 6G --label armdiag_off -- PY scripts/armdiag_offline.py
  --flow-src <L>/flow_ft-gdag2h_s1/policy.pt --rep-src <L>/refit-gendag3_noqd_s1/representation.pt --flow-sft
  <T>/target_adapt-flow_sft_b100_s1/policy.pt --rep-refit R_rz100=<T>/target_adapt-system0_refit_b100_s1/representation.pt
  [--rep-refit R_rz5k=...] --pack artifacts/runs/armtgt/armtgt6-data/pack-<target>_s0 --source-pack
  artifacts/packed/latent_pp_v6dart_s1_H16 --budget-seed 1701 --batches 12 --out artifacts/runs/armdiag/offline/<cell>.json`
  (lease 1790621228_f7df68, peak 1.51G). L = artifacts/runs/armv6/arm6-semfix, T = artifacts/runs/armtgt/armtgt6-semfix-<target>.
- Closed loop: `N=30 V=<variant> S=<seed> TGT=<target> ROUTES="..." bash scripts/armdiag_closedloop.sh`, which runs
  `scripts/ladder.py --target-dev-diagnostic --seed-start 3000000 --replan 8 --nfe 8 --max-steps 300 --prev-action zero`
  with RRP_GRASP_CONTACT=v2.1. Leases 1790620547_f877c2 / _f611f2, 1790620801_9f3e0a, 1790620802_76b359,
  1790621227_b6d2d6 (peaks ≤ 1.49G; declared 4G). The rz5k and joint300x300 rows used EXTRA_REP/EXTRA_FLOW (route
  gen_X_X / gen_src_X / gen_sft100_X, renamed in the repo copy).
- Joint matched probe and 5k refit: `PY scripts/armdiag_joint_adapt.py --flow <L>/flow_ft-gdag2h_s1/policy.pt --rep
  <L>/refit-gendag3_noqd_s1/representation.pt --pack .../pack-xarm7_pg2_s0 --seed 1701 --out
  artifacts/runs/armdiag/adapt/semfix-s1-xarm7_pg2/joint300x300`, plus refit_realizer(steps=5000, episode_budget=100,
  seed 1701) → `.../rz5k` (lease 1790620830_e419f0, peak 1.74G).
- Repo copies: `research/tracks/armdiag/{closedloop/<cell>/*.summary.json, offline/*.json, joint300x300_result.json}`.
- Code: `--target-dev-diagnostic` is a hidden ladder flag. It allows the three target bodies with dev seeds only, and
  the summary is labelled NON-SEALED DIAGNOSTIC. Without the flag, behaviour is unchanged.
- Incident: for about 6 minutes, three leases ran at once (the joint/rz5k training lease overlapped two closed-loop
  leases), one more than the requested limit of 2. Resource impact was small (peaks ≤ 1.74G).

## STEP A (D-135 addendum; dev seeds 3,000,000–3,000,029 only): choosing the joint-adaptation variant
All variants: budget 100 episodes (adapt seed 1700 + seed), **600 total updates** (= BC SFT b100), lr 1e-4, batch 128,
`rrp.training.joint_adapt` (joint_adapt_v1). split = 300 flow-SFT then 300 realizer updates (separate AdamW, constant
lr); joint = 600 updates of ONE AdamW over flow + realizer (each update touches both); g05 = half of every realizer
batch uses the CURRENT flow's free sample (NFE 8, detached) at the demo state instead of E's posterior sample (g0 = E
only). Trained in lease 1790621726_6cf932 (`V/S/TGT bash scripts/armdiag_stepA.sh`, peak 2.59G), evaluated with
`ROUTES="ja_split_g0 ja_split_g05 ja_joint_g0 ja_joint_g05" bash scripts/armdiag_closedloop.sh` (leases
1790622321_6e4aab, 1790622322_a02f24). Summaries: `research/tracks/armdiag/stepA/<cell>/`.

| dev 30 eps | split g0 | split g05 | joint g0 | joint g05 | BC SFT b100 (same dev seeds) |
|---|---|---|---|---|---|
| semfix s1 xarm7_pg2 | 13 | 14 | 17 | 13 | 18 |
| semfix s2 xarm7_pg2 | 11 | 16 | 15 | 6 | 27 |
| semfix s1 xarm7_tf3 | 22 | 26 | 19 | 24 | 30 |
| pooled / 90 | 46 | **56** | 51 | 43 | 75 |
Reading: flow-packet realizer training helps the split schedule (+10/90) but hurts the one-optimizer schedule (-8/90);
all four are within noise of each other (46–56 of 90), and none closes the gap to BC SFT (75/90): the per-update gap
shrinks (best 56 vs 75, previously 9 vs 18 on s1 pg2) but does not close. Chosen by the pre-declared rule "highest
pooled dev successes": **split, gen_frac 0.5**. (The earlier split probe `joint300x300`, 9/30 on s1 pg2, differs from
split g0 = 13/30 by the realizer's OneCycle schedule and sampling; both are within noise.)

## PREREG joint_adapt (D-136; committed and pushed BEFORE any sealed run)
- Status: a method ADDED AFTER D-135 (post-hoc; the variant was chosen on DEV seeds after D-135's sealed results were
  known), run on the SAME sealed target scenes as D-135. D-135 cells are not re-run or changed.
- Method (primary, update-matched): `target_adapt` method `joint_adapt`, joint_mode split, gen_frac 0.5
  (rrp.training.joint_adapt, joint_adapt_v1): total optimizer updates = BC SFT's SFT_STEPS = 150 / 300 / 600 at budgets
  5 / 20 / 100 (75/150/300 flow-SFT + 75/150/300 realizer updates), lr 1e-4, batch 128, the SAME budgeted demo
  episodes as flow SFT / refit / BC SFT (nested_budget_indices over the 150 v6dart target demos, adapt seed 1700 + seed),
  encoder and probes frozen; initial flow = final flow gdag2h, initial system 0 = final gendag3 of each v6 lineage.
- Secondary (NOT update-matched: 2x BC SFT's updates): D-135's own flow-SFT b + system-0 refit b checkpoints paired
  (150+150 / 300+300 / 600+600), evaluated once on the same sealed scenes.
- Cells: xarm7_pg2, xarm7_tf3 × semfix, nosem × lineage seeds 1, 2 × budgets 5 / 20 / 100 (primary 24 adaptations +
  24 sealed evals; secondary 24 sealed evals). panda_tf3 is not included (latent already reaches BC level there).
- Eval: configs/eval/latent_slice1.json sealed protocol unchanged (100 scenes from 2,000,000, infeasible excluded and
  counted, max_steps 300, replan 8, NFE 8, prev-action zero, privileged success evaluator, grasp_v2.1), each cell run
  exactly once (`dags/arm_targets_d136_joint.yaml`, lineage armja136-*; a resource-failed node may be rerun once after
  deleting its partial rows, as in D-135, and logged).
- Metric: privileged task success; per cell k/100 and pooled over the 2 seeds per (variant, target, budget) k/200 with
  Wilson 95% CI. Primary comparison: joint_adapt (matched) vs BC SFT (D-135 cells, same target/budget/demos/updates),
  pooled over seeds, difference with Newcombe 95% CI, per (variant, target, budget). Also reported against D-135's
  system-0 refit and flow SFT cells. Reading rule fixed now: "latent adapts as well as BC at equal updates" only if the
  Newcombe CI of (joint − BC) includes 0 or is positive; "worse than BC" if its upper bound is < 0.
- Not changed after this commit: variant, budgets, update counts, seeds, scenes, metric.

## D-136 sealed run (state: running since 12:15 PDT)
Coordinator: host user unit `rrp-armdiag-d136` (run-dag orchestrator only), peer code dir wt/armdiag at f43d8b7,
`--max-parallel 2 --max-parallel-gpu 1`. Ledger `artifacts/runs/armdiag/_dags/arm_targets_d136_joint/ledger.json` (host).
RESUME (completed nodes are skipped): `cd ~/work/rrp-wt/armdiag && systemd-run --user --unit rrp-armdiag-d136b
--setenv=RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/armdiag --setenv=PYTHONPATH=src --working-directory=$HOME/work/rrp-wt/armdiag
~/work/relational-robot-policy/.venv/bin/python -m rrp.cli run-dag dags/arm_targets_d136_joint.yaml --max-parallel 2 --max-parallel-gpu 1`
