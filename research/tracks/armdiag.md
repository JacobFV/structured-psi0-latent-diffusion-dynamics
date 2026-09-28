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
