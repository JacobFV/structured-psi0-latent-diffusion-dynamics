# track: ladder — closed-loop failure localization (correction 2026-09-25, item 3)

Owner: ladder track agent. Branch `track/ladder`, worktree `~/work/rrp-wt/ladder`, peer dir `/dev/shm/rrp-brandonin/wt/ladder`.
Raw outputs live on the peer store `artifacts/runs/ladder_*` (copied summaries under `research/tracks/ladder/` when final).

## SPRINT BEST ROUTE FINAL (frozen 2026-09-26 01:10 PDT, sprint_latent; for sprint_semantic / sprint_demo)
**Deployable route (R2): system i `learned:ladder_flow_jointfix_gdag1` -> system 0 `learned:ladder_rz_jointfix_gendag3_noqd`.**
No teacher, oracle or BC at run time. pick_place, 300 ticks, replan 8, NFE 8, standard stochastic sampling (noise scale 1),
prev-action input 0, privileged success evaluator. Matched dev seeds = the first 30 feasible from 3,000,000
(n_distractors = seed % 3). Fresh = the first 30 feasible from 3,000,100. Wilson 95%.

| rung (source label) | panda_pg2 | parm6_tf3 | failure stages (panda / parm6) | track q rad / TCP m (panda; parm6) |
|---|---|---|---|---|
| R0 scripted_teacher (privileged) | 30/30 [0.89,1.00] | 30/30 [0.89,1.00] | - | 0.025 / 0.021; 0.006 / 0.007 |
| reference: plain BC learned:direct1701_u12000 (sprint_bc) | 25/30 [0.66,0.93] | 27/30 [0.74,0.97] | late (grasp/lift/transport/place) | |
| R1 ORACLE DIAGNOSTIC, stateless: packet = E(chunk of learned:direct1701_u12000 at the current state) -> gendag3_noqd | 13/30 [0.27,0.61] | 27/30 [0.74,0.97] | lift 8, approach 6, grasp 2, transport 1 / place 2, transport 1 | 0.014 / 0.014; 0.007 / 0.007 |
| **R2 generated: flow_jointfix_gdag1 -> gendag3_noqd** | **10/30 [0.19,0.51]** | **22/30 [0.56,0.86]** | lift 7, approach 6, grasp 4, transport 3 / lift 4, place 2, approach 1, transport 1 | 0.012 / 0.012; 0.005 / 0.006 |
| R2, same checkpoints, 30 FRESH seeds | 13/30 [0.27,0.61] | 21/30 [0.52,0.83] | lift 8, approach 4, grasp 2, transport 2, place 1 / lift 4, transport 4, place 1 | 0.012 / 0.012; 0.005 / 0.006 |
| R2 pooled (60 seeds) | **23/60 = 0.38 [0.27,0.51]** | **43/60 = 0.72 [0.59,0.81]** | | |
| R2 on the 13 source-TRAINING bodies (seeds 4,000,000+, 24 each; gdag2 collection) | 209/312 = 0.67 overall | | | |
| historical: R1 shadow-teacher oracle (CONFOUNDED, D-050), best jfdag1 | 1/30 | 0/30 | approach | |

HELD-OUT source bodies (not in the pack, not in any DAgger buffer; same harness, first 30 feasible dev seeds from 3,000,000):
| rung | parm5s_tf3 | parm5l_pg2 |
|---|---|---|
| R0 scripted_teacher (privileged) | 30/30 [0.89,1.00] | 30/30 [0.89,1.00] |
| plain BC learned:direct1701_u12000 (ladder harness) | 26/30 [0.70,0.95] | 29/30 [0.83,0.99] |
| R1 ORACLE DIAGNOSTIC stateless -> gendag3_noqd | 28/30 [0.79,0.98] | 27/30 [0.74,0.97] |
| **R2 generated flow_jointfix_gdag1 -> gendag3_noqd** | **21/30 [0.52,0.83]** | **22/30 [0.56,0.86]** |
R2 failures: approach 3, lift 3, transport 2, place 1 / approach 5, lift 2, place 1. Raw: `artifacts/runs/ladder_v1/<robot>/{generated_zero_flowgdag1_rzgendag3_noqd,learned_bc_direct1701_u12000,oracle_zero_gendag3noqd_orcbc,teacher_heldout_ref}.summary.json`.
Tracking is never the failure: the joint tracker follows every rung's commands within ~1 cm TCP.
Raw (peer store = also in the peer path `/dev/shm/rrp-brandonin/repo/`): `artifacts/runs/ladder_v1/<robot>/generated_zero_flowgdag1_rzgendag3_noqd[_fresh3000100].{jsonl,summary.json}`,
`artifacts/runs/ladder_v1/<robot>/oracle_zero_gendag3noqd_orcbc.{jsonl,summary.json}`, `.../teacher_shadow_own.summary.json`,
`artifacts/runs/ladder_dagger_gdag2/generated_<robot>.summary.json`.

Checkpoints (peer store; sha256):
- system i (flow): `artifacts/runs/ladder_flow_jointfix_gdag1/policy.pt` 78fbee7f4f8df737d5074409f1d16379b031a459ea054a4b9192d64e487df873
  (config `configs/ladder/flow_jointfix_gdag1.json`: init `ladder_flow_jointfix/snap_final_s20000.pt`
  33b958884f3af12b59ed0a91f2b563101d9e8ace2521b143067d77113d4cc3bc, +4k steps generator DAgger, zero_prev_action,
  normalize_target, packet_semantic_weight 1.0, packet_tau_min 0.6). Latent space ls-80e5f25be2f0-wf22fe70f99d5.
- system 0 (+ the frozen Stage-A encoder E and probes P in the same bundle): `artifacts/runs/ladder_rz_jointfix_gendag3_noqd/representation.pt`
  f60cde41ed0671b84d1f698288d5a69a086f7d89f187b6562e195a287ff23622, realizer compat
  rz-ls-80e5f25be2f0-wf22fe70f99d5-r96d867118f17-none-v1; realizer_anchor true, realizer_drop_qd true (the runtime zeroes
  the joint-velocity input; handled by `realizer_node_feats` when loaded with `load_representation`).
- Encoder origin: `artifacts/runs/ladder_latent_sem_b1fix_anchor/representation.pt` 49b2e2e3...fb8 (Stage A with the
  semantic packet objective, trained jointly with the B-1 fix).
- USAGE: the flow's packets are addressed to the ORIGINAL jointfix realizer; always pass the system-0 bundle explicitly
  (`scripts/ladder.py --route generated --flow <flow> --rep <gendag3 bundle>`; `load_models` logs `realizer_override`).
  Tools that load only the flow checkpoint would use the jointfix realizer (0/30) — do not use them for this route.
- Labels: system i = learned (flow); system 0 = learned, with DAgger labels from the plan rows of a LEARNED stateless
  expert (BC learned:direct1701_u12000, trained on the same scripted-teacher demonstrations) at learner-visited states;
  the generator DAgger targets z* = E(chunk of that BC). DAgger seeds 3,200,000-4,000,000 (disjoint from dev 3,000,000+).

What made it work (all in-architecture; ablations on the same seeds in the live table below):
1. System 0 copied the current joint velocity (a B-1-like proprio shortcut): zeroing only qd collapsed its step gain from
   0.88 to 0.12; removing qd alone moved R1 stateless from 0/30, 0/30 to 0/30, 5/30.
2. The shadow teacher FSM is stale off its own trajectory (sprint_bc); the stateless BC expert gives valid packets and labels.
3. System-0 DAgger with that expert (3 rounds) + DAgger on system i's OWN generated packets + z-noise 0.3: R1 stateless
   0/30,0/30 -> 18/30,27/30; R2 0/30,0/30 -> 9/30,17/30.
4. Generator DAgger (flow fine-tuned on learner-visited contexts toward z* = E(BC chunk)): R2 9/30,17/30 -> 10/30,22/30.
Not solved: panda grasp/lift (pg2 parallel gripper alignment) and parm6 place; R2 < R1 stateless < BC.

Sem vs nosem (binding v4 bundles) with the same recipe, compressed to the sprint: NOT competent, so no deployable
comparison yet. R1 stateless after 3 BC-DAgger rounds: sem 1/30, 1/30; nosem 0/30, 4/30. R2 with their own flows
(12k, `ladder_flow_bindv4{sem,nosem}`): sem 0/30, 0/30; nosem 0/30, 1/30; on 13 training bodies sem 0/312, nosem 4/312.
Their generated-packet DAgger / flow DAgger round is still running (see the live section; will be appended).
The binding chain's own flows (`flow_binding_paired_{sem,nosem}_v4`) are deadlocked on prefetch (alert below).

## SPRINT BEST ROUTE (live; updated 2026-09-26 00:35 PDT by sprint_latent)
**Best DEPLOYABLE route (R2: system i flow -> system 0; no teacher, no oracle, no BC at run time):
learned:ladder_flow_jointfix_gdag1 (generator DAgger) -> system 0 learned:ladder_rz_jointfix_gendag3_noqd:
panda_pg2 10/30 [0.19,0.51], parm6_tf3 22/30 [0.56,0.86]** (matched dev seeds, NFE 8, standard sampling);
**30 FRESH seeds (3,000,100+): 13/30 [0.27,0.61] and 21/30 [0.52,0.83] -> pooled panda 23/60 = 0.38 [0.27,0.51],
parm6 43/60 = 0.72 [0.59,0.81].** (Plain BC on the dev seeds: 25/30, 27/30; teacher 30/30, 30/30.)
Binding v4 with the same recipe so far (R1 stateless, system 0 after 3 BC-DAgger rounds, no qd, z-noise): sem 1/30, 1/30;
the generated-packet rounds and flow DAgger for sem and nosem are running (`scripts/ladder_bindv4_chain.sh`).
Progression on the same seeds (panda / parm6): flow final + gendag1noqd 0/30, 9/30 -> + gendag2noqd 6/30, 12/30 ->
flow_ft + gendag2noqd 7/30, 16/30 -> flow_ft + gendag3noqd 9/30, 17/30 -> flow_gdag1 + gendag2noqd 9/30, 19/30 ->
**flow_gdag1 + gendag3noqd 10/30, 22/30**. Fresh-seed check (3,000,100+) of the round-2 system with the 20k flow: 9/30, 11/30.
Generator DAgger (`configs/ladder/flow_jointfix_gdag1.json`): flow fine-tuned 4k steps (lr 1e-4) from
`ladder_flow_jointfix/snap_final_s20000.pt` on 50% pack rows + 50% (public context at learner-visited R2 states, target
z* = E(chunk of BC learned:direct1701_u12000 at that state)); contexts from R2 rollouts of flow_ft + gendag2noqd on the 13
source-training bodies, seeds 3,900,000+ (`artifacts/runs/ladder_dagger_gdag1/*.genctx.pkl`). Offline: |z_gen - z_bc| /
|z_bc| 0.24 / 0.30 (flow_ft 0.25 / 0.32), system-0 error from the generated packet 0.73 / 2.69 of hold-still.
On the 13 source-TRAINING bodies, R2 flow_gdag1 -> gendag3noqd (the gdag2 collection rollouts, seeds 4,000,000+, 24 each):
**209/312 = 0.67** (raw peer `artifacts/runs/ladder_dagger_gdag2/generated_<robot>.summary.json`); earlier pair below.
On the 13 source-TRAINING bodies (seeds 3,800,000+, 24 each; the gen-DAgger round-3 collection rollouts, R2 flow_ft ->
gendag2noqd): 143/312 = 0.46 (parm6_pg2 20/24, parm7_pg2 19/24, parm5s_pg2 18/24, parm5_pg2 17/24 ... sawyer_tf3 3/24),
raw peer `artifacts/runs/ladder_dagger_gen3/generated_<robot>.summary.json`.
References on the matched seeds: R0 scripted_teacher 30/30, 30/30; plain BC learned:direct1701_u12000 25/30, 27/30;
R1 stateless oracle through the same system 0: 9/30, 27/30.
Sampling check (host, flow final + gendag2noqd, parm6): NFE 16 13/30 (vs 12/30 at NFE 8); initial-noise scale 0.5 4/30;
scale 0 (deterministic) 1/30 (approach 24) -> standard stochastic sampling is best; mode-seeking collapses the packet.
System 0 checkpoint: `artifacts/runs/ladder_rz_jointfix_gendag2_noqd/representation.pt`; flow:
`artifacts/runs/ladder_flow_jointfix_ft/policy.pt` (`configs/ladder/flow_jointfix_ft.json`, init from
`ladder_flow_jointfix/snap_final_s20000.pt`, +10k steps, lr 1e-4). Running: system 0 round 3 (`gendag3_noqd`), generator
DAgger (flow fine-tune on learner-visited contexts with target z* = E(BC chunk), `configs/ladder/flow_jointfix_gdag1.json`).
Best oracle-diagnostic route (R1 stateless: packet = E(chunk of BC learned:direct1701_u12000 at the current state)):
system 0 `gendag1qdd`: panda 18/30 [0.42,0.75], parm6 27/30 [0.74,0.97] (BC itself: 25/30, 27/30) -> with a valid
packet, system 0 is now close to BC-level on parm6. The remaining deployable gap is the GENERATOR's packet (grasp on panda,
place on parm6).
Reference on the same 30 seeds: R0 scripted_teacher 30/30, 30/30; plain BC learned:direct1701_u12000 25/30, 27/30.

**ALERT for the lead (23:20): the binding chain's flows `flow_binding_paired_{sem,nosem}_v4` are DEADLOCKED** (peer leases
1790399212_fa9d73 since 22:06 and 1790401602_231ce8 since 22:46: main process 0% CPU, 3 idle forked prefetch workers,
623 MiB GPU, empty train_log; config has "prefetch": true = the known fork-after-torch-init deadlock, see
infrastructure notes). They hold two GPU leases doing nothing. I am not stopping them (not my leases). For the
sem-vs-nosem deployable comparison I am training ladder-owned copies with prefetch off and 12k steps
(`configs/ladder/flow_bindv4{sem,nosem}.json` -> `artifacts/runs/ladder_flow_bindv4{sem,nosem}`) as soon as my GPU slots free.

Checkpoints (peer store; copies on host):
- system i: `artifacts/runs/ladder_flow_jointfix/snap_final_s20000.pt` (FlowPolicy, zero_prev_action, normalize_target,
  packet_tau_min 0.6, packet_semantic_weight 1.0, 20k steps; `configs/ladder/flow_jointfix.json`; latent space
  ls-80e5f25be2f0-wf22fe70f99d5 = Stage-A E of `ladder_latent_sem_b1fix_anchor`, trained jointly with the B-1 fix).
- system 0: `artifacts/runs/ladder_rz_jointfix_gendag1_noqd/representation.pt` (same E; system 0 refit from jfbcdag2,
  8k steps, lr 2e-4, 60% DAgger: BC-expert oracle-packet buffers bc1-bc3 + generated-packet buffers gen1 (route R2
  rollouts of flow_jointfix + jfbcdag1long, seeds 3,500,000+, 12/13 bodies), z-noise 0.3 on pack z, joint-velocity
  input removed (realizer_drop_qd), anchored input; `configs/ladder/rz_jointfix_gendag1_noqd.json`).
- HONEST LABELS: the DAgger labels of system 0 are the plan rows of a LEARNED stateless expert (plain BC
  learned:direct1701_u12000, itself trained on the same scripted-teacher demonstrations), not the scripted teacher;
  DAgger seeds (3.2M-3.6M) are disjoint from the dev seeds (3.0M). The flow never sees BC.
- Raw: peer `artifacts/runs/ladder_v1/<robot>/generated_zero_ladder_flow_jointfix_snap_final_s20000_rzgendag1noqd.{jsonl,summary.json}`.

**Mechanism found (lead item 2/3, 20:45): system 0 copies the current joint VELOCITY, not the packet.** At BC-visited
states (bindv4nosem, panda, 10 seeds, `artifacts/runs/ladder_localize/bias/`), system 0's commanded TCP step has gain 0.88
vs BC's (j>=1) and 0.61 on the first tick of a packet; zeroing ONLY the joint-velocity node input (col 27) collapses the
gain to 0.12 and the error to hold-still level at every j (0.011-0.017 vs hold 0.011-0.019). So the 1-step accuracy that
passes the offline gate (bindv4 15-19% of hold-still) comes from extrapolating the present motion; the packet contributes
little. In closed loop from rest this is a fixed point: a closed-loop trace (bindv4nosem, R1 stateless, panda seed
3000000, `artifacts/runs/ladder_smoke/oracle_trace_bindv4nosem_orcbc.jsonl`) shows the packet's plan row 5-15 cm from the
TCP while system 0 commands <1 cm steps, often in another direction; min TCP-cube 11 cm. Same class as B-1 (a
proprioceptive shortcut), via qd. The anchored input (col 28) itself is computed identically at runtime and in training
(runtime j>=1 errors are tiny; `realizer_node_feats` vs `LatentData.fetch`), so (3) is not a bug; but DAgger buffers
stored col 28 = 0 (fixed now: `_load_dagger(anchor=True)` recomputes it from the packet's j=0 row; jfbcdag1/jfbcdag1long
were trained with col 28 = 0 on their DAgger half). Fix under test (in-architecture): `realizer_drop_qd` = system 0
without the velocity input (training and runtime), refits `rz_jointfix_noqd` (host GPU) and `rz_bindv4sem_noqd` (peer GPU).
Why BC-expert DAgger helps: its buffer contains the learner's own slow/stalled states labelled with the plan row, which
penalizes the velocity copy (R1 stateless: jointfix 0/30,0/30 -> jfbcdag1 0/30,0/30 but reaching transport/place ->
jfbcdag1long 3/30, 11/30).
Gripper / place (lead item 4): with jfbcdag1long most parm6 failures are at place (13) and transport (5); in the DAgger-2
collection on 13 bodies (jfbcdag1) most failures are transport/place; next check.

| rung (source label) | system 0 | panda_pg2 | parm6_tf3 | failure stage (panda / parm6) | track q rad / TCP m (panda) |
|---|---|---|---|---|---|
| R0 scripted_teacher (privileged) | - | 30/30 [0.89,1] | 30/30 [0.89,1] | - | 0.025 / 0.021 |
| reference: plain BC learned:direct1701_u12000 (sprint_bc) | - | 25/30 [0.66,0.93] | 27/30 [0.74,0.97] | late (grasp/lift/transport) | |
| R1 oracle, shadow-teacher packet, re-anchored (ORACLE DIAGNOSTIC; CONFOUNDED: stale FSM, D-050) | jfdag1 | 1/30 | 0/30 | approach 25 / approach 29 | 0.042 / 0.037 |
| R1 oracle, STATELESS packet E(BC chunk at current state) (ORACLE DIAGNOSTIC) | jointfix | 0/30 [0,0.11] | 0/30 [0,0.11] | approach 18, grasp 5, lift 5, transport 1, place 1 / approach 20, grasp 4, lift 5, transport 1 | 0.011 / 0.012 |
| same | jfdag1 (shadow DAgger r1) | 0/30 | 0/30 | approach 30 / grasp 16, lift 4, transport 4, place 3 | 0.018 / 0.014 |
| same | jfdag2df08 (shadow DAgger r1+r2, 80% DAgger; D-049 fix A) | 0/30 | 0/30 | approach 29 / approach 27 | 0.023 / 0.019 |
| same | **gendag1qdd** (bcdag2 + DAgger on BC-oracle AND generated packets, z-noise 0.3, qd dropout 0.5) | **18/30 [0.42,0.75]** | **27/30 [0.74,0.97]** | lift 6, approach 4 / place 2, transport 1 | 0.016 / 0.015 |
| same | gendag1noqd (same, qd removed) | 18/30 [0.42,0.75] | 24/30 [0.63,0.90] | approach 4, lift 5, grasp 3 / place 4, transport 2 | 0.016 / 0.015 |
| same | gendag1 (same, qd kept) | 11/30 | 22/30 | | |
| same | jfbcdag3 (BC-DAgger round 3: bc1-bc4) | 21/30 [0.52,0.83] | 22/30 [0.56,0.86] | place 3, lift 2, grasp 2 / transport 5, place 3 | 0.018 / 0.017 |
| same | **jfbcdag2** (DAgger round 2: from jfbcdag1long, bc1+bc2, 16k) | **11/30 [0.22,0.54]** | **19/30 [0.46,0.78]** | place 4, grasp 4, approach 2, lift 5, transport 4 / place 6, transport 5 | 0.016 / 0.015 |
| same | jfbig@16.3k (4 layers x 256, z standardized, fresh, bc1+bc2; interrupted at 16.3k/32k by host stop) | 4/30 [0.05,0.30] | 20/30 [0.49,0.81] | lift 9, transport 9, grasp 5, place 2 / place 7, transport 2 | 0.015 / 0.012 |
| same | bindv4sem_bcdag2noqd (binding v4 sem E; recipe, BC-DAgger r1+r2, 8k) | 0/30 | 1/30 | approach 26 / transport 12, approach 7, grasp 5 | 0.013 / 0.016 |
| same | bindv4nosem_bcdag2noqd (binding v4 nosem E; same) | 0/30 | 1/30 | approach 29 / approach 16, grasp 6, place 4 | 0.011 / 0.010 |
| same | bindv4sem_bcdag1 (binding v4 sem E; system 0 + 1 round BC-expert DAgger, 8k) | 0/30 | running | grasp 9, approach 19, lift 2 / | 0.031 / 0.027 |
| same | bindv4nosem_bcdag1 (binding v4 nosem E; same) | 0/30 | running | approach 27, grasp 2, lift 1 / | 0.015 / 0.014 |
| same | jfnoqd (jointfix, no joint-velocity input, pack only 8k) | 0/30 | 5/30 [0.07,0.34] | grasp 14, approach 8, lift 5 / transport 13, approach 7, lift 4 | 0.011 / 0.013 |
| same | **jfbcdag1long** (jointfix E; system 0 16k steps, lr 3e-4, 50/50 BC-expert DAgger) | **3/30 [0.03,0.26]** | **11/30 [0.22,0.54]** | approach 5, grasp 5, lift 3, transport 11, place 3 / place 13, transport 5, approach 1 | 0.013 / 0.013 |
| same | bindv4nosem (binding v4, no semantic loss) | 0/30 | 1/30 [0.01,0.17] | approach 25 / approach 22 | 0.009 / 0.010 |
| same | bindv4sem (binding v4, semantic) | 0/30 | 0/30 | approach 28 / approach 30 | 0.007 / 0.009 |
| same | **jfbcdag1** (jointfix R + 1 round DAgger with the stateless BC expert, label = packet plan row j; D-049 fix B) | 0/30 [0,0.11] | 0/30 [0,0.11] | approach 9, grasp 10, lift 8, transport 3 / **transport 13, place 12**, lift 4, grasp 1 | 0.016 / 0.016 |
| R2 learned:ladder_flow_jointfix@4000 (system i, zero_prev_action, normalize_target, tau_min 0.6) | jointfix | 0/30 [0,0.11] | 0/30 [0,0.11] | approach 16, grasp 11, lift 2, place 1 / approach 21, grasp 7, transport 2 | 0.014 / 0.017 |
| R2 learned:ladder_flow_jointfix@8000 | jointfix | 0/30 | 0/30 | approach 24, grasp 5, lift 1 / approach 14, grasp 14, lift 2 | 0.012 / 0.014 |
| R2 @12000 | jointfix | 0/30 | 0/30 | approach 24 / grasp 12, approach 12 | 0.013 / 0.015 |
| R2 @16000 | jointfix | 0/30 | 0/30 | approach 20, lift 4, grasp 6 / grasp 20, approach 7 | 0.012 / 0.014 |
| R2 @8000 | jfbcdag1 | 0/30 | 0/30 | lift 12, approach 11, grasp 7 / grasp 20, approach 9 | 0.022 / 0.023 |
| R2 @20000 (final) | jointfix | 0/30 | 0/30 | approach 24, lift 5, grasp 1 / grasp 17, approach 11 | 0.012 / 0.014 |
| R2 @16000 | jfbcdag1long | 0/30 | 0/30 | grasp 21, approach 8, lift 1 / grasp 22, others 8 | 0.016 / 0.014 |
| R2 learned:ladder_flow_jointfix@20000 (final) | **jfbcdag2** | 0/30 [0,0.11] | **1/30 [0.01,0.17]** (first deployable-route success) | grasp 22, lift 4, approach 4 / grasp 20, transport 3, lift 2, approach 2, place 2 | 0.017 / 0.014 |
| **R2 @20000** | **gendag2noqd** (round 2 of generated-packet DAgger) | **6/30 [0.10,0.37]** | **12/30 [0.25,0.58]** | grasp 12, lift 9, transport 2, place 1 / place 6, approach 5, transport 5, grasp 1, lift 1 | 0.011 / 0.012 |
| R2 flow_jointfix_ft (+10k fine-tune) | gendag1noqd | 0/30 | 11/30 [0.22,0.54] | grasp 23 / place 10, approach 3, lift 3, transport 3 | 0.010 / 0.011 |
| R2 @20000 FRESH seeds 3,000,100+ | gendag1noqd | 2/30 [0.02,0.21] | 7/30 [0.12,0.41] | grasp 25 / transport 9, place 8, approach 3, lift 2, grasp 1 | 0.011 / 0.011 |
| R2 @20000 | gendag1qdd (same recipe, qd dropout 0.5 instead of removal) | 0/30 | 7/30 [0.12,0.41] | grasp 25, approach 5 / place 9, approach 7, lift 3, transport 3 | 0.011 / 0.011 |
| R2 @20000 | gendag1 (same recipe, qd kept) | 1/30 [0.01,0.17] | 6/30 [0.10,0.37] | grasp 25, approach 4 / place 13, transport 5 | 0.010 / 0.010 |
| R2 @20000 | jfbcdag3 (BC-DAgger round 3, oracle-packet buffers only) | 3/30 [0.03,0.26] | 1/30 | grasp 18, approach 6 / grasp 16, transport 6, approach 5 | 0.013 / 0.013 |
| **R2 @20000 (final)** | **gendag1noqd** (DAgger on generated + BC-oracle packets, z-noise, no velocity input) | 0/30 [0,0.11] | **9/30 [0.17,0.48]** | grasp 26, approach 4 / place 13, transport 4, lift 2, grasp 1, approach 1 | 0.011 / 0.011 |
| R2 @20000, FRESH seeds 3,000,100+ | jfbcdag2 | 0/30 | 0/30 | grasp 23, approach 4, lift 3 / grasp 12, approach 9, transport 6, lift 3 | 0.017 / 0.015 |
| R2 @20000 | jfznoise@11.1k (z-noise 0.3 refit, interrupted by host stop) | 0/30 | 1/30 [0.01,0.17] | grasp 25 / approach 12, grasp 11, transport 6 | 0.014 / 0.012 |
| R2 @20000, NFE 32 (host) | jfbcdag2 | 0/30 | 1/30 [0.01,0.17] | grasp 21, approach 5 / grasp 21, transport 3, place 2 | 0.018 / 0.015 |
| R2 @20000 (final) | jfbcdag1long | 0/30 [0,0.11] | 0/30 [0,0.11] | grasp 15, approach 11, lift 4 / grasp 18, approach 7, lift 2, transport 2, place 1 | 0.016 / 0.013 |
Generator gap at the final flow (BC-visited states, system 0 jfbcdag1long): |z_gen - z_bc|/|z_bc| 0.25 panda / 0.33 parm6
(0.50 / 0.54 at 4k), but system 0's arm error from the generated packet is still hold-still level (ratio 1.00 / 2.44)
and its gripper error 0.23 / 0.20 vs 0.06 / 0.04 from the oracle packet -> R2 now reaches the cube but fails the GRASP:
the generated packet's residual error is in directions system 0 is sensitive to (gripper timing/closing).
R2 grasp failure mechanism (trace `artifacts/runs/ladder_smoke/generated_trace_r2_bcdag2.jsonl`, parm6 seed 3000003): the
tool sits at the cube (|tcp-cube| < 1.5 cm) for ~150 ticks with the gripper commanded OPEN (-0.6) the whole time; the BC
plan at the same states says close (+0.65) at some replans. The generated packet does not carry the close decision.
Host note: the host broker's disk-reserve watchdog stopped all host jobs again at ~21:45 (300 GB free vs 322 GB reserve;
external growth); interrupted checkpoints are evaluated as such: `ladder_rz_jointfix_znoise` @11.1k/12k and
`ladder_rz_jointfix_big32k` @16.3k/32k (representation_interrupted.pt).
Next (running): system-0 DAgger on system i's OWN packets (route generated, labels = BC plan row j; buffers
`artifacts/runs/ladder_dagger_gen1/`), then refit from jfbcdag1long.

Stateless localization on BC-visited states (`scripts/ladder_localize.py`; BC episodes replayed exactly, 30/30 replay
consistent; z_bc = E(chunk BC actually executed next); system 0 NOT executed; 1-step normalized MSE vs BC's command):
| system 0 | arm err panda / parm6 | hold-still arm ref panda / parm6 | first tick after a new packet (j=0), panda |
|---|---|---|---|
| jointfix | 0.0049 / 0.0035 | 0.0114 / 0.0047 | 0.011 (= hold-still) |
| jfdag1 | 0.0094 / 0.0062 | same | 0.014 |
| jfdag2df08 | 0.0175 / 0.0110 (worse than holding still) | same | |
| jfbcdag1 | 0.0073 / 0.0060 (gate 0.64 / 1.29) | same | |
| jfbcdag1long | 0.0048 / 0.0031 (gate 0.42 / 0.67) | same | |
| jfbcdag2 | 0.0045 / 0.0041 (gate 0.39 / 0.89); from the GENERATED packet (flow final) 0.0140 / 0.0200 (gen gate 1.23 / 4.30) | same | |
| jfnoqd | 0.0055 / 0.0036 (gate 0.48 / 0.78) | same | |
| jfznoise@11.1k | 0.0042 / 0.0036 (gate 0.37 / 0.76); generated packet 0.0108 / 0.0150 (gen gate 0.95 / 3.22) | same | |
| jfbig@16.3k | 0.0061 / 0.0050 (gate 0.54 / 1.08); generated 0.0114 / 0.0157 (gen gate 1.00 / 3.36) | same | |
Generated-packet gate for gendag1 / qdd / noqd: 0.93 / 0.87 / 0.98 (panda), 3.07 / 3.14 / 3.25 (parm6): the gate compares
system 0's output with BC's command, while the generated packet encodes the flow's own (teacher-like) plan, so it does
not track R2 success (gendag1noqd has the worst gen gate and the best R2); treat it as uninformative for R2.
R2 status: 1/30 on parm6 with jfbcdag2 did not replicate on 30 fresh seeds (0/30; pooled 1/60). The generated-packet gate
(target <= 0.5) is not met by any system 0 so far (best 0.95 panda / 3.2 parm6).
| bindv4nosem | 0.0017 / 0.0016 (gate 0.15 / 0.35) | same | 0.0089 vs 0.0004-0.0012 |
| bindv4sem | 0.0021 / 0.0020 (gate 0.19 / 0.42) | same | 0.0098 vs 0.0008-0.0015 |
Gate metric = arm err / hold-still (target <= 0.20): jointfix 0.43 / 0.74; jfbcdag1 0.64 / 1.29; jfdag1 0.83 / 1.33.
Caveat: the gate is measured on BC's own (on-plan) states; the BC-expert DAgger refit is worse there but better in closed
loop (parm6 reaches transport/place in 25/30 instead of 1/30), so the gate is necessary-ish, not sufficient/aligned.
**Generator gap (flow_jointfix@4000 at the same BC states, `..__jointfix__flowjf_s4000.json`)**: |z_gen - z_bc| / |z_bc|
= 0.50 (panda) / 0.54 (parm6); system 0's arm error from the GENERATED packet 0.0106 / 0.0088 vs 0.0049 / 0.0035 from
the oracle packet (hold-still 0.0114 / 0.0047): through system 0, the generated packet is no better than holding
still. So at 4k flow steps BOTH stages are short: system 0 realizes ~55%/25% of the motion from a perfect packet, and
the generator's packet loses the rest.
On the training pack itself (panda rows, jointfix, `scripts/ladder_packed_check.py`) system 0's arm error is 0.02-0.066
vs 0.03-0.09 for a zero action: it explains only ~30% of the teacher's 1-step motion (underfit), and on the teacher's
own clean trajectory its j=0 error is 3x its j>=1 error (0.0126 vs ~0.004).

Why system 0 is the bottleneck: (1) packets that encode a 25/30 controller's own chunks give 0/30 through system 0 in
closed loop; (2) system 0 explains only ~55% (panda) / ~25% (parm6) of BC's 1-step motion at BC's own states, and none
of it on the first tick of each packet; (3) R2 at 4k flow steps already fails at the same stages as R1-BC (approach/grasp),
so the generator is not visibly worse than the stateless oracle. Pending: the generator gap measured directly
(z_gen vs z_bc distance and system-0 error from generated vs oracle packets at BC states; leases 1790391692_2ffeb0,
1790391693_2ce338 -> `artifacts/runs/ladder_localize/<robot>/bc_direct1701_u12000__jointfix__flowjf_s4000.json`).
Shadow-teacher DAgger makes system 0 WORSE on BC-visited states (2x-3.5x error): withdrawn as a fix.

System-0 offline-first levers (lead 20:05): running (a)+(c)+(d) combined `rz_jointfix_big_bcdag1` (realizer 4 layers x 256,
z standardized at system-0 input, j=0 loss weight 3, fresh init, 12k steps, lr 3e-4, 50/50 BC-expert DAgger; peer GPU lease
1790391992_8bb97e, ~0.45 s/step, ETA ~21:40); (b) `rz_jointfix_bcdag1_long` (2 layers, init jointfix, 16k steps, lr 3e-4,
host GPU, ETA ~20:40); next on the host GPU: `rz_jointfix_big_pack` (same as big, pack only, no DAgger: teacher data only).
Each passes the offline gate (`scripts/ladder_gate.sh`) before closed-loop R1-BC / R2.

Compute now:
- HOST GPU: `ladder_flow_jointfix` (lease 1790389550_4baed9, 0.26 s/step, 20k steps, ETA ~20:55 PDT);
  `scripts/ladder_flow_watch.sh` snapshots every 4k steps and runs R2 on the peer (tags `zero_flowjf_s<step>`).
- HOST GPU: `ladder_rz_jointfix_bcdag1_long` (system-0 refit, 16k steps, lr 3e-4, 50/50 BC-expert DAgger; ETA ~20:40).
- PEER: binding v4 reps (binding chain, ~20:25) -> flows in the binding chain; ladder R1-BC evals of bindv4 sem/nosem
  start automatically when the reps land (host loop, tags `zero_bindv4{sem,nosem}_orcbc`).

## CURRENT ANSWER FOR ACCEPTANCE / GRPO / campaign (which route/checkpoint is competent)
- **No competent learned route exists yet, and none can with the current Stage-A system 0**: R0 teacher 30/30;
  R1 oracle (latent_sem_v1) 0/30 with deployment input, 0/30 with training-consistent own-prev input; R2 flow v2 0/30.
- **Root cause localized: bug B-1** (below). The Stage-A realizer is a copycat of the previous teacher command stored in
  node-feature column 28; at deployment that column is 0, which on the TRAINING pack itself makes system 0 no better than
  holding still (arm) and wrong on the gripper. This explains the acceptance finding (step-0 output -0.1 vs -1.0, packet
  fine) and the GRPO findings (small undirected steps, gripper stuck ~0.03 = "mode averaging" is actually copying a zero
  prev-action; no grasp/lift after a teacher prefix).
- **Fix in progress**: system 0 re-fit on the FROZEN sem_v1 encoder with column 28 zeroed (same latent space => existing
  flows stay usable; new realizer compat id). Running: peer lease 1790370807_bd05c2 `ladder_rz_sem_ft`
  (config `configs/ladder/rz_sem_v1_b1fix_ft.json`, warm start, 8k steps) -> `artifacts/runs/ladder_rz_sem_v1_b1fix_ft/`.
  Then R1/R2 are rerun on it. If the frozen z lacks what R needs without the crutch, Stage A must be retrained with
  `"zero_prev_action": true` (binding track: please set it for revised representations).
- Until then: do NOT treat any learned closed-loop number (acceptance, GRPO, baselines, 4-way campaign) as evidence
  about the architecture; they are dominated by B-1.

### B-1 on the training pack itself (`scripts/ladder_t0_check.py`, raw `artifacts/runs/ladder_smoke/t0_check_sem_panda.json`)
latent_sem_v1 E+R, panda_pg2 rows, j=0, 96 rows each; normalized MSE (arm hold-still reference = arm_label_sq):
| input | rows | arm err | arm hold-still | grip err | grip pred / label | z norm |
|---|---|---|---|---|---|---|
| stored (teacher prev cmd in col 28) | t=0 (col 28 is 0 there too) | 0.112 | 0.114 | 0.314 | -0.44 / -1.0 | 97.7 |
| stored | t=1 | 0.005 | 0.079 | 0.0002 | -1.01 / -1.0 | 96.5 |
| stored | t>=5 | 0.005 | 0.082 | 0.005 | -0.36 / -0.35 | 35.4 |
| col 28 zeroed (= deployment) | t=1 | 0.144 | 0.079 | 0.319 | -0.44 / -1.0 | 96.3 |
| col 28 zeroed (= deployment) | t>=5 | 0.083 | 0.082 | 0.741 | -0.59 / -0.35 | 34.6 |
The only training rows whose col 28 is 0 are t=0, and exactly there the realizer has no skill: the acceptance "step-0"
symptom is the same bug, not a knot-timing/phase-0/normalization/clipping problem (knot times, phase and normalization
are identical between pack and runtime; the +-6 output clip was not checked separately because the error is already present in the raw network output on the pack). The large step-0 z norm
is E's (98 at t<=1 vs 35 later), identical in pack and runtime. The acceptance parity check matched features because at
t=0 the column is 0 in both; from t=1 on it differs (`scripts/ladder_feature_parity.py`).

## bug B-1: previous-action feature zeroed in the wrong column
- Node features are `[static(26) | q, qd, PREV_ACTION, anchor(3), axis(3), jp(3), jr(3), lever(3)]` (NODE_DIM 44), so
  prev-action is column **28**. Datasets were collected before D-021 with the teacher's previous 1-step command there.
  D-021's load-time fix in `rrp.learning.data.episode_samples` tests/zeroes column **2** (a static column, always 0),
  so it never fires: packs built since then (incl. `latent_pp_v3dart_s1_H16`) carry the teacher's previous command in
  column 28 of node features and of the node rows of the morph bank. The deployed featurizer always writes 0 there.
- Evidence (all on the peer store):
  - Feature parity: replay of stored episode `pick_place_panda_pg2_s0` through the current featurizer/teacher
    (`scripts/ladder_feature_parity.py artifacts/datasets/pick_place_primary_v3dart pick_place_panda_pg2_s0`): every
    bank, relation, q0, local sensor and action is identical except node/morph column 28 (t >= 1). Packed column 28 is
    non-zero in 66% of node entries (first 20k rows).
  - Stage-A system 0 on the TEACHER's own clean trajectory (R0 route with a shadow system 0 fed oracle packets, not
    executed; panda_pg2 seeds 3000000-1; `artifacts/runs/ladder_smoke/teacher_shadow*.jsonl`), normalized 1-step MSE vs
    teacher label:
    | prev-action input | arm | gripper | reference: arm hold-still error |
    |---|---|---|---|
    | 0 (current deployment) | 0.030 | 0.78 (close 1.0, transport 1.8, lower 2.0: it OPENS while carrying) | 0.018 |
    | own previous command (training-consistent) | 0.0017 | 0.028 | 0.018 |
    | training pack, same robot (`packed_check_sem_panda.json`) | 0.004-0.011 | 0.001-0.03 | 0.03-0.09 |
    With the deployed input, system 0 is worse than holding still on the arm and inverts the gripper while carrying.
- Fix (opt-in, default keeps running jobs/resumes bit-identical): `PackedChunkDataset(..., zero_prev_action=True)` /
  config key `"zero_prev_action": true` for `train_representation` / `train_latent_flow` zeroes column 28 on node rows at
  load (`rrp.learning.packed.PREV_ACTION_COL`; guard test `tests/unit/test_prev_action_col.py`). Verified: only node
  column 28 and morph rows < n_nodes column 28 change. Stage A and Stage B must be retrained with it for a
  deployment-consistent model. Alternatively deploy with the prev-action input = own previous command (ladder
  `--prev-action own`); this is training-consistent in form but not in semantics (teacher vs own command, the D-021 copycat
  concern) — the ladder measures both.
- Affects also: old direct-action baselines (dev5/dev6), GRPO bases, binding-track representations (unless they set the
  flag), dualarm pack (if built from pre-D-021 datasets; check column 28).

## ladder table (panda_pg2 unless noted; dev seeds = first 30 feasible from 3,000,000; 300 ticks; replan 8; NFE 8)
Raw: peer `artifacts/runs/ladder_v1/<robot>/<route>_<tag>.jsonl` (+ `.summary.json`); summarize with
`scripts/ladder_peek.py <files>`. `prev` = what the deployed featurizer puts in node column 28 (B-1): `zero` = current
deployment; `own` = system 0's own previous command (training-consistent form). Label error = normalized 1-step MSE of
system 0 vs the shadow teacher's command at the visited state; on R0 it is the shadow system 0 on the teacher trajectory.
| rung | system 0 | prev | success (Wilson 95%) | failed stage | min TCP-cube | arm / grip label err | track q (rad) / TCP (m) |
|---|---|---|---|---|---|---|---|
| R0 teacher (privileged) | shadow: sem_v1 R | own | 30/30 (0.89-1.0) | - | 0.003 | 0.0017 / 0.023 | 0.025 / 0.021 |
| R0 teacher | shadow: sem_v1 R | zero | 30/30 (running) | - | | | |
| R0 teacher | shadow: refit R @2k (B-1 fix) | zero | 8/8 | - | 0.003 | 0.0074 / 0.056 | 0.025 / 0.020 |
| R1 oracle (ORACLE DIAGNOSTIC) | sem_v1 R | zero | 0/30 (0-0.11) | approach 30 | 0.42 | (diverged) | 0.006 / 0.005 (barely moves: 0.014 rad/tick) |
| R1 oracle | sem_v1 R | own | 0/30 (0-0.11) | approach 30 | 0.28 | copycat drift | 0.073 / 0.10 |
| R1 oracle | refit R @2k | zero | 0/16 | approach 14, grasp 2 | 0.084 | | 0.015 / 0.020 |
| R1 oracle, re-anchored expert | refit R @2k | zero | 0/16 | approach 13, grasp 1, lift 2 | 0.095 | | 0.017 / 0.022 |
| R2 flow v2@24543 | sem_v1 R | zero | 0/30 (0-0.11) | approach 30 | 0.44 | | 0.007 / 0.007 |
| R2 flow v2@24543 | sem_v1 R | own | 0/30 (0-0.11) | approach 30 | 0.33 | | 0.032 / 0.031 |
| R2 flow v2@24543 | refit R @2k | zero | 0/16 | approach 15, lift 1 | 0.22 | | 0.035 / 0.038 |
parm6_tf3 (seeds 3000003..3000041, 30 feasible): R0 30/30 (track 0.006 rad / 0.007 m); shadow sem_v1 R on the teacher
trajectory: prev own arm/grip 0.0018/0.0017, prev zero 0.0087/0.53 (arm hold-still reference 0.0048) -> B-1 on a
second body too.

Tracking: the joint tracker follows every route's commands closely (R0 0.025 rad mean lag at teacher speeds); failures are
never tracker failures.

Oracle vs generated z at the same states (R2 rows, `oracle_cmp`): probes read the generated packet as well as the oracle
packet (held_by/acting_on/subtask 1.0, rel_pos 0.045 vs 0.043 m). With the OLD system 0 the action from generated vs
oracle z differs by only 0.002 (it barely reads z; copycat of col 28). With the REFIT system 0 the difference is 0.89
(normalized arm MSE; hold-still 0.017): once system 0 actually uses z, the generator's z (flow v2, itself trained with
the B-1 input) drives it very differently from the oracle z -> the generator is the second failure point.

R1 failure mode with the refit realizer (trace `artifacts/runs/ladder_smoke/oracle_ticks_zero_rz2k_re.jsonl`): the arm
moves toward the cube but swings sideways first and the tool tilts progressively (tool z-axis 20-25 deg off vertical by
t=40-90); it descends next to/onto the cube and pushes it. Compounding 1-step error (covariate shift) + off-manifold
packets; the teacher's relabel at those states is a large wrist correction (0.4-0.6 rad).

## fixes tested so far (all system-0 realizer variants share the FROZEN sem_v1 encoder unless noted)
| system 0 | on-teacher arm/grip err (prev 0) | R1 panda (30) | R1 panda re-anchored (30) | R1 parm6_tf3 (30) | R1 13 bodies x 24 (seeds 3.2M/3.3M) |
|---|---|---|---|---|---|
| sem_v1 R (B-1) | 0.030 / 0.78 | 0/30, approach 30 | - | - | - |
| refit @2k (col 28 zeroed) | 0.0074 / 0.056 | 0/16 | 0/16 | - | 1/312 (sawyer_pg2); min TCP-cube 0.11-0.55 m |
| + DAgger round 1 (`rz_sem_v1_dagger1`) | 0.0123 / 0.052 | running | 0/30: approach 19, grasp 11 | 0/30: approach 17, grasp 5, lift 5, transport 3 | 0/312; min TCP-cube 0.04-0.12 m; parm6 reach 11/24 |
DAgger round 1 = 13 bodies x 24 R1 re-anchored rollouts of the @2k realizer (seeds 3,200,000+; 94k learner states),
mixed 50/50 with the pack, 4k steps from the @4k refit (`configs/ladder/rz_sem_v1_dagger1.json`). It brings the hand to the
cube and, on parm6_tf3, carries the cube to the zone in 3/30 (placing fails), but it also teaches large wrist corrections
(cmd step 0.35 rad/tick on panda, tracking lag 0.11 rad) and raises the on-teacher error.

### fixed-packet disturbance (`latent_eval.disturbance_test`, panda_pg2, seeds 3000100+, 10 x joints 1 and 3, +0.12 rad,
packet held 8 ticks; final TCP deviation from the undisturbed rollout; raw `ladder_v1/panda_pg2/disturbance_oracle_*.jsonl`)
| system 0 (oracle packet) | closed-loop system 0 | open-loop deltas | replay of absolute targets (servo only) |
|---|---|---|---|
| sem_v1 R, prev 0 | 0.034 m | 0.042 m | 0.0007 m |
| sem_v1 R, prev own | 0.048 m | 0.061 m | 0.0010 m |
| DAgger 1 | 0.038 m | 0.058 m | 0.0052 m |
System 0 does NOT return to the packet's plan after a push: it realizes deltas relative to the CURRENT state, so a
disturbance persists (3-5 cm), whereas plain absolute-target replay recovers to <1 cm. Same mechanism as the drift under
compounding error. Proposed fix under test: an ANCHORED system-0 input — node column 28 (free after the B-1 fix) carries
the joint displacement since the packet's anchor state, so the realizer can express targets relative to the plan, not
only to the present (`realizer_anchor: true`; `rrp.control.latent_realizer.realizer_node_feats`).

## running now
- `ladder_rz_sem_v1_anchor` (anchored refit on frozen sem_v1 E, CPU), lease 1790375880_472857.
- `ladder_rz_sem_v1_dagger2` (DAgger round 2: r1+r2 buffers, from dagger1, CPU), lease 1790375896_89c2cb.
- `ladder_flow_sem_v2_b1fix_ft` (flow v2 fine-tuned 4k steps with col 28 zeroed, GPU), lease 1790374277_f0def8.
- next: Stage A retrained jointly with the fix + anchor (`configs/ladder/rep-latent_sem_b1fix_anchor.json`, GPU) and
  the pure 8k refit evaluation (`scripts/ladder_eval_rz.sh`).

## infrastructure notes
- Host `rrp ops run` is broken right now: every lease fails with systemd status 219/CGROUP ("Failed to create cgroup ...
  Cannot allocate memory"); a 1 GiB lease fails with "MemoryHigh out of range". All ladder work runs on the peer.
- `_prefetch` (forked collate workers) deadlocks when the parent already initialized torch on CPU-only or CUDA before
  forking (seen twice: a CPU refit and a flow train with `"prefetch": true`, both stuck on a futex with idle workers).
  Use the serial path (`prefetch_workers: 0` / no `prefetch`). On the contended peer GPU the prefetch refit ran at
  0.73 s/step, the serial CPU refit at 0.33 s/step.

## bug B-1: previous-action feature zeroed in the wrong column
- Node features are `[static(26) | q, qd, PREV_ACTION, anchor(3), axis(3), jp(3), jr(3), lever(3)]` (NODE_DIM 44), so
  prev-action is column **28**. Datasets were collected before D-021 with the teacher's previous 1-step command there.
  D-021's load-time fix in `rrp.learning.data.episode_samples` tests/zeroes column **2** (a static column, always 0),
  so it never fires: packs built since then (incl. `latent_pp_v3dart_s1_H16`) carry the teacher's previous command in
  column 28 of node features and of the node rows of the morph bank. The deployed featurizer always writes 0 there.
- Evidence (all on the peer store):
  - Feature parity: replay of stored episode `pick_place_panda_pg2_s0` through the current featurizer/teacher
    (`scripts/ladder_feature_parity.py artifacts/datasets/pick_place_primary_v3dart pick_place_panda_pg2_s0`): every
    bank, relation, q0, local sensor and action is identical except node/morph column 28 (t >= 1). Packed column 28 is
    non-zero in 66% of node entries (first 20k rows).
  - Stage-A system 0 on the TEACHER's own clean trajectory (R0 route with a shadow system 0 fed oracle packets, not
    executed; panda_pg2 seeds 3000000-1; `artifacts/runs/ladder_smoke/teacher_shadow*.jsonl`), normalized 1-step MSE vs
    teacher label:
    | prev-action input | arm | gripper | reference: arm hold-still error |
    |---|---|---|---|
    | 0 (current deployment) | 0.030 | 0.78 (close 1.0, transport 1.8, lower 2.0: it OPENS while carrying) | 0.018 |
    | own previous command (training-consistent) | 0.0017 | 0.028 | 0.018 |
    | training pack, same robot (`packed_check_sem_panda.json`) | 0.004-0.011 | 0.001-0.03 | 0.03-0.09 |
    With the deployed input, system 0 is worse than holding still on the arm and inverts the gripper while carrying.
- Fix (opt-in, default keeps running jobs/resumes bit-identical): `PackedChunkDataset(..., zero_prev_action=True)` /
  config key `"zero_prev_action": true` for `train_representation` / `train_latent_flow` zeroes column 28 on node rows at
  load (`rrp.learning.packed.PREV_ACTION_COL`; guard test `tests/unit/test_prev_action_col.py`). Verified: only node
  column 28 and morph rows < n_nodes column 28 change. Stage A and Stage B must be retrained with it for a
  deployment-consistent model. Alternatively deploy with the prev-action input = own previous command (ladder
  `--prev-action own`); this is training-consistent in form but not in semantics (teacher vs own command, the D-021 copycat
  concern) — the ladder measures both.
- Affects also: old direct-action baselines (dev5/dev6), GRPO bases, binding-track representations (unless they set the
  flag), dualarm pack (if built from pre-D-021 datasets; check column 28).

## ladder table (panda_pg2 unless noted; dev seeds = first 30 feasible from 3,000,000; 300 ticks; replan 8; NFE 8)
Raw: peer `artifacts/runs/ladder_v1/<robot>/<route>_<tag>.jsonl` (+ `.summary.json`); summarize with
`scripts/ladder_peek.py <files>`. `prev` = what the deployed featurizer puts in node column 28 (B-1): `zero` = current
deployment; `own` = system 0's own previous command (training-consistent form). Label error = normalized 1-step MSE of
system 0 vs the shadow teacher's command at the visited state; on R0 it is the shadow system 0 on the teacher trajectory.
| rung | system 0 | prev | success (Wilson 95%) | failed stage | min TCP-cube | arm / grip label err | track q (rad) / TCP (m) |
|---|---|---|---|---|---|---|---|
| R0 teacher (privileged) | shadow: sem_v1 R | own | 30/30 (0.89-1.0) | - | 0.003 | 0.0017 / 0.023 | 0.025 / 0.021 |
| R0 teacher | shadow: sem_v1 R | zero | 30/30 (running) | - | | | |
| R0 teacher | shadow: refit R @2k (B-1 fix) | zero | 8/8 | - | 0.003 | 0.0074 / 0.056 | 0.025 / 0.020 |
| R1 oracle (ORACLE DIAGNOSTIC) | sem_v1 R | zero | 0/30 (0-0.11) | approach 30 | 0.42 | (diverged) | 0.006 / 0.005 (barely moves: 0.014 rad/tick) |
| R1 oracle | sem_v1 R | own | 0/30 (0-0.11) | approach 30 | 0.28 | copycat drift | 0.073 / 0.10 |
| R1 oracle | refit R @2k | zero | 0/16 | approach 14, grasp 2 | 0.084 | | 0.015 / 0.020 |
| R1 oracle, re-anchored expert | refit R @2k | zero | 0/16 | approach 13, grasp 1, lift 2 | 0.095 | | 0.017 / 0.022 |
| R2 flow v2@24543 | sem_v1 R | zero | 0/30 (0-0.11) | approach 30 | 0.44 | | 0.007 / 0.007 |
| R2 flow v2@24543 | sem_v1 R | own | 0/30 (0-0.11) | approach 30 | 0.33 | | 0.032 / 0.031 |
| R2 flow v2@24543 | refit R @2k | zero | 0/16 | approach 15, lift 1 | 0.22 | | 0.035 / 0.038 |
parm6_tf3 (seeds 3000003..3000041, 30 feasible): R0 30/30 (track 0.006 rad / 0.007 m); shadow sem_v1 R on the teacher
trajectory: prev own arm/grip 0.0018/0.0017, prev zero 0.0087/0.53 (arm hold-still reference 0.0048) -> B-1 on a
second body too.

Tracking: the joint tracker follows every route's commands closely (R0 0.025 rad mean lag at teacher speeds); failures are
never tracker failures.

Oracle vs generated z at the same states (R2 rows, `oracle_cmp`): probes read the generated packet as well as the oracle
packet (held_by/acting_on/subtask 1.0, rel_pos 0.045 vs 0.043 m). With the OLD system 0 the action from generated vs
oracle z differs by only 0.002 (it barely reads z; copycat of col 28). With the REFIT system 0 the difference is 0.89
(normalized arm MSE; hold-still 0.017): once system 0 actually uses z, the generator's z (flow v2, itself trained with
the B-1 input) drives it very differently from the oracle z -> the generator is the second failure point.

R1 failure mode with the refit realizer (trace `artifacts/runs/ladder_smoke/oracle_ticks_zero_rz2k_re.jsonl`): the arm
moves toward the cube but swings sideways first and the tool tilts progressively (tool z-axis 20-25 deg off vertical by
t=40-90); it descends next to/onto the cube and pushes it. Compounding 1-step error (covariate shift) + off-manifold
packets; the teacher's relabel at those states is a large wrist correction (0.4-0.6 rad).

## fixes being tested
1. Realizer refit on frozen E with col 28 zeroed (B-1): `configs/ladder/rz_sem_v1_b1fix_ft.json`, lease 1790370807_bd05c2.
2. System-0 DAgger: R1 (re-anchored expert) rollouts of the current refit realizer on all 13 source-train bodies, seeds
   3,200,000+ (24 feasible each), buffer = oracle posterior at each replan + learner-visited states with the shadow
   teacher's command; mixed 50/50 with the pack in `refit_realizer` (`"dagger": [...]`). Collection leases
   1790373150_81c015, 1790373151_61c3bd, 1790373151_2e8975, 1790373152_9c8793 -> `artifacts/runs/ladder_dagger_r1/<robot>.npz`.
3. Generator: flow retrain on the same frozen E with the B-1 fix (`configs/ladder/flow_sem_v2_b1fix.json`, 20k steps),
   lease 1790373182_289454 -> `artifacts/runs/ladder_flow_sem_v2_b1fix/`.

## method (code: `src/rrp/evaluation/ladder.py`, CLI `scripts/ladder.py`)
- Matched scenes: `feasible_seeds(robot, 3_000_000, n)` (same list for every rung), `n_distractors = seed % 3`.
- R0 `teacher`: scripted teacher (privileged) -> joint-target tracker. R1 `oracle`: ORACLE DIAGNOSTIC, frozen Stage-A E
  encodes the teacher's next 16 commands rolled out from the current closed-loop state (snapshot/restore; normalized with
  q0 at t, fp16-rounded like the pack), packet source `target_encoder_oracle`; frozen system 0 realizes it every tick.
  R2 `generated`: system-i flow (ODE Euler, NFE 8) from public observations. Replan every 8 ticks, validity 0.8 s,
  300 ticks.
- A shadow teacher advances once per tick at the executed state (DART-like) to give phase labels, the relabelled teacher
  command (label error `lab_err_*`, normalized MSE; `lab_step_arm` = hold-still reference), and in R2 the same-state oracle
  packet (z distance, system-0 action from each, probe readouts). It never controls R1/R2 except through E in R1.
- Tracking: `track_q_rad` mean |q_cmd - q_achieved| after the 50 ms tick; `track_tcp_m` |FK(q_cmd) - TCP achieved|.
- Failure stage from privileged geometry: approach (TCP within 2.5 cm of cube) -> grasp (held truth) -> lift (+4 cm while
  held) -> transport (cube within 4 cm of zone xy while held) -> place (privileged success).
- Checkpoint identities (sha256):
  - `latent_sem_v1/representation.pt` 8946aa9d2d0da5fca149786e867a4138ca9596976258abde2ef20aef4708db04 (ls-8db814f941f5)
  - `latent_nosem_v1/representation.pt` 49cea78db52c71c0cdbf3894453d0cdca59a5d329bbfac7d3a1a3b23388dd4cf
  - `ladder_ckpts/flow_latent_sem_v2_step24543.pt` (frozen copy of flow_latent_sem_v2/policy_interrupted.pt)
    ed886740e76ffe5f2638a169fc93c60164b5bafeddfbdcf1ae37cc1de4ab74c2
  - `grpo_base_snapshots/flow_latent_sem_v2_step22000.pt` bb6f454b93c4471708768adab34d9e05fef78da001738cd4da53d5c06a9b5df0

## log
- [verified] smoke: R0 panda 2/2 (track_q 0.026 rad, track_tcp 1.9 cm); R1 (prev=0) 0/2, never approaches (min TCP-cube
  0.47 m), label error 2.1 -> led to B-1. Commands in this file's bug section; raw `artifacts/runs/ladder_smoke/`.
- [running] lease 1790369958_1a7c76 `ladder_r01_panda`: R0 (shadow, prev own), R1 prev own, R1 prev zero; n=30 panda_pg2
  -> `artifacts/runs/ladder_v1/panda_pg2/{teacher_shadow_own,oracle_own,oracle_zero}.jsonl`.
- [running] lease 1790369958_fe952f `ladder_r2_panda`: R2 flow v2 step 24543, prev own / zero -> `.../generated_v2s24543_{own,zero}.jsonl`.

## sprint (2026-09-25 19:00 → 03:00 PDT, agent sprint_latent)
Scope: bring the corrected latent path (B-1 fixed) as close as possible to competent on pick_place (panda_pg2, parm6_tf3,
dev seeds from 3,000,000) with honest ladder evidence. Oracle rows are ORACLE DIAGNOSTIC (teacher-encoded packet).
- [running] DAgger round 2 on jointfix+jfdag1: collection `scripts/ladder_dagger_collect.sh` with
  `ladder_rz_jointfix_dagger1/representation.pt`, 13 bodies x 24, SEED=3300000, 4 peer CPU leases
  -> `artifacts/runs/ladder_dagger_jf2/` (R1 on training bodies: 0/312; min TCP-cube 4-6 cm on parm*/panda, 9-25 cm on
  parm7/sawyer/ur5e). Refits: `configs/ladder/rz_jointfix_dagger2.json` (jf1+jf2 buffers, dagger_frac 0.5, init from
  jfdag1) and D-049 fix A `rz_jointfix_dagger2_df08.json` (same, dagger_frac 0.8).
- [running] host GPU flow on the jointfix bundle (zero_prev_action, normalize_target, 20k steps):
  `configs/ladder/flow_jointfix.json`, host lease 1790388397_d67260 -> `artifacts/runs/ladder_flow_jointfix/` (host).
- 19:25 lead re-prioritized (sprint_bc: plain BC with the B-1 fix is competent, 25-28/30; the shadow teacher FSM is stale
  on learner-visited states, so R1-with-shadow and shadow-DAgger labels are confounded). Shadow-teacher DAgger stopped:
  round-2 collection (above) is recorded; refit `rz_jointfix_dagger2` (df 0.5) stopped at 1.3k/4k steps; only the
  df 0.8 refit (D-049 fix A) is allowed to finish for the record. The host broker stopped every job at ~19:15
  (disk below reserve: 300 GB free vs 322 GB reserve); the jointfix flow restarted on the host at 19:26 (lease
  1790389550_4baed9, packet_tau_min 0.6 added, 0.25 s/step) with a watcher `scripts/ladder_flow_watch.sh` that snapshots
  every 4k steps and runs R2 on the peer (tags `zero_flowjf_s<step>`).
- [completed] **Stateless localization on BC-visited states** (`scripts/ladder_localize.py`): BC (learned:direct1701_u12000)
  runs the 30 matched dev seeds; each episode is replayed exactly (replay consistent 30/30 on both robots); every 8
  ticks z_bc = E(the chunk BC actually executed next) (ORACLE DIAGNOSTIC, no teacher state); shadow system 0 realizes it;
  1-step normalized arm MSE vs BC's executed command. Raw: peer `artifacts/runs/ladder_localize/<robot>/bc_direct1701_u12000__<tag>.json`.
  | system 0 | panda_pg2 arm / grip | parm6_tf3 arm / grip | hold-still arm ref (panda / parm6) | err at j=0 vs j=1..7 (panda) |
  |---|---|---|---|---|
  | jointfix (Stage A joint, B-1 fixed) | 0.0049 / 0.063 | 0.0035 / 0.058 | 0.0114 / 0.0047 | 0.011 vs 0.0035-0.0049 |
  | jfdag1 (shadow DAgger r1) | 0.0094 / 0.062 | 0.0062 / 0.044 | same | 0.014 vs 0.0078-0.0099 |
  BC success in pass 1: 24/30 panda, 26/30 parm6. Reading: the jointly trained system 0 realizes packets encoding BC's
  chunks with ~40% (panda) / ~75% (parm6) of the hold-still error, and its first tick after each new packet (j=0) is as
  bad as holding still. Shadow-teacher DAgger DOUBLED the error on BC-visited states (consistent with stale labels).
  Closed-loop version (R1 with the stateless BC expert, `--oracle-expert bc`, `scripts/ladder_eval_orcbc.sh`): running.
