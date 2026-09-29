# project status — structured-psi0-latent-diffusion-dynamics (formerly relational robot policy)
Open questions / planned experiments: docs/experiments_roadmap.md (D-126). Stated-but-unimplemented code: docs/intentions_backlog.md (D-123).

Ψ₀ line (W10): in this repo since D-140 (`rrp.policies.psi0`, `rrp.envs.simple`); notes research/tracks/psi0.md, P-decisions = appendix P of research/decisions.md.

Updated: 2026-09-27 21:30 PDT (records agent, D-123 item 10). Overall: **in_progress** (not complete). Plan: `docs/strategy.md` (D-094).
Evidence: `research/reports/evidence_matrix.md`. Decisions: `research/decisions.md` (D-001..D-123). Open intentions: `docs/intentions_backlog.md` (D-123).

## current state (2026-09-27 21:30)
Strategy (D-094): physics credibility first, one pipeline for all bodies, additive-then-subtractive restructure. Since D-115 the
host does no heavy compute (editing, git, unit tests, orchestration only); everything heavy runs on the peer (admission D-106,
memory declarations ≥ 1.35 × measured peak, D-117). Peer priority: W8 > W7 / arm lineages > W10 > W12 (D-121, D-122).
This table mirrors the status column of `docs/strategy.md`.

| id | workstream | state | status (decisions) | where |
|---|---|---|---|---|
| W1 | physics realism: contact v2, reward schedule, slip gate; actuators/latency | running | round complete: contact v2 installed for anymal_c, go2 (clearance floor), t1 w8d (sourced limits); sourced torque limits are the default (`sourced_v1`); permanent standing; h1/g1 parked; actuator dynamics and latency opt-in (D-101, D-103, D-107, D-114) | research/tracks/contact.md |
| W2 | repo hygiene (phase 0) | verified | 2026-09-26, main 3c796fd | research/tracks/hygiene.md |
| W3 | provenance and contracts (phase 1) | verified | 2026-09-26, main 510f052 | research/tracks/provenance.md |
| W4 | package restructure with shims (phase 2) | completed | 2026-09-27, P6 f1db76d; legacy packages are shims only | research/tracks/restructure.md |
| W5 | unified pipeline + run-dag (phases 3–4) | verified (arm) | arm parity row-identical (D-096); legged used in production by W8; dual skeleton only; `ladder.py` main → library pending | research/tracks/pipeline.md |
| W6 | robustness sweeps + motion-quality gates | verified | harness + metrics (D-108); variant-level anymal_c sweep (D-112); gates in code, `validate_tracker` stage (D-114); watchdog accounting fixes (D-116, D-117) | research/tracks/robust.md |
| W7 | arm expert (teacher v2, grasp v2/v2.1), then GRPO with anchor evals | running | teacher v2 (D-097), v2 BC (D-102), grasp v2 (D-110), grasp v2.1 + DART diagnosis (D-118), v6dart accepted conditional on the v6 BC expert (D-121); v6 BC 1701 training; grasp_v2 re-eval of old routes running (interim); GRPO not started | research/tracks/armexpert.md |
| W8 | legged regeneration on contact v2 | running | anymal_c done (D-105; robustness D-112); go2 done (D-113); t1 on sourced-limit w8d under the D-113 gate exception: 39/43 nodes, learned routes mostly fall (interim) | research/tracks/legged8.md |
| W9 | held-out target bodies; loco-manipulation | planned | not started; top backlog item (D-123 #1) | docs/intentions_backlog.md |
| W10 | Ψ₀ / SIMPLE benchmark (migrated into rrp, D-140) | paused (D-140) | step 1: 3/6 released checkpoints reproduce (D-120); step 2 TabletopGraspMP L0: released 20/20, Ψ₀ direct 19/20, Ψ₀ + structure 0/20 (D-141: structured route likely has an integration bug; diagnosis next) | research/tracks/psi0.md |
| W11 | rrp as an installable core for psi1z | verified | 2026-09-26, main b7dc677 (rrp.core API 1.0, py3.11 + 3.12); psi1z pin bumped to 68a6657 (P-020) | research/tracks/core.md |
| W12 | feature-centric coordination (anchor-relative packets, contact-event knots) | implementing | phase A (design + code, no heavy compute); compute after the arm v6 lineages (D-122) | docs/strategy.md W12 |
| W13 | HUMANOID program (owner top priority) | planned | P0 plan + sealed humanoid split committed (D-138); P1a GPU-sim bake-off and P1b per-body trackers next | research/tracks/humanoid.md |
| R0 | arm seed-2 replication | completed | D-095 | research/tracks/ladder.md |
| R1 | arm nosem recipe ablation | completed | D-099 | research/tracks/ladder.md |

## evidence summary (through D-123)
Earlier evidence (D-044..D-094) is summarised in the history section below and in `research/reports/evidence_matrix.md`. Results since D-095:
- **Arm, semantic supervision (2 seeds, grasp_v1 physics).** D-095: nosem is below both semantic lineages in all 8 body × seed cells
  (deployable successes pooled over both seeds: frozen sem 247/480, semfix 253/480, nosem 29/480; sem vs semfix is seed noise).
  D-099: neither z-noise 0 nor keeping qd rescues seed-1 nosem (0–3/240 vs sem 146/240), so the deficit arises upstream of the
  system-0 recipe. These numbers were measured under grasp_v1, where grasps held by interpenetration (D-108).
- **Arm physics.** Teacher v2 (minimum-jerk, touch-confirmed grasp) 4810/4811 feasible episodes (D-097; revised 4811/4811, D-102);
  grasp contact v2 removes interpenetration (held penetration tf3 3.3 → 0.09 mm; episodes > 3 mm 2407 → 0) (D-110); grasp v2.1 + phase-gated
  DART: v6dart passes every dataset gate (penetration ≤ 3 mm 0.9943) and is accepted conditional on the v6 BC expert (D-118, D-121).
- **Arm routes re-evaluated under grasp_v2 (INTERIM, 35/56 cells, not final; D-121).** The semantic routes hold or improve (semfix s1
  parm6 45 → 62/90, s2 38 → 60/90; frozen sem s2 44 → 66/90); frozen sem s1 collapses on tf3 (parm6 70 → 9/90); nosem stays far below
  (s1 1/90, s2 19/90 on parm6); BC under grasp_v2 parm6 68/90. The re-eval is still running; do not cite it as final.
- **Legged contact v2 (W1).** Skating fixed on every accepted v2 tracker (stance slip t1 0.86 → 0.151, go2 0.26 → 0.02, anymal_c 0.34 → 0.03;
  D-101); go2 passes with a permanent clearance floor, t1 learns a stepping turn in place (D-103); t1 torque limits were 2–3× the
  manufacturer's and g1 hip roll 58% too strong: sourced limits are now the body-model default (`sourced_v1`, D-107).
- **Legged semantic context-halt replicates under contact v2 on 2/2 bodies × 3/3 seeds.** anymal_c (D-105): halt Δforward semfix −0.19 m
  [−0.26, −0.13] vs nosem +0.21 m [+0.14, +0.28], every seed ordered (p = 0.05); semfix also steers more toward context goals.
  go2 (D-113): −0.29 m [−0.34, −0.24] vs +0.37 m [+0.32, +0.42], difference −0.66 m (contact v1: −0.13 vs +0.46, D-090); goal steering
  not ordered. R2 success 27–30/30 per go2 model; anymal_c 78/90 vs 83/90 by the privileged end check, 88–90/90 by public success.
- **t1 on sourced limits (interim, W8 track notes).** Labelled "t1 dataset fails D-112 slip gate (86.2% < 95%), tracker w8d fails lab forward
  0.72 …" (D-113 exception). Learned t1 routes mostly fall at gait onset (BC 0/30); the fixsem-vs-nosem halt comparison is not measurable on t1.
- **Robustness (W6).** anymal_c: the semantic route is slightly LESS robust than nosem; at variant level (3/3 training seeds) semantic
  supervision costs −0.046 [−0.056, −0.036] public success averaged over 33 single-factor levels and moves no break-point (friction 0.4, gain
  0.7, push 1.0 m/s, terrain 8 cm) (D-108, D-112). Arm robustness under grasp_v1 was not meaningful (interpenetration, D-108).
- **Gates (W6).** Tracker, dataset (legged and arm) and policy gates are enforced in code (D-112, D-114). anymal_c and go2 trackers and
  datasets pass; t1 w8d fails CoT/joint margin; g1_src is a real stomper (4.09 BW); arm joint margin report-only on procedural bodies.
- **Ψ₀ reproduction (W10).** The released checkpoints reproduce on our Isaac-5.1/aarch64/path-traced stack for 3 of 6 tasks
  (TabletopGraspMP 10/10, BendPickMP 10/10, HandoverTeleop 7/10); XMovePick / XMoveBendPick / LocomotionPickBetweenTables do not, consistent
  with a render gap on locomotion cues (D-120). Released TabletopGraspMP 20/20 in step 2 (D-120 addendum). Source: third-party released
  checkpoint (upstream Ψ₀), not our model.
- **Still not tested:** the latent route on the sealed target bodies (arm and legged held-out); fair baselines there (BC seeds 1702/1703,
  SFT budgets); dual-arm learned models; GRPO with anchor evals (D-123 top 10).

Frozen arm route (grasp_v1): system i ladder_flow_jointfix_gdag2h (sha d0d64918…) → system 0 ladder_rz_jointfix_gendag3_noqd (sha f60cde41…).
Always pass the system-0 bundle explicitly (research/tracks/ladder.md "SPRINT BEST ROUTE FINAL"). Under grasp_v2 its seed-1 instance
collapses on tf3 (interim, D-121).

## running work and resume pointers (2026-09-27 21:30)
Never sync into a peer dir with running jobs; never stop another agent's lease.
| work | owner / coordinator | state | resume |
|---|---|---|---|
| W8 t1 sourced-limit lineage | legged8 agent; host `run-dag dags/legged_v2_t1sl.yaml --retry-failed` (cwd ~/work/rrp-wt/legged8), peer dir wt/legged8w2 | running: 39/43 nodes done, nosem s2 evals/edits last | research/tracks/legged8.md "t1 sourced-limit results"; rerun the same run-dag command (idempotent ledger `artifacts/runs/legged8/_dags/legged_v2_t1sl/ledger.json`) |
| W7 v6 BC expert, seed 1701 (then 1702) | arm agent; `../armexpert_bcv6_launch.sh` (peer, GRASP v2.1) | running since ~20:50 | research/tracks/armexpert.md "v6dart"; condition of D-121 (must match the v2 BC expert) |
| W7 grasp_v2 re-evaluation of the old arm routes | arm agent; `scripts/armexpert_gc2_reeval.sh` (peer CPU), pass 2 (21 jobs) from 21:07 | running (35/56 cells in D-121) | research/tracks/armexpert.md "D-110 regeneration … re-evaluation" |
| W7 arm lineage set on v6dart (semfix/nosem × 2 seeds) | arm agent | planned: after the v6 BC passes its condition | D-110 (3), D-121; the v4dart set (`arm_lineage_v2`, 4/100 nodes) is stopped for good (D-110) |
| W10 step 2 (TabletopGraspMP direct vs structured) | psi0mig agent | evals done (D-141); structured 0/20 under offline diagnosis | research/tracks/psi0.md |
| W12 phase A | w12 agent | implementing (host-light) | D-122; phases B–D after the arm v6 lineages |
| records (this refresh) | records agent | completed on merge | docs/intentions_backlog.md "Repo structure / docs" |

## resolved environment
- repo: `~/work/relational-robot-policy` (GitHub: JacobFV/structured-psi0-latent-diffusion-dynamics, renamed 2026-09-25) (host `Dell-gb10-1`, aarch64 GB10). Handoff in `docs/handoff/`.
- peer: `gb10-direct` (hostname promaxgb10-4dfb, direct link). Workspace (RAM-backed, D-003): `/dev/shm/rrp-brandonin/{repo,venv,cache,bin}`.
- host venv: `.venv` (CPU only). peer venv: `/dev/shm/rrp-brandonin/venv` (torch cu130, mujoco).
- enforced parents: host broker 2 CPU / 8 GiB / 0 GPU, `gpu_authorized: false` (D-115: no heavy compute on the host); disk reserve 100 GB (D-086). Peer: all of it for this project (D-008/D-026) but admission-capped at 108 GiB declared RAM+GPU, 19.97 CPU, 8 GPU slots (D-106); declare ≥ 1.35 × measured peak (D-117). See `configs/resources.local.json`.
- peer store: cold run dirs live on the peer disk behind symlinks (`~/rrp-peer-data/runs-archive/`, D-111); watchdog counts project memory as anon+shmem+kernel (D-116/D-117).
- watchdogs: `rrp-watchdog-host.service`, `rrp-watchdog-peer.service` (user services inside rrp-control.slice).

## how to run anything (always under a lease)
```bash
cd ~/work/relational-robot-policy
PYTHONPATH=src python3 -m rrp.cli ops status
PYTHONPATH=src python3 -m rrp.cli ops run --cpu 1 --mem 2G --label NAME -- <cmd>
# peer (agents: export RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/<track> first, and use scripts/peer_run.sh; see AGENTS.md):
scripts/peer_sync.sh push
ssh gb10-direct 'cd /dev/shm/rrp-brandonin/repo && PATH=/dev/shm/rrp-brandonin/bin:$PATH PYTHONPATH=src RRP_NODE=peer RRP_REPO=$PWD python3 -m rrp.cli ops run --gpu --gpu-mem 8G --cpu 4 --mem 16G --label NAME -- /dev/shm/rrp-brandonin/venv/bin/python ...'
```

## completed / verified
- P02/P03 resource safety (cgroup enforcement, broker, watchdog; peer = whole machine per D-008, 3 GPU slots D-009).
- P01, P04-P06 contracts + task runtime; P07/P08 sim + labelled scripted teacher; P10 service; P11 UI (15/15 browser tests, artifacts/video/workbench-demo.webm).
- P09 (arms): procedural arm family (5/6/7 DOF), menagerie panda/fr3/ur5e/sawyer/xarm7/z1/lite6, two gripper modules with adapter standoffs; teacher ledger artifacts/assets/teacher_validation_pick_place_20seeds.txt.
- Frozen split research/splits/primary_v1.json; dataset artifacts/datasets/pick_place_primary_v1 (peer): 3791 success / 106 teacher failures / 453 infeasible.

- Dual-arm track (verified, scripted_teacher only): support_insert + handover scenarios (`src/rrp/sim/dual_scenarios.py`, `tasks/handover.json`), DualSession public estimators, dual teachers, ALOHA import, MultiFeaturizer (feat-multi-v1+feat-v2); teacher v2 validation 16 SI pairs / 9 handover pairs x 30 seeds (`artifacts/assets/dual_teacher_validation/*_v3*`, D-019); dataset `artifacts/datasets/support_insert_primary_v2` (teacher v2; v1 archived; split `research/splits/primary_v1_support_insert.json`); functional composition report `research/reports/functional_composition.md`; details `research/reports/dual_arm_tasks.md`.

## parallel tracks as of 2026-09-25 (historical; current table above) (D-033; brief: research/tracks/BRIEF.md; each track's resume file: research/tracks/<track>.md)
| track | worktree / branch | goal | blocks |
|---|---|---|---|
| lead chain | main checkout, peer /dev/shm/rrp-brandonin/repo | stage B v2/v3 flows + dev eval/disturbance/videos | — |
| binding | ~/work/rrp-wt/binding, track/binding | fix D-032 counterexample failure (binding-sensitive stage A) + flows | four-way latent cells |
| acceptance | ~/work/rrp-wt/acceptance | causal edits, composition, latency (reusable CLI) | — |
| baselines | ~/work/rrp-wt/baselines | sealed latent_slice1 cells for direct-action + codec baselines; aggregation | — |
| grpo | ~/work/rrp-wt/grpo | packet-policy GRPO (system i only) | target RL stage |
| dualarm | ~/work/rrp-wt/dualarm | support_insert/handover on latent path (M=2 assemblies) | — |
| legged_vlm | ~/work/rrp-wt/legged_vlm | legged/humanoid + VLM system II on latent path | — |
Host data mirror: ~/work/rrp-data/datasets only (packed removed, D-034: host disk reserve); packed-data training runs on the peer.

## blockers and known limits
- Legged: all results through D-092 use contact v1 (trackers skate, D-093). h1 cannot turn in place; g1 has no sourced-limit tracker that passes the gates (g1_src stomps, D-114); t1 w8d fails the lab gate and the t1 dataset fails the D-112 slip gate (D-113 exception).
- Arm: every arm result before D-110 used grasp_v1 (grasps held by interpenetration, D-108); grasp_v2 re-evaluation is interim (D-121). Procedural arms touch joint limits (joint margin report-only; limit-aware IK not implemented, D-114).
- Dual arm: panda+tf3 insertion-arm target teacher-weak (3/30, 8/30, D-019); sawyer support / sawyer->panda_tf3 handover pairs fail; the public insertion-depth estimator can false-positive on jams (D-018).
- Host: no heavy compute at all (D-115). Peer: disk ~97% used (check before large downloads); memory pressure sheds leases (declare ≥ 1.35 × peak, D-117).
- Open intentions (44 items, top 10 ranked): `docs/intentions_backlog.md` (D-123).

## resume
Read this file, `docs/strategy.md`, `research/decisions.md` (latest entries) and the relevant `research/tracks/<track>.md`.
Check `rrp ops status` on both nodes and `systemctl --user list-units 'rrp-*'` before starting anything; attach to running
owned jobs instead of duplicating. Budgets: `ops/resource-ledger.jsonl` on each node.

# history (kept for accounting; superseded by the sections above)

## state as of 2026-09-26 18:40 (historical; superseded by "current state" above)
Strategy adopted (D-094): physics credibility first, one pipeline for all bodies, additive-then-subtractive restructure.
Workstream owners, gates and sequencing live in `docs/strategy.md`; this table mirrors its status column.

| id | workstream | status | where |
|---|---|---|---|
| W1 | physics realism: contact v2, staged reward schedule, slip gate; then actuators/latency | running | research/tracks/contact.md (t1, h1 trackers on the host) |
| W2 | repo hygiene (phase 0) | completed | research/tracks/hygiene.md |
| W3 | provenance and contracts (phase 1) | implementing | docs/strategy.md §W3 |
| W4 | package restructure with shims | planned (after W2, W3) | docs/architecture.md |
| W5 | unified pipeline + DAG orchestration | planned (after W4, R0) | |
| W6 | robustness sweeps + motion-quality gates | planned (after W1 trackers) | |
| W7 | arm expert smoothing, then GRPO with anchor evals | planned (after R0) | |
| W8 | legged regeneration on contact v2 | planned (after W1 gate, W3) | |
| W9 | held-out target bodies; loco-manipulation | planned | |
| R0 | arm seed-2 replication of D-091 (sejf2 / sfjf2 / nsjf2) | running | research/tracks/ladder.md "ARM SEED-2 REPLICATION" |

Legged results through D-092 use contact model v1 (trackers skate, D-093); legged data is not regenerated until the
v2 trackers pass the W1 gate. Do not sync into peer dirs with running jobs (`/dev/shm/rrp-brandonin/wt/ladder`, `repo`).

### evidence summary as of 2026-09-26 (through D-095)
- Supported, on DEPLOYABLE routes (no teacher/oracle/BC at run time): task → packet → behaviour.
  - go2: goal direction (D-071).
  - parm6 arm: goal + binding edits (D-074, D-075; 80 seeds).
  - panda arm: binding redirection (D-077).
  - BC ignores the binding edit (D-065).
- Competence: the latent route matches BC on go2 and hexapod6 (D-070, D-076), and on t1 only for nosem (D-079, D-082) until the D-085 fix; with it t1 fixed sem ≈ nosem (81/90 vs 83/90, D-087).
  - Arm is partial: frozen route pooled 36/90 panda_pg2, 70/90 parm6_tf3 (D-078, D-080).
- Semantic supervision (after the D-085 bounded-NLL fix): essential on the arm (both training seeds: nosem 29/480 vs frozen sem 247/480 vs bounded-NLL sem 253/480 deployable successes; nosem lowest in every body × seed cell; D-089, D-091, D-095); on go2/hexapod6 a task-context "halt" slows only the sem packet, replicated in direction over 3 seeds (D-088, D-090), and pooled-only on the t1 humanoid (D-092); t1 success equal (81/90 vs 83/90, D-087). The D-084 "sem hurts" reading was the D-085 defect.
- Not tested: the latent route on the sealed target bodies. BC: new gripper 78–85/100, unseen xarm7 0/100 (D-064). g1 has no competent BC.
- Root-cause fixes along the way: B-1 prev-action column (D-044/045); system-0 velocity-copy shortcut (D-056); stale stateful teacher as oracle/DAgger expert (D-050); compatibility-ID fingerprinting (D-038).

Frozen arm route: system i ladder_flow_jointfix_gdag2h (sha d0d64918…) → system 0 ladder_rz_jointfix_gendag3_noqd (sha f60cde41…).
Always pass the system-0 bundle explicitly (research/tracks/ladder.md "SPRINT BEST ROUTE FINAL"). Legged: research/tracks/legged_vlm.md
"LEGGED RESEARCH RESULT FINAL". Semantic suite: research/tracks/acceptance.md "SPRINT SEMANTIC RESULTS". BC: research/tracks/baselines.md.

## END OF DEMO SPRINT (2026-09-26 ~04:00 PDT; historical) — sprint state and resume
Demo page (private artifact, owner-shareable): https://claude.ai/artifact/1LYxCtzDDCdEFvbrZJ83ot. Source: docs/demo/artifact.html
(standalone: docs/demo/index.html), built by scripts/demo/build_page.py from raw outputs (resume steps: research/tracks/demo.md).
Single current evidence statement: research/reports/evidence_matrix.md. The decisions log (D-044..D-092) has every result, with raw paths.

Evidence summary and frozen routes: moved to the top of this file.

Completed after sprint end: t1 seeds 2–3 (D-084); like-for-like BC control on parm6 (D-083). Nothing is running as of 05:35 (all sprint agents finished).
On hold (not started): seeds 1702/1703 and SFT budgets of the sealed four-way campaign; GRPO on target bodies; dual-arm and VLM training.
Most valuable next steps:
1. A nosem counterpart of the frozen arm route (same recipe), to isolate semantic supervision on the arm.
2. Latent route on the sealed target bodies (four-way comparison).
3. System-0/approach fixes for panda.
Operational notes: the host is shared. External memory pressure (other projects' processes) sheds our host jobs; the disk reserve is 100 GB (D-086; 300 GB at the time, D-036).
Launch loops must check exit codes and be bounded (D-061).

## DEMO SPRINT (2026-09-25 19:00 → 2026-09-26 05:00 PDT; user: "bring back as many subagents as you need ... no stopping", ≥80% host, 100% peer)
Plan and rules: research/tracks/BRIEF.md "DEMO SPRINT". Agents and their outputs:
| agent | worktree / notes | delivers |
|---|---|---|
| sprint_latent | ~/work/rrp-wt/ladder, research/tracks/ladder.md "SPRINT BEST ROUTE" | best B-1-fixed route: binding v4 oracle ladder, DAgger rounds, flows (host + peer GPU), R0/R1/R2 table |
| sprint_semantic | ~/work/rrp-wt/acceptance, acceptance.md "SPRINT SEMANTIC RESULTS" | approach-level semantic edits (rebind, goal, arm swap, plus controls); sem vs nosem; edit videos |
| sprint_bc | ~/work/rrp-wt/baselines, baselines.md "SPRINT BC RESULT" | plain BC positive control (B-1 fixed) with a learning curve, competence and videos |
| sprint_demo | ~/work/rrp-wt/demo, docs/demo/index.html | demo page, videos, evidence matrix (owner for the sprint) |
The lead publishes the page, audits the claims and rebalances resources (a watchdog alerts when a GPU is underused).

## now (2026-09-25 17:10) — lead working solo (no subagents); B-1 fix pipeline-wide (D-045); refits on the frozen v1 encoder fail (D-046)
Deciding experiments, all deployment-consistent (zero_prev_action):
- Stage A retrained jointly with fix + anchored system 0: peer lease 8d7f01 -> artifacts/runs/ladder_latent_sem_b1fix_anchor
  (ladder dir /dev/shm/rrp-brandonin/wt/ladder).
- binding v4 = paired pipeline + fix + anchor, sem & nosem: peer units rrp-binding-chain-v4-{sem,nosem} (scripts/binding_chain_v4.sh,
  dir wt/binding; logs repo/ops/logs/binding_chain_v4_*.log). The chain runs rep -> probes -> counterfactuals -> flow -> dev eval -> eval-binding.
- Oracle-route ladder evals auto-start when each of these representations lands: peer units rrp-ladder-wait-{jointfix,bindv4sem,bindv4nosem}
  (outputs wt/ladder artifacts/runs/ladder_v1/<robot>/oracle_zero_<tag>*.summary.json).
- Plain BC with the fix: direct-action baseline seed-1701 source on the HOST (unit rrp-b1fix-baseline_direct_action,
  scripts/baselines_host_b1fix.sh, root artifacts/runs/latent_slice1_b1fix); codec baseline on the PEER (unit rrp-b1fix-codec, dir wt/lead).
Stopped as B-1-contaminated (kept, never resume): flow_latent_{sem_v3,nosem_v2}, rep_binding_paired_*_v3, baseline sources in wt/baselines latent_slice1/.
Resume: `systemctl --user list-units 'rrp-*'` on both nodes; each script above is idempotent/resumable. Track notes: research/tracks/*.md on origin/track/*.
After a peer reboot: push source, run `scripts/peer_bootstrap.sh`, and restore runs from `~/rrp-peer-data/artifacts-snapshot-20260921/runs/` (on the PEER).

## earlier work log (2026-09-21, superseded by the track table; kept for accounting)
- lead: codec + structured/unstructured BC policies training on peer (artifacts/runs/codec_dev_v1, dev_structured_direct, dev_unstructured_direct).
- agent tracks: legged/humanoid breadth; dual-arm support_insert/handover; GRPO/EXPO-FT; psi0 VLM backbone + object QA.
