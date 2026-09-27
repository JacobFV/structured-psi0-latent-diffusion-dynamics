# project status — structured-psi0-latent-diffusion-dynamics (formerly relational robot policy)

Updated: 2026-09-26 18:40 PDT. Overall: **in_progress** (not complete). Plan: `docs/strategy.md` (D-094).
Evidence: `research/reports/evidence_matrix.md`. Decisions: `research/decisions.md` (D-001..D-094).

## current state (2026-09-26 evening)
Strategy adopted (D-094): physics credibility first, one pipeline for all bodies, additive-then-subtractive restructure.
Workstream owners, gates and sequencing live in `docs/strategy.md`; this table mirrors its status column.

| id | workstream | status | where |
|---|---|---|---|
| W1 | physics realism: contact v2, staged reward schedule, slip gate; then actuators/latency | running | research/tracks/contact.md (t1, h1 trackers on the host) |
| W2 | repo hygiene (phase 0) | completed | research/tracks/hygiene.md |
| W3 | provenance and contracts (phase 1) | implementing | docs/strategy.md §W3 |
| W4 | package restructure with shims | planned (after W2, W3) | docs/repo_structure_audit.md |
| W5 | unified pipeline + DAG orchestration | planned (after W4, R0) | |
| W6 | robustness sweeps + motion-quality gates | planned (after W1 trackers) | |
| W7 | arm expert smoothing, then GRPO with anchor evals | planned (after R0) | |
| W8 | legged regeneration on contact v2 | planned (after W1 gate, W3) | |
| W9 | held-out target bodies; loco-manipulation | planned | |
| R0 | arm seed-2 replication of D-091 (sejf2 / sfjf2 / nsjf2) | running | research/tracks/ladder.md "ARM SEED-2 REPLICATION" |

Legged results through D-092 use contact model v1 (trackers skate, D-093); legged data is not regenerated until the
v2 trackers pass the W1 gate. Do not sync into peer dirs with running jobs (`/dev/shm/rrp-brandonin/wt/ladder`, `repo`).

## evidence summary (through D-092)
- Supported, on DEPLOYABLE routes (no teacher/oracle/BC at run time): task → packet → behaviour.
  - go2: goal direction (D-071).
  - parm6 arm: goal + binding edits (D-074, D-075; 80 seeds).
  - panda arm: binding redirection (D-077).
  - BC ignores the binding edit (D-065).
- Competence: the latent route matches BC on go2 and hexapod6 (D-070, D-076), and on t1 only for nosem (D-079, D-082) until the D-085 fix; with it t1 fixed sem ≈ nosem (81/90 vs 83/90, D-087).
  - Arm is partial: frozen route pooled 36/90 panda_pg2, 70/90 parm6_tf3 (D-078, D-080).
- Semantic supervision (after the D-085 bounded-NLL fix): essential on the arm (nosem route 3/240 vs frozen sem 146/240 vs bounded-NLL sem 124/240 deployable successes; D-089, D-091, one seed; seed 2 running); on go2/hexapod6 a task-context "halt" slows only the sem packet, replicated in direction over 3 seeds (D-088, D-090), and pooled-only on the t1 humanoid (D-092); t1 success equal (81/90 vs 83/90, D-087). The D-084 "sem hurts" reading was the D-085 defect.
- Not tested: the latent route on the sealed target bodies. BC: new gripper 78–85/100, unseen xarm7 0/100 (D-064). g1 has no competent BC.
- Root-cause fixes along the way: B-1 prev-action column (D-044/045); system-0 velocity-copy shortcut (D-056); stale stateful teacher as oracle/DAgger expert (D-050); compatibility-ID fingerprinting (D-038).

Frozen arm route: system i ladder_flow_jointfix_gdag2h (sha d0d64918…) → system 0 ladder_rz_jointfix_gendag3_noqd (sha f60cde41…).
Always pass the system-0 bundle explicitly (research/tracks/ladder.md "SPRINT BEST ROUTE FINAL"). Legged: research/tracks/legged_vlm.md
"LEGGED RESEARCH RESULT FINAL". Semantic suite: research/tracks/acceptance.md "SPRINT SEMANTIC RESULTS". BC: research/tracks/baselines.md.

## resolved environment
- repo: `~/work/relational-robot-policy` (GitHub: JacobFV/structured-psi0-latent-diffusion-dynamics, renamed 2026-09-25) (host `Dell-gb10-1`, aarch64 GB10). Handoff in `docs/handoff/`.
- peer: `gb10-direct` (hostname promaxgb10-4dfb, direct link). Workspace (RAM-backed, D-003): `/dev/shm/rrp-brandonin/{repo,venv,cache,bin}`.
- host venv: `.venv` (CPU only). peer venv: `/dev/shm/rrp-brandonin/venv` (torch cu130, mujoco).
- enforced parents (D-033/D-086): host `rrp.slice` 80% of free CPU/memory (12.98 CPU / 31.7 GiB at last init, D-086), disk reserve fixed 100 GB (D-086), host GPU allowed (3 leases); peer unrestricted (whole machine, D-008/D-026). See `configs/resources.local.json`.
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
- Legged: contact v1 trackers skate (D-093); g1 has no competent BC positive control (D-073).
- Dual arm: panda+tf3 insertion-arm target teacher-weak (3/30, 8/30, D-019); sawyer support / sawyer->panda_tf3 handover pairs fail; the public insertion-depth estimator can false-positive on jams (D-018).
- Host memory is squeezed by other projects' processes; the memory-PSI watchdog sheds host jobs (keep them ≤24 GiB).

## resume
Read this file, `docs/strategy.md`, `research/decisions.md` (latest entries) and the relevant `research/tracks/<track>.md`.
Check `rrp ops status` on both nodes and `systemctl --user list-units 'rrp-*'` before starting anything; attach to running
owned jobs instead of duplicating. Budgets: `ops/resource-ledger.jsonl` on each node.

# history (kept for accounting; superseded by the sections above)

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
