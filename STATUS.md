# project status — structured-psi0-latent-diffusion-dynamics (formerly relational robot policy)

**ACTIVE: corrected architecture (R38) on `main`. Current priority (authoritative): research/corrections/2026-09-25-causal-semantics-priorities.md — a competent source controller and causally meaningful packet semantics before any expansion. Old direct-action path = baseline only.**

Updated: 2026-09-25 14:00 PDT (session 2). Evidence: research/reports/evidence_matrix.md. Overall: **in_progress** (not complete).

## resolved environment
- repo: `~/work/relational-robot-policy` (GitHub: JacobFV/structured-psi0-latent-diffusion-dynamics, renamed 2026-09-25) (host `Dell-gb10-1`, aarch64 GB10). Handoff in `docs/handoff/`.
- peer: `gb10-direct` (hostname promaxgb10-4dfb, direct link). Workspace (RAM-backed, D-003): `/dev/shm/rrp-brandonin/{repo,venv,cache,bin}`.
- host venv: `.venv` (CPU only). peer venv: `/dev/shm/rrp-brandonin/venv` (torch cu130, mujoco).
- enforced parents (D-033/D-036): host `rrp.slice` 80% of free CPU/memory (14.97 CPU / 34.2 GiB at last init), disk reserve fixed 300 GB, host GPU allowed (2 leases); peer unrestricted (whole machine, D-008/D-026). See `configs/resources.local.json`.
- watchdogs: `rrp-watchdog-host.service`, `rrp-watchdog-peer.service` (user services inside rrp-control.slice).

## how to run anything (always under a lease)
```bash
cd ~/work/relational-robot-policy
PYTHONPATH=src python3 -m rrp.cli ops status
PYTHONPATH=src python3 -m rrp.cli ops run --cpu 1 --mem 2G --label NAME -- <cmd>
# peer:
scripts/peer_sync.sh push
ssh gb10-direct 'cd /dev/shm/rrp-brandonin/repo && PATH=/dev/shm/rrp-brandonin/bin:$PATH PYTHONPATH=src RRP_NODE=peer RRP_REPO=$PWD python3 -m rrp.cli ops run --gpu --gpu-mem 8G --cpu 4 --mem 16G --label NAME -- /dev/shm/rrp-brandonin/venv/bin/python ...'
```

## completed / verified
- P02/P03 resource safety (cgroup enforcement, broker, watchdog; peer = whole machine per D-008, 3 GPU slots D-009).
- P01, P04-P06 contracts + task runtime; P07/P08 sim + labelled scripted teacher; P10 service; P11 UI (15/15 browser tests, artifacts/video/workbench-demo.webm).
- P09 (arms): procedural arm family (5/6/7 DOF), menagerie panda/fr3/ur5e/sawyer/xarm7/z1/lite6, two gripper modules with adapter standoffs; teacher ledger artifacts/assets/teacher_validation_pick_place_20seeds.txt.
- Frozen split research/splits/primary_v1.json; dataset artifacts/datasets/pick_place_primary_v1 (peer): 3791 success / 106 teacher failures / 453 infeasible.

- Dual-arm track (verified, scripted_teacher only): support_insert + handover scenarios (`src/rrp/sim/dual_scenarios.py`, `tasks/handover.json`), DualSession public estimators, dual teachers, ALOHA import, MultiFeaturizer (feat-multi-v1+feat-v2); teacher v2 validation 16 SI pairs / 9 handover pairs x 30 seeds (`artifacts/assets/dual_teacher_validation/*_v3*`, D-019); dataset `artifacts/datasets/support_insert_primary_v2` (teacher v2; v1 archived; split `research/splits/primary_v1_support_insert.json`); functional composition report `research/reports/functional_composition.md`; details `research/reports/dual_arm_tasks.md`.

## parallel tracks (D-033; brief: research/tracks/BRIEF.md; each track's resume file: research/tracks/<track>.md)
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

## budgets used (see ops/resource-ledger.jsonl on each node)
- peer GPU device-hours: ~0.001 of 96. CPU core-hours: <0.1 of 512. Calendar day 1 of 7.

## blockers
- dual-arm: panda+tf3 insertion-arm target teacher-weak (3/30, 8/30, D-019); sawyer support / sawyer->panda_tf3 handover pairs fail (recorded); public insertion-depth estimator can false-positive on jams (D-018).
- none yet.

## resume
Read this file, `research/tasks.json`, `research/decisions.md`. Check `ops status` on both nodes and `systemctl --user list-units 'rrp-*'` before starting anything; attach to running owned jobs instead of duplicating.
