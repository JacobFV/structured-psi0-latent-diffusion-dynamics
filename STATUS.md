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

## now (2026-09-25 14:00)
- Lead chain moved to the HOST GPU (the peer GPU was time-sliced 9 ways; flow_latent_sem_v2 had dropped to 0.3 steps/s):
  user unit `rrp-chain-host` runs `RRP_NODE=host scripts/latent_chain_v2.sh latent_sem_v2 latent_nosem_v2 latent_sem_v3`
  (it resumed sem_v2 exactly at step 24,543). Log: ops/logs/latent_chain_host.out. Outputs: artifacts/runs/flow_latent_*_v{2,3}
  on the HOST (copy them to the peer store for other tracks). The peer chain and `rrp-chain-v3` are stopped.
- Resume the lead chain (idempotent): `systemctl --user status rrp-chain-host`; if it is not running,
  `cd ~/work/relational-robot-policy && systemd-run --user --unit rrp-chain-host --collect -p WorkingDirectory=$PWD --setenv=RRP_NODE=host --setenv=PATH=$PATH bash -c "exec bash scripts/latent_chain_v2.sh latent_sem_v2 latent_nosem_v2 latent_sem_v3 >> ops/logs/latent_chain_host.out 2>&1"`.
- Tracks: see the table above. Each track's resume steps are in research/tracks/<track>.md; the branches are
  track/<name> on origin, and the worktrees are ~/work/rrp-wt/<name>. On hold (D-037): baselines seeds 1702/1703 and SFT
  budgets; GRPO runs; legged/VLM breadth.
- After a peer reboot: push source, run `scripts/peer_bootstrap.sh` (it restores the venv, the menagerie on disk and the
  data links), and copy `~/rrp-peer-data/artifacts-snapshot-20260921/runs/*` into repo/artifacts/runs/ if missing (the snapshot is on the PEER).
- Host data: ~/work/rrp-data/{datasets,packed} (linked from artifacts/); menagerie at .cache/assets (pinned SHA).

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
