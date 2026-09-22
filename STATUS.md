# project status — relational robot policy

Updated: 2026-09-21 (session 1). Overall: **in_progress** (not complete).

## resolved environment
- repo: `~/work/relational-robot-policy` (host `Dell-gb10-1`, aarch64 GB10). Handoff in `docs/handoff/`.
- peer: `gb10-direct` (hostname promaxgb10-4dfb, direct link). Workspace (RAM-backed, D-003): `/dev/shm/rrp-brandonin/{repo,venv,cache,bin}`.
- host venv: `.venv` (CPU only). peer venv: `/dev/shm/rrp-brandonin/venv` (torch cu130, mujoco).
- enforced parents: host `rrp.slice` 7.11 CPU / 20.0 GiB / swap 0 / host GPU off; peer `rrp.slice` ~15.9 CPU / ~88 GiB / 1 GPU slot. See `configs/resources.local.json`.
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

- Dual-arm track (verified, scripted_teacher only): support_insert + handover scenarios (`src/rrp/sim/dual_scenarios.py`, `tasks/handover.json`), DualSession public estimators, dual teachers, ALOHA import, MultiFeaturizer (feat-multi-v1+feat-v2); teacher validation 16 SI pairs / 9 handover pairs x 30 seeds (`artifacts/assets/dual_teacher_validation/*final*`, `*handover_30seeds_v2*`); dataset `artifacts/datasets/support_insert_primary_v1` (1260 eps, 910 success; split `research/splits/primary_v1_support_insert.json`); functional composition report `research/reports/functional_composition.md`; details `research/reports/dual_arm_tasks.md`.

## current work (parallel tracks)
- lead: codec + structured/unstructured BC policies training on peer (artifacts/runs/codec_dev_v1, dev_structured_direct, dev_unstructured_direct).
- agent tracks: legged/humanoid breadth; dual-arm support_insert/handover; GRPO/EXPO-FT; psi0 VLM backbone + object QA.

## budgets used (see ops/resource-ledger.jsonl on each node)
- peer GPU device-hours: ~0.001 of 96. CPU core-hours: <0.1 of 512. Calendar day 1 of 7.

## blockers
- dual-arm: ALOHA handover teacher 0/30 (low-gain servo sag vs. gated integral term); panda+tf3 insertion-arm target teacher-weak (3/30, 11/30); public insertion-depth estimator can false-positive on jams (D-016).
- none yet.

## resume
Read this file, `research/tasks.json`, `research/decisions.md`. Check `ops status` on both nodes and `systemctl --user list-units 'rrp-*'` before starting anything; attach to running owned jobs instead of duplicating.
