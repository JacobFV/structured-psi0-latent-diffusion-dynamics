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
- P02 discovery + telemetry, P03 enforcement/broker/watchdog: 41 tests pass (`artifacts/receipts/ops/enforcement-tests-host.txt`).

## current work
- P01 contracts, P04-P06 task runtime; source audit (psi0/SONIC/Z-1/EXPO-FT/menagerie/topoformer).

## budgets used (see ops/resource-ledger.jsonl on each node)
- peer GPU device-hours: ~0.001 of 96. CPU core-hours: <0.1 of 512. Calendar day 1 of 7.

## blockers
- none yet.

## resume
Read this file, `research/tasks.json`, `research/decisions.md`. Check `ops status` on both nodes and `systemctl --user list-units 'rrp-*'` before starting anything; attach to running owned jobs instead of duplicating.
