# decision log

Append-only. Each entry: prior assumption, evidence, decision, expected effect on claims.

## D-001 2026-09-21 repository location
`~/work/relational-robot-policy` did not exist; created fresh. Handoff preserved verbatim in `docs/handoff/` (SHA256SUMS verified OK).

## D-002 2026-09-21 peer identity (R02)
Candidates from literal `~/.ssh/config` aliases matching gb10/spark (no scanning, BatchMode, StrictHostKeyChecking=yes). Receipt: `artifacts/receipts/ops/peer-discovery.json`.
- `gb10-direct` -> hostname promaxgb10-4dfb, aarch64, NVIDIA GB10 driver 580.126.09, 20 cores, ~115 GiB available, no GPU compute processes, route over the physical direct link (enp1s0f1np1 10.100.216.0/24). `promaxgb10-4dfb` alias is the same machine via Tailscale (same machine-id).
- `spark-ec4d` is a THIRD GB10 (distinct machine-id) running a foreign vLLM EngineCore (92 GB); it is not the physically connected peer and is not used.
- `spark-gb10` unreachable (no route).
Decision: peer = `gb10-direct`. Local notes (`~/SSH-REMOTE-DIRECTIONS.md`) predate the direct link and were not trusted over live identity checks.

## D-003 2026-09-21 peer persistent disk is effectively unavailable
Peer root FS: 1.9 TiB, 188 GiB free (90% used). Reserve = max(20 GiB, 10% capacity) ≈ 186 GiB -> policy allows ~0 new persistent bytes. Decision: all peer project files (venv, caches, datasets, checkpoints) live in RAM-backed `/dev/shm/rrp-brandonin` (tmpfs; charged to our memory cgroup, so inside the peer memory budget). Durable artifacts are pulled to the host (host new-disk budget ~59 GiB). Consequence: a peer reboot loses the peer workspace; everything must be reproducible from the host repo + lockfiles, and checkpoints are synced back after each training segment. This tmpfs must be removed at final shutdown or listed as retained.

## D-004 2026-09-21 GPU memory is not visible to memcg on GB10 (R03)
Measured on peer (`gpu_memcg_probe`): a 3 GiB CUDA allocation inside a lease with MemoryMax=2 GiB succeeded; lease memory.current barely changed. `torch.cuda.mem_get_info` reported only ~6 GiB "free" (MemFree, excluding reclaimable page cache) of 119.6 GiB. Decision: leases carry a declared `gpu_memory_bytes` that is added to the aggregate memory sum; every GPU workload calls `rrp.ops.gpu.apply_cap()` (torch per-process memory fraction; tested: allocation beyond cap raises OOM in-process); system MemAvailable watchdog is the backstop. Host GPU remains disabled: cgroups cannot bound it.

## D-005 2026-09-21 host state at admission
Host had swap fully used (16 GiB) by other workloads and ~40-47 GiB available at admission, load ~4.6. Prior OOM incident on 2026-09-13 noted (`~/GB10-OOM-NOTE-2026-09-13.md`), probably a GPU workload. Host parent quota: CPUQuota 7.11 cores, MemoryMax 20.0 GiB, MemoryHigh 0.8x, swap 0, TasksMax 4096 (`configs/resources.local.json`). Watchdog lowers CPU quota live and revokes newest leases when the live memory limit falls below leased totals.

## D-006 2026-09-21 MemoryHigh behaviour
A child exceeding MemoryHigh with no swap stalls in reclaim (PSI full ~89%) instead of dying. The watchdog therefore sheds on sustained project memory PSI (tested with a bounded fixture in `tests/integration/test_resource_enforcement.py`).
