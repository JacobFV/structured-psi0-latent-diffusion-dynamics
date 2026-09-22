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

## D-007 2026-09-21 INCIDENT: brief unintended host GPU use; fixed with hard device exclusion
A 128x96 MuJoCo render probe (20 frames, <1 s, lease 1790017490_e25c21) created an EGL context on the host NVIDIA GB10 because the lease runner's environment allowlist dropped the Mesa vendor override and `CUDA_VISIBLE_DEVICES=""` does not affect EGL. No other workload was affected (probe finished in <1 s), but it violated the host-GPU-off rule. Fix: every non-GPU lease now runs with systemd `PrivateDevices=yes` (verified: /dev/nvidia* absent inside the unit) and forced Mesa llvmpipe EGL env; host GPU leases are refused in code. Regression test: `tests/integration/test_host_gpu_exclusion.py`. Host software rendering measured at ~18 fps for 128x96 on one core, so data rendering belongs on the peer.

## D-008 2026-09-21 user correction: the whole peer may be used
User instruction (supersedes the handoff's 80%-of-free peer default): "you can use all the peer". Peer policy now: 100% of free CPU/RAM/disk minus a small OS safety reserve (6 GiB RAM, 10 GiB disk); system-wide PSI shedding threshold on the peer raised to 80 (project-level stalls still shed). Persistent peer disk (~178 GiB usable) may now be used, but the RAM-backed workspace remains for speed; durable artifacts still sync to the host. Host limits (<=50% of currently free, no host GPU) are unchanged.

## D-009 2026-09-21 peer GPU concurrency 3
With the whole peer authorized (D-008) and five parallel engineering tracks, the peer broker admits up to 3 concurrent GPU leases. Each declares gpu_memory_bytes (counted in the aggregate memory limit) and applies the in-process CUDA cap; the system MemAvailable watchdog remains the backstop. Timing/latency measurements must be taken with no other GPU lease active (recorded per measurement).
