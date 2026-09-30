# two-node resource and machine safety contract

## meaning of the user's 50% rule

The host is a shared production-like machine. At lease admission, measure a conservative idle-capacity window and permit this entire project at most half of the free amount. All project processes are aggregated. A new subagent is not a new allowance. Use reserves in addition to the cap. Never infer free resources from the number printed on the computer's specifications.

For RAM available A, total T, project-resident P and reserve R:

```text
startup_limit <= min(0.5 * A_start, A_start - R)
live_limit <= min(startup_limit, 0.5 * (A_now + P), A_now + P - R)
```

Only add back accurately attributed project residency when estimating free-without-project; otherwise use the smaller conservative measurement. Do not repeatedly halve already-project-reduced available memory and call the result a coherent budget. Never count CPU and GPU memory on GB10 as independent additive pools. Use OS and CUDA telemetry conservatively; unavailable telemetry is not zero usage.

Host reserve defaults: max(8 GiB, 10% physical RAM); disk reserve max(20 GiB, 10% filesystem capacity). Cap project new disk footprint at half the initially free space and preserve the live reserve. Count datasets, extracted archives, wheels, build caches, videos, checkpoints, logs, and temporary double-buffered saves.

CPU admission uses a 10-second idle window on allowed CPUs, constrained by existing affinity and cgroup quota. Allocate <=50% of conservatively measured idle CPU capacity. Fractional CPU quotas are valid; do not round 0.4 free cores up to one busy core. BLAS/OpenMP threads, Rust/C++ compile jobs, browser workers, data workers, and child processes share this quota. Nice/ionice are cooperative priority hints, not hard limits.

## implement enforcement BEFORE workloads

Create a user-level parent cgroup/slice or an already-available nonprivileged container mechanism with real aggregate CPU/memory/pid enforcement. Test it with bounded child-process fixtures. Broker leases are reservations under that parent, not independent cgroups whose caps sum above the host allocation. MemoryHigh <=0.8*MemoryMax; project swap allowance is zero where supported; configure group OOM to affect only owned descendants. Do not assume cgroup accounting bounds every driver allocation; host GPU work remains off by default.

Do not create unlimited compilation processes while installing the limiter. Bootstrap inspection and a small standard-library daemon may run with one low-priority CPU worker, no large allocations, and bounded output. If enforceable host quotas are unavailable, do not launch host training/build/browser stress jobs. Prefer the peer; perform only lightweight host coordination and record the limitation. No sudo to “fix” this automatically.

Implement watchdog checkpoints every 2 seconds during compute, lease renewals, atomic quota updates, process ownership, start-time identity checks, checkpoint-before-termination when time permits, and a hard timeout fallback. Detect available-memory drops, swap growth, sustained memory PSI pressure, disk reserve violation, thermal throttle, failed telemetry, or missed heartbeat. Lower allocations or stop only owned jobs. Do not SIGSTOP a GPU workload as a memory-release strategy; its allocations remain resident.

On pressure: stop admitting work; ask owned jobs to checkpoint; terminate their process groups/cgroups after a bounded grace period; report the event; resume only after a stable window. Checkpointing itself requires a reserved disk/memory allowance. An emergency shutdown may lose the latest unsaved progress; it must not crash another project.

## GPU policy

Default host GPU allocation is zero for training, inference, simulation rendering and automated browser WebGL. Use the peer for these. A controlled host GPU profile is an optional optimization only after verifying hardware/runtime isolation and memory limits at or below half current spare capacity without altering existing workloads. `CUDA_VISIBLE_DEVICES`, PyTorch allocator fractions, MPS active-thread percentages, and observed utilization are not by themselves an aggregate free-capacity guarantee. Do not claim MIG/MPS support or protected bandwidth without actually testing that platform.

Automated host GUI checks use a dedicated software-rendered browser profile or streamed peer-rendered frames. Do not hijack the user's browser. The interactive UI must offer low-load streaming mode instead of obligatorily opening a heavy local WebGL scene. GUI/browser/encoder processes belong in the resource ledger too.

Peer defaults: use up to 80% of currently free CPU/RAM after reserves, one heavy GPU owner at a time, and the same memory/disk/thermal watchdog. If another peer GPU workload is present, do not take the whole accelerator; seek a proven safe allocation or wait while doing other work. A physically connected second computer is still a separate shared-capable system.

## discovery and networking

Prefer `ROBOT_PEER` if supplied. Otherwise inspect relevant literal aliases in existing SSH configuration and known local GB10 notes; try only likely configured GB10 targets with batch authentication, short connection timeout, existing known-host verification and bounded command output. Do not inspect private-key contents. Do not enumerate unrelated networks or change SSH trust. Confirm host and peer differ using machine identity and hostnames. Redact unique identifiers in public-facing reports.

Audit architecture, OS, CUDA/driver, available environments, actual GPU name/capability, filesystem, cgroup delegation and running project jobs. Do not assume x86 wheels work on ARM64 or the newest NVIDIA marketing throughput describes training performance. Verify MuJoCo/rendering, PyTorch kernels, optional attention backends, and simulator acceleration on the actual peer.

Two GB10s are not one shared allocator. Start with independent jobs: peer learner/GPU renderer, bounded host coordinator/CPU tests. Only consider distributed training after isolated jobs work and measured communication justifies it. Do not alter network configuration, launch privileged connectivity playbooks, or assume GPUDirect RDMA support. Existing SSH transport is sufficient for submission and artifact sync. Bound rsync/network IO; do not recursively sync model caches or private home directories.

## bounded campaign and retries

Defaults in `config/resources.json` are first-campaign ceilings: 96 peer GPU device-hours, 512 aggregate CPU core-hours, seven active calendar days. These are chosen caps, not predicted completion time. Reserve 20% of remaining budget for confirmation, 10% for audits/reporting; update the ledger before each release. Count failed and capped jobs. GPU device time means elapsed GPU-owned job occupancy, not an invented utilization-normalized FLOP estimate.

Escalation per newly implemented workload: bounded unit fixture -> 30-second smoke -> 2-minute profile -> 10-minute learning pilot -> measured training run capped at 6 hours. Longer objectives use checkpoints and resumable segments; no unlimited job timeout. At most two unchanged infrastructure retries, then diagnose. At most three materially distinct development recipes per named hypothesis before an explicit analysis decision. New experiments require a stated discriminating question, not merely unused compute.

Source builds and installations count against CPU/disk budget. Remote model downloads require size/license verification and safe space for extraction. Never delete user caches to make a model fit. If the final envelope is exhausted, stop owned compute, deliver evidence plus exact resume queue, and mark unfinished obligations.

## required proof before the first training launch

Show: two verified machines or a declared peer blocker; conservative free-capacity baseline; effective parent quota files; bounded-child enforcement test; aggregate lease accounting test; peer memory/driver audit; watchdog pressure-test result; owned-only shutdown test; available disk including checkpoint reserve; and a resource manifest attached to the proposed run. A pretty dashboard or read-only preflight JSON is insufficient.

Sources: NVIDIA's DGX Spark system/optimization guides describe ARM64 and unified memory; the official connectivity guide treats two-node setup as networking; the MPS documentation describes thread limits rather than a universal host resource reservation. Exact source URLs are in `docs/09_sources_and_topoformer.md`.
