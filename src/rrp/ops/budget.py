"""Conservative resource budgets implementing docs/handoff/docs/03_resource_safety.md.

Pure arithmetic. Measurement lives in telemetry.py and enforcement in cgroup.py.
The host budget is half of CURRENTLY FREE capacity (never of nominal totals),
constrained further by reserves. Fractional CPU is never rounded up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

GIB = 1024 ** 3


class BudgetError(ValueError):
    code = "invalid_capacity"


def _finite_nonneg(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BudgetError(f"{name} must be a finite nonnegative number, got {value!r}")
    v = float(value)
    if not math.isfinite(v) or v < 0:
        raise BudgetError(f"{name} must be a finite nonnegative number, got {value!r}")
    return v


@dataclass(frozen=True)
class RolePolicy:
    free_cpu_fraction: float
    free_memory_fraction: float
    free_disk_fraction: float
    memory_reserve_gib: float = 8.0
    memory_reserve_total_fraction: float = 0.10
    disk_reserve_gib: float = 20.0
    disk_reserve_total_fraction: float = 0.10

    @classmethod
    def host(cls) -> "RolePolicy":
        # D-027: host GPU at 80% of FREE memory (unified CPU+GPU). D-033: user raised host CPU to 80% of free cores.
        return cls(0.8, 0.8, 0.5)

    @classmethod
    def peer(cls) -> "RolePolicy":
        # D-008: user authorized using ALL of the peer; keep only a small OS safety reserve
        return cls(1.0, 1.0, 1.0, memory_reserve_gib=6.0, memory_reserve_total_fraction=0.0,
                   disk_reserve_gib=10.0, disk_reserve_total_fraction=0.0)

    def validate(self, role: str) -> None:
        for name in ("free_cpu_fraction", "free_memory_fraction", "free_disk_fraction"):
            upper = 1.0 if role != "host" else (0.5 if name == "free_disk_fraction" else 0.8)
            v = _finite_nonneg(getattr(self, name), name)
            if v > upper:
                raise BudgetError(f"{role} {name} {v} exceeds protected maximum {upper}")


@dataclass(frozen=True)
class ResourceBudget:
    role: str
    memory_gib: float
    cpu_cores: float
    new_disk_gib: float
    memory_reserve_gib: float
    disk_reserve_gib: float
    host_gpu_enabled: bool
    basis: dict

    @property
    def memory_bytes(self) -> int:
        return int(math.floor(self.memory_gib * GIB))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["memory_bytes"] = self.memory_bytes
        return d


def memory_reserve_gib(total_ram_gib: float, policy: RolePolicy) -> float:
    return max(policy.memory_reserve_gib, policy.memory_reserve_total_fraction * total_ram_gib)


def disk_reserve_gib(total_disk_gib: float, policy: RolePolicy) -> float:
    return max(policy.disk_reserve_gib, policy.disk_reserve_total_fraction * total_disk_gib)


def compute_budget(*, role: str, total_ram_gib: float, available_ram_gib: float,
                   free_cpu_cores: float, free_disk_gib: float, total_disk_gib: float,
                   policy: RolePolicy | None = None) -> ResourceBudget:
    policy = policy or (RolePolicy.host() if role == "host" else RolePolicy.peer())
    if role not in {"host", "peer"}:
        raise BudgetError("role must be host or peer")
    policy.validate(role)
    total_ram = _finite_nonneg(total_ram_gib, "total_ram_gib")
    avail = _finite_nonneg(available_ram_gib, "available_ram_gib")
    cpu = _finite_nonneg(free_cpu_cores, "free_cpu_cores")
    free_disk = _finite_nonneg(free_disk_gib, "free_disk_gib")
    total_disk = _finite_nonneg(total_disk_gib, "total_disk_gib")
    if total_ram <= 0 or total_disk <= 0:
        raise BudgetError("total memory and disk must be positive")
    if avail > total_ram:
        raise BudgetError("available memory exceeds total memory")
    if free_disk > total_disk:
        raise BudgetError("free disk exceeds total disk")
    mres = memory_reserve_gib(total_ram, policy)
    dres = disk_reserve_gib(total_disk, policy)
    mem = max(0.0, min(policy.free_memory_fraction * avail, avail - mres))
    disk = max(0.0, min(policy.free_disk_fraction * free_disk, free_disk - dres))
    cores = policy.free_cpu_fraction * cpu  # fractional; never rounded up
    return ResourceBudget(
        role=role, memory_gib=mem, cpu_cores=cores, new_disk_gib=disk,
        memory_reserve_gib=mres, disk_reserve_gib=dres,
        host_gpu_enabled=False,
        basis=dict(total_ram_gib=total_ram, available_ram_gib=avail, free_cpu_cores=cpu,
                   free_disk_gib=free_disk, total_disk_gib=total_disk,
                   policy=asdict(policy)),
    )


def host_budget(*, total_ram_gib: float, available_ram_gib: float, free_cpu_cores: float,
                free_disk_gib: float, total_disk_gib: float) -> ResourceBudget:
    return compute_budget(role="host", total_ram_gib=total_ram_gib,
                          available_ram_gib=available_ram_gib, free_cpu_cores=free_cpu_cores,
                          free_disk_gib=free_disk_gib, total_disk_gib=total_disk_gib)


def live_memory_limit_bytes(*, startup_limit: int, available_now: int, project_resident: int,
                            reserve: int, fraction: float = 0.5,
                            project_attribution_accurate: bool = True) -> int:
    """live_limit <= min(startup, f*(A_now+P), A_now+P-R).

    P is only added back when project residency is accurately attributed (cgroup
    memory.current). Otherwise the smaller, conservative A_now is used alone.
    """
    for n, v in (("startup_limit", startup_limit), ("available_now", available_now),
                 ("project_resident", project_resident), ("reserve", reserve)):
        _finite_nonneg(v, n)
    free_wo_project = available_now + (project_resident if project_attribution_accurate else 0)
    return int(max(0, min(startup_limit, fraction * free_wo_project, free_wo_project - reserve)))


def live_cpu_limit(*, startup_limit: float, idle_now: float, project_usage: float,
                   fraction: float = 0.5) -> float:
    _finite_nonneg(idle_now, "idle_now")
    _finite_nonneg(project_usage, "project_usage")
    return max(0.0, min(startup_limit, fraction * (idle_now + project_usage)))
