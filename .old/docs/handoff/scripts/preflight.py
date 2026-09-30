#!/usr/bin/env python3
"""Read-only Linux preflight and conservative budget proposal; NOT a limiter.

Uses only the Python standard library. It does not SSH, import CUDA/PyTorch,
install software, start workloads, or change system configuration. An optional
report is created exclusively, never overwriting an existing file.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time
from typing import Any

GIB = 1024 ** 3


def nonnegative_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite nonnegative number')
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be a finite nonnegative number')
    return float(value)


def calculate_budget(snapshot: dict[str, Any], role: str,
                     policy: dict[str, Any]) -> dict[str, Any]:
    """Calculate an admission proposal, never authorization/enforcement."""
    if role not in {'host', 'peer'}:
        raise ValueError('role must be host or peer')
    values = {k: nonnegative_number(snapshot[k], k) for k in (
        'memory_total_bytes', 'memory_available_bytes', 'disk_total_bytes',
        'disk_free_bytes', 'idle_cpu_cores')}
    if values['memory_total_bytes'] <= 0 or values['disk_total_bytes'] <= 0:
        raise ValueError('total memory and disk capacity must be positive')
    if values['memory_available_bytes'] > values['memory_total_bytes']:
        raise ValueError('available memory exceeds total memory')
    if values['disk_free_bytes'] > values['disk_total_bytes']:
        raise ValueError('free disk exceeds total disk')
    cfg = policy[role]
    fractions = {k: nonnegative_number(cfg[k], k) for k in (
        'free_cpu_fraction', 'free_memory_fraction', 'free_disk_fraction')}
    upper = 0.5 if role == 'host' else 0.8
    if any(v > upper for v in fractions.values()):
        raise ValueError(f'{role} free-resource fractions cannot exceed {upper}')
    mem_reserve = max(
        nonnegative_number(cfg['memory_reserve_gib'], 'memory reserve') * GIB,
        nonnegative_number(cfg['memory_reserve_total_fraction'], 'memory reserve fraction')
        * values['memory_total_bytes'])
    disk_reserve = max(
        nonnegative_number(cfg['disk_reserve_gib'], 'disk reserve') * GIB,
        nonnegative_number(cfg['disk_reserve_total_fraction'], 'disk reserve fraction')
        * values['disk_total_bytes'])
    mem_cap = max(0, min(values['memory_available_bytes'] * fractions['free_memory_fraction'],
                         values['memory_available_bytes'] - mem_reserve))
    disk_cap = max(0, min(values['disk_free_bytes'] * fractions['free_disk_fraction'],
                          values['disk_free_bytes'] - disk_reserve))
    return {
        'role': role,
        'memory_limit_bytes': math.floor(mem_cap),
        'new_disk_limit_bytes': math.floor(disk_cap),
        'cpu_quota_cores': values['idle_cpu_cores'] * fractions['free_cpu_fraction'],
        'memory_reserve_bytes': math.ceil(mem_reserve),
        'disk_reserve_bytes': math.ceil(disk_reserve),
        'gpu_admission': 'disabled_pending_enforcement',
        'heavy_work_authorized': False,
        'warning': 'initial measurement only; implement/test aggregate enforcement and live watchdog before heavy work',
    }


def parse_meminfo(text: str) -> dict[str, int]:
    result: dict[str, int] = {}
    keys = {'MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'}
    for line in text.splitlines():
        if ':' not in line:
            continue
        key, rest = line.split(':', 1)
        if key not in keys:
            continue
        fields = rest.split()
        if len(fields) != 2 or fields[1] != 'kB':
            raise ValueError(f'unsupported meminfo units for {key}')
        count = int(fields[0])
        if count < 0:
            raise ValueError(f'negative meminfo value for {key}')
        result[key] = count * 1024
    if not {'MemTotal', 'MemAvailable'} <= result.keys():
        raise ValueError('MemTotal and MemAvailable are required; free memory will not be guessed')
    return result


def read_cpu_counters() -> dict[str, list[int]]:
    result = {}
    for line in Path('/proc/stat').read_text().splitlines():
        fields = line.split()
        if fields and fields[0].startswith('cpu') and fields[0][3:].isdigit():
            result[fields[0]] = [int(v) for v in fields[1:9]]
    return result


def idle_cores_between(before: dict[str, list[int]], after: dict[str, list[int]],
                       allowed: set[int]) -> float:
    idle_cores = 0.0
    if not allowed:
        raise ValueError('no allowed CPUs')
    for index in sorted(allowed):
        name = f'cpu{index}'
        if name not in before or name not in after:
            raise ValueError(f'CPU telemetry missing for {name}')
        a, b = before[name], after[name]
        if len(a) < 8 or len(b) < 8:
            raise ValueError(f'incomplete CPU counters for {name}')
        delta = [b[i] - a[i] for i in range(8)]
        if any(v < 0 for v in delta) or sum(delta) <= 0:
            raise ValueError('CPU counter reset or no elapsed counter time')
        # iowait is deliberately not considered available compute capacity.
        idle_cores += delta[3] / sum(delta)
    return idle_cores


def cgroup_headroom() -> dict[str, Any]:
    """Best-effort read of cgroup-v2 ancestor limits, not proof of delegation."""
    result: dict[str, Any] = {'memory_headroom_bytes': None, 'cpu_quota_cores': None,
                              'enforcement_verified': False, 'warnings': []}
    try:
        rows = Path('/proc/self/cgroup').read_text().splitlines()
        relative = next(row.split('::', 1)[1] for row in rows if row.startswith('0::'))
        base = Path('/sys/fs/cgroup').resolve()
        node = (base / relative.lstrip('/')).resolve()
        if not node.is_relative_to(base):
            raise ValueError('unexpected cgroup path outside visible hierarchy')
        while True:
            cpu = node / 'cpu.max'
            if cpu.is_file():
                quota, period = cpu.read_text().strip().split()
                if quota != 'max':
                    q = int(quota) / int(period)
                    result['cpu_quota_cores'] = min(q, result['cpu_quota_cores']) if result['cpu_quota_cores'] is not None else q
            mem, current = node / 'memory.max', node / 'memory.current'
            if mem.is_file() and current.is_file():
                limit = mem.read_text().strip()
                if limit != 'max':
                    available = max(0, int(limit) - int(current.read_text().strip()))
                    old = result['memory_headroom_bytes']
                    result['memory_headroom_bytes'] = available if old is None else min(old, available)
            if node == base:
                break
            node = node.parent
    except (OSError, ValueError, StopIteration, ZeroDivisionError) as exc:
        result['warnings'].append(f'cgroup inspection incomplete: {type(exc).__name__}')
    return result


def gpu_query() -> dict[str, Any]:
    executable = shutil.which('nvidia-smi')
    if not executable:
        return {'status': 'unavailable', 'rows': [], 'warning': 'no GPU capacity inferred'}
    try:
        run = subprocess.run([executable,
            '--query-gpu=name,utilization.gpu,memory.total,memory.used',
            '--format=csv,noheader,nounits'], capture_output=True, text=True,
            timeout=5, check=False)
        return {'status': 'reported' if run.returncode == 0 else 'query_failed',
                'rows': run.stdout[:8192].splitlines(),
                'warning': 'raw diagnostic only; N/A is unknown; CPU/GPU memory may share one pool; not an allocation lease'}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'status': 'unavailable', 'rows': [], 'warning': type(exc).__name__}


def inspect_machine(samples: int, interval: float, disk_path: Path) -> dict[str, Any]:
    if platform.system() != 'Linux':
        raise ValueError('this lightweight helper requires Linux /proc; do not guess capacities on another OS')
    if samples < 2 or samples > 31 or not math.isfinite(interval) or not 0.1 <= interval <= 5:
        raise ValueError('samples must be 2..31 and interval 0.1..5 seconds')
    allowed = set(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else set(range(os.cpu_count() or 1))
    before = read_cpu_counters(); idle_samples = []
    mem_samples = []; cg_samples = []
    # Measure all admission inputs over the same conservative window.
    for _ in range(samples - 1):
        time.sleep(interval)
        after = read_cpu_counters()
        idle_samples.append(idle_cores_between(before, after, allowed))
        before = after
        mem_samples.append(parse_meminfo(Path('/proc/meminfo').read_text()))
        cg_samples.append(cgroup_headroom())
    total = min(x['MemTotal'] for x in mem_samples)
    available = min(x['MemAvailable'] for x in mem_samples)
    headroom = [x['memory_headroom_bytes'] for x in cg_samples if x['memory_headroom_bytes'] is not None]
    if headroom:
        available = min(available, min(headroom))
    idle = min(idle_samples)
    quotas = [x['cpu_quota_cores'] for x in cg_samples if x['cpu_quota_cores'] is not None]
    if quotas:
        idle = min(idle, min(quotas))
    disk = shutil.disk_usage(disk_path.resolve(strict=True))
    return {
        'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'hostname': socket.gethostname(), 'architecture': platform.machine(),
        'platform': platform.platform(), 'allowed_cpu_count': len(allowed),
        'idle_cpu_cores': idle, 'idle_cpu_samples': idle_samples,
        'memory_total_bytes': total, 'memory_available_bytes': available,
        'swap_used_bytes_samples': [x.get('SwapTotal', 0)-x.get('SwapFree', 0) for x in mem_samples],
        'disk_total_bytes': disk.total, 'disk_free_bytes': disk.free,
        'disk_path': str(disk_path.resolve()), 'cgroup_read': cg_samples[-1],
        'gpu_diagnostic': gpu_query(), 'inspection_window_seconds': (samples-1)*interval,
        'limitations': ['snapshot estimates only, not resource enforcement',
                       'external load may change after inspection',
                       'no SSH peer was contacted',
                       'disk IO/network/GPU bandwidth are not isolated by this helper'],
    }


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write('\n')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=['host','peer'], default='host')
    parser.add_argument('--policy', type=Path, default=Path(__file__).resolve().parents[1]/'config/resources.json')
    parser.add_argument('--samples', type=int, default=11)
    parser.add_argument('--interval', type=float, default=1.0)
    parser.add_argument('--disk-path', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, help='create a NEW JSON file; existing files are never overwritten')
    args = parser.parse_args()
    try:
        policy = json.loads(args.policy.read_text())
        snapshot = inspect_machine(args.samples, args.interval, args.disk_path)
        output = {'schema_version':'1.0', 'snapshot':snapshot,
                  'proposed_budget':calculate_budget(snapshot,args.role,policy),
                  'workloads_started':False, 'enforcement_installed':False}
        if args.output:
            write_json_exclusive(args.output, output)
            print(f'created {args.output}; no workload authorized or started')
        else:
            print(json.dumps(output, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'preflight failed closed: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
