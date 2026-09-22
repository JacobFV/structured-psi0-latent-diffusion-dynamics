"""Statistics computed from raw episode rows only."""
from __future__ import annotations

import math


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def success_counts(rows) -> tuple[int, int]:
    rows = list(rows)
    return sum(1 for r in rows if r["success"]), len(rows)


def summarize_attempts(outcomes):
    from dataclasses import dataclass

    @dataclass
    class S:
        attempted: int
        successes: int
        success_rate: float

    outs = list(outcomes)
    k = sum(1 for o in outs if o == "success")
    return S(len(outs), k, k / len(outs) if outs else float("nan"))
