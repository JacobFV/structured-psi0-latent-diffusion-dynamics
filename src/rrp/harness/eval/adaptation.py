"""Nested adaptation budgets and sustained-threshold (censored) crossing."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def nested_budget_indices(n: int, budgets: list[int], seed: int) -> dict[int, list[int]]:
    """One fixed permutation per (target, seed); budget k takes its first k items => nested."""
    if max(budgets) > n:
        raise ValueError(f"budget {max(budgets)} exceeds available {n}")
    perm = np.random.default_rng(seed).permutation(n).tolist()
    return {b: perm[:b] for b in sorted(budgets)}


@dataclass
class Crossing:
    budget: float | None
    censored: bool
    provisional: bool = False


def first_sustained_crossing(budgets, success, threshold: float, require_next: bool = True) -> Crossing:
    for i, (b, s) in enumerate(zip(budgets, success)):
        if s >= threshold:
            if not require_next:
                return Crossing(b, False)
            if i + 1 < len(success):
                if success[i + 1] >= threshold:
                    return Crossing(b, False)
                continue
            return Crossing(b, False, provisional=True)   # last checkpoint: unconfirmed
    return Crossing(None, True)
