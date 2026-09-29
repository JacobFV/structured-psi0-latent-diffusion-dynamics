"""Encoded-input identifiability audit (topoformer extended-02 lesson).

Before attributing a behavioral failure to learning, check that the ACTUAL public policy
tensors differ between counterfactual situations that require different actions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Distinguishability:
    status: str          # distinguishable | unidentifiable_under_current_input | not_required
    max_abs_diff: float
    detail: str = ""


def distinguishability(x_a, x_b, different_required_action: bool, tol: float = 1e-9) -> Distinguishability:
    a, b = np.asarray(x_a, float), np.asarray(x_b, float)
    if a.shape != b.shape:
        return Distinguishability("distinguishable", float("inf"), "different tensor shapes")
    d = float(np.abs(a - b).max()) if a.size else 0.0
    if not different_required_action:
        return Distinguishability("not_required", d)
    return Distinguishability("distinguishable" if d > tol else "unidentifiable_under_current_input", d)


def flatten_input(pi) -> np.ndarray:
    """Concatenate every tensor the policy can see (tokens, kinds, relations, pointers, text)."""
    parts = []
    for b in sorted(pi.tokens):
        parts += [pi.tokens[b].ravel(), pi.token_kind[b].ravel().astype(float), pi.pointer_text[b].ravel()]
    parts += [pi.act_node_feats.ravel(), pi.relations.ravel().astype(float), pi.pointers.ravel().astype(float)]
    return np.concatenate(parts)
