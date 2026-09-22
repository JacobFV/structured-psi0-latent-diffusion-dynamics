"""Encoded-input identifiability audit (topoformer extended-02 lesson).

Before attributing a behavioral failure to learning, check that the ACTUAL public policy
tensors differ between counterfactual situations that require different actions.
"""
from __future__ import annotations

import copy
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


def audit_counterfactuals(session_factory) -> list[dict]:
    """Build matched counterfactual pairs on a real session and compare encoded inputs.
    Returns one row per mandatory intervention with its identifiability status."""
    from rrp.data.collect import featurizer_for
    from rrp.tasks.receipts import Receipt
    rows = []

    def enc(s):
        s.step(None)      # one sensing/control tick so trackers and estimators see the change
        return flatten_input(featurizer_for(s)(s.observe()))

    # 1) rejection memory: same physical state, different rejection history
    s1, s2 = session_factory(), session_factory()
    s2.runtime.request_event("place", s2.runtime.graph_version)   # rejected: prerequisite_unsatisfied
    s2.runtime.request_event("place", s2.runtime.graph_version)
    r = distinguishability(enc(s1), enc(s2), True)
    rows.append(dict(intervention="rejection_memory", **r.__dict__))

    # 2) role swap: actor binding changed to a different manipulator entity -> must differ
    s1, s2 = session_factory(), session_factory()
    ev = copy.deepcopy(s2.runtime.store.document())
    s2.runtime.apply_edit([{"op": "add_entity", "entity": {"id": "gripper_b", "type": "manipulator",
                                                         "descriptor": "second manipulator"}},
                           {"op": "bind_role", "event_id": "grasp", "role": "actor", "ordinal": 0,
                            "binding": {"kind": "entity", "entity": {"id": "gripper_b", "version": 0}}}],
                          s2.runtime.graph_version, "cf-role-swap")
    r = distinguishability(enc(s1), enc(s2), True)
    rows.append(dict(intervention="role_swap", **r.__dict__))

    # 3) graph rewiring (dependency removed)
    s1, s2 = session_factory(), session_factory()
    s2.runtime.apply_edit([{"op": "remove_dependency", "src": "grasp", "dst": "place", "mode": "completed"}],
                          s2.runtime.graph_version, "cf-rewire")
    r = distinguishability(enc(s1), enc(s2), True)
    rows.append(dict(intervention="graph_rewiring", **r.__dict__))

    # 4) provenance: valid receipt from the bound producer vs same-type receipt from a foreign event
    s1, s2 = session_factory(), session_factory()
    s1.runtime.receipts.put(Receipt("grasp", 0, "placed_ref", "frame_estimate", 0, {"pos": [0.4, 0.0, 0.1]}))
    s2.runtime.receipts.put(Receipt("foreign", 0, "placed_ref", "frame_estimate", 0, {"pos": [0.4, 0.0, 0.1]}))
    r = distinguishability(enc(s1), enc(s2), True)
    rows.append(dict(intervention="foreign_vs_correct_provenance", **r.__dict__))

    # 5) stale (invalidated) vs valid same-event output
    s1, s2 = session_factory(), session_factory()
    for s in (s1, s2):
        s.runtime.receipts.put(Receipt("grasp", 0, "o", "frame_estimate", 0, {"pos": [0.4, 0.0, 0.1]}))
    s2.runtime.receipts.invalidate("grasp", 0, "o", "stale")
    r = distinguishability(enc(s1), enc(s2), True)
    rows.append(dict(intervention="stale_vs_valid_output", **r.__dict__))

    # 6) irrelevant object moved (should NOT require a different action for the task object)
    s1, s2 = session_factory(n_distractors=1), session_factory(n_distractors=1)
    s2.teleport_object("distractor0", [0.3, 0.3, 0.023])
    r = distinguishability(enc(s1), enc(s2), False)
    rows.append(dict(intervention="irrelevant_object_moved", **r.__dict__))
    return rows
