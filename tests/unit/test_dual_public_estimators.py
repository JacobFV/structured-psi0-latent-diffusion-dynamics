"""Leakage guard for the dual-arm public estimators and receipts (support_insert).

Public predicate estimates and the locate receipt must not change when privileged-only
information (entity->sim-body map, privileged layout) is corrupted, and the hole frame receipt
must come from noisy detections, not from the simulator's true hole pose.
"""
import copy

import numpy as np

from rrp.control.dual_teachers import SupportInsertTeacher
from rrp.morphology.variants import registered_variants
from rrp.sim.dual import DualSession
from rrp.sim.dual_scenarios import build_support_insert


def _session():
    W = registered_variants()
    return DualSession(build_support_insert([W["parm5l_pg2"](), W["parm6_pg2"]()], 3), seed=3)


def _estimates(s):
    return {(p, tuple(a)): s.estimate(p, a)[:2] for p, a in s._task_predicates()}


def test_public_estimates_ignore_privileged_bus():
    s = _session()
    t = SupportInsertTeacher(s)
    for _ in range(60):
        s.step(t.act())
    before = _estimates(s)
    s.scenario.meta["privileged_layout"] = {k: None for k in s.scenario.meta["privileged_layout"]}
    for o in s.scenario.objects:
        o.task_entity = None
    after = _estimates(s)
    assert before == after
    r = s.runtime.receipts.latest("locate", 0, "hole_frame")
    assert r is not None and r.valid
    true_hole = s._body_pose("hole")[0]
    err = np.linalg.norm(np.array(r.value["pos"]) - true_hole)
    assert 0.0 < err < 0.006          # averaged noisy detections: close to, never equal to, the truth
    assert r.value["estimator"] == "static_feature_average/v1"
