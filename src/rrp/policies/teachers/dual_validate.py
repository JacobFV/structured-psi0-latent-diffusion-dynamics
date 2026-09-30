"""Dual-arm session factory for the scripted teachers (scripted_teacher, privileged).

Pair keys: "<left_robot>__<right_robot>" (two separately mounted catalogue robots) or a dual-arm body key such as
"aloha". The teacher validation itself (`run_one`, `summarize_validation`, `validate_main`: `rrp suite dual-validate`)
lives in `rrp.harness.eval.dual_teacher_quality`, which drives the episodes through `harness.rollout`.
"""
from __future__ import annotations


def make_session(task: str, pair: str, seed: int):
    from rrp.envs.mujoco.dual import make_dual_env
    return make_dual_env(task=task, body=pair, seed=seed)
