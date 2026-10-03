"""Arm physics perturbations are per-SUBSTEP hooks (perturb.install_arm wraps apply_substep, "called right before every
mj_step"): the generator-based Session.step (warpeval) must keep that interleaving on the CPU path, else the push window is
sampled once per control step at the step-start time and `push_applied_s` under-counts by the substep factor."""
import pytest

from rrp.envs.mujoco.perturb import PhysicsPerturbation, install_arm


def test_arm_push_is_applied_per_substep():
    from rrp.harness.eval.evaluate import env_factory
    s = env_factory("mujoco/arm", "pick_place", "parm5_pg2", scene=dict(n_distractors=0))(0)
    dt = float(s.model.opt.timestep)
    t0 = float(s.data.time)                                  # the reset settle has advanced time already
    pert = PhysicsPerturbation(push_impulse_Ns=2.0, push_time_s=t0 + 0.05, push_duration_s=0.1, push_dir_rad=0.0)
    st = install_arm(s, pert, seed=0)
    assert s.batch_reason() is not None                      # never batched on Warp
    t_end = t0 + 0.3
    while s.data.time < t_end:
        s.step(None)
    rec = st["push"].record()
    assert rec["push_applied_s"] == pytest.approx(0.1, abs=1.5 * dt)
    assert rec["push_impulse_applied_Ns"] == pytest.approx(2.0, rel=0.05)
