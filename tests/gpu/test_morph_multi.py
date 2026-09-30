"""MorphMultiEnv on the real Warp engine (peer GPU only; RRP_TEST_GPU=1 under an ops lease): two morphologies, 10 ticks, independent per-group reset."""
import os

import pytest
import torch

pytest.importorskip("mujoco_warp")
pytest.importorskip("warp")
if not torch.cuda.is_available() or not os.environ.get("RRP_GPU_MEMORY_BYTES"):
    pytest.skip("peer GPU lease required", allow_module_level=True)
pytestmark = pytest.mark.menagerie


def test_two_groups_step_ten_ticks_and_reset_independently():
    from rrp.envs.mujoco.morph_obs import NS
    from rrp.envs.warp.tracker_env import MorphMultiEnv
    from rrp.ops.workload import apply_cap
    apply_cap()
    env = MorphMultiEnv([("h1", 8), ("t1", 8)], seed=5, nconmax=48, njmax=320)
    e_h, e_t = env.envs
    assert (env.N, env.nA, e_h.nA, e_t.nA) == (16, NS, 10, 12)
    obs = env.observe()
    assert obs.shape == (16, env.obs_dim) and env.privileged().shape == (16, env.priv_dim)
    gen = torch.Generator(device=env.dev).manual_seed(0)
    for _ in range(10):
        obs, priv, r, d, tmo = env.step(0.3 * torch.randn(16, NS, device=env.dev, generator=gen))
        assert obs.shape == (16, env.obs_dim) and priv.shape == (16, env.priv_dim)
        assert r.shape == d.shape == tmo.shape == (16,)
        assert torch.isfinite(obs).all() and torch.isfinite(priv).all() and torch.isfinite(r).all()
    # reset only some h1 worlds: the t1 group's physical state is untouched, the reset h1 worlds return to the start pose
    t_before, h_before = e_t.qpos.clone(), e_h.qpos.clone()
    mask = torch.zeros(e_h.N, dtype=torch.bool, device=env.dev)
    mask[:3] = True
    e_h._reset(mask)
    assert torch.equal(e_t.qpos, t_before)
    assert torch.equal(e_h.qpos[3:], h_before[3:])
    assert not torch.equal(e_h.qpos[:3], h_before[:3]) and torch.all(e_h.qvel[:3] == 0)
    env.step(torch.zeros(16, NS, device=env.dev))
    st = env.pop_stats()
    assert set(st["per_group"]) == {"h1", "t1"}
