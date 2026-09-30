"""MorphMultiEnv (D14): CPU fake-env test of the routing / concatenation and of the group dimension contract.

`MorphMultiEnv(env_cls=...)` builds the real per-body `MorphSpec` (slot map, signs, context; host MuJoCo only) around
whatever engine `env_cls` returns, so a torch-CPU fake engine exercises the production routing without warp / CUDA.
The engines are h1 (10 leg actuators) and t1 (12, different slot map): every group must agree on `obs_dim`, `priv_dim`,
the 14 canonical slot actions and the morph_v1 layout, whatever its own actuator count.
"""
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.menagerie

from rrp.envs.mujoco.morph_obs import CTX_DIM, DYN_DIM, NS, OBS_DIM  # noqa: E402
from rrp.envs.warp.tracker_env import ADAPT, MorphMultiEnv  # noqa: E402
from rrp.envs.warp.model import build_model  # noqa: E402

EXTRA = 2          # a public extra-observation block (terrain-scan stand-in) so its concatenation is exercised too


class FakeEngine:
    """The WarpTrackerEnv surface MorphMultiEnv reads, on CPU: real body binding, deterministic fake state."""

    def __init__(self, keys, nworld, seed=1, obs_noise=0.0, priv_pad=0, done_at=None, **_):
        self.dev = torch.device("cpu")
        self.variant_keys, self.N, self.K = list(keys), int(nworld), len(keys)
        self.m, self.meta, self.b, self.adaptations = build_model(keys[0], "v2", ADAPT)
        b = self.b
        self.body, self.nA, self.nf, self.qa, self.da = keys[0], b.n, b.nf, b.qa, b.da
        self.pol_qadr = torch.as_tensor(np.asarray(b.pol_qadr), dtype=torch.long)
        self.pol_dadr = torch.as_tensor(np.asarray(b.pol_dadr), dtype=torch.long)
        self.q0 = torch.as_tensor(np.asarray(b.q0), dtype=torch.float32).expand(self.N, -1).clone()
        self.extra_dim, self.priv_dim = EXTRA, b.priv_dim + 1 + EXTRA + priv_pad
        self.cfg0, self.dt = None, 0.02
        g = torch.Generator().manual_seed(seed)
        self.qpos = torch.randn(self.N, self.m.nq, generator=g)
        self.qpos[:, self.qa + 3:self.qa + 7] = torch.tensor([1.0, 0, 0, 0])      # identity root orientation
        self.qvel = torch.randn(self.N, self.m.nv, generator=g)
        self.cmd = torch.randn(self.N, 3, generator=g)
        self.phase = torch.rand(self.N, generator=g)
        self.t = torch.zeros(self.N)
        self.done_at = (done_at or {}).get(keys[0], {})     # body -> tick -> bool mask of the worlds that end then
        self.got = []                                # every action tensor this engine received
        self.n_pop = 0

    def set_alpha(self, a):
        self.alpha = a
        return {"engine": self.body}

    def _gravity_body(self):
        return torch.tensor([0.0, 0.0, -1.0]).expand(self.N, 3).clone()

    def clock_obs(self):
        return torch.stack([torch.sin(2 * np.pi * self.phase), torch.cos(2 * np.pi * self.phase)], -1)

    def extra_obs(self):
        return torch.full((self.N, EXTRA), 0.5)

    def privileged(self, fc):
        return torch.full((self.N, self.priv_dim), float(len(self.body)))

    def step(self, a):
        self.got.append(a.clone())
        tick = len(self.got)
        d = self.done_at.get(tick, torch.zeros(self.N, dtype=torch.bool))
        r = a.sum(-1)
        self.n_pop += int(d.sum())
        return None, self.privileged(d), r, d, torch.zeros_like(d)

    def pop_stats(self):
        return dict(episodes=self.n_pop, ret_sum=1.0, len_sum=2.0, falls=0, successes=self.n_pop,
                    gm=dict(track_err=1.0, cmd=2.0, steps=3.0))


def _env(nh=4, nt=6, **kw):
    """h1 x nh worlds, then t1 x nt worlds (the second group is bigger and has more actuators)."""
    env = MorphMultiEnv([("h1", nh), ("t1", nt)], seed=3, obs_noise=0.0, env_cls=FakeEngine, **kw)
    for g in env.specs:                 # h1 / t1 have no mirrored joints (all signs +1): flip every other slot so the
        g["sign"][::2] *= -1            # sign handling in observe and step is not vacuous
    return env


def test_groups_agree_on_dims_and_concatenate():
    env = _env()
    e_h, e_t = env.envs
    assert (e_h.nA, e_t.nA) == (10, 12) and env.nA == NS == 14      # engines differ, the interface is the 14 canonical slots
    assert env.N == 10 and env.slices == [slice(0, 4), slice(4, 10)] and env.names == ["h1", "t1"]
    assert env.obs_dim == OBS_DIM + EXTRA and env.dyn_dim == DYN_DIM and env.ctx_dim == CTX_DIM
    obs, priv = env.observe(), env.privileged()
    assert obs.shape == (10, env.obs_dim) and priv.shape == (10, env.priv_dim)
    assert e_h.priv_dim == e_t.priv_dim == env.priv_dim
    # priv is the per-group critic block, concatenated in group order (the fake tags it with the body-name length)
    assert torch.all(priv[:4] == len("h1")) and torch.all(priv[4:] == len("t1"))
    # obs layout (morph_v1): gyro 3 | grav 3 | cmd 3 | q 14 | qd 14 | last 14 | clock 2 | ctx 103 | extra
    for e, g, sl in zip(env.envs, env.specs, env.slices):
        o = obs[sl]
        ctx = o[:, DYN_DIM:DYN_DIM + CTX_DIM]
        assert torch.equal(ctx, g["ctx"]) and torch.equal(o[:, -EXTRA:], torch.full((e.N, EXTRA), 0.5))
        assert torch.all(o[:, 9 + 2 * NS:9 + 3 * NS] == 0)          # nothing sent yet
        q = o[:, 9:9 + NS]
        want = torch.zeros(e.N, NS)
        want[:, g["slots"]] = g["sign"] * (e.qpos[:, e.pol_qadr] - e.q0)[:, g["idx"]]
        assert torch.allclose(q, want)
    assert not torch.equal(obs[0, DYN_DIM:DYN_DIM + CTX_DIM], obs[5, DYN_DIM:DYN_DIM + CTX_DIM])    # morphology differs
    absent = [j for j in range(NS) if j not in env.specs[0]["slots"].tolist()]
    assert absent and torch.all(obs[:4, 9:9 + NS][:, absent] == 0)  # slots the body lacks stay zero (and padded in noise)


def test_actions_route_to_the_owning_group_with_slot_signs():
    env = _env()
    a = torch.arange(1, env.N * NS + 1, dtype=torch.float32).reshape(env.N, NS) / 50
    env.step(a)
    for e, g, sl in zip(env.envs, env.specs, env.slices):
        assert len(e.got) == 1
        got, slot = e.got[0], a[sl].clamp(-5, 5)
        assert got.shape == (e.N, e.nA)
        want = torch.zeros(e.N, e.nA)
        want[:, g["idx"]] = g["sign"] * slot[:, g["slots"]]
        assert torch.equal(got, want)                 # rows of another group never reach this engine
    big = torch.full((env.N, NS), 9.0)
    env.step(big)
    assert env.envs[0].got[1].abs().max() == 5.0      # canonical actions are clamped to +-5 before routing


def test_step_concatenates_and_resets_last_action_per_group():
    ends = torch.zeros(6, dtype=torch.bool)
    ends[[1, 4]] = True                               # only t1 worlds 1 and 4 (global rows 5 and 8) end on tick 1
    env = MorphMultiEnv([("h1", 4), ("t1", 6)], seed=3, obs_noise=0.0, env_cls=FakeEngine, done_at={"t1": {1: ends}})
    a = torch.ones(env.N, NS)
    obs, priv, r, d, tmo = env.step(a)
    assert obs.shape == (10, env.obs_dim) and priv.shape == (10, env.priv_dim)
    assert r.shape == d.shape == tmo.shape == (10,)
    assert d.tolist() == [False] * 5 + [True, False, False, True, False]
    assert torch.allclose(r[:4], torch.full((4,), 10.0)) and torch.allclose(r[4:], torch.full((6,), 12.0))    # per-engine widths
    last = obs[:, 9 + 2 * NS:9 + 3 * NS]
    pm = torch.cat([g["pm"][None].expand(e.N, -1) for e, g in zip(env.envs, env.specs)])
    keep = ~d
    assert torch.equal(last[keep], pm[keep])                                  # last slot action, present slots only
    assert torch.all(last[d] == 0)                                            # finished worlds start clean
    st = env.pop_stats()
    assert st["episodes"] == 2 and st["successes"] == 2 and st["gm"]["cmd"] == 4.0
    assert st["per_group"]["t1"]["episodes"] == 2 and st["per_group"]["h1"]["episodes"] == 0
    assert st["per_group"]["h1"]["track_rel_err"] == 0.5


def test_padded_slots_get_no_observation_noise():
    env = MorphMultiEnv([("h1", 8)], seed=3, obs_noise=1.0, env_cls=FakeEngine)
    o = env.observe()
    absent = [j for j in range(NS) if j not in env.specs[0]["slots"].tolist()]
    assert absent and torch.all(o[:, 9:9 + NS][:, absent] == 0) and torch.all(o[:, 9 + NS:9 + 2 * NS][:, absent] == 0)
    assert o[:, :3].std() > 0                                                  # noise is on for the real channels


def test_a_group_with_another_critic_width_cannot_be_concatenated():
    """The trainer needs ONE priv_dim; a group that disagrees fails loudly instead of yielding a ragged batch."""
    class Wide(FakeEngine):
        def __init__(self, keys, nworld, **kw):
            super().__init__(keys, nworld, priv_pad=1 if keys[0] == "t1" else 0, **kw)
    env = MorphMultiEnv([("h1", 2), ("t1", 2)], seed=3, obs_noise=0.0, env_cls=Wide)
    with pytest.raises(RuntimeError):
        env.privileged()
