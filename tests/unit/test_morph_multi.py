"""MorphMultiEnv (D14): CPU fake-env test of the routing / concatenation and of the group dimension contract.

`MorphMultiEnv(env_cls=...)` builds the real per-body `MorphSpec` (slot map, signs, context; host MuJoCo only) around
whatever engine `env_cls` returns, so a torch-CPU fake engine exercises the production routing without warp / CUDA.
The engines are h1 (10 leg actuators) and t1 (12, different slot map): every group must agree on `obs_dim`, `priv_dim`,
the 14 canonical slot actions and the morph_v1 layout, whatever its own actuator count.
"""
import types

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.menagerie

from rrp.envs.mujoco.morph_obs import (CTX_DIM, DYN_DIM, NS, OBS_DIM, UP_MAX, UP_PRIV_DIM, UPPER_DIM,  # noqa: E402
                                       MorphSpec, obs_format)
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
    with pytest.raises(ValueError, match="critic width"):
        MorphMultiEnv([("h1", 2), ("t1", 2)], seed=3, obs_noise=0.0, env_cls=Wide)


# ---------------------------------------------------------------------------------------------------------------- morph_v2 (HS2)
class FakeUpper(FakeEngine):
    """FakeEngine + the wholebody surface of WarpTrackerEnv (upper_body, held joints, payload / offset critic tail)."""

    def __init__(self, keys, nworld, upper_body=False, require_payload=True, **kw):
        super().__init__(keys, nworld, **kw)
        b = self.b
        self.upper_body, self.require_payload = bool(upper_body), require_payload
        self.upper_amp, self.upper_speed, self.payload_frac = 0.4, 1.5, 0.08
        self.ramps = []
        self.cfg0 = types.SimpleNamespace(options=lambda: {})
        if self.upper_body:
            assert len(b.held_act)
            self.held_qadr = torch.as_tensor(np.asarray(b.held_qadr), dtype=torch.long)
            self.held_dadr = torch.as_tensor(np.asarray(b.held_dadr), dtype=torch.long)
            self.q0_held = torch.as_tensor(np.asarray(b.q0_held), dtype=torch.float32).expand(self.N, -1).clone()
            self.upper_dim = 2 * len(b.held_act)
            self.priv_dim += 1 + len(b.held_act)
        else:
            self.upper_dim = 0

    def set_upper_ramp(self, amp, payload_frac):
        self.upper_amp, self.payload_frac = amp, payload_frac
        self.ramps.append((amp, payload_frac))

    def upper_meta(self):
        return dict(amp=self.upper_amp, payload_frac=self.payload_frac)



def _env2(groups, **kw):
    return MorphMultiEnv(groups, seed=3, obs_noise=0.0, env_cls=FakeUpper, upper_body=True, **kw)


def test_morph_v2_dims_padding_and_the_numpy_twin():
    env = _env2([("h1", 3), ("t1", 4)])
    assert env.upper_dim == UPPER_DIM and env.obs_dim == OBS_DIM + EXTRA + UPPER_DIM
    assert obs_format(True) == "morph_v2" and obs_format(False) == "morph_v1"
    e_h, e_t = env.envs
    assert (e_h.upper_dim, e_t.upper_dim) == (18, 22)              # 9 and 11 upper joints: different per body ...
    obs, priv = env.observe(), env.privileged()
    assert obs.shape == (7, env.obs_dim) and priv.shape == (7, env.priv_dim)     # ... one actor / critic width
    base = e_h.priv_dim - (1 + 9)
    assert env.priv_dim == base + UP_PRIV_DIM == e_t.priv_dim - (1 + 11) + UP_PRIV_DIM
    for e, g, sl in zip(env.envs, env.specs, env.slices):
        blk = obs[sl, -UPPER_DIM:]
        n = g["n_up"]
        assert n in (9, 11)
        q, qd = blk[:, :UP_MAX], blk[:, UP_MAX:2 * UP_MAX]
        assert torch.allclose(q[:, :n], e.qpos[:, e.held_qadr] - e.q0_held) and torch.all(q[:, n:] == 0)
        assert torch.allclose(qd[:, :n], e.qvel[:, e.held_dadr] * 0.05) and torch.all(qd[:, n:] == 0)
        # the numpy deployment twin (MorphSpec.upper_obs) produces the same row from the same state
        import mujoco
        d = mujoco.MjData(e.m)
        d.qpos[:] = e.qpos[0].numpy()
        d.qvel[:] = e.qvel[0].numpy()
        sp = MorphSpec(e.m, e.b, e.meta)
        assert np.allclose(sp.upper_obs(d), blk[0].numpy(), atol=1e-6)
        pr = priv[sl]                                               # critic tail: payload fraction (faked) + offsets, padded
        assert pr.shape[1] == env.priv_dim and torch.all(pr[:, -(UP_MAX - n):] == 0)
    assert not torch.equal(obs[0, -UPPER_DIM + 2 * UP_MAX:], obs[3, -UPPER_DIM + 2 * UP_MAX:])   # static descriptors differ by body
    assert torch.all(obs[:, -UPPER_DIM + 2 * UP_MAX:][:, 0:10][:1, 0] == 1.0)                        # slot 0 present


def test_a_group_without_an_upper_body_contributes_a_zero_block_and_a_zero_critic_tail():
    env = _env2([("h1", 3), ("phum_1", 2)])
    e_h, e_p = env.envs
    assert e_h.upper_body and not e_p.upper_body and not e_p.require_payload
    obs, priv = env.observe(), env.privileged()
    assert obs.shape == (5, env.obs_dim) and priv.shape == (5, env.priv_dim)
    assert torch.all(obs[3:, -UPPER_DIM:] == 0) and torch.all(priv[3:, -UP_PRIV_DIM:] == 0)
    assert torch.any(obs[:3, -UPPER_DIM:] != 0)
    obs2, priv2, *_ = env.step(torch.zeros(5, NS))                   # the step path pads the same way
    assert priv2.shape == (5, env.priv_dim) and obs2.shape == obs.shape


def test_upper_ramp_reaches_only_the_groups_with_an_upper_body():
    env = _env2([("h1", 3), ("phum_1", 2)])
    env.set_upper_ramp(0.2, 0.04)
    assert env.envs[0].ramps == [(0.2, 0.04)] and env.envs[1].ramps == []
    assert env.upper_meta()["amp"] == 0.2 and env.upper_meta()["groups"]["phum_1"] is None


def test_without_upper_body_the_layout_is_morph_v1_unchanged():
    env = _env()
    assert env.upper_dim == 0 and env.obs_dim == OBS_DIM + EXTRA


# ------------------------------------------------------------------------------------ PPO on the wholebody env (2 fake updates)
class StubUpperEnv:
    """The WarpTrackerEnv surface `warp_tracker_ppo.train` reads, on CPU (a `*_ub` per-body run): random observations, a reward
    that depends on the action, and a recorded set_upper_ramp call per iteration."""

    def __init__(self, N=8, upper=True):
        g = torch.Generator().manual_seed(0)
        self.dev, self.N, self.nf, self.nA, self.dt = torch.device("cpu"), N, 2, 5, 0.02
        self.extra_dim, self.upper_dim = 0, 6 if upper else 0
        self.obs_dim, self.priv_dim = 11 + self.upper_dim, 4
        self.b = types.SimpleNamespace(kind="humanoid")
        self.meta, self.adaptations, self.cfg0 = {}, [], types.SimpleNamespace(options=lambda: {})
        self.g, self.ramps, self.upper_body = g, [], upper

    def set_alpha(self, a):
        return {}

    def set_upper_ramp(self, amp, pl):
        self.ramps.append((amp, pl))

    def upper_meta(self):
        return dict(amp=self.ramps[-1][0] if self.ramps else None)

    def observe(self):
        return torch.randn(self.N, self.obs_dim, generator=self.g)

    def privileged(self, fc=None):
        return torch.randn(self.N, self.priv_dim, generator=self.g)

    def step(self, a):
        d = torch.zeros(self.N, dtype=torch.bool)
        return self.observe(), self.privileged(), -a.pow(2).sum(-1), d, d.clone()

    def pop_stats(self):
        return dict(episodes=0, ret_sum=0.0, len_sum=0.0, falls=0, successes=0, gm=dict(track_err=0.0, cmd=0.0, steps=0.0))


def _ppo_args(tmp_path, *extra):
    from rrp.harness.train.warp_tracker_ppo import build_args
    return build_args(["--out", str(tmp_path), "--iters", "2", "--horizon", "4", "--epochs", "1", "--minibatches", "2",
                       "--hidden", "16,16", "--ckpt-every", "1", "--alpha-schedule", "fixed:1.0", *extra])


def _logs(path):
    import json
    return [json.loads(x) for x in (path / "train_log.jsonl").read_text().splitlines()]


def test_ub_recipe_two_ppo_updates_ramp_and_meta(tmp_path):
    from rrp.harness.train.warp_tracker_ppo import train
    args = _ppo_args(tmp_path, "--body", "t1", "--upper-body", "--upper-amp", "0.4", "--upper-amp0", "0.1", "--upper-ramp", "1.0")
    env = StubUpperEnv()
    train(args, env, dev=torch.device("cpu"), engine="fake")
    assert env.ramps == [(0.1, 0.0), (pytest.approx(0.25), pytest.approx(0.04))]           # linear over the 2 iterations
    log = _logs(tmp_path)
    assert [r["upper_amp"] for r in log] == [pytest.approx(0.1), pytest.approx(0.25)] and log[1]["payload_frac"] == pytest.approx(0.04)
    import json
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["upper_obs"] is True and meta["upper_ramp"]["ramp"] == 1.0 and meta["obs_dim"] == env.obs_dim
    st = torch.load(tmp_path / "actor.pt", weights_only=False)
    assert st["meta"]["upper_ramp"]["amp0"] == 0.1 and st["actor"]["0.weight"].shape[1] == env.obs_dim


def test_no_upper_body_means_no_ramp_and_no_upper_meta(tmp_path):
    from rrp.harness.train.warp_tracker_ppo import train
    env = StubUpperEnv(upper=False)
    train(_ppo_args(tmp_path, "--body", "t1"), env, dev=torch.device("cpu"), engine="fake")
    assert env.ramps == [] and "upper_amp" not in _logs(tmp_path)[0]


@pytest.mark.menagerie
def test_morph_v2_recipe_two_ppo_updates_on_the_shared_env(tmp_path):
    """The shared wholebody tracker: MorphMultiEnv groups (h1, t1 with upper bodies + a legs-only phum) through the real trainer."""
    import json
    from rrp.harness.train.warp_tracker_ppo import train
    groups = [[["h1"], 3], [["t1"], 3], [["phum_1"], 2]]
    args = _ppo_args(tmp_path, "--groups", json.dumps(groups), "--upper-body", "--upper-ramp", "0.5")
    env = MorphMultiEnv(groups, seed=3, obs_noise=0.0, env_cls=FakeUpper, upper_body=True)
    train(args, env, groups=groups, dev=torch.device("cpu"), engine="fake")
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["obs_format"] == "morph_v2" and meta["obs_dim"] == OBS_DIM + EXTRA + UPPER_DIM and meta["shared"]
    assert meta["train_bodies"] == ["h1", "phum_1", "t1"] and meta["upper_obs"] is True
    assert [r["upper_amp"] for r in _logs(tmp_path)] == [pytest.approx(0.1), pytest.approx(0.4)]
    assert [e.ramps for e in env.envs][2] == []                                               # the legs-only group is never ramped
    assert env.envs[0].ramps[0] == (pytest.approx(0.1), 0.0)
    st = torch.load(tmp_path / "actor.pt", weights_only=False)
    assert st["actor"]["0.weight"].shape[1] == OBS_DIM + EXTRA + UPPER_DIM
