"""gait_v2 reward schedule: priors decay to a floor, natural terms ramp, permanent terms fixed; gated alpha."""
import pytest

from rrp.envs.legged_core import MIN_STOP_SHARE, NATURAL_TERMS, PERMANENT_STANDING_TERMS, PRIOR_TERMS, RewardCfg
from rrp.training.reward_schedule import AlphaGate, window_metrics


@pytest.mark.parametrize("kind", ["humanoid", "quadruped"])
def test_effective_weights(kind):
    c0 = RewardCfg.for_kind(kind, "gait_v2")
    e0, e1, eh = c0.effective(0.0), c0.effective(1.0), c0.effective(0.5)
    for t in PRIOR_TERMS:
        assert getattr(e0, t) == pytest.approx(getattr(c0, t))
        assert getattr(e1, t) == pytest.approx(0.1 * getattr(c0, t))
        assert getattr(eh, t) == pytest.approx(0.55 * getattr(c0, t))
    for t in NATURAL_TERMS:
        assert getattr(e0, t) == pytest.approx(getattr(c0, t))
        assert getattr(e1, t) == pytest.approx(c0.natural_max[t])
        assert abs(getattr(e1, t)) >= abs(getattr(e0, t))
    for t in ("track_lin", "track_ang", "slip", "termination", "orient", "limits", "height") + PERMANENT_STANDING_TERMS:
        assert getattr(e1, t) == getattr(c0, t)          # permanent
    assert not set(PERMANENT_STANDING_TERMS) & set(PRIOR_TERMS)
    assert c0.stand_contact > 0 and c0.stand_vel < 0
    assert c0.slip < 0 and c0.effective(3.0).alpha == 1.0


def test_gate_advance_hold_backoff():
    g = AlphaGate(step=0.1, warmup=10)
    good = dict(track_rel_err=0.2, fall_rate=0.02, slip_ratio=0.1)
    assert g.update(5, good) == "skip" and g.alpha == 0
    assert g.update(10, good) == "advance" and g.alpha == pytest.approx(0.1)
    assert g.update(11, dict(good, slip_ratio=0.25)) == "hold" and g.alpha == pytest.approx(0.1)
    assert g.update(12, dict(good, fall_rate=0.3)) == "backoff" and g.alpha == 0
    for i in range(20):
        g.update(20 + i, good)
    assert g.alpha == 1.0


def test_window_metrics():
    st = [dict(ret=1, len=10, fell=True), dict(ret=1, len=10, fell=False),
          dict(gm=dict(steps=2, track_err=0.2, cmd=1.0, slip=0.1, speed=1.0, power=10.0, cot_den=20.0))]
    m = window_metrics(st)
    assert m["fall_rate"] == 0.5 and m["track_rel_err"] == pytest.approx(0.2)
    assert m["slip_ratio"] == pytest.approx(0.1) and m["cot"] == pytest.approx(0.5)


def test_stop_share_in_every_sampler():
    """At least MIN_STOP_SHARE zero commands from the default and the teacher-mix samplers (standing is trained)."""
    import warnings
    import numpy as np
    from rrp.bodies.legged import legged_body
    from rrp.envs.legged_core import LeggedEnv
    warnings.filterwarnings("ignore")
    for body in ("t1", "go2"):
        env = LeggedEnv(lambda: legged_body(body), 1, 0, contact="v2")
        for mix in ("default", "teacher"):
            env.cmd_mix = mix
            zeros = 0
            for _ in range(3000):
                env._sample_cmd(0)
                zeros += not np.any(env.cmd[0])
            assert zeros / 3000 >= MIN_STOP_SHARE - 0.02, (body, mix, zeros / 3000)
