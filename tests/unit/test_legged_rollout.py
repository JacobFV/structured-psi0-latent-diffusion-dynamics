"""RP4 (D-146, docs/architecture.md 14.1): the legged tracker-validation episode runs on `harness.rollout` + hooks.
Rows of `tracker_validation.run_episode` on the procedural hexapod (CPG tracker, ideal and v1lat actuator, a lateral
kick, an early fall, the trajectory recorder) were recorded from the private mj_step loop BEFORE the port
(`loop.legged.tracker_validation` in tests/data/golden.json) and must stay byte-identical (floats rounded to 5
decimals). Plumbing only; no number here is a result."""
import pytest

from tests.unit.test_golden import golden  # noqa: F401  (recording fixture)
from tests.unit.test_ladder_rollout import _digest

pytest.importorskip("mujoco")


def _rows():
    from rrp.bodies.actuator import ActuatorModel
    from rrp.bodies.legged import legged_body, standalone_model
    from rrp.envs.mujoco.legged_core import LeggedBinding
    from rrp.envs.mujoco.legged_tracker import CPGTracker
    from rrp.harness.eval import tracker_validation as tv

    def build():
        model, _, meta = standalone_model(legged_body("hexapod6"), contact="v1")
        b = LeggedBinding(model, meta)
        return model, b, CPGTracker(b, meta), meta

    rows = {}
    model, b, tr, meta = build()
    sc = tv.scripts(b)
    short = lambda k, T, **kw: dict(sc[k], T=T, **kw)                                        # noqa: E731
    rows["stand"] = tv.run_episode(model, b, tr, short("stand", 0.8), 1000)
    rows["forward_rec"] = tv.run_episode(model, b, tr, short("forward", 2.4), 1001, record=True)
    rows["turn"] = tv.run_episode(model, b, tr, short("arc", 1.6), 1002)
    rows["push"] = tv.run_episode(model, b, tr, short("push_fwd", 1.6, push=(0.8, 0.3)), 1003)
    act = ActuatorModel(model, b, 1, None, name=meta["name"], randomize=False, latency_ms=10.0, mode="v1lat")
    rows["forward_lat"] = tv.run_episode(model, b, tr, short("forward", 1.6), 1004, act=act)
    model, b, tr, meta = build()
    b.min_h = 10.0                                                                          # falls on the first tick
    rows["fall"] = tv.run_episode(model, b, tr, short("forward", 1.6), 1005)
    return rows


def test_tracker_validation_episodes(golden):
    rows = _rows()
    assert rows["fall"]["fell"] and rows["fall"]["fell_t"] == 0.0 and not rows["stand"]["fell"]
    golden("loop.legged.tracker_validation", _digest(rows))
