"""Pointer system 0 (rrp.policies.pointer): knot/tick mapping and the engineered packet encoding (pure), and the
oracle -> engineered system 0 route through rrp eval's harness on every cw/* task (marked `computerworld`)."""
import numpy as np
import pytest

from rrp.policies.pointer import (ENG_DIM, KNOT_TIMES, decode_slot, encode_commands, packet_ticks, tick_slot)


def test_tick_slots_cover_knots_at_10hz():
    assert packet_ticks(0.1) == [(0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
    assert tick_slot(0.7, 0.1) is None and tick_slot(0.05, 0.1) == (1, 0)


def test_engineered_encoding_roundtrip():
    cmds = [{"pointer": [0.1, -0.2], "button": [0.0]}, {"pointer": [0.12, -0.2], "button": [1.0], "key": [17.0]},
            None, {"pointer": [0.3, 0.1], "button": [0.0], "wheel": [-2.0]}]
    z = encode_commands(cmds, 0.1, 109, depths=[0.0, 0.004, 0.0, 0.002])
    assert z.shape == (len(KNOT_TIMES), 1, ENG_DIM)
    for j, (k, s) in enumerate(packet_ticks(0.1)):
        d = decode_slot(z, k, s, 109)
        if j >= len(cmds) or cmds[j] is None:
            assert d is None
            continue
        c = cmds[j]
        assert (d["x"], d["y"]) == pytest.approx(c["pointer"], abs=1e-6)
        assert d["button"] == (c["button"][0] >= 0.5)
        assert d["key"] == int(c.get("key", [-1])[0]) and d["wheel"] == int(c.get("wheel", [0])[0])


@pytest.mark.computerworld
@pytest.mark.parametrize("task", ["cw/calc_sum", "cw/open_type", "cw/drag_window", "cw/fill_form"])
def test_oracle_packets_drive_every_task(task):
    from rrp.harness.eval.evaluate import evaluate
    from rrp.policies.base import make_policy
    pol = make_policy("pointer_oracle")
    eps = evaluate(pol, "computerworld", task, "cw_pointer", [0, 1, 2])
    assert [e.outcome for e in eps] == ["success"] * 3
    assert all(e.metrics["packets"] > 0 and e.metrics["command_rejections"] == 0 for e in eps)
    assert all(s.stats.rejected == 0 for s in pol.s0)
