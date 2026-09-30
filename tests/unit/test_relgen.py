"""relgen foundation skeletons (D-144 F4): exact batch allocation, steer grammar + validation + logging, scheduler
replay contract, full-world floor monotonicity. The former `ReadoutProbe == PacketProbe under the arm preset`
equivalence test lived here; PacketProbe is now deleted for real (D-144 R1 follow-up, archived research/tracks/rel-r1c.md)
so the comparison can no longer be run -- the invariant it proved is now load-bearing on the golden hashes in
tests/unit/test_golden.py and the key-set checks in tests/unit/test_relations_r1_probes.py instead."""
import pytest

from rrp.harness.data.mix import allocate
from rrp.harness.data.relgen.curriculum import Scheduler, SchedulerConfig, parse_steer


def test_allocate_is_exact():
    a = allocate(10, {"x": 0.25, "y": 0.25})
    assert sum(a.values()) == 10 and a["main"] == 5
    with pytest.raises(ValueError):
        allocate(4, {"x": 0.8, "y": 0.5})


def test_steer_grammar_and_validation():
    assert parse_steer("boost ix.support x2 for 5000").value == 2.0
    assert parse_steer("set full_world>=0.6").cmp == ">="
    assert parse_steer("revert to 1200").steps == 1200
    assert parse_steer({"op": "drop", "factor": "geo.depth3d"}).factor == "geo.depth3d"
    with pytest.raises(ValueError):
        parse_steer("teleport ix.support")
    s = Scheduler(SchedulerConfig(factors=("a", "b")), seed=1)
    ok = s.steer(parse_steer("boost a x2 for 100", author="lead", reason="a lags"), step=10)
    bad = s.steer(parse_steer("drop zzz"), step=10)
    assert ok.at == 10 and not ok.reason.startswith("REJECTED") and bad.reason.startswith("REJECTED")
    assert len(s.steer_log) == 2                                  # invalid steers are logged too


def test_scheduler_replay_and_floor():
    cfg = SchedulerConfig(factors=("a", "b", "c"), interval=10, ramp_steps=100)
    s = Scheduler(cfg, seed=3)
    s.steer(parse_steer("boost b x3 for 30"), step=20)
    s.steer(parse_steer("set full_world>=0.7"), step=40)
    seq = []
    for t in range(0, 80, 5):
        if t % cfg.interval == 0:
            s.decide(t)
        seq.append(s.sample(t, 32))
    _, seq2 = Scheduler.replay(cfg, 3, s.steer_log, s.metrics_log, range(0, 80, 5), 32)
    assert seq == seq2
    fw = [h.full_world for h in s.history]
    assert all(b >= a for a, b in zip(fw, fw[1:])) and fw[-1] >= 0.7
    st = s.history[2]                                             # step 20: b boosted
    assert st.share["b"] > st.share["a"]
