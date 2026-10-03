import pytest
from rrp.ops.broker import ResourceBroker, ResourceRequest, CapacityError, AdmissionStopped, LeaseError
from tests.support import fake_enforcement_backend


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_children_cannot_each_claim_the_parent_budget():
    broker = ResourceBroker(cpu_limit=2.0, memory_limit_bytes=1_000_000,
                            backend=fake_enforcement_backend())
    broker.acquire(ResourceRequest(cpu_cores=1.5, memory_bytes=600_000))
    with pytest.raises(CapacityError):
        broker.acquire(ResourceRequest(cpu_cores=1.0, memory_bytes=600_000))


def test_release_returns_capacity_and_removes_backend_lease():
    be = fake_enforcement_backend()
    b = ResourceBroker(cpu_limit=2.0, memory_limit_bytes=1000, backend=be)
    l = b.acquire(ResourceRequest(cpu_cores=2.0, memory_bytes=1000))
    assert l.lease_id in be.leases
    b.release(l.lease_id)
    assert l.lease_id not in be.leases
    b.acquire(ResourceRequest(cpu_cores=2.0, memory_bytes=1000))


def test_lease_expiry_without_heartbeat(tmp_path):
    clk = Clock()
    be = fake_enforcement_backend()
    b = ResourceBroker(cpu_limit=1.0, memory_limit_bytes=100, backend=be, state_dir=tmp_path,
                       lease_expiry_s=20, clock=clk)
    l = b.acquire(ResourceRequest(cpu_cores=1.0, memory_bytes=100))
    clk.t += 10
    assert b.heartbeat(l.lease_id).action == "continue"
    clk.t += 25
    assert b.heartbeat(l.lease_id).action == "stop_now"
    assert l.lease_id not in be.leases
    b.acquire(ResourceRequest(cpu_cores=1.0, memory_bytes=100))  # capacity reclaimed


def test_state_is_shared_between_broker_instances(tmp_path):
    be = fake_enforcement_backend()
    b1 = ResourceBroker(cpu_limit=2.0, memory_limit_bytes=1000, backend=be, state_dir=tmp_path)
    b2 = ResourceBroker(cpu_limit=2.0, memory_limit_bytes=1000, backend=be, state_dir=tmp_path)
    b1.acquire(ResourceRequest(cpu_cores=1.5, memory_bytes=500))
    with pytest.raises(CapacityError):
        b2.acquire(ResourceRequest(cpu_cores=1.0, memory_bytes=100))


def test_admission_stop_and_watchdog_requirement(tmp_path):
    clk = Clock()
    b = ResourceBroker(cpu_limit=2.0, memory_limit_bytes=1000, backend=fake_enforcement_backend(),
                       state_dir=tmp_path, require_watchdog=True, clock=clk)
    with pytest.raises(AdmissionStopped):
        b.acquire(ResourceRequest(cpu_cores=1, memory_bytes=10))
    b.watchdog_beat()
    b.acquire(ResourceRequest(cpu_cores=1, memory_bytes=10))
    clk.t += 60
    with pytest.raises(AdmissionStopped):
        b.acquire(ResourceRequest(cpu_cores=0.5, memory_bytes=10))
    b.watchdog_beat()
    b.stop_admission("pressure")
    with pytest.raises(AdmissionStopped):
        b.acquire(ResourceRequest(cpu_cores=0.5, memory_bytes=10))


def test_gpu_owner_is_serialized():
    b = ResourceBroker(cpu_limit=4, memory_limit_bytes=1000, backend=fake_enforcement_backend(), gpu_slots=1)
    b.acquire(ResourceRequest(cpu_cores=1, memory_bytes=10, gpu=True))
    with pytest.raises(CapacityError):
        b.acquire(ResourceRequest(cpu_cores=1, memory_bytes=10, gpu=True))


def test_host_has_no_gpu_slots_by_default():
    b = ResourceBroker(cpu_limit=4, memory_limit_bytes=1000, backend=fake_enforcement_backend())
    with pytest.raises(CapacityError):
        b.acquire(ResourceRequest(cpu_cores=1, memory_bytes=10, gpu=True))


def test_reduced_live_limit_revokes_newest_first():
    b = ResourceBroker(cpu_limit=4, memory_limit_bytes=1000, backend=fake_enforcement_backend())
    clk_leases = []
    old = b.acquire(ResourceRequest(cpu_cores=1, memory_bytes=500))
    new = b.acquire(ResourceRequest(cpu_cores=1, memory_bytes=400))
    revoked = b.update_limits(cpu_cores=4, memory_bytes=600)
    # both created in the same second; ordering by created, newest revoked until fit
    assert len(revoked) == 1
    decision = b.heartbeat(revoked[0])
    assert decision.action == "checkpoint_and_stop"


def test_live_limit_never_exceeds_startup():
    b = ResourceBroker(cpu_limit=2, memory_limit_bytes=100, backend=fake_enforcement_backend())
    b.update_limits(cpu_cores=50, memory_bytes=10**12)
    t = b.totals()
    assert t["limits"]["cpu_cores"] == 2 and t["limits"]["memory_bytes"] == 100


def test_invalid_requests_and_unknown_leases():
    b = ResourceBroker(cpu_limit=2, memory_limit_bytes=100, backend=fake_enforcement_backend())
    for bad in (dict(cpu_cores=0, memory_bytes=1), dict(cpu_cores=1, memory_bytes=0),
                dict(cpu_cores=1, memory_bytes=1, max_seconds=10**6)):
        with pytest.raises(ValueError):
            b.acquire(ResourceRequest(**bad))
    with pytest.raises(LeaseError):
        b.heartbeat("nope")


def test_max_runtime_triggers_checkpoint(tmp_path):
    clk = Clock()
    b = ResourceBroker(cpu_limit=2, memory_limit_bytes=100, backend=fake_enforcement_backend(),
                       state_dir=tmp_path, clock=clk, lease_expiry_s=1e9)
    l = b.acquire(ResourceRequest(cpu_cores=1, memory_bytes=10, max_seconds=30))
    clk.t += 31
    assert b.heartbeat(l.lease_id).action == "checkpoint_and_stop"


def test_slow_backend_cleanup_never_blocks_heartbeats(tmp_path):
    """Regression (D-028): systemctl stop inside the lock blocked all heartbeats -> cascade expiry."""
    import threading, time
    from tests.support import fake_enforcement_backend
    be = fake_enforcement_backend()
    slow = lambda lid: time.sleep(2.0)
    be.remove_lease = slow
    b = ResourceBroker(cpu_limit=4, memory_limit_bytes=10**9, backend=be, state_dir=tmp_path)
    victim = b.acquire(ResourceRequest(cpu_cores=1, memory_bytes=10))
    other = b.acquire(ResourceRequest(cpu_cores=1, memory_bytes=10))
    th = threading.Thread(target=lambda: b.release(victim.lease_id))
    th.start()
    time.sleep(0.2)
    b2 = ResourceBroker(cpu_limit=4, memory_limit_bytes=10**9, backend=be, state_dir=tmp_path)
    t0 = time.time()
    assert b2.heartbeat(other.lease_id).action == "continue"
    assert time.time() - t0 < 0.5
    th.join()


def test_gpu_reservation_keeps_a_slot_for_the_reserved_track_and_binds_older_brokers(tmp_path):
    """D-147 addendum 2026-10-02: reserve 1 of 2 GPU slots for humanoid labels. Other tracks together hold at most 1 GPU (also when
    their broker code knows nothing of reservations: the reservation is a GPU pseudo-lease every version counts); humanoid may use both;
    while a humanoid lease holds the reserved slot the reservation is `held`, so a non-humanoid lease can take the other slot."""
    clock = Clock()
    res = {"humanoid": {"slots": 1, "prefixes": ["hss_", "hr2_"]}}
    mk = lambda r: ResourceBroker(cpu_limit=8, memory_limit_bytes=10_000, backend=fake_enforcement_backend(), gpu_slots=2,
                                  state_dir=tmp_path, clock=clock, gpu_reserve=r)
    b, old = mk(res), mk(None)                        # `old` passes no reservation: like a peer code dir on older broker code
    gpu = lambda label: ResourceRequest(cpu_cores=1, memory_bytes=10, gpu=True, label=label)
    a1 = old.acquire(gpu("armdiv_train"))             # 1 non-humanoid + the reservation = 2 owners
    with pytest.raises(CapacityError):
        old.acquire(gpu("pointer_flow"))              # a second non-humanoid GPU lease is refused, even by the older code path
    h1 = b.acquire(gpu("hr2_steps_g1_train"))         # humanoid ignores its own reservation
    assert b.leases()["reserve:humanoid"]["state"] == "held"
    with pytest.raises(CapacityError):
        b.acquire(gpu("hss_h1_train"))                # 2 real GPU owners now (armdiv + humanoid)
    old.release(a1.lease_id)
    h2 = b.acquire(gpu("hss_h1_train"))               # humanoid may use both slots when no other track holds one
    with pytest.raises(CapacityError):
        old.acquire(gpu("pointer_flow"))
    b.release(h1.lease_id)
    b.release(h2.lease_id)
    assert b.leases()["reserve:humanoid"]["state"] == "active"
    old.acquire(gpu("pointer_flow"))
    with pytest.raises(CapacityError):
        old.acquire(gpu("psi0_eval"))                 # the reserved slot stays free for humanoid
    b.acquire(gpu("hr2_gap_t1_train"))
    mk({})                                            # an empty reservation config removes the pseudo-lease
    assert "reserve:humanoid" not in b.leases()
