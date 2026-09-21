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
