from rrp.ops.watchdog import evaluate, WatchdogConfig, WatchdogState, run_loop
from rrp.ops.broker import ResourceBroker, ResourceRequest
from tests.support import fake_enforcement_backend

G = 1024 ** 3


def cfg(**kw):
    base = dict(memory_reserve_bytes=12 * G, disk_reserve_bytes=100 * G, startup_memory_bytes=20 * G,
                startup_cpu_cores=6.0, disk_path="/")
    base.update(kw)
    return WatchdogConfig(**base)


def ok_sample(**kw):
    s = dict(memory_available=46 * G, swap_free=0, psi_full_avg10=0.0, project_memory=1 * G,
             project_psi_full_avg10=0.0, disk_free=400 * G, thermal_c=60.0, gpu_temp_c=None,
             idle_cores=12.0, project_cpu_cores=0.5, telemetry_errors=[])
    s.update(kw)
    return s


def test_ok_sample():
    v = evaluate(ok_sample(), cfg(), WatchdogState())
    assert v.level == "ok"
    assert v.live_memory_bytes == 20 * G
    assert v.live_cpu_cores == 6.0


def test_memory_below_reserve_is_emergency():
    v = evaluate(ok_sample(memory_available=10 * G), cfg(), WatchdogState())
    assert v.level == "emergency"


def test_external_load_rise_sheds_when_project_exceeds_live_limit():
    # others consume memory: A_now=16G, project 10G -> live=min(20, 13, 14)=13G; project 10G ok
    v = evaluate(ok_sample(memory_available=16 * G, project_memory=10 * G), cfg(), WatchdogState())
    assert v.level == "ok" and v.live_memory_bytes == 13 * G
    # further drop: A_now=13G, project 12G -> live=min(20, 12.5, 13)=12.5 G > 12 ok; A_now=12.5 -> reserve breach
    v2 = evaluate(ok_sample(memory_available=13 * G, project_memory=16 * G), cfg(), WatchdogState())
    assert v2.level == "shed"


def test_swap_growth_escalates_only_recently_and_when_tight():
    st = WatchdogState()
    evaluate(ok_sample(swap_free=4 * G), cfg(), st)
    assert evaluate(ok_sample(swap_free=4 * G - 300 * 1024**2), cfg(), st).level == "stop_admission"
    assert evaluate(ok_sample(swap_free=2 * G, memory_available=20 * G), cfg(), st).level == "shed"
    st2 = WatchdogState()
    evaluate(ok_sample(swap_free=4 * G), cfg(), st2)
    assert evaluate(ok_sample(swap_free=2 * G), cfg(), st2).level == "stop_admission"   # plenty of RAM: no kill
    for _ in range(40):                                                                   # old growth ages out
        v = evaluate(ok_sample(swap_free=2 * G), cfg(), st2)
    assert v.level == "ok"


def test_psi_must_be_sustained_before_shedding():
    st = WatchdogState()
    levels = [evaluate(ok_sample(psi_full_avg10=50.0), cfg(), st).level for _ in range(3)]
    assert levels == ["stop_admission", "stop_admission", "shed"]


def test_telemetry_failure_stops_admission():
    v = evaluate(ok_sample(memory_available=None, telemetry_errors=["meminfo"]), cfg(), WatchdogState())
    assert v.level == "stop_admission"


def test_disk_and_thermal():
    assert evaluate(ok_sample(disk_free=50 * G), cfg(), WatchdogState()).level == "shed"
    assert evaluate(ok_sample(gpu_temp_c=96.0), cfg(), WatchdogState()).level == "shed"
    assert evaluate(ok_sample(thermal_c=96.0, cpu_freq_ratio=1.0), cfg(), WatchdogState()).level == "ok"
    assert evaluate(ok_sample(thermal_c=98.0, cpu_freq_ratio=1.0), cfg(), WatchdogState()).level == "stop_admission"
    assert evaluate(ok_sample(thermal_c=93.0, cpu_freq_ratio=0.5), cfg(), WatchdogState()).level == "shed"
    assert evaluate(ok_sample(thermal_c=101.0), cfg(), WatchdogState()).level == "shed"
    assert evaluate(ok_sample(gpu_thermal_throttle=True), cfg(), WatchdogState()).level == "shed"


def test_loop_revokes_only_owned_leases_and_hard_stops_after_grace(tmp_path):
    be = fake_enforcement_backend()
    b = ResourceBroker(cpu_limit=4, memory_limit_bytes=8 * G, backend=be, state_dir=tmp_path / "b")
    l1 = b.acquire(ResourceRequest(cpu_cores=1, memory_bytes=G))
    samples = iter([ok_sample(memory_available=10 * G)] * 3)
    run_loop(b, be, cfg(), interval_s=0.0, log_path=tmp_path / "wd.jsonl", max_iterations=3,
             checkpoint_grace_s=0.0, emergency_grace_s=0.0, sample_fn=lambda c, s: next(samples))
    assert b.totals()["admission_stopped"]
    assert l1.lease_id not in be.leases  # hard stop after (zero) emergency grace
    assert b.leases()[l1.lease_id]['state'] == 'released'
