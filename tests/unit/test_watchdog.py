from rrp.orchestration.watchdog import evaluate, WatchdogConfig, WatchdogState, run_loop
from rrp.orchestration.broker import ResourceBroker, ResourceRequest
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


def test_peer_ram_store_shmem_is_subtracted_from_the_startup_cap():
    # D-111 follow-up (fake peer sample): 108 GiB cap sized on an empty RAM store; 48 GiB now in tmpfs (Shmem).
    # MemAvailable already excludes Shmem, so only the static startup term is reduced: 108 - 48 = 60 GiB.
    peer = dict(memory_reserve_bytes=6 * G, disk_reserve_bytes=10 * G, startup_memory_bytes=108 * G,
                startup_cpu_cores=19.97, disk_path="/", fraction=1.0, psi_full_avg10_shed=101.0)
    s = ok_sample(memory_available=70 * G, project_memory=5 * G, shmem=48 * G)
    assert evaluate(s, cfg(**peer), WatchdogState()).live_memory_bytes == 69 * G        # min(108, 75, 75 - 6)
    s2 = ok_sample(memory_available=100 * G, project_memory=5 * G, shmem=48 * G)
    off = evaluate(s2, cfg(**peer), WatchdogState())
    on = evaluate(s2, cfg(**peer, subtract_shmem=True), WatchdogState())
    assert off.live_memory_bytes == 99 * G and on.live_memory_bytes == 60 * G
    # a store larger than the cap clamps at 0 (admission of new declared memory stops; nothing negative)
    assert evaluate(ok_sample(memory_available=100 * G, project_memory=0, shmem=200 * G), cfg(**peer, subtract_shmem=True),
                    WatchdogState()).live_memory_bytes == 0
    # unknown Shmem (telemetry gap) leaves the limit unchanged
    assert evaluate(ok_sample(memory_available=100 * G, project_memory=5 * G), cfg(**peer, subtract_shmem=True),
                    WatchdogState()).live_memory_bytes == 99 * G


def test_ram_store_excess_sheds_only_after_sustained_pressure():
    # 108 GiB cap, 60 GiB in the RAM store -> startup term 48 GiB; project 50 GiB exceeds it ONLY because of the store term.
    peer = dict(memory_reserve_bytes=6 * G, disk_reserve_bytes=10 * G, startup_memory_bytes=108 * G, startup_cpu_cores=19.97,
                disk_path="/", fraction=1.0, psi_full_avg10_shed=101.0, subtract_shmem=True, shmem_shed_grace_s=30.0,
                sample_interval_s=2.0)
    c, st = cfg(**peer), WatchdogState()
    calm = ok_sample(memory_available=40 * G, project_memory=50 * G, shmem=60 * G, psi_full_avg10=0.0)
    for _ in range(40):                                   # no pressure: never shed, admission stays stopped
        v = evaluate(calm, c, st)
        assert v.level == "stop_admission" and v.live_memory_bytes == 48 * G
    hot = dict(calm, psi_full_avg10=43.0)
    levels = [evaluate(hot, c, st).level for _ in range(15)]
    assert levels[:14] == ["stop_admission"] * 14 and levels[14] == "shed"      # 15 samples x 2 s = 30 s
    st2 = WatchdogState()
    for _ in range(10):
        evaluate(hot, c, st2)
    assert evaluate(calm, c, st2).level == "stop_admission" and st2.shmem_excess_count == 0   # a calm sample resets
    # an excess that exists WITHOUT the store term still sheds immediately (unchanged behaviour)
    over = ok_sample(memory_available=40 * G, project_memory=120 * G, shmem=60 * G)    # > 108 even without the store
    assert evaluate(over, c, WatchdogState()).level == "shed"


def test_d116_ram_store_in_project_memory_does_not_block_admission():
    """D-116: tmpfs pages written by jobs are charged to rrp.slice memory.current, so project_memory already contains the
    RAM store. With the deployed peer config (subtract_shmem=False) a large store, high MemAvailable and PSI 0 must give
    'ok' (no stop_admission); subtracting Shmem again (the D-111 follow-up design) double-counts it and blocks admission."""
    peer = dict(memory_reserve_bytes=6 * G, disk_reserve_bytes=10 * G, startup_memory_bytes=108 * G, startup_cpu_cores=19.97,
                disk_path="/", fraction=1.0, psi_full_avg10_shed=101.0)
    s = ok_sample(memory_available=60 * G, project_memory=70 * G, shmem=48 * G, psi_full_avg10=0.0)   # 48 G store inside 70 G
    v = evaluate(s, cfg(**peer, subtract_shmem=False), WatchdogState())
    assert v.level == "ok" and v.live_memory_bytes == 108 * G
    bad = evaluate(s, cfg(**peer, subtract_shmem=True), WatchdogState())
    assert bad.level == "stop_admission" and bad.live_memory_bytes == 60 * G                         # the D-116 failure mode


def test_d116_deployed_peer_watchdog_does_not_subtract_shmem():
    import importlib
    import inspect
    src = inspect.getsource(importlib.import_module("rrp.cli.main").cmd_ops_watchdog)
    assert "subtract_shmem=False" in src


# ------------------------------------------------------------ D-116 (b): project memory = anon + shmem + kernel
def _cg(anon=None, shmem=None, kernel=None, file=0, current=None):
    stat = {k: v for k, v in dict(anon=anon, shmem=shmem, kernel=kernel, file=file).items() if v is not None}
    return dict(memory_current=current, memory_stat=stat or None)


def test_project_memory_counts_nonreclaimable_only():
    from rrp.orchestration.watchdog import project_memory_fields
    # 32 G anon + 23 G RAM store (shmem) + 0.5 G kernel; 11 G reclaimable cache; memory.current 66.5 G
    f = project_memory_fields(_cg(anon=32 * G, shmem=23 * G, kernel=G // 2, file=34 * G, current=66 * G + G // 2), None)
    assert f["project_memory"] == 55 * G + G // 2 and f["project_memory_source"].startswith("memory.stat")
    assert f["project_memory_current"] == 66 * G + G // 2                      # recorded for comparison


def test_project_memory_fallback_and_gpu():
    from rrp.orchestration.watchdog import project_memory_fields
    f = project_memory_fields(_cg(current=40 * G), None)                          # memory.stat unreadable
    assert f["project_memory"] == 40 * G and "fallback" in f["project_memory_source"]
    f = project_memory_fields(_cg(anon=10 * G, shmem=2 * G, kernel=0, current=20 * G), 5 * G)   # + CUDA/unified GPU bytes
    assert f["project_memory"] == 17 * G and f["project_gpu_bytes"] == 5 * G
    f = project_memory_fields(_cg(current=40 * G), 3 * G)
    assert f["project_memory"] == 43 * G
    assert project_memory_fields(None, None)["project_memory"] is None


def test_kernel_key_fallback_and_parser():
    from rrp.orchestration.telemetry import nonreclaimable_bytes, parse_memory_stat
    s = parse_memory_stat("anon 100\nfile 50\nshmem 20\nslab 5\nkernel_stack 1\npagetables 2\nbogus x\n")
    assert s["anon"] == 100 and "bogus" not in s
    assert nonreclaimable_bytes(s) == 100 + 20 + 5 + 1 + 2                       # no `kernel` key (older kernels)
    assert nonreclaimable_bytes(dict(s, kernel=9)) == 129
    assert nonreclaimable_bytes({"anon": 1}) is None and parse_memory_stat("") is None


def test_d116_case_with_memory_stat_project_is_ok():
    """D-116 with the new project measure: a 23 G RAM store and 11 G of cache inside rrp.slice, MemAvailable 31.6 G, PSI 0 ->
    ok (no stop_admission, no shed) under the deployed peer config; cache no longer inflates the limit."""
    from rrp.orchestration.watchdog import project_memory_fields
    peer = cfg(memory_reserve_bytes=6 * G, disk_reserve_bytes=10 * G, startup_memory_bytes=108 * G, startup_cpu_cores=19.97,
               disk_path="/", fraction=1.0, psi_full_avg10_shed=101.0, subtract_shmem=False)
    pm = project_memory_fields(_cg(anon=32 * G, shmem=23 * G, kernel=G // 2, file=34 * G, current=66 * G + G // 2), None)
    s = ok_sample(memory_available=31 * G + G // 2, psi_full_avg10=0.0, shmem=23 * G, **{k: pm[k] for k in ("project_memory",)})
    v = evaluate(s, peer, WatchdogState())
    assert v.level == "ok"
    assert v.live_memory_bytes == 87 * G - 6 * G                                  # A + P_nr - R = 31.5 + 55.5 - 6
    old = evaluate(dict(s, project_memory=66 * G + G // 2), peer, WatchdogState())
    assert old.live_memory_bytes == 98 * G - 6 * G                                # memory.current counted the 11 G cache twice
