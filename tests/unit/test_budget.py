import math
import pytest
from rrp.ops.budget import host_budget, compute_budget, BudgetError, live_memory_limit_bytes, live_cpu_limit, RolePolicy


def test_half_free_not_half_total():
    b = host_budget(total_ram_gib=128, available_ram_gib=40,
                    free_cpu_cores=6, free_disk_gib=200,
                    total_disk_gib=1000)
    assert b.memory_gib <= 20
    assert b.cpu_cores <= 3
    assert b.new_disk_gib <= 100
    assert not b.host_gpu_enabled


def test_fractional_cpu_not_rounded_up():
    b = host_budget(total_ram_gib=128, available_ram_gib=40, free_cpu_cores=0.8,
                    free_disk_gib=200, total_disk_gib=1000)
    assert b.cpu_cores == pytest.approx(0.4)


def test_reserve_dominates_when_little_is_free():
    # 14 GiB free, reserve 12.8 GiB -> only 1.2 GiB usable, not 7
    b = host_budget(total_ram_gib=128, available_ram_gib=14, free_cpu_cores=4,
                    free_disk_gib=50, total_disk_gib=1000)
    assert b.memory_gib == pytest.approx(14 - 12.8)
    assert b.new_disk_gib == 0.0  # disk reserve 100 GiB > free 50 GiB


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), "3", True, None])
def test_invalid_capacity_rejected(bad):
    with pytest.raises(BudgetError):
        host_budget(total_ram_gib=128, available_ram_gib=bad, free_cpu_cores=1,
                    free_disk_gib=10, total_disk_gib=100)


def test_zero_totals_and_inconsistent_values_rejected():
    with pytest.raises(BudgetError):
        host_budget(total_ram_gib=0, available_ram_gib=0, free_cpu_cores=1, free_disk_gib=1, total_disk_gib=1)
    with pytest.raises(BudgetError):
        host_budget(total_ram_gib=10, available_ram_gib=11, free_cpu_cores=1, free_disk_gib=1, total_disk_gib=1)


def test_host_fraction_cannot_exceed_half():
    with pytest.raises(BudgetError):
        compute_budget(role="host", total_ram_gib=128, available_ram_gib=100, free_cpu_cores=10,
                       free_disk_gib=100, total_disk_gib=1000, policy=RolePolicy(0.6, 0.5, 0.5))


def test_peer_policy_allows_eighty_percent_after_reserve():
    b = compute_budget(role="peer", total_ram_gib=120, available_ram_gib=100, free_cpu_cores=20,
                       free_disk_gib=500, total_disk_gib=1000)
    assert b.memory_gib == pytest.approx(80)
    assert b.cpu_cores == pytest.approx(16)


def test_live_limit_does_not_repeatedly_halve_project_reduced_memory():
    G = 1024 ** 3
    startup = 20 * G
    # project uses 10 GiB; available now 30 GiB => free without project 40 GiB => 20 GiB
    lim = live_memory_limit_bytes(startup_limit=startup, available_now=30 * G, project_resident=10 * G,
                                  reserve=12 * G)
    assert lim == 20 * G
    # unattributed project residency uses the conservative smaller value
    lim2 = live_memory_limit_bytes(startup_limit=startup, available_now=30 * G, project_resident=10 * G,
                                   reserve=12 * G, project_attribution_accurate=False)
    assert lim2 == 15 * G
    # external load rises: available drops to 14 GiB with project 10 GiB -> min(12, 24-12=12)
    lim3 = live_memory_limit_bytes(startup_limit=startup, available_now=14 * G, project_resident=10 * G,
                                   reserve=12 * G)
    assert lim3 == 12 * G


def test_live_cpu_limit_tracks_external_load():
    assert live_cpu_limit(startup_limit=6, idle_now=12, project_usage=0) == 6
    assert live_cpu_limit(startup_limit=6, idle_now=2, project_usage=2) == 2
