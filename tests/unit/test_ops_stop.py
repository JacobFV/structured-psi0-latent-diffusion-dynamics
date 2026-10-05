"""Bug fix, found 2026-10-05: `rrp ops stop --owned-only --lease ID` on one node, for a lease that
actually lives on the OTHER node, used to call the local systemd backend directly for any id and report
success regardless. `systemctl --user stop` on a unit that was never loaded on this node's manager is a
silent no-op (exit 0, nothing stopped) -- so stopping a peer lease from the host reported success while
the peer job kept running. No remote (ssh) stopping is added here; the fix is local-only: refuse loudly
when the lease/unit isn't known on this node, and verify the unit is actually inactive before reporting
success when it is.

These are fakes only -- no real systemd units, brokers or config files are touched.
"""
from __future__ import annotations

import pytest

from rrp.ops.runtime import stop_lease, LeaseNotLocal


class FakeBroker:
    """Stand-in for ResourceBroker: only the `.leases()` read `stop_lease` uses."""

    def __init__(self, lease_ids=()):
        self._ids = set(lease_ids)

    def leases(self):
        return {lid: {} for lid in self._ids}


class FakeBackend:
    """Stand-in for SystemdUserBackend. `units` maps unit name -> ActiveState ('inactive' means not
    loaded / LoadState not-found, matching what `systemctl --user show` returns for a unit this node's
    manager has never heard of)."""

    def __init__(self, units: dict[str, str] | None = None, stop_works: bool = True):
        self.units = dict(units or {})
        self.stop_works = stop_works
        self.remove_lease_calls: list[str] = []

    def unit_state(self, unit: str) -> dict:
        state = self.units.get(unit)
        if state is None:
            return {"LoadState": "not-found", "ActiveState": "inactive"}
        return {"LoadState": "loaded", "ActiveState": state}

    def remove_lease(self, lease_id: str) -> None:
        self.remove_lease_calls.append(lease_id)
        if self.stop_works:
            from rrp.ops.cgroup import job_unit, lease_slice
            self.units[job_unit(lease_id)] = "inactive"
            self.units[lease_slice(lease_id)] = "inactive"


def test_stop_refuses_a_lease_this_node_does_not_know_and_never_touches_the_backend():
    """The reported bug: a lease acquired on the OTHER node. This node's broker never saw it and its
    systemd --user manager has never loaded its unit. The fix must raise, not call remove_lease (which
    would silently no-op and look like success), and must not claim the lease was stopped."""
    broker = FakeBroker(lease_ids=[])             # not in this node's broker state
    backend = FakeBackend(units={})                # unit never loaded here either
    with pytest.raises(LeaseNotLocal, match="remote123"):
        stop_lease("remote123", broker=broker, backend=backend)
    assert backend.remove_lease_calls == []         # never blindly called the backend for an unknown lease


def test_stop_succeeds_and_verifies_inactive_for_a_lease_known_here():
    from rrp.ops.cgroup import job_unit
    broker = FakeBroker(lease_ids=["local1"])
    backend = FakeBackend(units={job_unit("local1"): "active"}, stop_works=True)
    result = stop_lease("local1", broker=broker, backend=backend)
    assert result == {"lease_id": "local1", "unit": job_unit("local1"), "active_state": "inactive"}
    assert backend.remove_lease_calls == ["local1"]


def test_stop_raises_if_the_unit_is_still_active_after_stopping():
    """A stop that didn't actually work (e.g. the unit refuses SIGTERM) must never be reported as success."""
    from rrp.ops.cgroup import job_unit
    broker = FakeBroker(lease_ids=["stuck1"])
    backend = FakeBackend(units={job_unit("stuck1"): "active"}, stop_works=False)
    with pytest.raises(RuntimeError, match="still"):
        stop_lease("stuck1", broker=broker, backend=backend)


def test_cli_lease_stop_exits_nonzero_and_never_prints_a_success_line_for_an_unknown_lease(monkeypatch, capsys):
    """CLI-level regression for the exact bug report: old code printed {"stopped_leases": [...]} and
    returned 0 for a lease it never actually touched. The fixed CLI must exit non-zero and must not print
    a stdout success line that claims the lease was stopped."""
    import argparse
    from rrp.cli.main import cmd_ops_stop

    def fake_stop_lease(lease_id, **kw):
        raise LeaseNotLocal(f"lease {lease_id!r} is not known to this node")

    monkeypatch.setattr("rrp.ops.runtime.stop_lease", fake_stop_lease)
    a = argparse.Namespace(owned_only=True, lease=["peer_lease_1"], all_project_jobs=False)
    with pytest.raises(SystemExit) as e:
        cmd_ops_stop(a)
    assert e.value.code != 0
    out = capsys.readouterr().out
    assert "stopped_leases" not in out      # no success line on stdout for an unverified stop
