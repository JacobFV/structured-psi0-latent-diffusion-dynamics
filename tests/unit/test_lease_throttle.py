"""D-117: per-lease memory.high throttling — watchdog sample deltas/warnings, the job exit record, and the `ops run`
declaration warning. Fake cgroup files only."""
from __future__ import annotations

import json
from types import SimpleNamespace

from rrp.ops import telemetry
from rrp.ops.watchdog import WatchdogState, lease_high_fields, throttle_warnings

EV = "low 0\nhigh {h}\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n"


def _lease(root, lid, high):
    d = root / f"rrp-lease-{lid}.slice"
    d.mkdir(parents=True, exist_ok=True)
    (d / "memory.events").write_text(EV.format(h=high))


def test_parse_and_scan_fake_lease_cgroups(tmp_path):
    assert telemetry.parse_memory_events(EV.format(h=7)) == dict(low=0, high=7, max=0, oom=0, oom_kill=0, oom_group_kill=0)
    assert telemetry.parse_memory_events("") is None
    _lease(tmp_path, "111_aa", 0)
    _lease(tmp_path, "222_bb", 5)
    (tmp_path / "rrp-lease-333_cc.slice").mkdir()                 # no memory.events (unit gone): skipped
    ev = telemetry.lease_memory_events(tmp_path)
    assert set(ev) == {"111_aa", "222_bb"} and ev["222_bb"]["high"] == 5
    assert telemetry.lease_memory_events(tmp_path / "missing") == {}


def test_sample_deltas_and_warnings(tmp_path):
    st = WatchdogState()
    _lease(tmp_path, "111_aa", 0)
    _lease(tmp_path, "222_bb", 5)
    f = lease_high_fields(st, telemetry.lease_memory_events(tmp_path))
    assert f["throttled_leases"] == ["222_bb"] and f["lease_memory_high"] == {"222_bb": dict(high=5, delta=5)}
    f = lease_high_fields(st, telemetry.lease_memory_events(tmp_path))          # no new events: not throttled now
    assert f["throttled_leases"] == [] and f["lease_memory_high"]["222_bb"]["delta"] == 0
    _lease(tmp_path, "222_bb", 9)
    _lease(tmp_path, "111_aa", 2)
    f = lease_high_fields(st, telemetry.lease_memory_events(tmp_path))
    assert f["throttled_leases"] == ["111_aa", "222_bb"] and f["lease_memory_high"]["222_bb"]["delta"] == 4
    leases = {"222_bb": {"request": {"label": "armv2_pack", "memory_bytes": 10 * 2 ** 30}}}
    w = throttle_warnings(f, leases)
    assert len(w) == 2 and "armv2_pack" in w[1] and "+4 events" in w[1] and "memory.high 8.0G" in w[1]
    (tmp_path / "rrp-lease-222_bb.slice" / "memory.events").unlink()            # lease ended
    f = lease_high_fields(st, telemetry.lease_memory_events(tmp_path))
    assert "222_bb" not in st.lease_high and f["throttled_leases"] == []


def test_exit_record_carries_high_events(tmp_path, monkeypatch, capsys):
    import rrp.ops.child as child
    lease_dir = tmp_path / "rrp-lease-L1.slice"
    lease_dir.mkdir()
    (lease_dir / "memory.events").write_text(EV.format(h=42))
    (lease_dir / "memory.peak").write_text(str(9 * 2 ** 30))
    (tmp_path / "ops").mkdir()

    class Br:
        def leases(self):
            return {"L1": {"request": {"label": "w7_pack", "memory_bytes": 10 * 2 ** 30, "cpu_cores": 2}}}

        def release(self, *a, **k):
            pass
    monkeypatch.setattr(child, "make_broker", lambda require_watchdog=False: (Br(), None))
    monkeypatch.setattr(child, "ops_root", lambda: tmp_path)
    monkeypatch.setattr(child.telemetry, "slice_cgroup_path", lambda name: tmp_path / name)
    monkeypatch.setattr(child, "run_leased_child", lambda br, lid, cmd: SimpleNamespace(returncode=0, stopped_by=None,
                                                                                          heartbeats=3))
    log = tmp_path / "job.log"
    monkeypatch.setattr(child.os, "dup2", lambda a, b: None)                   # keep pytest's stdout
    assert child.main(["--lease", "L1", "--log", str(log), "--", "true"]) == 0
    rec = json.loads((tmp_path / "ops" / "resource-ledger.jsonl").read_text().splitlines()[-1])
    assert rec["memory_high_events"] == 42 and rec["memory_peak_bytes"] == 9 * 2 ** 30 and rec["oom_kill"] == 0
    assert "memory_high_events=42 (THROTTLED" in capsys.readouterr().out


def test_ops_run_declaration_warning(tmp_path):
    from rrp.ops.runtime import mem_declaration_warning
    led = tmp_path / "ledger.jsonl"
    G = 2 ** 30
    led.write_text("\n".join(json.dumps(r) for r in [dict(label="w7_pack", memory_peak_bytes=8 * G),
                                                     dict(label="w7_pack", memory_peak_bytes=9 * G),
                                                     dict(label="other", memory_peak_bytes=50 * G)]) + "\nnot json\n")
    w = mem_declaration_warning("w7_pack", int(10.8 * G), led)                   # 1.2 x the 9 G peak (D-117 case)
    assert w and "9.00G" in w and "12.2G" in w
    assert mem_declaration_warning("w7_pack", int(12.5 * G), led) is None        # >= 1.35 x peak
    assert mem_declaration_warning("new_label", 1 * G, led) is None             # no history
    assert mem_declaration_warning("w7_pack", 1 * G, tmp_path / "missing.jsonl") is None
