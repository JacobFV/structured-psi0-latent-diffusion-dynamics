"""read_cgroup reports the cgroup v2 memory high-water mark (used to declare realistic lease memory)."""
from rrp.ops.telemetry import read_cgroup


def test_memory_peak_read(tmp_path):
    (tmp_path / "memory.current").write_text("1000\n")
    (tmp_path / "memory.peak").write_text("5600000000\n")
    (tmp_path / "cpu.stat").write_text("usage_usec 42\n")
    info = read_cgroup(tmp_path)
    assert info["memory_peak"] == 5600000000 and info["memory_current"] == 1000 and info["cpu_usage_usec"] == 42


def test_memory_peak_missing(tmp_path):
    (tmp_path / "memory.current").write_text("1\n")
    assert read_cgroup(tmp_path)["memory_peak"] is None
