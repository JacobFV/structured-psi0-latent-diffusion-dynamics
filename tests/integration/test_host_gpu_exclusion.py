"""Host GPU is off: leased host jobs cannot open NVIDIA device nodes and render in software."""
import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.cgroup
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run(code):
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"))
    return subprocess.run([sys.executable, "-m", "rrp.cli", "ops", "run", "--cpu", "0.5", "--mem", "512M",
                           "--label", "gpu_exclusion_test", "--max-seconds", "120", "--", sys.executable, "-c", code],
                          capture_output=True, text=True, env=env, timeout=180, cwd=REPO)


@pytest.mark.skipif(os.environ.get("RRP_NODE", "host") != "host", reason="host-only policy")
def test_leased_host_job_cannot_see_nvidia_devices():
    r = _run("import os,glob; print('DEVICES', glob.glob('/dev/nvidia*') + glob.glob('/dev/dri/*'))")
    assert "DEVICES []" in r.stdout, r.stdout + r.stderr


@pytest.mark.skip(reason="host GPU authorized by the user (D-027); non-GPU leases still hide devices")
def test_host_gpu_lease_is_refused():
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"))
    r = subprocess.run([sys.executable, "-m", "rrp.cli", "ops", "run", "--gpu", "--gpu-mem", "1G", "--cpu", "0.5",
                        "--mem", "256M", "--label", "x", "--", "true"], capture_output=True, text=True, env=env,
                       timeout=60, cwd=REPO)
    assert r.returncode != 0
    assert "capacity" in (r.stderr + r.stdout).lower() or "disabled" in (r.stderr + r.stdout).lower()
