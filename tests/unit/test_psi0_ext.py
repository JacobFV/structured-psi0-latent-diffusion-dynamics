"""ops/bin/psi0_ext.sh writes the rrp_simple_compat.pth line reproducibly (round-3 psi-venv; the peer psi venv lacked it).
Shell-only: a throw-away venv, no torch, no network."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "ops/bin/psi0_ext.sh"
LINE = ('import os; os.environ.get("RRP_SIMPLE_COMPAT") == "1" and '
        '__import__("rrp.envs.simple.compat", fromlist=["x"]).install()')


def _run(ext, *args):
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", "HOME": str(ext), "RRP_PSI0_EXT": str(ext)})


@pytest.fixture
def ext(tmp_path):
    for name in ("simple", "psi"):
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(tmp_path / "venvs" / name)], check=True)
    return tmp_path


def _pth(ext, name):
    return next((ext / "venvs" / name).glob("lib/python*/site-packages")) / "rrp_simple_compat.pth"


def test_compat_pth_both_venvs_idempotent(ext):
    r = _run(ext, "compat-pth")
    assert r.returncode == 0, r.stderr
    for name in ("simple", "psi"):
        assert _pth(ext, name).read_text() == LINE + "\n"
    m = {n: _pth(ext, n).stat().st_mtime_ns for n in ("simple", "psi")}
    assert _run(ext, "compat-pth").returncode == 0      # second run: same content, file untouched
    assert {n: _pth(ext, n).stat().st_mtime_ns for n in ("simple", "psi")} == m


def test_compat_pth_single_venv_and_missing(ext):
    assert _run(ext, "compat-pth", "psi").returncode == 0
    assert _pth(ext, "psi").exists() and not _pth(ext, "simple").exists()
    r = _run(ext, "compat-pth", "nonesuch")
    assert r.returncode != 0 and "no venv" in r.stderr


def test_env_installers_call_compat_pth():
    text = SCRIPT.read_text()
    assert text.count("compat_pth ") >= 3       # definition use in simple-env, psi-env and the subcommand
    assert "> $SP/rrp_simple_compat.pth" not in text     # no second, non-idempotent writer
