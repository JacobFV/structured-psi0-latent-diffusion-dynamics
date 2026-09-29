"""W4 CLI: one argparse tree in rrp.cli, the same commands and flags as before, no swallowed ImportError, and the
`ops` layer still runs on a bootstrap python without the heavy dependencies."""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOP = ["doctor", "ops", "workbench", "task", "assets", "data", "train", "evaluate", "campaign", "latency", "analyze",
       "latent", "adapt"]
# commands used by scripts/ and configs (grep "-m rrp.cli" in scripts/), plus one per remaining group
USED = [
    "ops run", "ops status", "ops stop", "ops init", "ops start-watchdog", "ops shrink", "ops discover", "doctor",
    "latent train-representation", "latent train-flow", "latent fit-probes", "latent semantic-edits",
    "latent counterfactuals", "latent evaluate", "latent eval-binding", "latent disturbance", "latent arm-edits",
    "latent causal", "latent composition", "latent grpo", "latent cell", "latent latency", "latent pack-dual",
    "latent evaluate-dual", "latent teacher-ref-dual", "latent pair-index-dual",
    "campaign baseline-cell", "campaign cell", "data generate", "data pack", "train policy", "train codec", "evaluate",
    "latency", "analyze", "adapt", "workbench", "task validate", "assets validate",
]
BLOCKED = ("numpy", "torch", "mujoco", "pydantic", "fastapi", "scipy", "jsonschema")


def _run(code, *args):
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    return subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, env=env, timeout=120)


BOOTSTRAP = f"""
import sys, runpy
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {BLOCKED!r}:
            raise ModuleNotFoundError(f"blocked for the bootstrap test: {{name}}", name=name)
sys.meta_path.insert(0, Block())
sys.argv = ["rrp"] + sys.argv[1:]
runpy.run_module("rrp.cli", run_name="__main__", alter_sys=True)
"""


@pytest.mark.parametrize("argv", [["--help"], ["ops", "--help"], ["ops", "run", "--help"], ["latent", "--help"]])
def test_cli_registers_everything_without_heavy_deps(argv):
    r = _run(BOOTSTRAP, *argv)
    assert r.returncode == 0, r.stderr[-3000:]
    if argv == ["--help"]:
        for c in TOP:
            assert c in r.stdout


def test_used_commands_parse():
    from rrp.cli.main import build_parser
    for cmd in USED:
        with pytest.raises(SystemExit) as e:
            build_parser().parse_args(cmd.split() + ["--help"])
        assert e.value.code == 0, cmd


def test_no_swallowed_import_errors_in_cli():
    for p in (REPO / "src" / "rrp" / "cli").glob("*.py"):
        for n in ast.walk(ast.parse(p.read_text())):
            if isinstance(n, ast.ExceptHandler) and n.type is not None:
                names = {x.id for x in ast.walk(n.type) if isinstance(x, ast.Name)}
                assert not names & {"ImportError", "ModuleNotFoundError"}, f"{p.name}:{n.lineno}"


def test_cli_modules_are_stdlib_only_at_import():
    """Module-level imports of the command modules: stdlib or rrp.cli only (heavy imports go inside commands)."""
    for p in (REPO / "src" / "rrp" / "cli").glob("*.py"):
        for n in ast.parse(p.read_text()).body:
            if isinstance(n, ast.ImportFrom) and n.module:
                assert n.module.split(".")[0] not in BLOCKED and (not n.module.startswith("rrp")
                                                                 or n.module.startswith("rrp.cli")), f"{p.name}: {n.module}"
            elif isinstance(n, ast.Import):
                assert all(a.name.split(".")[0] not in BLOCKED and not a.name.startswith("rrp") for a in n.names), p.name
