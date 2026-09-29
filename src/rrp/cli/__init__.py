"""`rrp` command-line interface (one argparse tree; W4 consolidated the former rrp/cli*.py chain here).

`python -m rrp.cli ...` and the `rrp` console script (`rrp.cli:main`) are unchanged. The command modules import only the
standard library at module level (the `ops` commands must run on a bootstrap python without the project venv); every
heavy dependency is imported inside a command function. Import errors are never swallowed.
"""
from rrp.cli.main import main  # noqa: F401  (console-script entry point `rrp.cli:main`)
