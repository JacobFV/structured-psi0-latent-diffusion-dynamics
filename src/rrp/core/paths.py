"""Where rrp finds its repo-level data (artifacts/, recipes/, .cache/assets) (W11, docs/architecture.md).

Before W11 every module used `Path(__file__).resolve().parents[3]`, i.e. the rrp source checkout. That is still the
default when rrp runs from a checkout (src layout: `<checkout>/src/rrp/...`), so nothing changes there. When rrp is
INSTALLED as a package (e.g. psi1z depends on it by git sha), `parents[3]` is a directory inside the venv; then:

- `RRP_HOME` (env) names the data root explicitly (e.g. the rrp checkout, or the consumer's own project dir);
- otherwise `rrp_home()` falls back to the current working directory (the consumer's project root).

`is_checkout()` says whether this rrp is a source checkout. Package data needed at runtime (the task graphs) lives
inside the package itself, `src/rrp/tasks/graphs/*.json` (D-145), so it needs no data root.
Stdlib only (contracts layer).
"""
from __future__ import annotations

import os
from pathlib import Path

_CHECKOUT = Path(__file__).resolve().parents[3]          # <checkout> when running from src/rrp/core/paths.py


def is_checkout(root: Path | None = None) -> bool:
    """True when rrp is imported from a source checkout (src layout), not from an installed wheel."""
    r = root or _CHECKOUT
    return (r / "src" / "rrp" / "__init__.py").is_file() and (r / "pyproject.toml").is_file()


def rrp_home() -> Path:
    """Data root: $RRP_HOME, else the source checkout this rrp runs from, else the current working directory."""
    env = os.environ.get("RRP_HOME")
    if env:
        return Path(env).expanduser()
    if is_checkout():
        return _CHECKOUT
    return Path.cwd()
