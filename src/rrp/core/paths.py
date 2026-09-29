"""Where rrp finds its repo-level data (artifacts/, configs/, tasks/, .cache/assets) (W11, docs/architecture.md).

Before W11 every module used `Path(__file__).resolve().parents[3]`, i.e. the rrp source checkout. That is still the
default when rrp runs from a checkout (src layout: `<checkout>/src/rrp/...`), so nothing changes there. When rrp is
INSTALLED as a package (e.g. psi1z depends on it by git sha), `parents[3]` is a directory inside the venv; then:

- `RRP_HOME` (env) names the data root explicitly (e.g. the rrp checkout, or the consumer's own project dir);
- otherwise `rrp_home()` falls back to the current working directory (the consumer's project root).

`is_checkout()` says whether this rrp is a source checkout. Package data needed at runtime (task specs) is shipped in
the wheel under `rrp/_data/` and used when the data root does not have its own copy (`data_path`).
Stdlib only (contracts layer).
"""
from __future__ import annotations

import os
from pathlib import Path

_CHECKOUT = Path(__file__).resolve().parents[3]          # <checkout> when running from src/rrp/core/paths.py
_PACKAGE_DATA = Path(__file__).resolve().parents[1] / "_data"


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


def package_data(*parts: str) -> Path:
    """A file shipped inside the rrp package (wheel), e.g. package_data("tasks", "pick_place.json")."""
    return _PACKAGE_DATA.joinpath(*parts)


def data_path(*parts: str) -> Path:
    """rrp_home()/<parts> if it exists, else the packaged copy (rrp/_data/<parts>), else rrp_home()/<parts>."""
    p = rrp_home().joinpath(*parts)
    if p.exists():
        return p
    q = package_data(*parts)
    return q if q.exists() else p
