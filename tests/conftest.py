"""Shared pytest hooks: skip tests whose external inputs are absent in a fresh checkout.

Markers (registered in pyproject.toml):
- ``menagerie``: needs the MuJoCo Menagerie assets under ``.cache/assets/mujoco_menagerie``
  (fetch with ``scripts/fetch_menagerie.sh``; the main checkout symlinks them).
- ``packed_data``: needs the packed training data ``artifacts/packed/latent_pp_v3dart_s1_H16`` (not in git).
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MENAGERIE = REPO / ".cache" / "assets" / "mujoco_menagerie"
PACKED = REPO / "artifacts" / "packed" / "latent_pp_v3dart_s1_H16"

_REQUIREMENTS = {
    "menagerie": (MENAGERIE, "Menagerie assets not fetched (scripts/fetch_menagerie.sh)"),
    "packed_data": (PACKED, "packed data not present (artifacts/packed is not in git)"),
}


def pytest_collection_modifyitems(config, items):
    for marker, (path, reason) in _REQUIREMENTS.items():
        if path.exists():
            continue
        skip = pytest.mark.skip(reason=f"{reason}: {path.relative_to(REPO)}")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)
