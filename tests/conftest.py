"""Shared pytest hooks: skip tests whose external inputs are absent in a fresh checkout.

Markers (registered in pyproject.toml):
- ``menagerie``: needs the MuJoCo Menagerie assets under ``.cache/assets/mujoco_menagerie``
  (fetch with ``scripts/fetch_menagerie.sh``; the main checkout symlinks them).
- ``packed_data``: needs the packed training data ``artifacts/packed/latent_pp_v3dart_s1_H16`` (not in git).
"""


from __future__ import annotations

import os as _os

# D-115 / D-125: the host has no GPU slots and its GPU is often full (other projects). Tests that touch CUDA (e.g. torch
# Adam's device check) then fail with CUDA OOM. Hide CUDA for the unit suite unless explicitly requested.
if _os.environ.get("RRP_TEST_GPU") != "1":
    _os.environ["CUDA_VISIBLE_DEVICES"] = ""

import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
# rrp.bodies.importers reads the assets from rrp_home() ($RRP_HOME, else this checkout): skip on the same path
MENAGERIE = Path(os.environ.get("RRP_HOME", REPO)).expanduser() / ".cache" / "assets" / "mujoco_menagerie"
PACKED = REPO / "artifacts" / "packed" / "latent_pp_v3dart_s1_H16"

_REQUIREMENTS = {
    "menagerie": (MENAGERIE, "Menagerie assets not fetched (scripts/fetch_menagerie.sh)"),
    "packed_data": (PACKED, "packed data not present (artifacts/packed is not in git)"),
}


def pytest_collection_modifyitems(config, items):
    for marker, (path, reason) in _REQUIREMENTS.items():
        if path.exists():
            continue
        skip = pytest.mark.skip(reason=f"{reason}: {path}")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)
