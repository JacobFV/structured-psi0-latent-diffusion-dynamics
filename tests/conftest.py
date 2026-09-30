"""Shared pytest hooks: skip tests whose external inputs are absent in a fresh checkout (D-146 round 2, G0).

A test that needs untracked weights, Menagerie assets, warp or the peer SKIPS with the missing thing named; it never fails.
The two helpers below are the one way to do it from inside a test: ``need_weights(path)`` and ``need_assets()``.
Tests slower than 20 s carry ``@pytest.mark.slow``; the merge gate runs ``-m "not slow"``, the integration check runs everything.

Markers (registered in pyproject.toml):
- ``slow``: over 20 s on the host (excluded from the merge gate).
- ``menagerie``: needs the MuJoCo Menagerie assets under ``.cache/assets/mujoco_menagerie``
  (fetch with ``ops/bin/fetch_menagerie.sh``; the main checkout symlinks them).
- ``packed_data``: needs the packed training data ``artifacts/packed/latent_pp_v3dart_s1_H16`` (not in git).
- ``computerworld``: needs the optional extra ``computerworld==0.2.0`` (research/tracks/pointer.md has install steps).
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

_ASSETS_REASON = f"Menagerie assets not fetched (ops/bin/fetch_menagerie.sh): {MENAGERIE}"
_REQUIREMENTS = {
    "menagerie": (MENAGERIE, _ASSETS_REASON),
    "packed_data": (PACKED, f"packed data not present (artifacts/packed is not in git): {PACKED}"),
}


def need_weights(path) -> Path:
    """Skip the calling test unless the untracked weights file / directory `path` exists; the reason names it."""
    p = Path(path)
    if not p.exists():
        pytest.skip(f"needs untracked weights {p} (peer store; not in git)")
    return p


def need_assets() -> Path:
    """Skip the calling test unless the Menagerie assets are fetched (same check as the ``menagerie`` marker)."""
    if not MENAGERIE.exists():
        pytest.skip(_ASSETS_REASON)
    return MENAGERIE


def pytest_collection_modifyitems(config, items):
    import importlib.util
    if importlib.util.find_spec("computerworld") is None:
        skip = pytest.mark.skip(reason="optional extra `computerworld` not installed")
        for item in items:
            if "computerworld" in item.keywords:
                item.add_marker(skip)
    for marker, (path, reason) in _REQUIREMENTS.items():
        if path.exists():
            continue
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)
