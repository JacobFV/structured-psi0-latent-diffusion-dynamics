"""Atomic checkpoints with version provenance; incompatible versions are never silently loaded."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from rrp.core.provenance import file_digest


def require_compatible_versions(saved: dict, requested: dict):
    for k, v in requested.items():
        if k in saved and saved[k] != v:
            raise ValueError(f"checkpoint {k} mismatch: saved {saved[k]!r} != requested {v!r}")


def save_checkpoint(path: Path, *, model, optimizer=None, step: int, versions: dict, config: dict,
                    data_cursor: dict | None = None, extra: dict | None = None, source: str | None = None) -> dict:
    """`provenance` (W3) is added to the state and the sidecar: weights fingerprint of the model state_dict,
    git sha, versions and the meaning-changing training flags (zero_prev_action, realizer_drop_qd, ...)."""
    from rrp.core.provenance import make_provenance, training_flags
    from rrp.policies.relations.base import provenance as factor_provenance, stamp_versions
    specs = model.factor_specs() if callable(getattr(model, "factor_specs", None)) else None
    factors = factor_provenance(specs) if specs is not None else None   # relation factors (D-144): hash + provenance
    from rrp.policies.features import kinfeat
    versions = stamp_versions(versions, specs, [kinfeat.VERSION] if kinfeat.resolved() else [])  # versions["factors"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sd = model.state_dict()
    prov = make_provenance(source or f"learned:{path.parent.name}/{path.name}", weights=dict(model=sd),
                           versions={k: v for k, v in (versions or {}).items() if v is not None},
                           featurizer_version=(versions or {}).get("featurizer"), flags=training_flags(config))
    state = dict(model=sd, optimizer=optimizer.state_dict() if optimizer else None, step=step,
                 versions=versions, config=config, data_cursor=data_cursor or {},
                 rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                          cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
                 extra=extra or {}, provenance=prov.to_dict(), factors=factors)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    digest = file_digest(tmp)
    os.replace(tmp, path)            # previous known-good file replaced only after full write
    meta = dict(path=str(path), sha256_16=digest, step=step, versions=versions, provenance=prov.to_dict())
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


def load_checkpoint(path: Path, *, requested_versions: dict | None = None, map_location="cpu", specs=None,
                    allow_factor_mismatch: bool = False) -> dict:
    """`specs` (the resolved factor list the caller is about to build) must match the saved structure hash
    (`relations.base.require_factors`); `allow_factor_mismatch` is for a deliberate cross-structure load."""
    state = torch.load(path, map_location=map_location, weights_only=False)
    check_kinfeat(state, path)
    if requested_versions:
        require_compatible_versions(state["versions"], requested_versions)
    if specs is not None:
        from rrp.policies.relations.base import require_factors
        require_factors(state.get("versions"), specs, allow_factor_mismatch)
    return state


def check_kinfeat(state: dict, path=None) -> None:
    """A model trained with `feat.base_axes` features must run with them, and vice versa (never silently mixed).
    Current checkpoints fold this into the combined `versions["factors"]` string (see `save_checkpoint`); old
    checkpoints (pre-R12c) stored a standalone `versions["kinfeat"]` key -- read here too, an on-disk legacy remap
    only (D-144 addendum decision b), never a live env-var read."""
    from rrp.policies.features import kinfeat
    if not isinstance(state, dict) or "model" not in state:
        return
    v = state.get("versions") or {}
    saved = v.get("kinfeat")                      # legacy standalone key
    if saved is None and isinstance(v.get("factors"), str):
        saved = kinfeat.VERSION if kinfeat.VERSION in v["factors"].split("+") else None
    now = kinfeat.VERSION if kinfeat.resolved() else None
    if saved != now:
        raise ValueError(f"checkpoint {path}: kinfeat {saved!r} != current (ambient feat.base_axes) -> {now!r}")


def checkpoint_provenance(state: dict, path=None):
    """Provenance of a loaded checkpoint; pre-W3 checkpoints -> legacy=True ('unfingerprinted'), with the
    flags that their stored config does state."""
    from rrp.core.provenance import Provenance, legacy_provenance, training_flags, UNFINGERPRINTED
    if isinstance(state.get("provenance"), dict):
        return Provenance.from_json(state["provenance"])
    src = f"learned:{Path(path).parent.name}/{Path(path).name}" if path else "learned:unknown"
    v = state.get("versions") or {}
    return legacy_provenance(src, flags=training_flags(state.get("config")), notes=UNFINGERPRINTED,
                             versions={k: str(x) for k, x in v.items() if x is not None})
