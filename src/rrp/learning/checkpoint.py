"""Atomic checkpoints with version provenance; incompatible versions are never silently loaded."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def require_compatible_versions(saved: dict, requested: dict):
    for k, v in requested.items():
        if k in saved and saved[k] != v:
            raise ValueError(f"checkpoint {k} mismatch: saved {saved[k]!r} != requested {v!r}")


def save_checkpoint(path: Path, *, model, optimizer=None, step: int, versions: dict, config: dict,
                    data_cursor: dict | None = None, extra: dict | None = None) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict() if optimizer else None, step=step,
                 versions=versions, config=config, data_cursor=data_cursor or {},
                 rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                          cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
                 extra=extra or {})
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()[:16]
    os.replace(tmp, path)            # previous known-good file replaced only after full write
    meta = dict(path=str(path), sha256_16=digest, step=step, versions=versions)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


def load_checkpoint(path: Path, *, requested_versions: dict | None = None, map_location="cpu") -> dict:
    state = torch.load(path, map_location=map_location, weights_only=False)
    if requested_versions:
        require_compatible_versions(state["versions"], requested_versions)
    return state
