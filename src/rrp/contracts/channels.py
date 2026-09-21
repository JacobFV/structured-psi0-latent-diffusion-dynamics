"""Public deployment transport vs privileged bus.

The public transport serializes only public contract types and refuses anything that
is, contains, or references privileged truth. Deserialization uses strict models, so a
hostile payload smuggling `object_poses` etc. into a PolicyObservation is rejected.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .errors import PrivilegedLeakError
from .observation import PolicyObservation, PrivilegedTruth

PRIVILEGED_KEYS = frozenset({"object_poses", "object_entity_map", "event_completion_truth", "held_by",
                             "predicates", "reward", "labels", "privileged_truth", "sim_object"})
PUBLIC_KINDS = frozenset({"policy_observation", "action_chunk"})


def _scan(obj: Any, path: str = "$"):
    if isinstance(obj, dict):
        if obj.get("kind") == "privileged_truth":
            raise PrivilegedLeakError(f"privileged payload at {path}", path=path)
        for k, v in obj.items():
            if k in PRIVILEGED_KEYS:
                raise PrivilegedLeakError(f"privileged key {k!r} at {path}", path=path, key=k)
            _scan(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _scan(v, f"{path}[{i}]")


def serialize_public(obj: BaseModel) -> bytes:
    if isinstance(obj, PrivilegedTruth):
        raise PrivilegedLeakError("PrivilegedTruth cannot use the public transport")
    if not isinstance(obj, BaseModel):
        raise PrivilegedLeakError("public transport only accepts typed public contracts")
    d = obj.model_dump(mode="json")
    _scan(d)
    return json.dumps(d, separators=(",", ":")).encode()


def deserialize_observation(data: bytes | str) -> PolicyObservation:
    d = json.loads(data)
    _scan(d)
    return PolicyObservation.model_validate(d)


class PrivateBus:
    """Append-only privileged record stream, stored separately from public episode files."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if "public" in self.path.name:
            raise PrivilegedLeakError("private bus path must not be a public file")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, truth: PrivilegedTruth):
        with open(self.path, "a") as f:
            f.write(truth.model_dump_json() + "\n")
