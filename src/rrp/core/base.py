from __future__ import annotations

import hashlib
import json
import math
from typing import Annotated, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, PlainSerializer, BeforeValidator, WithJsonSchema


class Strict(BaseModel):
    """Unknown fields rejected; assignment validated; no silent coercion of shapes."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=False,
                              arbitrary_types_allowed=True)


def _to_array(v: Any) -> np.ndarray:
    if isinstance(v, dict) and set(v) >= {"dtype", "shape", "data"}:
        a = np.asarray(v["data"], dtype=v["dtype"]).reshape(v["shape"])
    else:
        a = np.asarray(v)
    if a.dtype == object:
        raise ValueError("object arrays are not allowed")
    if a.dtype.kind in "fc" and not np.isfinite(a).all():
        raise ValueError("array contains non-finite values")
    return a


def _ser_array(a: np.ndarray) -> dict:
    return {"dtype": str(a.dtype), "shape": list(a.shape), "data": a.reshape(-1).tolist()}


NDArray = Annotated[np.ndarray, BeforeValidator(_to_array), PlainSerializer(_ser_array, return_type=dict),
                    WithJsonSchema({"type": "object", "properties": {"dtype": {"type": "string"},
                                    "shape": {"type": "array", "items": {"type": "integer"}},
                                    "data": {"type": "array"}}, "required": ["dtype", "shape", "data"]})]


def finite(x: float, name: str = "value") -> float:
    if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x):
        raise ValueError(f"{name} must be finite")
    return float(x)


def content_hash(obj: Any) -> str:
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="json")
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
