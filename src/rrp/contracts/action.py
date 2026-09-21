from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from .base import Strict, NDArray

Source = Literal["teacher", "scripted_teacher", "learned", "user", "debug", "random", "mock", "privileged_teacher"]


class GroupCommand(Strict):
    group: str
    values: NDArray          # [horizon, width] in the group's declared units
    mask: NDArray            # [horizon, width] bool


class ActionChunk(Strict):
    """A versioned chunk of native commands. The controller rejects stale/mismatched chunks."""
    observation_id: str
    graph_version: int
    runtime_version: int
    robot_spec_hash: str
    controller_version: str
    policy_version: str
    codec_version: str | None
    start_time: float
    dt: float = Field(gt=0)
    horizon: int = Field(ge=1)
    command_groups: list[GroupCommand]
    sampling_seed: int | None
    source: Source
    expires_at: float | None = None

    @model_validator(mode="after")
    def _shapes(self):
        for g in self.command_groups:
            if g.values.ndim != 2 or g.values.shape[0] != self.horizon:
                raise ValueError(f"group {g.group}: values must be [horizon={self.horizon}, width]")
            if g.mask.shape != g.values.shape:
                raise ValueError(f"group {g.group}: mask shape mismatch")
            if not np.isfinite(g.values).all():
                raise ValueError(f"group {g.group}: non-finite command")
        return self

    def group(self, name: str) -> GroupCommand:
        for g in self.command_groups:
            if g.group == name:
                return g
        raise KeyError(name)


class NativeCommand(Strict):
    """Single control-step command sent to the simulator controller."""
    controller_version: str
    groups: dict[str, list[float]]
    source: Source
    chunk_ref: str | None = None
