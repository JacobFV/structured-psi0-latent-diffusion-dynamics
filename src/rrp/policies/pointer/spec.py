"""Pointer-body constants, relation-factor config and the geometry of one env (torch-free; docs/architecture.md 3, 5).

Knots, tick counts and pixel scales are declared here once. Anything that depends on the env (screen size, pixel
pitch, control rate) is read from the env spec through `PointerGeometry.from_spec`, never typed in a consumer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rrp.policies.nets.pointer_vocab import (KNOT_TIMES, LC, LI, N_BOUND, N_KEYCLS, N_ROLE, N_SYM, NH, NW,  # noqa: F401
                                             POINTER_FACTORS_PRESET, UI_CARRIES, WF)
from rrp.policies.teachers.computerworld import MAX_STEP_PX      # the scripted teachers' speed (px per tick)

VALIDITY_S = 0.8
ENG_VERSION = "cw_pointer_eng.v1"
SLOT_FIELDS = ("x", "y", "depth", "button", "key", "wheel", "flag")
SLOT_W = len(SLOT_FIELDS)
ENG_DIM = 2 * SLOT_W
POINTER_KINDS = frozenset({"cartesian_position", "button", "discrete"})
PHASES = ("idle", "move", "press", "release", "drag", "type")
ROLE_IDS = {"null": 0, "button": 1, "textbox": 2, "label": 3, "form": 4, "heading": 5, "text": 6, "link": 7,
            "checkbox": 8}


@dataclass
class PolicyConfig:
    """Pointer-net relation factors (docs/relations.md; D-144 R20 follow-up). `factors=None` (the default) resolves
    to `POINTER_FACTORS_PRESET` ("none"): unchanged nets, byte-identical checkpoints. `factors=["preset:ui"]` turns
    on `ui.label_for` / `ui.same_window` / `ui.focus_next` / `ui.above` / `ui.drag_to` (`catalog.py` §ui) at `UICtx`'s
    widget self-attention; add `geo.*` names (or `"geo.*"` itself) to also enable the PaPE / bilinear geometry
    factors over the same tokens' `pos3d` / `cam_uvd` fields."""
    factors: list | None = None

    def specs(self):
        from rrp.policies.relations.base import resolve
        return resolve(self.factors, default=POINTER_FACTORS_PRESET)


@dataclass(frozen=True)
class PointerGeometry:
    """Screen size (px), pixel pitch (m) and control rate (Hz) of one env, from its `EnvSpec` (`from_spec`). Every
    normalization, step unit and tick phase of the pointer stack derives from this."""
    screen_px: tuple[int, int]
    m_per_px: float
    control_hz: float

    @classmethod
    def from_spec(cls, spec) -> "PointerGeometry":
        w, h = spec.frame["screen_px"]
        return cls((int(w), int(h)), float(spec.frame["m_per_px"]), float(spec.control_hz))

    @property
    def half(self) -> np.ndarray:
        """Half extents (m) of the screen: normalization of every position feature to [-1, 1]."""
        return np.array(self.screen_px, np.float32) * self.m_per_px / 2

    @property
    def half_m(self) -> tuple[float, float]:
        """`half` in float64 (exact python arithmetic). The trainers normalize demo positions with this; the public
        features use the float32 `half`. The two differ by 1 ulp, and both are kept as they were so that existing
        checkpoints and frozen features stay bit-identical."""
        return self.screen_px[0] * self.m_per_px / 2, self.screen_px[1] * self.m_per_px / 2

    @property
    def half_px(self) -> tuple[float, float]:
        return self.screen_px[0] / 2, self.screen_px[1] / 2

    @property
    def dt(self) -> float:
        return 1.0 / self.control_hz

    @property
    def step_m(self) -> float:
        """Realizer pointer-step unit: the scripted teachers' MAX_STEP_PX in metres."""
        return MAX_STEP_PX * self.m_per_px

    def as_dict(self) -> dict:
        return dict(screen_px=list(self.screen_px), m_per_px=self.m_per_px, control_hz=self.control_hz)

    @classmethod
    def from_dict(cls, d: dict) -> "PointerGeometry":
        return cls((int(d["screen_px"][0]), int(d["screen_px"][1])), float(d["m_per_px"]), float(d["control_hz"]))
