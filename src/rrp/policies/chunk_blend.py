"""Overlapping-chunk blending for learned policies (D-126 #7, D-102 (4)): removes the command step at chunk / packet
boundaries. Default OFF everywhere (`chunk_blend="none"` = the historical executor, byte-identical).

Modes (one definition for both paths):
  crossfade  over the first L ticks after a new chunk / packet starts, the executed command is
             (1 - w_i) * previous + w_i * new with w_i = (i + 1) / (L + 1), i = 0..L-1 (ticks since the switch);
             from tick L on the new one alone. The previous source is what WOULD have been executed without the
             replan: for BC the previous (already blended) chunk's rows, for system 0 the previous packet realized at
             the current state and its own phase.
  ensemble   temporal ensembling (ACT-style): every still-valid chunk / packet that covers the tick contributes, with
             weight exp(-decay * age_rank), age_rank 0 = the NEWEST (decay > 0 favours the newest; decay = 0 is the
             plain mean). Old sources expire at the end of their chunk (BC) or at their packet's valid_until (system 0).
Blending uses only public quantities the policy already produced (its own earlier outputs); it adds no inputs.
A graph edit / invalidation drops every older packet (system 0 never blends across an invalidation).
Settings are recorded in the eval summary (`chunk_blend`), and the D-112 chunk-boundary step metric
(rrp.envs.motion_quality, `chunk_vel_step_max`) measures the effect.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CHUNK_BLEND_MODES = ("none", "crossfade", "ensemble")
CHUNK_BLEND_VERSION = "chunk_blend_v1"


@dataclass(frozen=True)
class BlendConfig:
    mode: str = "none"
    ticks: int = 4                 # crossfade length L (ticks)
    decay: float = 0.0             # ensemble weight exp(-decay * age_rank), rank 0 = newest

    def __post_init__(self):
        if self.mode not in CHUNK_BLEND_MODES:
            raise ValueError(f"chunk_blend {self.mode!r} (known: {CHUNK_BLEND_MODES})")
        if self.ticks < 1:
            raise ValueError("blend ticks must be >= 1")

    @property
    def on(self) -> bool:
        return self.mode != "none"

    def record(self) -> dict:
        return dict(mode=self.mode, ticks=self.ticks, decay=self.decay, version=CHUNK_BLEND_VERSION)


def crossfade_weight(i: int, L: int) -> float:
    """Weight of the NEW source i ticks after the switch."""
    return 1.0 if i >= L else (i + 1) / (L + 1)


def ensemble_weights(n: int, decay: float) -> np.ndarray:
    """Normalized weights for n sources ordered NEWEST first."""
    w = np.array([math.exp(-decay * r) for r in range(n)], float)
    return w / w.sum()


# ------------------------------------------------------------------------------------------ BC (ActionChunk rows)
def _mix(rows: list[dict], weights) -> dict:
    out = {}
    for g in rows[0]:
        out[g] = [float(sum(w * float(r[g][c]) for w, r in zip(weights, rows))) for c in range(len(rows[0][g]))]
    return out


def blend_bc_chunk(session, rows: list[dict], now: float, dt: float, cfg: BlendConfig) -> list[dict]:
    """rows: the new chunk's native command rows (list of {group: values}); returns the rows to submit. History is
    stored on the session (`_rrp_blend_hist`: [(start_time, rows)], newest last)."""
    hist = getattr(session, "_rrp_blend_hist", None) or []
    live = []
    for t0, r in hist:                                     # the part of each earlier chunk that covers ticks >= now
        off = int(round((now - t0) / dt))
        if 0 <= off < len(r):
            live.append(r[off:])
    out = []
    for i, row in enumerate(rows):
        cover = [r[i] for r in live if i < len(r)]          # older sources covering tick i (oldest first)
        if not cover:
            out.append({g: list(v) for g, v in row.items()})
            continue
        if cfg.mode == "crossfade":
            w = crossfade_weight(i, cfg.ticks)
            out.append(_mix([row, cover[-1]], [w, 1.0 - w]))   # previous = the latest earlier (blended) chunk
        else:
            srcs = [row] + cover[::-1]                           # newest first
            out.append(_mix(srcs, ensemble_weights(len(srcs), cfg.decay)))
    keep = [(now, out if cfg.mode == "crossfade" else rows)]    # crossfade chains the executed rows; ensemble raw
    if cfg.mode == "ensemble":
        keep = [(t0, r) for t0, r in hist if int(round((now - t0) / dt)) < len(r)] + keep
    session._rrp_blend_hist = keep
    return out


# ------------------------------------------------------------------------------------------ system 0 (packets)
class PacketBlender:
    """Per-system-0 history of accepted packets for blending (newest last). `realize(packet)` must return the
    realizer's normalized output for that packet at the CURRENT state and tick (its own phase)."""

    def __init__(self, cfg: BlendConfig):
        self.cfg = cfg
        self.hist: list[tuple[object, float]] = []           # (packet, time it was accepted)

    def accepted(self, packet, now: float):
        self.hist.append((packet, float(now)))
        if self.cfg.mode == "crossfade":
            self.hist = self.hist[-2:]

    def clear(self):
        self.hist = []

    def blend(self, a_new: np.ndarray, now: float, dt: float, realize) -> np.ndarray:
        if len(self.hist) < 2:
            return a_new
        cur, t_sw = self.hist[-1]
        old = [(p, t) for p, t in self.hist[:-1] if now <= p.valid_until]
        self.hist = old + [(cur, t_sw)]
        if not old:
            return a_new
        if self.cfg.mode == "crossfade":
            i = int(round((now - t_sw) / dt))
            w = crossfade_weight(i, self.cfg.ticks)
            if w >= 1.0:
                return a_new
            return w * a_new + (1.0 - w) * realize(old[-1][0])
        srcs = [a_new] + [realize(p) for p, _ in old[::-1]]     # newest first
        w = ensemble_weights(len(srcs), self.cfg.decay)
        return sum(wi * s for wi, s in zip(w, srcs))
