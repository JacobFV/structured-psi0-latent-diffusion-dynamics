"""Caption strip for labelled demo-video frames (moved from scripts/render_episode.py, D-126; that script re-exports it)."""
from __future__ import annotations

import numpy as np


def caption(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Black strip at the top of `frame` with one white text line per entry of `lines` (14 px per line)."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 14 * len(lines) + 6], fill=(0, 0, 0))
    for i, t in enumerate(lines):
        d.text((6, 3 + 14 * i), t, fill=(255, 255, 255))
    return np.asarray(im)
