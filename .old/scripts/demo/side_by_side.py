"""Compose labelled clips of the SAME scene side by side (demo sprint). Each input keeps its own caption; shorter clips
hold their last frame. Usage: side_by_side.py out.mp4 a.mp4 b.mp4 [c.mp4 ...] (peer: imageio + imageio_ffmpeg)."""
import sys

import imageio.v2 as imageio
import numpy as np

out, ins = sys.argv[1], sys.argv[2:]
clips = [[f for f in imageio.get_reader(p)] for p in ins]
h = min(c[0].shape[0] for c in clips)
n = max(len(c) for c in clips)
frames = []
for t in range(n):
    row = [c[min(t, len(c) - 1)][:h] for c in clips]
    frames.append(np.concatenate(row, axis=1))
imageio.mimsave(out, frames, fps=20, quality=5, macro_block_size=8)
print(out, len(frames), "frames")
