"""Side-by-side of two clips (left, right) -> out, streamed frame by frame (W7 grasp contact before/after)."""
import sys
import imageio
import numpy as np
a, b, out = sys.argv[1:4]
ra, rb = imageio.get_reader(a), imageio.get_reader(b)
w = imageio.get_writer(out, fps=20, quality=6)
ia, ib = iter(ra), iter(rb)
la = lb = None
while True:
    x = next(ia, None); y = next(ib, None)
    if x is None and y is None:
        break
    la = x if x is not None else la
    lb = y if y is not None else lb
    w.append_data(np.concatenate([la, np.full((la.shape[0], 6, 3), 255, np.uint8), lb], 1))
w.close()
