"""Side-by-side of two clips (left, right) -> out, with a white separator (W7 grasp contact before/after)."""
import sys
import imageio
import numpy as np
a, b, out = sys.argv[1:4]
fa, fb = list(imageio.get_reader(a)), list(imageio.get_reader(b))
n = max(len(fa), len(fb))
fa += [fa[-1]] * (n - len(fa)); fb += [fb[-1]] * (n - len(fb))
imageio.mimsave(out, [np.concatenate([x, np.full((x.shape[0], 6, 3), 255, np.uint8), y], 1) for x, y in zip(fa, fb)],
                fps=20, quality=6)
