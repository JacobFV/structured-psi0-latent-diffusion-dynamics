import numpy as np, sys, json
from pathlib import Path
from rrp.learning.data import load_episodes, episode_samples
ds = Path(sys.argv[1])
for robot in sys.argv[2].split(","):
    eps = load_episodes(ds, robots={robot}, limit_per_robot=20)
    A = np.concatenate([np.stack([s.a for s in episode_samples(p, q, 16, 4)]) for p, q in eps])  # [S,H,N]
    first = A[:, 0, :]
    print(robot, "N", A.shape[2], "absmax per node", np.round(np.abs(A).max((0, 1)), 2).tolist(),
          "std per node", np.round(A.std((0, 1)), 2).tolist(), "frac|a|>3", round(float((np.abs(A) > 3).mean()), 4),
          "first-step std", np.round(first.std(0), 3).tolist(), flush=True)
