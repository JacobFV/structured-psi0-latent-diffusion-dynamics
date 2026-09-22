"""Time one deterministic replay+render of a dataset episode (peer, EGL)."""
import sys, time
from pathlib import Path
t0 = time.time()
from rrp.data.vlm_features import replay_render, select_episodes
from rrp.data.collect import read_episode
ds = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/datasets/pick_place_primary_v3")
e = select_episodes(ds, ["parm5_pg2"], 1)[0]
print(e, f"select {time.time() - t0:.1f}s", flush=True)
ref = read_episode(e["path"])["q0"]
t1 = time.time()
r = replay_render("parm5_pg2", e["seed"], e["n_distractors"], ref_q0=ref)
print("replay_s", round(time.time() - t1, 2), "frames", len(r["t"]), "q0_max_dev", r["q0_max_dev"], r["cameras"], r["text"],
      r["frames"][0][0].shape, float(r["frames"][0][0].std()), flush=True)
