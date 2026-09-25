"""Pack ONE source of a pack-dual config into <out_dir>.part<k> (lets sources pack in parallel; `rrp latent
pack-dual` then skips finished parts and concatenates)."""
import json, sys
from pathlib import Path
from rrp.learning.packed import pack_dataset
cfg = json.loads(open(sys.argv[1]).read())
k = int(sys.argv[2])
src = cfg["sources"][k]
out = Path(cfg["out_dir"])
d = out.parent / f"{out.name}.part{k}"
meta = pack_dataset(Path(src["dataset"]), d, set(src["robots"]), cfg["horizon"], stride=1,
                    statuses=tuple(src.get("statuses", ["success"])),
                    include_dart_failures=src.get("include_dart_failures", False),
                    seeds=tuple(src["seeds"]) if src.get("seeds") else None, limits=cfg["limits"],
                    multi_m=cfg["multi_m"], exclude_episodes=tuple(src.get("exclude_episodes", ())))
print(json.dumps({kk: meta[kk] for kk in ("n", "robots")}))
