"""Pack ONE source of a pack-dual config into <out_dir>.part<k>, in seed chunks to bound memory (pack_dataset
buffers a whole call in RAM), then concatenate the chunks (dual_latent.concat_packed). `rrp latent pack-dual`
later skips finished parts and concatenates the parts.
Usage: binding_pack_part.py <config> <k> <seed_lo> <seed_hi> <chunk_seeds>"""
import json, shutil, sys
from pathlib import Path
from rrp.learning.dual_latent import concat_packed
from rrp.learning.packed import pack_dataset
cfg = json.loads(open(sys.argv[1]).read())
k, lo, hi, step = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
src = cfg["sources"][k]
out = Path(cfg["out_dir"])
d = out.parent / f"{out.name}.part{k}"
chunks = []
for s0 in range(lo, hi, step):
    c = out.parent / f"{out.name}.part{k}.c{s0}"
    if not (c / "meta.json").exists():
        meta = pack_dataset(Path(src["dataset"]), c, set(src["robots"]), cfg["horizon"], stride=1,
                            statuses=tuple(src.get("statuses", ["success"])),
                            include_dart_failures=src.get("include_dart_failures", False),
                            seeds=(s0, min(s0 + step, hi)), limits=cfg["limits"], multi_m=cfg["multi_m"],
                            exclude_episodes=tuple(src.get("exclude_episodes", ())))
        print(s0, meta["n"], flush=True)
    chunks.append(c)
meta = concat_packed(chunks, d)
meta["config_source"] = src
(d / "meta.json").write_text(json.dumps(meta, indent=1))
for c in chunks:
    shutil.rmtree(c)
print(json.dumps({kk: meta[kk] for kk in ("n", "robots")}))
