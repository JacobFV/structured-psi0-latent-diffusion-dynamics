"""Timing probe: data sample+fetch cost per batch (CPU) for the latent packed dataset."""
import random, time
from pathlib import Path
from rrp.learning.latent_train import LatentData
d = LatentData(Path("artifacts/packed/latent_pp_v3dart_s1_H16"))
rng = random.Random(0)
t = time.time()
for _ in range(20):
    sel, tgt, j = d.sample(128, rng, 12)
    t1 = time.time(); d.fetch(sel, tgt, "cpu")
print("fetch s/batch", (time.time() - t) / 20)
