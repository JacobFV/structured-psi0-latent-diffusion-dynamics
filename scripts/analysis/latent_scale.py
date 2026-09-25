"""Compare encoder-mean latent scale of two frozen representations on the training distribution.

Flow velocity target is v = z - eps, so the zero-knowledge flow MSE floor per dim is E[z^2] + 1 (masked mean).
Writes JSON with per-representation statistics and the implied floor.
"""
import json, random, sys
from pathlib import Path
import torch
from rrp.learning.latent_train import LatentData, load_representation, _dev
from rrp.model.semantic_latent import assembly_tokens

out = Path(sys.argv[1]); reps = sys.argv[2:]
dev = _dev()
data = LatentData(Path("artifacts/packed/latent_pp_v3dart_s1_H16"))
res = {}
for rp in reps:
    lcfg, E, R, P, _ = load_representation(Path(rp), dev)
    rng = random.Random(123); zs = []
    with torch.no_grad():
        for _ in range(20):
            sel, tgt, j = data.sample(128, rng, 0)
            batch, a, v, lab, r = data.fetch(sel, tgt, dev)
            af, am, ai = assembly_tokens(batch)
            mu, _ = E(batch, a, v, af, am, ai)                      # [B,K,M,d]
            m = am[:, None, :].expand(-1, mu.shape[1], -1)
            zs.append(mu[m].float().cpu())
    z = torch.cat(zs)                                              # [n, d]
    sd = z.std(0)
    res[rp] = dict(n=z.shape[0], dz=z.shape[1], rms=float(z.pow(2).mean().sqrt()), second_moment=float(z.pow(2).mean()),
                   mean_abs_dim_mean=float(z.mean(0).abs().mean()), dim_std_median=float(sd.median()),
                   dim_std_max=float(sd.max()), dims_std_gt_0p1=int((sd > 0.1).sum()),
                   norm_p50=float(z.norm(dim=1).median()), norm_p99=float(z.norm(dim=1).quantile(0.99)),
                   flow_floor_zero_knowledge=float(z.pow(2).mean() + 1.0),
                   flow_floor_mean_known=float(z.var(0).mean() + 1.0))
out.write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
