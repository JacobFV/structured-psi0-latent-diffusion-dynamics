"""Diagnostic (track binding): can the Stage-A encoder learn to read the supplied binding at all?
Trains E + a fresh probe on the FOCUS query only, with counterfactual-binding copies (frac 0.5), for a few
thousand steps, and reports held-out cf focus_follows over time. Variants:
  full     : as in Stage A (context + demonstrated trajectory)
  no_traj  : demonstrated actions zeroed (removes the trajectory shortcut)
Usage: binding_diag_learnability.py <variant> <steps> <out.json>"""
import json, random, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from rrp.learning.latent_train import LatentData, _dev, binding_cf_metrics
from rrp.model.binding_aug import augment
from rrp.model.latent_probes import PacketProbe
from rrp.model.semantic_latent import LatentConfig, TargetEncoder, assembly_tokens

variant, steps, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
dev = _dev()
torch.manual_seed(0)
cfg = LatentConfig()
E, P = TargetEncoder(cfg).to(dev), PacketProbe(cfg.dz, cfg.knots).to(dev)
opt = torch.optim.AdamW(list(E.parameters()) + list(P.parameters()), lr=3e-4, weight_decay=1e-4)
data = LatentData(Path("artifacts/packed/latent_pp_v3dart_s1_H16"))
rng, rng_ev = random.Random(0), random.Random(123)
log, t0 = [], time.time()


def evaluate():
    E.eval(); P.eval()
    agg = {}
    g = torch.Generator().manual_seed(7)
    for _ in range(8):
        sel, tgt, j = data.sample(128, rng_ev, 0)
        b, a, v, lab, r = data.fetch(sel, tgt, dev)
        if variant == "no_traj":
            a = torch.zeros_like(a)
        for k, (x, n) in binding_cf_metrics(E, P, b, a, v, lab, g).items():
            s_, n_ = agg.get(k, (0, 0)); agg[k] = (s_ + x, n_ + n)
    E.train(); P.train()
    return {k: x / n for k, (x, n) in agg.items() if n and k in ("cf_focus_follows", "cf_focused_on", "cf_focused_on_pos", "cf_rel_z_dist")}


for step in range(1, steps + 1):
    sel, tgt, j = data.sample(128, rng, 0)
    b, a, v, lab, r = data.fetch(sel, tgt, dev)
    if variant == "no_traj":
        a = torch.zeros_like(a)
    b, a, v, lab, nf, _ = augment(b, a, v, lab, 0.5)
    af, am, ai = assembly_tokens(b)
    mu, _ = E(b, a, v, af, am, ai)
    S = b.bank_tokens["scene"].shape[1]
    o = P(mu, am, S)
    m = (b.bank_mask["scene"] & lab["slot_valid"].bool()).float()
    loss = (F.binary_cross_entropy_with_logits(o["focused_on"][..., 0], lab["focus"].float(), reduction="none") * m).sum() / m.sum()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 250 == 0:
        rec = dict(step=step, t=round(time.time() - t0), loss=float(loss), **evaluate())
        log.append(rec)
        print(json.dumps(rec), flush=True)
        out.write_text(json.dumps(dict(variant=variant, log=log), indent=1))
