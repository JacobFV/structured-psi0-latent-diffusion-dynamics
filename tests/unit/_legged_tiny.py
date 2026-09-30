"""Tiny random-weight legged bundle + deterministic row digests for the D-126 deploy-eval tests (plumbing only:
these weights are not a trained model and no number from them is a result)."""
import hashlib
import json
from pathlib import Path

import torch

DZ, W = 8, 32


def tiny_bundle(tmp: Path, seed: int = 0, *, upper_trained: bool = False, factors=None):
    """Random-weight legged bundle. Default: a bare pre-stamp `legged-none` checkpoint (no stamp, legs-only result). With
    `factors` (a spec list, e.g. ["preset:probes:legged-v1", "preset:legged-r19"]) the rep and flow are built on it and saved
    through the trainer's own writer (stamp + factor list). `upper_trained` is the trainer's `result` declaration."""
    from rrp.policies.nets.legged_latent import LeggedEncoder, LeggedFlow, LeggedRealizer, build_legged_rep, legged_probe, legged_specs
    from rrp.policies.features.legged import H
    torch.manual_seed(seed)
    res = dict(latent_space_version="legged-ls-tiny", **({"upper_trained": True} if upper_trained else {}))
    rep = Path(tmp) / "rep" / "representation.pt"
    rep.parent.mkdir(parents=True, exist_ok=True)
    flow = Path(tmp) / "flow" / "policy.pt"
    flow.parent.mkdir(parents=True, exist_ok=True)
    if factors is None:
        E, R, P = LeggedEncoder(dz=DZ, D=W, H=H), LeggedRealizer(dz=DZ, D=W), legged_probe(dz=DZ)
        F = LeggedFlow(dz=DZ, D=W, layers=1)
        torch.save(dict(E=E.state_dict(), R=R.state_dict(), P=P.state_dict(),
                        cfg=dict(name="tiny", latent=dict(dz=DZ, width=W)), result=res), str(rep))
        torch.save(dict(flow=F.state_dict(), cfg=dict(representation=str(rep), width=W, layers=1)), str(flow))
        return rep, flow
    from rrp.harness.train.legged_latent_train import _save
    specs = legged_specs(factors)
    lc = dict(dz=DZ, width=W, factors=list(factors))
    E, R, P = build_legged_rep(lc, specs)
    F = LeggedFlow(dz=DZ, D=W, layers=1, factors=specs)
    cfg = dict(name="tiny", latent=lc)
    _save(rep, E=E.state_dict(), R=R.state_dict(), P=P.state_dict(), cfg=cfg, result=res, specs=specs)
    _save(flow, flow=F.state_dict(), cfg=dict(representation=str(rep), width=W, layers=1, latent=lc), specs=specs)
    return rep, flow


def row_digest(row) -> str:
    r = {k: v for k, v in row.items() if k not in ("wall_s", "checkpoint_provenance")}
    for p in r.get("packets") or []:
        p.pop("generated_at", None)
    return hashlib.sha256(json.dumps(r, sort_keys=True, default=str).encode()).hexdigest()
