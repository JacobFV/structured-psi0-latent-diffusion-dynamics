"""Tiny random-weight legged bundle + deterministic row digests for the D-126 deploy-eval tests (plumbing only:
these weights are not a trained model and no number from them is a result)."""
import hashlib
import json
from pathlib import Path

import torch

DZ, W = 8, 32


def tiny_bundle(tmp: Path, seed: int = 0):
    from rrp.policies.nets.legged_latent import LeggedEncoder, LeggedFlow, LeggedProbe, LeggedRealizer
    from rrp.policies.features.legged import H
    torch.manual_seed(seed)
    E, R, P = LeggedEncoder(dz=DZ, D=W, H=H), LeggedRealizer(dz=DZ, D=W), LeggedProbe(dz=DZ)
    F = LeggedFlow(dz=DZ, D=W, layers=1)
    rep = Path(tmp) / "rep" / "representation.pt"
    rep.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(E=E.state_dict(), R=R.state_dict(), P=P.state_dict(),
                    cfg=dict(name="tiny", latent=dict(dz=DZ, width=W)), result=dict(latent_space_version="legged-ls-tiny")),
               str(rep))
    flow = Path(tmp) / "flow" / "policy.pt"
    flow.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(flow=F.state_dict(), cfg=dict(representation=str(rep), width=W, layers=1)), str(flow))
    return rep, flow


def row_digest(row) -> str:
    r = {k: v for k, v in row.items() if k not in ("wall_s", "checkpoint_provenance")}
    for p in r.get("packets") or []:
        p.pop("generated_at", None)
    return hashlib.sha256(json.dumps(r, sort_keys=True, default=str).encode()).hexdigest()
