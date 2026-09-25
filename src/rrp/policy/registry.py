"""Workbench policy registry: fixed names -> loaders (no user-supplied paths)."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _latent_loader(ck: Path):
    def load():
        import torch
        from rrp.policy.latent_runner import LatentPolicy
        from rrp.learning.checkpoint import load_checkpoint
        from rrp.learning.latent_train import load_representation
        from rrp.service.sessions import LatentStack
        dev = "cpu"                                   # host workbench: CPU unless a GPU lease runs it
        pol = LatentPolicy.from_checkpoint(ck, device=dev)
        rep = load_checkpoint(ck, map_location="cpu")["config"]["representation"]
        _, _, R, P, _ = load_representation(REPO / rep if not Path(rep).is_absolute() else Path(rep), dev)
        return LatentStack(pol, R, P, name=ck.parent.name)
    return load


def register_workbench_policies(policies: dict):
    for ck in sorted((REPO / "artifacts" / "runs").glob("*/policy.pt")):
        cfg = ck.parent / "config.json"
        if cfg.exists() and '"representation"' in cfg.read_text():
            policies[f"latent:{ck.parent.name}"] = _latent_loader(ck)
