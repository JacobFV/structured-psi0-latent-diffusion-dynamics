"""Re-target a context Batch so the generated entities are controllable assemblies (latent path)."""
from __future__ import annotations

import dataclasses

import torch

from .batch import Batch
from .semantic_latent import assembly_tokens


def assembly_batch(batch: Batch, max_m: int = 2) -> Batch:
    af, am, ai = assembly_tokens(batch, max_m)
    B, M = am.shape
    R = batch.ctx_rel.shape[-1]
    act_rel = torch.gather(batch.ctx_rel, 1, ai[:, :, None, None].expand(-1, -1, batch.ctx_rel.shape[2], R))
    act_rel = act_rel & am[:, :, None, None]
    node_rel = torch.zeros(B, M, M, R, dtype=torch.bool, device=af.device)
    extra = dict(batch.extra, node_ctx_index=ai)
    return dataclasses.replace(batch, node_feats=af, node_mask=am, act_rel=act_rel, node_rel=node_rel, extra=extra)
