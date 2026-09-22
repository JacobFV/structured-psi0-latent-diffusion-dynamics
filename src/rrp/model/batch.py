"""Collate PolicyInputs into padded tensors.

Context tokens are the concatenation of the four banks (morph, scene, task, interact) in a
fixed order, each padded to the batch max. Relations become multi-hot tensors:
  ctx_rel  [B, C, C, R]  context->context (query row, key column)
  act_rel  [B, N, C, R]  action node (query) -> context (key)
  node_rel [B, N, N, R]  action node -> action node (kinematic parent)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rrp.data.features import BANKS, N_REL, HASH_DIM

MORPH_DIM = 35
BANK_DIMS = {"morph": MORPH_DIM, "scene": 24, "task": 59, "interact": 32}
NODE_DIM = 35


@dataclass
class Batch:
    bank_tokens: dict          # bank -> [B, T, F]
    bank_mask: dict            # bank -> [B, T] bool (True = valid)
    bank_kind: dict            # bank -> [B, T] long
    bank_text: dict            # bank -> [B, T, HASH_DIM] (serialized pointer text, unstructured mode)
    bank_offset: dict          # bank -> int offset into concatenated context
    ctx_mask: torch.Tensor     # [B, C]
    node_feats: torch.Tensor   # [B, N, F]
    node_mask: torch.Tensor    # [B, N]
    ctx_rel: torch.Tensor      # [B, C, C, R] bool
    act_rel: torch.Tensor      # [B, N, C, R] bool
    node_rel: torch.Tensor     # [B, N, N, R] bool
    pointers: torch.Tensor     # [B, P, 2] (src ctx idx, dst ctx idx), -1 padded
    extra: dict

    def to(self, device):
        def mv(x):
            if isinstance(x, torch.Tensor):
                return x.to(device, non_blocking=True)
            if isinstance(x, dict):
                return {k: mv(v) for k, v in x.items()}
            return x
        return Batch(**{k: mv(v) for k, v in self.__dict__.items()})

    @property
    def B(self):
        return self.node_feats.shape[0]


def collate_inputs(inputs: list, extra_tokens: dict | None = None) -> Batch:
    B = len(inputs)
    T = {b: max(max(len(pi.tokens[b]) for pi in inputs), 1) for b in BANKS}
    offs, o = {}, 0
    for b in BANKS:
        offs[b] = o
        o += T[b]
    C = o
    toks, masks, kinds, texts = {}, {}, {}, {}
    for b in BANKS:
        F = BANK_DIMS[b]
        x = np.zeros((B, T[b], F), np.float32)
        m = np.zeros((B, T[b]), bool)
        k = np.zeros((B, T[b]), np.int64)
        t = np.zeros((B, T[b], HASH_DIM), np.float32)
        for i, pi in enumerate(inputs):
            n = len(pi.tokens[b])
            x[i, :n, :pi.tokens[b].shape[1]] = pi.tokens[b]
            m[i, :n] = True
            k[i, :n] = pi.token_kind[b]
            pt = pi.pointer_text[b]
            t[i, :len(pt)] = pt
        toks[b], masks[b], kinds[b], texts[b] = (torch.from_numpy(x), torch.from_numpy(m), torch.from_numpy(k),
                                                 torch.from_numpy(t))
    N = max(pi.act_node_feats.shape[0] for pi in inputs)
    nf = np.zeros((B, N, NODE_DIM), np.float32)
    nm = np.zeros((B, N), bool)
    ctx_rel = np.zeros((B, C, C, N_REL), bool)
    act_rel = np.zeros((B, N, C, N_REL), bool)
    node_rel = np.zeros((B, N, N, N_REL), bool)
    P = max(max(len(pi.pointers) for pi in inputs), 1)
    ptr = -np.ones((B, P, 2), np.int64)
    for i, pi in enumerate(inputs):
        n = pi.act_node_feats.shape[0]
        nf[i, :n] = pi.act_node_feats
        nm[i, :n] = True
        for qb, qi, kb, ki, r in pi.relations:
            kc = offs[BANKS[kb]] + ki
            if qb == -1:
                if kb == 0 and ki < n and r == 16 and False:
                    pass
                act_rel[i, qi, kc, r] = True
            else:
                ctx_rel[i, offs[BANKS[qb]] + qi, kc, r] = True
        for j, (sb, si, db, di) in enumerate(pi.pointers):
            ptr[i, j] = (offs[BANKS[sb]] + si, offs[BANKS[db]] + di)
    # node->node kinematic relation derived from morph relations among action-node tokens
    kp = 16
    for i, pi in enumerate(inputs):
        n = pi.act_node_feats.shape[0]
        for qb, qi, kb, ki, r in pi.relations:
            if qb == 0 and kb == 0 and r == kp and qi < n and ki < n:
                node_rel[i, qi, ki, kp] = True
                node_rel[i, ki, qi, kp] = True
    ctx_mask = torch.cat([masks[b] for b in BANKS], 1)
    return Batch(toks, masks, kinds, texts, offs, ctx_mask, torch.from_numpy(nf), torch.from_numpy(nm),
                 torch.from_numpy(ctx_rel), torch.from_numpy(act_rel), torch.from_numpy(node_rel),
                 torch.from_numpy(ptr), dict(extra_tokens or {}))
