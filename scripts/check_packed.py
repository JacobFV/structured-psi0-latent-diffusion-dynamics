"""Packed batches must equal pickle-path batches for the same samples."""
import random, tempfile, numpy as np, torch
from pathlib import Path
from rrp.learning.packed import pack_dataset, PackedChunkDataset
from rrp.learning.data import load_episodes, ChunkDataset, collate_samples
ds = Path("artifacts/datasets/pick_place_primary_v3")
with tempfile.TemporaryDirectory() as d:
    pack_dataset(ds, Path(d), {"panda_pg2"}, 16, stride=4, limit_per_robot=2)
    P = PackedChunkDataset(Path(d))
    eps = load_episodes(ds, robots={"panda_pg2"}, limit_per_robot=2)
    C = ChunkDataset(eps, 16, stride=4)
    assert len(P) == len(C), (len(P), len(C))
    sel = [0, 3, 7]
    b1, a1, v1, l1, e1 = P.collate(np.array(sel))
    b2, a2, v2, l2, e2 = collate_samples([C.samples[i] for i in sel])
    for bk in b2.bank_tokens:
        assert torch.allclose(b1.bank_tokens[bk], b2.bank_tokens[bk], atol=2e-2), bk
    assert torch.equal(b1.ctx_rel, b2.ctx_rel) and torch.equal(b1.act_rel, b2.act_rel) and torch.equal(b1.node_rel, b2.node_rel)
    assert torch.equal(b1.pointers[:, :b2.pointers.shape[1]], b2.pointers)
    assert torch.allclose(a1, a2, atol=2e-3) and torch.equal(v1, v2)
    assert torch.equal(l1["held"], l2["held"]) and torch.equal(l1["focus"], l2["focus"])
    print("packed == pickle path: OK", len(P))
