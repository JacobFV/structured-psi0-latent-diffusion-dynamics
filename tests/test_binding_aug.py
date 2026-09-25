"""Binding counterfactual edit: the public focus rule recomputed from edited relations must equal the swapped
labels (labels follow the binding), and factual relations must reproduce the stored labels."""
from pathlib import Path

import numpy as np
import pytest
import torch

PACKED = Path("artifacts/packed/latent_pp_v3dart_s1_H16")


@pytest.mark.skipif(not PACKED.exists(), reason="packed data not present")
def test_rebind_focus_follows_binding():
    from rrp.learning.packed import PackedChunkDataset
    from rrp.model.binding_aug import augment, focus_from_batch, slot_has_edges
    ds = PackedChunkDataset(PACKED)
    sel = np.sort(np.random.default_rng(0).choice(len(ds), 256, replace=False))
    batch, a, v, lab, _ = ds.collate(sel)
    S = batch.bank_tokens["scene"].shape[1]
    assert torch.equal(focus_from_batch(batch), lab["focus"].bool()[:, :S])
    g = torch.Generator().manual_seed(0)
    b2, a2, v2, l2, nf, info = augment(batch, a, v, lab, 1.0, g)
    assert nf == 256 and b2.B > 256 and torch.equal(a2[nf:], a[info["pick"]])
    assert torch.equal(focus_from_batch(b2), l2["focus"].bool())
    # edges moved, tokens did not; physical labels unchanged
    assert torch.equal(b2.bank_tokens["scene"][nf:], batch.bank_tokens["scene"][info["pick"]])
    assert torch.equal(l2["held"][nf:], lab["held"][info["pick"]])
    he = slot_has_edges(b2)[nf:]
    b = torch.arange(len(info["pick"]))
    assert he[b, info["dst"]].all()
    # a large share of the counterfactuals really change the focus label
    changed = (l2["focus"][nf:] != lab["focus"][info["pick"]]).any(1).float().mean()
    assert changed > 0.5
