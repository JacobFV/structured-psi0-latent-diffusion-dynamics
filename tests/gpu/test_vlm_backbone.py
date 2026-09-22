"""Real rendered images + task text through the ACTUAL frozen VLM adapter (peer GPU only)."""
import os

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
if not torch.cuda.is_available() or not os.environ.get("RRP_GPU_MEMORY_BYTES"):
    pytest.skip("peer GPU lease required", allow_module_level=True)
os.environ.setdefault("MUJOCO_GL", "egl")


def test_real_images_through_adapter_and_resampler():
    from rrp.ops.gpu import apply_cap
    from rrp.model.backbone import BackboneSpec, VLMBackbone, Resampler, Renderer, task_text
    from rrp.sim.fixtures import make_pick_place_session
    apply_cap()
    vlm = VLMBackbone(BackboneSpec(), device="cuda")
    prov = vlm.provenance()
    assert prov["hidden_width"] == 2048 and prov["n_layers"] == 28 and prov["vocab"] >= 153000
    frames, texts = [], []
    for seed in (1, 2):
        s = make_pick_place_session(seed=seed, n_distractors=2)
        r = Renderer(s.model, 256)
        im = r.render(s.data, r.cameras(["front"]))
        assert im[0].std() > 5            # a real rendered scene, not a blank frame
        frames.append(im)
        texts.append(task_text(s))
        r.close()
    enc = vlm.encode(frames, texts)
    t = enc["tokens"]
    assert t.shape == (2, enc["n_visual"] + 8, 2, 2048) and enc["n_visual"] == 64
    assert torch.isfinite(t.float()).all() and t.float().std() > 0
    assert (t[0].float() - t[1].float()).abs().mean() > 1e-3        # different scenes -> different features
    blank = vlm.encode([[np.zeros_like(frames[0][0])]], texts[:1])["tokens"]
    assert (blank[0].float() - t[0].float()).abs().mean() > 1e-3
    rs = Resampler(2048, 2, dim=256, queries=16)
    out = rs(t, enc["n_visual"])
    assert out.shape == (2, 16, 256)
