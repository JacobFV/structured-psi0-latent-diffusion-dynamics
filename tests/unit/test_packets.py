"""W5 dedup: ladder.make_packet == packets.arm_packet == latent_semantic_edits.build_packet (layout-insensitive), and
the device helper keeps both old behaviours."""
from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pytest


def _fakes():
    asm = [NS(kind="gripper", frame=NS(link="hand")), NS(kind="arm", frame=NS(link="base"))]
    f = NS(spec=NS(assemblies=asm, spec_hash="0123456789abcdef"))
    o = NS(object_descriptors=[NS(slot=0), NS(slot=1)], observation_id="obs:0123456789abcdef")
    s = NS(data=NS(time=1.25), runtime=NS(graph_version=1, runtime_version=1), observe=lambda: o)
    return f, s, o


def test_make_packet_equals_arm_packet_on_strided_z():
    from rrp.harness.eval.ladder import make_packet
    from rrp.harness.eval.latent_semantic_edits import build_packet
    from rrp.harness.eval.packets import arm_packet
    f, s, o = _fakes()
    big = np.arange(4 * 2 * 16, dtype=np.float64).reshape(4, 2, 16) / 7.0
    z = big[:, :1, ::2]                                    # non-contiguous view
    kw = dict(lsv="ls-a", rcv="rz-b", knot_times=[0.1, 0.2, 0.3, 0.4])
    a = make_packet(s, f, z, kw["knot_times"], "ls-a", "rz-b", 0.8, source="target_encoder_oracle", policy_version="v")
    b = arm_packet(f, s, o, z, **kw, source="target_encoder_oracle", name="v", validity=0.8)
    c = build_packet(f, s, o, z, **kw, source="target_encoder_oracle", name="v", sampling={}, validity=0.8)
    for p in (b, c):
        da, dp = a.model_dump(), p.model_dump()
        for d in (da, dp):
            d.pop("generated_at")
            d["z"] = np.asarray(d["z"]).tolist()
        assert da == dp
    assert np.asarray(a.z).dtype == np.float32 and np.array_equal(np.asarray(a.z), z.astype(np.float32))


def test_select_device(monkeypatch):
    import torch
    import rrp.ops.workload as w
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def boom():
        raise RuntimeError("cap")
    monkeypatch.setattr(w, "apply_cap", boom)
    with pytest.raises(RuntimeError):
        w.select_device(on_cap_error="raise")
    assert w.select_device(on_cap_error="ignore").type == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert w.select_device().type == "cpu"
