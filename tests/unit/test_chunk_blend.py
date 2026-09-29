"""D-126 #7 overlapping-chunk blending: weights, BC chunk rows, system-0 packets; default none = unchanged."""
from types import SimpleNamespace

import numpy as np
import pytest

from rrp.policies.chunk_blend import (BlendConfig, PacketBlender, blend_bc_chunk, crossfade_weight,
                                         ensemble_weights)


def test_weights():
    assert [crossfade_weight(i, 3) for i in range(5)] == [0.25, 0.5, 0.75, 1.0, 1.0]
    w = ensemble_weights(3, 0.0)
    assert np.allclose(w, 1 / 3)
    w = ensemble_weights(2, np.log(2))            # newest first: 1 : 0.5
    assert np.allclose(w, [2 / 3, 1 / 3])
    with pytest.raises(ValueError):
        BlendConfig("smooth")
    assert not BlendConfig().on


def _rows(v, n=16):
    return [{"arm": [float(v), float(v)], "gripper": [float(v)]} for _ in range(n)]


def test_bc_crossfade_and_ensemble_rows():
    s = SimpleNamespace()
    cfg = BlendConfig("crossfade", ticks=3)
    first = blend_bc_chunk(s, _rows(0.0), now=0.0, dt=0.05, cfg=cfg)
    assert first == _rows(0.0)                                   # nothing to blend with
    out = blend_bc_chunk(s, _rows(1.0), now=0.4, dt=0.05, cfg=cfg)   # 8 ticks later: previous rows 8.. overlap
    assert [r["arm"][0] for r in out[:5]] == pytest.approx([0.25, 0.5, 0.75, 1.0, 1.0])
    assert out[9]["gripper"][0] == 1.0                           # beyond the old chunk: new rows alone
    out2 = blend_bc_chunk(s, _rows(2.0), now=0.8, dt=0.05, cfg=cfg)    # chains the EXECUTED (blended) rows
    assert out2[0]["arm"][0] == pytest.approx(0.25 * 2.0 + 0.75 * 1.0)
    s = SimpleNamespace()
    ens = BlendConfig("ensemble", decay=0.0)
    blend_bc_chunk(s, _rows(0.0), now=0.0, dt=0.05, cfg=ens)
    out = blend_bc_chunk(s, _rows(1.0), now=0.4, dt=0.05, cfg=ens)
    assert out[0]["arm"][0] == pytest.approx(0.5) and out[7]["arm"][0] == pytest.approx(0.5)
    assert out[8]["arm"][0] == pytest.approx(1.0)                # old chunk ended
    out = blend_bc_chunk(s, _rows(4.0), now=0.8, dt=0.05, cfg=ens)
    assert out[0]["arm"][0] == pytest.approx((4.0 + 1.0) / 2)   # the first chunk expired; raw rows are ensembled


def test_packet_blender():
    pk = lambda t, v: SimpleNamespace(valid_from=t, valid_until=t + 0.8, v=v)
    b = PacketBlender(BlendConfig("crossfade", ticks=2))
    realize = lambda p: np.full(3, p.v)
    b.accepted(pk(0.0, 0.0), 0.0)
    assert np.allclose(b.blend(np.zeros(3), 0.0, 0.05, realize), 0)
    b.accepted(pk(0.4, 1.0), 0.4)
    assert np.allclose(b.blend(np.ones(3), 0.4, 0.05, realize), 1 / 3)
    assert np.allclose(b.blend(np.ones(3), 0.45, 0.05, realize), 2 / 3)
    assert np.allclose(b.blend(np.ones(3), 0.5, 0.05, realize), 1.0)
    b.clear()                                                    # invalidation: no blending across it
    b.accepted(pk(0.5, 5.0), 0.5)
    assert np.allclose(b.blend(np.full(3, 5.0), 0.5, 0.05, realize), 5.0)
    e = PacketBlender(BlendConfig("ensemble"))
    e.accepted(pk(0.0, 0.0), 0.0)
    e.accepted(pk(0.4, 2.0), 0.4)
    assert np.allclose(e.blend(np.full(3, 2.0), 0.5, 0.05, realize), 1.0)
    assert np.allclose(e.blend(np.full(3, 2.0), 0.85, 0.05, realize), 2.0)     # first packet expired at 0.8


def test_system0_blend_single_matches_batched_and_default_is_unchanged():
    torch = pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    from rrp.policies.system0 import LatentRealizer, LatentSystem0, batched_ticks
    from rrp.envs.mujoco.fixtures import make_pick_place_session
    from tests.unit.test_latent_grpo import DZ, _actor, _policy
    from rrp.harness.train.flow_sde import SDEConfig
    torch.manual_seed(1)
    m = _policy()
    R = LatentRealizer(DZ, width=32, heads=2, layers=1)
    with torch.no_grad():
        R.out.weight.normal_(0, 0.5)

    def run(blend):
        a = _actor(m, SDEConfig(nfe=4))                          # same sampler seed: identical packets every run
        S = [make_pick_place_session(seed=sd) for sd in (3, 4)]
        s0 = [LatentSystem0(R, a.featurizer(s), latent_space_version="ls-t", realizer_compat_version="rz-t")
              for s in S]
        if blend:
            for x in s0:
                x.configure_blend("crossfade", 3)
        pk1 = a.packets(S)
        for x, p, s in zip(s0, pk1, S):
            x.receive(p, now=float(s.data.time), graph_version=s.runtime.graph_version)
        for s in S:
            s.step(None)
        pk2 = a.packets(S)
        for x, p, s in zip(s0, pk2, S):
            x.receive(p, now=float(s.data.time), graph_version=s.runtime.graph_version)
        return S, s0, pk1, pk2
    S, s0, _, _ = run(True)
    single = [x.tick(s, s.controller_version()) for x, s in zip(s0, S)]
    S2, s02, _, _ = run(True)
    batched = batched_ticks(s02, S2)
    for c1, c2 in zip(single, batched):
        for g in c1.groups:
            assert np.allclose(c1.groups[g], c2.groups[g], atol=1e-5)
    S3, s03, _, _ = run(False)
    plain = batched_ticks(s03, S3)
    assert s03[0].blender is None
    diff = max(float(np.abs(np.asarray(c1.groups["arm"]) - np.asarray(c2.groups["arm"])).max())
               for c1, c2 in zip(batched, plain))
    assert diff > 1e-6                                           # blending is active right after the switch
