"""Leakage guards for the legged latent packet: system 0 must not read task/goal context; probes see only z."""
import torch

from rrp.control.legged_latent import NODE_STATIC_DIM, ASM_DIM, GLOBAL_DIM
from rrp.model.legged_latent import LeggedRealizer, LeggedProbe, LeggedFlow


def _batch(B=3, N=12, M=5):
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g)
    return dict(node_static=r(B, N, NODE_STATIC_DIM), node_asm=torch.randint(0, 4, (B, N), generator=g),
                asm_static=r(B, M, ASM_DIM), node_mask=torch.ones(B, N, dtype=torch.bool),
                asm_mask=torch.ones(B, M, dtype=torch.bool), asm_is_leg=torch.tensor([[1, 1, 1, 1, 0]] * B).bool(),
                body_asm=torch.full((B,), 4), q=r(B, N), qd=r(B, N), imu=r(B, 6), asm_touch=r(B, M), osc=torch.rand(B),
                ctx=r(B, GLOBAL_DIM))


def test_system0_ignores_task_context_but_uses_packet():
    torch.manual_seed(0)
    R = LeggedRealizer(dz=8, D=32).eval()
    b = _batch()
    z = torch.randn(3, 4, 5, 8)
    ph = torch.zeros(3)
    a0 = R(z, b, ph)
    b2 = dict(b, ctx=torch.randn_like(b["ctx"]))          # task/goal/localization context changed
    assert torch.allclose(a0, R(z, b2, ph))
    assert not torch.allclose(a0, R(z + 1.0, b, ph))       # the packet matters


def test_probe_reads_only_packet():
    P = LeggedProbe(dz=8).eval()
    b = _batch()
    z = torch.randn(3, 4, 5, 8)
    o1 = P(z, b["asm_mask"], b["body_asm"])
    o2 = P(z[[1, 0, 2]], b["asm_mask"], b["body_asm"])
    assert torch.allclose(o1["goal"][0], o2["goal"][1])      # answers move with z, nothing else is an input
    Pm = LeggedProbe(dz=8, metadata_only=True).eval()
    m1, m2 = Pm(z, b["asm_mask"], b["body_asm"]), Pm(torch.randn_like(z), b["asm_mask"], b["body_asm"])
    assert torch.allclose(m1["goal"], m2["goal"])            # control probe has no access to z


def test_flow_conditions_on_public_context():
    torch.manual_seed(0)
    F = LeggedFlow(dz=8, D=32, layers=1).eval()
    b = _batch()
    g = torch.Generator().manual_seed(1)
    z1 = F.sample(b, nfe=2, generator=g)
    g = torch.Generator().manual_seed(1)
    z2 = F.sample(dict(b, ctx=torch.randn_like(b["ctx"])), nfe=2, generator=g)
    assert not torch.allclose(z1, z2)
