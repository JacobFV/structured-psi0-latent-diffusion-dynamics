"""Leakage guards for the legged latent packet: system 0 must not read task/goal context; probes see only z.

Also the R4 (docs/relations.md section 10) migration checks: `route.own_assembly` (preset `legged-s0`) reproduces
the former inline own/body-assembly `-inf` mask bit for bit; `RelBlock` keeps the `block`/`run_block` module key
names (`n1,x,n2,s,n3,m`), so old `LeggedEncoder` / `LeggedRealizer` / `LeggedFlow` checkpoints load strictly;
`legged_probe` + `legged_probe_read` (`ReadoutProbe` on preset `probes:legged-v1`) reproduce the former
`LeggedProbe`'s outputs bit for bit once a legacy state dict is converted by `remap_legged_probe_state`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from rrp.policies.features.legged import NODE_STATIC_DIM, ASM_DIM, GLOBAL_DIM
from rrp.policies.nets.attention import MHA
from rrp.policies.nets.flow import MLP
from rrp.policies.nets.legged_latent import (LeggedRealizer, LeggedEncoder, LeggedFlow, legged_probe,
                                             legged_probe_read, remap_legged_probe_state, N_SUBTASK)
from rrp.policies.relations.base import PRESETS


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
    P = legged_probe(dz=8, max_m=5).eval()
    b = _batch()
    z = torch.randn(3, 4, 5, 8)
    o1 = legged_probe_read(P, z, b["asm_mask"], b["body_asm"])
    o2 = legged_probe_read(P, z[[1, 0, 2]], b["asm_mask"], b["body_asm"])
    assert torch.allclose(o1["goal"][0], o2["goal"][1])      # answers move with z, nothing else is an input
    Pm = legged_probe(dz=8, max_m=5, metadata_only=True).eval()
    m1 = legged_probe_read(Pm, z, b["asm_mask"], b["body_asm"])
    m2 = legged_probe_read(Pm, torch.randn_like(z), b["asm_mask"], b["body_asm"])
    assert torch.allclose(m1["goal"], m2["goal"])            # control probe has no access to z


def test_flow_conditions_on_public_context():
    torch.manual_seed(0)
    F_ = LeggedFlow(dz=8, D=32, layers=1).eval()
    b = _batch()
    g = torch.Generator().manual_seed(1)
    z1 = F_.sample(b, nfe=2, generator=g)
    g = torch.Generator().manual_seed(1)
    z2 = F_.sample(dict(b, ctx=torch.randn_like(b["ctx"])), nfe=2, generator=g)
    assert not torch.allclose(z1, z2)


# ------------------------------------------------------------------ R4: route.own_assembly (docs/relations.md #10)
def _legacy_own_body_bias(b, K, M, N):
    """The pre-migration inline own/body-assembly mask (bit-exact reference)."""
    knot_asm = torch.arange(M).repeat(K)
    body = b["body_asm"][:, None, None]
    own = (b["node_asm"][:, :, None] == knot_asm[None, None]) | (knot_asm[None, None] == body)
    own = own & b["asm_mask"][:, None, :].repeat(1, 1, K)
    B = b["node_asm"].shape[0]
    return torch.zeros(B, 1, N, K * M).masked_fill(~own[:, None], float("-inf"))


def test_route_own_assembly_matches_legacy_mask():
    """`route.own_assembly` (preset `legged-s0`, params.also_key_field="body") reproduces the former inline
    own/body-assembly bias exactly: same -inf pattern for every (node, knot) pair."""
    torch.manual_seed(0)
    R = LeggedRealizer(dz=8, D=32).eval()
    B, K, M, N = 3, 4, 5, 12
    b = _batch(B=B, N=N, M=M)
    knot_valid = b["asm_mask"][:, None].expand(B, K, M).reshape(B, K * M)
    new_bias = R._route_bias(b, K, M, knot_valid, torch.device("cpu"))
    legacy_bias = _legacy_own_body_bias(b, K, M, N)
    assert torch.equal(torch.isinf(new_bias), torch.isinf(legacy_bias))
    finite = ~torch.isinf(legacy_bias)
    assert torch.allclose(new_bias[finite], legacy_bias[finite])


def test_route_own_assembly_off_control_is_unmasked():
    """control="off" on route.own_assembly (via the `factors` override) drops the routing mask entirely: every
    node may read every valid knot (a red/green check that the factor, not a hard-coded mask, drives routing)."""
    torch.manual_seed(0)
    R = LeggedRealizer(dz=8, D=32, factors=[{"name": "route.own_assembly", "control": "off"}]).eval()
    B, K, M, N = 3, 4, 5, 12
    b = _batch(B=B, N=N, M=M)
    knot_valid = b["asm_mask"][:, None].expand(B, K, M).reshape(B, K * M)
    bias = R._route_bias(b, K, M, knot_valid, torch.device("cpu"))
    assert bias is None or not torch.isinf(bias).any()


def test_legged_s0_preset_registered():
    assert "legged-s0" in PRESETS and "probes:legged-v1" in PRESETS
    assert len(PRESETS["legged-s0"]) == 1 and PRESETS["legged-s0"][0]["name"] == "route.own_assembly"
    assert PRESETS["legged-s0"][0]["params"]["also_key_field"] == "body"
    assert list(PRESETS["probes:legged-v1"]) == [f"probe.legged.{q}" for q in
                                                 ("contact", "goal", "disp", "subtask", "fall")]


def test_relblock_module_keys_match_legacy_block():
    """RelBlock keeps the `block()`/`run_block()` module names (n1,x,n2,s,n3,m), so old LeggedEncoder /
    LeggedRealizer / LeggedFlow state dicts load strictly (docs/relations.md section 3.5)."""
    for model in (LeggedEncoder(dz=8, D=16, layers=1), LeggedRealizer(dz=8, D=16, layers=1),
                 LeggedFlow(dz=8, D=16, layers=1)):
        prefixes = {k[len("blocks.0."):].split(".")[0] for k in model.state_dict() if k.startswith("blocks.0.")}
        assert {"n1", "x", "n2", "s", "n3", "m"} <= prefixes


# ------------------------------------------------------------------ R4: LeggedProbe -> ReadoutProbe key map
class _OldLeggedProbe(nn.Module):
    """The pre-migration `LeggedProbe` (kept here only as the golden reference for `remap_legged_probe_state`)."""

    def __init__(self, dz=32, K=4, D=128, heads=4, max_m=11, metadata_only=False, seed=1234):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("asm_code", F.normalize(torch.randn(max_m, 16, generator=g), dim=-1))
        self.register_buffer("knot_code", F.normalize(torch.randn(K, 16, generator=g), dim=-1))
        self.metadata_only = metadata_only
        self.z_in = nn.Linear(dz, D)
        self.code = nn.Linear(16, D)
        self.kcode = nn.Linear(16, D)
        self.qtype = nn.Embedding(5, D)
        self.const = nn.Parameter(torch.zeros(1, 1, D))
        self.a1, self.a2 = MHA(D, heads), MHA(D, heads)
        self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.mlp = MLP(D, D, 2 * D)
        self.heads = nn.ModuleDict(dict(contact=nn.Linear(D, 1), goal=nn.Linear(D, 4), disp=nn.Linear(D, 6),
                                        subtask=nn.Linear(D, N_SUBTASK), fall=nn.Linear(D, 1)))

    def forward(self, z, asm_mask, body_asm):
        B, K, M, _ = z.shape
        tpos = self.kcode(self.knot_code)[None, :, None] + self.code(self.asm_code[:M])[None, None]
        t = (self.const.expand(B, K * M, -1) + tpos.reshape(1, K * M, -1)) if self.metadata_only else \
            (self.z_in(z) + tpos).reshape(B, K * M, -1)
        km = asm_mask[:, None].expand(B, K, M).reshape(B, K * M)

        def read(q):
            r = q + self.a1(self.n1(q), kv=t, key_mask=km)
            r = r + self.a2(self.n2(r), kv=t, key_mask=km)
            return r + self.mlp(r)
        qc = (self.qtype.weight[0] + tpos).reshape(1, K * M, -1).expand(B, -1, -1)
        contact = self.heads["contact"](read(qc)).reshape(B, K, M)
        bc = self.code(self.asm_code[body_asm])
        out = dict(contact=contact)
        for i, k in enumerate(("goal", "disp", "subtask", "fall"), start=1):
            out[k] = self.heads[k](read((self.qtype.weight[i] + bc)[:, None]))[:, 0]
        return out


def test_legacy_legged_probe_checkpoint_loads_via_key_map():
    torch.manual_seed(0)
    old = _OldLeggedProbe(dz=8, K=4, D=32, heads=4, max_m=11, seed=1234).eval()
    new = legged_probe(dz=8, K=4, D=32, heads=4, max_m=11, seed=1234).eval()
    new.load_state_dict(remap_legged_probe_state(old.state_dict()))

    B, K, M = 3, 4, 5
    z = torch.randn(B, K, M, 8)
    asm_mask = torch.ones(B, M, dtype=torch.bool)
    body_asm = torch.tensor([4, 4, 2])                        # varying per-sample body assembly index

    o_old, o_new = old(z, asm_mask, body_asm), legged_probe_read(new, z, asm_mask, body_asm)
    for k in o_old:
        assert o_old[k].shape == o_new[k].shape
        assert torch.allclose(o_old[k], o_new[k], atol=1e-5), k


def test_remap_legged_probe_state_is_idempotent_on_new_layout():
    P = legged_probe(dz=8, K=4, D=32, max_m=11).eval()
    st = P.state_dict()
    assert remap_legged_probe_state(st).keys() == st.keys()
    for k in st:
        assert torch.equal(remap_legged_probe_state(st)[k], st[k])
