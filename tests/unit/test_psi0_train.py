"""One forward / backward per Psi0 training head (StageA, DirectHead, StructuredHead) on random tensors (D-146 K1);
the structure fix of architecture 14.5 (D-146 P1): constant-input mask, forced packet use, gate, factor-stamped checkpoint.

The upstream Psi0 action transformer is an external dependency, so the two heads that wrap it get a tiny stand-in
header with the same call signature (`header_forward`); everything else (probe, labels, flow loss) is production code.
"""
from types import SimpleNamespace

import json

import pytest

try:
    import torch
    import torch.nn as nn
    from rrp.policies.nets.checkpoint import save_checkpoint
    from rrp.policies.psi0 import nets as N
    from rrp.policies.psi0 import train as T
    from rrp.policies.relations.base import FactorError
except ImportError:          # the unit suite must pass without the ml extra
    torch = N = T = None
pytestmark = pytest.mark.skipif(torch is None, reason="needs torch")

B = 4


def _batch():
    torch.manual_seed(0)
    lab = dict(hand_dist=torch.rand(B, N.K, 2), contact=torch.randint(0, 2, (B, N.K, 2)), lift=torch.randint(0, 2, (B, N.K)),
               target_pos=torch.randn(B, N.K, 3), active_hand=torch.tensor([0, 1, -1, 1]), base_disp=torch.randn(B, 3))
    return dict(state0=torch.randn(B, 36), actions=torch.randn(B, N.TP, N.DA), amask=torch.ones(B, N.TP, N.DA),
                j=torch.tensor([0, 3, 7, 1]), state_j=torch.randn(B, 36), labels=lab,
                hidden=torch.randn(B, 5, N.VIEW_DIM), mask=torch.ones(B, 5, dtype=torch.long),
                ent=torch.zeros(B, 5))


class _StubHeader(nn.Module):
    """Same call signature as `ActionTransformerModel`; output depends on the noisy action tokens."""

    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, hidden_states, timestep, joint_attention_kwargs, vlm_attn_mask, return_dict):
        return SimpleNamespace(action=self.lin(joint_attention_kwargs["action_hidden_embeds"]))


@pytest.fixture
def stub_header(monkeypatch):
    monkeypatch.setattr(N, "build_header", lambda cfg, action_dim, horizon: _StubHeader(action_dim))


def _step(model, b, keys):
    loss, logs, parts = model.loss(b)
    assert torch.isfinite(loss) and set(keys) <= set(parts)
    loss.backward()
    return [p for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0]


def test_stage_a_forward_backward():
    torch.manual_seed(0)
    A = N.StageA(D=32, probe_D=32)
    assert _step(A, _batch(), ("rec", "kl", "sem"))


def test_direct_head_forward_backward(stub_header):
    H = N.DirectHead(SimpleNamespace())
    assert _step(H, _batch(), ("flow",))


def test_structured_head_forward_backward_reaches_the_semantic_term(stub_header):
    torch.manual_seed(0)
    A = N.StageA(D=32, probe_D=32)
    H = N.StructuredHead(SimpleNamespace(), A, torch.zeros(N.K, N.M, N.DZ), torch.ones(N.K, N.M, N.DZ), w_sem=0.1,
                         sigma_max_sem=1.1)               # sigma <= 1: every sample keeps the semantic term
    got = _step(H, _batch(), ("flow", "sem"))
    assert got and all(not p.requires_grad for p in A.parameters())      # frozen stage A; the header still learns


# ---------------------------------------------------------------- D-146 P1: structure fix (architecture 14.5)
CONST_DIM = 31                # a torso-command state dim: constant -1 in the demonstrations, +1 in closed loop (D-141)


class _Cache(torch.utils.data.Dataset if torch else object):
    """Synthetic feature cache with the keys `psi0.data.collate` reads; state dim CONST_DIM is constant -1."""

    def __init__(self, n=64):
        g = torch.Generator().manual_seed(0)
        s = torch.randn(n, 36, generator=g)
        s[:, N.G.STATE_DIM:] = 0
        s[:, CONST_DIM] = -1.0
        self.s, self.a = s, torch.randn(n, N.TP, N.DA, generator=g)
        self.items = [dict(fr=i) for i in range(n)]

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        return dict(hidden=torch.zeros(1, 4, dtype=torch.bfloat16), ent=torch.zeros(0), state0=self.s[i], actions=self.a[i],
                    amask=torch.ones(N.TP, N.DA), j=torch.tensor(0), state_j=self.s[i], ep=0, fr=i, has_labels=False)


def _train_stage_a(cache, steps=15, **kw):
    torch.manual_seed(0)
    A = N.StageA(D=32, probe_D=32, **kw)
    opt = torch.optim.Adam(A.parameters(), 3e-3)
    b = T.collate([cache[i] for i in range(len(cache))])
    for _ in range(steps):
        loss, _, _ = A.loss(dict(b), w_sem=0.0)
        opt.zero_grad(); loss.backward(); opt.step()
    return A.eval(), b


def _torso_shift(A, b):
    """Max |R output| change when the constant torso-command dim flips from -1 (training) to +1 (closed loop)."""
    z = torch.randn(b["state_j"].shape[0], N.K, N.M, N.DZ)
    with torch.no_grad():
        outs = []
        for v in (-1.0, 1.0):
            s = b["state_j"].clone(); s[:, CONST_DIM] = v
            outs.append(A.R(A.morph, z, s, b["j"].float() / N.TP))
    return float((outs[1] - outs[0]).abs().max())


def test_constant_torso_command_shift_red_green():
    cache = _Cache()
    A, b = _train_stage_a(cache)
    assert _torso_shift(A, b) > 1e-3                              # red: R reads the training-constant dim (the D-141 OOD)
    masked = A.fit_state_mask(T.state_std(cache))                 # the production fit over the cache
    assert masked == [CONST_DIM]
    assert _torso_shift(A, b) == 0.0                              # green: the masked dim cannot move R


def test_forced_packet_use_hinge_and_gate(tmp_path):
    cache = _Cache(16)
    A = N.StageA(D=32, probe_D=32)
    b = T.collate([cache[i] for i in range(16)])
    _, logs, parts = A.loss(dict(b), w_sem=0.0)
    assert "perm" in parts and "perm_gap" in logs
    with torch.no_grad():                                         # an R that ignores its packet: no gap, hinge = margin
        A.R.z_in.weight.zero_(); A.R.z_in.bias.zero_()
        _, logs, _ = A.loss(dict(b), w_sem=0.0, p_state_dim=0.0, p_state_all=0.0)
        em, ee = A.packet_gap(b, torch.zeros(N.K, N.M, N.DZ))
    assert abs(logs["perm_gap"]) < 1e-5 and abs(logs["perm_hinge"] - N.PACKET_MARGIN) < 1e-5
    assert float((em - ee).abs().max()) < 1e-5
    rep = T.packet_use(A, cache, torch.zeros(N.K, N.M, N.DZ), stride=2, batch=4)           # the heldout gate report
    assert rep["frames"] == 8 and not rep["passed"] and abs(rep["gap"]) < 1e-5
    ck = tmp_path / "stage_a.pt"
    save_checkpoint(ck, model=A, step=0, versions=dict(psi0_nets=N.NETS_VERSION), config=dict(stage_a=A.cfg))
    gate = tmp_path / "packet_gate.json"
    bad = dict(packet_use=dict(gap=0.0, margin=N.PACKET_MARGIN, passed=False, stage_a_sha256_16=T.file_digest(ck)))
    gate.write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="does not use the packet"):
        T.require_gate(gate, ck)
    with pytest.raises(SystemExit, match="no packet-use gate"):
        T.require_gate(tmp_path / "missing.json", ck)
    bad["packet_use"].update(passed=True, gap=0.5)
    gate.write_text(json.dumps(bad))
    assert T.require_gate(gate, ck)["gap"] == 0.5
    bad["packet_use"]["stage_a_sha256_16"] = "0" * 16            # a gate of another stage A does not count
    gate.write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="no packet_use for"):
        T.require_gate(gate, ck)


def test_stage_a_checkpoint_carries_mask_and_factors(tmp_path):
    cache = _Cache()
    A, _ = _train_stage_a(cache, steps=1, factors=["preset:psi0-dims", "preset:s0-psi0", "preset:probes:psi0-v1"])
    A.fit_state_mask(T.state_std(cache))
    ck = tmp_path / "stage_a.pt"
    meta = save_checkpoint(ck, model=A, step=1, versions=dict(psi0_nets=N.NETS_VERSION), config=dict(stage_a=A.cfg))
    assert meta["versions"]["factors"].startswith("fx-")
    B = N.load_stage_a(ck)
    assert B.morph.state_keep[CONST_DIM] == 0 and B.morph.state_keep.sum() == N.G.STATE_DIM - 1     # mask restored
    assert B.specs == A.specs
    with pytest.raises(FactorError):                                                      # a different structure is refused
        N.load_stage_a(ck, factors=["preset:s0-psi0", "preset:probes:psi0-v1"])
    N.load_stage_a(ck, factors=["preset:s0-psi0", "preset:probes:psi0-v1"], allow_factor_mismatch=True)
    old = tmp_path / "old.pt"                                                             # a pre-P1 file has no config
    torch.save(dict(model=A.state_dict(), args={}), old)
    with pytest.raises(ValueError, match="not a P1 stage-A checkpoint"):
        N.load_stage_a(old)
    with pytest.raises(FactorError, match="applies to no site"):             # family check at construction
        N.StageA(D=32, probe_D=32, factors=["ui.same_window"])
