"""One forward / backward per Psi0 training head (StageA, DirectHead, StructuredHead) on random tensors (D-146 K1).

The upstream Psi0 action transformer is an external dependency, so the two heads that wrap it get a tiny stand-in
header with the same call signature (`header_forward`); everything else (probe, labels, flow loss) is production code.
"""
from types import SimpleNamespace

import pytest

try:
    import torch
    import torch.nn as nn
    from rrp.policies.psi0 import nets as N
except ImportError:          # the unit suite must pass without the ml extra
    torch = N = None
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
