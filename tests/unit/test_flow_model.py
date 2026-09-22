import copy
import numpy as np
import pytest
import torch

from rrp.sim.fixtures import make_pick_place_session
from rrp.data.collect import featurizer_for
from rrp.model.batch import collate_inputs
from rrp.model.flow import FlowPolicy, PolicyConfig

torch.manual_seed(0)


@pytest.fixture(scope="module")
def inputs():
    s = make_pick_place_session(seed=3, n_distractors=1)
    f = featurizer_for(s)
    out = [f(s.observe())]
    for _ in range(5):
        s.step(None)
    out.append(f(s.observe()))
    return out


def small(**kw):
    cfg = PolicyConfig(width=64, heads=2, ctx_layers=1, blocks=2, horizon=4, **kw)
    return FlowPolicy(cfg).double()


def dbl(b):
    import dataclasses
    def cv(x):
        if isinstance(x, torch.Tensor) and x.dtype == torch.float32:
            return x.double()
        if isinstance(x, dict):
            return {k: cv(v) for k, v in x.items()}
        return x
    return dataclasses.replace(b, **{k: cv(v) for k, v in b.__dict__.items()})


def test_zero_bias_equals_no_bias(inputs):
    b = dbl(collate_inputs(inputs))
    m_true = small(bias_mode="true")
    m_none = small(bias_mode="none")
    m_none.load_state_dict(m_true.state_dict())
    z = torch.randn(2, 4, b.node_feats.shape[1], 1, dtype=torch.float64)
    tau = torch.tensor([0.3, 0.7], dtype=torch.float64)
    v1 = m_true.velocity(z, tau, m_true.prepare(b))
    v2 = m_none.velocity(z, tau, m_none.prepare(b))
    assert torch.allclose(v1, v2, atol=1e-10)
    # nonzero bias changes outputs
    with torch.no_grad():
        for mod in m_true.modules():
            if hasattr(mod, "w") and isinstance(mod.w, torch.nn.Parameter):
                mod.w.fill_(1.0)
        m_true.out.weight.normal_()
    v3 = m_true.velocity(z, tau, m_true.prepare(b))
    m_none.out.weight.data.copy_(m_true.out.weight.data)
    v4 = m_none.velocity(z, tau, m_none.prepare(b))
    assert not torch.allclose(v3, v4)


def test_cached_equals_uncached_across_sampler_steps(inputs):
    b = dbl(collate_inputs(inputs))
    m = small()
    with torch.no_grad():
        m.out.weight.normal_()
    cache = m.prepare(b)
    z = torch.randn(2, 4, b.node_feats.shape[1], 1, dtype=torch.float64)
    for tau in (0.0, 0.5, 0.9):
        t = torch.full((2,), tau, dtype=torch.float64)
        assert torch.allclose(m.velocity(z, t, cache), m.velocity(z, t, m.prepare(b)), atol=1e-12)


def test_node_permutation_equivariance(inputs):
    pi = inputs[0]
    N = pi.act_node_feats.shape[0]
    perm = np.random.default_rng(0).permutation(N)
    # permute action nodes jointly: node features, relations, noise; morph action-node tokens too
    pj = copy.deepcopy(pi)
    inv = np.argsort(perm)
    pj.act_node_feats = pi.act_node_feats[perm]
    mt = pi.tokens["morph"].copy()
    mt[:N] = pi.tokens["morph"][perm]
    pj.tokens["morph"] = mt
    R = pi.relations.copy()
    for row in R:
        if row[0] == -1:
            row[1] = inv[row[1]]
        if row[0] == 0 and row[1] < N:
            row[1] = inv[row[1]]
        if row[2] == 0 and row[3] < N:
            row[3] = inv[row[3]]
    pj.relations = R
    P = pi.pointers.copy()
    for row in P:
        if row[0] == 0 and row[1] < N:
            row[1] = inv[row[1]]
        if row[2] == 0 and row[3] < N:
            row[3] = inv[row[3]]
    pj.pointers = P
    m = small()
    with torch.no_grad():
        m.out.weight.normal_()
        for mod in m.modules():
            if hasattr(mod, "w") and isinstance(mod.w, torch.nn.Parameter):
                mod.w.normal_()
    b1, b2 = dbl(collate_inputs([pi])), dbl(collate_inputs([pj]))
    z = torch.randn(1, 4, N, 1, dtype=torch.float64)
    t = torch.tensor([0.4], dtype=torch.float64)
    v1 = m.velocity(z, t, m.prepare(b1))
    v2 = m.velocity(z[:, :, perm], t, m.prepare(b2))
    assert torch.allclose(v1[:, :, perm], v2, atol=1e-9)


def test_padding_nodes_do_not_change_valid_outputs(inputs):
    m = small()
    with torch.no_grad():
        m.out.weight.normal_()
    b1 = dbl(collate_inputs([inputs[0]]))
    big = copy.deepcopy(inputs[1])
    big.act_node_feats = np.concatenate([big.act_node_feats, big.act_node_feats[:2]], 0)  # longer sample
    b2 = dbl(collate_inputs([inputs[0], big]))
    N = b1.node_feats.shape[1]
    z1 = torch.randn(1, 4, N, 1, dtype=torch.float64)
    z2 = torch.randn(2, 4, b2.node_feats.shape[1], 1, dtype=torch.float64)
    z2[0, :, :N] = z1[0]
    z2[0, :, N:] = 1e3   # garbage in padding
    t1 = torch.tensor([0.3], dtype=torch.float64)
    v1 = m.velocity(z1, t1, m.prepare(b1))
    v2 = m.velocity(z2, torch.tensor([0.3, 0.3], dtype=torch.float64), m.prepare(b2))
    assert torch.allclose(v1[0], v2[0, :, :N], atol=1e-9)
    assert torch.all(v2[0, :, N:] == 0)


def test_aux_readout_gradients_reach_action_expert_blocks(inputs):
    from rrp.learning.data import Sample, collate_samples
    m = FlowPolicy(PolicyConfig(width=32, heads=2, ctx_layers=1, blocks=3, horizon=4, aux=True))
    S = inputs[0].tokens["scene"].shape[0]
    N = inputs[0].act_node_feats.shape[0]
    labels = dict(held=np.ones(S, bool), contact=np.zeros(S, bool), visible=np.ones(S, bool), focus=np.zeros(S, bool),
                  slot_valid=np.ones(S, bool), rel_tcp=np.zeros((S, 3), np.float32),
                  future_disp=np.zeros((S, 3), np.float32), gaze=np.zeros(S, np.float32), status={})
    smp = [Sample(pi, np.zeros((4, N), np.float32), np.ones((4, N), bool), labels, np.zeros((4, 4), np.float32), {})
           for pi in inputs]
    batch, a, v, lab, eff = collate_samples(smp)
    _, logs = m.loss(batch, a, v, lab, aux_weight=1.0)
    m.zero_grad()
    # aux-only gradient
    cache = m.prepare(batch)
    z = torch.randn_like(a)
    vel, hidden = m.velocity(z, torch.full((2,), 0.5), cache, return_hidden=True)
    aux, _ = m.readout(hidden, cache, batch, lab, z)
    aux.backward()
    g_block0 = sum(p.grad.abs().sum() for p in m.blocks[0].parameters() if p.grad is not None)
    assert g_block0 > 0, "auxiliary loss must shape system-i denoising blocks"
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_unstructured_baseline_has_no_pointer_messages(inputs):
    b = collate_inputs(inputs)
    m = FlowPolicy(PolicyConfig(width=32, heads=2, ctx_layers=1, blocks=1, horizon=4, structured=False,
                                bias_mode="none", aux=False))
    c = m.prepare(b)
    assert all(x is None for x in c.act_bias)
    n_params_s = sum(p.numel() for p in FlowPolicy(PolicyConfig(width=32, heads=2, ctx_layers=1, blocks=1,
                                                                horizon=4, aux=False)).parameters())
    n_params_u = sum(p.numel() for p in m.parameters())
    assert n_params_s == n_params_u   # matched parameter budget (pointer vs text projections are same size)
