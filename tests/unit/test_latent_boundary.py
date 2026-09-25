"""R38 boundary tests (mechanical, small untrained modules on a real fixture session)."""
import json
import subprocess
import sys
import numpy as np
import pytest
import torch

from rrp.sim.fixtures import make_pick_place_session
from rrp.contracts.latent_action import LatentActionChunk
from rrp.control.latent_realizer import LatentRealizer, LatentSystem0
from rrp.model.flow import FlowPolicy, PolicyConfig
from rrp.model.latent_probes import PacketProbe, probe_loss
from rrp.policy.latent_runner import LatentPolicy

DZ, K = 16, 4
torch.manual_seed(0)


def tiny_policy():
    m = FlowPolicy(PolicyConfig(width=32, heads=2, ctx_layers=1, blocks=1, horizon=K, latent_dim=DZ, aux=False))
    with torch.no_grad():
        m.out.weight.normal_(0, 0.1)
    return LatentPolicy(m, knot_times=(0.1, 0.3, 0.5, 0.7), latent_space_version="ls-test", realizer_compat_version="rz-test",
                        device="cpu")


def tiny_realizer():
    r = LatentRealizer(DZ, width=32, heads=2, layers=1)
    with torch.no_grad():
        r.out.weight.normal_(0, 0.5)
    return r


def test_independent_consumer_in_separate_process(tmp_path):
    s = make_pick_place_session(seed=3)
    pol, R = tiny_policy(), tiny_realizer()
    p = pol.packets([s])[0]
    (tmp_path / "packet.json").write_bytes(p.to_bytes())
    torch.save(R.state_dict(), tmp_path / "r.pt")
    s0 = LatentSystem0(R, pol.featurizer(s), latent_space_version="ls-test", realizer_compat_version="rz-test")
    s0.receive(p, now=float(s.data.time), graph_version=s.runtime.graph_version)
    cmd = s0.tick(s, s.controller_version())
    code = f"""
import json, torch
from rrp.contracts.latent_action import LatentActionChunk
from rrp.control.latent_realizer import LatentRealizer, LatentSystem0
from rrp.sim.fixtures import make_pick_place_session
from rrp.data.collect import featurizer_for
p = LatentActionChunk.from_bytes(open(r'{tmp_path}/packet.json','rb').read())
R = LatentRealizer({DZ}, width=32, heads=2, layers=1); R.load_state_dict(torch.load(r'{tmp_path}/r.pt'))
s = make_pick_place_session(seed=3)          # fresh instance: no upstream model, cache or hidden state
s0 = LatentSystem0(R, featurizer_for(s), latent_space_version='ls-test', realizer_compat_version='rz-test')
s0.receive(p, now=float(s.data.time), graph_version=s.runtime.graph_version)
print(json.dumps(s0.tick(s, s.controller_version()).groups))
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    other = json.loads(out.stdout.strip().splitlines()[-1])
    for g in cmd.groups:
        assert np.allclose(cmd.groups[g], other[g], atol=1e-5)


def test_no_semantic_bypass_and_version_invalidation():
    s = make_pick_place_session(seed=4, n_distractors=1)
    pol, R = tiny_policy(), tiny_realizer()
    p = pol.packets([s])[0]
    s0 = LatentSystem0(R, pol.featurizer(s), latent_space_version="ls-test", realizer_compat_version="rz-test")
    s0.receive(p, now=float(s.data.time), graph_version=s.runtime.graph_version)
    snap = s.snapshot()
    c1 = s0.tick(s, s.controller_version())
    # change inaccessible upstream context: object/goal positions and descriptors seen only by system i
    s.restore(snap)
    s.teleport_object("cube", [0.30, -0.25, 0.023])
    s.teleport_object("target_zone", [0.50, 0.25, 0.0005])
    s.tracker.update(float(s.data.time), {0: np.array([0.3, -0.25, 0.023])})
    c2 = s0.tick(s, s.controller_version())
    for g in c1.groups:
        assert np.allclose(c1.groups[g], c2.groups[g], atol=1e-6), "system 0 must not read scene/task context"
    # graph edit = version change -> packet invalidated (not a semantic dependence)
    s.runtime.apply_edit([{"op": "set_priority", "event_id": "grasp", "priority": 2}], s.runtime.graph_version, "e1")
    assert s0.tick(s, s.controller_version()) is None and s0.packet is None


def test_metadata_only_probe_ignores_latent():
    P = PacketProbe(DZ, K, width=32, metadata_only=True)
    z1, z2 = torch.randn(2, K, 1, DZ), torch.randn(2, K, 1, DZ)
    m = torch.ones(2, 1, dtype=torch.bool)
    o1, o2 = P(z1, m, 4), P(z2, m, 4)
    for k in o1:
        assert torch.allclose(o1[k], o2[k])


def test_semantic_loss_on_predicted_clean_latent_reaches_flow_model_only_via_packet():
    from rrp.model.latent_batch import assembly_batch
    from rrp.model.batch import collate_inputs
    s = make_pick_place_session(seed=5, n_distractors=1)
    pol = tiny_policy()
    b = assembly_batch(collate_inputs([pol.featurizer(s)(s.observe())] * 2))
    P = PacketProbe(DZ, K, width=32)
    for q in P.parameters():
        q.requires_grad_(False)                       # frozen probe; gradient must still flow through its input
    Sn = b.bank_tokens["scene"].shape[1]
    lab = dict(held=torch.zeros(2, Sn, dtype=torch.bool), contact=torch.zeros(2, Sn, dtype=torch.bool),
               visible=torch.ones(2, Sn, dtype=torch.bool), focus=torch.zeros(2, Sn, dtype=torch.bool),
               gaze=torch.zeros(2, Sn), rel_tcp=torch.zeros(2, Sn, 3), future_disp=torch.zeros(2, Sn, 3),
               subtask=torch.zeros(2, dtype=torch.long))
    smask = b.bank_mask["scene"]
    model = pol.model
    model.train()
    target = torch.randn(2, K, 1, DZ)
    valid = b.node_mask[:, None, :].expand(-1, K, -1)
    loss, logs = model.loss(b, target, valid, None, packet_loss_fn=lambda zc: probe_loss(P(zc, b.node_mask, Sn), lab, smask),
                            packet_weight=1.0)
    assert model.readout is None                     # no hidden-state readout route exists in this model
    model.zero_grad()
    # isolate the packet-semantic term: recompute only it at an intermediate noise level
    cache = model.prepare(b)
    eps = torch.randn_like(target)
    tau = torch.full((2,), 0.5)
    z_tau = 0.5 * eps + 0.5 * target
    v = model.velocity(z_tau, tau, cache)
    z_hat = z_tau + 0.5 * v
    sem, _ = probe_loss(P(z_hat, b.node_mask, Sn), lab, smask)
    sem.backward()
    g = [p.grad for p in model.blocks.parameters() if p.grad is not None]
    assert g and all(torch.isfinite(x).all() for x in g) and sum(float(x.abs().sum()) for x in g) > 0
    assert all(p.grad is None for p in P.parameters())
