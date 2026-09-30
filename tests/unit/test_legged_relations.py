"""HL (D-146): legged relation sites. `legged-rel-v1` edges + `leg.foothold` / `leg.com_support` at the ctx>ctx, act>ctx
and act>act sites of the legged encoder / flow / BC net, with `legged-none` (the default) leaving every state-dict key
and output unchanged."""
import math

import pytest
import torch

from rrp.policies.features.legged import NODE_STATIC_DIM, ASM_DIM, GLOBAL_DIM
from rrp.policies.nets import legged_latent as LL
from rrp.policies.relations.base import FAMILIES, PRESETS, PrivilegedInput, estimates_loss, resolve

DZ, D = 8, 32
LEG_L, LEG_R, BODY, ARM_L, ARM_R = 0, 1, 2, 3, 4


def batch(B=2, terrain=True, seed=0):
    """A tiny humanoid: two 3-joint legs (assemblies 0 / 1, mirror pair), a 1-joint body (2), two 2-joint arms (3, 4)."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    chains = [(LEG_L, 3), (LEG_R, 3), (BODY, 1), (ARM_L, 2), (ARM_R, 2)]
    asm_of = [a for a, n in chains for _ in range(n)]
    depth = [d for a, n in chains for d in range(n)]
    N, M = len(asm_of), 5
    ns = r(B, N, NODE_STATIC_DIM)
    ns[..., 9] = torch.tensor(depth, dtype=torch.float32) / 6
    st = torch.zeros(B, M, ASM_DIM)
    kind = [0, 0, 1, 2, 2]
    pos = [(0.1, 0.1, -0.5), (0.1, -0.1, -0.5), (0.0, 0.0, 0.0), (0.0, 0.2, 0.1), (0.0, -0.2, 0.1)]
    side = [1, -1, 0, 1, -1]
    for m in range(M):
        st[:, m, kind[m]] = 1
        st[:, m, 3:6] = torch.tensor(pos[m])
        st[:, m, 7] = side[m]
        st[:, m, 8] = 1.0
    b = dict(node_static=ns, node_asm=torch.tensor(asm_of)[None].repeat(B, 1), asm_static=st,
             node_mask=torch.ones(B, N, dtype=torch.bool), asm_mask=torch.ones(B, M, dtype=torch.bool),
             asm_is_leg=torch.tensor([[1, 1, 0, 0, 0]] * B).bool(), body_asm=torch.full((B,), BODY),
             q=r(B, N), qd=r(B, N), imu=r(B, 6), asm_touch=r(B, M), osc=torch.rand(B, generator=g), ctx=r(B, GLOBAL_DIM))
    if terrain:
        b["terrain"] = 0.1 * r(B, LL.SCAN_DIM)
        b["terrain_valid"] = torch.ones(B, LL.SCAN_DIM, dtype=torch.bool)
    return b


def labelled(b):
    B = b["q"].shape[0]
    fc = torch.full((B, 5), -2)
    fc[:, LEG_L], fc[:, LEG_R] = 37, -1                 # left foot lands on cell 37, right foot is planted
    return dict(b, foothold_cell=fc, com_support=torch.randn(B, 1), com_support_valid=torch.ones(B, dtype=torch.bool))


def excite(module, scale=0.7, seed=3):
    """The bilinear g and the edge weights are zero-initialised (a no-op at init): give every site parameter a value."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in module.named_parameters():
            if ".b." in "." + n or "rsites" in n:
                p.copy_(scale * torch.randn(p.shape, generator=g))


def specs_of(*names, off=()):
    items = [{"name": n, **({"control": "off"} if n in off else {})} for n in names]
    return resolve(items, family="legged")


# ------------------------------------------------------------------ registry
def test_preset_is_nine_edges_plus_foothold_and_com_support():
    assert len(LL.LEGGED_REL_VOCAB) == 9
    names = [s.name for s in resolve(["preset:legged"], family="legged")]
    assert names == [f"edge.{n}" for n in LL.LEGGED_REL_VOCAB] + ["leg.foothold", "leg.com_support"]
    assert resolve(None, default="legged-none", family="legged") == ()
    assert list(PRESETS["legged-none"]) == []
    for fam in ("legged", "humanoid"):
        sites = FAMILIES[fam].sites
        assert all("edges:legged-rel-v1" in sites[s] for s in LL.LEGGED_SITES)


def test_scan_constants_match_the_terrain_core():
    core = pytest.importorskip("rrp.envs.mujoco.legged_core")
    assert (LL.SCAN_NX, LL.SCAN_NY, LL.SCAN_DX, LL.SCAN_X0, LL.SCAN_Y0) == \
        (core.SCAN_NX, core.SCAN_NY, core.SCAN_DX, core.SCAN_X0, core.SCAN_Y0)
    xy = LL.scan_xy()
    assert xy.shape == (core.SCAN_DIM, 2)
    ix, iy = 3, 5
    assert torch.allclose(xy[ix * core.SCAN_NY + iy], torch.tensor([core.SCAN_X0 + ix * core.SCAN_DX,
                                                                    core.SCAN_Y0 + iy * core.SCAN_DX]))


# ------------------------------------------------------------------ graph
def test_edges_kinematics_mirror_and_terrain():
    b = batch()
    tok, rc = LL.legged_graph(b, K=2)
    ch = {n: i for i, n in enumerate(LL.LEGGED_REL_VOCAB)}
    E = rc.edges["ctx>ctx"].data
    J, L, Fo, Ce = (tok.sl(k) for k in ("joint", "limb", "foot", "cell"))
    kp = E[0, J, J, ch["kin_parent"]]
    assert kp[1, 0] and kp[2, 1] and not kp[0, 1] and not kp[3, 2]          # chain: joint 1 -> parent 0; leg boundary cut
    assert torch.equal(E[0, J, J, ch["kin_child"]], kp.T)
    mj = E[0, J, J, ch["mirror"]]
    assert mj[0, 3] and mj[4, 1] and not mj[0, 4]                          # equal depth in the mirror leg only
    assert E[0, L, L, ch["mirror"]][LEG_L, LEG_R] and E[0, L, L, ch["mirror"]][ARM_L, ARM_R]
    assert not E[0, L, L, ch["mirror"]][BODY].any()
    assert E[0, L, L, ch["limb_adjacent"]][LEG_L, ARM_L] is not None
    assert E[0, L, L, ch["limb_adjacent"]][LEG_L, LEG_R] and not E[0, L, L, ch["limb_adjacent"]][LEG_L, ARM_L]
    assert E[0, J, L, ch["node_in_assembly"]][4, LEG_R] and not E[0, J, L, ch["node_in_assembly"]][4, LEG_L]
    assert E[0, L, Fo, ch["foot_of"]][LEG_L, LEG_L] and not E[0, L, Fo, ch["foot_of"]][BODY].any()
    over = E[0, Fo, Ce, ch["over_cell"]]
    assert over[LEG_L].any() and not over[BODY].any() and not over[ARM_L].any()
    xy = LL.scan_xy()[over[LEG_L]]
    assert (xy[:, 0] - 0.1 >= LL.OVER_CELL_DX[0] - 1e-6).all() and ((xy[:, 1] - 0.1).abs() <= LL.OVER_CELL_DY + 1e-6).all()
    A = rc.edges["act>ctx"].data
    assert A.shape[:3] == (2, 2 * 5, tok.T) and rc.edges["act>act"].data.shape == (2, 10, 10, 9)
    assert not A[..., ch["kin_parent"]].any()                                # the act side has no kinematic-tree edge
    AA = rc.edges["act>act"].data
    assert AA[0, LEG_L, 5 + LEG_L, ch["same_assembly"]] and AA[0, LEG_L, LEG_L, ch["same_node"]]


def test_graph_is_public_only_and_labels_are_training_only():
    b = labelled(batch())
    _, rc = LL.legged_graph(b, K=2, deploy=False)
    ys = rc.sets["ctx"].labels
    assert ys["foothold_next"].shape[1] == ys["foothold_next"].shape[2]
    assert ys["foothold_next"].sum() == 2                                    # one landing cell per sample (planted: none)
    assert ys["foothold_next.valid"][:, :, :].sum() > 0
    _, rc = LL.legged_graph(b, K=2, deploy=True)
    assert rc.sets["ctx"].labels == {}
    _, rc = LL.legged_graph(batch(), K=2)
    assert rc.sets["ctx"].labels == {}                                       # missing keys: masked, not an error


# ------------------------------------------------------------------ legged-none is the old net
@pytest.mark.parametrize("cls,kw", [(LL.LeggedEncoder, dict(dz=DZ, D=D, layers=1, H=4)),
                                    (LL.LeggedFlow, dict(dz=DZ, D=D, layers=1))])
def test_legged_none_keeps_keys_and_outputs(cls, kw):
    torch.manual_seed(0)
    old = cls(**kw)
    torch.manual_seed(0)
    new = cls(**kw, factors=None)
    assert list(old.state_dict()) == list(new.state_dict())
    assert not any("rsites" in k or ".b." in k or k.startswith("ctx.limb") for k in new.state_dict())
    assert not new.ctx.extended
    b = batch(terrain=False)
    if cls is LL.LeggedEncoder:
        beh = torch.randn(2, b["q"].shape[1], 4)
        assert torch.equal(old(b, beh)[0], new(b, beh)[0])
    else:
        g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
        assert torch.equal(old.sample(b, nfe=2, generator=g1), new.sample(b, nfe=2, generator=g2))


# ------------------------------------------------------------------ the factors act
def _encode(E, b, beh, attn=None):
    with torch.no_grad():
        mu, lv, rc = E.encode(b, beh, attn=attn)
    return mu, rc


def _enc(specs):
    torch.manual_seed(0)
    E = LL.LeggedEncoder(dz=DZ, D=D, layers=1, H=4, factors=specs).eval()
    excite(E)
    return E


def test_toggling_foothold_and_kin_parent_changes_attention_and_output():
    """Acceptance (research/readiness.md HL): on a tiny humanoid batch, switching `leg.foothold` and `edge.kin_parent` off
    (control="off": same parameters, term removed) changes the ctx attention and the packet."""
    b = batch()
    beh = torch.randn(2, b["q"].shape[1], 4)
    names = ("edge.same_node", "edge.kin_parent", "edge.over_cell", "leg.foothold")
    on = _enc(specs_of(*names))
    a_on, a_kp, a_fh = [], [], []
    mu_on, _ = _encode(on, b, beh, a_on)
    for off, sink in (("edge.kin_parent", a_kp), ("leg.foothold", a_fh)):
        E = _enc(specs_of(*names, off=(off,)))
        E.load_state_dict(on.state_dict())
        mu, _ = _encode(E, b, beh, sink)
        assert not torch.allclose(mu, mu_on, atol=1e-6), off
        assert not torch.allclose(sink[0], a_on[0], atol=1e-7), off
        assert torch.isfinite(mu).all()
    # kin_parent shifts the attention of a joint towards its parent, foothold shifts foot -> cell rows
    tok, _ = LL.legged_graph(b, K=4)
    J, Fo, Ce = tok.sl("joint"), tok.sl("foot"), tok.sl("cell")
    assert (a_kp[0][0][:, J, J] - a_on[0][0][:, J, J]).abs().max() > 1e-5
    assert (a_fh[0][0][:, Fo, Ce] - a_on[0][0][:, Fo, Ce]).abs().max() > 1e-6


def test_act_sites_change_the_flow_velocity():
    b = batch()
    names = ("edge.same_node", "edge.mirror", "edge.same_assembly", "leg.foothold")
    torch.manual_seed(0)
    F1 = LL.LeggedFlow(dz=DZ, D=D, layers=1, factors=specs_of(*names)).eval()
    excite(F1)
    F0 = LL.LeggedFlow(dz=DZ, D=D, layers=1, factors=specs_of(*names, off=names)).eval()
    F0.load_state_dict(F1.state_dict())
    z, tau = torch.randn(2, 4, 5, DZ), torch.tensor([0.3, 0.7])
    with torch.no_grad():
        v1, v0 = F1.velocity(z, tau, F1.prepare(b), b), F0.velocity(z, tau, F0.prepare(b), b)
    assert not torch.allclose(v1, v0, atol=1e-6)


def _bc(specs=None, H=4, **kw):
    from rrp.policies.nets.legged_bc import LeggedBC
    torch.manual_seed(0)
    return LeggedBC(D=D, heads=4, enc_layers=1, dec_layers=2, H=H, factors=specs, **kw).eval()


def test_bc_legged_none_is_the_pre_relations_net():
    from rrp.policies.nets.legged_bc import build
    old, new = _bc(H=40), build(dict(model=dict(width=D, enc_layers=1, dec_layers=2)))
    assert not old.extended and not any("rsites" in k or ".b." in k or k.startswith("foot") for k in old.state_dict())
    assert list(old.state_dict()) == list(new.state_dict())
    b = batch(terrain=False)
    a = torch.randn(2, b["q"].shape[1], 40)
    xt, tau = torch.randn_like(a), torch.tensor([0.3, 0.7])
    with torch.no_grad():
        new.load_state_dict(old.state_dict())
        assert torch.equal(old.velocity(xt, tau, old.prepare(b), b), new.velocity(xt, tau, new.prepare(b), b))


def test_bc_relations_toggle_changes_attention_and_actions():
    """The BC positive control takes the same `legged-rel-v1` factors: switching `leg.foothold` / `edge.kin_parent` off
    changes the encoder attention and the decoded joint velocity; the build reads `model.factors` from the config."""
    from rrp.policies.nets.legged_bc import build
    b = batch()
    names = ("edge.same_node", "edge.kin_parent", "edge.same_assembly", "edge.over_cell", "leg.foothold")
    on = _bc(specs_of(*names))
    excite(on)
    assert on.extended
    cfg = dict(model=dict(width=D, enc_layers=1, dec_layers=2, factors=["preset:legged"]))
    assert build(cfg).extended and not build(dict(model=dict(width=D))).extended
    a = torch.randn(2, b["q"].shape[1], 4)
    xt, tau = torch.randn_like(a), torch.tensor([0.3, 0.7])
    att_on = []
    with torch.no_grad():
        c = on.prepare(b, attn=att_on)
        v_on = on.velocity(xt, tau, c, b)
    assert torch.isfinite(v_on).all()
    tok, _ = LL.legged_graph(b, K=1)
    for off in ("edge.kin_parent", "leg.foothold"):
        m = _bc(specs_of(*names, off=(off,)))
        m.load_state_dict(on.state_dict())
        att = []
        with torch.no_grad():
            v = m.velocity(xt, tau, m.prepare(b, attn=att), b)
        assert not torch.allclose(v, v_on, atol=1e-6), off
        assert not torch.allclose(att[0], att_on[0], atol=1e-7), off


def test_bc_loss_trains_the_pair_estimate_and_deploy_refuses_it():
    b = labelled(batch())
    m = _bc(specs_of("edge.same_node", "leg.foothold")).train()
    excite(m)
    a = torch.randn(2, b["q"].shape[1], 4)
    loss = m.loss(b, a, torch.ones(2, b["q"].shape[1], dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters()
                                        if ".b.f" in n or ".b.w" in n)
    m.set_deploy()
    assert m.prepare(dict(b))[3].sets["ctx"].labels == {}          # labels dropped in a deploy graph


def test_extended_net_trains_the_pair_estimate():
    """`estimates_loss` supervises the foothold pair logits with the batch label; the loss has gradient."""
    from rrp.policies.relations.base import estimates_loss
    b = labelled(batch())
    specs = specs_of("edge.same_node", "leg.foothold")
    torch.manual_seed(0)
    E = LL.LeggedEncoder(dz=DZ, D=D, layers=1, H=4, factors=specs)
    excite(E)
    mu, lv, rc = E.encode(b, torch.randn(2, b["q"].shape[1], 4))
    assert ("pair", "leg.foothold") in rc.estimates
    loss, logs, metrics = estimates_loss(rc, specs)
    loss.backward()
    assert loss.item() > 0 and "probe_leg.foothold" in logs
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in E.ctx.enc[0]["b"].parameters())
    _, rc2 = None, LL.legged_graph(batch(), K=4)[1]           # no label keys: masked, count 0
    E2 = LL.LeggedEncoder(dz=DZ, D=D, layers=1, H=4, factors=specs)
    _, _, rc3 = E2.encode(batch(), torch.randn(2, b["q"].shape[1], 4))
    l3, _, m3 = estimates_loss(rc3, specs)
    assert m3["leg.foothold_acc"] == (0.0, 0) and l3.item() == 0.0


def test_deploy_mode_refuses_the_estimate_loss_and_drops_labels():
    b = labelled(batch())
    specs = specs_of("edge.same_node", "leg.foothold")
    E = LL.LeggedEncoder(dz=DZ, D=D, layers=1, H=4, factors=specs).set_deploy(True)
    _, _, rc = E.encode(b, torch.randn(2, b["q"].shape[1], 4))
    assert rc.sets["ctx"].labels == {}
    with pytest.raises(PrivilegedInput):
        estimates_loss(rc, specs)


# ------------------------------------------------------------------ probe read / com_support
def test_com_support_readout_and_terms():
    from rrp.policies.nets.probes import readout_loss
    specs = resolve(["preset:probes:legged-v1", "leg.com_support"], family="legged")
    P = LL.legged_probe(dz=DZ, max_m=5, factors=specs)
    z = torch.randn(2, 4, 5, DZ)
    b = labelled(batch())
    out = LL.legged_probe_read(P, z, b["asm_mask"], b["body_asm"])
    assert out["com_support"].shape == (2, 2) and out["contact"].shape == (2, 4, 5)
    lab = dict(contact_k=torch.zeros(2, 4, 5), goal=torch.randn(2, 2), goal_valid=torch.ones(2, dtype=torch.bool),
               disp=torch.randn(2, 3), subtask=torch.zeros(2, dtype=torch.long), fall=torch.zeros(2),
               com_support=b["com_support"], com_support_valid=b["com_support_valid"])
    pred, labels, masks = LL.probe_terms(out, lab, b)
    total, logs = readout_loss(pred, labels, specs, masks)
    assert {"probe_com_support", "probe_contact", "probe_goal"} <= set(logs) and math.isfinite(float(total))
    lab.pop("com_support"), lab.pop("com_support_valid")                    # absent label: masked, no term
    _, logs2 = readout_loss(*(lambda p, l, m: (p, l, specs, m))(*LL.probe_terms(out, lab, b)))
    assert "probe_com_support" not in logs2


# ------------------------------------------------------------------ trainer (tiny synthetic shard, CPU, a few steps)
def _write_shard(root, terrain=True, T=24, n_eps=3):
    import json
    import numpy as np
    from rrp.policies.features.legged import GLOBAL_DIM as G
    b = batch(B=1, terrain=False)
    rng = np.random.default_rng(0)
    N, M = b["q"].shape[1], 5
    d = dict(node_static=b["node_static"][0].numpy(), asm_static=b["asm_static"][0].numpy(),
             node_asm=b["node_asm"][0].numpy(), nf=np.array(2), n_policy=np.array(N),
             ep=np.repeat(np.arange(n_eps), T), q=rng.normal(size=(n_eps * T, N)).astype(np.float32),
             qd=rng.normal(size=(n_eps * T, N)).astype(np.float32), a=rng.normal(size=(n_eps * T, N)).astype(np.float32),
             imu=rng.normal(size=(n_eps * T, 6)).astype(np.float32), osc=rng.random(n_eps * T).astype(np.float32),
             touch=rng.random((n_eps * T, M)).astype(np.float32), contact=(rng.random((n_eps * T, M)) > 0.5),
             ctx=rng.normal(size=(n_eps * T, G)).astype(np.float32), ev=rng.integers(0, 4, n_eps * T),
             pose=rng.normal(size=(n_eps * T, 3)).astype(np.float32))
    if terrain:
        d["terrain"] = (0.05 * rng.normal(size=(n_eps * T, LL.SCAN_DIM))).astype(np.float32)
        d["terrain_valid"] = np.ones((n_eps * T, LL.SCAN_DIM), bool)
        fc = np.full((n_eps * T, M), -2)
        fc[:, 0] = rng.integers(-1, LL.SCAN_DIM, n_eps * T)
        fc[:, 1] = -1
        d["foothold_cell"] = fc
        d["com_support"] = rng.normal(size=n_eps * T).astype(np.float32)
        d["com_support_valid"] = rng.random(n_eps * T) > 0.2
    (root / "tinybody").mkdir(parents=True)
    np.savez(root / "tinybody" / "s0.npz", **d)
    (root / "tinybody" / "s0.json").write_text(json.dumps(dict(episodes=[
        dict(seed=(1, 2, 19)[i], status="ok", waypoints=dict(a=[1.0, 0.0], b=[2.0, 0.0])) for i in range(n_eps)])))


def _cfg(root, factors, steps=3):
    return dict(name="hl", data=str(root), bodies=["tinybody"], steps=steps, batch_size=8, ckpt_every=1000,
                latent=dict(dz=DZ, width=D, beta_kl=1e-3, factors=factors))


PROBES = "preset:probes:legged-v1"


@pytest.fixture(autouse=True)
def _short_evals(monkeypatch):
    """The trainer's held-out evals default to 40 / 30 batches of 256: two are enough for a plumbing check."""
    import functools
    from rrp.harness.train import legged_latent_train as T
    monkeypatch.setattr(T, "eval_rep", functools.partial(T.eval_rep, n_batches=2))
    monkeypatch.setattr(T, "eval_flow", functools.partial(T.eval_flow, n_batches=2))


def test_train_rep_runs_the_relational_losses_and_stamps_versions(tmp_path):
    import json
    from rrp.harness.train import legged_latent_train as T
    _write_shard(tmp_path / "d")
    cfg = _cfg(tmp_path / "d", [PROBES, "preset:legged"])
    res = T.train_rep(cfg, tmp_path / "out")
    st = torch.load(tmp_path / "out" / "representation.pt", weights_only=False)
    specs = T._legged_specs(cfg["latent"])
    assert st["_provenance"]["versions"]["factors"].startswith(
        __import__("rrp.policies.relations.base", fromlist=["compat_hash"]).compat_hash(specs))
    assert [f["name"] for f in st["factors"]][-2:] == ["leg.foothold", "leg.com_support"]
    log = json.loads((tmp_path / "out" / "train_log.jsonl").read_text().splitlines()[-1]) \
        if (tmp_path / "out" / "train_log.jsonl").read_text().strip() else {}
    assert "com_support_mae" in res["eval"]["probes"] and res["eval"]["probes"]["com_support_mae"] is not None
    assert any(k.startswith("ctx.limb") for k in st["E"]) and any(".b." in k for k in st["E"])
    cfg2, E, R, P, rres, specs2 = T.load_legged_rep(tmp_path / "out" / "representation.pt", torch.device("cpu"))
    assert E.ctx.extended and [s.name for s in specs2] == [s.name for s in specs] and "com_support" in P.heads
    del log


def test_legged_none_trainer_is_unchanged_and_legacy_checkpoints_load(tmp_path):
    from rrp.harness.train import legged_latent_train as T
    from rrp.policies.relations.base import FactorError
    _write_shard(tmp_path / "d", terrain=False)
    cfg = _cfg(tmp_path / "d", [PROBES])
    T.train_rep(cfg, tmp_path / "out")
    st = torch.load(tmp_path / "out" / "representation.pt", weights_only=False)
    assert not any(k.startswith("ctx.limb") or ".b." in k for k in st["E"])
    _, E, _, _, _, specs = T.load_legged_rep(tmp_path / "out" / "representation.pt", torch.device("cpu"))
    assert not E.ctx.extended and not T.relational_specs(specs)
    # a checkpoint written before the stamp existed is legged-none: it loads, but not into a relational config
    st["_provenance"]["versions"].pop("factors")
    torch.save(st, tmp_path / "legacy.pt")
    T.load_legged_rep(tmp_path / "legacy.pt", torch.device("cpu"))
    st["cfg"]["latent"]["factors"] = [PROBES, "preset:legged"]
    torch.save(st, tmp_path / "legacy_rel.pt")
    with pytest.raises(FactorError):
        T.load_legged_rep(tmp_path / "legacy_rel.pt", torch.device("cpu"))
    # a stamped checkpoint refuses a changed factor structure
    st2 = torch.load(tmp_path / "out" / "representation.pt", weights_only=False)
    st2["cfg"]["latent"]["factors"] = [PROBES, "edge.same_node"]
    torch.save(st2, tmp_path / "mismatch.pt")
    with pytest.raises(FactorError):
        T.load_legged_rep(tmp_path / "mismatch.pt", torch.device("cpu"))


def test_train_flow_uses_the_representations_factors_and_is_sealed(tmp_path):
    from rrp.harness.train import legged_latent_train as T
    from rrp.core.sealed import SealedSplitError
    _write_shard(tmp_path / "d")
    T.train_rep(_cfg(tmp_path / "d", [PROBES, "preset:legged"]), tmp_path / "rep")
    fcfg = dict(representation=str(tmp_path / "rep" / "representation.pt"), steps=2, batch_size=8, width=D, layers=1,
                semantic_weight=0.5)
    res = T.train_flow(fcfg, tmp_path / "flow")
    st = torch.load(tmp_path / "flow" / "policy.pt", weights_only=False)
    assert any(".b." in k for k in st["flow"]) and any("rsites" in k for k in st["flow"])
    assert st["_provenance"]["versions"]["factors"] and "com_support_mae" in res["eval"]["oracle"]
    calls = []
    orig = T.SealedSplit.load
    T.SealedSplit.load = staticmethod(lambda *a, **k: type("S", (), {"assert_dataset_allowed": lambda self, *a: (calls.append(a),
                                                                     (_ for _ in ()).throw(SealedSplitError("no", code="x")))[1]})())
    try:
        with pytest.raises(SealedSplitError):
            T.train_rep(_cfg(tmp_path / "d", [PROBES]), tmp_path / "out_sealed")
    finally:
        T.SealedSplit.load = orig
    assert calls


def test_bc_trainer_builds_from_model_factors_and_stamps_them(tmp_path):
    """`legged_bc.train` (unchanged) builds the net from `model.factors`; the checkpoint carries the factor list and the
    `versions["factors"]` stamp, and `load_bc` rebuilds the extended net from the config alone."""
    from rrp.harness.train import legged_bc as TB
    from rrp.policies.nets.legged_bc import load_bc
    from rrp.policies.relations.base import compat_hash
    _write_shard(tmp_path / "d")
    cfg = dict(name="hlbc", data=str(tmp_path / "d"), bodies=["tinybody"], steps=2, batch_size=8, ckpt_every=1000,
               snap_every=10 ** 6, model=dict(width=D, enc_layers=1, dec_layers=1, factors=["preset:legged"]))
    res = TB.train(cfg, tmp_path / "out")
    st = torch.load(tmp_path / "out" / "policy.pt", weights_only=False)
    assert [f["name"] for f in st["factors"]][-2:] == ["leg.foothold", "leg.com_support"]
    assert st["_provenance"]["versions"]["factors"].startswith(compat_hash(LL.legged_specs(cfg["model"]["factors"])))
    m, _ = load_bc(tmp_path / "out" / "policy.pt", torch.device("cpu"))
    assert m.extended and "chunk_mse" in res["eval"]
