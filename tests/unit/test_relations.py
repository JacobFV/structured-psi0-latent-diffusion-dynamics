"""Relation-factor registry (D-144): resolution, hashing, legacy config mapping, operator math (PaPE / rotation /
equality are exact identities of the q/k augmentation), zero-init equivalence, RNG isolation, deploy guard."""
import math

import pytest
import torch

from rrp.policies.nets.attention import MHA
from rrp.policies.relations import base as RB
from rrp.policies.relations.base import (Algebra, EdgeSet, FactorDef, FactorError, FactorSpec, PrivilegedInput, RelCtx,
                                         TokenSet, compat_hash, register_factor, resolve)
from rrp.policies.relations.catalog import ARM_REL_VOCAB
from rrp.policies.relations.ops import FactorSite

register_factor(FactorDef("test.pape", "1", field="pos3d", op="sqdiff+diff", form="aug", sources=("given", "gt"),
                          params=(("p", 3),)))
register_factor(FactorDef("test.pape_q", "1", field="pos3d", op="sqdiff+diff", form="aug",
                          params=(("p", 3), ("frame", "query"), ("orient", "orient"))))
register_factor(FactorDef("test.rot", "1", field="orient", op="rel_rot", form="aug"))
register_factor(FactorDef("test.same", "1", field="entity_id", op="same", form="aug", params=(("n_ids", 8),)))
register_factor(FactorDef("test.bil", "1", field="hidden", op="bilinear", form="aug", params=(("rank", 4),)))
register_factor(FactorDef("test.anc", "1", field="edges:*", op="ancestor", form="bias", params=(("edge", "kin_parent"),)))
register_factor(FactorDef("test.planned", "1", field="pos3d", op="diff", form="aug", status="planned"))


def test_resolve_presets_globs_overrides():
    arm = resolve(None, default="arm")
    assert [s.name for s in arm] == [f"edge.{n}" for n in ARM_REL_VOCAB] + ["msg.incidence"]
    off = resolve(["preset:arm", {"name": "edge.*", "control": "rewired"}, "id.slot_handle"])
    assert {s.control for s in off if s.name.startswith("edge.")} == {"rewired"} and off[-1].name == "id.slot_handle"
    assert [s.name for s in off][:17] == [s.name for s in arm][:17]          # order = parameter-row order
    with pytest.raises(FactorError):
        resolve(["edge.nope"])
    with pytest.raises(FactorError):
        resolve(["test.planned"])
    with pytest.raises(FactorError):
        resolve([{"name": "edge.kin_parent", "control": "serialized"}])
    with pytest.raises(FactorError):
        resolve([{"name": "edge.kin_parent", "gate": "task"}])                  # gate not allowed by the entry


def test_compat_hash_ignores_controls_but_not_structure():
    a = compat_hash(resolve(None, default="arm"))
    assert a == compat_hash(resolve(["preset:arm", {"name": "edge.*", "control": "zero"}]))
    assert a != compat_hash(resolve(["preset:arm", "id.slot_handle"]))
    assert a != compat_hash(resolve(["preset:arm", {"name": "edge.kin_parent", "heads": [0]}]))


def test_legacy_config_mapping():
    from rrp.policies.nets.flow import PolicyConfig
    assert PolicyConfig.from_dict(dict(width=8, structured=True, bias_mode="true")).factors is None
    c = PolicyConfig.from_dict(dict(structured=False, bias_mode="none", slot_handles=True))
    ctl = {s.name: s.control for s in c.specs()}
    assert ctl["edge.kin_parent"] == "off" and ctl["msg.incidence"] == "serialized" and "id.slot_handle" in ctl


def _sets(B=2, Q=5, K=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    def rot():
        q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g))
        return q * torch.sign(torch.det(q))
    tq = TokenSet("q", torch.ones(B, Q, dtype=torch.bool),
                  fields={"pos3d": torch.randn(B, Q, 3, generator=g),
                          "orient": torch.stack([torch.stack([rot() for _ in range(Q)]) for _ in range(B)]).view(B, Q, 9),
                          "entity_id": torch.randint(0, 8, (B, Q, 1), generator=g)})
    tk = TokenSet("k", torch.ones(B, K, dtype=torch.bool),
                  fields={"pos3d": torch.randn(B, K, 3, generator=g),
                          "orient": torch.stack([torch.stack([rot() for _ in range(K)]) for _ in range(B)]).view(B, K, 9),
                          "entity_id": torch.randint(0, 8, (B, K, 1), generator=g)})
    return RelCtx(sets={"q": tq, "k": tk})


def _randomize(site, seed=3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in site.parameters():
            p.copy_(torch.randn(p.shape, generator=g))


def test_pape_augmentation_is_eq9_up_to_query_constants():
    rc = _sets()
    site = FactorSite(2, 16, "q>k", resolve(["test.pape"]), ("pos3d",))
    _randomize(site)
    x = torch.randn(2, 5, 16)
    qa, ka = site.augment(rc, x)
    logit = qa @ ka.transpose(-1, -2)                                         # [B,H,Q,K]
    m = site.f["test__pape"]
    a = m.g[None, :, None, None] * torch.nn.functional.softplus(m.a(x))
    b = m.b(x)
    rq, rk = rc.sets["q"].fields["pos3d"], rc.sets["k"].fields["pos3d"]
    u = torch.einsum("hmp,bkqp->bhkqm", m.Wp, rk[:, :, None] - rq[:, None])   # [B,H,K,Q,m]: W_p (r_j - r_i)
    ref = (-(a[:, :, None] * u ** 2) + b[:, :, None] * u).sum(-1).transpose(2, 3)
    d = logit - ref                                                           # must be constant over keys j
    assert torch.allclose(d, d[..., :1].expand_as(d), atol=1e-4)
    assert torch.allclose(logit.softmax(-1), ref.softmax(-1), atol=1e-5)


def test_pape_query_frame_is_rotation_invariant():
    rc = _sets()
    site = FactorSite(2, 16, "q>k", resolve(["test.pape_q"]), ("pos3d",))
    _randomize(site)
    x = torch.randn(2, 5, 16)
    qa, ka = site.augment(rc, x)
    R, _ = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(9)))
    t = torch.tensor([0.3, -1.0, 2.0])
    for n in ("q", "k"):
        ts = rc.sets[n]
        ts.fields["pos3d"] = ts.fields["pos3d"] @ R.T + t
        ts.fields["orient"] = (R @ ts.fields["orient"].view(*ts.fields["orient"].shape[:2], 3, 3)).flatten(-2)
    qa2, ka2 = site.augment(rc, x)
    assert torch.allclose((qa @ ka.transpose(-1, -2)).softmax(-1), (qa2 @ ka2.transpose(-1, -2)).softmax(-1), atol=1e-4)


def test_rel_rot_and_same_are_exact():
    rc = _sets()
    site = FactorSite(2, 16, "q>k", resolve(["test.rot", "test.same"]), ("orient", "entity_id"))
    _randomize(site)
    x = torch.randn(2, 5, 16)
    qa, ka = site.augment(rc, x)
    got = qa @ ka.transpose(-1, -2)
    Bq = site.f["test__rot"].B(x).view(2, 2, 5, 3, 3)
    Rq = rc.sets["q"].fields["orient"].view(2, 5, 3, 3)
    Rk = rc.sets["k"].fields["orient"].view(2, 6, 3, 3)
    rot = torch.einsum("bhqst,bqus,bkut->bhqk", Bq, Rq, Rk)                   # <B_i, R_i^T R_j>_F
    iq, ik = rc.sets["q"].fields["entity_id"][..., 0], rc.sets["k"].fields["entity_id"][..., 0]
    same = site.f["test__same"].g[None, :, None, None] * (iq[:, :, None] == ik[:, None, :]).float()[:, None]
    assert torch.allclose(got, rot + same, atol=1e-4)


def test_mha_augmentation_matches_explicit_logits():
    torch.manual_seed(0)
    att = MHA(16, 2)
    x = torch.randn(2, 5, 16)
    qa, ka = torch.randn(2, 2, 5, 4), torch.randn(2, 2, 5, 4)
    bias = torch.randn(2, 2, 5, 5)
    y = att(x, bias=bias, q_aug=qa, k_aug=ka)
    y2, w = att(x, bias=bias, q_aug=qa, k_aug=ka, need_weights=True)
    q = att.q(x).view(2, 5, 2, 8).transpose(1, 2)
    k, _ = att.kv(x)
    ref = (q @ k.transpose(-1, -2) / math.sqrt(8) + qa @ ka.transpose(-1, -2) + bias).softmax(-1)
    assert torch.allclose(w, ref, atol=1e-5) and torch.allclose(y, y2, atol=1e-5)


def test_new_factor_is_zero_init_and_rng_isolated():
    from rrp.policies.nets.batch import collate_inputs
    from rrp.policies.nets.flow import FlowPolicy, PolicyConfig
    pytest.importorskip("mujoco")
    from rrp.envs.mujoco.fixtures import make_pick_place_session
    from rrp.policies.features.featurizer import featurizer_for
    s = make_pick_place_session(seed=3)
    b = collate_inputs([featurizer_for(s)(s.observe())])
    torch.manual_seed(0)
    m0 = FlowPolicy(PolicyConfig(width=32, heads=2, ctx_layers=2, blocks=1, horizon=4, latent_dim=2)).eval()
    torch.manual_seed(0)
    m1 = FlowPolicy(PolicyConfig(width=32, heads=2, ctx_layers=2, blocks=1, horizon=4, latent_dim=2,
                                 factors=["preset:arm", "test.bil"])).eval()
    sd1 = m1.state_dict()
    assert all(torch.equal(v, sd1[k]) for k, v in m0.state_dict().items())   # same init for every shared parameter
    assert any("test__bil" in k for k in sd1)
    z = torch.randn(1, 4, b.node_feats.shape[1], 2)
    with torch.no_grad():
        v0 = m0.velocity(z, torch.tensor([0.4]), m0.prepare(b))
        v1 = m1.velocity(z, torch.tensor([0.4]), m1.prepare(b))
    assert torch.equal(v0, v1)                                                # zero-bias equivalence at init


def test_closure_op_and_edge_controls():
    A = torch.zeros(1, 4, 4, 1, dtype=torch.bool)
    A[0, 1, 0, 0] = A[0, 2, 1, 0] = A[0, 3, 2, 0] = True                      # chain 3 -> 2 -> 1 -> 0 (child -> parent)
    rc = RelCtx(sets={"n": TokenSet("n", torch.ones(1, 4, dtype=torch.bool))},
                edges={"n>n": EdgeSet(("kin_parent",), A)})
    site = FactorSite(1, 8, "n>n", resolve(["test.anc"]), ("edges:arm-rel-v1",))
    with torch.no_grad():
        site.f["test__anc"].w.fill_(1.0)
    b = site.bias(rc)[0, 0]
    assert b[3].tolist() == [1, 1, 1, 0] and b[0].tolist() == [0, 0, 0, 0]


def test_deploy_guard():
    specs = resolve(["preset:arm", {"name": "test.pape", "control": "gt"}])
    with pytest.raises(PrivilegedInput):
        RB.assert_deployable(specs)
    RB.assert_deployable(resolve(None, default="arm"))
    rc = _sets()
    rc.sets["q"].labels["pos3d"] = rc.sets["q"].fields["pos3d"]
    rc.sets["k"].labels["pos3d"] = rc.sets["k"].fields["pos3d"]
    site = FactorSite(2, 16, "q>k", resolve([{"name": "test.pape", "control": "gt"}]), ("pos3d",))
    site.augment(rc, torch.randn(2, 5, 16))                                   # training: allowed
    rc.deploy = True
    with pytest.raises(PrivilegedInput):
        site.augment(rc, torch.randn(2, 5, 16))


# ------------------------------------------------------------------ R15: membership / graph (id.*, kin.*)
# catalog.py §graph: id.same_body, id.same_assembly, kin.ancestor, kin.sibling, kin.mirror. Section is disjoint from
# the rest of this file (only new functions below); no change to `base.py` / `ops.py` was needed.
def _undirected_hop_distance(parent: dict, n: int, k: int) -> "torch.Tensor":
    """[n, n] bool, distance(i, j) == k in the undirected graph of a `parent` map (child -> parent, -1 = root).
    Reference BFS used to check `HopOp` independently of its implementation."""
    import collections
    adj = collections.defaultdict(set)
    for c, p in parent.items():
        if p is not None and p >= 0:
            adj[c].add(p)
            adj[p].add(c)
    out = torch.zeros(n, n, dtype=torch.bool)
    for src in range(n):
        dist = {src: 0}
        q = collections.deque([src])
        while q:
            u = q.popleft()
            if dist[u] == k:
                continue
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        for j, d in dist.items():
            if d == k and j != src:
                out[src, j] = True
    return out


def _ancestor_closure(parent: dict, n: int) -> "torch.Tensor":
    """[n, n] bool, j is an ancestor of i (any number of >=1 hops up `parent`). Reference used to check `ClosureOp`."""
    out = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n):
        j = parent.get(i, -1)
        seen = set()
        while j is not None and j >= 0 and j not in seen:
            out[i, j] = True
            seen.add(j)
            j = parent.get(j, -1)
    return out


def test_graph_factors_registered_and_resolve_on_arm_psi0_legged():
    from rrp.policies.relations.base import get_factor
    for name in ("id.same_body", "id.same_assembly", "kin.ancestor", "kin.sibling"):
        d = get_factor(name)
        assert d.status == "implemented"
    # arm: the arm preset's edges (kin_parent lives in arm-rel-v1) plus every graph factor.
    arm = resolve(["preset:arm", "preset:graph"])
    assert {"id.same_body", "id.same_assembly", "kin.ancestor", "kin.sibling"} <= {s.name for s in arm}
    # Ψ₀: the psi0-dims preset's edges (g1-dim-rel-v1 also carries kin_parent) plus every graph factor.
    psi0 = resolve(["preset:psi0-dims", "preset:graph"])
    assert {"id.same_body", "id.same_assembly", "kin.ancestor", "kin.sibling"} <= {s.name for s in psi0}
    # legged: no edge vocabulary at all (docs/relations.md §2), only per-token assembly_id / entity_id fields.
    legged = resolve(["preset:graph"])
    assert {s.name for s in legged} == {"id.same_body", "id.same_assembly", "kin.ancestor", "kin.sibling"}
    site = FactorSite(1, 8, "n>n", legged, ("assembly_id", "entity_id"))
    assert {s.name for s in site.specs} == {"id.same_body", "id.same_assembly"}   # edge factors inert here


def test_kin_ancestor_and_sibling_on_arm_morphology_fixture():
    """Small synthetic arm-style morph tree over `edges:arm-rel-v1`'s `kin_parent` channel (0 root; 1, 2 children of
    0; 3, 4 children of 1; 5, 6 children of 2): `kin_parent[i, j] = 1` iff j is i's direct parent."""
    from rrp.policies.relations.base import get_factor, spec
    from rrp.policies.relations.ops import OPS
    n = 7
    parent = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    A = torch.zeros(1, n, n, 1, dtype=torch.bool)
    for c, p in parent.items():
        A[0, c, p, 0] = True
    rc = RelCtx(sets={"n": TokenSet("n", torch.ones(1, n, dtype=torch.bool))},
               edges={"n>n": EdgeSet(("kin_parent",), A)})

    anc = OPS["ancestor"].value(get_factor("kin.ancestor"), spec("kin.ancestor"), None, rc, "n>n")
    assert torch.equal(anc[0], _ancestor_closure(parent, n))
    assert anc[0, 3].tolist() == [True, True, False, False, False, False, False]      # 3's ancestors: 1, 0
    assert anc[0].sum() == sum(len(_ancestor_closure(parent, n)[i].nonzero()) for i in range(n))  # sanity, not vacuous

    sib = OPS["hop"].value(get_factor("kin.sibling"), spec("kin.sibling"), None, rc, "n>n")
    expect = _undirected_hop_distance(parent, n, 2)
    assert torch.equal(sib[0], expect)
    # exact pairs at undirected distance 2: the 3 true sibling pairs plus the 4 grandparent<->grandchild pairs
    # through the root (kin.sibling's doc says both occur; hops=2 does not disambiguate them)
    got_pairs = {(i, j) for i in range(n) for j in range(n) if sib[0, i, j]}
    assert got_pairs == {(1, 2), (2, 1), (3, 4), (4, 3), (5, 6), (6, 5),
                         (0, 3), (3, 0), (0, 4), (4, 0), (0, 5), (5, 0), (0, 6), (6, 0)}


def test_kin_ancestor_sibling_and_same_assembly_on_g1_morphology_fixture():
    """The real G1 morphology (`rrp.bodies.g1_simple`, D-098): 36 command-dim tokens, real kinematic parent chain and
    tree. `kin.ancestor` / `kin.sibling` are checked against independent BFS references; `id.same_assembly`
    against the body module's own `same_assembly` edge channel. (`kin.mirror` was deleted in round 2: the mirror
    pairing is the `mirror` edge channel of g1-dim-rel-v1, with no field factor over it.)"""
    from rrp.bodies import g1_simple as G
    from rrp.policies.relations.base import get_factor, spec
    from rrp.policies.relations.ops import OPS
    n = G.ACTION_DIM
    parent = {i: int(G.DIM_PARENT[i]) for i in range(n)}
    rel = torch.from_numpy(G.relation_matrix()).permute(1, 2, 0).unsqueeze(0)         # [1, T, T, R]
    asm = torch.from_numpy(G.DIM_ASM).view(1, n, 1)
    ts = TokenSet("dims", torch.ones(1, n, dtype=torch.bool),
                 fields={"assembly_id": asm})
    rc = RelCtx(sets={"dims": ts}, edges={"dims>dims": EdgeSet(G.RELATIONS, rel)})

    anc = OPS["ancestor"].value(get_factor("kin.ancestor"), spec("kin.ancestor"), None, rc, "dims>dims")[0]
    assert torch.equal(anc, _ancestor_closure(parent, n))
    assert bool(anc.any(-1).sum()) and int(anc.sum(-1).max()) == int(G.DIM_DEPTH.max())   # root-to-leaf depth check

    sib = OPS["hop"].value(get_factor("kin.sibling"), spec("kin.sibling"), None, rc, "dims>dims")[0]
    assert torch.equal(sib, _undirected_hop_distance(parent, n, 2))
    # concrete real sibling: l_thumb0 / l_middle0 / l_index0 (dims 0, 3, 5) share the immediate parent l_wrist_yaw (20)
    for i, j in ((0, 3), (0, 5), (3, 5)):
        assert bool(sib[i, j]) and bool(sib[j, i])

    same_asm = OPS["same"].value(get_factor("id.same_assembly"), spec("id.same_assembly"), None, rc, "dims>dims")[0]
    assert torch.equal(same_asm, torch.from_numpy(G.relation_matrix()[G.RELATIONS.index("same_assembly")]))


def test_bilinear_pair_estimate_feeds_graph_factor():
    """ix.support's pair estimate (bilinear, params.emits) -> estimated edges:support-v1 -> ix.force_flow closure;
    deployable with source probe, refused in deploy mode with source gt."""
    A = torch.tensor([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.float32)       # 0 supports 1 supports 2
    x = torch.eye(3)[None]                                                            # one-hot token hiddens [1,3,3]

    def run(specs, deploy=False, gt=None):
        rc = RelCtx(sets={"ctx": TokenSet("ctx", torch.ones(1, 3, dtype=torch.bool))}, deploy=deploy)
        if gt is not None:
            rc.edges["ctx>ctx#support-v1@gt"] = EdgeSet(("support",), gt[None, :, :, None], prov="privileged")
        site = FactorSite(1, 3, "ctx>ctx", resolve(specs), ("hidden",))
        assert [s.name for s in site.specs] == ["ix.support", "ix.force_flow"]     # force_flow applies via emits
        with torch.no_grad():
            m = site.f["ix__support"]
            m.U.zero_()
            m.U[:3] = 10 * torch.eye(3)                                               # u_i = 10 e_i (rank 8, 3 used)
            m.V.zero_()
            m.V[:3] = A - 0.5                                                         # logit_ij = 10 (A_ij - 1/2)
            site.f["ix__force_flow"].w.fill_(1.0)
        site.augment(rc, x, x)
        return rc, site.bias(rc)[0, 0]

    rc, b = run(["ix.support", "ix.force_flow"])
    assert torch.allclose(torch.sigmoid(rc.estimates[("pair", "ix.support")][0]) > 0.5, A.bool())
    assert rc.edges["ctx>ctx#support-v1"].prov == "estimated"
    assert b.tolist() == [[0, 1, 1], [0, 0, 1], [0, 0, 0]]                          # closure: 0 -> 2 upstream
    _, b_dep = run(["ix.support", "ix.force_flow"], deploy=True)                     # estimated: deployable
    assert torch.equal(b, b_dep)
    gt_graph = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=torch.float32)
    _, b_gt = run(["ix.support", {"name": "ix.force_flow", "source": "gt"}], gt=gt_graph)
    assert b_gt.tolist() == [[0, 0, 0], [1, 0, 0], [1, 1, 0]]
    with pytest.raises(PrivilegedInput):
        run(["ix.support", {"name": "ix.force_flow", "source": "gt"}], deploy=True, gt=gt_graph)
    with pytest.raises(PrivilegedInput):
        RB.assert_deployable(resolve(["ix.support", {"name": "ix.force_flow", "source": "gt"}]))
