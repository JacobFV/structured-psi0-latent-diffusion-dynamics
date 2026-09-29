"""Ψ₀ line (W10): invariants where a silent bug would invalidate results, and goldens of the nets moved from psi1z (D-140).

The golden digests below were recorded by running the ORIGINAL psi1z code (6f5e2b3, on rrp 68a6657, its pin) through
`_goldens()`; the migrated `rrp.policies.psi0.nets` must reproduce them byte for byte. Random-weight nets: plumbing only,
no number here is a result. Net tests skip without torch; the SIMPLE worker plumbing needs neither torch nor Isaac."""
import hashlib
import inspect
import json
import math

import numpy as np
import pytest

from rrp.bodies import g1_simple as G
from rrp.envs.simple import worker as W

try:
    import torch
    from rrp.policies.psi0 import nets as N
except ImportError:          # the unit suite must pass without the ml extra
    torch = N = None
torch_only = pytest.mark.skipif(torch is None, reason="needs torch")

GOLDEN = {'context_tokens': '94f6347ff56ce972:93d4a1923daa17ff',
 'probe_meta_only': 'c00936f5b8223d4f:13fcd9a0e80173b6',
 'probe_metrics_grasp0': '{"active_hand_acc": [1.0, 3], "base_cmd_vx_mae_norm": [22.50524, 20], "base_disp_xy_err": '
                         '[9.25428, 4], "contact_acc": [23.0, 40], "contact_pos_recall": [23.0, 23], '
                         '"hand_dist_mae": [15.51641, 40], "lift_acc": [10.0, 20], "target_pos_err": [41.11239, 20]}',
 'probe_metrics_grasp1': '{"active_hand_acc": [1.0, 3], "base_cmd_vx_mae_norm": [22.50524, 20], "base_disp_xy_err": '
                         '[9.25428, 4], "contact_acc": [23.0, 40], "contact_pos_recall": [23.0, 23], '
                         '"grasp_face_acc": [3.0, 4], "grasp_pt_err": [4.9498, 4], "hand_dist_mae": [15.51641, 40], '
                         '"lift_acc": [10.0, 20], "target_pos_err": [41.11239, 20]}',
 'read_mask': 'ea5d4d70a39de316',
 'stageA_fwd_grasp0': '844ac41c0bc775c2',
 'stageA_fwd_grasp1': 'b2922bec6758aef3',
 'stageA_init_grasp0': 'aec3c13e3f2fcc5c',
 'stageA_init_grasp1': 'f7496f7a6c754491',
 'stageA_loss_grasp0': '22431ef1b26a021a',
 'stageA_loss_grasp1': 'a4ec096fba9995c7',
 'tables': 'c77392ff60a56f90'}


def _dg(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        a = a.detach().float().numpy() if torch.is_tensor(a) else np.asarray(a)
        h.update(np.ascontiguousarray(a).tobytes()); h.update(str(a.shape).encode())
    return h.hexdigest()[:16]


def _sd(m):
    return _dg(*[v for _, v in sorted(m.state_dict().items())])


def _batch(B=4, g=0):
    torch.manual_seed(100 + g)
    lab = dict(hand_dist=torch.rand(B, N.K, 2), contact=torch.randint(0, 2, (B, N.K, 2)), lift=torch.randint(0, 2, (B, N.K)),
               target_pos=torch.randn(B, N.K, 3), active_hand=torch.tensor([0, 1, -1, 1])[:B], base_disp=torch.randn(B, 3),
               grasp_pt=torch.randn(B, 2, 3), grasp_face=torch.randint(0, 6, (B, 2)),
               grasp_valid=torch.tensor([[True, False], [False, True], [False, False], [True, True]])[:B])
    return dict(state0=torch.randn(B, 36), actions=torch.randn(B, N.TP, N.DA), amask=torch.ones(B, N.TP, N.DA),
                j=torch.tensor([0, 3, 7, 1])[:B], state_j=torch.randn(B, 36), labels=lab)


def _goldens() -> dict:
    out = {"tables": _dg(G.node_static(), G.relation_matrix(), G.asm_static(), G.SIMPLE_QPOS_INDEX, G.DIM_MIRROR, G.DIM_DEPTH),
           "read_mask": _dg(N.read_mask())}
    for grasp in (False, True):
        torch.manual_seed(0); A = N.StageA(grasp=grasp)
        out[f"stageA_init_grasp{int(grasp)}"] = _sd(A)
        b = _batch(); torch.manual_seed(1)
        loss, _, _ = A.loss(b, w_kl=1e-4, w_sem=0.1, z_noise=0.0, w_grasp=float(grasp))
        out[f"stageA_loss_grasp{int(grasp)}"] = _dg(loss)
        with torch.no_grad():
            mu, lv = A.E(A.morph, b["state0"], b["actions"])
            r = A.R(A.morph, mu, b["state_j"], b["j"].float() / N.TP)
            p = A.P(mu)
        out[f"stageA_fwd_grasp{int(grasp)}"] = _dg(mu, lv, r, *[p[k] for k in sorted(p)])
        out[f"probe_metrics_grasp{int(grasp)}"] = json.dumps(
            {k: [round(float(s), 5), n] for k, (s, n) in sorted(N.probe_metrics(p, b["labels"]).items())})
    torch.manual_seed(0); P = N.PacketProbe(metadata_only=True)
    out["probe_meta_only"] = _sd(P) + ":" + _dg(*[v for _, v in sorted(P(torch.randn(2, N.K, N.M, N.DZ)).items())])
    torch.manual_seed(0); C = N.ContextTokens()
    torch.manual_seed(5)
    hid = torch.randn(2, 7, 2048).to(torch.bfloat16); ent = torch.zeros(2, 7, dtype=torch.bool); ent[0, 3:5] = True
    with torch.no_grad():
        v, m = C(dict(hidden=hid, mask=torch.ones(2, 7, dtype=torch.bool), ent=ent, state0=torch.randn(2, 36)))
    out["context_tokens"] = _sd(C) + ":" + _dg(v, m)
    return out


@torch_only
def test_nets_match_psi1z_goldens():
    got = _goldens()
    assert {k: got[k] for k in GOLDEN} == GOLDEN


def test_action_layout_matches_upstream_agent():
    """Our 36-dim layout must match SIMPLE's psi0 agents: STATE_SLICES order (thumb 29:32, middle 34:36, index 32:34,
    right hand 36:43, left arm 15:22, right arm 22:29) and waist roll/pitch/yaw = qpos 13, 14, 12."""
    upstream = list(range(29, 32)) + list(range(34, 36)) + list(range(32, 34)) + list(range(36, 43)) \
        + list(range(15, 22)) + list(range(22, 29)) + [13, 14, 12]
    assert G.SIMPLE_QPOS_INDEX.tolist() == upstream
    assert [i for s, e in W.STATE_SLICES for i in range(s, e)] == upstream[:28]      # the worker's state construction
    assert G.ACTION_DIM == 36 and G.STATE_DIM == 32
    assert [G.DIM_NAMES[i] for i in (31, 32, 35)] == ["base_height", "base_vx", "base_target_yaw"]


@torch_only
def test_realizer_direct_reads_follow_reads_table():
    """System 0 cross-attention mask: command dim i may read knot (k, m) iff assembly m is in READS[asm(i)]."""
    rm = N.Morph().read_mask.reshape(N.DA, N.K, N.M)
    for i in range(N.DA):
        allowed = {G.ASM_INDEX[a] for a in N.READS[G.ASSEMBLIES[G.DIM_ASM[i]]]}
        for mm in range(N.M):
            assert bool(rm[i, :, mm].all()) == (mm in allowed) and bool(rm[i, :, mm].any()) == (mm in allowed)


@torch_only
def test_realizer_single_block_without_dim_mixing_is_local():
    """With dim-to-dim self-attention removed, a left-hand dim gets zero gradient from right-hand / right-arm / base /
    torso knots: the only cross-assembly path is the dim self-attention (a routing prior, not an information barrier)."""
    torch.manual_seed(0)
    m = N.Morph(); R = N.Realizer(D=64, layers=1)
    for L in R.blocks:
        L["s"].forward = lambda x, **k: torch.zeros_like(x)
    R.dims.layers = torch.nn.ModuleList()
    z = torch.randn(1, N.K, N.M, N.DZ, requires_grad=True)
    R(m, z, torch.randn(1, 36), torch.zeros(1))[0, :, 0].sum().backward()
    g = z.grad.abs().sum((0, 1, 3))
    for far in ("right_hand", "right_arm", "base", "torso"):
        assert g[G.ASM_INDEX[far]] == 0, far
    assert g[G.ASM_INDEX["left_hand"]] > 0


@torch_only
def test_realizer_inputs_exclude_task_and_actions():
    """System 0 may see only z, morphology, current state and phase (no actions, labels, images or instruction)."""
    assert list(inspect.signature(N.Realizer.forward).parameters) == ["self", "morph", "z", "state", "phase"]


@torch_only
def test_probe_nll_bounded_below():
    """Bounded probe NLL (D-085): log-variance floored at -4, so the loss cannot run to -inf on a perfect mean."""
    B = 4
    out = N.PacketProbe(D=32)(torch.randn(B, N.K, N.M, N.DZ))
    lab = dict(hand_dist=out["hand_dist"][..., 0].detach(), contact=torch.zeros(B, N.K, 2), lift=torch.zeros(B, N.K),
               target_pos=out["target_pos"][..., :3].detach(), active_hand=torch.zeros(B, dtype=torch.long),
               base_disp=out["base_disp"][:, :3].detach(), base_cmd=out["base_cmd"][..., :2].detach())
    for k in ("hand_dist", "target_pos", "base_disp", "base_cmd"):
        out[k] = out[k].detach().clone(); d = out[k].shape[-1] // 2; out[k][..., d:] = -100.0
    total, logs = N.probe_loss(out, lab, lv_min=-4.0)
    assert np.isfinite(float(total)) and logs["probe_target_pos"] >= 3 * 0.5 * (-4.0 + np.log(2 * np.pi)) - 1e-4


@torch_only
def test_cmd_labels_come_from_targets_only():
    """base_cmd labels are the demonstrated (vx, vyaw) at knot steps: labels only, derived from b['actions']."""
    a = torch.randn(2, N.TP, N.DA)
    b = N.add_cmd_labels(dict(actions=a, labels={}))
    assert torch.equal(b["labels"]["base_cmd"], a[:, list(N.KNOT_STEPS)][..., list(N.CMD_DIMS)])


@torch_only
def test_grasp_head_off_by_default_and_loss_only_when_weighted():
    torch.manual_seed(0); a = N.StageA(D=32, probe_D=32)
    torch.manual_seed(0); b = N.StageA(D=32, probe_D=32, grasp=False)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa) and not any("grasp" in k for k in sa)
    torch.manual_seed(0); P = N.PacketProbe(D=32, grasp=True)
    out = P(torch.randn(3, N.K, N.M, N.DZ))
    assert out["grasp"].shape == (3, 2, 6 + N.N_FACES)
    lab = {k: v[:3] for k, v in _batch()["labels"].items()}
    assert "probe_grasp_pt" not in N.probe_loss(out, lab, w_grasp=0.0)[1]
    assert {"probe_grasp_pt", "probe_grasp_face"} <= set(N.probe_loss(out, lab, w_grasp=1.0)[1])
    assert N.probe_metrics(out, lab)["grasp_face_acc"][1] == 2          # only the valid (sample, hand) pairs are scored


@torch_only
def test_load_stage_a_infers_grasp_head(tmp_path):
    for g in (False, True):
        torch.manual_seed(1); A = N.StageA(grasp=g)
        torch.save(dict(model=A.state_dict()), tmp_path / "a.pt")
        B = N.load_stage_a(tmp_path / "a.pt")
        sa, sb = A.state_dict(), B.state_dict()
        assert B.P.grasp == g and sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_worker_state32_matches_body_layout():
    q = np.arange(43, dtype=np.float32) / 10
    s = W.state32(q, [0.1, 0.2, 0.3, 0.74])
    assert s.shape == (32,) and np.allclose(s[:28], q[G.SIMPLE_QPOS_INDEX[:28]]) and np.allclose(s[28:], [0.1, 0.2, 0.3, 0.74])


def test_chunk_client_asks_once_and_checks_the_state_shown_to_the_policy():
    c = W.ChunkClient()
    st = {"states": np.ones((1, 32), np.float32)}
    with pytest.raises(W.NeedChunk):
        c.query_action({}, "i", st)
    c.rows, c.expect = np.zeros((24, 36), np.float32), np.ones(32, np.float32)
    assert c.query_action({}, "i", st)[0].shape == (24, 36) and c.rows is None and c.queries == 1
    c.rows, c.expect = np.zeros((24, 36), np.float32), np.zeros(32, np.float32)
    with pytest.raises(RuntimeError, match="state shown to the policy"):
        c.query_action({}, "i", st)


class _FakeWorker:
    """Test double for the SIMPLE backend: same request surface, no simulator."""

    def __init__(self, **kw):
        self.kw, self.n = kw, 0

    def init_info(self):
        return dict(self.kw, joint_names=[f"j{i}" for i in range(43)])

    def reset(self, episode, split="eval"):
        self.n = 0
        return dict(step=0, episode=episode)

    def step(self, rows=None):
        if self.n == 0 and rows is None:
            raise W.NeedChunk()
        self.n += 1
        return dict(step=self.n, rows=None if rows is None else np.asarray(rows).shape)

    def truth(self):
        return dict(source="privileged:sim", step=self.n)

    def close(self):
        pass


def test_worker_rpc_roundtrip_and_explicit_errors():
    import threading
    from multiprocessing.connection import Client, Listener
    key = b"k" * 32
    with Listener(("127.0.0.1", 0), authkey=key) as L:
        th = threading.Thread(target=lambda: W.serve(Client(L.address, authkey=key), _FakeWorker), daemon=True)
        th.start()
        conn = L.accept()

        def call(m, **kw):
            conn.send((m, kw)); return conn.recv()
        assert call("init", task="T")[1]["task"] == "T"
        assert call("reset", episode=3) == ("ok", dict(step=0, episode=3))
        assert call("step")[:2] == ("err", "chunk_required")                         # never a silent hold
        assert call("step", rows=np.zeros((24, 36)))[1] == dict(step=1, rows=(24, 36))
        assert call("truth")[1]["source"] == "privileged:sim"
        assert call("nope")[1] == "ValueError"
        assert call("close") == ("ok", None)
        th.join(5)


def test_levels_validated():
    with pytest.raises(ValueError, match="level"):
        W.Worker("G1WholebodyTabletopGraspMP-v0", level=3)
