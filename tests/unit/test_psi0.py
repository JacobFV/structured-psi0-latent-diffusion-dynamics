"""Ψ₀ line (W10): invariants where a silent bug would invalidate results, and goldens of the nets moved from psi1z (D-140).

The golden digests below were recorded by running the ORIGINAL psi1z code (6f5e2b3, on rrp 68a6657, its pin) through
`_goldens()`; the migrated `rrp.policies.psi0.nets` must reproduce them byte for byte. Random-weight nets: plumbing only,
no number here is a result. Net tests skip without torch; the SIMPLE worker plumbing needs neither torch nor Isaac.

D-144 R5 (Ψ₀ on the relation-factor foundation) moved the Realizer's READS mask to the `route.assembly_reads` factor
and replaced `PacketProbe` with `ReadoutProbe` (`probes:psi0-v1`): `GOLDEN` below keeps exactly the psi1z-original
entries that this migration does not touch (`tables`, `read_mask`, `context_tokens` — the morphology tables, the
legacy mask helper's own output, and `ContextTokens`, none of which changed); the probe-dependent entries
(`stageA_*`, `probe_*`) are superseded by `GOLDEN_R5`, a fresh regression baseline recorded on this migrated code
(not psi1z), and the Realizer's numerical equivalence is checked directly (`test_route_assembly_reads_matches_read_mask`,
`test_realizer_output_matches_pre_r5_formula`) rather than folded into a combined probe+realizer digest."""
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
    from rrp.policies.relations.base import RelCtx, TokenSet
except ImportError:          # the unit suite must pass without the ml extra
    torch = N = None
torch_only = pytest.mark.skipif(torch is None, reason="needs torch")

GOLDEN = {'context_tokens': '94f6347ff56ce972:93d4a1923daa17ff', 'read_mask': 'ea5d4d70a39de316',
         'tables': 'c77392ff60a56f90'}

# Recorded once on this migration (D-144 R5); a change here must be explained by an intentional probe/StageA change,
# not by an accidental one (rebuild via `python -c "from tests.unit.test_psi0 import _goldens_r5; print(_goldens_r5())"`).
GOLDEN_R5 = {'probe_metrics_grasp0': '{"active_hand_acc": [1.0, 3], "base_cmd_mae": [49.12035, 40], '
                                     '"base_disp_mae": [13.06765, 12], "contact_acc": [17.0, 40], '
                                     '"hand_dist_mae": [45.98371, 40], "lift_acc": [10.0, 20], '
                                     '"target_pos_mae": [62.74125, 60]}',
            'probe_metrics_grasp1': '{"active_hand_acc": [1.0, 3], "base_cmd_mae": [30.73586, 40], '
                                    '"base_disp_mae": [11.98346, 12], "contact_acc": [17.0, 40], '
                                    '"grasp_face_acc": [3.0, 4], "grasp_pt_err": [4.10538, 4], '
                                    '"hand_dist_mae": [11.72864, 40], "lift_acc": [10.0, 20], '
                                    '"target_pos_mae": [57.10979, 60]}'}


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
    torch.manual_seed(0); C = N.ContextTokens()
    torch.manual_seed(5)
    hid = torch.randn(2, 7, 2048).to(torch.bfloat16); ent = torch.zeros(2, 7, dtype=torch.bool); ent[0, 3:5] = True
    with torch.no_grad():
        v, m = C(dict(hidden=hid, mask=torch.ones(2, 7, dtype=torch.bool), ent=ent, state0=torch.randn(2, 36)))
    out["context_tokens"] = _sd(C) + ":" + _dg(v, m)
    return out


def _goldens_r5() -> dict:
    """probe metrics of the migrated (ReadoutProbe-based) StageA on the same fixture; not psi1z-comparable (the
    probe architecture changed, D-144 R5) but a regression baseline for this migration going forward."""
    out = {}
    for grasp in (False, True):
        torch.manual_seed(0); A = N.StageA(grasp=grasp)
        b = N.add_cmd_labels(_batch())
        with torch.no_grad():
            mu, _ = A.E(A.morph, b["state0"], b["actions"])
            p = N.run_probe(A.P, mu)
        out[f"probe_metrics_grasp{int(grasp)}"] = json.dumps(
            {k: [round(float(s), 5), n] for k, (s, n) in sorted(N.probe_metrics(p, b["labels"], A.P.specs).items())})
    return out


@torch_only
def test_nets_match_psi1z_goldens():
    """The subset of `_goldens()` the R5 migration does not touch (morphology tables, the legacy `read_mask()`
    helper, `ContextTokens`) still matches the original psi1z digests byte for byte."""
    got = _goldens()
    assert {k: got[k] for k in GOLDEN} == GOLDEN


@torch_only
def test_probe_metrics_match_r5_baseline():
    assert _goldens_r5() == GOLDEN_R5


@torch_only
def test_route_assembly_reads_matches_read_mask():
    """The `route.assembly_reads` FactorSite bias the Realizer now uses reproduces the exact -inf pattern of the
    legacy `read_mask()` fill it replaced (D-144 R5): same routing, same numbers, different mechanism."""
    m = N.Morph(); B = 3
    R = N.Realizer(D=16, heads=2, layers=1)
    rc = RelCtx(sets={
        "dims": TokenSet("dims", torch.ones(B, N.DA, dtype=torch.bool), fields={"assembly_id": m.dim_asm[None, :, None].expand(B, -1, -1)}),
        "knots": TokenSet("knots", torch.ones(B, N.K * N.M, dtype=torch.bool), fields={"assembly_id": R.knot_asm[None, :, None].expand(B, -1, -1)})})
    got = R.route.bias(rc)
    want = torch.zeros(B, 1, N.DA, N.K * N.M).masked_fill(~m.read_mask[None, None], float("-inf"))
    assert torch.equal(got, want)


@torch_only
def test_realizer_output_matches_pre_r5_formula():
    """The Realizer's forward output is byte-identical to the pre-R5 formula (an inline masked_fill of the same
    read_mask, run through the same RelBlock math) given the same weights and inputs — the factor-routed mask is a
    mechanism change, not a numerical one (R5 acceptance: "Realizer output byte-identical on a seeded fixture")."""
    torch.manual_seed(3)
    R = N.Realizer(D=32, heads=4, layers=2)
    m = N.Morph()
    B = 4
    z = torch.randn(B, N.K, N.M, N.DZ)
    state = torch.randn(B, 36)
    phase = torch.rand(B)
    got = R(m, z, state, phase)

    # the pre-R5 formula, reimplemented inline against the SAME modules (dims, z_in, asm, kt, out) and blocks
    ph = torch.stack([torch.sin(math.pi * phase), torch.cos(math.pi * phase)], -1)
    x = R.dims(m, state, extra=ph[:, None].expand(-1, N.DA, -1))
    kt = torch.as_tensor(N.KNOT_STEPS, dtype=z.dtype) / N.TP
    rel = kt[None] - phase[:, None]
    tok = R.z_in(z) + R.kt(N.sinusoidal(rel, R.D))[:, :, None] + R.asm(m.asm_static)[None, None]
    tok = tok.reshape(B, N.K * N.M, -1)
    bias = torch.zeros(1, 1, N.DA, N.K * N.M, dtype=tok.dtype).masked_fill(~m.read_mask[None, None], float("-inf"))
    for L in R.blocks:
        x = x + L.x(L.n1(x), kv=tok, bias=bias)
        x = x + L.s(L.n2(x))
        x = x + L.m(L.n3(x))
    want = R.out(x).transpose(1, 2)
    assert torch.equal(got, want)


@torch_only
def test_probes_psi0_v1_preset_and_grasp_optional():
    from rrp.policies.relations.base import resolve
    core = resolve(None, default="probes:psi0-v1")
    assert [s.name for s in core] == [f"probe.psi0.{q}" for q in
                                      ("hand_dist", "contact", "lift", "target_pos", "active_hand", "base_disp", "base_cmd")]
    assert not N.probe_has_grasp(N.new_probe())
    assert N.probe_has_grasp(N.new_probe(grasp=True))


@torch_only
def test_load_tolerant_drops_pre_r5_packet_probe_keeps_rest():
    """An old (pre-D-144) stage-A checkpoint's `P.*` (PacketProbe: `a1`/`a2`/`kcode`/`code` submodules, a different
    architecture than `ReadoutProbe`) does not load; `R` / `E` / `morph` still load strictly."""
    torch.manual_seed(11); A = N.StageA()
    old = {k: v for k, v in A.state_dict().items() if not k.startswith("P.")}
    old.update({"P.asm_code": torch.randn(N.M, 16), "P.code.weight": torch.randn(192, 16),
               "P.kcode.weight": torch.randn(192, 16), "P.a1.q.weight": torch.randn(192, 192)})
    torch.manual_seed(22)
    B = N.load_tolerant(N.StageA(), dict(old))
    sb = B.state_dict()
    assert all(torch.equal(sb[k], old[k]) for k in old if not k.startswith("P."))
    assert not torch.equal(sb["P.asm_code"], old["P.asm_code"])       # P kept its own fresh init


@torch_only
def test_load_tolerant_loads_post_r5_checkpoint_strictly():
    torch.manual_seed(4); A = N.StageA(D=32, probe_D=32)
    B = N.load_tolerant(N.StageA(D=32, probe_D=32), dict(A.state_dict()))
    sa, sb = A.state_dict(), B.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


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
        L.s.forward = lambda x, **k: torch.zeros_like(x)
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
    P = N.new_probe(D=32)
    out = N.run_probe(P, torch.randn(B, N.K, N.M, N.DZ))
    lab = dict(hand_dist=out["hand_dist"][..., 0].detach(), contact=torch.zeros(B, N.K, 2), lift=torch.zeros(B, N.K),
               target_pos=out["target_pos"][..., :3].detach(), active_hand=torch.zeros(B, dtype=torch.long),
               base_disp=out["base_disp"][:, :3].detach(), base_cmd=out["base_cmd"][..., :2].detach())
    for k in ("hand_dist", "target_pos", "base_disp", "base_cmd"):
        out[k] = out[k].detach().clone(); d = out[k].shape[-1] // 2; out[k][..., d:] = -100.0
    total, logs = N.probe_loss(out, lab, P.specs, lv_min=-4.0)
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
    torch.manual_seed(0); P = N.new_probe(D=32, grasp=True)
    out = N.run_probe(P, torch.randn(3, N.K, N.M, N.DZ))
    assert out["grasp_pt"].shape == (3, 2, 6) and out["grasp_face"].shape == (3, 2, N.N_FACES)
    lab = {k: v[:3] for k, v in _batch()["labels"].items()}
    assert "probe_grasp_pt" not in N.probe_loss(out, lab, P.specs, w_grasp=0.0)[1]
    assert {"probe_grasp_pt", "probe_grasp_face"} <= set(N.probe_loss(out, lab, P.specs, w_grasp=1.0)[1])
    assert N.probe_metrics(out, lab, P.specs)["grasp_face_acc"][1] == 2  # only the valid (sample, hand) pairs are scored


@torch_only
def test_load_stage_a_restores_grasp_head_from_config(tmp_path):
    """D-146 P1: `load_stage_a` rebuilds the net from the checkpoint's own config (no inference from weights)."""
    from rrp.policies.nets.checkpoint import save_checkpoint
    for g in (False, True):
        torch.manual_seed(1); A = N.StageA(grasp=g)
        save_checkpoint(tmp_path / "a.pt", model=A, step=0, versions={}, config=dict(stage_a=A.cfg))
        B = N.load_stage_a(tmp_path / "a.pt")
        sa, sb = A.state_dict(), B.state_dict()
        assert N.probe_has_grasp(B.P) == g and sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


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


# ------------------------------------------------------------------ SimpleEnv x psi0_replay x simple/<Task> through harness.rollout
_FAKE_WORKER = r'''
import sys, numpy as np
from multiprocessing.connection import Client
import os
from rrp.envs.simple import worker as W

class Fake:
    """Upstream-agent queue semantics without a simulator: 60-step stand warm-up on MP tasks, one row per step,
    asks for rows when the queue is empty, TimeLimit at 100 steps, success when >= 40 recorded rows ran."""
    def __init__(self, task, level=0, **kw):
        if level not in W.LEVELS:
            raise ValueError("level")
        self.task, self.mp = task, task.endswith("MP-v0")
    def init_info(self):
        return dict(task=self.task, level=0, sim_mode="fake", mp=self.mp, max_steps=100, agent="Fake",
                    joint_names=[f"j{i}" for i in range(43)], render="none", uid_fixes=[], compat_log=[], upstream={})
    def reset(self, episode, split="eval"):
        self.q, self.n, self.ran, self.warm, self.done = [], 0, 0, (60 if self.mp else 0), False
        return self._pack()
    def _need(self):
        return not self.q and self.warm == 0
    def _pack(self):
        return dict(joint_qpos=np.zeros(43, np.float32), image=np.zeros((4, 6, 3), np.uint8), instruction="pick up the box",
                    psi0_state=np.zeros(32, np.float32), chunk_request=self._need(), step=self.n, time=self.n * 0.02,
                    done=self.done, stabilize_steps=0)
    def step(self, rows=None):
        if rows is not None:
            assert self._need(), "rows submitted while the queue is not empty"
            self.q = list(rows)
        if self.warm:
            self.warm -= 1
        elif not self.q:
            raise W.NeedChunk()
        else:
            self.q.pop(0); self.ran += 1
        self.n += 1
        self.done = self.n >= 100
        return self._pack()
    def truth(self):
        return dict(source="privileged:sim", success=self.ran >= 40, terminated=False, truncated=self.done,
                    palm={"left": np.zeros(3), "right": np.ones(3)}, pelvis=np.zeros(7), objects={"target": np.zeros(7)},
                    contact={"left:target": False, "right:target": self.ran > 10},
                    contact_point={"left": np.full(3, np.nan), "right": np.ones(3)}, reward=float(self.ran) / 100)
    def render(self):
        return np.zeros((4, 6, 3), np.uint8)
    def close(self):
        pass

host, port = sys.argv[sys.argv.index("--address") + 1].rsplit(":", 1)
W.serve(Client((host, int(port)), authkey=bytes.fromhex(os.environ["RRP_SIMPLE_AUTHKEY"])), Fake)
'''


def _fake_env(tmp_path, task, seed, **kw):
    import sys
    from rrp.envs.simple import SimpleEnv
    script = tmp_path / "fake_worker.py"
    script.write_text(_FAKE_WORKER)
    env = SimpleEnv(task, worker_cmd=[sys.executable, str(script)], start_timeout=60, **kw)
    env.reset(seed)
    return env


@pytest.mark.parametrize("task,n_rows,outcome", [("G1WholebodyTabletopGraspMP-v0", 60, "success"),
                                                  ("G1WholebodyHandoverTeleop-v0", 30, "rejected")])
def test_replay_rollout_through_simple_env(tmp_path, task, n_rows, outcome):
    from rrp.harness.rollout import rollout
    from rrp.policies.base import negotiate
    from rrp.policies.psi0 import make_replay
    from rrp.policies.psi0.data import LabelRecorder
    from rrp.tasks.spec import get_task
    if N is None:
        pytest.skip("policies.psi0.data needs torch")
    pol = make_replay(task=f"simple/{task}")
    pol.recorded_rows = lambda e: np.full((n_rows, 36), e, np.float32)
    t = get_task(f"simple/{task}")
    rec = LabelRecorder(tmp_path / "labels", mp=task.endswith("MP-v0"))
    eps = rollout(lambda sd: _fake_env(tmp_path, task, sd), pol, t, [0, 1], batch=2, hooks=[rec])
    assert [e.outcome for e in eps] == [outcome, outcome] and all(e.env_id == "simple" and e.body == "g1_simple" for e in eps)
    assert eps[0].source == f"replay:{task}" and eps[0].success_public is None
    if outcome == "rejected":            # 30 recorded rows, then no rows: explicit rejection, never a silent hold
        assert eps[0].failure_reason == "chunk_required" and eps[0].steps == 31
    else:                                # MP: 60 stand steps + 40 rows until the fake TimeLimit
        assert eps[0].steps == 100 and eps[0].success_privileged
    lab = np.load(tmp_path / "labels" / "episode_000001.npz")
    assert lab["contact__right"].shape[0] == eps[1].steps - (60 if task.endswith("MP-v0") else 0)
    env = _fake_env(tmp_path, task, 0)
    try:
        sp = env.spec
        assert sp.action_kinds() == {"psi0"} and sp.bodies[0].robot_spec_hash == G.spec_hash()
        from rrp.envs.simple import env_spec
        assert env_spec(task=sp.task).model_dump(exclude={"provenance"}) == sp.model_dump(exclude={"provenance"})
        assert negotiate(pol.info, sp, t).ok
        other = get_task("simple/G1WholebodyBendPickMP-v0")
        from rrp.policies.psi0 import make_direct
        assert any("not in" in r for r in negotiate(make_direct(task=f"simple/{task}").info, sp, other).reasons)
    finally:
        env.close()


def test_simple_tasks_registered_and_arm_policies_declined():
    from rrp.envs.base import ActionSpace, BodyInfo, EnvSpec
    from rrp.policies.base import POLICIES, PolicyInfo, Requirements, negotiate
    from rrp.policies.psi0 import make_direct, make_structured
    from rrp.tasks.spec import SIMPLE_TASKS, TASKS
    assert {f"simple/{k}" for k in SIMPLE_TASKS} <= set(TASKS) and len(SIMPLE_TASKS) == 6
    assert POLICIES["psi0_direct"] == "rrp.policies.psi0:make_direct" and POLICIES["psi0_structured"] == "rrp.policies.psi0:make_structured"
    d, s = make_direct(), make_structured()           # construction is cheap: no weights touched before reset
    assert d.info.source.startswith("upstream:psi0-released/") and d.info.variant == "released" and s.info.version == "unhashed"
    from rrp.envs.simple import env_spec
    simple = env_spec(task="simple/G1WholebodyTabletopGraspMP-v0")         # static: no worker, no Isaac
    assert simple.env_id == "simple" and simple.action_kinds() == {"psi0"} and simple.bodies[0].key == "g1_simple"
    with pytest.raises(ValueError):
        env_spec(task="simple/G1WholebodyTabletopGraspMP-v0", body="panda")
    assert negotiate(d.info, simple, TASKS["simple/G1WholebodyTabletopGraspMP-v0"]).ok
    latent = PolicyInfo("latent", "learned", "v", Requirements(frozenset({"joint_position", "gripper"}),
                                                              body_families=frozenset({"arm"})))
    assert any("needs gripper, joint_position; env offers psi0" in r for r in negotiate(latent, simple).reasons)
    arm = EnvSpec(env_id="mujoco/arm", backend="mujoco", task="pick_place", control_hz=20.0,
                  bodies=[BodyInfo(robot=0, family="arm", key="panda", robot_spec_hash="h")],
                  action_spaces=[ActionSpace(group="arm", kind="joint_position", width=7, rate_hz=20.0)],
                  capabilities=["proprio", "chunk_executor"])
    r = negotiate(d.info, arm, TASKS["pick_place"]).reasons
    assert any("needs psi0" in x for x in r) and any("g1_simple" in x for x in r)


def test_viz_psi0_reads_the_folded_p_log(tmp_path):
    from rrp.viz.export.common import Config
    from rrp.viz.export.psi0 import build_psi0
    repo = tmp_path / "repo"
    (repo / "research/tracks").mkdir(parents=True)
    (repo / "research/decisions.md").write_text(
        "# decision log\n\n## D-140 2026-09-29 one repo\nbody\n\n# Appendix P: psi1z decisions\nintro\n\n"
        "## P-012 2026-09-27 XMovePick not reproduced (rrp D-120)\nbody P\n")
    (repo / "research/tracks/psi0.md").write_text("# track psi0\n| a | b |\n|---|---|\n| 1 | 2 |\n")
    d = build_psi0(Config(repo=repo, out=tmp_path / "out", wt_root=None, main_checkout=None, rrp_data=None), [])
    d = d.get("data", d)
    assert [p["id"] for p in d["p_decisions"]] == ["P-012"] and d["crosswalk"][0]["rrp"] == ["D-120"]
    assert d["notes_markdown"].startswith("# track psi0") and d["notes_tables"]


def test_eval_released_stamps_upstream_source(tmp_path, monkeypatch):
    """The eval_r2 stage of the released arm returns an `upstream:` source (the manifest uses it), never `learned:`."""
    import json as _json
    from types import SimpleNamespace
    from rrp.harness.pipelines import psi0 as P
    out = tmp_path / "o"; out.mkdir()
    def run(argv, log_to=None):
        (out / "released.summary.json").write_text(_json.dumps(dict(successes=1, attempted=1)))
    ctx = SimpleNamespace(opts=dict(arm="released", task="G1WholebodyBendPickMP-v0"), out=out, root=tmp_path, run=run,
                          rc=SimpleNamespace(config=dict(options=dict(task="G1WholebodyBendPickMP-v0"))))
    monkeypatch.setattr(P, "_task", lambda c: "G1WholebodyBendPickMP-v0")
    res = P.eval_r2(ctx)
    assert res["source"].startswith("upstream:psi0-released/") and res["metrics"]["success"] == "1/1"
    ctx.opts = dict(arm="direct"); ctx.inp = lambda k: "ckpt.pt"
    assert "source" not in P.eval_r2(ctx)            # our arms keep the stage default (learned:<run>)


def test_eval_r2_level_option_reaches_the_env(tmp_path, monkeypatch):
    """options.level -> `rrp eval --env-kw level=<n>` (SIMPLE dr-level-<n> eval configs); absent / 0 adds nothing."""
    import json as _json
    from types import SimpleNamespace
    from rrp.harness.pipelines import psi0 as P
    seen = []
    def run(argv, log_to=None):
        seen.append(list(argv)); (tmp_path / "released.summary.json").write_text(_json.dumps(dict(successes=0, attempted=1)))
    monkeypatch.setattr(P, "_task", lambda c: "G1WholebodyBendPickMP-v0")
    for lv in (None, 2):
        o = dict(arm="released") if lv is None else dict(arm="released", level=lv)
        P.eval_r2(SimpleNamespace(opts=o, out=tmp_path, root=tmp_path, run=run))
    assert "--env-kw" not in seen[0] and seen[1][seen[1].index("--env-kw") + 1] == "level=2"
