"""W12 feature-centric coordination: anchor geometry, contact segmentation / event-aligned knots / anchor-relative
targets, bounded anchor probes, the anchor-input system 0 variant, contact metrics on synthetic trajectories and the
contact edit specs."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from rrp.data.contact_segments import (PHASE_ID, ContactRecording, SegmentParams, anchor_relative_targets,
                                       debounced_intervals, event_aligned_knot_times, event_ticks, segment,
                                       uniform_knot_times)
from rrp.evaluation.contact_metrics import (TASK_CONTACT_SPECS, arm_contact_motion, contact_sequence,
                                            dual_contact_motion, receipt_latency, relative_drift, settle_latency,
                                            stance_drift, swap_hands)
from rrp.features.anchor_frame import (ANCHOR_BLOCK, ANCHOR_INPUT_DIM, anchor_inputs, axis_angle, frame_from_normal,
                                       geodesic_angle, pose_vec, relative_pose, rot6d, rot6d_to_mat, rotz, yaw_about)

DOWN = np.diag([1.0, -1.0, -1.0])          # tool z pointing down (rotation about x by pi)


# ----------------------------------------------------------------------------------------------- geometry
def test_rot6d_roundtrip_and_geodesic():
    R = axis_angle([0.3, -0.5, 0.8], 1.1)
    assert np.allclose(rot6d_to_mat(rot6d(R)), R, atol=1e-9)
    Rn = rot6d_to_mat(rot6d(R) + np.array([0.05, 0, 0, 0, 0.02, 0]))           # any 6-vector -> a rotation
    assert np.allclose(Rn.T @ Rn, np.eye(3), atol=1e-9) and np.linalg.det(Rn) == pytest.approx(1.0)
    assert geodesic_angle(R, R @ axis_angle([0, 0, 1], 0.4)) == pytest.approx(0.4)
    assert yaw_about(rotz(0.7)) == pytest.approx(0.7)
    assert yaw_about(axis_angle([1, 0, 0], 0.3) @ rotz(-0.5)) == pytest.approx(-0.5, abs=0.05)


def test_anchor_frame_and_relative_pose():
    f = frame_from_normal([1.0, 2.0, 0.5], [0, 0, 2.0])
    assert np.allclose(f.R, np.eye(3)) and not f.yaw_defined                      # x = projected world +x
    f2 = frame_from_normal([0, 0, 0], [1.0, 0, 0])                               # normal parallel to the reference
    assert np.allclose(f2.R.T @ f2.R, np.eye(3)) and np.allclose(f2.R[:, 2], [1, 0, 0])
    p, R = relative_pose(f, [1.1, 2.0, 0.7], rotz(0.3))
    assert np.allclose(p, [0.1, 0.0, 0.2]) and np.allclose(R, rotz(0.3))
    v = pose_vec(p, R)
    assert v.shape == (9,) and np.allclose(v[:3], [1.0, 0.0, 2.0])               # decimetres


# ----------------------------------------------------------------------------------------------- segmentation
def _one_arm_recording(T=140, dt=0.02):
    """Manipulator 'left' descends onto 'box' (contact 20..100 with chatter at 50), slides +x 40-60, pivots 70-80, lifts."""
    pos = np.zeros((T, 1, 3))
    R = np.repeat(DOWN[None, None], T, 0)
    for t in range(T):
        x = 0.0 if t < 40 else (min(t, 60) - 40) * 0.002                        # 0.1 m/s slide
        z = 0.1 - 0.004 * min(t, 20) if t <= 100 else (t - 100) * 0.004
        pos[t, 0] = [x, 0.0, z + 0.02]
        if 70 <= t < 80:
            R[t, 0] = DOWN @ rotz((t - 69) * 0.02)                               # 1 rad/s pivot about the tool z
        elif t >= 80:
            R[t, 0] = DOWN @ rotz(0.2)
    contact = np.zeros((T, 1), bool)
    contact[20:100, 0] = True
    contact[50, 0] = False                                                        # chatter: must not break the interval
    return ContactRecording(dt=dt, manipulators=["left"], tcp_pos=pos, tcp_R=R, objects=["box"],
                            obj_pos=np.zeros((T, 1, 3)), obj_R=np.repeat(np.eye(3)[None, None], T, 0),
                            pairs=[("left", "box")], contact=contact, held=np.zeros((T, 1, 1), bool))


def test_debounced_intervals():
    x = np.zeros(30, bool)
    x[5:20] = True
    x[10] = False
    x[25] = True                                                                  # single-tick blip: no interval
    assert debounced_intervals(x, min_on=2, min_off=3) == [(5, 20)]
    y = np.ones(10, bool)
    assert debounced_intervals(y) == [(0, 10)]


def test_segment_phases_and_events():
    rec = _one_arm_recording()
    seg = segment(rec, SegmentParams(slide_v=0.05, pivot_w=0.5, smooth=1))
    assert seg["intervals"][0] == [(20, 100)]
    ph = seg["phase"][:, 0]
    assert ph[21] == PHASE_ID["make"] and ph[30] == PHASE_ID["maintain"]
    assert ph[50] == PHASE_ID["slide"] and ph[75] == PHASE_ID["pivot"]
    assert ph[98] == PHASE_ID["release"] and ph[15] == PHASE_ID["approach"] and ph[120] == PHASE_ID["free"]
    kinds = [k for _, k in event_ticks(seg)]
    assert kinds[0] == "make" and "break" in kinds and "slide_start" in kinds and "pivot_start" in kinds


def test_event_aligned_knots():
    t, tags = event_aligned_knot_times([], t0=0, dt=0.02)
    assert np.allclose(t, uniform_knot_times()) and np.allclose(t, [0.1, 0.3, 0.5, 0.7]) and tags == ["uniform"] * 4
    t, tags = event_aligned_knot_times([(10, "make"), (11, "slide_start"), (30, "break")], t0=0, dt=0.02)
    assert len(t) == 4 and np.all(np.diff(t) > 0) and 0.2 in np.round(t, 6) and 0.6 in np.round(t, 6)
    assert "make" in tags and "break" in tags and "slide_start" not in tags      # too close to the make (gap 0.1 s)
    t, _ = event_aligned_knot_times([(100, "make")], t0=0, dt=0.02)              # outside the horizon -> uniform
    assert np.allclose(t, uniform_knot_times())


# ----------------------------------------------------------------------------------------------- anchor targets
def _two_arm_recording(T=60, dt=0.02):
    """'left' supports 'fixture' from t=5 (static); 'right' holds 'peg' from t=10, peg rotated 0.3 rad about the
    right tool z inside the hand; right TCP 0.1 m above/aside the left anchor."""
    tcp = np.zeros((T, 2, 3))
    tcp[:, 0] = [0.4, 0.2, 0.05]
    tcp[:, 1] = [0.4, 0.0, 0.15]
    R = np.repeat(np.stack([DOWN, DOWN])[None], T, 0)
    obj_pos = np.zeros((T, 2, 3))
    obj_pos[:, 0] = [0.4, 0.2, 0.0]
    obj_pos[:, 1] = [0.4, 0.0, 0.10]
    obj_R = np.repeat(np.stack([np.eye(3), DOWN @ rotz(0.3)])[None], T, 0)
    pairs = [("left", "fixture"), ("left", "peg"), ("right", "fixture"), ("right", "peg")]
    contact = np.zeros((T, 4), bool)
    contact[5:, 0] = True
    contact[10:, 3] = True
    held = np.zeros((T, 2, 2), bool)
    held[10:, 1, 1] = True
    normal = np.zeros((T, 4, 3))
    normal[:, 0] = [0, 0, 1.0]
    return ContactRecording(dt=dt, manipulators=["left", "right"], tcp_pos=tcp, tcp_R=R, objects=["fixture", "peg"],
                            obj_pos=obj_pos, obj_R=obj_R, pairs=pairs, contact=contact, contact_normal=normal,
                            held=held)


def test_anchor_relative_targets_known_values():
    rec = _two_arm_recording()
    seg = segment(rec)
    tg = anchor_relative_targets(rec, seg, t0=20, knot_ticks=[25, 30, 35, 40])
    # right TCP in the support anchor (left TCP at make, normal = -tool z = +z world): world offset (0, -0.2, 0.1)
    assert tg["tcp_in_support_mask"].all()
    assert np.allclose(tg["tcp_in_support"][0, 1, :3], [0.0, -2.0, 1.0], atol=1e-5)
    # the left hand's "other" anchor is the right hand's live grasp contact (another contact point)
    assert np.allclose(tg["tcp_in_support"][0, 0, :3], [0.0, 2.0, -1.0], atol=1e-5)
    assert np.allclose(rot6d_to_mat(tg["tcp_in_support"][0, 1, 3:]), DOWN, atol=1e-6)
    # held peg in the right TCP frame: rotation rotz(0.3), offset (0,0,-0.05) world = (0,0,0.05) tool
    assert tg["held_in_tcp_mask"][:, 1].all()
    assert np.allclose(rot6d_to_mat(tg["held_in_tcp"][0, 1, 3:]), rotz(0.3), atol=1e-6)
    assert np.allclose(tg["held_in_tcp"][0, 1, :3], [0, 0, 0.5], atol=1e-5)
    assert tg["held_in_support_mask"][:, 1].all()
    # the left hand's own anchor: its TCP sits exactly at it
    assert np.allclose(tg["tcp_in_own"][0, 0, :3], 0, atol=1e-6)
    # recorded contact normal (+z world) in the left tool frame (tool z down) = -z
    assert tg["normal_in_tcp_mask"][0, 0] and np.allclose(tg["normal_in_tcp"][0, 0], [0, 0, -1])
    assert tg["contact_persist"][:, 0].all() and tg["contact_persist"][:, 3].all() and not tg["contact_persist"][:, 1].any()
    assert (tg["phase"][:, 0] == PHASE_ID["maintain"]).all()


# ----------------------------------------------------------------------------------------------- deploy-time input
def _receipts():
    return [dict(type="contact_anchor", value={"pos": [0.4, 0.2, 0.05], "normal": [0, 0, 1.0], "tangent": None},
                 valid=True, created_at=1.0, participants=["left"], event_id="support", attempt=0, version=0,
                 covariance_diag=[1e-4] * 3),
            dict(type="contact_anchor", value={"pos": [9, 9, 9], "normal": [0, 0, 1.0]}, valid=False,
                 created_at=1.5, participants=["left"], event_id="support", attempt=0, version=1),
            dict(type="frame_estimate", value={"pos": [0.5, -0.1, 0.045], "quat_wxyz": [1, 0, 0, 0]}, valid=True,
                 created_at=0.5, participants=["camera"], event_id="locate", attempt=0, version=0),
            dict(type="completion_receipt", value={"x": 1}, valid=True, created_at=0.1, participants=["right"])]


def test_anchor_inputs_public_only_and_roles():
    tcp = {"left": (np.array([0.4, 0.2, 0.05]), DOWN), "right": (np.array([0.4, 0.0, 0.15]), DOWN)}
    X, mask = anchor_inputs(tcp, _receipts(), ["left", "right"], now=2.0)
    assert X.shape == (2, ANCHOR_INPUT_DIM) and mask.all()
    own, other, frame = (X[:, i * ANCHOR_BLOCK:(i + 1) * ANCHOR_BLOCK] for i in range(3))
    assert own[0, 0] == 1 and other[0, 0] == 0          # left: its own anchor (the invalid v1 is ignored)
    assert own[1, 0] == 0 and other[1, 0] == 1          # right: the left's support anchor is "other"
    assert np.allclose(other[1, 1:4], [0.0, -2.0, 1.0], atol=1e-5)
    assert frame[0, 0] == 1 and frame[1, 0] == 1
    assert other[1, 13] == pytest.approx(math.log1p(1.0))
    # shifting the receipt (context edit) shifts the relative position by -delta; nothing else is read
    r2 = _receipts()
    r2[0]["value"] = dict(r2[0]["value"], pos=[0.43, 0.2, 0.05])
    X2, _ = anchor_inputs(tcp, r2, ["left", "right"], now=2.0)
    d = X2[1, ANCHOR_BLOCK + 1:ANCHOR_BLOCK + 4] - other[1, 1:4]
    assert np.allclose(d, [-0.3, 0, 0], atol=1e-5)
    X3, m3 = anchor_inputs(tcp, [], ["left", None], now=2.0)
    assert not X3.any() and list(m3) == [True, False]


# ----------------------------------------------------------------------------------------------- probes
def _probe_batch(B=3, K=4, M=2, C=4):
    rec = _two_arm_recording()
    seg = segment(rec)
    tgs = [anchor_relative_targets(rec, seg, t0=20, knot_ticks=[25 + i, 30, 35, 40]) for i in range(B)]
    from rrp.models.anchor_probes import batch_targets
    return batch_targets(tgs, np.ones((B, M), bool), n_pairs=C)


def test_anchor_probe_bounded_nll_and_grad_norms():
    from rrp.models.anchor_probes import (AnchorProbe, anchor_probe_loss, anchor_probe_metrics,
                                          bounded_gaussian_nll, nll_lower_bound, per_loss_grad_norms)
    torch.manual_seed(0)
    lab = _probe_batch()
    P = AnchorProbe(dz=8, knots=4)
    z = torch.randn(3, 4, 2, 8)
    out = P(z, torch.ones(3, 2, dtype=torch.bool), n_pairs=4)
    assert out["tcp_in_own"].shape == (3, 4, 2, 18) and out["contact_persist"].shape == (3, 4, 4)
    total, L = anchor_probe_loss(out, lab)
    assert torch.isfinite(total) and set(L) >= {"tcp_in_support", "held_in_tcp", "contact_phase", "contact_persist"}
    # bounded: a perfect mean with an absurdly small requested variance cannot go below the floor
    tgt = torch.randn(5, 9)
    pred = torch.cat([tgt, torch.full((5, 9), -30.0)], -1)
    assert float(bounded_gaussian_nll(pred, tgt, torch.ones(5, dtype=torch.bool))) == pytest.approx(nll_lower_bound(9))
    g = per_loss_grad_norms(L, P.parameters())
    assert set(g) == set(L) and all(v >= 0 for v in g.values()) and g["tcp_in_support"] > 0
    met = anchor_probe_metrics(out, lab)
    assert met["held_in_tcp_rot_err_deg"][1] == 3 * 4 and met["contact_phase_acc"][1] == 3 * 4 * 2
    # metadata-only control ignores z
    Pm = AnchorProbe(dz=8, knots=4, metadata_only=True)
    a = Pm(z, torch.ones(3, 2, dtype=torch.bool))["tcp_in_own"]
    b = Pm(torch.randn_like(z), torch.ones(3, 2, dtype=torch.bool))["tcp_in_own"]
    assert torch.allclose(a, b)


# ----------------------------------------------------------------------------------------------- system 0 variant
def test_anchor_realizer_zero_init_equals_base():
    from rrp.controllers.anchor_realizer import AnchorLatentRealizer
    from rrp.controllers.latent_realizer import LatentRealizer
    from rrp.models.batch import NODE_DIM
    torch.manual_seed(1)
    base = LatentRealizer(dz=8, width=32, layers=2).eval()
    var = AnchorLatentRealizer.from_base(base).eval()
    B, K, M, N = 2, 4, 2, 5
    z = torch.randn(B, K, M, 8)
    zm = torch.ones(B, M, dtype=torch.bool)
    kt = torch.tensor([0.1, 0.3, 0.5, 0.7])
    ph = torch.tensor([0.05, 0.4])
    nf = torch.randn(B, N, NODE_DIM)
    nm = torch.ones(B, N, dtype=torch.bool)
    loc = torch.randn(B, N, 4)
    na = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 0, 1, 0]])
    anc = torch.randn(B, M, ANCHOR_INPUT_DIM)
    with torch.no_grad():
        y0 = base(z, zm, kt, ph, nf, nm, loc, na)
        y1 = var(z, zm, kt, ph, nf, nm, loc, na, anchor=anc)
        y2 = var(z, zm, kt[None].expand(B, -1), ph, nf, nm, loc, na, anchor=None)   # per-sample knot times
        assert torch.allclose(y0, y1, atol=1e-6) and torch.allclose(y0, y2, atol=1e-6)
        var.anchor_in.weight.normal_()
        y3 = var(z, zm, kt, ph, nf, nm, loc, na, anchor=anc)
        assert not torch.allclose(y0, y3)
        # routing: every node receives the anchor features of its OWN packet slot
        seen = {}
        h = var.anchor_in.register_forward_hook(lambda mod, inp, out: seen.update(x=inp[0].clone()))
        var(z, zm, kt, ph, nf, nm, loc, na, anchor=anc)
        h.remove()
        assert torch.equal(seen["x"], torch.stack([anc[b, na[b]] for b in range(B)]))


# ----------------------------------------------------------------------------------------------- metrics
def test_relative_drift_known_values():
    T = 50
    Rg = np.repeat(rotz(0.5)[None], T, 0) @ np.repeat(axis_angle([1, 0, 0], 0.2)[None], T, 0)
    Ro = np.stack([Rg[t] @ rotz(0.004 * t) for t in range(T)])                # object twists in hand: 0.2 rad over 50
    p = np.zeros((T, 3))
    d = relative_drift(Rg, p, Ro, p + np.array([0, 0, 0.1]), [(10, 50)])
    assert d["rot_max"] == pytest.approx(0.004 * 39, rel=1e-6) and d["pos_max"] == pytest.approx(0.0, abs=1e-12)
    common = np.stack([rotz(0.01 * t) for t in range(T)])                      # both rotate together: no drift
    d2 = relative_drift(common, p, common, p, [(0, 50)])
    assert d2["rot_max"] == pytest.approx(0.0, abs=1e-6)
    assert relative_drift(common, p, common, p, [(3, 4)])["n"] == 0


def test_contact_sequence_fidelity():
    spec = TASK_CONTACT_SPECS["handover"]
    ok = {"make:left|bar": 1.0, "make:right|bar": 2.0, "break:left|bar": 2.5}
    iv = {"left|bar": [(1.0, 2.5)]}
    r = contact_sequence(ok, spec, ref_times=ok, intervals=iv)
    assert r["order_error"] == 0 and r["missing"] == 0 and r["kendall_order_error"] == 0 and r["timing_mae_s"] == 0
    bad = {"make:left|bar": 1.0, "make:right|bar": 3.0, "break:left|bar": 2.5}  # giver lets go before the receiver
    r = contact_sequence(bad, spec, ref_times=ok, intervals={"left|bar": [(1.0, 2.5)]})
    assert r["precedence_violations"] == 1 and r["maintained_violations"] == 1 and r["order_error"] == pytest.approx(2 / 3)
    assert r["kendall_order_error"] == pytest.approx(1 / 3) and r["timing_mae_s"] == pytest.approx(1 / 3)
    sw = swap_hands(spec)
    assert sw["precedence"][0] == ("make:right|bar", "make:left|bar")
    r = contact_sequence({"make:left|bar": 1.0}, spec)
    assert r["missing"] == 2 and r["order_error"] is None


def test_reanchoring_and_receipt_latency():
    rec = _one_arm_recording()
    seg = segment(rec)
    lat = settle_latency(rec, 0, 20, 100, v_tol=0.01, hold=5)
    assert lat == pytest.approx(0.0)                                          # at rest in the anchor until the slide
    rec.tcp_pos[20:30, 0, 0] += np.linspace(0, 0.05, 10)                        # at rest from tick 29 on
    rec.tcp_pos[30:, 0, 0] += 0.05
    assert settle_latency(rec, 0, 20, 40, v_tol=0.01, hold=5) == pytest.approx(0.18)
    log = [dict(op="put", type="contact_anchor", participants=["left"], created_at=0.5),
           dict(op="put", type="contact_anchor", participants=["left"], created_at=0.46)]
    assert receipt_latency(0.4, "left", log) == pytest.approx(0.06)
    assert receipt_latency(0.6, "left", log) is None
    assert seg["events"][0].kind == "make"


def test_dual_contact_motion_keys_and_values():
    rec = _two_arm_recording()
    for t in range(20, 60):                                                    # the peg twists 0.01 rad/tick in hand
        rec.obj_R[t, 1] = DOWN @ rotz(0.3 + 0.01 * (t - 19))
    out = dual_contact_motion(rec, task="support_insert")
    assert out["cf_held_rot_drift_grip_max_rad"] == pytest.approx(0.01 * 37, abs=1e-6)   # trimmed segment 13..57
    assert out["cf_held_rot_drift_support_max_rad"] == pytest.approx(out["cf_held_rot_drift_grip_max_rad"])
    assert out["cf_support_anchor_slip_max_m"] == pytest.approx(0.0)
    assert out["cf_contact_missing"] == 1                                      # the peg is never released
    a = arm_contact_motion(rec)
    assert a["cf_n_held_segments"] == 1


def test_stance_drift():
    T = 40
    pos = np.zeros((T, 2, 3))
    yaw = np.zeros((T, 2))
    st = np.zeros((T, 2), bool)
    st[5:25, 0] = True
    pos[6:25, 0, 0] = np.linspace(0, 0.01, 19)                                 # 1 cm slide during stance
    yaw[6:25, 0] = np.linspace(0, 0.1, 19)
    st[10:30, 1] = True                                                        # a clean stance
    d = stance_drift(pos, yaw, st, skip=1)
    assert d["n_stances"] == 2 and d["pos_max"] == pytest.approx(0.01) and d["yaw_max"] == pytest.approx(0.1)
    assert d["pos_mean"] == pytest.approx(0.005)


# ----------------------------------------------------------------------------------------------- edit specs
def test_contact_edit_specs_and_anchor_shift_edit():
    from rrp.evaluation.edit_harness import (EDIT_KINDS, anchor_shift_target, contact_edit_conditions,
                                             paired_effect, probe_guided_edit, projected_shift, shift_receipt_value)
    from rrp.models.anchor_probes import AnchorProbe
    cs = contact_edit_conditions((0.03, 0, 0))
    names = [c.name for c in cs]
    assert len(set(names)) == len(names) and all(c.kind in EDIT_KINDS and c.prediction for c in cs)
    kinds = {c.kind for c in cs}
    assert {"control", "semantic", "irrelevant_control", "negative_control"} <= kinds
    assert shift_receipt_value({"pos": [1, 2, 3], "normal": [0, 0, 1]}, [0.1, 0, 0])["pos"] == [1.1, 2, 3]
    rows = [dict(seed=s, condition=c, x=[v, 0, 0]) for s in range(6) for c, v in (("control", 0.0), ("e", 0.03))]
    eff = paired_effect(rows, projected_shift("x", [2.0, 0, 0]), "e", n_boot=200, n_perm=200)
    assert eff["mean_diff"] == pytest.approx(0.03)
    torch.manual_seed(0)
    P = AnchorProbe(dz=8, knots=4).eval()
    readout = lambda z: P(z[None] if z.dim() == 3 else z, torch.ones(1, 2, dtype=torch.bool))
    z0 = np.random.default_rng(0).standard_normal((4, 2, 8)).astype(np.float32)
    tl = anchor_shift_target(readout, z0, [0.03, 0, 0], slot=1)
    before = float(tl(torch.as_tensor(z0)).detach())
    z1, info = probe_guided_edit(z0, tl, steps=60, lr=0.05)
    assert float(tl(torch.as_tensor(z1)).detach()) < 0.5 * before and info["delta_norm"] > 0
