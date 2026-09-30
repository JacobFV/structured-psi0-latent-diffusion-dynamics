"""HD2: the legged trainers are task-agnostic and train the `upper` action group where it was commanded.

A tiny humanoid pack (two 3-joint legs + a 1-joint body + two 2-joint arms: 6 policy rows, 5 held rows) is written in the
HD1 shard format for `h_reach` (wholebody: upper targets valid) and `h_steps` (base velocity: upper not commanded,
foothold labels). Plumbing only: random data, three CPU steps, no number here is a result."""
import functools
import json

import numpy as np
import pytest
import torch

from rrp.harness.data.mix import RelgenError
from rrp.harness.train import legged_bc as TB
from rrp.harness.train import legged_latent_train as T
from rrp.policies.features.legged import GLOBAL_DIM, target_slot_cols
from rrp.policies.nets import legged_latent as LL
from rrp.policies.relations.base import estimates_loss
from .test_legged_relations import DZ, D, PROBES, batch

NPOL = 6


@pytest.fixture(autouse=True)
def _short_evals(monkeypatch):
    monkeypatch.setattr(T, "eval_rep", functools.partial(T.eval_rep, n_batches=2))
    monkeypatch.setattr(T, "eval_flow", functools.partial(T.eval_flow, n_batches=2))
    monkeypatch.setattr(TB, "eval_bc", functools.partial(TB.eval_bc, n_batches=2))


def _write_pack(root, task, wholebody, n_eps=3, T_=24, foothold=True, legacy=False):
    """One shard in the latent-collector format. `legacy`: a pre-HD1 waypoint pack (no task / task_view / upper keys, the
    episode `waypoints` meta)."""
    b = batch(B=1, terrain=False)
    ns = b["node_static"][0].numpy().copy()
    ns[:, LL.IS_POLICY_COL] = [1.0] * NPOL + [0.0] * (ns.shape[0] - NPOL)
    rng = np.random.default_rng(0)
    N, M, n = ns.shape[0], 5, n_eps * T_
    ctx = rng.normal(size=(n, GLOBAL_DIM)).astype(np.float32)
    ctx[:, target_slot_cols(0)][:, 3] = 1.0                       # entity slot 0 located
    ctx[:, target_slot_cols(1)][:, 3] = 0.0                       # slot 1 not
    d = dict(node_static=ns, asm_static=b["asm_static"][0].numpy(), node_asm=b["node_asm"][0].numpy(), nf=np.array(2),
             n_policy=np.array(NPOL), ep=np.repeat(np.arange(n_eps), T_),
             q=rng.normal(size=(n, N)).astype(np.float32), qd=rng.normal(size=(n, N)).astype(np.float32),
             a=rng.normal(size=(n, NPOL)).astype(np.float32), imu=rng.normal(size=(n, 6)).astype(np.float32),
             osc=rng.random(n).astype(np.float32), touch=rng.random((n, M)).astype(np.float32),
             contact=rng.random((n, M)) > 0.5, ctx=ctx, ev=np.zeros(n, int),
             pose=rng.normal(size=(n, 3)).astype(np.float32),
             terrain=(0.05 * rng.normal(size=(n, LL.SCAN_DIM))).astype(np.float32),
             terrain_valid=np.ones((n, LL.SCAN_DIM), bool), com_support=rng.normal(size=n).astype(np.float32),
             com_support_valid=np.ones(n, bool))
    if foothold:
        fc = np.full((n, M), -2)
        fc[:, 0] = rng.integers(0, LL.SCAN_DIM, n)                # the left foot is swinging toward a scan cell
        fc[:, 1] = -1
        d["foothold_cell"] = fc
    if not legacy:
        d["upper"] = rng.normal(size=(n, N - NPOL)).astype(np.float32)
        d["upper_valid"] = np.full(n, wholebody)
    (root / "tinybody").mkdir(parents=True)
    np.savez(root / "tinybody" / "s0.npz", **d)
    eps = [dict(seed=(1, 2, 19)[i], status="ok", **(dict(waypoints=dict(a=[1.0, 0.0], b=[2.0, 0.0])) if legacy else
                                                     dict(control="wholebody" if wholebody else "base_velocity")))
           for i in range(n_eps)]
    meta = dict(episodes=eps) if legacy else dict(task=task, task_view=T.task_view_of(task).as_dict(), episodes=eps)
    (root / "tinybody" / "s0.json").write_text(json.dumps(meta))
    return d


def _cfg(root, factors, steps=3):
    return dict(name="hd2", data=str(root), bodies=["tinybody"], steps=steps, batch_size=8, ckpt_every=1000,
                latent=dict(dz=DZ, width=D, beta_kl=1e-3, factors=factors))


def test_event_goal_slots_come_from_the_task_graph():
    assert T.event_goal_slots("waypoint_contact") == (0, 1, 1, -1)       # walk_to_a, walk_to_b, halt (reference b), done
    assert T.event_goal_slots("h_reach") == (0, -1, -1, -1)
    assert T.event_goal_slots("h_loco_pick") == (0, 1, -1, -1)           # walk to the crate, then pick the box
    assert T.event_goal_slots("h_place") == (0, 2, -1, -1)               # pick the box, place at the mark (slot 2)


def test_h_reach_pack_is_task_agnostic_and_covers_the_upper_rows(tmp_path):
    d = _write_pack(tmp_path / "d", "h_reach", wholebody=True)
    data = T.LeggedData(tmp_path / "d", ["tinybody"], torch.device("cpu"))          # no `waypoints` in the episodes
    assert data.upper_trained and data.action_groups == ["legs", "upper"]
    am = data.A["amask"]
    assert am[:, :NPOL + 5].all() and not am[:, NPOL + 5:].any()
    assert np.allclose(data.A["a"][:, NPOL:NPOL + 5].numpy(), d["upper"])
    i = torch.arange(0, 6)
    lab = data.labels(i)
    assert lab["goal_valid"].all() and torch.allclose(lab["goal"], data.A["ctx"][i][:, 8:10])
    tgt = data.beh(i)                                                                # target chunks carry the upper rows
    assert tgt.shape[1] == T.MAX_N and tgt[:, NPOL:NPOL + 5].abs().sum() > 0
    bb = TB.bc_batch(data, i)
    assert "foothold_cell" in bb[0] and bb[2][:, NPOL:NPOL + 5].all()


@pytest.mark.slow
def test_rep_flow_bc_three_steps_on_an_h_reach_pack(tmp_path):
    _write_pack(tmp_path / "d", "h_reach", wholebody=True)
    cfg = _cfg(tmp_path / "d", [PROBES, "preset:legged"])
    res = T.train_rep(cfg, tmp_path / "rep")
    assert res["upper_trained"] is True and res["action_groups"] == ["legs", "upper"]
    assert res["eval"]["realize_mse_by_group"]["upper"] > 0 and res["eval"]["realize_mse_by_group"]["legs"] > 0
    st = torch.load(tmp_path / "rep" / "representation.pt", weights_only=False)
    assert st["result"]["upper_trained"] is True
    _, E, R, P, rres, specs = T.load_legged_rep(tmp_path / "rep" / "representation.pt", torch.device("cpu"))
    data = T.LeggedData(tmp_path / "d", ["tinybody"], torch.device("cpu"))
    i = data.sample(8, np.random.default_rng(0))
    j = torch.randint(0, 5, (8,))
    _, logs, _ = T.rep_step(E, R, P, data, i, j, specs, 1e-3, train=False)
    assert logs["real_upper"] > 0 and logs["real_legs"] > 0
    fres = T.train_flow(dict(representation=str(tmp_path / "rep" / "representation.pt"), steps=3, batch_size=8, width=D,
                             layers=1, semantic_weight=0.5), tmp_path / "flow")
    assert fres["upper_trained"] is True and fres["action_groups"] == ["legs", "upper"]
    bcfg = dict(name="hd2bc", data=str(tmp_path / "d"), bodies=["tinybody"], steps=3, batch_size=8, ckpt_every=1000,
                snap_every=10 ** 6, model=dict(width=D, enc_layers=1, dec_layers=1, factors=["preset:legged"]))
    bres = TB.train(bcfg, tmp_path / "bc")
    assert bres["upper_trained"] is True and bres["eval"]["chunk_mse"] > 0


def test_h_steps_pack_supervises_the_foothold_estimate_and_leaves_upper_untrained(tmp_path):
    _write_pack(tmp_path / "d", "h_steps", wholebody=False)
    data = T.LeggedData(tmp_path / "d", ["tinybody"], torch.device("cpu"))
    assert not data.upper_trained and data.action_groups == ["legs"] and not data.A["amask"][:, NPOL:].any()
    cfg = _cfg(tmp_path / "d", [PROBES, "preset:legged"])
    res = T.train_rep(cfg, tmp_path / "rep")
    assert res["upper_trained"] is False and res["action_groups"] == ["legs"]
    assert res["eval"]["realize_mse_by_group"]["upper"] is None
    _, E, R, P, _, specs = T.load_legged_rep(tmp_path / "rep" / "representation.pt", torch.device("cpu"))
    i = data.sample(8, np.random.default_rng(0))
    b = data.train_batch(i)
    _, _, rc = E.encode(b, data.beh(i))
    el, logs, metrics = estimates_loss(rc, specs)
    assert float(el) > 0 and any("foothold" in k for k in logs) and any(k.startswith("leg.foothold") for k in metrics)


def test_bc_three_steps_on_an_h_reach_pack_legged_none(tmp_path):
    """The fast (legged-none) BC path on the wholebody pack: the upper rows are in the loss and the result declares them."""
    _write_pack(tmp_path / "d", "h_reach", wholebody=True)
    bres = TB.train(dict(name="hd2bc", data=str(tmp_path / "d"), bodies=["tinybody"], steps=3, batch_size=8,
                         ckpt_every=1000, snap_every=10 ** 6, model=dict(width=D, enc_layers=1, dec_layers=1)),
                    tmp_path / "bc")
    assert bres["upper_trained"] is True and bres["action_groups"] == ["legs", "upper"] and bres["eval"]["chunk_mse"] > 0


def test_old_waypoint_pack_still_trains_and_labels_its_goal_from_the_context(tmp_path):
    _write_pack(tmp_path / "d", None, wholebody=False, legacy=True)
    data = T.LeggedData(tmp_path / "d", ["tinybody"], torch.device("cpu"))
    assert not data.upper_trained
    lab = data.labels(torch.arange(4))
    assert lab["goal_valid"].all()                                # event 0 of waypoint_contact: waypoint_a, slot 0
    res = T.train_rep(_cfg(tmp_path / "d", [PROBES]), tmp_path / "rep")
    assert res["upper_trained"] is False and res["action_groups"] == ["legs"]


@pytest.mark.parametrize("stage", ["rep", "flow", "bc"])
def test_relgen_curriculum_is_refused_not_ignored(tmp_path, stage):
    _write_pack(tmp_path / "d", "h_reach", wholebody=True)
    cfg = dict(_cfg(tmp_path / "d", [PROBES]), curriculum={"factors": ["leg.foothold"]})
    with pytest.raises(RelgenError, match="legged"):
        {"rep": T.train_rep, "flow": T.train_flow, "bc": TB.train}[stage](cfg, tmp_path / "out")
