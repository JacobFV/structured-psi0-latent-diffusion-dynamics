"""Readiness R2 unit PC (pointer closure): the collector runs on `harness.rollout` and keeps its demos byte-identical; the
teacher's drag target is stored per episode and is what `ui.drag_to` is supervised with (no net proxy); `--curriculum` mixes
relgen rows through `relation_batches` and refuses shards that carry no pointer public inputs; the checkpoint stamps / checks its
factor structure. CPU, tiny packs; the CW tests need the `computerworld` extra."""
from __future__ import annotations

import hashlib
import json
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rrp.harness.train import pointer as T  # noqa: E402
from tests.unit.test_pointer_train_smoke import GEOM, _args, _geometry, _pack  # noqa: E402,F401  (_geometry: autouse fixture)


def _digest(ep):
    h = hashlib.sha256()
    for r in ep["rows"]:
        for k in sorted(r):
            h.update(k.encode())
            h.update(np.asarray(r[k]).tobytes())
    for t in ep["tabs"]:
        for k in sorted(t):
            h.update(np.asarray(t[k]).tobytes())
    h.update(json.dumps(ep["goal"], sort_keys=True).encode())
    h.update(ep["instr"].encode())
    return h.hexdigest()[:16], len(ep["rows"])


# ------------------------------------------------------------------------------------------------ collect on harness.rollout
@pytest.mark.computerworld
def test_collect_on_rollout_reproduces_the_frozen_demos_byte_for_byte():
    """Digests recorded from the pre-rollout collector (same seeds, DART noise and rng stream): the port changed the
    driver, not one demo byte."""
    from rrp.harness.train.pointer.collect import collect_episode
    a = collect_episode("cw/calc_sum", 3, dart_px=0.0, rng=random.Random(0))
    b = collect_episode("cw/open_type", 5, dart_px=4.0, rng=random.Random(0), env_kw={"strings": "procedural"})
    assert _digest(a) == ("c2c83c9c713bc7ce", 30)
    assert _digest(b) == ("81dfc0a7eb242537", 19)


@pytest.mark.computerworld
def test_the_demo_teacher_is_a_marked_privileged_policy_and_runs_through_the_rollout_harness():
    from rrp.harness.train.pointer import collect as C
    pol = C.DemoTeacher("cw/calc_sum", dart_px=0.0, rng=random.Random(0))
    assert pol.info.source == "scripted_teacher" and pol.info.requires.privileged
    import inspect
    assert "rollout(" in inspect.getsource(C.collect_episode)            # the one rollout harness, not a private loop


@pytest.mark.computerworld
def test_the_pack_stores_the_teachers_drag_and_drag_to_is_supervised_with_it(tmp_path):
    from rrp.envs.base import make_env
    from rrp.harness.train.pointer.collect import collect_episode, write_pack
    from rrp.policies.pointer import PointerGeometry
    kw = dict(dart_px=0.0, rng=random.Random(0), env_kw={"strings": "procedural"})
    drag, other = (collect_episode(t, s, **kw) for t, s in (("cw/drag_window", 1), ("cw/calc_sum", 3)))
    assert drag and other and other["drag"] is None
    env = make_env("computerworld", task="cw/drag_window", body="cw_pointer", seed=1, strings="procedural")
    geom = PointerGeometry.from_spec(env.spec)
    env.close()
    drag["seed"], other["seed"] = 1, 3
    write_pack(tmp_path / "d.npz", [drag], "cw/drag_window", geom)
    write_pack(tmp_path / "o.npz", [other], "cw/calc_sum", geom)
    data = T.Demos([str(tmp_path / "d.npz"), str(tmp_path / "o.npz")], "cpu", geom)
    assert data.has_drag
    n = len(drag["rows"])
    y, valid = data.drag_labels(torch.arange(data.N))
    s, d = drag["drag"]["src_slot"], drag["drag"]["dst_slot"]
    tab0 = data.t["wmask"][data.t["tab"][:n]]
    assert (y[:n, s, d, 0] == tab0[:, s].float() * tab0[:, d].float()).all()          # the pair, where both widgets are present
    assert y[:n].sum() == float((tab0[:, s] & tab0[:, d]).sum()) and valid[:n].any()
    assert y[n:].sum() == 0 and valid[n:].any()                                        # calc_sum never drags: a real negative


# ------------------------------------------------------------------------------------------------ Demos.drag_labels
def test_drag_labels_true_pair_negatives_and_no_label_when_the_target_is_absent(tmp_path):
    p = tmp_path / "p.npz"
    _pack(p, geometry=True, seed0=10)                                     # widgets 0..4 present; even episodes drag 1 -> 3
    data = T.Demos([str(p)], "cpu", GEOM)
    ix = torch.arange(data.N)
    y, valid = data.drag_labels(ix)
    ep = data.t["ep"][ix]
    drags = (ep % 2 == 0)
    assert y.shape[1:] == (valid.shape[1], valid.shape[2], 1)
    assert (y[drags, 1, 3, 0] == 1).all() and y[drags].sum() == drags.sum()         # exactly one true pair per dragging tick
    assert y[~drags].sum() == 0 and valid[~drags][:, :5, :5].all()                   # teacher never drags here: valid negatives
    assert not valid[:, 5:, :].any() and not valid[:, :, 5:].any()                   # absent widgets are never candidates
    z = dict(np.load(p))                                                             # drag handle 7 is not in the table
    z["ep_drag"] = np.where(z["ep_drags"][:, None], np.array([7, 3]), -1).astype(np.int16)
    np.savez_compressed(tmp_path / "gone.npz", **z)
    g = T.Demos([str(tmp_path / "gone.npz")], "cpu", GEOM)
    y2, v2 = g.drag_labels(torch.arange(g.N))
    assert y2.sum() == 0 and not v2[drags].any() and v2[~drags].any()               # absent target: no label, never a negative


def test_an_old_pack_without_the_drag_has_no_drag_labels(tmp_path):
    p = tmp_path / "p.npz"
    _pack(p, geometry=True, seed0=10)
    z = dict(np.load(p))
    z.pop("ep_drag"), z.pop("ep_drags")
    np.savez_compressed(tmp_path / "old.npz", **z)
    data = T.Demos([str(tmp_path / "old.npz")], "cpu", GEOM)
    assert not data.has_drag
    with pytest.raises(ValueError, match="ep_drag"):
        data.drag_labels(torch.arange(4))


def test_the_drag_to_loss_reads_the_teacher_label_not_a_proxy(tmp_path, monkeypatch):
    """Red/green: flip the teacher's labels and the drag_to probe loss the trainer logs changes; without any label the
    trainer refuses instead of falling back to the net's own focused-widget proxy."""
    from rrp.harness.train.pointer import train as TR
    from rrp.policies.relations.base import FactorError
    p = tmp_path / "p.npz"
    _pack(p, geometry=True, seed0=10)
    fx = ["preset:ui"]

    def loss_with(label_fn, name):
        if label_fn:
            monkeypatch.setattr(T.data.Demos if hasattr(T, "data") else __import__("rrp.harness.train.pointer.data",
                                fromlist=["Demos"]).Demos, "drag_labels", label_fn)
        T.cmd_bc(_args(tmp_path, [p], out=str(tmp_path / name), w_xy=5.0, factors=fx, seed=0))
        return torch.load(tmp_path / name, weights_only=False)["metrics"]["fx_probe_ui.drag_to"]

    from rrp.harness.train.pointer.data import Demos
    real = Demos.drag_labels
    base = loss_with(None, "a.pt")
    assert base == loss_with(None, "a2.pt")                               # deterministic at a fixed seed
    flipped = loss_with(lambda self, ix: (1.0 - real(self, ix)[0], real(self, ix)[1]), "b.pt")
    assert np.isfinite(flipped) and flipped != base                       # the teacher label is what the head learns from
    monkeypatch.setattr(Demos, "drag_labels", real)
    specs = TR.factor_specs(_args(tmp_path, [p], out="x", factors=fx), Demos([str(p)], "cpu", GEOM))
    ctx = type("C", (), {"last_rc": object()})()
    with pytest.raises(FactorError, match="teacher's drag label"):
        TR.factor_loss(ctx, specs, None)


# ------------------------------------------------------------------------------------------------ relation_batches stream
CURRICULUM = json.dumps({"shards": "relgen_shards", "interval": 1, "ramp_steps": 10})
MIX = [json.dumps({"name": "ui.drag_to", "mix": 0.5})]


def test_relation_batches_refuses_shards_without_policy_inputs(tmp_path, monkeypatch):
    from rrp.harness.data.relgen import Label
    from rrp.harness.train.pointer import relmix
    from rrp.policies.relations.base import FactorError
    p = tmp_path / "p.npz"
    _pack(p, geometry=True, seed0=10)
    row = dict(inputs={}, labels={"drag_to": Label(value=np.zeros((5, 5, 1)), valid=np.ones((5, 5), bool), prov="gt", version="2")},
               provenance={})
    monkeypatch.setattr(relmix, "load_shard_rows", lambda *a: [row])
    with pytest.raises(FactorError, match="no pointer public inputs"):
        T.cmd_bc(_args(tmp_path, [p], out=str(tmp_path / "x.pt"), w_xy=5.0, factors=MIX, curriculum=CURRICULUM))


def test_relation_batches_mixes_relgen_rows_that_carry_the_pointer_inputs(tmp_path, monkeypatch):
    import rrp.harness.data.mix as mix
    from rrp.harness.data.relgen import Label
    from rrp.harness.train.pointer import relmix
    p = tmp_path / "p.npz"
    _pack(p, geometry=True, seed0=10)
    data = T.Demos([str(p)], "cpu", GEOM)
    ix = torch.arange(6)
    b, _, _ = data.batch(ix)
    y, v = data.drag_labels(ix)
    rows = [dict(inputs={k: t[i].numpy() for k, t in b.items()},
                 labels={"drag_to": Label(value=y[i].numpy(), valid=v[i].numpy(), prov="gt", version="2")}, provenance={"seed": i})
            for i in range(6)]
    monkeypatch.setattr(relmix, "load_shard_rows", lambda *a: rows)
    monkeypatch.setattr(mix, "load_shard_rows", lambda *a: rows)
    out = tmp_path / "bc.pt"
    a = _args(tmp_path, [p], out=str(out), w_xy=5.0, factors=MIX, curriculum=CURRICULUM, steps=4, batch=8)
    T.cmd_bc(a)
    sched = [json.loads(x) for x in (tmp_path / "bc.relgen" / "schedule.jsonl").read_text().splitlines()]
    assert len(sched) == 4 and all("share" in s for s in sched)              # one decision per step, written next to the checkpoint
    m = torch.load(out, weights_only=False)["metrics"]
    assert np.isfinite(m["fx_probe_ui.drag_to"]) and np.isfinite(m["rel_fx_probe_ui.drag_to"])   # the relgen rows were trained on


def test_a_mix_factor_without_a_curriculum_is_refused(tmp_path):
    from rrp.policies.relations.base import FactorError
    p = tmp_path / "p.npz"
    _pack(p, geometry=True, seed0=10)
    with pytest.raises(FactorError, match="--curriculum"):
        T.cmd_bc(_args(tmp_path, [p], out=str(tmp_path / "x.pt"), w_xy=5.0, factors=MIX))
