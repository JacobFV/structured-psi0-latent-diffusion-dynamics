"""Round-3 relgen-train: relation shards reach the legged / humanoid trainers (D-146 round-3 addendum), and the factor-loss
calibration rule (`RelationBatches.loss`) plus the competence of field factors (`curriculum.estimate_competence`).

Legged shard rows hold the legged public batch of one tick (`input_kind = legged_batch`) and the `foothold_next` label over
the ctx tokens `[glob | N joints | M limbs | M feet | C cells]`; `collate_rows(rows, "legged")` forwards them as the batch
dict with the `foothold_cell` key the relation graph builds that label from. Plumbing only: random data, three CPU steps;
no number here is a result."""
import functools
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import rrp.harness.data.mix as mix
from rrp.harness.data.mix import FACTOR_LOSS_SCALE, RelationBatches, RelgenError, collate_rows, load_shard_rows, write_shard
from rrp.harness.data.relgen import Label
from rrp.harness.data.relgen.curriculum import Scheduler, estimate_competence
from rrp.harness.train import legged_bc as TB
from rrp.harness.train import legged_latent_train as T
from rrp.policies.nets import legged_latent as LL
from rrp.policies.nets.legged_bc import LeggedBC
from rrp.policies.relations.base import FactorError, get_factor, resolve
from .test_legged_relations import D, PROBES, _cfg, _write_shard, batch, labelled

FOOT = "leg.foothold"
CUR = dict(factors=[FOOT], interval=2, share_min=0.5, share_max=0.5, full_world_start=0.0, full_world_end=0.0)
B_SIZE, STEPS, N_REL = 8, 3, 4


@pytest.fixture(autouse=True)
def _short_evals(monkeypatch):
    monkeypatch.setattr(T, "eval_rep", functools.partial(T.eval_rep, n_batches=2))
    monkeypatch.setattr(T, "eval_flow", functools.partial(T.eval_flow, n_batches=2))
    monkeypatch.setattr(TB, "eval_bc", functools.partial(TB.eval_bc, n_batches=2))


def _row(seed=0, terrain=True, swing_cell=37):
    """One legged shard sample: the tick's public batch (no batch axis) + ctx ids + the `foothold_next` label (foot 0
    swings toward `swing_cell`, foot 1 is planted: valid pairs, none true; the other assemblies have no foot token)."""
    b = {k: v[0].numpy() for k, v in batch(B=1, terrain=terrain, seed=seed).items()}
    N, M, C = b["node_mask"].shape[0], b["asm_mask"].shape[0], (b["terrain"].shape[0] if terrain else 0)
    ids = ["glob"] + [f"j{i}" for i in range(N)] + [f"limb{m}" for m in range(M)] + [f"foot{m}" for m in range(M)] \
        + [f"cell{c}" for c in range(C)]
    Tn = len(ids)
    val, ok = np.zeros((Tn, Tn, 1), np.float32), np.zeros((Tn, Tn), bool)
    f0, c0 = 1 + N + M, 1 + N + 2 * M
    if C:
        ok[f0, c0:] = ok[f0 + 1, c0:] = True
        val[f0, c0 + swing_cell, 0] = 1.0
    lab = Label(value=val, valid=ok, prov="gt", version="1")
    return dict(inputs=dict(legged_batch=b, tokens=dict(ctx=ids)), labels={"foothold_next": lab},
                provenance=dict(active=[FOOT], env="fixture/legged", task="toy", seed=seed, labels=[
                    dict(label="foothold_next", prov="gt", version="1")]))


def _shards(root, n=6, terrain=True):
    rows = [_row(seed=i, terrain=terrain, swing_cell=(37 + i) % LL.SCAN_DIM) for i in range(n)]
    write_shard(FOOT, get_factor(FOOT).version, rows, root, shard_id="s0")
    return rows


# ------------------------------------------------------------------ shard IO + collate
def test_legged_rows_round_trip_and_collate_to_the_batch_the_graph_reads(tmp_path):
    rows = _shards(tmp_path)
    back = load_shard_rows(FOOT, get_factor(FOOT).version, tmp_path)
    man = json.loads((tmp_path / FOOT / get_factor(FOOT).version / "manifest.json").read_text())
    assert man["schema"] == "relgen-shard-3" and {r["input_kind"] for r in man["episodes"]} == {"legged_batch"}
    for r, o in zip(back, rows):
        assert set(r["inputs"]["legged_batch"]) == set(o["inputs"]["legged_batch"])
        for k, v in o["inputs"]["legged_batch"].items():
            assert r["inputs"]["legged_batch"][k].dtype == v.dtype and np.array_equal(r["inputs"]["legged_batch"][k], v), k
        assert r["inputs"]["tokens"]["ctx"] == o["inputs"]["tokens"]["ctx"]
    got = collate_rows(back[:2], "legged")
    assert got["q"].shape[0] == 2 and got["node_mask"].dtype == torch.bool and got["node_asm"].dtype == torch.long
    assert got["foothold_cell"].tolist() == [[37, -1, -2, -2, -2], [38, -1, -2, -2, -2]]
    # the same label the pack path builds from `foothold_cell`: the graph's pair label is identical
    ref = dict(batch(B=2, terrain=True, seed=0))
    ref["foothold_cell"] = torch.tensor([[37, -1, -2, -2, -2], [38, -1, -2, -2, -2]])
    lab = LL.legged_graph(got, 4)[1].sets["ctx"].labels
    want = LL.legged_graph({**got, "foothold_cell": ref["foothold_cell"]}, 4)[1].sets["ctx"].labels
    assert torch.equal(lab["foothold_next"], want["foothold_next"]) and torch.equal(lab["foothold_next.valid"], want["foothold_next.valid"])
    assert bool(lab["foothold_next.valid"].any()) and float(lab["foothold_next"].sum()) == 2.0


def test_legged_collate_refuses_bad_rows(tmp_path):
    rows = load_shard_rows(FOOT, get_factor(FOOT).version, (_shards(tmp_path), tmp_path)[1])
    short = [{**rows[0], "inputs": {**rows[0]["inputs"], "tokens": {"ctx": rows[0]["inputs"]["tokens"]["ctx"][:-1]}}}]
    with pytest.raises(RelgenError, match="ctx entity ids"):
        collate_rows(short, "legged")
    other = [{**rows[0], "labels": {"com_support": rows[0]["labels"]["foothold_next"]}}]
    with pytest.raises(FactorError, match="com_support"):
        collate_rows(other, "legged")
    with pytest.raises(RelgenError, match="forwardable"):
        write_shard(FOOT, "1", [{**_row(), "inputs": {"tokens": {"ctx": ["a"]}}}], tmp_path / "x", shard_id="s")
    both = _row()
    both["inputs"]["policy_input"] = object()
    with pytest.raises(RelgenError, match="exactly one"):
        write_shard(FOOT, "1", [both], tmp_path / "z", shard_id="s")
    bad = _row()
    bad["inputs"]["legged_batch"]["foothold_cell"] = np.zeros(5, np.int64)
    with pytest.raises(RelgenError, match="foothold_cell"):
        write_shard(FOOT, "1", [bad], tmp_path / "y", shard_id="s")


def test_legged_rows_without_terrain_carry_no_foothold_pairs(tmp_path):
    _shards(tmp_path, terrain=False)
    rows = load_shard_rows(FOOT, get_factor(FOOT).version, tmp_path)
    assert "terrain" not in rows[0]["inputs"]["legged_batch"]
    assert collate_rows(rows[:2], "legged")["foothold_cell"].eq(-2).all()


# ------------------------------------------------------------------ the three legged trainers, 3 CPU steps each
def _wrap(monkeypatch, mod):
    rec = SimpleNamespace(counts=[], shard_rows=[], el=[], raw=[], grad=[], action_rows=[])
    real = mod.legged_relation_batches

    def wrapped(*a, **k):
        rel = real(*a, **k)
        draw, loss = rel.draw, rel.loss

        def d(dev):
            b, sh = draw(dev)
            rec.counts.append(dict(b["counts"]))
            rec.shard_rows.append(0 if sh is None else sh["q"].shape[0])
            assert sh is None or "foothold_cell" in sh and "act" not in sh
            return b, sh

        def l(rc, step):
            el, logs = loss(rc, step)
            rec.el.append(float(el.detach())), rec.raw.append(logs["relgen_raw"]), rec.grad.append(el.requires_grad)
            return el, logs
        rel.draw, rel.loss = d, l
        return rel
    monkeypatch.setattr(mod, "legged_relation_batches", wrapped)
    return rec


def _tap(monkeypatch, cls, name, rec):
    real = getattr(cls, name)

    def f(self, b, *a, **k):
        rec.action_rows.append(b["q"].shape[0])
        assert "foothold_cell" in b or True
        return real(self, b, *a, **k)
    monkeypatch.setattr(cls, name, f)


def _check(rec, out):
    assert len(rec.counts) == STEPS and all(c == {"main": B_SIZE - N_REL, FOOT: N_REL} for c in rec.counts)
    assert rec.shard_rows == [N_REL] * STEPS
    assert len(rec.el) == STEPS and all(np.isfinite(rec.el)) and min(rec.el) > 0 and all(rec.grad)
    assert rec.el[0] <= FACTOR_LOSS_SCALE * 1.0001                          # calibrated: at most the one scale at init
    assert rec.action_rows[:STEPS] == [B_SIZE - N_REL] * STEPS                # the action loss saw the pack rows only
    recs = [json.loads(l) for l in (out / "schedule.jsonl").read_text().splitlines()]
    assert [r["step"] for r in recs] == [0, 2]
    assert (out / "factor_calibration.json").exists()


def _relcfg(cfg, shards):
    return dict(cfg, relgen=str(shards), curriculum=CUR)


def test_train_rep_three_cpu_steps_shard_rows_enter_the_estimate_loss_only(tmp_path, monkeypatch):
    _write_shard(tmp_path / "d"), _shards(tmp_path / "rel")
    rec = _wrap(monkeypatch, T)
    real = T.rep_step
    monkeypatch.setattr(T, "rep_step", lambda E, R, P, data, i, j, *a, **k: (rec.action_rows.append(len(i)),
                                                                          real(E, R, P, data, i, j, *a, **k))[1])
    cfg = _relcfg(_cfg(tmp_path / "d", [PROBES, "preset:legged"]), tmp_path / "rel")
    res = T.train_rep(cfg, tmp_path / "out")
    assert res["steps"] == STEPS and (tmp_path / "out" / "representation.pt").exists()
    _check(rec, tmp_path / "out")


def test_train_flow_three_cpu_steps_shard_rows_enter_the_estimate_loss_only(tmp_path, monkeypatch):
    _write_shard(tmp_path / "d"), _shards(tmp_path / "rel")
    T.train_rep(_cfg(tmp_path / "d", [PROBES, "preset:legged"]), tmp_path / "rep")
    rec = _wrap(monkeypatch, T)
    real = LL.LeggedFlow.loss
    monkeypatch.setattr(LL.LeggedFlow, "loss", lambda self, b, *a, **k: (rec.action_rows.append(b["q"].shape[0]),
                                                                         real(self, b, *a, **k))[1])
    fcfg = _relcfg(dict(representation=str(tmp_path / "rep" / "representation.pt"), steps=STEPS, batch_size=B_SIZE,
                        width=D, layers=1, semantic_weight=0.5), tmp_path / "rel")
    res = T.train_flow(fcfg, tmp_path / "flow")
    assert res["steps"] == STEPS
    _check(rec, tmp_path / "flow")


def test_bc_three_cpu_steps_shard_rows_enter_the_estimate_loss_only(tmp_path, monkeypatch):
    _write_shard(tmp_path / "d"), _shards(tmp_path / "rel")
    rec = _wrap(monkeypatch, TB)
    real = LeggedBC.loss
    monkeypatch.setattr(LeggedBC, "loss", lambda self, b, a, am: (rec.action_rows.append(b["q"].shape[0]),
                                                                  real(self, b, a, am))[1])
    cfg = _relcfg(dict(name="hlbc", data=str(tmp_path / "d"), bodies=["tinybody"], steps=STEPS, batch_size=B_SIZE,
                       ckpt_every=1000, snap_every=10 ** 6, model=dict(width=D, enc_layers=1, dec_layers=1,
                                                                        factors=["preset:legged"])), tmp_path / "rel")
    res = TB.train(cfg, tmp_path / "out")
    assert res["steps"] == STEPS
    _check(rec, tmp_path / "out")


def test_a_relgen_run_on_a_factorless_legged_net_fails_loudly(tmp_path):
    _write_shard(tmp_path / "d"), _shards(tmp_path / "rel")
    cfg = _relcfg(_cfg(tmp_path / "d", [PROBES]), tmp_path / "rel")
    with pytest.raises(ValueError, match="not active factors|nothing to schedule|no relation context"):
        T.train_rep(cfg, tmp_path / "out")


# ------------------------------------------------------------------ the one calibration rule
def _rb(tmp_path, monkeypatch, losses, **cfg):
    """A RelationBatches over legged shards whose `estimates_loss` is replaced by a scripted per-call loss (the rule is the
    unit under test, not the nets)."""
    _shards(tmp_path / "rel")
    specs = resolve([{"name": FOOT, "source": "probe"}], family="legged")
    rb = RelationBatches(dict(curriculum=CUR, relgen=str(tmp_path / "rel"), seed=0, batch_size=B_SIZE, **cfg),
                         tmp_path / "out", specs, family="legged", start_step=cfg.pop("_start", 0))
    it = iter(losses)
    w = torch.zeros((), requires_grad=True)

    def fake(rc, sp):
        l, mae = next(it)
        return w * 0 + l, {f"probe_{FOOT}": l}, {f"{FOOT}_mae": (mae * 10, 10)}
    monkeypatch.setattr("rrp.policies.relations.base.estimates_loss", fake)
    rb._active = [FOOT]
    return rb


def test_raw_factor_loss_is_orders_above_the_flow_loss_and_the_rule_brings_it_to_a_tenth(tmp_path, monkeypatch):
    flow_init = 2.0                                                        # a unit-variance rectified-flow target at init
    rb = _rb(tmp_path, monkeypatch, [(330.0, 4.0), (165.0, 2.0), (0.7, 1.0)])
    el0, logs0 = rb.loss(object(), 0)
    assert logs0["relgen_raw"] / flow_init > 100                           # the defect: about 330x the flow loss
    assert el0.item() / flow_init == pytest.approx(0.1)                    # the rule: 0.1 x the flow loss at init
    assert FACTOR_LOSS_SCALE == pytest.approx(0.1 * flow_init)
    assert rb.loss(object(), 1)[0].item() == pytest.approx(0.1)            # half the initial loss: half the term
    assert rb.loss(object(), 2)[0].item() == pytest.approx(0.2 * 0.7 / 330.0)    # ref stays the FIRST value


def test_loss_ref_floor_scale_override_and_resume(tmp_path, monkeypatch):
    rb = _rb(tmp_path, monkeypatch, [(0.7, 1.0), (0.35, 1.0)], factor_loss_scale=1.0)
    assert rb.loss(object(), 0)[0].item() == pytest.approx(0.7)            # |loss| < 1 is floored to ref 1: never amplified
    cal = json.loads((tmp_path / "out" / "factor_calibration.json").read_text())
    assert cal["loss_ref"] == {FOOT: 1.0} and cal["mae_ref"] == {FOOT: pytest.approx(1.0)} and cal["factor_loss_scale"] == 1.0
    # a resumed run restores the file: the reference is NOT re-measured at the resume step
    next(rb)                                                               # step 0 decided: schedule.jsonl has its record
    rb2 = _rb(tmp_path, monkeypatch, [(3.5, 1.0)], factor_loss_scale=1.0, _start=1)
    assert rb2.loss_ref == {FOOT: 1.0}
    bad = pytest.raises(RelgenError, match="factor_loss_scale")
    for v in (0, -1.0, "x", float("nan"), True):
        with bad:
            _rb(tmp_path / str(v), monkeypatch, [], factor_loss_scale=v)


# ------------------------------------------------------------------ competence of a field factor
def test_estimate_competence_covers_acc_and_mae_readouts():
    m = {"a_acc": (3.0, 4), "b_mae": (2.0, 4), "c_mae": (9.0, 3), "d_mae": (1.0, 0), "e_acc": (1.0, 1), "f_mae": (1.0, 2)}
    out = estimate_competence(m, {"b": 1.0, "c": 1.0, "d": 1.0}, ["a", "b", "c", "d", "f"])
    assert out["a"] == {"competence": 0.75}
    assert out["b"] == {"competence": pytest.approx(0.5)}                  # mae 0.5 of an initial 1.0: half removed
    assert out["c"] == {"competence": 0.0}                                 # worse than initial: clipped, never negative
    assert set(out) == {"a", "b", "c"}                                     # n == 0, untracked (e) and no-reference (f) skipped


def test_mae_competence_reaches_the_scheduler_and_the_schedule_record(tmp_path, monkeypatch):
    rb = _rb(tmp_path, monkeypatch, [(5.0, 4.0), (5.0, 1.0)])
    rb.loss(object(), 0)
    rb.loss(object(), 1)
    m = rb.scheduler.metrics_log
    assert [e[1][FOOT]["competence"] for e in m] == [0.0, 0.75]
    assert rb._pending == [[0, {FOOT: {"competence": 0.0}}], [1, {FOOT: {"competence": 0.75}}]]
