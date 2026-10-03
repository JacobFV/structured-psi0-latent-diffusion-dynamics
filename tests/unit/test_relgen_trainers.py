"""RG (readiness round 2): relgen shards reach the arm trainers (docs/architecture.md 14.2, docs/relations.md 5.5).

A tiny fixture pack (two scripted-teacher episodes) and tiny relgen shards (real fixture sessions, the real featurizer and
the real `geo.pos3d` label) drive 3 CPU steps of each arm trainer (`train_representation`, `train_latent_flow`,
`train_policy`) with `params.curriculum` on: a mix row's factor loss is non-zero and differentiable, its action loss is
zero (shard rows never enter the flow / realizer loss), `schedule.jsonl` is written, and `Scheduler.replay_records`
reproduces the batch composition the run drew.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import rrp.harness.train.behavior as behavior
import rrp.harness.train.latent_train as latent_train
from rrp.envs.mujoco.fixtures import make_pick_place_session
from rrp.harness.data.collect import collect_teacher_episode, write_episode
from rrp.harness.data.manifest import write_manifest
from rrp.harness.data.mix import FACTOR_LOSS_SCALE, RelgenError, collate_rows, load_shard_rows
from rrp.harness.data.packed import pack_dataset
from rrp.harness.data.relgen import load_families
from rrp.harness.data.relgen.curriculum import Scheduler
from rrp.harness.pipelines.relations import SnapshotCollector, _write_all
from rrp.policies.nets.batch import BANKS
from rrp.policies.nets.flow import FlowPolicy
from rrp.policies.relations import catalog  # noqa: F401  (registers geo.pos3d & co)
from rrp.policies.relations.base import spec

load_families()
B_SIZE, STEPS, N_REL = 8, 3, 4
CUR = dict(factors=["geo.pos3d"], interval=2, share_min=0.5, share_max=0.5, full_world_start=0.0, full_world_end=0.0)
POS3D = {"name": "geo.pos3d", "source": "probe"}         # `given` (the default) has no readout head: nothing to supervise
LATENT = dict(width=32, heads=2, ctx_layers=1, enc_layers=1, knots=2, knot_times=(0.1, 0.3), dz=8, horizon=6,
              realizer_layers=1, max_phase_ticks=2, name="tiny", encoder_factors=["preset:arm", POS3D])
POLICY = dict(width=32, heads=2, ctx_layers=1, blocks=1, factors=["preset:arm", POS3D])


def _collect(seeds_distractors):
    seeds = [sd for sd, _ in seeds_distractors]
    col = SnapshotCollector("fixture/arm", "pick_place", [spec("geo.pos3d")], seeds, rng=np.random.default_rng(0),
                            every=2, max_per_episode=3)
    for i, (sd, nd) in enumerate(seeds_distractors):
        s = make_pick_place_session(seed=sd, n_distractors=nd)
        col.on_reset(i, s, s.observe())
        for _ in range(4):
            col.on_step(i, s, None, SimpleNamespace(observation=s.observe()))
    return col.by_factor


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("relgen_trainers")
    metas = [write_episode(collect_teacher_episode(make_pick_place_session(seed=sd), max_steps=12, episode_id=f"e{sd}"),
                           root / "ds" / "episodes") for sd in (3, 4)]
    write_manifest(root / "ds", "tiny", metas, extra={})
    pack_dataset(root / "ds", root / "pack", None, 6, statuses=("success", "failure"))
    _write_all(_collect([(11, 0), (12, 0), (13, 2)]), root / "relgen", 5)
    return SimpleNamespace(root=root, pack=root / "pack", relgen=root / "relgen")


# ------------------------------------------------------------------ the shard row -> forwardable batch
def test_collected_rows_carry_the_featurizer_input_and_entity_ids(world):
    rows = load_shard_rows("geo.pos3d", "1", world.relgen)
    assert len(rows) == 9
    for r in rows:
        pi, ids = r["inputs"]["policy_input"], r["inputs"]["tokens"]["ctx"]
        assert len(ids) == sum(len(pi.tokens[b]) for b in BANKS)
        assert {"gripper", "cube", "target"} <= set(ids)                     # manipulator token + the two scene objects
        lab = r["labels"]["pos3d"]
        assert lab.valid.shape == (len(ids),) and lab.valid.sum() == sum(i is not None for i in ids)
        assert r["provenance"]["env"] == "fixture/arm"


def test_collate_rows_remaps_labels_into_the_padded_layout(world):
    rows = load_shard_rows("geo.pos3d", "1", world.relgen)
    short, long_ = rows[0], rows[-1]                                          # seed 11 (2 scene tokens) / seed 13 (4)
    assert len(short["inputs"]["tokens"]["ctx"]) < len(long_["inputs"]["tokens"]["ctx"])
    batch = collate_rows([long_, short], "arm")
    lab = batch.extra["relation_labels"]["ctx"]
    C = batch.ctx_mask.shape[1]
    assert lab["pos3d"].shape == (2, C, 3) and lab["pos3d.valid"].shape == (2, C)
    for i, row in enumerate((long_, short)):
        pi, own = row["inputs"]["policy_input"], row["labels"]["pos3d"]
        pos = np.concatenate([batch.bank_offset[b] + np.arange(len(pi.tokens[b])) for b in BANKS])
        assert np.array_equal(lab["pos3d"][i, pos].numpy(), own.value.astype(np.float32))
        assert np.array_equal(lab["pos3d.valid"][i, pos].numpy(), own.valid)
        assert int(lab["pos3d.valid"][i].sum()) == int(own.valid.sum())        # nothing lands on a padded / other slot
        assert bool(batch.ctx_mask[i, pos].all())
    assert int(batch.ctx_mask[1].sum()) < int(batch.ctx_mask[0].sum())         # the short row really is padded


def test_collate_rows_refuses_a_label_the_family_does_not_attach_to_ctx(world):
    from rrp.policies.relations.base import FactorError
    rows = load_shard_rows("geo.pos3d", "1", world.relgen)[:2]
    for r in rows:
        r["labels"] = {"not_a_label": r["labels"]["pos3d"]}
    with pytest.raises(FactorError, match="not_a_label"):
        collate_rows(rows, "arm")


# ------------------------------------------------------------------ the three arm trainers, 3 CPU steps each
def _spy(monkeypatch, mod, action_loss_attr=None):
    """Record what the trainer's hook drew (composition, shard rows), each shard factor loss, and the size of every batch the
    ACTION loss saw (+ whether it carried relation labels)."""
    rec = SimpleNamespace(rel=None, counts=[], shard_rows=[], raw=[], el=[], el_grad=[], action_rows=[], action_labelled=[])
    real = mod.relation_batches

    def wrapped(*a, **k):
        rel = real(*a, **k)
        rec.rel = rel
        draw, loss = rel.draw, rel.loss

        def d(dev):
            b, sh = draw(dev)
            rec.counts.append(dict(b["counts"]))
            rec.shard_rows.append(0 if sh is None else sh.B)
            return b, sh

        def l(rc, step):
            el, logs = loss(rc, step)
            rec.raw.append(logs["relgen_raw"])
            rec.el.append(float(el.detach()))
            rec.el_grad.append(el.requires_grad)
            return el, logs
        rel.draw, rel.loss = d, l
        return rel
    monkeypatch.setattr(mod, "relation_batches", wrapped)

    def tap(fn, batch_of):
        def f(*a, **k):
            b = batch_of(a, k)
            rec.action_rows.append(b.B)
            rec.action_labelled.append("relation_labels" in b.extra)
            return fn(*a, **k)
        return f
    return rec, tap


def _check(rec, out: Path, seed: int):
    assert len(rec.counts) == STEPS and all(c == {"main": B_SIZE - N_REL, "geo.pos3d": N_REL} for c in rec.counts)
    assert rec.shard_rows == [N_REL] * STEPS
    assert len(rec.el) == STEPS and all(np.isfinite(rec.el)) and min(rec.el) > 0 and all(rec.el_grad)   # factor loss on
    assert not any(rec.action_labelled)                                      # no action loss on a shard row
    assert rec.el[0] <= FACTOR_LOSS_SCALE * 1.0001 and rec.raw[0] > 0         # calibrated: at most the one scale at init
    recs = [json.loads(l) for l in (out / "schedule.jsonl").read_text().splitlines()]
    assert [r["step"] for r in recs] == [0, 2]                               # interval 2 over 3 steps
    obs = [m for _, m in recs[1]["metrics"]]                                 # the field factor's competence reaches the scheduler
    assert obs and all(0.0 <= m["geo.pos3d"]["competence"] <= 1.0 for m in obs) and obs[0]["geo.pos3d"]["competence"] == 0.0
    _, seq = Scheduler.replay_records(rec.rel.cfg, seed, recs, steps=range(STEPS), n=B_SIZE)
    assert [[(sorted(fs), c) for fs, c in s if c > 0] for s in seq] == [[(["geo.pos3d"], N_REL)]] * STEPS


@pytest.fixture(scope="module")
def stage_a(world, tmp_path_factory):
    """Stage A on the tiny pack with shards on; module-scoped because stage B loads its representation."""
    mp = pytest.MonkeyPatch()
    rec, tap = _spy(mp, latent_train, "representation_step")
    mp.setattr(latent_train, "evaluate_representation", lambda *a, **k: {})
    mp.setattr(latent_train, "representation_step", tap(latent_train.representation_step, lambda a, k: a[3][0]))
    out = tmp_path_factory.mktemp("rep")
    cfg = dict(latent=LATENT, packed_dir=str(world.pack), seed=1, steps=STEPS, batch_size=B_SIZE, zero_prev_action=True,
              relgen=str(world.relgen), curriculum=CUR)
    try:
        res = latent_train.train_representation(cfg, out)
    finally:
        mp.undo()
    return SimpleNamespace(rec=rec, out=out, res=res)


def test_stage_a_three_cpu_steps_shard_rows_enter_the_estimate_loss_only(stage_a):
    assert stage_a.res["steps"] == STEPS and (stage_a.out / "representation.pt").exists()
    _check(stage_a.rec, stage_a.out, 1)
    assert stage_a.rec.action_rows == [B_SIZE - N_REL] * STEPS               # the realizer / KL loss saw the pack rows only


def test_flow_three_cpu_steps_shard_rows_enter_the_estimate_loss_only(world, stage_a, monkeypatch, tmp_path):
    rec, tap = _spy(monkeypatch, latent_train)
    monkeypatch.setattr(latent_train, "evaluate_generated", lambda *a, **k: {})
    monkeypatch.setattr(FlowPolicy, "loss", tap(FlowPolicy.loss, lambda a, k: a[1]))
    cfg = dict(representation=str(stage_a.out / "representation.pt"), packed_dir=str(world.pack), seed=2, steps=STEPS,
               batch_size=B_SIZE, zero_prev_action=True, snapshot_every=2, policy=dict(POLICY, name="tinyflow"),
               relgen=str(world.relgen), curriculum=CUR)
    res = latent_train.train_latent_flow(cfg, tmp_path)
    assert res["steps"] == STEPS and (tmp_path / "policy.pt").exists()
    _check(rec, tmp_path, 2)
    assert rec.action_rows == [B_SIZE - N_REL] * STEPS                       # the flow's MSE saw the pack rows only


def test_a_scheduled_factor_without_an_estimate_head_fails_loudly(world, monkeypatch, tmp_path):
    monkeypatch.setattr(latent_train, "evaluate_representation", lambda *a, **k: {})
    from rrp.harness.data.mix import SILENT_STEPS_MAX         # a factor that NEVER writes a term fails after this many steps
    cfg = dict(latent=dict(LATENT, encoder_factors=["preset:arm", "geo.pos3d"]), packed_dir=str(world.pack), seed=1,
               steps=SILENT_STEPS_MAX + 2, batch_size=B_SIZE, zero_prev_action=True, relgen=str(world.relgen), curriculum=CUR)
    with pytest.raises(RelgenError, match="source: probe"):
        latent_train.train_representation(cfg, tmp_path)


def test_bc_three_cpu_steps_shard_rows_enter_the_estimate_loss_only(world, monkeypatch, tmp_path):
    rec, tap = _spy(monkeypatch, behavior)
    monkeypatch.setattr(FlowPolicy, "loss", tap(FlowPolicy.loss, lambda a, k: a[1]))
    cfg = dict(packed_dir=str(world.pack), seed=3, epochs=1, batch_size=B_SIZE, horizon=6, zero_prev_action=True,
               policy=dict(POLICY, horizon=6, name="tinybc"), relgen=str(world.relgen), curriculum=CUR)
    res = behavior.train_policy(cfg, tmp_path)
    assert res["steps"] == STEPS
    _check(rec, tmp_path, 3)
    assert rec.action_rows == [B_SIZE] * STEPS                               # the loader's batch is whole; shard rows are extra


def test_trainers_without_a_curriculum_draw_no_shards(world, monkeypatch, tmp_path):
    """No `params.curriculum` / `inputs.relgen` = the pre-hook trainer: no schedule.jsonl, no shard forward."""
    monkeypatch.setattr(latent_train, "evaluate_representation", lambda *a, **k: {})
    cfg = dict(latent=LATENT, packed_dir=str(world.pack), seed=1, steps=2, batch_size=B_SIZE, zero_prev_action=True)
    latent_train.train_representation(cfg, tmp_path)
    assert not (tmp_path / "schedule.jsonl").exists()
    with pytest.raises(ValueError, match="params.curriculum"):
        latent_train.train_representation(dict(cfg, relgen=str(world.relgen)), tmp_path / "x")


# ------------------------------------------------------------------ the recipes: relgen is an input of the arm trainers
@pytest.mark.parametrize("fset,factors", [("geo", ["geo.depth3d", "geo.normal_align"]), ("ix", ["ix.contact", "ix.held_by", "ix.support"]),
                                          ("task", ["task.next_contact"])])
def test_relations_recipe_feeds_shards_to_f0(fset, factors):
    from rrp.harness.dag import load_dag, plan_dag
    from rrp.harness.pipelines.base import _load_families
    _load_families()
    plan = plan_dag(load_dag(f"recipes/relations/relations_{fset}.yaml"), source="t")
    f0, base = plan.nodes[f"F0@{fset}.s1"], plan.nodes["F0@base.s1"]
    assert f"relgen@{fset}.s1" in f0.deps and "relgen" in f0.rc.inputs
    assert f0.rc.params["curriculum"]["factors"] == factors
    # the F0 factor list builds in the arm family (the presets' handover / same_track / given-normal members do not),
    # and every scheduled factor is a probe-sourced one with a readout: it writes the estimate the loss supervises
    from rrp.policies.relations.base import effective_source, resolve
    specs = {s.name: s for s in resolve(f0.rc.params["policy"]["factors"], default="arm", family="arm")}
    for f in factors:
        assert f in specs and effective_source(specs[f]) == "probe", f
    assert "relgen" not in base.rc.inputs and "curriculum" not in base.rc.params
    assert "caveat" not in load_dag(f"recipes/relations/relations_{fset}.yaml")
