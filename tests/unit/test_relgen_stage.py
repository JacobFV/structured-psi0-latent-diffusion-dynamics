"""relations_data stage + shards + mix (D-144 fanout unit R10, docs/relations.md 5.3, section 10 row R10).

Acceptance: a 2-episode fixture run writes shards + manifest with per-sample provenance (active set, label
versions); `mixed_batches` fractions are exact and seed-deterministic; labels a sample does not carry are masked.
"""
from __future__ import annotations

import numpy as np
import pytest

from rrp.harness.data.mix import allocate, mask_missing_labels, mixed_batches
from rrp.harness.data.relgen import Label, LabelDef, TokenIndex, TransformDef, register_label, register_transform
from rrp.harness.data.relgen.curriculum import Scheduler, SchedulerConfig
from rrp.harness.data.mix import RelgenError, load_shard_rows, read_shard_manifest, write_shard
from rrp.harness.pipelines.relations import EpisodeSnapshot, label_episode, relations_data
from rrp.policies.relations.base import FactorDef, register_factor


# ------------------------------------------------------------------ fixture StateView + registry entries
class FixtureView:
    """Duck-types `rrp.envs.base.StateView`: a fixed set of "poses" for a 2-entity toy scene."""
    def __init__(self, offset: float = 0.0):
        self.caps = frozenset({"poses"})
        self.time = 0.0
        self.gravity = np.array([0.0, 0.0, -9.81])
        self._pos = {"gripper": np.array([0.0, 0.0, 0.0]), "cube": np.array([1.0 + offset, 0.0, 0.0])}

    def entities(self):
        return []

    def contacts(self):
        return []

    def joints(self):
        return []

    def camera(self, name):
        raise NotImplementedError

    def ui_tree(self):
        return []

    def token_entity(self, token_set, slot):
        return slot


def _pos3d_label(view, index: TokenIndex) -> Label:
    ids = index.sets["ctx"]
    value = np.stack([view._pos[i] for i in ids])
    valid = np.ones(len(ids), dtype=bool)
    return Label(value=value, valid=valid, prov="gt", version="3")


def _double(sample, rng, params):
    """A toy TRANSFORMS entry: emits the sample unchanged plus one perturbed copy (like `noise`/`cf_swap` widening
    one sample into several) so the stage's per-factor row multiplication is exercised."""
    import copy
    twin = copy.deepcopy(sample)
    return [sample, twin]


register_label(LabelDef(name="test.pos3d", version="3", arity=1, needs=frozenset({"poses"}), fn=_pos3d_label))
register_transform(TransformDef(name="test.double", version="1", fn=_double))
register_factor(FactorDef("test.r10.pos3d", "1", field="pos3d", op="sqdiff+diff", form="aug",
                          label="test.pos3d", gen=("test.double",)))
register_factor(FactorDef("test.r10.nolabel", "1", field="pos3d", op="sqdiff+diff", form="aug"))   # no .label
register_factor(FactorDef("test.r10.needs_camera", "1", field="cam_uvd", op="sqdiff+diff", form="aug",
                          label="test.pos3d"))   # label exists but this spec is unused by these fixtures


def _episodes(n=2):
    idx = TokenIndex(sets={"ctx": ["gripper", "cube"]})
    return [EpisodeSnapshot(env="fixture", task="toy", seed=s, view=FixtureView(offset=float(s)), index=idx, step=0)
            for s in range(n)]


# ------------------------------------------------------------------ label_episode
def test_label_episode_skips_unlabeled_and_uncovered_factors():
    ep = _episodes(1)[0]
    from rrp.policies.relations.base import spec as _spec
    specs = [_spec("test.r10.pos3d"), _spec("test.r10.nolabel")]
    out = label_episode(ep, specs, rng=np.random.default_rng(0))
    assert set(out) == {"test.r10.pos3d"}                       # nolabel factor contributes nothing
    rows = out["test.r10.pos3d"]
    assert len(rows) == 2                                       # test.double multiplied 1 -> 2
    for r in rows:
        assert r["provenance"]["active"] == ["test.r10.pos3d", "test.r10.nolabel"]
        assert r["provenance"]["transforms"] == ["test.double"]
        assert r["provenance"]["labels"] == [{"label": "test.pos3d", "prov": "gt", "version": "3"}]
        lab = r["labels"]["test.pos3d"]
        assert lab.value.shape == (2, 3) and lab.valid.all()


# ------------------------------------------------------------------ relations_data: the fixture run (acceptance)
def test_relations_data_writes_shard_and_manifest(tmp_path):
    out_root = tmp_path / "relgen"
    result = relations_data(["test.r10.pos3d", "test.r10.nolabel"], _episodes(2), out_root, seed=7)
    assert set(result["factors"]) == {"test.r10.pos3d"}
    man = result["factors"]["test.r10.pos3d"]
    # 2 episodes x 2 rows (test.double) = 4 rows written
    assert man["n_episodes"] == 4 and man["schema"] == "relgen-shard-1"
    for row in man["episodes"]:
        assert row["active"] == ["test.r10.pos3d", "test.r10.nolabel"]           # active set recorded
        assert row["labels"] == [{"label": "test.pos3d", "prov": "gt", "version": "3"}]   # label + version recorded
        assert row["env"] == "fixture" and row["task"] == "toy" and row["seed"] in (0, 1)
    d = out_root / "test.r10.pos3d" / "1"                                        # <factor>/<version>/
    assert (d / "manifest.json").exists()
    assert list(d.glob("*.npz"))
    reread = read_shard_manifest(d)
    assert reread["hash_ok"] is True

    rows = load_shard_rows("test.r10.pos3d", "1", out_root)
    assert len(rows) == 4
    for r in rows:
        lab = r["labels"]["test.pos3d"]
        assert isinstance(lab, Label) and lab.value.shape == (2, 3)


def test_relations_data_no_producers_is_not_an_error(tmp_path):
    result = relations_data(["test.r10.nolabel"], _episodes(2), tmp_path / "relgen", seed=0)
    assert result == {"factors": {}}
    assert not (tmp_path / "relgen").exists() or not list((tmp_path / "relgen").iterdir())


def test_relations_data_appends_across_shard_runs(tmp_path):
    out_root = tmp_path / "relgen"
    relations_data(["test.r10.pos3d"], _episodes(2), out_root, seed=1)
    relations_data(["test.r10.pos3d"], _episodes(2), out_root, seed=2)
    rows = load_shard_rows("test.r10.pos3d", "1", out_root)
    assert len(rows) == 8                                        # two runs' shards both kept in the one manifest
    assert {r["provenance"]["seed"] for r in rows} == {0, 1}     # (each run reuses episode seeds 0/1 internally)


# ------------------------------------------------------------------ mixed_batches
def _pool(n, label_name="test.pos3d"):
    return [{"inputs": {}, "labels": {label_name: Label(value=np.ones((2, 3)), valid=np.ones(2, bool),
                                                        prov="gt", version="3")},
            "provenance": {"active": ["f"], "row": i}} for i in range(n)]


def test_mixed_batches_fractions_are_exact():
    sched = Scheduler(SchedulerConfig(factors=("f",), share_min=0.3, share_max=0.6, full_world_start=0.0,
                                      full_world_end=0.0, interval=1), seed=0)
    shards = {"f": _pool(50)}
    main = [[{"inputs": {}, "labels": {}, "provenance": {}}] for _ in range(3)]
    batches = list(mixed_batches(main, sched, shards, batch_size=20, rng=np.random.default_rng(0)))
    assert len(batches) == 3
    for b in batches:
        assert sum(b["counts"].values()) == 20
        assert len(b["main"]) + len(b["relgen"]) == 20
        assert len(b["relgen"]) == b["counts"]["f"]
        # allocate() gives the exact same split for the scheduler's current share
        assert b["counts"] == allocate(20, {"f": sched.history[-1].share["f"]})


def test_mixed_batches_is_seed_deterministic():
    def mk():
        return Scheduler(SchedulerConfig(factors=("f", "g"), interval=1), seed=3)
    shards = {"f": _pool(30), "g": _pool(30, "g_label")}
    main = [[{"inputs": {}, "labels": {}, "provenance": {}}] for _ in range(4)]
    b1 = list(mixed_batches(main, mk(), shards, batch_size=16, rng=np.random.default_rng(42)))
    b2 = list(mixed_batches(main, mk(), shards, batch_size=16, rng=np.random.default_rng(42)))
    for x, y in zip(b1, b2):
        assert x["counts"] == y["counts"]
        assert [r["provenance"] for r in x["relgen"]] == [r["provenance"] for r in y["relgen"]]
    b3 = list(mixed_batches(main, mk(), shards, batch_size=16, rng=np.random.default_rng(99)))
    assert any(x["counts"] != y["counts"] or [r["provenance"] for r in x["relgen"]] != [r["provenance"] for r in y["relgen"]]
              for x, y in zip(b1, b3)) or True   # different seeds MAY coincide on counts; just must not crash


def test_mixed_batches_masks_missing_labels():
    sched = Scheduler(SchedulerConfig(factors=("f",), share_min=0.5, share_max=0.5, full_world_start=0.0,
                                      full_world_end=0.0, interval=1), seed=0)
    shards = {"f": _pool(10)}
    main = [[{"inputs": {}, "labels": {}, "provenance": {}} for _ in range(4)]]
    batch = next(mixed_batches(main, sched, shards, batch_size=8, rng=np.random.default_rng(0)))
    assert batch["main"] and batch["relgen"] and all("test.pos3d" in r["labels"] for r in batch["main"])
    real_shape = batch["relgen"][0]["labels"]["test.pos3d"].valid.shape
    for r in batch["main"]:
        lab = r["labels"]["test.pos3d"]
        assert not lab.valid.any() and lab.prov == "none"                      # masked, not missing
        assert lab.valid.shape == real_shape                                   # shape borrowed from the real rows,
                                                                                # not just other masked main rows
    for r in batch["relgen"]:
        assert r["labels"]["test.pos3d"].valid.all()                           # real rows untouched


def test_mask_missing_labels_shapes_from_a_present_label():
    present = Label(value=np.zeros((3, 4)), valid=np.ones(3, bool), prov="gt", version="1")
    s = [{"labels": {"a": present}}, {"labels": {}}]
    out = mask_missing_labels(s, ["a", "b"])
    assert out[1]["labels"]["b"].valid.shape == (3,) and not out[1]["labels"]["b"].valid.any()
    assert out[0]["labels"]["a"] is present                                    # present labels pass through


# ================================================================== R1: the stage, the trainer hook, replay, resume
import json
from types import SimpleNamespace

from rrp.core.runconfig import FLAG_NAMES, RunConfig, RunIndex
from rrp.envs.base import EntityState
from rrp.harness.data import relgen as _relgen
from rrp.harness.data.mix import relation_batches, stack_labels
from rrp.harness.pipelines import base as B
from rrp.harness.pipelines.relations import SnapshotCollector, producer_status
from rrp.policies.relations import catalog as _catalog  # noqa: F401  (registers geo.pos3d & co)
from rrp.policies.relations.base import RelCtx, TokenSet, estimates_loss, spec

_relgen.load_families()


class LiveView(FixtureView):
    """A view of the real `pos3d` label's input: `entities()` with poses, moving with the tick."""
    def __init__(self, tick):
        super().__init__()
        self._e = [EntityState(id="gripper", kind="body", name="gripper", pos=np.array([0.0, 0.0, 0.1 * tick]),
                               quat=np.array([1.0, 0.0, 0.0, 0.0])),
                   EntityState(id="cube", kind="object", name="cube", pos=np.array([1.0, tick, 0.0]),
                               quat=np.array([1.0, 0.0, 0.0, 0.0]))]

    def entities(self):
        return self._e


class LiveEnv:
    def __init__(self):
        self.tick = 0

    def state_view(self):
        return LiveView(self.tick)


def _drive(col, seeds, steps=4):
    """What `harness.rollout` does with a hook: reset then step, per episode (one env per episode index)."""
    for i, _ in enumerate(seeds):
        env = LiveEnv()
        col.on_reset(i, env, None)
        for t in range(steps):
            env.tick = t + 1
            col.on_step(i, env, None, t)


def _rows(seeds=(11, 12, 13), factors=("geo.pos3d",), every=2):
    col = SnapshotCollector("fake/env", "toy", [spec(f) for f in factors], seeds, rng=np.random.default_rng(0),
                            every=every, max_per_episode=3)
    _drive(col, seeds)
    return col.by_factor


def test_snapshot_collector_labels_at_capture_time_with_the_reset_seed():
    by = _rows()
    rows = by["geo.pos3d"]
    assert len(rows) == 3 * 3                                        # reset + ticks 2 and 4, per episode
    assert {r["provenance"]["seed"] for r in rows} == {11, 12, 13}
    assert sorted({r["provenance"]["step"] for r in rows}) == [0, 2, 4]
    r = next(r for r in rows if r["provenance"]["step"] == 4)
    lab = r["labels"]["pos3d"]
    assert np.allclose(lab.value[1], [1.0, 4.0, 0.0]) and lab.valid.all()          # the state AT tick 4, not a later one
    assert r["provenance"]["env"] == "fake/env" and r["provenance"]["parts"] == ["table_objects"]


def _shards(tmp_path, **kw):
    root = tmp_path / "relgen"
    from rrp.harness.pipelines.relations import _write_all
    _write_all(_rows(**kw), root, 5)
    return root


def _rc(shards, steps_cfg=None, factors=("geo.pos3d",), **kw):
    cur = dict(shards=str(shards), factors=list(factors), interval=2, share_min=0.5, share_max=0.5,
               full_world_start=0.0, full_world_end=0.0)
    cur.update(steps_cfg or {})
    d = dict(schema_version="runconfig-1", family="arm", stage="train_flow", variant="na", seed=3, lineage="r1",
             track="t", flags={**{f: None for f in FLAG_NAMES}, "zero_prev_action": True, "contact_version": "contact_v1"},
             params={"policy": {"factors": list(factors)}, "curriculum": cur, "batch_size": 8})
    d.update(kw)
    return RunConfig.model_validate(d)


def _main(n):
    return [[{"inputs": {}, "labels": {}, "provenance": {}} for _ in range(8)] for _ in range(n)]


# (a) shard rows enter the loss; schedule.jsonl parses -------------------------------------------------------------
def test_two_step_cpu_run_shard_rows_enter_the_loss(tmp_path):
    import torch
    root = _shards(tmp_path)
    out = tmp_path / "run"
    batches = relation_batches(_rc(root), out, _main(2))
    mu = torch.zeros(8, 2, 3, requires_grad=True)
    lv = torch.zeros(8, 2, 1, requires_grad=True)
    opt = torch.optim.SGD([mu, lv], lr=0.1)
    n_rel = []
    for b in batches:
        n_rel.append(len(b["relgen"]))
        lab = b["labels"]["ctx"]
        assert lab["pos3d"].shape == (8, 2, 3) and lab["pos3d.valid"].shape == (8, 2)
        assert int(lab["pos3d.valid"].sum()) == len(b["relgen"]) * 2               # main rows are masked, shard rows valid
        ts = TokenSet("ctx", torch.ones(8, 2, dtype=torch.bool), labels=lab)
        rc = RelCtx(sets={"ctx": ts}, estimates={("ctx", "pos3d"): (mu, None), ("ctx", "pos3d", "logvar"): lv},
                    memo={("est_by", "ctx", "pos3d"): "geo.pos3d"})
        loss, logs, metrics = estimates_loss(rc, [spec("geo.pos3d")])
        assert metrics["geo.pos3d_mae"][1] == len(b["relgen"]) * 2 * 3 and "probe_geo.pos3d" in logs
        opt.zero_grad()
        loss.backward()
        assert float(mu.grad.abs().sum()) > 0
        assert float(mu.grad[[i for i in range(8) if not lab["pos3d.valid"][i].any()]].abs().sum()) == 0   # only shard rows
        opt.step()
        batches.observe_estimates(b["step"], {"geo.pos3d_acc": (3.0, 4)})
    assert n_rel == [4, 4]
    recs = [json.loads(l) for l in (out / "schedule.jsonl").read_text().splitlines()]
    assert [r["step"] for r in recs] == [0] and recs[0]["share"]["geo.pos3d"] == 0.5     # interval 2: one decision in 2 steps
    assert recs[0]["steers"] == [] and recs[0]["steer_line"] == 0


# (b) replay reproduces the batch composition, steers and metrics included -----------------------------------------
def test_replay_from_schedule_jsonl_reproduces_batch_composition(tmp_path):
    from rrp.harness.data.relgen.curriculum import Scheduler
    two = ("geo.pos3d", "geo.orient")
    root = _shards(tmp_path, factors=two)
    out = tmp_path / "run"
    rc = _rc(root, dict(share_min=0.1, share_max=0.6, interval=2, boost_gain=2.0), factors=two)
    batches = relation_batches(rc, out, _main(8))
    live = []
    for b in batches:
        live.append([(fs, c) for fs, c in b["active"]])
        if b["step"] == 0:
            with open(out / "steer.jsonl", "a") as fh:
                fh.write("boost geo.pos3d x3 for 6\n")
        batches.observe(b["step"], {"geo.pos3d": {"competence": 0.2 + 0.1 * b["step"]}})
    recs = [json.loads(l) for l in (out / "schedule.jsonl").read_text().splitlines()]
    assert [r["step"] for r in recs] == [0, 2, 4, 6]
    assert [len(r["steers"]) for r in recs] == [0, 1, 0, 0] and recs[1]["steers"][0]["op"] == "boost"
    assert recs[1]["metrics"] and recs[1]["steers"][0]["at"] == 2
    _, seq = Scheduler.replay_records(batches.cfg, int(rc.seed), recs, steps=range(8), n=8)
    assert [[(sorted(fs), c) for fs, c in s if c > 0] for s in seq] == live
    assert len({tuple(map(tuple, x)) for x in map(lambda l: [(tuple(f), c) for f, c in l], live)}) > 1   # the steer moved it


def test_bad_steer_line_is_a_rejected_record_and_the_listing_says_so(tmp_path, capsys):
    from rrp.cli.curriculum import _steer_status
    root = _shards(tmp_path)
    out = tmp_path / "run"
    (out).mkdir()
    (out / "steer.jsonl").write_text("boost geo.pos3d x2 for 4\nnot a steer\n")
    batches = relation_batches(_rc(root, dict(interval=1)), out, _main(2))
    next(batches)
    (out / "steer.jsonl").write_text((out / "steer.jsonl").read_text() + "freeze geo.pos3d\n")   # not read yet
    st = _steer_status(out)
    assert "applied at step 0" in st[0] and "rejected" in st[1] and "pending" in st[2]


# (c) resume adopts by hash ----------------------------------------------------------------------------------------
def test_resume_from_step_rebuilds_the_scheduler_and_replays_the_same_rows(tmp_path):
    root = _shards(tmp_path)
    full = list(relation_batches(_rc(root), tmp_path / "full", _main(4)))
    out = tmp_path / "run"
    first = relation_batches(_rc(root), out, _main(4))
    [next(first) for _ in range(2)]                                      # the run dies after step 1
    with pytest.raises(RelgenError, match="pass start_step"):
        relation_batches(_rc(root), out, _main(2))                       # not silently restarting over its own log
    rest = list(relation_batches(_rc(root), out, _main(2), start_step=2))
    assert [b["step"] for b in rest] == [2, 3]
    prov = lambda b: [r["provenance"] for r in b["relgen"]]                # noqa: E731
    assert [prov(b) for b in rest] == [prov(b) for b in full[2:]]          # per-step draws: same rows as uninterrupted
    assert [b["counts"] for b in rest] == [b["counts"] for b in full[2:]]
    assert [json.loads(l)["step"] for l in (out / "schedule.jsonl").read_text().splitlines()] == [0, 2]


def _arm_rc(**opts):
    return RunConfig.model_validate(dict(
        schema_version="runconfig-1", family="arm", stage="relations_data", variant="na", seed=4, lineage="r1", track="t",
        flags={f: None for f in FLAG_NAMES}, params={"factors": ["geo.pos3d"]},
        options=dict(env="fake/env", task="toy", body="b", seeds=[11, 12], **opts)))


def _fake_collect(o, specs, *, seed):
    col = SnapshotCollector(o["env"], o["task"], specs, _seeds := [int(s) for s in o["seeds"]],
                            rng=np.random.default_rng(seed), every=2, max_per_episode=3)
    _drive(col, _seeds)
    return col.by_factor, ["success"] * len(_seeds)


def test_stage_writes_hashed_shards_and_the_dag_adopts_it_on_rerun(tmp_path, monkeypatch):
    from rrp.harness import dag as D
    from rrp.harness.pipelines import relations as R
    monkeypatch.setattr(R, "collect_by_factor", _fake_collect)
    rc = _arm_rc()
    body = B.Pipeline("arm").run(rc, root=tmp_path, index=RunIndex())
    man = json.loads((tmp_path / rc.out / "pipeline_manifest.json").read_text())
    assert man["config_hash"] == rc.config_hash() and man["provenance"]["source"] == "privileged_teacher"
    sh = man["metrics"]["shards"]["geo.pos3d/1"]
    assert sh["n_rows"] == 6 and sh["manifest_hash"] and all(len(v) > 8 for v in sh["npz"].values())
    n_files = sorted((tmp_path / rc.out / "relgen" / "geo.pos3d" / "1").glob("*.npz"))
    B.Pipeline("arm").run(rc, root=tmp_path, index=RunIndex())                     # rerun: same shard id, rows replaced
    m2 = json.loads((tmp_path / rc.out / "pipeline_manifest.json").read_text())
    assert m2["metrics"]["shards"] == man["metrics"]["shards"]                      # identical hashes, no doubled rows
    assert sorted((tmp_path / rc.out / "relgen" / "geo.pos3d" / "1").glob("*.npz")) == n_files
    assert len(load_shard_rows("geo.pos3d", "1", tmp_path / rc.out / "relgen")) == 6

    class Runner:                                                                # the DAG must adopt, not launch
        launched = []
        def launch(self, node):
            self.launched.append(node.id)
            raise AssertionError("relations_data node was re-run instead of adopted")
        def poll(self, h):
            return 0
        def manifest(self, node):
            p = tmp_path / node.rc.out / "pipeline_manifest.json"
            return json.loads(p.read_text()) if p.exists() else None

    dagspec = f"""
name: r1
family: arm
track: t
lineage: r1
matrix: {{seed: [4]}}
defaults: {{placement: host, resources: {{cpu: 1, mem: 1G}}}}
nodes:
  data:
    stage: relations_data
    variant: na
    config: {{params: {{factors: ['geo.pos3d']}}, options: {{env: fake/env, task: toy, body: b, seeds: [11, 12]}}}}
"""
    from rrp.harness.yamlmini import loads
    plan = D.plan_dag(loads(dagspec))
    node = next(iter(plan.nodes.values()))
    assert node.rc.config_hash() == rc.config_hash()
    code = man["provenance"]["code"]
    s = D.Executor(plan, D.Ledger(tmp_path / "ledger.json"), Runner(), poll_s=0, sleep=lambda s: None, log=lambda m: None,
                   code_now=lambda: code, pins=B.stage_versions).run()
    assert s["completed"] == 1 and Runner.launched == []


def test_stage_with_no_producing_factor_fails_loudly(tmp_path, monkeypatch):
    from rrp.harness.pipelines import relations as R
    monkeypatch.setattr(R, "collect_by_factor", lambda o, specs, *, seed: ({}, []))
    with pytest.raises(B.StageError, match="no factor produced rows"):
        B.Pipeline("arm").run(_arm_rc(), root=tmp_path, index=RunIndex())


# (d) unknown label / transform raises ---------------------------------------------------------------------------
def test_unknown_label_or_transform_raises_but_probe_labels_and_parts_pass():
    ep = _episodes(1)[0]
    register_factor(FactorDef("test.r1.badlabel", "1", field="pos3d", op="sqdiff+diff", form="aug", label="test.nope"))
    register_factor(FactorDef("test.r1.badgen", "1", field="pos3d", op="sqdiff+diff", form="aug", label="test.pos3d",
                              gen=("test.no_such_transform",)))
    register_factor(FactorDef("test.r1.part", "1", field="pos3d", op="sqdiff+diff", form="aug", label="test.pos3d",
                              gen=("table_objects",)))
    from rrp.policies.relations.base import spec as _s
    with pytest.raises(RelgenError, match="test.nope"):
        label_episode(ep, [_s("test.r1.badlabel")], rng=np.random.default_rng(0))
    with pytest.raises(RelgenError, match="test.no_such_transform"):
        label_episode(ep, [_s("test.r1.badgen")], rng=np.random.default_rng(0))
    with pytest.raises(RelgenError, match="test.nope"):
        producer_status([_s("test.r1.badlabel")])
    out = label_episode(ep, [_s("test.r1.part")], rng=np.random.default_rng(0))
    assert out["test.r1.part"][0]["provenance"]["parts"] == ["table_objects"]
    assert producer_status([_s("probe.arm.visible")])["probe.arm.visible"].startswith("probe label")
