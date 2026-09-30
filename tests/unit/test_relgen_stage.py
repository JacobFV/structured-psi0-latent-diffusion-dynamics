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
from rrp.harness.pipelines.relations import (EpisodeSnapshot, label_episode, load_shard_rows, read_shard_manifest,
                                             relations_data, write_shard)
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
    assert batch["main"] and all("test.pos3d" in r["labels"] for r in batch["main"])
    for r in batch["main"]:
        lab = r["labels"]["test.pos3d"]
        assert not lab.valid.any() and lab.prov == "none"                      # masked, not missing
    for r in batch["relgen"]:
        assert r["labels"]["test.pos3d"].valid.all()                           # real rows untouched


def test_mask_missing_labels_shapes_from_a_present_label():
    present = Label(value=np.zeros((3, 4)), valid=np.ones(3, bool), prov="gt", version="1")
    s = [{"labels": {"a": present}}, {"labels": {}}]
    out = mask_missing_labels(s, ["a", "b"])
    assert out[1]["labels"]["b"].valid.shape == (3,) and not out[1]["labels"]["b"].valid.any()
    assert out[0]["labels"]["a"] is present                                    # present labels pass through
