"""D-146 C3 (audit D7 / D1): pointer recipes (multi-seed lineage, UI factors vs none, run-once sealed evaluation on a declared
split) and the ComputerWorld scene parts of relgen. No torch, no simulator: stage argv are recorded from a stubbed runner."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from rrp.core.runconfig import RunIndex
from rrp.core.sealed import SealedSplitError
from rrp.harness import dag
from rrp.harness.data import relgen
from rrp.harness.pipelines import base as B
from rrp.harness.pipelines import pointer as P

ROOT = Path(__file__).resolve().parents[2]
V1 = "research/splits/cworld_pointer_v1.json"
SYN = "research/splits/synthetic_v9.json"
TASKS = ["cw/calc_sum", "cw/drag_window"]


def _plan(name):
    return dag.plan_dag(dag.load_dag(ROOT / "recipes/pointer" / f"{name}.yaml"), source="t")


def _ctx(node, root, **opts):
    rc = node.rc.model_copy(update=dict(options={**node.rc.options, **opts}))
    return B.StageContext(rc=rc, index=RunIndex(), root=root)


@pytest.fixture
def root(tmp_path):
    (tmp_path / "research/splits").mkdir(parents=True)
    shutil.copy(ROOT / V1, tmp_path / V1)
    v1 = json.loads((ROOT / V1).read_text())
    seeds = {t: {"dev": [0, 1], "sealed_id": [10, 11]} for t in TASKS}
    seeds["cw/calc_sum"]["sealed_heldout"] = [20]                   # drag_window has no held-out variants
    (tmp_path / SYN).write_text(json.dumps(dict(v1, split_id="synthetic_v9", seeds=seeds)))
    return tmp_path


@pytest.fixture
def calls(monkeypatch):
    rec = []
    monkeypatch.setattr(B.StageContext, "run", lambda self, argv, **kw: rec.append(list(argv)))
    monkeypatch.setattr(B.StageContext, "run_parallel", lambda self, jobs, w: rec.extend(list(j[0]) for j in jobs))
    return rec


# ------------------------------------------------------------------------------------------------ flags and split passthrough
def test_list_params_become_repeated_values_and_empty_lists_are_absent():
    assert P._flags(dict(steps=5, w_sem=0.5, factors=["preset:ui", {"name": "ui.above", "mix": 0}])) == [
        "--steps", "5", "--w-sem", "0.5", "--factors", "preset:ui", '{"name": "ui.above", "mix": 0}']
    assert P._flags(dict(steps=5, factors=[], lr=None)) == ["--steps", "5"]
    assert P._flags(dict(factors=["preset:ui"], steps=1), skip=("factors",)) == ["--steps", "1"]


def test_every_training_stage_passes_the_split_and_the_factors(root, calls):
    plan = _plan("pointer_ui")
    for nid in ("rep@semfix.s1", "flow@semfix.s1"):
        ctx = _ctx(plan.nodes[nid], root, split=SYN)
        for k, v in ctx.rc.input_paths(ctx.index).items():
            if k == "data":
                (root / v).mkdir(parents=True, exist_ok=True)
                (root / v / "t.npz").write_text("x")
            else:
                (root / v).parent.mkdir(parents=True, exist_ok=True)
                (root / v).write_text("x")
        calls.clear()
        B._REGISTRY[(ctx.rc.family, ctx.rc.stage)].fn(ctx)
        argv = calls[0]
        assert argv[argv.index("--split") + 1] == SYN
        i = argv.index("--factors")
        assert argv[i + 1] == "preset:ui" and not argv[i + 2].startswith("preset")


def test_a_missing_split_is_refused_before_any_job(root, calls):
    plan = _plan("pointer_seeds")
    ctx = _ctx(plan.nodes["collect"], root)                               # v2 is C2's file: not in this tree
    assert not (root / ctx.opts["split"]).exists()
    with pytest.raises(B.StageError, match="declare the split"):
        P._seeds(ctx, "dev", "cw/calc_sum")
    ev = _ctx(_plan("pointer_sealed").nodes["sealed_latent@semfix.s1"], root)
    with pytest.raises(B.StageError, match="declare the split"):
        P.eval_r2(ev)
    assert not calls


# ------------------------------------------------------------------------------------------------ sealed guard
def _sealed(root, seed=1, variant="semfix", node="sealed_latent", **opts):
    return _ctx(_plan("pointer_sealed").nodes[f"{node}@{variant}.s{seed}"], root, split=SYN, tasks=TASKS, **opts)


def _events(root):
    p = root / P.SEALED_LOG
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def test_sealed_cells_run_once_and_only_infrastructure_failures_reopen(root, calls):
    ctx = _sealed(root)
    P.eval_r2(ctx)                                                        # first run: both sets, drag_window has no heldout
    ev = _events(root)
    assert sorted(e["event"] for e in ev) == ["done"] * 3 + ["start"] * 3
    cells = {e["cell"] for e in ev}
    assert len(cells) == 3 and all(c.startswith("synthetic_v9|") for c in cells)
    seeds = [a[a.index("--seeds") + 1] for a in calls]
    assert sorted(seeds) == ["10,11", "10,11", "20"]
    for out in (root / ctx.rc.out).glob("*.jsonl"):
        out.unlink()                                                      # even with the outputs gone
    n = len(calls)
    with pytest.raises(SealedSplitError, match="already ran"):
        P.eval_r2(ctx)
    assert len(calls) == n                                                # nothing was launched


def test_the_splits_env_kwargs_reach_every_eval_job(root, calls):
    f = root / SYN
    f.write_text(json.dumps(dict(json.loads(f.read_text()), env_kw={"strings": "procedural", "level": 2})))
    P.eval_r2(_sealed(root))
    assert calls and all(a.count("--env-kw") == 2 for a in calls)
    assert all("strings=procedural" in a and "level=2" in a for a in calls)


def test_a_crashed_sealed_attempt_stays_open_until_released_with_a_reason(root, monkeypatch, calls):
    ctx = _sealed(root)
    monkeypatch.setattr(B.StageContext, "run_parallel", lambda self, jobs, w: (_ for _ in ()).throw(B.StageError("node lost")))
    with pytest.raises(B.StageError):
        P.eval_r2(ctx)
    ev = _events(root)
    assert {e["event"] for e in ev} == {"start"}
    with pytest.raises(SealedSplitError, match="already ran"):            # a bad result / crash is not a free retry
        P.eval_r2(ctx)
    cid = ev[0]["cell"]
    with pytest.raises(SealedSplitError, match="reason"):
        P.release_sealed_cell(root, cid, "  ")
    with pytest.raises(SealedSplitError, match="no open attempt"):
        P.release_sealed_cell(root, "synthetic_v9|sealed_id|cw/none|x", "typo")
    for e in ev:
        P.release_sealed_cell(root, e["cell"], "peer OOM-killed the job (infra)")
    monkeypatch.setattr(B.StageContext, "run_parallel", lambda self, jobs, w: calls.extend(list(j[0]) for j in jobs))
    P.eval_r2(ctx)                                                        # released: one more attempt
    assert [e["event"] for e in _events(root)].count("done") == 3
    with pytest.raises(SealedSplitError, match="already ran"):
        P.eval_r2(ctx)


def test_the_method_is_its_frozen_checkpoints_so_other_seeds_and_arms_are_other_cells(root, calls):
    P.eval_r2(_sealed(root, seed=1))
    P.eval_r2(_sealed(root, seed=2))
    P.eval_r2(_sealed(root, seed=1, variant="nosem"))
    P.eval_r2(_sealed(root, seed=1, variant="nosem", node="sealed_bc", policy="pointer_bc"))
    assert len({e["cell"] for e in _events(root)}) == 12


def test_the_consumed_v1_sealed_sets_are_refused_but_dev_is_not(root, calls):
    ctx = _sealed(root)
    ctx = B.StageContext(rc=ctx.rc.model_copy(update=dict(options={**ctx.opts, "split": V1})), index=RunIndex(), root=root)
    with pytest.raises(SealedSplitError, match="consumed"):
        P.eval_r2(ctx)
    assert not calls and not _events(root)
    dev = B.StageContext(rc=ctx.rc.model_copy(update=dict(options={**ctx.opts, "seed_sets": ["dev"]})), index=RunIndex(), root=root)
    P.eval_r2(dev)
    assert calls and not _events(root)                                    # dev evaluation is unguarded and unlogged


def test_only_the_declared_seed_sets_exist(root, calls):
    with pytest.raises(B.StageError, match="dev | sealed_id"):
        P.eval_r2(_sealed(root, seed_sets=["train"]))


# ------------------------------------------------------------------------------------------------ recipes
def test_lineage_recipe_has_three_seeds_and_every_arm_gets_them():
    plan = _plan("pointer_seeds")
    for seed in (1, 2, 3):
        for n in ("rep@semfix", "rep@nosem", "flow@semfix", "flow@nosem", "eval_dev@semfix", "eval_dev@nosem", "bc@nosem",
                  "eval_bc@nosem", "flow_eng@nosem", "eval_eng@nosem"):
            assert f"{n}.s{seed}" in plan.nodes, (n, seed)
    assert {n for n in plan.nodes if "@" not in n} == {"collect", "eval_oracle"}
    assert len({n.rc.run_id for n in plan.nodes.values()}) == len(plan.nodes)
    for n in plan.nodes.values():
        assert n.rc.options["split"] == "research/splits/cworld_pointer_v2.json"
        assert set(n.rc.options.get("seed_sets") or [n.rc.options.get("seed_set", "dev")]) == {"dev"}
    for n in plan.nodes.values():                                          # the engineered flow has no probe head: nosem, w_sem 0
        if n.rc.stage == "train_flow" and n.rc.params.get("target") == "eng":
            assert n.rc.params["w_sem"] == 0 and n.rc.variant == "nosem"


def test_ui_arm_differs_from_none_only_in_the_factors_and_shares_the_demos():
    seeds, ui = _plan("pointer_seeds"), _plan("pointer_ui")
    assert ui.nodes["collect"].rc.config_hash() == seeds.nodes["collect"].rc.config_hash()
    for nid in ("rep@semfix.s2", "flow@nosem.s3"):
        a, b = dict(seeds.nodes[nid].rc.params), dict(ui.nodes[nid].rc.params)
        assert b.pop("factors") == ["preset:ui"] and "factors" not in a and a == b
        assert ui.nodes[nid].rc.inputs.get("data") == seeds.nodes[nid].rc.inputs.get("data")
    assert not any(n.startswith(("bc", "eval_bc", "flow_eng", "eval_eng", "eval_oracle")) for n in ui.nodes)


def test_copy_recipe_arms_differ_only_in_eng_version_and_key_head():
    plan = _plan("pointer_copy")
    assert {n.rc.seed for n in plan.nodes.values() if n.rc.stage == "train_flow"} == {1, 2, 3}
    for s in (1, 2, 3):
        free, copy, bc = (plan.nodes[f"{k}@nosem.s{s}"].rc.params for k in ("flow_eng", "flow_copy", "bc"))
        assert free.pop("eng_version") == copy.pop("eng_version") == "cw_pointer_eng.v2"
        assert "key_head" not in free and copy.pop("key_head") == "copy" and bc["key_head"] == "copy"
        assert free == copy                                            # steps, batch, target, w_sem: identical
    data = {n.rc.inputs.get("data") for n in plan.nodes.values() if "data" in n.rc.inputs}
    assert len(data) == 1                                              # one collect run for every arm


def test_sealed_recipes_evaluate_frozen_checkpoints_of_every_seed_on_both_sealed_sets():
    for name, src, variants in (("pointer_sealed", "pointer-s3", 2), ("pointer_ui_sealed", "pointer-ui", 2)):
        plan = _plan(name)
        sealed = [n for n in plan.nodes.values() if n.rc.options.get("seed_sets")]
        assert sealed and all(n.rc.options["seed_sets"] == ["sealed_id", "sealed_heldout"] for n in sealed)
        assert all(n.rc.options["split"] == "research/splits/cworld_pointer_v2.json" for n in plan.nodes.values())
        lat = [n for n in plan.nodes.values() if n.name == "sealed_latent"]
        assert len(lat) == 3 * variants
        assert {n.rc.seed for n in lat} == {1, 2, 3}
        assert all(f"{src}-{n.rc.variant}/train_flow_s{n.rc.seed}/" in n.rc.input_paths(RunIndex())["flow"] for n in lat)
        assert not any(d.startswith(("rep", "flow@", "collect")) for n in plan.nodes.values() for d in n.deps)   # trains nothing


# ------------------------------------------------------------------------------------------------ ComputerWorld scene parts
relgen.load_families()


def test_cw_parts_are_registered_for_the_computerworld_env_only():
    for name, act in (("cw_viewport", "viewport"), ("cw_depth", "zstack")):
        part = relgen.PARTS[name]
        assert part.envs == ("computerworld",) and part.activates == frozenset({act})
    with pytest.raises(relgen.ComposeError):
        relgen.compose({"viewport"}, "mujoco/arm", np.random.default_rng(0))


def test_compose_builds_make_env_scene_kwargs_deterministically():
    import inspect

    from rrp.envs.computerworld import ComputerWorldEnv
    params = set(inspect.signature(ComputerWorldEnv.__init__).parameters)
    a = relgen.compose({"viewport", "zstack"}, "computerworld", np.random.default_rng(3))
    b = relgen.compose({"viewport", "zstack"}, "computerworld", np.random.default_rng(3))
    assert a.kwargs == b.kwargs and a.parts == ("cw_depth", "cw_viewport") and a.active == {"viewport", "zstack"}
    assert set(a.kwargs) <= params and a.kwargs["depth"] in ("stack", "constant") and not a.entities
    assert sorted(p["part"] for p in a.provenance["parts"]) == ["cw_depth", "cw_viewport"]
    seen = {tuple(relgen.compose({"viewport"}, "computerworld", np.random.default_rng(s)).kwargs.values()) for s in range(12)}
    assert len(seen) > 1                                                   # the rng actually varies the viewport


def test_cw_parts_vary_only_the_named_factor():
    d = relgen.compose({"viewport", "zstack"}, "computerworld", np.random.default_rng(1))
    for part, factor, keys in (("cw_viewport", "viewport", {"width", "height", "m_per_px"}), ("cw_depth", "zstack", {"depth"})):
        out = relgen.PARTS[part].vary(d, np.random.default_rng(0), factor)
        assert out and all(o.kwargs != d.kwargs for o in out)
        for o in out:
            assert {k for k in d.kwargs if o.kwargs[k] != d.kwargs[k]} <= keys
            assert o.provenance["vary"] == {"part": part, "factor": factor}
        assert d.kwargs == relgen.compose({"viewport", "zstack"}, "computerworld", np.random.default_rng(1)).kwargs  # not mutated
    with pytest.raises(ValueError, match="unsupported factor"):
        relgen.PARTS["cw_depth"].vary(d, np.random.default_rng(0), "orient")
