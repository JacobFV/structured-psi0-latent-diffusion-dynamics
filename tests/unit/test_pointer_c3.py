"""D-146 C3 + readiness R2 unit PC (audit D7 / D1): pointer recipes (multi-seed lineage, UI factors vs none, run-once sealed
evaluation on the pinned split) and the ComputerWorld scene parts of relgen. The sealed guard is `core.sealed.SealedSplit`
(pinned `cworld_pointer_v2`, run-once log `artifacts/runs/pointer/sealed_log.jsonl`). No torch, no simulator: stage argv are
recorded from a stubbed runner."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from rrp.core.runconfig import RunIndex
from rrp.core.sealed import SPLITS, SealedSplit, SealedSplitError
from rrp.harness import dag
from rrp.harness.data import relgen
from rrp.harness.pipelines import base as B
from rrp.harness.pipelines import pointer as P

ROOT = Path(__file__).resolve().parents[2]
V1 = "research/splits/cworld_pointer_v1.json"
V2 = "research/splits/cworld_pointer_v2.json"
LOG = SPLITS["cworld_pointer_v2"]["log"]
TASKS = ["cw/calc_sum", "cw/open_type", "cw/drag_window", "cw/fill_form"]
N_CELLS = 7                                                             # 4 tasks on sealed_id + 3 with held-out variants


def _plan(name):
    return dag.plan_dag(dag.load_dag(ROOT / "recipes/pointer" / f"{name}.yaml"), source="t")


def _ctx(node, root, **opts):
    rc = node.rc.model_copy(update=dict(options={**node.rc.options, **opts}))
    return B.StageContext(rc=rc, index=RunIndex(), root=root)


@pytest.fixture
def root(tmp_path):
    if not (ROOT / V2).exists() or not (ROOT / V1).exists():
        pytest.skip(f"{V1} / {V2} missing in this checkout")
    (tmp_path / "research/splits").mkdir(parents=True)
    for f in (V1, V2):
        shutil.copy(ROOT / f, tmp_path / f)
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
        ctx = _ctx(plan.nodes[nid], root, split=V2)
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
        assert argv[argv.index("--split") + 1] == V2
        i = argv.index("--factors")
        assert argv[i + 1] == "preset:ui" and not argv[i + 2].startswith("preset")


def test_a_missing_split_is_refused_before_any_job(root, calls):
    plan = _plan("pointer_seeds")
    ctx = _ctx(plan.nodes["collect"], root, split="research/splits/not_declared.json")
    with pytest.raises(B.StageError, match="declare the split"):
        P._seeds(ctx, "dev", "cw/calc_sum")
    ev = _ctx(_plan("pointer_sealed").nodes["sealed_latent@semfix.s1"], root, split="research/splits/not_declared.json")
    with pytest.raises(B.StageError, match="declare the split"):
        P.eval_r2(ev)
    assert not calls


def test_the_default_split_of_the_pipeline_and_the_trainers_is_v2():
    from rrp.harness.train.pointer import SPLIT_PATH_V2
    from rrp.harness.train.pointer.split import load_split
    import inspect
    assert SPLIT_PATH_V2.endswith("cworld_pointer_v2.json")
    assert inspect.signature(load_split).parameters["path"].default == SPLIT_PATH_V2
    assert P._split_path(B.StageContext(rc=_plan("pointer_seeds").nodes["collect"].rc.model_copy(update=dict(options={})),
                                        index=RunIndex(), root=ROOT)) == SPLIT_PATH_V2


def test_the_pipeline_module_imports_without_torch():
    import subprocess
    import sys
    code = "import sys, rrp.harness.pipelines.pointer, rrp.harness.train.pointer; sys.exit(int('torch' in sys.modules))"
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                       env={**__import__("os").environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"})
    assert r.returncode == 0, r.stderr[-400:]


def _stub_inputs(ctx, root):
    for k, v in ctx.rc.input_paths(ctx.index).items():
        if k == "data":
            (root / v).mkdir(parents=True, exist_ok=True)
            (root / v / "t.npz").write_text("x")
        else:
            (root / v).parent.mkdir(parents=True, exist_ok=True)
            (root / v).write_text("x")


def test_eng_and_bc_default_to_the_v2_code_with_the_copy_head_and_the_p_seeds_recipe_states_its_arm(root, calls):
    plan = _plan("pointer_seeds")
    eng, bc = plan.nodes["flow_eng@nosem.s1"], plan.nodes["bc@nosem.s1"]
    assert eng.rc.params["eng_version"] == "cw_pointer_eng.v1" and eng.rc.params["key_head"] == "free"   # stated, not defaulted
    assert bc.rc.params["key_head"] == "free"
    for node in (eng, bc):                                                # a node that says nothing gets the v2 defaults
        ctx = _ctx(node, root)
        ctx = B.StageContext(rc=ctx.rc.model_copy(update=dict(params={k: v for k, v in ctx.rc.params.items()
                                                                     if k not in ("eng_version", "key_head")})),
                             index=RunIndex(), root=root)
        _stub_inputs(ctx, root)
        calls.clear()
        B._REGISTRY[(ctx.rc.family, ctx.rc.stage)].fn(ctx)
        argv = calls[0]
        assert argv[argv.index("--key-head") + 1] == "copy"
        assert ("--eng-version" in argv) == (node is eng) and (node is not eng or argv[argv.index("--eng-version") + 1] == "cw_pointer_eng.v2")
    sm = _plan("pointer_smoke")
    assert {n.rc.options["split"] for n in sm.nodes.values()} == {V2}
    assert sm.nodes["collect"].rc.params["episodes"] == 4 and sm.nodes["flow@semfix.s1"].rc.params["steps"] == 20


# ------------------------------------------------------------------------------------------------ sealed guard (core.sealed)
def _sealed(root, seed=1, variant="semfix", node="sealed_latent", **opts):
    opts = {"split": V2, "tasks": TASKS, **opts}
    return _ctx(_plan("pointer_sealed").nodes[f"{node}@{variant}.s{seed}"], root, **opts)


def _events(root):
    p = root / LOG
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def _sp(root):
    return SealedSplit.load("cworld_pointer_v2", path=root / V2)


def test_sealed_cells_run_once_and_only_infrastructure_failures_reopen(root, calls):
    ctx = _sealed(root)
    P.eval_r2(ctx)                                                        # first run: both sets, drag_window has no heldout
    ev = _events(root)
    assert sorted(e["event"] for e in ev) == ["done"] * N_CELLS + ["start"] * N_CELLS
    cells = {e["cell"] for e in ev}
    assert len(cells) == N_CELLS and all(c.startswith("cw_pointer|pointer_latent:") for c in cells)
    assert {c.split("|")[-1] for c in cells} == {"sealed_id", "sealed_heldout"}
    n_seeds = sorted(len(a[a.index("--seeds") + 1].split(",")) for a in calls)
    assert n_seeds == [50] * 3 + [100] * 4                                # the split's declared lists, nothing else
    for out in (root / ctx.rc.out).glob("*.jsonl"):
        out.unlink()                                                      # even with the outputs gone
    n = len(calls)
    with pytest.raises(SealedSplitError, match="already ran") as e:
        P.eval_r2(ctx)
    assert e.value.code == "sealed_cell_rerun" and len(calls) == n        # nothing was launched


def test_a_started_cell_refuses_the_whole_attempt_before_any_other_cell_starts(root, calls):
    ctx = _sealed(root)
    sp, log = _sp(root), root / LOG
    cell = P._sealed_cells(ctx, sp, P._split(ctx), "sealed_id", ["cw/calc_sum"], "pointer_latent")[0]
    with sp.sealed_eval(cell, log):
        pass                                                              # one cell already done
    with pytest.raises(SealedSplitError, match="already ran"):
        P.eval_r2(ctx)
    assert not calls and len(_events(root)) == 2                          # no job, and no other cell was started


def test_the_splits_env_kwargs_reach_every_eval_job(root, calls):
    assert json.loads((root / V2).read_text())["env_kw"] == {"strings": "procedural"}
    P.eval_r2(_sealed(root))
    assert len(calls) == N_CELLS and all(a.count("--env-kw") == 1 and "strings=procedural" in a for a in calls)
    calls.clear()
    P.eval_r2(_sealed(root, seed=2, seed_sets=["dev"]))                   # dev evaluation gets them too
    assert len(calls) == 4 and all("strings=procedural" in a for a in calls)


def test_a_crashed_sealed_attempt_stays_open_until_an_infrastructure_failure_is_recorded(root, monkeypatch, calls):
    ctx = _sealed(root)
    monkeypatch.setattr(B.StageContext, "run_parallel", lambda self, jobs, w: (_ for _ in ()).throw(B.StageError("node lost")))
    with pytest.raises(B.StageError):
        P.eval_r2(ctx)
    ev = _events(root)
    assert {e["event"] for e in ev} == {"start"}                          # the first seed set's cells: open, never done
    with pytest.raises(SealedSplitError, match="already ran"):            # a bad result / crash is not a free retry
        P.eval_r2(ctx)
    sp, log = _sp(root), root / LOG
    with pytest.raises(SealedSplitError, match="reason"):
        sp.record_infrastructure_failure(ev[0]["cell"], "  ", log)
    for e in ev:
        sp.record_infrastructure_failure(e["cell"], "peer OOM-killed the job (infra)", log)
    monkeypatch.setattr(B.StageContext, "run_parallel", lambda self, jobs, w: calls.extend(list(j[0]) for j in jobs))
    P.eval_r2(ctx)                                                        # released: one more attempt
    assert [e["event"] for e in _events(root)].count("done") == N_CELLS
    with pytest.raises(SealedSplitError, match="already ran"):
        P.eval_r2(ctx)


def test_the_method_is_its_frozen_checkpoints_so_other_seeds_and_arms_are_other_cells(root, calls):
    P.eval_r2(_sealed(root, seed=1))
    P.eval_r2(_sealed(root, seed=2))
    P.eval_r2(_sealed(root, seed=1, variant="nosem"))
    P.eval_r2(_sealed(root, seed=1, variant="nosem", node="sealed_bc", policy="pointer_bc"))
    assert len({e["cell"] for e in _events(root)}) == 4 * N_CELLS


def test_the_consumed_v1_sealed_sets_are_refused_but_dev_is_not(root, calls):
    ctx = _sealed(root, split=V1)
    with pytest.raises(SealedSplitError, match="consumed") as e:
        P.eval_r2(ctx)
    assert e.value.code == "sealed_split_consumed" and not calls and not _events(root)
    P.eval_r2(_sealed(root, split=V1, seed_sets=["dev"]))
    assert calls and not _events(root)                                    # dev evaluation is unguarded and unlogged


def test_an_unpinned_split_has_no_sealed_evaluation(root, calls):
    f = root / "research/splits/mine.json"
    f.write_text(json.dumps(dict(json.loads((root / V2).read_text()), split_id="mine_v9")))
    with pytest.raises(SealedSplitError, match="pins no such split") as e:
        P.eval_r2(_sealed(root, split="research/splits/mine.json"))
    assert e.value.code == "sealed_split_unknown" and not calls


def test_an_edited_split_file_is_refused_by_the_pin(root, calls):
    f = root / V2
    f.write_text(f.read_text() + "\n")                                     # any byte change breaks the sha256 pin
    with pytest.raises(SealedSplitError) as e:
        P.eval_r2(_sealed(root))
    assert e.value.code == "sealed_split_hash" and not calls


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
        assert free.pop("key_head") == "free" and copy.pop("key_head") == "copy" and bc["key_head"] == "copy"
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


def test_max_seeds_truncates_dev_seeds_only(root, calls):
    ctx = _ctx(_plan("pointer_smoke").nodes["eval_dev@semfix.s1"], root)
    assert len(P._seeds(ctx, "dev", "cw/calc_sum").split(",")) == 2
    with pytest.raises(B.StageError, match="only the dev seeds"):
        P._seeds(ctx, "sealed_id", "cw/calc_sum")
    full = _ctx(_plan("pointer_seeds").nodes["eval_dev@semfix.s1"], root)
    assert len(P._seeds(full, "dev", "cw/calc_sum").split(",")) == 50


def test_stage_jobs_keep_the_inherited_pythonpath(root, monkeypatch):
    """`StageContext.env` drops a PYTHONPATH that already holds the repo src (the computerworld wheel dir with it); the
    pointer stages pass the inherited path on so collect / train / eval find the optional extra on the peer."""
    ctx = _ctx(_plan("pointer_smoke").nodes["collect"], root)
    monkeypatch.setenv("PYTHONPATH", f"{root / 'src'}:/peer/cw-site")
    assert P._env(ctx)["PYTHONPATH"] == f"{root / 'src'}:/peer/cw-site"
