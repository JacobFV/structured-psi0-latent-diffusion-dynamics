"""HA (round 2): the adapting stages of the humanoid transfer matrix (`adapt_refit`, `adapt_flow`, `adapt_bc`, `adapt_ppo`), their
acquisition accounting, the sealed cell ids, the `pending` driver status and `legged_latent_eval` on `h_*` tasks.

Plumbing only: random checkpoints trained three CPU steps on a random tiny pack (the HD2 fixture of test_legged_train_upper), the
adapting stages run on them, no number here is a result. No simulation, no GPU, no warp: the PPO plan is argv arithmetic."""
import functools
import json
import shutil
from pathlib import Path

import pytest
import torch

from rrp.core.runconfig import RunConfig, RunIndex
from rrp.core.sealed import SealedSplit, SealedSplitError
from rrp.harness import dag
from rrp.harness.eval import humanoid_eval as HE
from rrp.harness.eval import legged_latent_eval as LE
from rrp.harness.pipelines import humanoid as HP
from rrp.harness.pipelines.base import _REGISTRY, StageContext, StageError, _load_families
from rrp.harness.train import humanoid_adapt as HA
from rrp.harness.train import legged_bc as TB
from rrp.harness.train import legged_latent_train as T
from .test_legged_relations import D, PROBES
from .test_legged_train_upper import _cfg, _write_pack

ROOT = Path(__file__).resolve().parents[2]
BUDGET, STEPS, TASK, BODY = 5, 3, "h_reach", "tinybody"


# ---------------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(autouse=True)
def _short_evals(monkeypatch):
    monkeypatch.setattr(T, "eval_rep", functools.partial(T.eval_rep, n_batches=2))
    monkeypatch.setattr(T, "eval_flow", functools.partial(T.eval_flow, n_batches=2))
    monkeypatch.setattr(TB, "eval_bc", functools.partial(TB.eval_bc, n_batches=2))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """(90-150 s to build on the host, so every test that takes it is marked slow.) A repo-like root: the pack `pack/pack.json` + `pack/n5/tinybody/` and three random source checkpoints (rep, flow, BC)
    trained three steps on the same tiny data."""
    root = tmp_path_factory.mktemp("ha")
    _write_pack(root / "artifacts/runs/pack" / f"n{BUDGET}", TASK, wholebody=True)
    # _write_pack writes <root>/tinybody/; the pack wants <pack>/n5/tinybody/ (the body directory of the nested pack)
    rec = dict(task=TASK, body=BODY, budget=BUDGET, demos=BUDGET, teacher_ticks=1234, env_samples=0, updates=STEPS, seeds=[1, 2, 19])
    (root / "artifacts/runs/pack/pack.json").write_text(json.dumps(dict(records={f"{TASK}|{BODY}|n{BUDGET}": rec})))
    src, data = root / "artifacts/runs", root / "artifacts/runs/pack" / f"n{BUDGET}"
    T.train_rep(_cfg(data, [PROBES, "preset:legged"]), src / "src_rep")
    T.train_flow(dict(representation=str(src / "src_rep" / "representation.pt"), steps=3, batch_size=8, width=D, layers=1,
                      semantic_weight=0.5), src / "src_flow")
    TB.train(dict(name="bc", data=str(data), bodies=[BODY], steps=3, batch_size=8, ckpt_every=1000,
                  snap_every=10 ** 6, model=dict(width=D, enc_layers=1, dec_layers=1, factors=["preset:legged"])), src / "src_bc")
    return root


def _ctx(world, stage, inputs, params=None, adapt=None, tmp=None, seed=0):
    root = tmp
    if not (root / "artifacts").exists():
        shutil.copytree(world / "artifacts", root / "artifacts")
    rc = RunConfig.model_validate(dict(
        schema_version="runconfig-1", family="humanoid", stage=stage, variant="na", seed=seed, lineage="ha", track="humanoid",
        flags=dict(zero_prev_action=None, realizer_anchor=None, realizer_drop_qd=None, probe_lv_min=None, qd_dropout=None,
                   contact_version="contact_v1"),
        inputs=inputs, params={**dict(steps=STEPS, batch_size=8), **(params or {})},
        options=dict(adapt=adapt or dict(task=TASK, body=BODY, budget=BUDGET))))
    return StageContext(rc=rc, index=RunIndex(), root=root)


def _state(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _moved(a: dict, b: dict) -> bool:
    return any(not torch.equal(a[k], b[k]) for k in a)


INPUTS = dict(pack="runs/pack:pack.json", representation="runs/src_rep:representation.pt", init="runs/src_flow:policy.pt")
BC_INPUTS = dict(pack=INPUTS["pack"], init="runs/src_bc:policy.pt")
PACK_JSON = "artifacts/runs/pack/pack.json"


# ---------------------------------------------------------------------------------------------------- registration
def test_adapt_stages_are_registered_and_waypoint_free_check_is_gone():
    _load_families()
    for st in HA.ADAPT_STAGES:
        assert ("humanoid", st) in _REGISTRY, st
    assert not hasattr(HP, "check_waypoint_free")


# ---------------------------------------------------------------------------------------------------- tiny CPU runs of each stage
@pytest.mark.slow
def test_adapt_refit_moves_system_zero_only(world, tmp_path):
    ctx = _ctx(world, "adapt_refit", {k: INPUTS[k] for k in ("pack", "representation")}, tmp=tmp_path)
    res = HP.adapt_refit(ctx)
    new = _state(tmp_path / res["outputs"]["representation"])
    old = _state(tmp_path / "artifacts/runs/src_rep/representation.pt")
    assert not _moved(old["E"], new["E"]) and not _moved(old["P"], new["P"])       # the latent space is unchanged
    assert _moved(old["R"], new["R"])
    assert new["result"]["moved"].startswith("R (system 0) only") and new["result"]["n_heldout_rows"] == 0
    assert new["cfg"]["adapt"]["budget"] == BUDGET
    assert (ctx.out / "train_log.jsonl").exists() or STEPS < 50                      # the log rows are every 50 steps


@pytest.mark.slow
def test_adapt_flow_moves_system_i_only(world, tmp_path):
    ctx = _ctx(world, "adapt_flow", INPUTS, tmp=tmp_path)
    res = HP.adapt_flow(ctx)
    new = _state(tmp_path / res["outputs"]["policy"])
    old = _state(tmp_path / "artifacts/runs/src_flow/policy.pt")
    assert _moved(old["flow"], new["flow"]) and new["result"]["moved"] == "flow (system i) only"
    assert new["cfg"]["init"].endswith("src_flow/policy.pt") and new["cfg"]["adapt"]["task"] == TASK
    assert res["source_detail"].endswith("policy.pt")


@pytest.mark.slow
def test_adapt_bc_fine_tunes_the_whole_policy(world, tmp_path):
    ctx = _ctx(world, "adapt_bc", BC_INPUTS, tmp=tmp_path)
    res = HP.adapt_bc(ctx)
    new = _state(tmp_path / res["outputs"]["policy"])
    old = _state(tmp_path / "artifacts/runs/src_bc/policy.pt")
    assert _moved(old["model"], new["model"]) and "POSITIVE CONTROL" in new["result"]["source"]
    assert new["result"]["upper_trained"] is True


@pytest.mark.slow
def test_adapt_refit_then_flow_on_the_refit_representation(world, tmp_path):
    """The joint cell: the flow warm start reads the refit representation (E identical, so the source flow still applies)."""
    c1 = _ctx(world, "adapt_refit", {k: INPUTS[k] for k in ("pack", "representation")}, tmp=tmp_path)
    rep = HP.adapt_refit(c1)["outputs"]["representation"]
    rid = rep.removeprefix("artifacts/").rsplit("/", 1)[0]
    c2 = _ctx(world, "adapt_flow", dict(INPUTS, representation=f"{rid}:representation.pt"), tmp=tmp_path)
    c2.rc = c2.rc.model_copy(update=dict(seed=1))
    res = HP.adapt_flow(c2)
    assert _state(tmp_path / res["outputs"]["policy"])["cfg"]["representation"].endswith("adapt_refit_s0/representation.pt")


# ---------------------------------------------------------------------------------------------------- accounting
@pytest.mark.slow
def test_acquisition_json_is_the_pack_record_and_names_the_native_cell(world, tmp_path):
    ctx = _ctx(world, "adapt_bc", BC_INPUTS, tmp=tmp_path)
    HP.adapt_bc(ctx)
    acq = json.loads((ctx.out / "acquisition.json").read_text())
    rec = json.loads((tmp_path / PACK_JSON).read_text())["records"][f"{TASK}|{BODY}|n{BUDGET}"]
    assert {k: acq[k] for k in HE.ACQ_KEYS} == {k: rec[k] for k in HE.ACQ_KEYS} == dict(demos=5, teacher_ticks=1234, env_samples=0, updates=STEPS)
    assert acq["cell"] == f"{BODY}|adapt_bc|{TASK}|n{BUDGET}|adapt_bc|s0|evaluation"     # SL's native cell id
    assert len(acq["pack_sha256"]) == 64
    # the driver accepts exactly this record
    cfg = HE.validate_config(dict(
        name="x", task=TASK, bodies=[BODY], train_seeds=[0], levels={"2": dict(unit="demos", budgets=[BUDGET], updates={str(BUDGET): STEPS})},
        methods=[dict(name="bc_sft", level=2, kind="bc", source="bc", trained=True, budgeted=True, policy="legged_bc", producer="adapt_bc",
                      kw=dict(checkpoint="artifacts/runs/humanoid/ha/adapt_bc_s<seed>/policy.pt"),
                      acquisition="artifacts/runs/humanoid/ha/adapt_bc_s<seed>/acquisition.json")]))
    st = HE.cell_state(cfg, HE.expand_cells(cfg)[0], tmp_path, HE.load_packs(dict(cfg, pack=[PACK_JSON]), tmp_path))
    assert st["status"] == "ready" and st["acquisition"] == dict(demos=5, teacher_ticks=1234, env_samples=0, updates=STEPS)


@pytest.mark.slow
def test_unmatched_update_count_is_refused(world, tmp_path):
    ctx = _ctx(world, "adapt_bc", BC_INPUTS, params=dict(steps=STEPS + 1), tmp=tmp_path)
    with pytest.raises(StageError, match="matched count"):
        HP.adapt_bc(ctx)
    assert not (ctx.out / "acquisition.json").exists() and not (ctx.out / "policy.pt").exists()


@pytest.mark.slow
def test_adapt_options_are_required(world, tmp_path):
    ctx = _ctx(world, "adapt_bc", BC_INPUTS, adapt=dict(task=TASK, body=BODY), tmp=tmp_path)
    with pytest.raises(StageError, match="options.adapt"):
        HP.adapt_bc(ctx)


@pytest.mark.slow
def test_a_sealed_body_trains_only_on_adaptation_seeds(world, tmp_path):
    """n1 is sealed: the pack's seeds (1, 2, 19) are source-range seeds, the stage refuses before reading or writing anything, and
    the error names the native cell id."""
    ctx = _ctx(world, "adapt_bc", BC_INPUTS, tmp=tmp_path,
               adapt=dict(task=TASK, body="n1", budget=BUDGET))
    pk = json.loads((tmp_path / PACK_JSON).read_text())
    pk["records"][f"{TASK}|n1|n{BUDGET}"] = dict(pk["records"][f"{TASK}|{BODY}|n{BUDGET}"], body="n1")
    (tmp_path / PACK_JSON).write_text(json.dumps(pk))
    with pytest.raises(SealedSplitError, match=r"n1\|adapt_bc\|h_reach\|n5\|adapt_bc\|s0\|evaluation"):
        HP.adapt_bc(ctx)
    assert not (ctx.out / "policy.pt").exists()


@pytest.mark.slow
def test_the_adaptation_seed_range_is_accepted_for_a_sealed_body(world, tmp_path):
    ctx = _ctx(world, "adapt_bc", BC_INPUTS, tmp=tmp_path,
               adapt=dict(task=TASK, body="n1", budget=BUDGET))
    lo = SealedSplit.load().ranges["adaptation"][0]
    cid = HP._sealed_adapt_guard(ctx, dict(task=TASK, body="n1", budget=BUDGET), [lo, lo + 1], 0)
    assert cid == f"n1|adapt_bc|{TASK}|n{BUDGET}|adapt_bc|s0|evaluation"


# ---------------------------------------------------------------------------------------------------- Level 1: PPO accounting
def test_ppo_iterations_are_derived_from_the_env_sample_budget():
    p = HA.adapt_ppo_plan(dict(task="h_steps", body="h1", budget=1_000_000, mode="scratch"), dict(nworld=1000, horizon=25), out="o")
    assert (p["iters"], p["samples_per_iter"]) == (40, 25_000) and p["argv"][-3:] == ["--iters", "40", "--resume"]
    assert "--init-shared" not in p["argv"]
    acq = HA.ppo_acquisition(p)
    assert acq["env_samples"] == 1_000_000 and (acq["demos"], acq["teacher_ticks"], acq["updates"]) == (0, 0, 0)
    # the driver's expectation of a samples-unit level is exactly {env_samples: budget}
    cfg = dict(levels={"1": dict(unit="samples", budgets=[1_000_000])})
    assert HE.acquisition_of_pack({}, cfg, dict(level="1", budget=1_000_000)) == {k: acq[k] for k in HE.ACQ_KEYS}


def test_ppo_finetune_needs_init_and_scratch_refuses_one():
    base = dict(task="h_steps", body="h1", budget=1_000_000)
    kw = dict(args=dict(nworld=1000, horizon=25), out="o")
    with pytest.raises(ValueError, match="needs an `init`"):
        HA.adapt_ppo_plan(dict(base, mode="finetune"), kw["args"], out="o")
    with pytest.raises(ValueError, match="scratch starts from no actor"):
        HA.adapt_ppo_plan(dict(base, mode="scratch"), kw["args"], out="o", init="a.pt")
    p = HA.adapt_ppo_plan(dict(base, mode="finetune"), kw["args"], out="o", init="a.pt")
    assert p["argv"][p["argv"].index("--init-shared") + 1] == "a.pt"
    with pytest.raises(ValueError, match="mode"):
        HA.adapt_ppo_plan(dict(base, mode="resume"), kw["args"], out="o")


def test_ppo_budget_that_is_not_a_whole_number_of_iterations_is_refused():
    with pytest.raises(ValueError, match="not a multiple"):
        HA.adapt_ppo_plan(dict(task="h_steps", body="h1", budget=1_000_000, mode="scratch"), {}, out="o")      # 4096 * 24 per iteration


@pytest.mark.parametrize("opt", ["iters", "init_shared", "body", "out", "resume"])
def test_ppo_options_owned_by_the_stage_are_refused(opt):
    with pytest.raises(ValueError, match="set by the stage"):
        HA.adapt_ppo_plan(dict(task="h_steps", body="h1", budget=98_304, mode="scratch"), {opt: 1}, out="o")


def test_ppo_groups_other_than_the_adapted_body_are_refused():
    """D-147 (2026-10-03): `groups` is allowed only as one group of exactly the adapted body (shared-tracker fine-tune)."""
    for g in (1, '[[["t1"], 4096]]', '[[["h1"], 2048], [["t1"], 2048]]'):
        with pytest.raises(ValueError, match="not a per-body budget"):
            HA.adapt_ppo_plan(dict(task="h_steps", body="h1", budget=98_304, mode="scratch"), {"groups": g}, out="o")


def test_ppo_on_a_sealed_body_passes_only_an_adaptation_seed():
    a = dict(task="h_steps", body="n1", budget=1_000_000, mode="scratch")
    with pytest.raises(SealedSplitError, match="target-adaptation"):
        HA.adapt_ppo_plan(a, dict(nworld=1000, horizon=25), out="o")                     # the trainer's default seed is a source seed
    lo = SealedSplit.load().ranges["adaptation"][0]
    assert HA.adapt_ppo_plan(a, dict(nworld=1000, horizon=25, seed=lo), out="o")["seed"] == lo


@pytest.mark.slow
def test_ppo_stage_checks_the_trainers_own_log(world, tmp_path):
    """No simulation: the trainer subprocess is stubbed. The stage writes acquisition.json only when the trainer's log ends at
    exactly the budget."""
    _load_families()
    plan_args = dict(nworld=1000, horizon=25)
    rc = RunConfig.model_validate(dict(
        schema_version="runconfig-1", family="humanoid", stage="adapt_ppo", variant="na", seed=1, lineage="ha", track="humanoid",
        flags=dict(zero_prev_action=None, realizer_anchor=None, realizer_drop_qd=None, probe_lv_min=None, qd_dropout=None,
                   contact_version="contact_v1"),
        options=dict(adapt=dict(task="h_steps", body="h1", budget=1_000_000, mode="scratch"), args=plan_args)))
    ctx = StageContext(rc=rc, index=RunIndex(), root=tmp_path)
    calls = []

    def fake_run(argv, env=None, log_to=None, samples=1_000_000):
        calls.append(argv)
        ctx.out.mkdir(parents=True, exist_ok=True)
        (ctx.out / "actor.pt").write_bytes(b"x")
        (ctx.out / "meta.json").write_text(json.dumps(dict(body="h1", source_label="learned_tracker (test stub)", init_from=None)))
        (ctx.out / "train_log.jsonl").write_text(json.dumps(dict(it=39, samples=samples)) + "\n")
    ctx.run = fake_run
    res = HP.adapt_ppo(ctx)
    assert calls[0][:4] == ["-m", "rrp.cli", "train", "tracker-warp"] and "--iters" in calls[0]
    acq = json.loads((ctx.out / "acquisition.json").read_text())
    assert acq["env_samples"] == acq["measured_env_samples"] == 1_000_000 and acq["cell"] == f"h1|adapt_ppo|h_steps|n1000000|adapt_ppo|s1|evaluation"
    assert res["outputs"]["actor"].endswith("actor.pt") and res["source_detail"].startswith("learned_tracker")
    (ctx.out / "acquisition.json").unlink()
    ctx.run = functools.partial(fake_run, samples=999_999)
    with pytest.raises(StageError, match="log ends at 999999"):
        HP.adapt_ppo(ctx)
    assert not (ctx.out / "acquisition.json").exists()


# ---------------------------------------------------------------------------------------------------- native sealed cell ids + pending
def _cfg_with_producers(task):
    from .test_humanoid_transfer import _plan
    plan = _plan(ROOT / f"recipes/humanoid/transfer_{task}.yaml")
    cfg = HE.validate_config(json.loads(json.dumps(plan.nodes["eval@semfix.s0"].rc.options["transfer"])))
    by_name = {"semfix_zeroshot": "train_flow", "nosem_zeroshot": "train_flow", "bc_zeroshot": "train_bc", "semfix_refit": "adapt_refit",
               "nosem_refit": "adapt_refit", "semfix_joint": "adapt_flow", "nosem_joint": "adapt_flow", "bc_sft": "adapt_bc",
               "shared_morph_zeroshot": "tracker-install", "task_expert_zeroshot": "tracker-install", "ppo_finetune": "adapt_ppo",
               "ppo_scratch": "adapt_ppo"}
    for m in cfg["methods"]:
        if m["name"] in by_name:
            m["producer"] = by_name[m["name"]]
    return cfg


@pytest.mark.parametrize("task", ["h_steps", "h_walk"])
def test_a_dry_run_table_with_declared_producers_has_no_missing_run(task, tmp_path):
    """The HA acceptance: every cell is ready (the scripted teacher) or pending (a stage of the recipe produces it); `missing_run`
    is left for a run nothing plans. The methods' producers are declared in the config (HR owns the templates)."""
    cfg = _cfg_with_producers(task)
    plan = HE.run_matrix(cfg, root=tmp_path, out=tmp_path / "out", scope="dev", sealed_flag=False, run=False, pack=HE.load_packs(cfg, tmp_path))
    plan_s = HE.run_matrix(cfg, root=tmp_path, out=tmp_path / "out", scope="sealed", sealed_flag=False, run=False, pack=HE.load_packs(cfg, tmp_path))
    cells = plan["cells"] + plan_s["cells"]
    statuses = {c["status"] for c in cells}
    assert statuses <= {"ready", "pending"} and "pending" in statuses, statuses
    assert all(c["status"] == "ready" for c in cells if c["method"] == "teacher")
    t = HE.make_tables(cfg, cells)
    assert t["coverage"] and not any("missing_run" in k for k in t["coverage"])


def test_without_a_producer_a_missing_run_stays_missing_run(tmp_path):
    cfg = HE.validate_config(dict(
        name="x", task="h_walk", bodies=["t1"], train_seeds=[0], levels={"2": dict(unit="demos", budgets=[5], updates={"5": 150})},
        methods=[dict(name="z", level=2, kind="bc", source="bc", trained=True, policy="legged_bc", kw=dict(checkpoint="nowhere/policy.pt"))]))
    st = HE.cell_state(cfg, HE.expand_cells(cfg)[0], tmp_path, dict(records={}))
    assert st["status"] == "missing_run"
    cfg["methods"][0]["producer"] = "train_bc"
    assert HE.cell_state(cfg, HE.expand_cells(cfg)[0], tmp_path, dict(records={}))["status"] == "pending"
    with pytest.raises(ValueError, match="producer"):
        HE.validate_config(dict(cfg, methods=[dict(cfg["methods"][0], producer="magic")]))


def test_an_accounting_mismatch_is_unaccounted_even_with_a_producer(tmp_path):
    cfg = HE.validate_config(dict(
        name="x", task="h_walk", bodies=["t1"], train_seeds=[0], levels={"2": dict(unit="demos", budgets=[5], updates={"5": 150})},
        methods=[dict(name="b", level=2, kind="bc", source="bc", trained=True, budgeted=True, policy="legged_bc", producer="adapt_bc",
                      kw=dict(checkpoint="r/policy.pt"), acquisition="r/acquisition.json")]))
    (tmp_path / "r").mkdir()
    (tmp_path / "r/policy.pt").write_bytes(b"x")
    (tmp_path / "r/acquisition.json").write_text(json.dumps(dict(demos=5, teacher_ticks=1, env_samples=0, updates=149)))
    pack = dict(records={"h_walk|t1|n5": dict(demos=5, teacher_ticks=1, env_samples=0, updates=150)})
    assert HE.cell_state(cfg, HE.expand_cells(cfg)[0], tmp_path, pack)["status"] == "unaccounted"


def test_sealed_cell_is_the_native_cell_of_the_method_and_budget():
    cfg = HE.validate_config(dict(
        name="x", task="h_walk", bodies=["n1"], train_seeds=[0], levels={"2": dict(unit="demos", budgets=[5, 20], updates={"5": 150, "20": 300})},
        methods=[dict(name="bc_sft", level=2, kind="bc", source="bc", trained=True, budgeted=True, policy="legged_bc", producer="adapt_bc"),
                 dict(name="bc_zs", level=2, kind="bc", source="bc", trained=True, policy="legged_bc", producer="train_bc")]))
    scenes = list(range(2_000_000, 2_000_100))
    ids = [SealedSplit.cell_id(HE.sealed_cell(cfg, c, scenes)) for c in HE.expand_cells(cfg)]
    assert ids == ["n1|bc_sft|h_walk|n5|adapt_bc|s0|evaluation", "n1|bc_sft|h_walk|n20|adapt_bc|s0|evaluation",
                   "n1|bc_zs|h_walk|-|-|s0|evaluation"]


# ---------------------------------------------------------------------------------------------------- legged_latent_eval on h_* tasks
def test_legged_latent_eval_refuses_waypoint_only_options_for_h_tasks():
    for kw in (dict(arc_only=True), dict(video=True), dict(scenario=object())):
        with pytest.raises(ValueError, match="waypoint_contact only"):
            LE.run_episode(None, "h1", 7200, task="h_steps", **kw)
    with pytest.raises(ValueError, match="tracker="):
        LE.run_episode(None, "go2", 10000, tracker="go2:x")


@pytest.mark.parametrize("argv, msg", [(["--task", "h_steps", "--edit", "halt"], "waypoint_contact only"),
                                       (["--task", "waypoint_contact", "--tracker", "h1:x"], "--tracker"),
                                       (["--task", "pick"], "h_\\* task")])
def test_legged_latent_eval_cli_guards(argv, msg, capsys):
    with pytest.raises(SystemExit):
        LE.main(["--bodies", "h1", "--seeds", "1", "--out", "/dev/null", *argv])
    assert __import__("re").search(msg, capsys.readouterr().err)


def test_legged_latent_eval_runs_an_h_task_through_the_generic_path(monkeypatch):
    """run_task_episode over evaluate(): the scripted teacher of the task, the tracker spec in env_kw, the row's core keys."""
    from types import SimpleNamespace
    seen = {}
    ep = SimpleNamespace(outcome="timeout", failure_reason="fell", success_privileged=False, success_public=False, policy="t",
                         source="scripted_teacher", time=2.5, steps=5, metrics={})

    def fake_eval(policy, env_id, task, body, seeds, **kw):
        seen.update(policy=policy, env_id=env_id, task=task, body=body, seeds=list(seeds), env_kw=kw["env_kw"])
        return [ep]
    monkeypatch.setattr("rrp.harness.eval.evaluate.evaluate", fake_eval)
    monkeypatch.setattr("rrp.policies.base.make_policy", lambda name, **k: f"policy<{name}>")
    row, frames = LE.run_episode(None, "h1", 7200, task="h_steps", tracker="h1:v", max_s=None)
    assert seen == dict(policy="policy<teacher:h_steps>", env_id="mujoco/legged", task="h_steps", body="h1", seeds=[7200],
                        env_kw=dict(tracker="h1:v")) and frames == []
    assert (row["task"], row["success"], row["fell"], row["failure_stage"], row["tracker"]) == ("h_steps", False, True, "fell", "h1:v")
    assert row["source"] == "scripted_teacher" and row["sim_time"] == 2.5
