"""Humanoid track (D-145 P4a): the eval_tracker / warp train_tracker stage plumbing that replaced scripts/humanoid_*.sh, and the
lab-gate math. No simulation: the stage's subprocess is captured."""
import json

import pytest

from rrp.core.runconfig import RunConfig, RunIndex
from rrp.harness.eval.humanoid_eval import lab_gate
from rrp.harness.pipelines import legged
from rrp.harness.pipelines.base import StageContext

FLAGS = dict(zero_prev_action=None, realizer_anchor=None, realizer_drop_qd=None, probe_lv_min=None, qd_dropout=None,
             contact_version="contact_v2")


def _ctx(tmp_path, stage, options, inputs=None):
    rc = RunConfig.model_validate(dict(schema_version="runconfig-1", family="legged", stage=stage, variant="na", seed=1, lineage="l",
                                       track="humanoid", flags=FLAGS, options=options, inputs=inputs or {}))
    ctx = StageContext(rc=rc, index=RunIndex(), root=tmp_path)
    ctx.out.mkdir(parents=True)
    calls = []
    ctx.run = lambda argv, env=None, log_to=None: calls.append((argv, env))
    return ctx, calls


def test_lab_gate_thresholds():
    v = dict(gate=dict(no_fall_rate=1.0, forward_ratio=0.8, turn_ratio=0.5, contact_gate=dict(slip_ratio=0.149)))
    assert lab_gate(v)["passed"]
    for k, bad in (("no_fall_rate", 0.9), ("forward_ratio", 0.79), ("turn_ratio", 0.49)):
        assert not lab_gate(dict(gate=dict(v["gate"], **{k: bad})))["passed"]
    assert not lab_gate(dict(gate=dict(v["gate"], contact_gate=dict(slip_ratio=0.15))))["passed"]
    out = lab_gate(v, dict(verdict="fail", criteria=[dict(name="c", value=1, status="fail")]), dict(success=3, fell=1, n=4))
    assert out["d112_verdict"] == "fail" and out["waypoint_success"] == 3


def test_eval_tracker_steps_runs_one_tool_call_per_height(tmp_path):
    ctx, calls = _ctx(tmp_path, "eval_tracker", dict(task="steps", body="h1", actor="a.pt", seed0=7200, n=2, h_fracs=[0.1, 0.15]))
    ctx.run = lambda argv, env=None, log_to=None: (calls.append(argv), __import__("pathlib").Path(argv[argv.index("--out") + 1]).write_text(
        json.dumps(dict(n=2, success=1, fell=0, source="x"))))
    res = legged.eval_tracker(ctx)
    assert [c[c.index("--h-frac") + 1] for c in calls] == ["0.1", "0.15"]
    assert all(c[:5] == ["-m", "rrp.cli", "suite", "humanoid-steps", "h1"] for c in calls)
    assert set(res["outputs"]) == {"steps_h0.10", "steps_h0.15"}


def test_train_tracker_warp_seeds_the_run_dir_from_resume_from(tmp_path):
    src = tmp_path / "artifacts/runs/old"
    src.mkdir(parents=True)
    for f in ("checkpoint.pt", "actor.pt"):
        (src / f).write_bytes(b"x")
    (src / "meta.json").write_text(json.dumps(dict(body="shared")))
    ctx, calls = _ctx(tmp_path, "train_tracker", dict(engine="warp", recipe="shared_morph_v1", resume_from="artifacts/runs/old", pythonpath=["/w"]))
    res = legged.train_tracker(ctx)
    argv, env = calls[0]
    assert argv[:5] == ["-m", "rrp.cli", "train", "tracker-warp", "--out"] and "--resume" in argv and "--contact" not in argv
    assert env["PYTHONPATH"].startswith("/w:") and (ctx.out / "checkpoint.pt").exists()
    assert res["metrics"]["body"] == "shared"
    with pytest.raises(legged.StageError, match="no checkpoint.pt"):
        legged.train_tracker(_ctx(tmp_path / "b", "train_tracker", dict(engine="warp", resume_from="nope"))[0])


# ---------------------------------------------------------------------------------------------------- HR (D-146 round 2): recipes for every task and tracker run
import re
from pathlib import Path

from rrp.core.sealed import SealedSplit
from rrp.harness import dag
from rrp.harness.eval import humanoid_eval as HE
from rrp.harness.pipelines.base import _load_families

ROOT = Path(__file__).resolve().parents[2]
HUM = ROOT / "recipes/humanoid"
HELD = ("h_steps_carry", "h_gap_cart")
FULL = ("h_walk", "h_turn", "h_reach", "h_squat_pick", "h_place", "h_carry", "h_loco_pick", "h_steps", "h_gap")
TRACKER_RECIPES = ("trackers_steps_scan", "trackers_gap_ring", "trackers_wholebody_ub", "trackers_shared_morph_ub", "trackers_pins")


def _plan(path):
    _load_families()
    return dag.plan_dag(dag.load_dag(path))


def _cfg(plan, node="eval@semfix.s0"):
    return HE.validate_config(json.loads(json.dumps(plan.nodes[node].rc.options["transfer"])))


def test_no_recipe_leaves_a_placeholder_behind():
    hits = [p.name for p in HUM.rglob("*") if p.is_file() and "SET_WHEN_REGISTERED" in p.read_text()]
    assert not hits, hits


def test_every_registered_humanoid_task_has_a_transfer_recipe():
    from rrp.policies.teachers.humanoid import ALL_MANIP_TEACHERS
    tasks = set(ALL_MANIP_TEACHERS) | {"h_steps", "h_gap"}
    assert tasks == set(FULL) | set(HELD), tasks ^ (set(FULL) | set(HELD))
    for t in tasks:
        assert (HUM / f"transfer_{t}.yaml").is_file(), t


@pytest.mark.parametrize("task", HELD)
def test_held_out_transfer_recipe_is_evaluation_only(task):
    plan = _plan(HUM / f"transfer_{task}.yaml")
    stages = {n.rc.stage for n in plan.nodes.values()}
    assert stages <= {"eval_transfer", "sealed_eval"}, stages                 # no collect / pack / train / adapt
    cfg = _cfg(plan)
    assert not any(m.get("producer") for m in cfg["methods"]) and not any(m.get("budgeted") for m in cfg["methods"])
    assert plan.nodes["sealed@semfix.s0"].rc.options["sealed"] is False       # the sealed cells run once, the lead flips it
    assert HE.expand_cells(cfg)                                               # and there is something to evaluate


@pytest.mark.parametrize("task", FULL)
def test_transfer_config_paths_are_the_planned_outs_of_the_adapting_nodes(task, tmp_path):
    plan = _plan(HUM / f"transfer_{task}.yaml")
    cfg = _cfg(plan)
    by_out = {n.rc.out: n for n in plan.nodes.values()}
    split = SealedSplit.load()
    want_stage = {"train_flow": {"train_flow", "train_rep"}, "train_bc": {"train_bc"}, "adapt_refit": {"adapt_refit", "train_flow"},   # a joint / zero-shot
                  "adapt_flow": {"adapt_flow", "adapt_refit"}, "adapt_bc": {"adapt_bc"}, "adapt_ppo": {"adapt_ppo"}}  # latent policy = flow + rep
    seen = set()
    for cell in HE.expand_cells(cfg):
        m = HE.method_of(cfg, cell["method"])
        if m.get("producer") not in want_stage:
            continue
        kv = dict(body=cell["body"], task=task, budget=cell["budget"], seed=cell["train_seed"])
        paths = [HE.fmt(v, **kv) for v in list((m.get("kw") or {}).values()) + [m.get("tracker", ""), m.get("acquisition", "")]
                 if isinstance(v, str) and v.endswith((".pt", ".json"))]
        assert paths, (m["name"], "names no run path")
        for p in paths:
            node = by_out.get(p.rsplit("/", 1)[0])
            assert node is not None, f"{m['name']}: {p} is not the out of any node of the recipe"
            assert node.rc.stage in want_stage[m["producer"]], (m["name"], node.rc.stage)
            seen.add((m["name"], cell["body"], cell["budget"], cell["train_seed"]))
            if node.rc.stage == m["producer"] and m["producer"].startswith("adapt_"):
                a = node.rc.options["adapt"]
                assert (a["task"], a["body"], a["budget"]) == (task, cell["body"], cell["budget"]), (m["name"], a)
                if m["producer"] != "adapt_ppo":
                    sealed = split.is_sealed_body(cell["body"])
                    assert node.rc.inputs["pack"].split(":")[0].rsplit("/", 1)[-1] == ("pack-sealed_s0" if sealed else "pack_s0")
                    assert node.rc.params["steps"] == cfg["levels"]["2"]["updates"][str(cell["budget"])]      # the matched update count
                    assert node.rc.seed == cell["train_seed"]
    # every (method, body, budget, seed) of every producing method was checked: nothing was skipped silently
    want = {(m["name"], c["body"], c["budget"], c["train_seed"]) for c in HE.expand_cells(cfg) for m in cfg["methods"]
            if m["name"] == c["method"] and m.get("producer") in want_stage}
    assert seen == want


@pytest.mark.parametrize("task", FULL)
def test_adapt_nodes_declare_the_bodies_of_the_config_and_sealed_packs_only_for_sealed_bodies(task):
    plan = _plan(HUM / f"transfer_{task}.yaml")
    cfg = _cfg(plan)
    split = SealedSplit.load()
    assert {b for b in cfg["bodies"] if split.is_sealed_body(b)} == set(plan.nodes["sealed@bc.s0"].rc.options["transfer"]["bodies"]["sealed"])
    bodies = {n.rc.options["adapt"]["body"] for n in plan.nodes.values() if n.rc.stage in ("adapt_refit", "adapt_flow", "adapt_bc")}
    assert bodies == set(cfg["bodies"]), bodies ^ set(cfg["bodies"])
    for n in plan.nodes.values():
        if n.rc.stage in ("adapt_refit", "adapt_flow", "adapt_bc"):
            assert ("pack-sealed" in n.rc.inputs["pack"]) == split.is_sealed_body(n.rc.options["adapt"]["body"])


@pytest.mark.parametrize("task", FULL)
def test_a_dry_run_table_has_no_missing_run_for_a_planned_producer(task, tmp_path):
    cfg = _cfg(_plan(HUM / f"transfer_{task}.yaml"))
    assert all(m.get("producer") for m in cfg["methods"] if m["level"] != 0), [m["name"] for m in cfg["methods"] if not m.get("producer")]
    cells = []
    for scope in ("dev", "sealed"):
        cells += HE.run_matrix(cfg, root=tmp_path, out=tmp_path / "o", scope=scope, sealed_flag=False, run=False,
                               pack=HE.load_packs(cfg, tmp_path))["cells"]
    assert {c["status"] for c in cells} <= {"ready", "pending", "unaccounted"}
    assert all(c["status"] == "ready" for c in cells if c["method"] == "teacher")


@pytest.mark.parametrize("task", FULL)
def test_ppo_nodes_are_accepted_by_the_stage_plan(task):
    plan = _plan(HUM / f"transfer_{task}.yaml")
    from rrp.harness.train.humanoid_adapt import adapt_ppo_plan
    split = SealedSplit.load()
    nodes = [n for n in plan.nodes.values() if n.rc.stage == "adapt_ppo"]
    assert bool(nodes) == (task in ("h_steps", "h_gap"))
    for n in nodes:
        o = n.rc.options
        init = n.rc.inputs.get("init") if o["adapt"]["mode"] == "finetune" else None
        assert (init is not None) == (o["adapt"]["mode"] == "finetune")
        p = adapt_ppo_plan(o["adapt"], dict(o["args"], recipe=o["recipe"]), out="o", init=init)
        assert p["iters"] * p["samples_per_iter"] == o["adapt"]["budget"]
        if split.is_sealed_body(o["adapt"]["body"]):
            lo, hi = split.ranges["adaptation"]
            assert lo <= p["seed"] < hi


def test_gap_ppo_recipes_are_the_registry_recipes_without_a_warm_start():
    from rrp.harness.train.tracker_recipes import WARP_RECIPES
    for name, reg in (("gap_t1", "t1_gap_gpu_v1"), ("gap_h1", "h1_gap_gpu_v1")):
        got = json.loads((HUM / f"ppo/{name}.json").read_text())
        want = {k: v for k, v in WARP_RECIPES[reg].items() if k != "init_shared"}
        assert {k: v for k, v in got.items() if not k.startswith("_")} == {k: v for k, v in want.items() if not k.startswith("_")}
        assert "init_shared" not in got


def test_transfer_tracker_specs_are_versions_the_tracker_recipes_install():
    """Every `<body>:<version>` of a transfer recipe (collect trackers, task-expert and teacher reference) is an install target of a tracker recipe."""
    from rrp.harness import yamlmini
    installs = set()
    for name in TRACKER_RECIPES:
        for body, ver in (yamlmini.load((HUM / f"{name}.yaml").read_text()).get("lists", {}).get("installs") or {}).items():
            installs.add(f"{body}:{ver}")
    wb = yamlmini.load((HUM / "trackers_wholebody_ub.yaml").read_text())["lists"]["installs"]       # {kind: version}, every body
    installs |= {f"{b}:{v}" for b in ("t1", "g1", "h1") for v in wb.values()}
    used = set()
    for task in FULL + HELD:
        plan = _plan(HUM / f"transfer_{task}.yaml")
        cfg = _cfg(plan)
        for b, spec in (plan.nodes["collect_src"].rc.options.get("trackers") or {}).items() if "collect_src" in plan.nodes else ():
            used.add(spec)
        for m in cfg["methods"]:
            env_kw = (m.get("env_kw") or {}).get("tracker")
            for spec in (env_kw, m.get("tracker")):
                if spec and not spec.endswith(".pt"):
                    used |= {HE.fmt(spec, body=b) for b in cfg["bodies"] if not SealedSplit.load().is_sealed_body(b)}
    missing = {s for s in used if s not in installs and not s.startswith("shared:")}
    assert not missing, missing
    assert "shared:morph_v2_ub" in {f"{b}:{v}" for b, v in yamlmini.load((HUM / "trackers_shared_morph_ub.yaml").read_text())["lists"]["installs"].items()}


@pytest.mark.parametrize("name", TRACKER_RECIPES)
def test_tracker_recipes_plan_and_their_trainers_exist(name):
    plan = _plan(HUM / f"{name}.yaml")
    assert any(n.rc.stage == "validate_tracker" for n in plan.nodes.values())
    from rrp.harness.train.tracker_recipes import WARP_RECIPES
    for n in plan.nodes.values():
        r = (n.rc.options or {}).get("recipe")
        if n.rc.stage == "train_tracker" and r:
            assert r in WARP_RECIPES or r.endswith(".json"), r


def test_teacher_check_nodes_run_the_scripted_teacher_over_the_installed_tracker_on_dev_bodies():
    for task, scenes in (("h_carry", 12), ("h_steps_carry", 10)):
        plan = _plan(HUM / f"transfer_{task}.yaml")
        cfg = HE.validate_config(json.loads(json.dumps(plan.nodes["teacher_check"].rc.options["transfer"])))
        assert cfg["scenes"] == scenes and [m["source"] for m in cfg["methods"]] == ["scripted_teacher"]
        assert not any(SealedSplit.load().is_sealed_body(b) for b in cfg["bodies"])
        assert all((m.get("env_kw") or {}).get("tracker") for m in cfg["methods"])


def test_legged_lineage_lists_the_legged_relation_preset_and_the_control_arm_is_a_var():
    text = (ROOT / "recipes/templates/legged_lineage.yaml").read_text()
    assert text.count("'preset:{rel_preset}'") >= 3 and "rel_preset: legged" in text


@pytest.mark.parametrize("task", HELD)
def test_held_out_cells_plan_without_a_pack_and_report_the_missing_h_carry_checkpoints(task, tmp_path):
    cfg = _cfg(_plan(HUM / f"transfer_{task}.yaml"))
    cells = []
    for scope in ("dev", "sealed"):
        cells += HE.run_matrix(cfg, root=tmp_path, out=tmp_path / "o", scope=scope, sealed_flag=False, run=False,
                               pack=HE.load_packs(cfg, tmp_path))["cells"]
    st = {c["method"]: c["status"] for c in cells}
    assert st["teacher"] == "ready" and {st[m] for m in st if m != "teacher"} == {"missing_run"}      # nothing is faked, nothing skipped
    assert all("h_carry" in HE.fmt(HE.method_of(cfg, m)["kw"].get("flow") or HE.method_of(cfg, m)["kw"]["checkpoint"]) for m in st if m != "teacher")


@pytest.mark.parametrize("task", ("h_carry", "h_steps_carry"))
def test_teacher_check_config_plans_as_ready_cells(task, tmp_path):
    cfg = HE.validate_config(json.loads(json.dumps(_plan(HUM / f"transfer_{task}.yaml").nodes["teacher_check"].rc.options["transfer"])))
    plan = HE.run_matrix(cfg, root=tmp_path, out=tmp_path / "o", scope="dev", sealed_flag=False, run=False, pack=HE.load_packs(cfg, tmp_path))
    assert {c["status"] for c in plan["cells"]} == {"ready"} and {c["body"] for c in plan["cells"]} == {"t1", "g1", "h1"}
