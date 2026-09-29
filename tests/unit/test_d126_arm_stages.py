"""D-126 arm stages (grpo + anchors, target_eval, target_adapt, train_bc) and the anchor regression rule. Heavy functions
are replaced by fakes (no simulation on the host, D-127); the rule math is tested directly."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rrp.core.runconfig import RunConfig, RunIndex
from rrp.harness.pipelines.base import Pipeline, StageError
from rrp.harness.pipelines import base as pbase

EVAL_FLAGS = dict(zero_prev_action=True, realizer_anchor=None, realizer_drop_qd=None, probe_lv_min=None,
                  qd_dropout=None, contact_version="contact_v1")
META_FLAGS = dict(EVAL_FLAGS, zero_prev_action=None)
ROOT = Path(__file__).resolve().parents[2]


def _rc(stage, flags=EVAL_FLAGS, **kw) -> RunConfig:
    d = dict(schema_version="runconfig-1", family="arm", stage=stage, variant="semfix", seed=1, lineage="l", track="t",
             tag="x", flags=flags)
    d.update(kw)
    return RunConfig.model_validate(d)


def _touch(root: Path, *rels):
    for r in rels:
        p = root / r
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")


def _proto(root: Path):
    p = root / "configs/eval/latent_slice1.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text((ROOT / "configs/eval/latent_slice1.json").read_text())


# ---------------------------------------------------------------------------------------------- anchor rule
def test_anchor_verdict_rule():
    from rrp.harness.train.grpo_anchor import AnchorConfig, anchor_verdict
    cfg = AnchorConfig(robots=["panda_pg2", "parm6_tf3"], max_drop=0.10, max_drop_robot=0.20)
    ref = {"panda_pg2": {"success": 20, "n": 30}, "parm6_tf3": {"success": 25, "n": 30}}
    same = anchor_verdict(ref, ref, cfg)
    assert not same["regressed"] and same["pooled"]["drop"] == 0
    small = anchor_verdict(ref, {"panda_pg2": {"success": 18, "n": 30}, "parm6_tf3": {"success": 24, "n": 30}}, cfg)
    assert not small["regressed"]                                      # pooled drop 3/60 = 0.05
    pooled = anchor_verdict(ref, {"panda_pg2": {"success": 16, "n": 30}, "parm6_tf3": {"success": 22, "n": 30}}, cfg)
    assert pooled["regressed"] and "pooled" in pooled["reasons"][0]    # 7/60 > 0.10
    one = anchor_verdict(ref, {"panda_pg2": {"success": 13, "n": 30}, "parm6_tf3": {"success": 30, "n": 30}}, cfg)
    assert one["regressed"] and one["reasons"] == ["panda_pg2 drop 0.233 > 0.2"]   # pooled gain, one robot collapses
    lo, hi = one["pooled"]["diff_cur_minus_ref_ci95"]
    assert lo < 0 < hi
    with pytest.raises(ValueError):
        anchor_verdict(ref, {"panda_pg2": {"success": 1, "n": 29}, "parm6_tf3": {"success": 1, "n": 30}}, cfg)
    with pytest.raises(ValueError):
        AnchorConfig(robots=["xarm7_pg2"])
    with pytest.raises(ValueError):
        AnchorConfig(seed_start=2_000_000)


def test_anchor_tracker_stop_keeps_last_good_weights(tmp_path):
    torch = pytest.importorskip("torch")
    from rrp.harness.train.grpo_anchor import AnchorConfig, AnchorTracker
    m = torch.nn.Linear(2, 1)
    seq = iter([{"panda_pg2": {"success": 20, "n": 30}}, {"panda_pg2": {"success": 21, "n": 30}},
                {"panda_pg2": {"success": 5, "n": 30}}])
    tr = AnchorTracker(AnchorConfig(robots=["panda_pg2"], action="stop"), tmp_path, lambda tag: next(seq))
    tr.reference(m)
    with torch.no_grad():
        m.weight.fill_(1.0)
    assert not tr.check(m, "it5")["regressed"]
    good = m.weight.detach().clone()
    with torch.no_grad():
        m.weight.fill_(9.0)
    assert tr.check(m, "it10")["regressed"] and tr.stopped
    assert tr.restore_best(m) == "it5" and torch.equal(m.weight, good)
    rep = json.loads((tmp_path / "anchor_report.json").read_text())
    assert rep["kept_checkpoint"] == "it5" and rep["stopped_on_regression"] and len(rep["checks"]) == 2


# ---------------------------------------------------------------------------------------------- stages
def test_grpo_stage_latent_wiring(tmp_path, monkeypatch):
    import rrp.harness.train.latent_grpo as lg
    seen = {}

    def fake(cfg):
        seen["cfg"] = cfg
        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for f in ("policy.pt", "result.json", "anchor_report.json"):
            (out / f).write_text("{}")
        return dict(evals=[dict(tag="reference@0", successes=1, episodes=4)], accounting={"train_env_steps": 10},
                    budget=dict(budget=10, spent=10), anchor=dict(kept_checkpoint="latent_grpo@5"))
    monkeypatch.setattr(lg, "train_latent_grpo", fake)
    _touch(tmp_path, "artifacts/runs/f/policy.pt", "artifacts/runs/r/representation.pt")
    rc = _rc("grpo", inputs={"flow": "runs/f:policy.pt", "representation": "runs/r:representation.pt"},
             params=dict(robot="parm6_tf3", iters=50, budget_env_steps=100000, grpo={"lr": 2e-6, "sde": {"nfe": 8}}),
             options=dict(method="latent", anchor=dict(robots=["panda_pg2"], episodes=30, action="stop")))
    body = Pipeline("arm").run(rc, root=tmp_path, index=RunIndex())
    c = seen["cfg"]
    assert c.representation == "artifacts/runs/r/representation.pt" and c.budget_env_steps == 100000
    assert c.grpo.lr == 2e-6 and c.grpo.kl_coef == 0.05 and c.grpo.sde.nfe == 8 and c.snapshot_evals
    assert c.anchor["prev_action"] == "zero" and c.anchor["action"] == "stop"
    assert body["metrics"]["anchor"]["kept_checkpoint"] == "latent_grpo@5"
    with pytest.raises(StageError, match="budget_env_steps"):
        Pipeline("arm").run(rc.model_copy(update={"params": dict(robot="parm6_tf3")}), root=tmp_path, index=RunIndex())


def test_grpo_stage_bc_wiring(tmp_path, monkeypatch):
    import rrp.harness.train.adapt as ad
    seen = {}

    def fake(cfg):
        seen["cfg"] = cfg
        out = Path(cfg["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text("{}")
        (out / "policy_b100000.pt").write_text("x")
        (out / "anchor_report.json").write_text("{}")
        return dict(evals=[], counters={"new_transitions": 100000})
    monkeypatch.setattr(ad, "run", fake)
    _touch(tmp_path, "artifacts/runs/bc/policy.pt")
    rc = _rc("grpo", inputs={"bc_policy": "runs/bc:policy.pt"},
             params=dict(robot="parm6_tf3", budgets=[0, 100000], train_seed_start=3100000, eval_seed_start=3000000,
                         eval_episodes=64, max_steps=300),
             options=dict(method="bc", anchor=dict(robots=["panda_pg2"])))
    body = Pipeline("arm").run(rc, root=tmp_path, index=RunIndex())
    assert seen["cfg"]["method"] == "grpo" and seen["cfg"]["checkpoint"] == "artifacts/runs/bc/policy.pt"
    assert seen["cfg"]["anchor"]["robots"] == ["panda_pg2"] and "policy" in body["outputs"]


def test_target_eval_stage_argv_and_sealed_flag(tmp_path, monkeypatch):
    _proto(tmp_path)
    calls = []

    def fake_run(self, argv, *, env=None, log_to=None):
        calls.append(argv)
        out = self.root / argv[argv.index("--out") + 1]
        out.mkdir(parents=True, exist_ok=True)
        name = f"{argv[argv.index('--route') + 1]}_{argv[argv.index('--tag') + 1]}"
        (out / f"{name}.jsonl").write_text("")
        (out / f"{name}.summary.json").write_text(json.dumps(dict(n=97, success=50, scene_kind="target", smoke=False)))
    monkeypatch.setattr(pbase.StageContext, "run", fake_run)
    _touch(tmp_path, "artifacts/runs/f/policy.pt", "artifacts/runs/r/representation.pt", "artifacts/runs/bc/policy.pt")
    rc = _rc("target_eval", inputs={"flow": "runs/f:policy.pt", "representation": "runs/r:representation.pt"},
             options=dict(robot="xarm7_pg2", sealed_run=True, chunk_blend="crossfade"))
    body = Pipeline("arm").run(rc, root=tmp_path, index=RunIndex())
    a = calls[-1]
    assert a[:4] == ["-m", "rrp.cli", "suite", "target"] and "--sealed-run" in a and "--flow" in a
    assert a[a.index("--chunk-blend") + 1] == "crossfade" and a[a.index("--prev-action") + 1] == "zero"
    assert body["metrics"]["protocol"]["id"] == "latent_slice1"
    rc2 = _rc("target_eval", inputs={"bc_policy": "runs/bc:policy.pt"},
              options=dict(robot="xarm7_pg2", route="learned", bc_label="bcv6_1702"))
    Pipeline("arm").run(rc2, root=tmp_path, index=RunIndex())
    assert "--sealed-run" not in calls[-1] and "--policy" in calls[-1]      # target_eval itself then refuses


def test_target_eval_scene_rules():
    from rrp.harness.eval.target_eval import load_protocol, plan_scenes
    proto, _ = load_protocol(ROOT / "configs/eval/latent_slice1.json")
    assert plan_scenes(proto, "xarm7_tf3", sealed_run=True, smoke=False) == dict(kind="target", seed_start=2000000,
                                                                                episodes=100)
    assert plan_scenes(proto, "parm5s_tf3", sealed_run=True, smoke=False)["kind"] == "source_heldout"
    with pytest.raises(ValueError, match="sealed-run"):
        plan_scenes(proto, "xarm7_tf3", sealed_run=False, smoke=False)
    with pytest.raises(ValueError, match="target"):
        plan_scenes(proto, "panda_tf3", sealed_run=False, smoke=True)
    with pytest.raises(ValueError):
        plan_scenes(proto, "panda_pg2", sealed_run=True, smoke=False)      # a training body is not a sealed scene
    assert plan_scenes(proto, "panda_pg2", sealed_run=False, smoke=True)["seed_start"] == 3000000


def test_target_adapt_stage_protocol_budgets(tmp_path, monkeypatch):
    _proto(tmp_path)
    import rrp.harness.train.latent_train as lt
    import rrp.harness.train.sft as sft
    seen = {}

    def fake_flow(ck, pack, budget, *, seed, out_dir, steps, lr):
        seen.update(kind="flow", budget=budget, seed=seed, steps=steps, lr=lr)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "policy.pt").write_text("x")
        return dict(demo_episodes=budget)

    def fake_bc(ck, pack, budget, *, seed, out_dir, steps, lr):
        seen.update(kind="bc", budget=budget, steps=steps)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "policy.pt").write_text("x")
        return dict(demo_episodes=budget)
    monkeypatch.setattr(lt, "sft_latent_flow", fake_flow)
    monkeypatch.setattr(sft, "sft_packed", fake_bc)
    _touch(tmp_path, "artifacts/runs/f/policy.pt", "artifacts/packed/tx/meta.json", "artifacts/runs/bc/policy.pt")
    (tmp_path / "artifacts/packed/tx/meta.json").write_text(json.dumps({"robots": ["xarm7_pg2"]}))
    rc = _rc("target_adapt", flags=META_FLAGS, inputs={"flow": "runs/f:policy.pt", "packed_dir": "packed/tx"},
             options=dict(method="flow_sft", target="xarm7_pg2", budget=20, adapt_seed=1701))
    Pipeline("arm").run(rc, root=tmp_path, index=RunIndex())
    assert seen == dict(kind="flow", budget=20, seed=1701, steps=300, lr=1e-4)
    bad = rc.model_copy(update={"options": dict(rc.options, budget=7)})
    with pytest.raises(StageError, match="sft_budgets"):
        Pipeline("arm").run(bad, root=tmp_path, index=RunIndex())
    with pytest.raises(StageError, match="not only"):
        Pipeline("arm").run(rc.model_copy(update={"options": dict(rc.options, target="xarm7_tf3")}), root=tmp_path,
                            index=RunIndex())
    rc_bc = _rc("target_adapt", flags=META_FLAGS, inputs={"bc_policy": "runs/bc:policy.pt", "packed_dir": "packed/tx"},
                options=dict(method="bc_sft", target="xarm7_pg2", budget=100, adapt_seed=1703))
    Pipeline("arm").run(rc_bc, root=tmp_path, index=RunIndex())
    assert seen["kind"] == "bc" and seen["steps"] == 600


def test_arm_eval_r2_route_learned_and_blend_args(tmp_path, monkeypatch):
    calls = []

    def fake_run(self, argv, *, env=None, log_to=None):
        calls.append(argv)
        out = self.root / argv[argv.index("--out") + 1]
        out.mkdir(parents=True, exist_ok=True)
        name = f"{argv[argv.index('--route') + 1]}_{argv[argv.index('--tag') + 1]}"
        (out / f"{name}.jsonl").write_text("")
        (out / f"{name}.summary.json").write_text(json.dumps(dict(n=30, success=3, rate=0.1, wilson95=[0, 1])))
    monkeypatch.setattr(pbase.StageContext, "run", fake_run)
    _touch(tmp_path, "artifacts/runs/bc/policy.pt", "artifacts/runs/f/policy.pt", "artifacts/runs/r/representation.pt")
    rc = _rc("eval_r2", inputs={"bc_policy": "runs/bc:policy.pt"},
             options=dict(tag="bc", route="learned", robots=["panda_pg2"], bc_label="bcv6"))
    body = Pipeline("arm").run(rc, root=tmp_path, index=RunIndex())
    a = calls[-1]
    assert a[a.index("--route") + 1] == "learned" and a[a.index("--policy-label") + 1] == "bcv6"
    assert "--chunk-blend" not in a and body["manifest_hash"]
    rc2 = _rc("eval_r2", inputs={"flow": "runs/f:policy.pt", "representation": "runs/r:representation.pt"},
              options=dict(tag="g", robots=["panda_pg2"], chunk_blend="ensemble", blend_decay=0.5))
    Pipeline("arm").run(rc2, root=tmp_path, index=RunIndex())
    a = calls[-1]
    assert a[a.index("--chunk-blend") + 1] == "ensemble" and a[a.index("--blend-decay") + 1] == "0.5"


def test_default_off_configs_unchanged():
    """Adding the D-126 stages/options changes no existing planned node: the committed arm DAGs plan with the same
    config hashes (the flag spec of every pre-existing (family, stage) is unchanged)."""
    from rrp.core.runconfig import FLAG_SPEC
    assert FLAG_SPEC[("dual", "grpo")] == {"contact_version": "@meta"}
    assert FLAG_SPEC[("arm", "eval_r2")] == {"zero_prev_action": "@cli", "contact_version": "@meta"}
    assert FLAG_SPEC[("dual", "train_bc")] == {"contact_version": "@meta"}


def test_grasp_contact_option_sets_env_for_the_stage(tmp_path, monkeypatch):
    import os
    monkeypatch.delenv("RRP_GRASP_CONTACT", raising=False)
    seen = {}

    def fake_run(self, argv, *, env=None, log_to=None):
        seen["env"] = (env or {}).get("RRP_GRASP_CONTACT")
        out = self.root / argv[argv.index("--out") + 1]
        out.mkdir(parents=True, exist_ok=True)
        name = f"{argv[argv.index('--route') + 1]}_{argv[argv.index('--tag') + 1]}"
        (out / f"{name}.jsonl").write_text("")
        (out / f"{name}.summary.json").write_text(json.dumps(dict(n=30, success=1, rate=1, wilson95=[0, 1])))
    monkeypatch.setattr(pbase.StageContext, "run", fake_run)
    _touch(tmp_path, "artifacts/runs/f/policy.pt", "artifacts/runs/r/representation.pt")
    rc = _rc("eval_r2", inputs={"flow": "runs/f:policy.pt", "representation": "runs/r:representation.pt"},
             options=dict(tag="g", robots=["panda_pg2"], grasp_contact="v2.1"))
    Pipeline("arm").run(rc, root=tmp_path, index=RunIndex())
    assert seen["env"] == "v2.1" and "RRP_GRASP_CONTACT" not in os.environ
    monkeypatch.setenv("RRP_GRASP_CONTACT", "v1")
    with pytest.raises(StageError, match="contradicts"):
        Pipeline("arm").run(rc, root=tmp_path, index=RunIndex())


def test_latent_grpo_d126_fields_default_off():
    from dataclasses import asdict
    from rrp.harness.train.latent_grpo import D126_FIELDS, LatentGRPORunConfig
    d = asdict(LatentGRPORunConfig(checkpoint="c", robot="r", out_dir="o"))
    assert all(d[k] in (None, False) for k in D126_FIELDS)     # dropped from config.json: historical runs unchanged
