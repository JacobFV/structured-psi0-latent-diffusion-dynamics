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
