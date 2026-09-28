"""Pipeline stage wiring (W5): each family's stages call the existing entry points with the RunConfig's native config /
arguments and write a manifest with provenance. The heavy functions are replaced by fakes here; real runs are the
parity and smoke runs recorded in research/tracks/pipeline.md."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rrp.contracts.runconfig import RunConfig, RunIndex
from rrp.pipelines import MANIFEST, Pipeline, StageError
from rrp.pipelines import base as pbase

FLAGS_ARM = dict(zero_prev_action=True, realizer_anchor=True, realizer_drop_qd=True, probe_lv_min=None, qd_dropout=None,
                 contact_version="contact_v1")


def _rc(**kw) -> RunConfig:
    d = dict(schema_version="runconfig-1", family="arm", stage="refit", variant="semfix", seed=2, lineage="arm-semfix",
             track="t", tag="x", inputs={"representation": "runs/rep:representation.pt",
                                         "dagger": {"runs": ["runs/bc1"], "files": ["a.npz", "b.npz"]}},
             flags=FLAGS_ARM, params={"steps": 3, "seed": 2711})
    d.update(kw)
    return RunConfig.model_validate(d)


def _touch(root: Path, *rels):
    for r in rels:
        p = root / r
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")


def test_stage_registry():
    assert Pipeline("arm").stages() == ["collect", "pack", "train_rep", "probes", "train_flow", "flow_ft",
                                        "dagger_collect", "refit", "eval_r1", "eval_r2", "heldout", "edits", "train_bc",
                                        "grpo", "target_eval", "target_adapt"]
    assert "pack" not in Pipeline("legged").stages() and "refit" in Pipeline("legged").stages()
    assert Pipeline("dual").stages() == ["train_rep", "train_flow"]
    with pytest.raises(StageError, match="not implemented"):
        Pipeline("dual").spec("refit")


def test_arm_refit_calls_refit_realizer(tmp_path, monkeypatch):
    import rrp.training.latent_train as lt
    seen = {}

    def fake(cfg, out):
        seen.update(cfg=cfg, out=out, cwd=Path.cwd())
        out.mkdir(parents=True, exist_ok=True)
        (out / "representation.pt").write_bytes(b"w")
        return {"steps": 3, "latent_space_version": "ls-x", "realizer_compat_version": "rz-y", "interrupted": False}
    monkeypatch.setattr(lt, "refit_realizer", fake)
    rc = _rc()
    _touch(tmp_path, "artifacts/runs/rep/representation.pt", "artifacts/runs/bc1/a.npz", "artifacts/runs/bc1/b.npz")
    body = Pipeline("arm").run(rc, root=tmp_path, index=RunIndex())
    assert seen["cfg"] == rc.to_native(RunIndex()) and seen["cwd"] == tmp_path.resolve()
    assert seen["cfg"]["dagger"] == ["artifacts/runs/bc1/a.npz", "artifacts/runs/bc1/b.npz"]
    assert seen["cfg"]["realizer_drop_qd"] is True and seen["cfg"]["out_dir"] == "artifacts/runs/t/arm-semfix/refit-x_s2"
    m = json.loads((tmp_path / rc.out / MANIFEST).read_text())
    assert m["config_hash"] == rc.config_hash() and m["manifest_hash"]
    assert m["provenance"]["source"].startswith("learned:") and m["provenance"]["flags"]["zero_prev_action"] is True
    assert m["provenance"]["versions"] == {"latent_space_version": "ls-x", "realizer_compat_version": "rz-y"}
    assert m["inputs"]["dagger"][0]["digest"] and m["outputs"]["representation"]["digest"]


def test_missing_input_refused(tmp_path):
    with pytest.raises(StageError, match="missing inputs"):
        Pipeline("arm").run(_rc(), root=tmp_path, index=RunIndex())


def test_arm_eval_r2_runs_ladder_with_deployment_input(tmp_path, monkeypatch):
    calls = []

    def fake_run(self, argv, *, env=None, log_to=None):
        calls.append(argv)
        a = dict(zip(argv[1::2], argv[2::2]))
        out = self.root / argv[argv.index("--out") + 1]
        out.mkdir(parents=True, exist_ok=True)
        name = f"generated_{argv[argv.index('--tag') + 1]}"
        (out / f"{name}.jsonl").write_text(json.dumps({"source": "learned(system-i flow)"}) + "\n")
        (out / f"{name}.summary.json").write_text(json.dumps(dict(n=30, success=21, rate=0.7, wilson95=[0.5, 0.8])))
    monkeypatch.setattr(pbase.StageContext, "run", fake_run)
    rc = _rc(stage="eval_r2", tag="final", inputs={"flow": "runs/f:policy.pt", "representation": "runs/r:representation.pt"},
             flags=dict(FLAGS_ARM, realizer_anchor=None, realizer_drop_qd=None), params={},
             options={"tag": "final", "robots": ["panda_pg2"], "seed_starts": [3000000, 3000100]})
    _touch(tmp_path, "artifacts/runs/f/policy.pt", "artifacts/runs/r/representation.pt")
    body = Pipeline("arm").run(rc, root=tmp_path, index=RunIndex())
    assert len(calls) == 2 and calls[0][0] == "scripts/ladder.py"
    assert calls[0][calls[0].index("--prev-action") + 1] == "zero" and "--flow" in calls[0]
    assert calls[1][calls[1].index("--tag") + 1] == "zero_final_s3000100"
    assert body["metrics"]["_total"] == {"success": 42, "n": 60}
    assert body["metrics"]["panda_pg2/s3000000"]["source_kinds"] == {"learned": 1}
    with pytest.raises(StageError, match="target bodies"):
        Pipeline("arm").run(rc.model_copy(update={"options": dict(rc.options, robots=["xarm7_pg2"])}), root=tmp_path,
                            index=RunIndex())


def test_arm_heldout_refuses_training_bodies(tmp_path):
    rc = _rc(stage="heldout", tag="h", inputs={"flow": "runs/f:policy.pt", "representation": "runs/r:representation.pt"},
             flags=dict(FLAGS_ARM, realizer_anchor=None, realizer_drop_qd=None), params={},
             options={"tag": "h", "robots": ["panda_pg2"]})
    _touch(tmp_path, "artifacts/runs/f/policy.pt", "artifacts/runs/r/representation.pt")
    with pytest.raises(StageError, match="overlap"):
        Pipeline("arm").run(rc, root=tmp_path, index=RunIndex())


def test_legged_train_rep_calls_trainer_and_checks_contact(tmp_path, monkeypatch):
    import rrp.training.legged_latent_train as llt
    seen = {}

    def fake(cfg, out):
        seen.update(cfg=cfg, out=out)
        (out / "representation.pt").write_bytes(b"w")
        return {"steps": 1}
    monkeypatch.setattr(llt, "train_rep", fake)
    rc = RunConfig.model_validate(dict(
        schema_version="runconfig-1", family="legged", stage="train_rep", variant="semfix", seed=1, lineage="legged-go2",
        track="t", inputs={"data": "datasets/legged_x"},
        flags=dict(zero_prev_action=None, realizer_anchor=None, realizer_drop_qd=None, probe_lv_min=-4.0,
                   qd_dropout=0.5, contact_version="contact_v1"),
        params={"bodies": ["go2"], "latent": {"semantic_weight": 1.0}}))
    (tmp_path / "artifacts/datasets/legged_x").mkdir(parents=True)
    Pipeline("legged").run(rc, root=tmp_path, index=RunIndex())
    assert seen["cfg"]["latent"] == {"semantic_weight": 1.0, "probe_lv_min": -4.0, "qd_dropout": 0.5}
    assert seen["cfg"]["data"] == "artifacts/datasets/legged_x"
    # a dataset recorded with another contact version is refused
    from rrp.data.manifest import write_manifest
    from rrp.contracts.provenance import make_provenance, PhysicsProvenance
    phys = PhysicsProvenance(mujoco_version="3", timestep=0.002, integrator="euler", cone="elliptic", impratio=100.0,
                             solver="newton", iterations=100, ls_iterations=50, noslip_iterations=0,
                             contact_version="contact_v2")
    write_manifest(tmp_path / "artifacts/datasets/legged_x", "x", [], provenance=make_provenance("scripted_teacher", physics=phys),
                   filename="s0-1.manifest.json")
    with pytest.raises(StageError, match="contact version"):
        Pipeline("legged").run(rc, root=tmp_path, index=RunIndex())


def test_legged_contact_v2_refuses_unversioned_data_and_mismatched_rows(tmp_path):
    from rrp.pipelines.legged import check_contact_version, check_rows_contact
    from rrp.pipelines.base import StageContext
    rc = RunConfig.model_validate(dict(
        schema_version="runconfig-1", family="legged", stage="eval_r2", variant="semfix", seed=1, lineage="l",
        track="t", inputs={"flow": "runs/x:policy.pt"},
        flags=dict(zero_prev_action=None, realizer_anchor=None, realizer_drop_qd=None, probe_lv_min=None,
                   qd_dropout=None, contact_version="contact_v2"), options={"body": "anymal_c"}))
    ctx = StageContext(rc=rc, index=RunIndex(), root=tmp_path)
    (tmp_path / "d").mkdir()
    with pytest.raises(StageError, match="legacy data = contact_v1"):
        check_contact_version(ctx, "d")
    ok = dict(seed=1, contact_version="contact_v2",
              checkpoint_provenance={"flow": {"physics": {"contact_version": "contact_v2"}}, "representation": {}})
    assert check_rows_contact(ctx, [ok], "w") == {"contact_versions": ["contact_v2"]}
    rc2 = rc.model_copy(update={"options": {"body": "anymal_c", "tracker_sha256": "abc"}})
    with pytest.raises(StageError, match="tracker sha"):
        check_rows_contact(StageContext(rc=rc2, index=RunIndex(), root=tmp_path), [dict(ok, tracker_sha256="def")], "w")
    assert check_rows_contact(StageContext(rc=rc2, index=RunIndex(), root=tmp_path), [dict(ok, tracker_sha256="abc")], "w")
    with pytest.raises(StageError, match="scene contact_v1"):
        check_rows_contact(ctx, [dict(ok, contact_version="contact_v1")], "w")
    with pytest.raises(StageError, match="trained on contact_v1"):
        check_rows_contact(ctx, [dict(ok, checkpoint_provenance={"flow": {"physics": {"contact_version": "contact_v1"}}})], "w")
    assert __import__("rrp.pipelines.legged", fromlist=["physics_env"]).physics_env(ctx)["RRP_CONTACT_MODEL"] == "contact_v2"


def test_legged_eval_refuses_checkpoint_trained_on_other_contact(tmp_path):
    import torch
    from rrp.pipelines.legged import check_checkpoints_contact
    from rrp.pipelines.base import StageContext
    d = tmp_path / "artifacts/runs/x"
    d.mkdir(parents=True)
    torch.save({"_provenance": {"physics": {"contact_version": "contact_v2"}}, "cfg": {}}, d / "rep2.pt")
    torch.save({"_provenance": {"physics": None}, "cfg": {"representation": str(d / "rep2.pt")}}, d / "flow2.pt")
    torch.save({"cfg": {}}, d / "legacy.pt")

    def ctx(flow, cv="contact_v2"):
        rc = RunConfig.model_validate(dict(
            schema_version="runconfig-1", family="legged", stage="eval_r2", variant="semfix", seed=1, lineage="l",
            track="t", inputs={"flow": f"runs/x:{flow}"},
            flags=dict(zero_prev_action=None, realizer_anchor=None, realizer_drop_qd=None, probe_lv_min=None,
                       qd_dropout=None, contact_version=cv), options={"body": "anymal_c"}))
        return StageContext(rc=rc, index=RunIndex(), root=tmp_path)
    assert check_checkpoints_contact(ctx("flow2.pt")) == {"flow": "contact_v2"}     # inherited from its rep
    with pytest.raises(StageError, match="trained on contact_v1"):
        check_checkpoints_contact(ctx("legacy.pt"))
    with pytest.raises(StageError, match="trained on contact_v2"):
        check_checkpoints_contact(ctx("flow2.pt", "contact_v1"))


def test_legged_dataset_gate_d112():
    from rrp.pipelines.legged import dataset_gate
    ep = lambda sr, sig=0.1, st="success": dict(sigma=sig, status=st, motion=dict(slip_ratio=sr))
    g = dataset_gate([ep(0.05)] * 95 + [ep(0.3)] * 5)
    assert g["slip_ok"] and g["passed"] and g["slip_lt_0p15_frac"] == 0.95
    assert not dataset_gate([ep(0.05)] * 94 + [ep(0.3)] * 6)["passed"]
    assert not dataset_gate([ep(0.05)] * 99 + [ep(0.05, 0.0, "fell")])["passed"]        # a fall at noise 0
    assert dataset_gate([ep(0.05)] * 99 + [ep(0.05, 0.3, "fell")])["passed"]            # falls under noise allowed
    assert dataset_gate([dict(sigma=0.0, status="success")])["passed"] is None         # unmeasured
