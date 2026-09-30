"""armdiv restart (readiness A2, audit D25/D27): the held-out guard reads the checkpoint's real training bodies, the v8div
lineage pins its frozen inputs by sha256, one sealed-target constant, and the G4 protocol/recipes render."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rrp.core.runconfig import RunConfig, RunIndex
from rrp.harness.pipelines import arm as parm
from rrp.harness.pipelines import base as pbase
from rrp.harness.pipelines.base import Pipeline, StageError

ROOT = Path(__file__).resolve().parents[2]
FLAGS = dict(zero_prev_action=True, realizer_anchor=None, realizer_drop_qd=None, probe_lv_min=None, qd_dropout=None,
             contact_version="contact_v1")


def _rc(stage, inputs, options, flags=FLAGS, **kw):
    return RunConfig.model_validate(dict(
        schema_version="runconfig-1", family="arm", stage=stage, variant="semfix", seed=1, lineage="arm8div-semfix",
        track="armdiv", tag="h", inputs=inputs, flags=flags, params={}, options=options, **kw))


def _pack(tmp_path, robots):
    d = tmp_path / "artifacts/packed/p"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps(dict(robots=sorted(robots))))
    return d


def _fake_ckpt(monkeypatch, packed_dir):
    import rrp.policies.nets.checkpoint as ck
    monkeypatch.setattr(ck, "load_checkpoint", lambda p, map_location=None: {"config": {"packed_dir": str(packed_dir)}})


def _heldout(tmp_path, robots):
    (tmp_path / "artifacts/runs/r").mkdir(parents=True, exist_ok=True)
    for f in ("runs/f/policy.pt", "runs/r/representation.pt"):
        p = tmp_path / "artifacts" / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    rc = _rc("heldout", {"flow": "runs/f:policy.pt", "representation": "runs/r:representation.pt"},
             {"tag": "h", "robots": robots})
    return Pipeline("arm").run(rc, root=tmp_path, index=RunIndex())


def test_heldout_guard_uses_the_checkpoints_real_training_bodies(tmp_path, monkeypatch):
    # v8div pool bodies that are NOT in the v6 default list: the old guard let them through as "held out"
    _fake_ckpt(monkeypatch, _pack(tmp_path, ["pa2s0_pg2", "pa2s0_tf3", "parm6_tf3"]))
    assert "pa2s0_pg2" not in parm.TRAIN_BODIES
    with pytest.raises(StageError, match="overlap"):
        _heldout(tmp_path, ["pa2s0_pg2"])


def test_heldout_guard_refuses_an_unresolvable_training_set(tmp_path, monkeypatch):
    _fake_ckpt(monkeypatch, tmp_path / "artifacts/packed/missing")
    with pytest.raises(StageError, match="training bodies"):
        _heldout(tmp_path, ["parm5s_tf3"])


def test_heldout_guard_passes_a_truly_held_out_body(tmp_path, monkeypatch):
    calls = []

    def fake_run(self, argv, *, env=None, log_to=None):
        calls.append(argv)
        out = self.root / argv[argv.index("--out") + 1]
        out.mkdir(parents=True, exist_ok=True)
        name = f"generated_{argv[argv.index('--tag') + 1]}"
        (out / f"{name}.jsonl").write_text(json.dumps({"source": "learned(system-i flow)"}) + "\n")
        (out / f"{name}.summary.json").write_text(json.dumps(dict(n=30, success=15, rate=0.5, wilson95=[0.3, 0.7])))
    monkeypatch.setattr(pbase.StageContext, "run", fake_run)
    _fake_ckpt(monkeypatch, _pack(tmp_path, ["pa2s0_pg2", "parm6_tf3"]))
    _heldout(tmp_path, ["parm5s_tf3"])
    assert calls


def _ctx(tmp_path, inputs, options):
    rc = _rc("refit", inputs, options, flags=dict(FLAGS, realizer_anchor=True, realizer_drop_qd=True))
    return pbase.StageContext(rc=rc, index=RunIndex(), root=tmp_path)


def test_pin_sha256_checks_files_and_pack_meta(tmp_path):
    pack = _pack(tmp_path, ["a"])
    pol = tmp_path / "artifacts/runs/bc/policy.pt"
    pol.parent.mkdir(parents=True)
    pol.write_bytes(b"weights")
    sha = lambda b: hashlib.sha256(b).hexdigest()
    ins = {"packed_dir": "packed/p", "bc_policy": "runs/bc:policy.pt"}
    good = {"packed_dir": sha((pack / "meta.json").read_bytes()), "bc_policy": sha(b"weights")}
    parm._verify_pins(_ctx(tmp_path, ins, {"pin_sha256": good}))
    with pytest.raises(StageError, match="sha256 mismatch"):
        parm._verify_pins(_ctx(tmp_path, ins, {"pin_sha256": dict(good, bc_policy=sha(b"other"))}))
    with pytest.raises(StageError, match="placeholder"):
        parm._verify_pins(_ctx(tmp_path, ins, {"pin_sha256": dict(good, bc_policy="UNTRAINED")}))
    parm._verify_pins(_ctx(tmp_path, ins, {}))            # no pins declared: nothing to check


def test_sealed_constants_have_one_source():
    from rrp.bodies import armdiv
    from rrp.harness.eval import target_eval
    assert not hasattr(parm, "TARGET_BODIES") and not hasattr(target_eval, "SEALED_TARGETS")
    for k in ("xarm7_pg2", "xarm7_tf3", "panda_tf3", "gen3_pg2", "rizon4_tf3", "pa2s900002_pg2", "pa2s900003_tf3"):
        assert armdiv.is_sealed_target(k), k
    assert not armdiv.is_sealed_target("parm5s_tf3") and not armdiv.is_sealed_target("pa2s0_pg2")


PROTO = ROOT / "recipes/presets/eval-armdiv_v1.json"


def test_g4_protocol_targets_match_the_sealed_split():
    proto = json.loads(PROTO.read_text())
    split = json.loads((ROOT / "research/splits/armdiv_v1.json").read_text())
    want = [r for t in split["targets"].values() for r in t["robots"]]
    assert proto["targets"] == want and proto["split"] == "research/splits/armdiv_v1.json"
    assert proto["status"] == "SEALED_BEFORE_RESULTS" and proto["source_train_robots"] == split["source_train_robots"]
    assert proto["eval"]["seed_start"] == 2000000 and proto["sft_budgets"] == [0, 5, 20, 100]
    from rrp.bodies.armdiv import is_sealed_target
    assert all(is_sealed_target(t) for t in proto["targets"])
    assert not set(proto["targets"]) & set(proto["source_train_robots"])


def _plan(name):
    from rrp.harness.dag import load_dag, plan_dag
    return plan_dag(load_dag(ROOT / "recipes/armdiv" / f"{name}.yaml"), source="t")


def test_v8div_lineage_pins_its_frozen_inputs_and_has_the_v7_node_set():
    plan = _plan("arm_lineage_v8div")
    assert len(plan.nodes) == 88
    pins = {n.rc.options["pin_sha256"]["bc_policy"] for n in plan.nodes.values()
            if n.rc.stage in ("dagger_collect",)}
    assert pins == {"831470cbc314236f6b51a4f14533e118ae9b4ddf44217d70dc18c57e7d94d8cd"}
    stagea = [n for n in plan.nodes.values() if n.rc.stage == "train_rep"]
    assert stagea and all(len(n.rc.options["pin_sha256"]["packed_dir"]) == 64 for n in stagea)
    assert all(n.rc.lineage.startswith("arm8div-") for n in plan.nodes.values() if n.name != "newarms" or True)
    kf = _plan("arm_lineage_v8div_kinfeat")
    assert all(n.rc.options.get("kinfeat") == "v1" for n in kf.nodes.values())


@pytest.mark.parametrize("name,n", [("arm_targets_v8div_latent", None), ("arm_targets_v8div_bc", None)])
def test_g4_target_recipes_use_the_sealed_protocol_and_all_six_targets(name, n):
    plan = _plan(name)
    tg = {x.rc.options.get("target") or x.rc.options.get("robot") for x in plan.nodes.values()
          if x.rc.stage in ("target_eval", "target_adapt")}
    proto = json.loads(PROTO.read_text())
    assert set(proto["targets"]) <= tg
    ev = [x for x in plan.nodes.values() if x.rc.stage in ("target_eval", "target_adapt")]
    assert ev and all(x.rc.options["protocol"] == "recipes/presets/eval-armdiv_v1.json" for x in ev)
