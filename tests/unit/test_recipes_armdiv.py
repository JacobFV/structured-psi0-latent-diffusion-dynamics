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


def _dry_run_recipes():
    return sorted([*(ROOT / "recipes/armdiv").glob("*.yaml"), *(ROOT / "recipes/templates").glob("arm_*.yaml")])


@pytest.mark.parametrize("path", _dry_run_recipes(), ids=lambda p: p.stem)
def test_every_arm_recipe_and_template_dry_runs(path):
    """`rrp run-dag <recipe> --dry-run` (readiness AR): the plan AND its input-path resolution (a placeholder like
    `packed/SET_IN_INSTANCE` resolves; a bare `SET_IN_INSTANCE` is not a run id and used to crash arm_bc)."""
    from rrp.harness.dag import format_plan, load_dag, plan_dag
    out = format_plan(plan_dag(load_dag(path), source=str(path)))
    assert out.startswith("DAG ")


V6 = {   # sha256 of the eight v6 checkpoints (peer runs/armv6/arm6-<variant>/, taken 2026-09-30, D-146 round-3 addendum)
    ("semfix", 1): ("b4e9dff7384c23364c969646573deceaa0109237422b4aa48649831aa797a860",
                    "d2360bb4e1e2af45df9192d955d38aff02fd0cddef268a5f3a1a0726194389a7"),
    ("semfix", 2): ("854ad478991a8c4b4f025fbcf954cd927f69f9167d0d7a20feb93b7f82cd1d9f",
                    "f37b2c794f43d156a83d6085119191cde064e671280708a623c150d908dd5369"),
    ("nosem", 1): ("5288d3a108afbc9f5828dfed5a3f9a3bf01bca72bcbb869de0924983196ca960",
                   "c6f726ba3367a4b94c975bb7e80a71cb00ba5a14f3dba9812ce03144e1ffaee7"),
    ("nosem", 2): ("26cef27b4abc3cf80de183682e1c1c798c097f583a94ca15932a8c75e185ca0e",
                   "66dd44bb77df2055a2802a528b8e195963bba8ec307ce50b8bab3d5c9e7da22c"),
}


def test_v6ref_cells_are_pinned_on_exactly_the_v6_files_with_their_recorded_hashes(tmp_path):
    plan = _plan("arm_targets_v6ref")
    assert {n.rc.options["target"] for n in plan.nodes.values() if n.rc.stage == "target_adapt"} == {"gen3_pg2", "rizon4_tf3"}
    for n in plan.nodes.values():
        loads = {k for k in ("flow", "representation") if str(n.rc.inputs.get(k, "")).startswith("runs/armv6/")}
        assert set((n.rc.options.get("pin_sha256") or {})) == loads, n.name
        if n.rc.stage.startswith("target_"):
            assert n.rc.options["protocol"] == "recipes/presets/eval-armdiv_v1.json"
    for (variant, seed), (f, r) in V6.items():                     # each cell's zs node carries its own (variant, seed) hashes
        pins = plan.nodes[f"zs@{variant}.s{seed}.gen3_pg2"].rc.options["pin_sha256"]
        assert pins == {"flow": f, "representation": r}
    assert len({h for fr in V6.values() for h in fr}) == 8
    assert parm.pending_pins(plan.nodes) == []                     # nothing in the v6 reference recipe is a placeholder
    # a recorded pin is checked against the file's bytes (a different file at the pinned path refuses the node)
    ck = tmp_path / "artifacts/runs/armv6/arm6-nosem/flow_ft-gdag2h_s2/policy.pt"
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"v6")
    with pytest.raises(StageError, match="sha256 mismatch"):
        parm._verify_pins(_ctx(tmp_path, {"flow": "runs/armv6/arm6-nosem/flow_ft-gdag2h_s2:policy.pt"},
                               {"pin_sha256": {"flow": V6[("nosem", 2)][0]}}))


def test_no_pending_pins_left_after_t6_phase_a():
    """T6 phase A (D-147, 2026-10-02) filled the last two pins (BC 1702, kinfeat BC 1701): every G3/G4 recipe is pin-clean,
    and the pins are real 64-hex sha256 values, not placeholders."""
    for clean in ("arm_targets_v6ref", "arm_targets_v8div_latent", "arm_lineage_v8div", "arm_targets_v8div_bc",
                  "arm_lineage_v8div_kinfeat"):
        assert parm.pending_pins(_plan(clean).nodes) == [], clean
    pins = {n.rc.options["pin_sha256"]["bc_policy"] for n in _plan("arm_targets_v8div_bc").nodes.values()
            if (n.rc.options or {}).get("pin_sha256", {}).get("bc_policy")}
    assert "f4abaec350b760838e816134b65cbf029f8978483866134d75c3f8999d04706e" in pins


def test_pending_pins_flags_a_short_or_malformed_hash():
    class N:
        def __init__(self, o): self.rc = type("R", (), {"options": o})()
    nodes = {"a": N({"pin_sha256": {"flow": "abc123", "rep": "0" * 64, "x": None}}), "b": N({})}
    assert [(n, k) for n, k, _ in parm.pending_pins(nodes)] == [("a", "flow"), ("a", "x")]


@pytest.mark.parametrize("recipe,pending", [("arm_targets_v6ref", False), ("arm_targets_v8div_bc", False),
                                            ("arm_lineage_v8div_kinfeat", False), ("arm_targets_v8div_latent", False)])
def test_run_dag_dry_run_reports_pending_pins(recipe, pending, tmp_path, capsys):
    from rrp.cli.dag import cmd_run_dag
    import argparse
    a = argparse.Namespace(dag=str(ROOT / f"recipes/armdiv/{recipe}.yaml"), root=str(ROOT), point=None, only=None,
                           peer_repo=None, ledger=str(tmp_path / "l.json"), show_config=None, dry_run=True)
    assert cmd_run_dag(a) == 0
    out = capsys.readouterr().out
    assert ("PENDING pins:" in out) is pending
    if pending:
        assert "  PENDING bc_policy=PENDING_bcv7div" in out or "PENDING_bcv7divkf_1701" in out
    else:
        assert "PENDING" not in out
