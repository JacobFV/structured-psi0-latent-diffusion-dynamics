"""Pipeline / recipe contract (docs/architecture.md 14.2, unit F3): open stage registry, the RunConfig as the one channel
to child processes (kinfeat / feat.base_axes), version pins + the stale rule of run-dag adoption, code revision (untracked
files count), manifest factor versions read by the room. Everything host-light: tiny subprocesses, no simulator."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from rrp.core.runconfig import PIPELINE_STAGES, RunConfig, RunConfigError, register_family, unregister_family
from rrp.harness import dag as D
from rrp.harness.pipelines import base as B
from rrp.harness.yamlmini import loads

ROOT = Path(__file__).resolve().parents[2]
NOFLAGS = dict(zero_prev_action=None, realizer_anchor=None, realizer_drop_qd=None, probe_lv_min=None, qd_dropout=None,
               contact_version=None)
PROBES = [{"name": "probe.arm.visible", "weight": 1.0, "params": {"lv_min": -4.0}},
          {"name": "probe.arm.looking_at", "weight": 1.0, "params": {"lv_min": -4.0}}]


def _rc(stage="eval_r2", family="pointer", **kw) -> RunConfig:
    d = dict(schema_version="runconfig-1", family=family, stage=stage, variant="na", seed=0, lineage="f3", track="t",
             flags=NOFLAGS, params={}, options={})
    d.update(kw)
    return RunConfig.model_validate(d)


@pytest.fixture
def fam0():
    """Nothing registered yet: the test registers its family and stage through the open registry itself."""
    yield "f3fam"
    B.unregister_stages("f3fam")
    unregister_family("f3fam")


@pytest.fixture
def fam():
    """The real (pointer, eval_r2) pair with a do-nothing body that tests swap with `_register_probe`. Children rebuild
    the RunConfig in a fresh process, so the family must be one every process knows: a built-in."""
    B._load_families()
    key = ("pointer", "eval_r2")
    saved = B._REGISTRY[key]
    _register_probe(lambda ctx: dict(outputs={}, metrics={}))
    yield "pointer"
    B._REGISTRY[key] = saved


def _register_probe(fn):
    B._REGISTRY[("pointer", "eval_r2")] = B.StageSpec("pointer", "eval_r2", fn, "scripted_teacher", "test probe")


# --------------------------------------------------------------------------------------------- open registry
def test_register_stage_declares_family_and_stage(fam0):
    assert "f3_probe" not in PIPELINE_STAGES
    F = dict(family="f3fam", stage="f3_probe")
    with pytest.raises(Exception, match="f3_probe"):
        _rc(**F)
    B.register_stage("f3fam", "f3_probe", lambda ctx: dict(outputs={}, metrics={"ok": 1}), source="scripted_teacher")
    assert "f3_probe" in PIPELINE_STAGES and B.Pipeline("f3fam").stages() == ["f3_probe"]
    assert _rc(**F).stage == "f3_probe"
    with pytest.raises(ValueError, match="no stage"):
        _rc(family="arm", stage="f3_probe")        # a stage belongs to the families that declared it


def test_register_stage_flags_and_existing_pairs(fam0):
    B.register_stage("f3fam", "f3_probe", lambda ctx: {}, source="learned", flags=("contact_version",))
    with pytest.raises(ValueError, match="must be stated"):
        _rc(family="f3fam", stage="f3_probe")
    assert _rc(family="f3fam", stage="f3_probe", flags=dict(NOFLAGS, contact_version="c1")).flags.contact_version == "c1"
    with pytest.raises(RunConfigError, match="already declared"):
        B.register_stage("f3fam", "f3_probe", lambda ctx: {}, source="learned", flags=())
    B.register_stage("arm", "collect", B._REGISTRY[("arm", "collect")].fn, source="scripted_teacher")   # core pair, flags kept
    assert "contact_version" in __import__("rrp.core.runconfig", fromlist=["FLAG_SPEC"]).FLAG_SPEC[("arm", "collect")]


def test_stage_names_are_pruned_when_the_family_goes(fam0):
    B.register_stage("f3fam", "f3_probe", lambda ctx: {}, source="learned")
    B.unregister_stages("f3fam")
    unregister_family("f3fam")
    assert "f3_probe" not in PIPELINE_STAGES


def test_pipeline_modules_register_their_own_families():
    B._load_families()
    for f in ("arm", "dual", "legged", "psi0", "pointer"):
        assert B.Pipeline(f).stages()


def test_dag_names_a_missing_stage_cleanly():
    with pytest.raises(D.DagError, match="no stage"):
        D.plan_dag(loads("name: x\nfamily: pointer\ntrack: t\nlineage: l\nmatrix: {seed: [1]}\nnodes:\n  a: {stage: eval_transfer}\n"))


# --------------------------------------------------------------------------- the RunConfig reaches children
def _child_script(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "child.py"
    p.write_text(body)
    return p


PROBE_CHILD = '''
import json, os, sys
from rrp.policies.features import kinfeat
json.dump(dict(resolved=kinfeat.resolved(), env="RRP_KINFEAT" in os.environ), open(sys.argv[1], "w"))
'''


def _run_stage(tmp_path, rc, fn):
    _register_probe(fn)
    return B.Pipeline("pointer").run(rc, root=tmp_path, index=B.RunIndex())


def test_child_of_a_kinfeat_stage_sees_base_axes(tmp_path, fam):
    from rrp.policies.features import kinfeat
    script, out = _child_script(tmp_path, PROBE_CHILD), tmp_path / "seen.json"

    def stage(ctx):
        assert "RRP_KINFEAT" not in ctx.env()                 # env carries resources only
        ctx.run([str(script), str(out)])
        return dict(outputs={}, metrics={})
    _run_stage(tmp_path, _rc(options={"kinfeat": "v1"}), stage)
    assert json.loads(out.read_text()) == dict(resolved=True, env=False)
    assert kinfeat.resolved() is False                        # parent restored after the stage
    out.unlink()
    _run_stage(tmp_path, _rc(), stage)                        # no kinfeat: the child stays False
    assert json.loads(out.read_text()) == dict(resolved=False, env=False)
    assert (tmp_path / _rc().out / B.CONTEXT_FILE).exists()   # the channel is the rendered RunConfig


def test_child_module_target_and_exit_code(tmp_path, fam):
    def stage(ctx):
        ctx.run(["-m", "json.tool", "--help"])                # `-m module` targets run like python -m
        with pytest.raises(B.StageError, match="exit 3"):
            ctx.run(["-c", "raise SystemExit(3)"]) if False else ctx.run([str(_child_script(tmp_path, "raise SystemExit(3)"))])
        return dict(outputs={}, metrics={})
    _run_stage(tmp_path, _rc(), stage)


def test_factor_item_configures_the_featurizer_and_conflicts_are_refused(tmp_path, fam):
    script, out = _child_script(tmp_path, PROBE_CHILD), tmp_path / "seen.json"
    _run_stage(tmp_path, _rc(params={"factors": [{"name": "feat.base_axes"}]}),
               lambda ctx: (ctx.run([str(script), str(out)]), dict(outputs={}, metrics={}))[1])
    assert json.loads(out.read_text())["resolved"] is True
    with pytest.raises(B.StageError, match="stated twice"):
        _run_stage(tmp_path, _rc(options={"kinfeat": "v1"}, params={"factors": [{"name": "feat.base_axes", "control": "off"}]}),
                   lambda ctx: dict(outputs={}, metrics={}))


LOAD_CHILD = '''
import sys
from pathlib import Path
from rrp.policies.nets.checkpoint import load_checkpoint
load_checkpoint(Path(sys.argv[1]))
'''


@pytest.fixture
def kinfeat_checkpoint(tmp_path):
    import torch
    from rrp.policies.features import kinfeat
    from rrp.policies.nets.checkpoint import save_checkpoint
    prev = kinfeat.set_base_axes(True)
    try:
        save_checkpoint(tmp_path / "m.pt", model=torch.nn.Linear(2, 2), step=1, versions={}, config={})
    finally:
        kinfeat.set_base_axes(prev)
    return tmp_path / "m.pt"


def test_lineage_child_loads_its_kinfeat_checkpoint_and_a_mismatch_still_raises(tmp_path, fam, kinfeat_checkpoint):
    load = _child_script(tmp_path, LOAD_CHILD)
    stage = lambda ctx: (ctx.run([str(load), str(kinfeat_checkpoint)]), dict(outputs={}, metrics={}))[1]   # noqa: E731
    _run_stage(tmp_path, _rc(options={"kinfeat": "v1"}), stage)               # the lineage's child loads it
    with pytest.raises(B.StageError, match="exit 1"):                          # a non-kinfeat stage must not
        _run_stage(tmp_path, _rc(), stage)
    r = subprocess.run([sys.executable, str(load), str(kinfeat_checkpoint)], cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                       capture_output=True, text=True)                          # negative control: no context, no channel
    assert r.returncode != 0 and "kinfeat" in r.stderr


# --------------------------------------------------------------------------------- versions, pins, manifest
def test_manifest_carries_factor_versions_pins_and_provenance(tmp_path, fam):
    rc = _rc(params={"latent": {"factors": PROBES}})
    body = _run_stage(tmp_path, rc, lambda ctx: dict(outputs={}, metrics={}))
    pins = B.stage_versions(rc)
    assert body["pins"] == pins and pins["pipeline"] == str(B.PIPELINE_VERSION)
    assert pins["factors"].startswith("fx-") and pins["catalog"].startswith("cat-")
    m = json.loads((tmp_path / rc.out / B.MANIFEST).read_text())
    assert m["provenance"]["versions"]["factors"] == pins["factors"]
    assert [f["name"] for f in m["factors"]] == ["probe.arm.visible", "probe.arm.looking_at"] and m["factors"][0]["version"]
    assert m["provenance"]["code"]["src_tree"]                                 # revision incl. tree hashes
    assert B.stage_versions(_rc(options={"kinfeat": "v1"}))["factors"] == "kinfeat_v1"
    assert B.stage_versions(_rc(params={"latent": {"factors": PROBES}}, options={"kinfeat": "v1"}))["factors"] == \
        pins["factors"] + "+kinfeat_v1"                                        # the checkpoint's combined string
    assert B.stage_versions(_rc())["factors"] == ""


def test_pins_follow_meaning_not_weights(monkeypatch):
    from rrp.policies.relations import base as R
    a = B.stage_versions(_rc(params={"latent": {"factors": PROBES}}))
    heavier = [dict(PROBES[0], weight=3.0), PROBES[1]]
    assert B.stage_versions(_rc(params={"latent": {"factors": heavier}})) == a          # weight is not structure
    shifted = [dict(PROBES[0], params={"lv_min": -8.0}), PROBES[1]]
    assert B.stage_versions(_rc(params={"latent": {"factors": shifted}}))["factors"] != a["factors"]
    d = R.FACTORS["probe.arm.visible"]
    monkeypatch.setitem(R.FACTORS, "probe.arm.visible", replace(d, doc="reworded", op=d.op, version=d.version + "x"))
    b = B.stage_versions(_rc(params={"latent": {"factors": PROBES}}))
    assert b["catalog"] != a["catalog"] and b["factors"] != a["factors"]
    monkeypatch.setattr(B, "PIPELINE_VERSION", B.PIPELINE_VERSION + 1)
    assert B.stage_versions(_rc())["pipeline"] != a["pipeline"]


def test_every_recipe_node_has_resolvable_pins():
    bad = []
    for p in sorted((ROOT / "recipes").rglob("*.yaml")):
        for nid, n in D.plan_dag(D.load_dag(p), source="t").nodes.items():
            try:
                B.stage_versions(n.rc)
            except Exception as e:  # noqa: BLE001
                bad.append(f"{p.name}:{nid}: {e}")
    assert not bad, bad[:5]


def test_room_reads_pipeline_manifests(tmp_path, fam):
    from rrp.viz.export.relations import _factor_runs
    rc = _rc(params={"latent": {"factors": PROBES}})
    _run_stage(tmp_path, rc, lambda ctx: dict(outputs={}, metrics={}))
    cfg = SimpleNamespace(repo=tmp_path, main_checkout=None, rrp_data=None, now=time.time(), out=tmp_path / "out", live=False)
    rows = _factor_runs(cfg)
    assert len(rows) == 1 and rows[0]["file"].endswith("pipeline_manifest.json")
    assert rows[0]["factors"] == B.stage_versions(rc)["factors"] and rows[0]["stage"] == "eval_r2"
    assert [f["name"] for f in rows[0]["factor_set"]] == ["probe.arm.visible", "probe.arm.looking_at"]


# ------------------------------------------------------------------------------------ code revision counts untracked
def _git(root, *a):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True, capture_output=True)


def _repo(tmp_path):
    (tmp_path / "src/pkg").mkdir(parents=True)
    (tmp_path / "src/pkg/a.py").write_text("x = 1\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "c")
    return tmp_path


def test_untracked_files_count_as_dirty_and_move_the_src_tree(tmp_path):
    from rrp.core.provenance import code_provenance
    r = _repo(tmp_path)
    clean = code_provenance(r, fresh=True)
    assert clean.dirty is False and clean.src_tree and clean.tree
    (r / "docs").mkdir()
    (r / "docs/note.md").write_text("n")                       # untracked, outside src: dirty, same code
    c2 = code_provenance(r, fresh=True)
    assert c2.dirty is True and c2.src_tree == clean.src_tree
    (r / "src/pkg/new.py").write_text("y = 2\n")               # untracked module in src: different code
    c3 = code_provenance(r, fresh=True)
    assert c3.dirty is True and c3.src_tree != clean.src_tree
    (r / "src/pkg/a.py").write_text("x = 2\n")                 # tracked edit
    assert code_provenance(r, fresh=True).src_tree != c3.src_tree
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "d")
    c4 = code_provenance(r, fresh=True)
    assert c4.dirty is False and c4.tree != clean.tree and c4.git_sha != clean.git_sha     # all committed: clean again


def test_peer_sync_revision_matches_code_provenance(tmp_path):
    from rrp.core.provenance import code_provenance
    r = _repo(tmp_path)
    (r / "ops/bin").mkdir(parents=True)
    (r / "ops/bin/peer_sync.sh").write_text((ROOT / "ops/bin/peer_sync.sh").read_text())
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "ops")
    rev = lambda: json.loads(subprocess.run(["bash", str(r / "ops/bin/peer_sync.sh"), "revision"], capture_output=True,   # noqa: E731
                                            text=True, check=True).stdout)
    a, cp = rev(), code_provenance(r, fresh=True)
    assert (a["git_sha"], a["dirty"], a["tree"], a["src_tree"]) == (cp.git_sha, False, cp.tree, cp.src_tree)
    (r / "src/pkg/untracked.py").write_text("z = 3\n")
    b, cp = rev(), code_provenance(r, fresh=True)
    assert b["dirty"] is True and b["src_tree"] == cp.src_tree != a["src_tree"]     # untracked counted, same hash as python


def test_synced_copy_without_git_reads_revision_file(tmp_path):
    from rrp.core.provenance import code_provenance
    (tmp_path / ".rrp_revision").write_text(json.dumps(dict(git_sha="abc", dirty=True, tree="t1", src_tree="s1")))
    c = code_provenance(tmp_path, fresh=True)
    assert (c.git_sha, c.dirty, c.tree, c.src_tree) == ("abc", True, "t1", "s1")


# ------------------------------------------------------------------ adoption, stale, ledger revision (run-dag)
TOY = """
name: toy
family: pointer
track: t
lineage: toy
matrix: {seed: [1]}
defaults: {placement: host, resources: {cpu: 1, mem: 1G}}
nodes:
  a:
    stage: train_rep
    config: {params: {w_sem: 1.0, latent: {factors: FACTORS}}}
    tag: a
  b:
    stage: train_flow
    tag: b
    config: {inputs: {representation: '@a:rep.pt'}, params: {w_sem: 1.0}}
""".replace("FACTORS", json.dumps(PROBES))


class Runner:
    """Writes the manifest a real stage would (pins + code), or nothing when `write` is off."""
    def __init__(self, root, tree="T1"):
        self.root, self.tree, self.launched, self.jobs = root, tree, [], {}

    def manifest_for(self, node, tree=None, pins=None):
        p = self.root / node.rc.out
        p.mkdir(parents=True, exist_ok=True)
        (p / "pipeline_manifest.json").write_text(json.dumps(dict(
            config_hash=node.rc.config_hash(), metrics={}, pins=pins or B.stage_versions(node.rc),
            provenance=dict(code=dict(git_sha="g", dirty=False, tree="tr", src_tree=tree or self.tree)))))

    def launch(self, node):
        self.launched.append(node.id)
        self.manifest_for(node)
        self.jobs[f"L{len(self.launched)}"] = 0
        return dict(lease_id=f"L{len(self.launched)}", log="/x", placement="host")

    def poll(self, h):
        return self.jobs[h["lease_id"]]

    def manifest(self, node):
        p = self.root / node.rc.out / "pipeline_manifest.json"
        return json.loads(p.read_text()) if p.exists() else None


def _code(tree):
    return lambda: dict(git_sha="g", dirty=False, tree="tr", src_tree=tree)


def _ex(tmp, runner, tree="T1", **kw):
    return D.Executor(D.plan_dag(loads(TOY)), D.Ledger(tmp / "ledger.json"), runner, poll_s=0, sleep=lambda s: None,
                      log=lambda m: None, code_now=_code(tree), pins=B.stage_versions, **kw)


def _ledger(tmp):
    return json.loads((tmp / "ledger.json").read_text())["nodes"]


def test_ledger_records_revision_pins_and_code_per_attempt(tmp_path):
    r = Runner(tmp_path)
    assert _ex(tmp_path, r).run()["completed"] == 2
    e = _ledger(tmp_path)["a@s1"]
    assert e["attempts"][0]["revision"] == dict(git_sha="g", dirty=False, tree="tr", src_tree="T1")
    assert e["pins"]["pipeline"] == str(B.PIPELINE_VERSION) and e["code"]["src_tree"] == "T1"


def test_same_code_completed_nodes_stay_completed(tmp_path):
    r = Runner(tmp_path)
    _ex(tmp_path, r).run()
    r2 = Runner(tmp_path)
    assert _ex(tmp_path, r2).run() == dict(completed=2, failed=0, blocked=0, planned=0, running=0) and r2.launched == []


def test_other_code_makes_completed_nodes_stale_until_adopted(tmp_path):
    _ex(tmp_path, Runner(tmp_path)).run()
    r2 = Runner(tmp_path)
    s = _ex(tmp_path, r2, tree="T2").run()
    assert s["stale"] == 2 and s["completed"] == 0 and r2.launched == []           # reported, not silently adopted
    e = _ledger(tmp_path)["a@s1"]
    assert e["state"] == "stale" and "T1" in e["stale_reason"] and "T2" in e["stale_reason"]
    s = _ex(tmp_path, r2, tree="T2", adopt_stale=True).run()
    assert s["completed"] == 2 and "stale" not in s and r2.launched == []
    assert _ledger(tmp_path)["a@s1"]["adopted_trees"] == ["T2"]
    assert _ex(tmp_path, r2, tree="T2").run()["completed"] == 2                    # the adoption is remembered
    assert _ex(tmp_path, r2, tree="T3").run()["stale"] == 2                        # a third revision is not covered


def test_adopting_existing_outputs_needs_equal_pins_and_same_code(tmp_path):
    plan = D.plan_dag(loads(TOY))
    r = Runner(tmp_path)
    for n in plan.nodes.values():
        r.manifest_for(n, tree="T1")                                               # outputs of a hand-run
    assert _ex(tmp_path, r).run()["completed"] == 2 and r.launched == []           # same pins + code: adopted, no lease
    assert _ledger(tmp_path)["a@s1"]["adopted"] is True
    (tmp_path / "ledger.json").unlink()
    s = _ex(tmp_path, r, tree="T2").run()
    assert s["stale"] == 1 and s["blocked"] == 1 and r.launched == []              # b waits on a stale a
    assert _ledger(tmp_path)["b@s1"]["blocked_by"] == "stale"
    s = _ex(tmp_path, r, tree="T2", adopt_stale=True).run()
    assert s["completed"] == 2 and r.launched == []                                # the blocked node is re-planned


def test_other_pipeline_version_or_factors_are_not_adopted_and_rerun(tmp_path, monkeypatch):
    plan = D.plan_dag(loads(TOY))
    r = Runner(tmp_path)
    for n in plan.nodes.values():
        r.manifest_for(n)
    monkeypatch.setattr(B, "PIPELINE_VERSION", B.PIPELINE_VERSION + 1)
    assert _ex(tmp_path, r).run()["completed"] == 2 and r.launched == ["a@s1", "b@s1"]     # rerun, not adopted
    (tmp_path / "ledger.json").unlink()
    old = Runner(tmp_path)
    for n in plan.nodes.values():
        old.manifest_for(n, pins=dict(B.stage_versions(n.rc), factors="fx-other"))
    assert _ex(tmp_path, old).run()["completed"] == 2 and old.launched == ["a@s1", "b@s1"]


def test_completed_ledger_node_under_other_pins_needs_reset(tmp_path, monkeypatch):
    _ex(tmp_path, Runner(tmp_path)).run()
    monkeypatch.setattr(B, "PIPELINE_VERSION", B.PIPELINE_VERSION + 1)
    with pytest.raises(D.DagError, match="--reset"):
        _ex(tmp_path, Runner(tmp_path)).run()


def test_ledger_without_pins_is_stale_not_silently_trusted(tmp_path):
    _ex(tmp_path, Runner(tmp_path)).run()
    led = json.loads((tmp_path / "ledger.json").read_text())
    for e in led["nodes"].values():
        e.pop("pins"), e.pop("code")
    (tmp_path / "ledger.json").write_text(json.dumps(led))
    assert _ex(tmp_path, Runner(tmp_path)).run()["stale"] == 2


def test_run_dag_cli_has_adopt_stale():
    import argparse
    from rrp.cli.dag import register
    ap = argparse.ArgumentParser()
    register(ap.add_subparsers())
    assert ap.parse_args(["run-dag", "x", "--adopt-stale"]).adopt_stale is True
    assert ap.parse_args(["run-dag", "x"]).adopt_stale is False


def test_recipe_goldens_are_untouched_by_the_contract():
    from rrp.harness.dag import load_dag, plan_dag
    p = plan_dag(load_dag(ROOT / "recipes/armdiv/arm_lineage_v7div_kinfeat.yaml"))
    assert all(n.rc.options.get("kinfeat") == "v1" for n in p.nodes.values())       # still a plain option in the hash
