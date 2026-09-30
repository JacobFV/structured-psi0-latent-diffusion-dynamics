"""Readiness R2 unit FS: every checkpoint loader guards the factor stamp (an unstamped checkpoint reads as its family's
default preset, a stamped mismatch raises unless `allow_factor_mismatch`), and the pipeline base carries the run context
to grandchildren (`RRP_RUN_CONTEXT`). Host-light: tiny nets, tiny subprocesses."""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from rrp.harness.pipelines import base as B
from rrp.policies.bundles import load_representation
from rrp.policies.nets.checkpoint import load_checkpoint, save_checkpoint
from rrp.policies.nets.flow import FlowPolicy, PolicyConfig
from rrp.policies.nets.probes import ReadoutProbe
from rrp.policies.nets.semantic_latent import LatentConfig, TargetEncoder
from rrp.policies.relations.base import FactorError
from rrp.policies.system0 import make_realizer

ROOT = Path(__file__).resolve().parents[2]


def _strip_stamp(path: Path) -> None:
    """A checkpoint written before the stamp existed: same weights, no `versions["factors"]`."""
    st = torch.load(path, map_location="cpu", weights_only=False)
    st["versions"].pop("factors", None)
    torch.save(st, path)


# ------------------------------------------------------------------------------------------- arm bundle
def _bundle_ckpt(tmp_path, **cfg_kw):
    from rrp.harness.train.latent_train import _bundle
    cfg = LatentConfig(width=32, heads=2, ctx_layers=1, enc_layers=1, knots=2, dz=4, horizon=4, realizer_layers=1, **cfg_kw)
    E, R = TargetEncoder(cfg), make_realizer(cfg.dz, cfg.realizer_layers, factors=cfg.realizer_factors)
    P = ReadoutProbe(cfg.dz, cfg.knots, specs=["preset:probes:arm-packet-v1"])
    lv = cfg.version()
    fields = ("width", "heads", "ctx_layers", "enc_layers", "knots", "dz", "horizon", "realizer_layers", "realizer_factors",
              "encoder_factors")
    path = tmp_path / "rep.pt"
    save_checkpoint(path, model=_bundle(E, R, P), step=1, versions=dict(latent=lv),
                    config=dict(latent={f: getattr(cfg, f) for f in fields}, probe={}),
                    extra=dict(result=dict(latent_space_version=lv)))
    return path


def test_unstamped_arm_bundle_loads_as_the_arm_preset(tmp_path):
    path = _bundle_ckpt(tmp_path)
    _strip_stamp(path)
    assert "factors" not in load_checkpoint(path)["versions"]
    _, E, R, P, _ = load_representation(path, "cpu")                      # the default (arm) structure: read as `arm`
    assert [s.name for s in R.factor_specs()] == ["route.own_assembly"]


def test_unstamped_arm_bundle_with_a_non_default_list_is_refused(tmp_path):
    path = _bundle_ckpt(tmp_path, realizer_factors=[])                     # a structure only a stamped writer can produce
    _strip_stamp(path)
    with pytest.raises(FactorError, match="unstamped"):
        load_representation(path, "cpu")
    load_representation(path, "cpu", allow_factor_mismatch=True)           # a deliberate cross-structure load


def test_stamped_arm_bundle_mismatch_raises(tmp_path):
    path = _bundle_ckpt(tmp_path, realizer_factors=[])
    st = torch.load(path, map_location="cpu", weights_only=False)
    st["config"]["latent"]["realizer_factors"] = None                      # config edited after the stamp
    torch.save(st, path)
    with pytest.raises(FactorError):
        load_representation(path, "cpu")


# ---------------------------------------------------------------------------- flow policy (latent.py / bc.py)
def _flow_ckpt(tmp_path, **pc_kw):
    pc = PolicyConfig(width=32, heads=2, ctx_layers=1, blocks=1, horizon=4, **pc_kw)
    path = tmp_path / "policy.pt"
    save_checkpoint(path, model=FlowPolicy(pc), step=1, versions={}, config=dict(policy=dataclasses.asdict(pc)))
    return path, pc


def test_bc_loader_reads_unstamped_as_arm_and_refuses_a_stamped_mismatch(tmp_path):
    from rrp.policies.bc import LearnedPolicy
    path, pc = _flow_ckpt(tmp_path)
    LearnedPolicy.from_checkpoint(path)                                     # stamped, same list
    _strip_stamp(path)
    LearnedPolicy.from_checkpoint(path)                                     # unstamped: the arm preset
    (tmp_path / "b").mkdir()
    path2, _ = _flow_ckpt(tmp_path / "b", factors=["preset:arm", "id.slot_handle"])
    st = torch.load(path2, map_location="cpu", weights_only=False)
    st["config"]["policy"]["factors"] = None                                # config edited after the stamp
    torch.save(st, path2)
    with pytest.raises(FactorError):
        LearnedPolicy.from_checkpoint(path2)


def test_bc_loader_refuses_an_unstamped_non_default_list(tmp_path):
    from rrp.policies.bc import LearnedPolicy
    path, _ = _flow_ckpt(tmp_path, factors=["preset:arm", "id.slot_handle"])
    _strip_stamp(path)
    with pytest.raises(FactorError, match="unstamped"):
        LearnedPolicy.from_checkpoint(path)


def test_latent_loader_guards_the_flow_stamp(tmp_path):
    """`LatentPolicy.from_checkpoint`: the flow's stamp is checked before the weights load (the representation it names
    is a stub here: the guard fires first)."""
    from rrp.policies.latent import LatentPolicy
    (tmp_path / "flow").mkdir()
    path, pc = _flow_ckpt(tmp_path / "flow", factors=["preset:arm", "id.slot_handle"])
    rep = _bundle_ckpt(tmp_path)
    st = torch.load(path, map_location="cpu", weights_only=False)
    st["config"]["policy"] = dict(st["config"]["policy"], factors=None)     # edited after the stamp
    st["config"]["representation"] = str(rep)
    torch.save(st, path)
    with pytest.raises(FactorError):
        LatentPolicy.from_checkpoint(path)


# ------------------------------------------------------------------------------------- pipeline base cleanup
def test_register_alias_is_gone_and_register_stage_decorates():
    assert not hasattr(B, "register")
    B._load_families()
    assert ("psi0", "collect") in B._REGISTRY and ("pointer", "eval_r2") in B._REGISTRY
    try:
        @B.register_stage("f3fs", "collect", source="scripted_teacher")
        def fn(ctx):
            return {}
        assert B._REGISTRY[("f3fs", "collect")].fn is fn
    finally:
        B.unregister_stages("f3fs")
        from rrp.core.runconfig import unregister_family
        unregister_family("f3fs")


def _rc(**kw):
    from tests.unit.test_pipeline_contract import _rc as rc
    return rc(**kw)


def test_factor_items_read_the_site_lists_and_versions_compose_through_stamp_versions():
    from rrp.policies.relations.base import compat_hash, resolve, stamp_versions
    rc = _rc(params={"latent": {"factors": ["preset:probes:arm-packet-v1"], "encoder_factors": ["preset:arm"],
                                "realizer_factors": ["preset:s0-arm"]}})
    assert len(B._factor_items(rc)) == 3
    specs, _ = B._resolved_factors(rc)
    want = stamp_versions({}, specs, [])["factors"]
    assert B.stage_versions(rc)["factors"] == want
    assert compat_hash(specs) in want
    moved = _rc(params={"latent": {"factors": ["preset:probes:arm-packet-v1"], "encoder_factors": ["preset:arm"],
                                   "realizer_factors": []}})
    assert B.stage_versions(moved)["factors"] != want or len(B._factor_items(moved)) == 2
    assert B.stage_versions(_rc(options={"kinfeat": "v1"}))["factors"] == "kinfeat_v1"
    assert B.stage_versions(_rc())["factors"] == ""
    assert resolve(["preset:s0-arm"])                                        # sanity: the realizer preset resolves


GRAND = '''
import json, os, subprocess, sys
child = "import json,sys; from rrp.harness.pipelines.base import apply_run_context; from rrp.policies.features import kinfeat; " \\
        "apply_run_context(); json.dump(dict(resolved=kinfeat.resolved()), open(sys.argv[1], 'w'))"
subprocess.run([sys.executable, "-c", child, sys.argv[1]], check=True)          # default env: inherits os.environ
'''


def test_a_spawned_grandchild_sees_the_kinfeat_option(tmp_path):
    from tests.unit.test_pipeline_contract import _child_script, _register_probe, fam  # noqa: F401
    B._load_families()
    key = ("pointer", "eval_r2")
    saved = B._REGISTRY[key]
    try:
        script, out = _child_script(tmp_path, GRAND), tmp_path / "grand.json"
        _register_probe(lambda ctx: (ctx.run([str(script), str(out)]), dict(outputs={}, metrics={}))[1])
        B.Pipeline("pointer").run(_rc(options={"kinfeat": "v1"}), root=tmp_path, index=B.RunIndex())
        assert json.loads(out.read_text()) == dict(resolved=True)
        out.unlink()
        B.Pipeline("pointer").run(_rc(), root=tmp_path, index=B.RunIndex())
        assert json.loads(out.read_text()) == dict(resolved=False)
    finally:
        B._REGISTRY[key] = saved


def test_apply_run_context_without_a_config_reads_the_env_or_is_a_no_op(tmp_path, monkeypatch):
    from rrp.policies.features import kinfeat
    monkeypatch.delenv(B.CONTEXT_ENV, raising=False)
    with B.apply_run_context() as c:                                         # no context: nothing to apply
        assert c.specs == () and kinfeat.resolved() is False
    p = tmp_path / B.CONTEXT_FILE
    p.write_text(_rc(options={"kinfeat": "v1"}).to_json())
    monkeypatch.setenv(B.CONTEXT_ENV, str(p))
    with B.apply_run_context():
        assert kinfeat.resolved() is True
    assert kinfeat.resolved() is False


def test_cli_entry_applies_the_context_of_its_parent_stage(tmp_path):
    p = tmp_path / B.CONTEXT_FILE
    p.write_text(_rc(options={"kinfeat": "v1"}).to_json())
    code = ("import importlib; m = importlib.import_module('rrp.cli.main'); import rrp.cli.tools as t; "
            "from rrp.policies.features import kinfeat; "
            "t.dispatch = lambda argv: (print(kinfeat.resolved()) or True, 0); m.main(['x'])")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                       env=dict(__import__("os").environ, CUDA_VISIBLE_DEVICES="", **{B.CONTEXT_ENV: str(p)}))
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == "True"


def _peer_sync_rsync_args(tmp_path, peer_repo, **env):
    """Run `peer_sync.sh push` with stub ssh / rsync on PATH; return the rsync argument line."""
    import os
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    log = tmp_path / "rsync.log"
    (bin_ / "ssh").write_text("#!/bin/sh\ncat >/dev/null\nexit 0\n")
    (bin_ / "rsync").write_text(f"#!/bin/sh\necho \"$@\" >> {log}\n")
    for f in ("ssh", "rsync"):
        (bin_ / f).chmod(0o755)
    log.unlink(missing_ok=True)
    r = subprocess.run(["bash", str(ROOT / "ops/bin/peer_sync.sh"), "push"], cwd=ROOT, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                       env=dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", RRP_PEER_REPO=peer_repo, **env))
    assert r.returncode == 0, r.stderr
    return log.read_text().splitlines()[0]


def test_peer_sync_pushes_the_resources_config_into_a_per_agent_dir_not_the_shared_one(tmp_path):
    own = _peer_sync_rsync_args(tmp_path, "/dev/shm/rrp-brandonin/wt/fs-test")
    assert "resources.local.json" not in own                                 # synced with the tree
    shared = _peer_sync_rsync_args(tmp_path, "/dev/shm/rrp-brandonin/repo", RRP_ALLOW_SHARED_REPO="1")
    assert "--exclude ops/resources.local.json" in shared                    # the live shared config is never replaced


def test_run_dag_loads_the_pipeline_families_at_plan_time(tmp_path):
    """A fresh process that only calls `rrp run-dag --dry-run` plans an extension-family recipe (psi0): the stage
    modules register at plan time, nothing imported them before."""
    r = subprocess.run([sys.executable, "-m", "rrp.cli", "run-dag", "recipes/psi0/psi0_tabletop_step2.yaml", "--dry-run",
                        "--ledger", str(tmp_path / "ledger.json")], cwd=ROOT, capture_output=True, text=True,
                       env=dict(__import__("os").environ, CUDA_VISIBLE_DEVICES=""))
    assert r.returncode == 0, r.stderr[-800:]
    assert "train_rep" in r.stdout
