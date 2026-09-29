"""D-126 #28 privileged-information audit: static AST check, schema check, dynamic tripwire, ablation transforms and
a tiny runtime audit (random-weight bundle; plumbing only)."""
import json

import numpy as np
import pytest
import torch

from rrp.harness.eval import privileged_audit as pa

from ._legged_tiny import tiny_bundle


def test_static_check_clean_with_declared_exceptions():
    res = pa.static_check()
    assert res["violations"] == [], res["violations"]
    fns = {d["function"] for d in res["declared"]}
    assert "rrp.envs.mujoco.legged:LeggedSession._sense" in fns          # the truth+noise localization is reported


def test_static_check_catches_a_leak():
    import types, sys
    mod = types.ModuleType("rrp_tmp_leak")
    src = "def ctx(s, b):\n    x = s.base_pose_truth()\n    return s.data.qvel[b.da]\n"
    exec(compile(src, "rrp_tmp_leak.py", "exec"), mod.__dict__)
    sys.modules["rrp_tmp_leak"] = mod
    import inspect
    orig = inspect.getsource
    try:
        inspect.getsource = lambda obj: src if getattr(obj, "__module__", None) == "rrp_tmp_leak" else orig(obj)
        mod.ctx.__module__ = "rrp_tmp_leak"
        res = pa.static_check({"rrp_tmp_leak:ctx": {}})
    finally:
        inspect.getsource = orig
        del sys.modules["rrp_tmp_leak"]
    assert {v["name"] for v in res["violations"]} == {"base_pose_truth", "da"}


def test_public_observation_schema_has_no_privileged_fields():
    from pydantic import create_model
    from rrp.core.channels import schema_privileged_fields
    from rrp.core.observation import PolicyObservation
    assert schema_privileged_fields(PolicyObservation) == []
    Leaky = create_model("Leaky", __base__=PolicyObservation, object_poses=(dict, {}))
    assert schema_privileged_fields(Leaky) == ["$.object_poses"]


@pytest.mark.parametrize("bss", ["truth_noise", "estimator"])
def test_dynamic_tripwire(bss):
    r = pa.dynamic_tripwire("hexapod6", 0, bss)
    assert r["ctx_dim"] == 22 and r["observation_public"]


def test_ctx_groups_cover_public_context_exactly():
    idx = sorted(i for g in pa.LEGGED_GROUPS if g.where == "ctx" for i in range(*g.index))
    assert idx == list(range(22))


def test_transforms():
    bank = pa.ValueBank()
    for sd, val in ((1, 1.0), (2, 2.0)):
        c_rec, l_rec = bank.recorder(sd)
        for _ in range(3):
            c_rec(np.full(22, val, np.float32), None)
            l_rec(np.full(4, val), np.full(4, val), np.full(6, val), np.full(2, val), None)
    rng = np.random.default_rng(0)
    g = pa.GROUPS["ctx.speed"]
    ct, lt = pa.make_transforms(g, "zero", bank, 1, rng)
    c = ct(np.ones(22, np.float32), None)
    assert lt is None and (c[20:22] == 0).all() and (c[:20] == 1).all()
    ct, _ = pa.make_transforms(g, "shuffle", bank, 1, rng)
    assert (ct(np.ones(22, np.float32), None)[20:22] == 2.0).all()          # value from the other episode
    _, lt = pa.make_transforms(pa.GROUPS["local.touch"], "noise", bank, 1, rng)
    q, qd, imu, touch = lt(np.zeros(4), np.zeros(4), np.zeros(6), np.zeros(2), None)
    assert (q == 0).all() and np.allclose(touch, 1.5, atol=2.0) and (touch != 0).any()


def test_summary_flags():
    base = dict(n=4, changes_behaviour=True)
    res = {("identity", "none"): dict(base, changes_behaviour=False), ("ctx.speed", "zero"): base,
           ("ctx.osc", "zero"): base, ("local.q", "zero"): dict(base, changes_behaviour=False)}
    s = pa.summarize(res, threshold_m=0.1, expect_irrelevant=["ctx.osc"])
    flags = {(f["flag"], f["group"]) for f in s["flags"]}
    assert flags == {("privileged_derived_used", "ctx.speed"), ("expected_irrelevant_but_used", "ctx.osc")}
    s = pa.summarize(res, threshold_m=0.1, base_state_source="estimator")
    assert s["groups"]["ctx.speed"]["provenance"] == "declared_estimator" and not s["flags"]


def test_tiny_runtime_audit(tmp_path):
    torch.set_num_threads(1)
    rep, flow = tiny_bundle(tmp_path / "b")
    from rrp.policies.legged import LatentLeggedController
    make = lambda sd: LatentLeggedController(flow, torch.device("cpu"), nfe=2, seed=sd)
    s = pa.run_audit(make, ["hexapod6"], [3, 4], tmp_path / "audit", groups=["local.q"],
                     ablations=["shuffle"], max_s=0.5)
    assert set(s["groups"]) == {"identity", "local.q"}
    ident = s["groups"]["identity"]["ablations"]["none"]
    assert ident["n"] == 2 and ident["traj_dev_m"]["mean"] == 0.0          # deterministic rerun
    assert json.loads((tmp_path / "audit" / "summary.json").read_text())["version"] == "paudit-1"
    rows = (tmp_path / "audit" / "rows" / "hexapod6_local.q_shuffle.jsonl").read_text().splitlines()
    assert len(rows) == 2 and json.loads(rows[0])["audit"]["condition"] == "local.q_shuffle"
