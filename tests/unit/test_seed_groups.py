"""compute.seeds_per_job: n seeds of one stage in one process (harness/train/lockstep.py) are the independent runs.

Tiny CPU fixture (the relgen trainer tests' two-episode pack). Same seed -> the same init and the same per-step
draws whether a member trains alone or next to others; a failing member does not stop the rest."""
from __future__ import annotations

import pytest
import torch

import rrp.harness.train.latent_train as lt
from rrp.harness.train.lockstep import Outcome, run_members
from tests.unit.test_relgen_trainers import LATENT, POLICY, world  # noqa: F401  (the module-scoped fixture)

STEPS, B = 4, 4


@pytest.fixture(autouse=True)
def _no_eval(monkeypatch):
    monkeypatch.setattr(lt, "evaluate_representation", lambda *a, **k: {})
    monkeypatch.setattr(lt, "evaluate_generated", lambda *a, **k: {})


def _rep_cfg(world, seed):
    return dict(latent=dict(LATENT, encoder_factors=["preset:arm"]), packed_dir=str(world.pack), seed=seed, steps=STEPS,
                batch_size=B, zero_prev_action=True)


def _flow_cfg(rep, world, seed):
    return dict(representation=str(rep), packed_dir=str(world.pack), seed=seed, steps=STEPS, batch_size=B,
                zero_prev_action=True, snapshot_every=2, policy=dict(POLICY, factors=["preset:arm"], name="tinyflow"))


def _model(path):
    return torch.load(path, map_location="cpu", weights_only=False)["model"]


def _same_model(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], dict):
            _same_model(a[k], b[k])
        elif torch.is_tensor(a[k]):
            assert torch.equal(a[k], b[k]), k


@pytest.fixture(scope="module")
def rep(world, tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setattr(lt, "evaluate_representation", lambda *a, **k: {})
    d = tmp_path_factory.mktemp("rep0")
    try:
        lt.train_representation(_rep_cfg(world, 1), d)
    finally:
        mp.undo()
    return d / "representation.pt"


@pytest.mark.parametrize("mode", ["lockstep", "threads"])
def test_flow_group_is_the_independent_runs(world, rep, tmp_path, mode):
    seeds = (5, 6, 7)
    for s in seeds:
        lt.train_latent_flow(_flow_cfg(rep, world, s), tmp_path / f"solo{s}")
    outs, info = lt.train_latent_flow_group([_flow_cfg(rep, world, s) for s in seeds], [tmp_path / f"grp{s}" for s in seeds], mode)
    assert info["mode"] == mode and info["rng_exact"], "the flow loop must not draw from the global RNG"
    assert [o.ok for o in outs] == [True] * 3
    for s, o in zip(seeds, outs):
        assert o.value["steps"] == STEPS
        _same_model(_model(tmp_path / f"solo{s}" / "policy.pt"), _model(tmp_path / f"grp{s}" / "policy.pt"))
        assert (tmp_path / f"grp{s}" / "policy_last.pt").exists()
    # seeds really differ (a group is not one run copied n times)
    a, b = _model(tmp_path / "grp5" / "policy.pt"), _model(tmp_path / "grp6" / "policy.pt")
    assert any(not torch.equal(a[k], b[k]) for k in a if torch.is_tensor(a[k]))


def test_representation_group_swaps_the_global_rng_per_member(world, tmp_path):
    """Stage A draws posterior noise from the global torch RNG: lockstep must keep each member's stream apart."""
    seeds = (1, 2)
    for s in seeds:
        lt.train_representation(_rep_cfg(world, s), tmp_path / f"solo{s}")
    outs, info = lt.train_representation_group([_rep_cfg(world, s) for s in seeds], [tmp_path / f"grp{s}" for s in seeds])
    assert all(o.ok for o in outs) and info["rng_exact"]
    for s in seeds:
        _same_model(_model(tmp_path / f"solo{s}" / "representation.pt"), _model(tmp_path / f"grp{s}" / "representation.pt"))
    # threads share the one global stream: Stage A is NOT exact there, and the run says so
    outs, info = lt.train_representation_group([_rep_cfg(world, s) for s in seeds], [tmp_path / f"thr{s}" for s in seeds], "threads")
    assert all(o.ok for o in outs) and info["rng_exact"] is False


def test_a_failing_member_stops_alone(world, rep, tmp_path):
    good, bad = _flow_cfg(rep, world, 5), dict(_flow_cfg(rep, world, 6), representation=str(tmp_path / "missing.pt"))
    outs, _ = lt.train_latent_flow_group([good, bad, _flow_cfg(rep, world, 7)], [tmp_path / f"m{i}" for i in range(3)])
    assert [o.ok for o in outs] == [True, False, True]
    assert isinstance(outs[1].value, Exception)
    with pytest.raises(Exception):
        outs[1].unwrap()
    assert (tmp_path / "m0" / "policy.pt").exists() and (tmp_path / "m2" / "policy.pt").exists()


def test_members_may_finish_at_different_steps():
    def member(n):
        for i in range(n):
            yield
        return n
    outs, _ = run_members([member(2), member(5), member(0)])
    assert [o.value for o in outs] == [2, 5, 0] and all(isinstance(o, Outcome) for o in outs)


# ---------------------------------------------------------------------------------- planner / executor / run-group
import json as _json
from pathlib import Path as _Path

import pytest

from rrp.harness import dag as _dag
from rrp.harness.dag import Executor, Ledger, form_groups, format_plan, plan_dag, scale_resources
from rrp.harness.pipelines import base as _base
from rrp.harness.pipelines.base import stage_versions
from rrp.harness.yamlmini import loads

GROUPED = """
name: gseeds
family: arm
track: t
lineage: 'g-{variant}'
matrix: {variant: [nosem], seed: [1, 2, 3, 4, 5]}
defaults: {placement: host, resources: {cpu: 1, mem: 1G}}
base:
  flags: {zero_prev_action: true, realizer_anchor: true, realizer_drop_qd: false, probe_lv_min: -8.0, contact_version: contact_v1}
nodes:
  rep:
    stage: train_rep
    resources: {gpu: true, gpu_mem: 8G, cpu: 2, mem: 4G, max_seconds: 100}
    config:
      inputs: {dataset: 'datasets/x', packed_dir: 'packed/x'}
      params: {seed: '{seed}', steps: 3, name: 'p_{seed}', latent: {factors: [{name: probe.arm.visible, weight: 0.0}]}}
  flow:
    stage: train_flow
    resources: {gpu: true, gpu_mem: 8G, cpu: 2, mem: 4G, max_seconds: 100}
    config:
      inputs: {representation: '@rep:representation.pt', packed_dir: 'packed/x'}
      params: {seed: '{seed}', steps: 3, name: 'f_{seed}', packet_semantic_weight: 0.0}
  ev:
    stage: eval_r2
    config:
      inputs: {flow: '@flow:policy.pt', representation: '@rep:representation.pt'}
      options: {tag: ev, robots: [panda_pg2]}
"""


@pytest.fixture
def grouped(monkeypatch):
    monkeypatch.setattr(_dag, "_seeds_per_job", lambda rc: 2 if rc.stage in ("train_rep", "train_flow") else 1)
    return plan_dag(loads(GROUPED))


def test_planner_groups_siblings_and_scales_declarations(grouped):
    gs = list(grouped.groups.values())
    assert sorted(g.members for g in gs) == sorted([["rep@nosem.s1", "rep@nosem.s2"], ["rep@nosem.s3", "rep@nosem.s4"],
                                                    ["flow@nosem.s1", "flow@nosem.s2"], ["flow@nosem.s3", "flow@nosem.s4"]])
    g = grouped.groups[next(k for k, v in grouped.groups.items() if v.members[0] == "rep@nosem.s1")]
    one = grouped.nodes["rep@nosem.s1"].resources
    assert (g.resources.cpu, g.resources.mem, g.resources.gpu_mem, g.resources.max_seconds) == (4, "8G", "16G", 200)
    assert g.resources.gpu and one.cpu == 2                      # members keep their own single-node declaration
    assert not any("ev@" in m for g in gs for m in g.members)    # evals are untouched
    assert "GROUP grp-" in format_plan(grouped)
    again = form_groups(grouped.nodes, grouped.order)
    assert set(again) == set(grouped.groups)                     # stable ids
    assert scale_resources(one, 3).mem == "12G"
    sel = grouped.select(r"^rep", [{"seed": "1"}, {"seed": "2"}])
    assert [g.members for g in sel.groups.values()] == [["rep@nosem.s1", "rep@nosem.s2"]]


def test_default_plan_has_no_groups():
    assert plan_dag(loads(GROUPED)).groups == {}


def test_group_with_member_dependency_or_cycle_is_dissolved(monkeypatch):
    monkeypatch.setattr(_dag, "_seeds_per_job", lambda rc: 2 if rc.stage in ("train_rep", "train_flow") else 1)
    p = plan_dag(loads(GROUPED))
    # flow s1 -> rep s1 (group A), flow s2 -> rep s2: add a cross dependency rep s2 -> flow s1 by hand: cycle between groups
    p.nodes["rep@nosem.s2"].deps.append("flow@nosem.s1")
    g = form_groups(p.nodes, p.order)
    members = [m for x in g.values() for m in x.members]
    assert not ({"rep@nosem.s1", "rep@nosem.s2", "flow@nosem.s1", "flow@nosem.s2"} <= set(members))   # one side was dissolved


class GroupRunner:
    """FakeRunner for groups: a lease runs the whole member list; `fail` names members that produce no manifest."""

    def __init__(self, tmp, fail=(), left=1):
        self.root, self.fail, self.left = tmp, set(fail), left
        self.launches, self.jobs, self.polls, self.n, self.live_slots, self.peak = [], {}, 0, 0, set(), 0

    def _write(self, node):
        from rrp.core.provenance import code_provenance
        p = self.root / node.rc.out
        p.mkdir(parents=True, exist_ok=True)
        (p / "pipeline_manifest.json").write_text(_json.dumps(dict(
            config_hash=node.rc.config_hash(), metrics={}, pins=stage_versions(node.rc),
            provenance=dict(code=code_provenance().model_dump()))))

    def _lease(self, ids, rc, group=False):
        self.n += 1
        lid = f"L{self.n}"
        self.launches.append(tuple(ids))
        self.jobs[lid] = [rc, self.left]
        self.live_slots.add(lid)
        self.peak = max(self.peak, len(self.live_slots))
        return dict(lease_id=lid, log=f"/x/{lid}.log", placement="host")

    def launch(self, node):
        self.fail.discard(node.id)               # a retry of the failed member alone succeeds
        self._write(node)
        return self._lease([node.id], 0)

    def launch_group(self, group, members):
        for m in members:
            if m.id not in self.fail:
                self._write(m)
        bad = [m.id for m in members if m.id in self.fail]
        return self._lease([m.id for m in members], 87 if bad else 0, True)

    def poll(self, h):
        self.polls += 1
        st = self.jobs[h["lease_id"]]
        st[1] -= 1
        if st[1] > 0:
            return None
        self.live_slots.discard(h["lease_id"])
        return st[0]

    def manifest(self, node):
        p = self.root / node.rc.out / "pipeline_manifest.json"
        return _json.loads(p.read_text()) if p.exists() else None

    def _read_json(self, node, name):
        p = self.root / node.rc.out / name
        return _json.loads(p.read_text()) if p.exists() else None


def _gex(tmp, plan, runner, **kw):
    return Executor(plan, Ledger(tmp / "ledger.json"), runner, poll_s=0, sleep=lambda s: None, log=lambda m: None,
                    pins=stage_versions, **kw)


def test_executor_runs_a_group_as_one_lease(tmp_path, grouped):
    r = GroupRunner(tmp_path, left=2)
    ex = _gex(tmp_path, grouped.select(r"^(rep|flow)", [{"seed": s} for s in "1234"]), r)
    assert ex.run()["completed"] == 8
    assert r.launches == [("rep@nosem.s1", "rep@nosem.s2"), ("rep@nosem.s3", "rep@nosem.s4"),
                          ("flow@nosem.s1", "flow@nosem.s2"), ("flow@nosem.s3", "flow@nosem.s4")][:2] + r.launches[2:]
    assert len(r.launches) == 4 and all(len(x) == 2 for x in r.launches)
    assert r.polls <= 2 * 4 * 2                                  # one poll per lease per cycle, not per member
    e1, e2 = ex.ledger.node("rep@nosem.s1"), ex.ledger.node("rep@nosem.s2")
    assert e1["group"] == e2["group"] and e1["attempts"][-1]["lease_id"] == e2["attempts"][-1]["lease_id"]
    assert e1["resources"]["cpu"] == 2                           # the member's own declaration


def test_group_waits_for_every_member_and_counts_one_gpu_slot(tmp_path, grouped):
    plan = grouped.select(r"^(rep|flow)", [{"seed": s} for s in "1234"])
    seen = []

    class Watch(GroupRunner):
        def launch_group(self, group, members):
            seen.append(sorted(x.id for x in members))
            return super().launch_group(group, members)
    r = Watch(tmp_path, left=2)
    ex = _gex(tmp_path, plan, r, max_parallel=8, max_parallel_gpu=2, max_cpu=8)
    ex.run()
    # two group leases (declared cpu 4 each) fit under max_cpu 8 together; the member count would have said 4 GPU nodes
    assert r.launches[0] + r.launches[1] == ("rep@nosem.s1", "rep@nosem.s2", "rep@nosem.s3", "rep@nosem.s4")
    # a flow group starts only after BOTH its reps are done
    assert seen.index(["flow@nosem.s1", "flow@nosem.s2"]) > seen.index(["rep@nosem.s1", "rep@nosem.s2"])


def test_partial_failure_retries_the_member_alone(tmp_path, grouped):
    plan = grouped.select(r"^rep", [{"seed": "1"}, {"seed": "2"}])
    plan.nodes["rep@nosem.s2"].retries = 1
    r = GroupRunner(tmp_path, fail={"rep@nosem.s2"})
    ex = _gex(tmp_path, plan, r)
    assert ex.run()["completed"] == 2
    assert r.launches == [("rep@nosem.s1", "rep@nosem.s2"), ("rep@nosem.s2",)]
    assert "this member" in ex.ledger.node("rep@nosem.s2").get("last_error", "") or ex.ledger.node("rep@nosem.s2")["attempts"][0]["rc"] == 87
    assert ex.ledger.node("rep@nosem.s1")["state"] == "completed" and len(ex.ledger.node("rep@nosem.s1")["attempts"]) == 1


def test_partial_failure_without_retries_blocks_only_that_members_dependants(tmp_path, grouped):
    plan = grouped.select(r"^(rep|flow|ev)", [{"seed": "1"}, {"seed": "2"}])
    r = GroupRunner(tmp_path, fail={"rep@nosem.s2"})
    s = _gex(tmp_path, plan, r).run()
    assert s["failed"] == 1 and s["completed"] >= 1
    assert r.launches[0] == ("rep@nosem.s1", "rep@nosem.s2")


def test_adopted_member_drops_out_of_the_group(tmp_path, grouped):
    plan = grouped.select(r"^rep", [{"seed": "1"}, {"seed": "2"}])
    r = GroupRunner(tmp_path)
    r._write(plan.nodes["rep@nosem.s1"])                            # a finished manifest with this hash and these pins
    ex = _gex(tmp_path, plan, r)
    assert ex.run()["completed"] == 2
    assert r.launches == [("rep@nosem.s2",)]                        # one member left: a plain single-node launch
    assert ex.ledger.node("rep@nosem.s1").get("adopted")


def test_ops_runner_group_command_roundtrips(tmp_path, grouped):
    import base64
    import zlib
    from rrp.core.runconfig import RunConfig
    runner = _dag.OpsRunner(root=tmp_path)
    g = next(iter(grouped.groups.values()))
    cmd = runner.command_group([grouped.nodes[m] for m in g.members], "py")
    assert cmd[:6] == ["py", "-m", "rrp.cli", "stage", "run-group", "--group-b64"]
    rcs = [RunConfig.model_validate(x) for x in _json.loads(zlib.decompress(base64.b64decode(cmd[6])))]
    argv, _ = runner.launch_argv(g, [grouped.nodes[m] for m in g.members])
    assert argv[-6:-1] == cmd[1:6] and "--label" in argv and "--detach" in argv
    assert [r.config_hash() for r in rcs] == [grouped.nodes[m].rc.config_hash() for m in g.members]


def _fake_group_stage(monkeypatch, fail_seed=None):
    """arm/train_rep's group runner replaced by one that writes the declared outputs without training."""
    def fn(ctxs):
        outs = []
        for c in ctxs:
            if c.rc.seed == fail_seed:
                outs.append(RuntimeError("boom"))
                continue
            d = c.out
            d.mkdir(parents=True, exist_ok=True)
            (d / "representation.pt").write_bytes(b"x" + bytes([c.rc.seed]))
            outs.append(dict(outputs={"representation": str(d / "representation.pt")}, metrics={"seed": c.rc.seed}))
        return outs, dict(mode="lockstep", rng_exact=True)
    monkeypatch.setitem(_base._GROUPS, ("arm", "train_rep"), fn)


def test_run_group_writes_each_members_manifest_and_failure_reason(tmp_path, monkeypatch, grouped, capsys):
    _fake_group_stage(monkeypatch, fail_seed=2)
    rcs = [grouped.nodes[f"rep@nosem.s{i}"].rc for i in (1, 2, 3)]
    res, rec = _base.Pipeline("arm").run_group(rcs, root=tmp_path, check_inputs=False)
    assert isinstance(res[1], RuntimeError) and rec["size"] == 3 and rec["mode"] == "lockstep"
    for i in (0, 2):
        m = _json.loads((tmp_path / rcs[i].out / "pipeline_manifest.json").read_text())
        assert m["config_hash"] == rcs[i].config_hash() and m["pins"] == stage_versions(rcs[i])
        assert m["group"]["group_id"] == rec["group_id"] and m["group"]["index"] == i
        assert rcs[i].run_id in m["group"]["members"]
    assert not (tmp_path / rcs[1].out / "pipeline_manifest.json").exists()
    f = _json.loads((tmp_path / rcs[1].out / "group_failure.json").read_text())
    assert "boom" in f["error"] and f["run_id"] == rcs[1].run_id


def test_stage_main_run_group_exit_codes(tmp_path, monkeypatch, grouped, capsys):
    cfgs = tmp_path / "g.json"
    rcs = [grouped.nodes[f"rep@nosem.s{i}"].rc for i in (1, 2)]
    cfgs.write_text(_json.dumps([r.model_dump(mode="json") for r in rcs]))
    _fake_group_stage(monkeypatch)
    assert _base.stage_main(["run-group", "--group", str(cfgs), "--root", str(tmp_path), "--no-check-inputs"]) == 0
    _fake_group_stage(monkeypatch, fail_seed=1)
    assert _base.stage_main(["run-group", "--group", str(cfgs), "--root", str(tmp_path), "--no-check-inputs"]) == _base.GROUP_PARTIAL_EXIT
    assert not (tmp_path / rcs[0].out / "pipeline_manifest.json").exists()      # stale manifest of the failed member cleared
    assert (tmp_path / rcs[1].out / "pipeline_manifest.json").exists()


def test_run_group_rejects_members_that_differ_in_process_configuration(tmp_path, monkeypatch, grouped):
    _fake_group_stage(monkeypatch)
    a, b = grouped.nodes["rep@nosem.s1"].rc, grouped.nodes["rep@nosem.s2"].rc.model_copy(deep=True)
    b.options = dict(b.options, workers=9)
    with pytest.raises(_base.StageError):
        _base.Pipeline("arm").run_group([a, b], root=tmp_path, check_inputs=False)


def test_a_group_takes_one_parallel_slot(tmp_path, grouped):
    r = GroupRunner(tmp_path, left=3)
    ex = _gex(tmp_path, grouped.select(r"^rep", [{"seed": s} for s in "1234"]), r, max_parallel=2, max_parallel_gpu=2)
    assert ex.run()["completed"] == 4
    assert r.peak == 2                                           # two group leases (four running members) at once
