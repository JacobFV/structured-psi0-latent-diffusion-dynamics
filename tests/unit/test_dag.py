"""run-dag (W5): planning, ledger, retries and resume with a fake runner; the YAML subset. The rendered nodes of every kept
recipe are pinned by tests/unit/test_recipes.py (recipe.* goldens, D-145)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rrp.harness.dag import DagError, Executor, Ledger, load_dag, plan_dag
from rrp.harness.pipelines.base import stage_versions
from rrp.harness.yamlmini import YamlError, loads

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def arm_plan():
    return plan_dag(load_dag(ROOT / "recipes/templates/arm_lineage.yaml"))


def test_arm_dag_structure(arm_plan):
    assert len(arm_plan.nodes) == 25 * 6
    n = arm_plan.nodes["rzgendag1@semfix.s2"]
    assert set(n.deps) == {f"{d}@semfix.s2" for d in ("rzbcdag2", "bc1", "bc2", "bc3", "gen1")}
    col = arm_plan.nodes["gdag2@sem.s1"]
    assert col.rc.options["seed_start"] == 4000000 and col.rc.options["genctx"] is True
    assert arm_plan.nodes["gdag2@sem.s2"].rc.options["seed_start"] == 5100000
    assert arm_plan.nodes["stageA@nosem.s2"].rc.params["seed"] == 2706
    order = arm_plan.order
    assert order.index("Fgdag2h@semfix.s2") > order.index("gdag2@semfix.s2") > order.index("rzgendag3@semfix.s2")
    sel = arm_plan.select(r"^finalevals", [{"variant": "semfix", "seed": "2"}])
    assert "stageA@semfix.s2" in sel.nodes and all(k.endswith("@semfix.s2") for k in sel.nodes)


def test_legged_dag_plans():
    p = plan_dag(load_dag(ROOT / "recipes/templates/legged_lineage.yaml"))
    assert p.nodes and all(n.rc.family == "legged" for n in p.nodes.values())
    assert all(n.rc.flags.contact_version for n in p.nodes.values())


# ------------------------------------------------------------------------------------------ fake runner
class FakeRunner:
    def __init__(self, tmp: Path, fail: dict | None = None, refuse: int = 0):
        self.root, self.fail, self.refuse = tmp, dict(fail or {}), refuse
        self.launched, self.jobs, self.n = [], {}, 0

    def launch(self, node):
        from rrp.harness.dag import AdmissionRefused
        if self.refuse:
            self.refuse -= 1
            raise AdmissionRefused("aggregate limit")
        self.n += 1
        lid = f"L{self.n}"
        self.launched.append(node.id)
        rc = 1 if self.fail.get(node.id, 0) > 0 else 0
        if rc:
            self.fail[node.id] -= 1
        else:
            p = self.root / node.rc.out
            p.mkdir(parents=True, exist_ok=True)
            from rrp.core.provenance import code_provenance
            (p / "pipeline_manifest.json").write_text(json.dumps(dict(
                config_hash=node.rc.config_hash(), metrics={}, pins=stage_versions(node.rc),
                provenance=dict(code=code_provenance().model_dump()))))
        self.jobs[lid] = rc
        return dict(lease_id=lid, log=f"/x/{lid}.log", placement=node.placement)

    def poll(self, h):
        return self.jobs[h["lease_id"]]

    def manifest(self, node):
        p = self.root / node.rc.out / "pipeline_manifest.json"
        return json.loads(p.read_text()) if p.exists() else None


TOY = """
name: toy
family: arm
track: t
lineage: 'toy-{variant}'
matrix: {variant: [sem], seed: [1]}
defaults: {placement: host, resources: {cpu: 1, mem: 1G}}
base:
  flags: {zero_prev_action: true, realizer_anchor: true, realizer_drop_qd: false, probe_lv_min: -8.0, contact_version: contact_v1}
nodes:
  a:
    stage: refit
    tag: a
    config:
      inputs: {representation: 'runs/x:representation.pt'}
      params: {steps: 1}
  b:
    stage: refit
    tag: b
    retries: 1
    config:
      inputs: {representation: '@a:representation.pt'}
      params: {steps: 2}
  c:
    stage: eval_r2
    tag: c
    config:
      inputs: {flow: 'runs/f:policy.pt', representation: '@b:representation.pt'}
      options: {tag: c, robots: [panda_pg2]}
  d:
    stage: eval_r2
    tag: d
    deps: [a]
    config:
      inputs: {flow: 'runs/f:policy.pt', representation: 'runs/x:representation.pt'}
      options: {tag: d, robots: [panda_pg2]}
"""


def _ex(tmp, runner, plan=None):
    plan = plan or plan_dag(loads(TOY))
    return Executor(plan, Ledger(tmp / "ledger.json"), runner, poll_s=0, sleep=lambda s: None, log=lambda m: None, pins=stage_versions)


def test_plan_refs_and_order():
    p = plan_dag(loads(TOY))
    assert p.nodes["b@sem.s1"].deps == ["a@sem.s1"]
    assert p.nodes["b@sem.s1"].rc.input_paths()["representation"] == "artifacts/runs/t/toy-sem/refit-a_s1/representation.pt"
    assert p.order.index("c@sem.s1") > p.order.index("b@sem.s1") > p.order.index("a@sem.s1")


def test_run_retry_block_and_resume(tmp_path):
    r = FakeRunner(tmp_path, fail={"b@sem.s1": 2})         # b fails twice; retries=1 -> failed, c blocked
    s = _ex(tmp_path, r).run()
    assert s == dict(completed=2, failed=1, blocked=1, planned=0, running=0)
    led = json.loads((tmp_path / "ledger.json").read_text())["nodes"]
    assert len(led["b@sem.s1"]["attempts"]) == 2 and led["c@sem.s1"]["state"] == "blocked"
    assert r.launched.count("b@sem.s1") == 2
    # resume: completed nodes are not relaunched; failed stays failed without --retry-failed
    r2 = FakeRunner(tmp_path)
    s2 = _ex(tmp_path, r2).run()
    assert r2.launched == [] and s2["failed"] == 1
    # a manual retry (what --retry-failed does) runs b and c only
    lg = Ledger(tmp_path / "ledger.json")
    for k in ("b@sem.s1", "c@sem.s1"):
        lg.data["nodes"][k].update(state="planned", attempts=[])
    lg.save()
    r3 = FakeRunner(tmp_path)
    assert _ex(tmp_path, r3).run()["completed"] == 4 and r3.launched == ["b@sem.s1", "c@sem.s1"]


def test_admission_wait_is_not_an_attempt_and_manifest_adoption(tmp_path):
    r = FakeRunner(tmp_path, refuse=2)
    assert _ex(tmp_path, r).run()["completed"] == 4
    assert all(len(v["attempts"]) == 1 for v in json.loads((tmp_path / "ledger.json").read_text())["nodes"].values())
    # a fresh ledger adopts outputs whose manifest carries the same config hash (no lease)
    (tmp_path / "ledger.json").unlink()
    r2 = FakeRunner(tmp_path)
    assert _ex(tmp_path, r2).run()["completed"] == 4 and r2.launched == []


def test_config_change_is_refused(tmp_path):
    _ex(tmp_path, FakeRunner(tmp_path)).run()
    changed = plan_dag(loads(TOY.replace("params: {steps: 1}", "params: {steps: 5}")))
    with pytest.raises(DagError, match="--reset"):
        _ex(tmp_path, FakeRunner(tmp_path), changed).run()


def test_missing_flag_is_an_error():
    with pytest.raises(DagError, match="not stated"):
        plan_dag(loads(TOY.replace("realizer_drop_qd: false, ", "")))


def test_yaml_subset():
    d = loads("""
# comment
a: 1
b: [x, 'y z', {k: -2.5, l: [1, 2]}]
c:
  - p: q   # trailing comment
    r: {s: null}
  - plain
  -
    nested: true
d: "quoted # not a comment"
e:
f: '{1 + 2}'
""")
    assert d == {"a": 1, "b": ["x", "y z", {"k": -2.5, "l": [1, 2]}], "c": [{"p": "q", "r": {"s": None}}, "plain",
                 {"nested": True}], "d": "quoted # not a comment", "e": None, "f": "{1 + 2}"}
    with pytest.raises(YamlError):
        loads("a: &anchor 1")
    with pytest.raises(YamlError):
        loads("a: 1\na: 2")


def test_ops_runner_poll_reads_rc_file(tmp_path):
    from rrp.harness.dag import OpsRunner
    log = tmp_path / "1_x.log"
    (tmp_path / "1_x.rc").write_text("0")                       # the child writes the rc without a newline
    r = OpsRunner(tmp_path)
    assert r.poll(dict(lease_id="1", log=str(log), unit="rrp-job-w5-test-none.service", placement="host")) == 0
    (tmp_path / "1_x.rc").write_text("3")
    assert r.poll(dict(lease_id="1", log=str(log), unit="rrp-job-w5-test-none.service", placement="host")) == 3
    (tmp_path / "1_x.rc").unlink()
    h = dict(lease_id="1", log=str(log), unit="rrp-job-w5-test-none.service", placement="host")
    assert [r.poll(h) for _ in range(3)] == [None, None, -1] and h["unit_result"]


GLOBAL_TOY = """
name: gtoy
family: legged
track: t
lineage: 'l-{variant}'
label_prefix: 'g_{variant}{seed}'
matrix:
  variant: [semfix, nosem]
  seed: [0, 1]
axis_vars:
  variant:
    semfix: {sw: 1.0, lv: -4.0}
    nosem: {sw: 0.0, lv: -8.0}
vars:
  body: anymal_c
  pfx: 'p_{variant}_{seed}'
defaults:
  resources: {cpu: 1, mem: 1G}
base:
  flags: {contact_version: contact_v2}
nodes:
  collect:
    stage: collect
    scope: global
    config:
      options: {body: '{body}', seeds: 0-9}
  bc:
    stage: train_bc
    scope: global
    config:
      inputs: {data: '@collect'}
      params: {bodies: ['{body}'], steps: 1}
  rep:
    stage: train_rep
    per:
      variant:
        semfix:
          params:
            latent:
              factors: [{name: probe.legged.contact, weight: 1.0, params: {lv_min: -4.0}}]
    config:
      flags: {qd_dropout: 0.5}
      inputs: {data: '@collect'}
      params: {name: '{pfx}', bodies: ['{body}'], latent: {factors: [{name: probe.legged.contact, weight: '{sw}'}]}, seed: '{seed}'}
  r2:
    stage: eval_r2
    config:
      inputs: {flow: '@rep:representation.pt', bc: '@bc:policy.pt'}
      options: {body: '{body}'}
"""


def test_global_scope_nodes_are_planned_once_and_shared():
    p = plan_dag(loads(GLOBAL_TOY))
    assert sorted(k for k in p.nodes if "@" not in k) == ["bc", "collect"]
    assert len([k for k in p.nodes if k.startswith("rep@")]) == 4
    assert p.nodes["bc"].deps == ["collect"]
    for k in ("rep@semfix.s0", "rep@nosem.s1"):
        assert p.nodes[k].deps == ["collect"]
        assert p.nodes[k].rc.inputs["data"] == p.nodes["collect"].rc.run_id
    assert p.nodes["r2@nosem.s1"].deps == ["bc", "rep@nosem.s1"]
    assert p.nodes["collect"].rc.lineage == "gtoy" and p.nodes["collect"].rc.variant == "na"
    assert p.nodes["collect"].rc.flags.contact_version == "contact_v2"
    assert p.order.index("collect") < p.order.index("rep@semfix.s0")
    # a global node cannot use axis values
    with pytest.raises(Exception):
        plan_dag(loads(GLOBAL_TOY.replace("options: {body: '{body}', seeds: 0-9}", "options: {body: '{variant}', seeds: 0-9}")))


def test_gpu_and_cpu_caps(tmp_path):
    spec = loads(GLOBAL_TOY.replace("""  rep:
    stage: train_rep
""", """  rep:
    stage: train_rep
    resources: {gpu: true, cpu: 3, mem: 1G}
""").replace("""  r2:
    stage: eval_r2
""", """  r2:
    stage: eval_r2
    resources: {cpu: 10, mem: 1G}
"""))
    plan = plan_dag(spec)

    class Slow(FakeRunner):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.left, self.live, self.peak_gpu, self.peak_cpu = {}, set(), 0, 0

        def launch(self, node):
            h = super().launch(node)
            self.left[h["lease_id"]], h["nid"] = 2, node.id
            self.live.add(node.id)
            self.peak_gpu = max(self.peak_gpu, sum(plan.nodes[k].resources.gpu for k in self.live))
            self.peak_cpu = max(self.peak_cpu, sum(plan.nodes[k].resources.cpu for k in self.live))
            return h

        def poll(self, h):
            self.left[h["lease_id"]] -= 1
            if self.left[h["lease_id"]] > 0:
                return None
            self.live.discard(h["nid"])
            return self.jobs[h["lease_id"]]
    r = Slow(tmp_path)
    ex = Executor(plan, Ledger(tmp_path / "l.json"), r, max_parallel=8, max_parallel_gpu=2, max_cpu=16,
                  poll_s=0, sleep=lambda s: None, log=lambda m: None, pins=stage_versions)
    assert ex.run()["completed"] == len(plan.nodes)
    assert r.peak_gpu == 2 and r.peak_cpu <= 16


def test_point_filter_keeps_global_nodes():
    p = plan_dag(loads(GLOBAL_TOY)).select(None, [{"variant": "semfix"}])
    assert {"collect", "bc"} <= set(p.nodes) and "r2@semfix.s0" in p.nodes and "rep@nosem.s0" not in p.nodes


def test_shared_budget_counts_other_ledgers(tmp_path):
    from rrp.harness.dag import _gib
    assert _gib("24G") == 24 and _gib("512M") == 0.5
    other = tmp_path / "_dags" / "other" / "ledger.json"
    other.parent.mkdir(parents=True)
    other.write_text(json.dumps(dict(schema="dag-ledger-1", nodes={
        "x": dict(state="running", resources=dict(cpu=2, mem="4G", gpu=True, gpu_mem="2G")),
        "y": dict(state="completed", resources=dict(cpu=9, mem="20G", gpu=True))})))
    plan = plan_dag(loads(GLOBAL_TOY.replace("""  rep:
    stage: train_rep
""", """  rep:
    stage: train_rep
    resources: {gpu: true, cpu: 2, mem: 4G, gpu_mem: 2G}
""")))
    ex = Executor(plan, Ledger(tmp_path / "_dags" / "gtoy" / "ledger.json"), FakeRunner(tmp_path), max_parallel_gpu=2,
                  max_cpu=8, max_mem_gib=28, budget_dir=tmp_path / "_dags", poll_s=0, sleep=lambda s: None, log=lambda m: None, pins=stage_versions)
    assert ex._fits("rep@semfix.s0", [])                     # 1 GPU elsewhere + 1 = 2
    assert not ex._fits("rep@semfix.s1", ["rep@semfix.s0"])  # would be 3 GPU nodes in the track
    assert ex._fits("collect", ["rep@semfix.s0", "rep@nosem.s0"])      # cpu 2+2+2+1 = 7 <= 8, mem 18+1 <= 28
    ex.max_cpu = 6
    assert not ex._fits("collect", ["rep@semfix.s0", "rep@nosem.s0"])  # cpu 7 > 6
    ex.max_cpu, ex.max_mem_gib = 8, 18.5
    assert not ex._fits("collect", ["rep@semfix.s0", "rep@nosem.s0"])  # mem 19 > 18.5


def test_caveat_goes_to_notes_and_ledger_without_changing_hashes(tmp_path):
    a = plan_dag(loads(GLOBAL_TOY))
    b = plan_dag(loads("caveat: 'gate exception X'\n" + GLOBAL_TOY))
    assert b.nodes["rep@semfix.s0"].rc.note == "gate exception X"
    assert all(a.nodes[k].rc.config_hash() == b.nodes[k].rc.config_hash() for k in a.nodes)
    ex = Executor(b, Ledger(tmp_path / "l.json"), FakeRunner(tmp_path), poll_s=0, sleep=lambda s: None, log=lambda m: None, pins=stage_versions)
    ex.run()
    assert json.loads((tmp_path / "l.json").read_text())["caveat"] == "gate exception X"


def test_admission_clock_does_not_tick_while_held_by_own_gpu_cap(tmp_path, monkeypatch):
    """D-147 T7 (bendpick 10-04): two GPU nodes, max_parallel_gpu 1. One is refused by the broker once, then the other
    is admitted and runs longer than admission_timeout_s; the first must not fail on admission time spent waiting
    behind our own running node."""
    import rrp.harness.dag as D
    now = [0.0]
    monkeypatch.setattr(D.time, "time", lambda: now[0])
    toy = TOY.replace("defaults: {placement: host, resources: {cpu: 1, mem: 1G}}",
                      "defaults: {placement: host, resources: {cpu: 1, mem: 1G, gpu: true, gpu_mem: 1G}}")
    plan = plan_dag(loads(toy))

    class R(FakeRunner):
        """Broker refuses b once and d twice (d: once before b is admitted, once after b finished) -- the 10-04 sequence."""
        def __init__(s, tmp):
            super().__init__(tmp); s.left = {"b@sem.s1": 1, "d@sem.s1": 2}; s.polls = 0

        def launch(s, node):
            if s.left.get(node.id, 0) > 0:
                s.left[node.id] -= 1
                raise D.AdmissionRefused("gpu owners 3 > slots 2")
            return super().launch(node)

        def poll(s, h):
            s.polls += 1
            now[0] += 1000.0                # every poll of a running node: +1000 s (the long probes_gen of 10-04)
            return None if s.polls % 5 else s.jobs[h["lease_id"]]

    ex = Executor(plan, Ledger(tmp_path / "ledger.json"), R(tmp_path), poll_s=0, sleep=lambda s: None, log=lambda m: None,
                  pins=stage_versions, max_parallel_gpu=1, admission_timeout_s=3000.0)
    s = ex.run()
    assert s["completed"] == 4 and s["failed"] == 0, s
