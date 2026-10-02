"""Fanout unit R11 (docs/relations.md section 10): the competence-driven scheduler policy and `compose`.

F4's placeholder-policy tests (uniform shares, steer grammar, floor monotonicity, replay contract) live in
`tests/unit/test_relgen.py` and are untouched by this unit; these tests cover what R11 fills in: signals (EMA
competence / plateau / interference / failures), promotion / demotion / drop-back, responsive rate-limited shares,
`compose`'s minimal cover / requires-closure / conflict-rejection, `interference` / `interfering_pairs`, and the
`rrp steer` / `rrp suite relations-curriculum` CLI surfaces.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from rrp.cli import curriculum as curriculum_cli
from rrp.harness.data.relgen import PARTS, ComposeError, ScenePart, compose
from rrp.harness.data.relgen.curriculum import (Scheduler, SchedulerConfig, interference, interfering_pairs,
                                                 max_interference, parse_steer)


# ==================================================================== compose
# All fixture part NAMES, ACTIVATION TAGS and the ENV below are namespaced ("_ut11") so `compose`'s tests are
# isolated from whatever the real catalog (other fanout units' `PARTS` entries, e.g. R16's "grasp_target" activating
# "contact", R17's "stack" activating "support", both for env "mujoco/arm") happens to have registered by the time
# this test runs in the full suite -- a real part registered at another test module's import time is process-global
# and outlives that module (docs/relations.md 5.2's PARTS has no per-test reset), so reusing a real env name or a
# generic tag like "support"/"contact" here would silently pick up real parts and make `compose`'s result (and this
# test's assertions on it) depend on suite ordering / which sibling units have merged.
_UT_ENV = "unit-test-env_ut11"
_UT_OTHER_ENV = "unit-test-other-env_ut11"


@pytest.fixture
def parts_sandbox():
    """Registers/unregisters test-only ScenePart fixtures in the real PARTS registry, restoring (not just deleting)
    any entry a fixture name happens to shadow, so this is safe even if a name collides with something real."""
    added: dict[str, object] = {}   # name -> the PARTS entry that was there before (or _MISSING)
    _MISSING = object()

    def _add(name, activates, requires=(), conflicts=(), envs=(), build=None):
        name = f"{name}_ut11"
        if name not in added:
            added[name] = PARTS.get(name, _MISSING)
        p = ScenePart(name=name, version="1", activates=frozenset(activates), requires=frozenset(requires),
                      conflicts=frozenset(conflicts), envs=tuple(envs),
                      build=build or (lambda draft, rng, _n=name: draft.entities.append({"name": _n})))
        PARTS[name] = p
        return p

    yield _add
    for n, prev in added.items():
        if prev is _MISSING:
            PARTS.pop(n, None)
        else:
            PARTS[n] = prev


def test_compose_minimal_cover(parts_sandbox):
    parts_sandbox("part_x", {"ut.x"})
    parts_sandbox("part_y", {"ut.y"})
    parts_sandbox("part_xy", {"ut.x", "ut.y"})       # one part covering both beats combining two single-purpose parts
    draft = compose({"ut.x", "ut.y"}, _UT_ENV, np.random.default_rng(0))
    assert draft.parts == ("part_xy_ut11",)
    assert draft.active == frozenset({"ut.x", "ut.y"})


def test_compose_closes_requires_transitively(parts_sandbox):
    parts_sandbox("stack_part", {"ut.support"}, requires={"ut.contact"})
    parts_sandbox("contact_part", {"ut.contact"}, requires={"ut.table"})
    parts_sandbox("table_part", {"ut.table"})
    draft = compose({"ut.support"}, _UT_ENV, np.random.default_rng(0))
    assert set(draft.parts) == {"stack_part_ut11", "contact_part_ut11", "table_part_ut11"}
    assert draft.active == frozenset({"ut.support", "ut.contact", "ut.table"})


def test_compose_rejects_conflicts(parts_sandbox):
    parts_sandbox("a", {"ut.x"}, conflicts={"ut.y"})
    parts_sandbox("b", {"ut.y"})
    with pytest.raises(ComposeError):
        compose({"ut.x", "ut.y"}, _UT_ENV, np.random.default_rng(0))


def test_compose_unsatisfiable_requires_raises(parts_sandbox):
    parts_sandbox("needs_ghost", {"ut.x"}, requires={"ut.ghost"})   # nothing activates "ut.ghost"
    with pytest.raises(ComposeError):
        compose({"ut.x"}, _UT_ENV, np.random.default_rng(0))


def test_compose_no_covering_part_raises(parts_sandbox):
    parts_sandbox("unrelated", {"ut.y"})
    with pytest.raises(ComposeError):
        compose({"ut.x"}, _UT_ENV, np.random.default_rng(0))


def test_compose_env_compatibility(parts_sandbox):
    parts_sandbox("this_env_only", {"ut.x"}, envs=(_UT_ENV,))
    with pytest.raises(ComposeError):
        compose({"ut.x"}, _UT_OTHER_ENV, np.random.default_rng(0))
    draft = compose({"ut.x"}, _UT_ENV, np.random.default_rng(0))
    assert draft.parts == ("this_env_only_ut11",)


def test_compose_empty_active_set_is_a_noop():
    draft = compose((), _UT_ENV, np.random.default_rng(0))
    assert draft.active == frozenset() and draft.parts == () and draft.entities == []


def test_compose_merges_and_dedupes_events(parts_sandbox):
    parts_sandbox("a", {"ut.x"}, build=lambda d, r: d.events.append({"kind": "e1"}))
    parts_sandbox("b", {"ut.y"}, build=lambda d, r: (d.events.append({"kind": "e1"}), d.events.append({"kind": "e2"})))
    draft = compose({"ut.x", "ut.y"}, _UT_ENV, np.random.default_rng(0))
    assert draft.events == [{"kind": "e1"}, {"kind": "e2"}]     # the "e1" duplicate across parts is merged away


def test_compose_resolves_non_overlapping_layout(parts_sandbox):
    parts_sandbox("a", {"ut.x"}, build=lambda d, r: d.entities.append({"name": "o1", "pos": (0.0, 0.0, 0.0), "radius": 0.2}))
    parts_sandbox("b", {"ut.y"}, build=lambda d, r: d.entities.append({"name": "o2", "pos": (0.05, 0.0, 0.0), "radius": 0.2}))
    draft = compose({"ut.x", "ut.y"}, _UT_ENV, np.random.default_rng(0))
    xs = sorted(e["pos"][0] for e in draft.entities)
    assert xs[1] - xs[0] >= 0.2 + 0.2 + 0.15 - 1e-9    # radius + radius + min_gap clearance


def test_compose_is_deterministic(parts_sandbox):
    parts_sandbox("small", {"ut.contact"})
    parts_sandbox("big", {"ut.contact", "ut.support"})
    a = compose({"ut.contact"}, _UT_ENV, np.random.default_rng(0))
    b = compose({"ut.contact"}, _UT_ENV, np.random.default_rng(0))
    assert a.parts == b.parts and a.active == b.active


# ==================================================================== competence-driven scheduler policy
def test_default_trajectory_promotes_to_full_depth_and_mostly_full_world():
    factors = ("a", "b", "c")
    cfg = SchedulerConfig(factors=factors, interval=10, ramp_steps=200,
                          promote=tuple((f, 0.8) for f in factors))
    s = Scheduler(cfg, seed=0)
    for i in range(60):
        step = i * cfg.interval
        comp = min(0.95, 0.3 + 0.03 * i)                       # every factor steadily improves
        s.observe(step, {f: {"competence": comp} for f in factors})
        s.decide(step)
    last = s.history[-1]
    assert all(last.level[f] == len(factors) for f in factors), last.level     # every factor reaches max depth
    assert last.full_world >= 0.75                                             # mostly full-world (rising floor)
    fw = [h.full_world for h in s.history]
    assert all(b >= a for a, b in zip(fw, fw[1:]))                             # floor never decreases


def test_never_observed_factor_keeps_level_one_and_flat_baseline_share():
    """A factor `observe()` never reports for behaves like the F4 placeholder: level 1, flat baseline weight."""
    cfg = SchedulerConfig(factors=("a", "b"), interval=10)
    s = Scheduler(cfg, seed=0)
    st = s.decide(0)
    assert st.level == {"a": 1, "b": 1}
    assert st.share["a"] == pytest.approx(st.share["b"])


def test_struggling_factor_share_rises_bounded_and_rate_limited():
    # 5 easy factors dilute the baseline (each gets a modest share); "bad" starts at that same baseline (interval 0,
    # good competence) then starts struggling from interval 1 on, so the rise is genuinely rate-limited rather than
    # jumping straight to the ceiling on the very first (un-rate-limited, no-previous-share) decision.
    factors = tuple(f"good{i}" for i in range(5)) + ("bad",)
    cfg = SchedulerConfig(factors=factors, interval=10, ramp_steps=10 ** 9, max_step_change=0.05)
    s = Scheduler(cfg, seed=1)
    shares_bad = []
    for i in range(25):
        step = i * cfg.interval
        metrics = {f: {"competence": 0.9} for f in factors if f != "bad"}
        metrics["bad"] = ({"competence": 0.9} if i == 0 else
                          {"competence": 0.1, "plateau": 0.5, "interference": 0.6, "failures": 0.3})
        s.observe(step, metrics)
        st = s.decide(step)
        shares_bad.append(st.share["bad"])
    assert all(cfg.share_min - 1e-9 <= v <= cfg.share_max + 1e-9 for v in shares_bad)     # always in bounds
    assert shares_bad[0] < 0.2                                                            # started at the baseline
    assert shares_bad[-1] == pytest.approx(cfg.share_max, abs=1e-6)                       # boosted to the ceiling
    assert all(v1 <= v2 + 1e-9 for v1, v2 in zip(shares_bad, shares_bad[1:]))             # monotonic rise, no oscillation
    assert all(b - a <= cfg.max_step_change + 1e-9 for a, b in zip(shares_bad, shares_bad[1:]))   # rate-limited
    assert s.history[-1].level["bad"] == 1                                                # never promoted (low competence)


def test_dropback_holds_the_level_for_its_cooldown_no_oscillation():
    """A factor promoted to k=2 that then starts interfering heavily (while staying individually competent) drops
    back exactly once and STAYS down for its whole cooldown window -- the "bounded period" that prevents it from
    immediately promoting back up and oscillating every interval."""
    factors = ("f", "g")
    cfg = SchedulerConfig(factors=factors, interval=10, promote=(("f", 0.5), ("g", 0.99)))
    s = Scheduler(cfg, seed=2)
    for i in range(6):                                          # phase 1: f earns a promotion to k=2
        step = i * cfg.interval
        s.observe(step, {"f": {"competence": 0.9}, "g": {"competence": 0.0}})
        s.decide(step)
    assert s.history[-1].level["f"] == 2
    reasons, levels = [], []
    # f's competence NEVER drops (only its interference/plateau/failures do), so without recovery-gating the level
    # would promote right back up once a fixed cooldown timer expired and immediately drop again -- forever. Run
    # well past the fixed cooldown window to prove that does not happen.
    for i in range(6, 40):
        step = i * cfg.interval
        s.observe(step, {"f": {"competence": 0.9, "plateau": 1.0, "interference": 1.0, "failures": 1.0},
                         "g": {"competence": 0.0}})
        st = s.decide(step)
        reasons += [r for r in st.reasons if r.startswith("drop-back f")]
        levels.append(st.level["f"])
    assert len(reasons) == 1, reasons                             # exactly one drop-back event, never repeats
    first_drop = next(i for i, lvl in enumerate(levels) if lvl == 1)
    assert all(v == 1 for v in levels[first_drop:])                # held at k=1 permanently (struggle never recovers)


def test_pins_and_freezes_override_the_policy():
    cfg = SchedulerConfig(factors=("a", "b"), interval=10)
    s = Scheduler(cfg, seed=0)
    s.steer(parse_steer("pin a k=3"), step=0)
    st = s.decide(0)
    assert st.level["a"] == 3
    s2 = Scheduler(cfg, seed=0)
    s2.decide(0)
    s2.steer(parse_steer("freeze a"), step=10)
    st2 = s2.decide(10)
    assert st2.share["a"] == pytest.approx(s2.history[0].share["a"])


def test_invalid_steer_is_rejected_and_logged_not_applied():
    cfg = SchedulerConfig(factors=("a",), interval=10)
    s = Scheduler(cfg, seed=0)
    bad = s.steer(parse_steer("pin zzz k=2"), step=0)
    assert bad.reason.startswith("REJECTED")
    st = s.decide(0)
    assert "zzz" not in st.level


def test_replay_reproduces_sampling_with_observed_metrics():
    cfg = SchedulerConfig(factors=("a", "b", "c"), interval=10, ramp_steps=100)
    s = Scheduler(cfg, seed=4)
    for i in range(8):
        step = i * cfg.interval
        s.observe(step, {"a": {"competence": 0.2 + 0.05 * i}, "b": {"competence": 0.9},
                         "c": {"competence": 0.5, "interference": 0.3}})
    s.steer(parse_steer("boost c x2 for 40"), step=20)
    seq = []
    for t in range(0, 80, 5):
        if t % cfg.interval == 0:
            s.decide(t)
        seq.append(s.sample(t, 32))
    _, seq2 = Scheduler.replay(cfg, 4, s.steer_log, s.metrics_log, range(0, 80, 5), 32)
    assert seq == seq2
    hist2 = [h.level for h in s.history]
    s3, _ = Scheduler.replay(cfg, 4, s.steer_log, s.metrics_log, range(0, 80, 5), 32)
    assert [h.level for h in s3.history] == hist2                 # signals/levels replay identically too


# ==================================================================== interference
def _rec(active, **metric):
    return {"active": frozenset(active), "metric": metric}


def test_interference_detects_a_planted_pair():
    # "x" does well alone or with "z", but worse whenever "y" is co-active -- a planted interfering pair (x, y).
    records = [_rec(["x"], x=0.9), _rec(["x"], x=0.85), _rec(["x", "z"], x=0.88, z=0.7),
              _rec(["x", "y"], x=0.4, y=0.6), _rec(["x", "y"], x=0.35, y=0.65),
              _rec(["y"], y=0.7), _rec(["z"], z=0.7)]
    v = interference(records, "x", "y")
    assert v is not None and v > 0.3                        # without y minus with y: a large positive gap
    vz = interference(records, "x", "z")
    assert vz is not None and vz < 0.1                       # z does not (positively) interfere with x

    pairs = interfering_pairs(records, ["x", "y", "z"], threshold=0.1)
    assert pairs and pairs[0][:2] == ("x", "y")              # the planted pair is flagged; (x, z) is not (v <= 0)
    assert all(g != "z" for _, g, _ in pairs)

    mi = max_interference(records, ["x", "y", "z"])
    assert mi["x"] == pytest.approx(v)                       # max_g interference(x, g) is the (x, y) value


def test_interference_none_without_comparable_records():
    records = [_rec(["x", "y"], x=0.5, y=0.5)]     # "x" is never seen WITHOUT "y"
    assert interference(records, "x", "y") is None
    assert max_interference(records, ["x", "y"]) == {"x": 0.0, "y": 0.0}


def test_interference_feeds_the_struggle_score_via_observe():
    """The doc's intended loop: an eval harness computes `max_interference` from its records and passes it into
    `observe()`, where it raises the struggling factor's share same as a hand-set `interference` value would."""
    records = [_rec(["bad"], bad=0.9), _rec(["bad", "good"], bad=0.2, good=0.9), _rec(["good"], good=0.9)]
    i_f = max_interference(records, ["bad", "good"])
    cfg = SchedulerConfig(factors=("bad", "good"), interval=10, ramp_steps=10 ** 9)
    s = Scheduler(cfg, seed=0)
    for step in (0, 10, 20):
        s.observe(step, {"bad": {"competence": 0.3, "interference": i_f["bad"]}, "good": {"competence": 0.9}})
        s.decide(step)
    assert s.history[-1].share["bad"] > s.history[-1].share["good"]


# ==================================================================== schedule.jsonl export
def test_export_writes_one_jsonl_line_per_decision(tmp_path):
    cfg = SchedulerConfig(factors=("a",), interval=10)
    s = Scheduler(cfg, seed=0)
    path = tmp_path / "schedule.jsonl"
    for step in (0, 10):
        s.decide(step)
        s.export(path)
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert [l["step"] for l in lines] == [0, 10]


# ==================================================================== CLI: rrp steer / rrp suite relations-curriculum
def test_cli_steer_appends_a_json_line_cli_grammar(tmp_path):
    class _A:
        run = str(tmp_path)
        op = ["boost", "ix.support", "x2", "for", "5000"]      # argparse.REMAINDER: one token per bare word
    assert curriculum_cli.cmd_steer(_A()) == 0
    lines = (tmp_path / "steer.jsonl").read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["op"] == "boost" and rec["factor"] == "ix.support" and rec["value"] == 2.0 and rec["steps"] == 5000


def test_cli_steer_appends_a_json_line_json_form_with_author_reason(tmp_path):
    # a shell-quoted JSON object arrives as REMAINDER's single element; this is the only way to set author/reason
    # (see cmd_steer's docstring note: separate --author/--reason flags conflict with REMAINDER swallowing them).
    class _A:
        run = str(tmp_path)
        op = ['{"op": "pin", "factor": "geo.above", "value": 2, "author": "lead", "reason": "lags"}']
    assert curriculum_cli.cmd_steer(_A()) == 0
    rec = json.loads((tmp_path / "steer.jsonl").read_text().splitlines()[0])
    assert rec["op"] == "pin" and rec["author"] == "lead" and rec["reason"] == "lags"


def test_cli_steer_rejects_unparseable_op(tmp_path):
    class _A:
        run = str(tmp_path)
        op = ["teleport", "ix.support"]
    with pytest.raises(ValueError):
        curriculum_cli.cmd_steer(_A())


def test_cli_suite_relations_curriculum_reports_history_and_interference(tmp_path, capsys):
    cfg = SchedulerConfig(factors=("a", "b"), interval=10, ramp_steps=10 ** 9)
    s = Scheduler(cfg, seed=0)
    for step in (0, 10):
        s.observe(step, {"a": {"competence": 0.4}, "b": {"competence": 0.9}})
        s.decide(step)
        s.export(tmp_path / "schedule.jsonl")
    (tmp_path / "eval_records.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"active": ["a"], "metric": {"a": 0.9}}, {"active": ["a", "b"], "metric": {"a": 0.3, "b": 0.9}},
        {"active": ["b"], "metric": {"b": 0.9}}]) + "\n")
    rc = curriculum_cli.suite_main([str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "schedule history" in out and "competence by factor" in out and "interfering pairs" in out
    assert "a vs b" in out


def test_cli_suite_relations_curriculum_handles_no_schedule_yet(tmp_path, capsys):
    rc = curriculum_cli.suite_main([str(tmp_path)])
    assert rc == 0
    assert "no decisions recorded" in capsys.readouterr().out


def test_cli_registers_top_level_steer_command():
    """`rrp steer` is a genuine top-level command (docs/relations.md 5.5: "rrp steer <run> <op> ..."), registered
    the same way `run-dag` is (cli.dag): via its own `register(sub)`, not a `(group, name)` TOOLS entry."""
    import argparse
    import importlib
    # `rrp.cli.__init__` re-exports `main` (the function) as its own `main` attribute, shadowing the `rrp.cli.main`
    # submodule on the package object -- so `import rrp.cli.main as x` (attribute access) gets the function, not the
    # submodule; `importlib.import_module` goes through `sys.modules` and gets the real submodule.
    cli_main = importlib.import_module("rrp.cli.main")
    p = cli_main.build_parser()
    sub = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction))
    assert "steer" in sub.choices


def test_suite_relations_curriculum_is_a_tool_entry():
    from rrp.cli.tools import TOOLS
    assert ("suite", "relations-curriculum") in TOOLS
    mod, fn = TOOLS[("suite", "relations-curriculum")][0].split(":")
    assert mod == "rrp.cli.curriculum" and fn == "suite_main"


def test_suite_relations_compare_tables_depth_and_paired_interference(tmp_path, capsys):
    root = tmp_path / "runs"

    def jl(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    for s, seed in (("base", 1), ("geo", 1)):
        run = root / f"relations-{s}" / f"train_flow_s{seed}"
        jl(run / "train_log.jsonl", [{"step": 20000, "flow": 0.5, "relgen": 0.1}])
        if s == "geo":
            jl(run / "schedule.jsonl", [{"step": 0, "level": {"geo.depth3d": 1}, "signals": {"geo.depth3d": {"competence": 0.2}}},
                                        {"step": 1000, "level": {"geo.depth3d": 2}, "signals": {"geo.depth3d": {"competence": 0.9}}}])
        ok = {"base": [1, 1, 0, 0], "geo": [1, 1, 1, 0]}[s]
        jl(root / f"relations-{s}" / "eval_r2_s1" / "bodyA" / "generated_dev_s1.jsonl",
           [{"seed": 10 + i, "privileged_success": bool(v)} for i, v in enumerate(ok)])
    curriculum_cli.compare_main(["--root", str(root), "--sets", "geo", "--seeds", "1"])
    t = json.loads((root / "tables" / "tables.json").read_text())
    assert {(r["factor"], r["depth"]) for r in t["competence_by_depth"]} == {("geo.depth3d", 1), ("geo.depth3d", 2)}
    pooled = [r for r in t["vs_base"] if r["body"] == "pooled"][0]
    assert pooled["n_pairs"] == 4 and pooled["set_only"] == 1 and pooled["base_only"] == 0
    assert pooled["rate_set"] == 0.75 and pooled["rate_base"] == 0.5 and pooled["delta"]["mean"] == 0.25
    assert "competence by factor" in capsys.readouterr().out
