"""D-126 legged/physics options (#11, #13, #14, #15): every option is additive and default-off.

The golden test pins the DEFAULT reward + observation streams of the tracker env to values computed with the pre-D-126 code
(origin/main 4181111, same seeds and actions), so a default-config tracker episode is proven unchanged.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _stream(body, contact, ov=None, actuator="v1", steps=40, **kw):
    from rrp.bodies.legged import legged_body
    from rrp.envs.mujoco.legged_core import LeggedEnv
    env = LeggedEnv(lambda: legged_body(body), 2, 5, contact=contact, reward_overrides=ov, actuator=actuator, **kw)
    rng = np.random.default_rng(0)
    R, O = [], []
    for _ in range(steps):
        o, p, r, d, tm = env.step(rng.normal(0, 0.3, (2, env.b.n)))
        R.append(r.copy())
        O.append(o.copy())
    h = hashlib.sha256(np.round(np.array(R), 8).tobytes() + np.round(np.array(O), 5).tobytes()).hexdigest()[:16]
    return h, env


# computed with origin/main 4181111 (pre-D-126) on the host, mujoco 3.14.0
GOLDEN = {"hexapod6_v1": "102eabf110fa8abd"}
GOLDEN_MENAGERIE = {"go2_v2": "ccab0059741a5dd4", "t1_v2_turn": "5eaab8329c9a5e40", "t1_v2_v1lat": "e04b85f404195529"}


def _need_314():
    import mujoco
    if mujoco.__version__ != "3.14.0":
        pytest.skip(f"golden hashes were computed with mujoco 3.14.0, not {mujoco.__version__}")


def test_default_tracker_env_is_byte_identical_procedural():
    _need_314()
    assert _stream("hexapod6", "v1")[0] == GOLDEN["hexapod6_v1"]


@pytest.mark.menagerie
def test_default_tracker_env_is_byte_identical_menagerie():
    _need_314()
    assert _stream("go2", "v2")[0] == GOLDEN_MENAGERIE["go2_v2"]
    turn = {"turn_lin": 1.5, "yaw_lin_all": 1.0, "sigma_ang": 0.03, "clearance_floor": -2.0}
    assert _stream("t1", "v2", turn)[0] == GOLDEN_MENAGERIE["t1_v2_turn"]
    assert _stream("t1", "v2", None, "v1lat")[0] == GOLDEN_MENAGERIE["t1_v2_v1lat"]
    assert _stream("t1", "v2", None, "ideal")[0] == _stream("t1", "v2", None, "v1")[0]     # "ideal" is the v1 alias


def test_reward_defaults_record_nothing_new():
    from rrp.envs.mujoco.legged_core import D126_DEFAULTS, RewardCfg
    for kind in ("humanoid", "quadruped", "hexapod"):
        for v in ("gait_v1", "gait_v2"):
            c = RewardCfg.for_kind(kind, v)
            assert c.options() == {}
            assert not set(c.weights()) & set(D126_DEFAULTS)
    c = RewardCfg.for_kind("humanoid", "gait_v2")
    from dataclasses import replace
    c2 = replace(c, yaw_progress_cap=1.0, ref_gait="clock")
    assert c2.options() == {"yaw_progress_cap": 1.0, "ref_gait": "clock"} and c2.weights()["yaw_progress_cap"] == 1.0


def test_limit_margin_aggregation():
    from rrp.envs.mujoco.legged_core import limit_margin_penalty
    lo, hi = np.zeros(4), np.ones(4)
    q = np.array([0.5, 0.5, 0.5, 0.0])          # one joint AT its limit: per-joint penalty 1
    assert limit_margin_penalty(q, lo, hi) == pytest.approx(0.25)
    assert limit_margin_penalty(q, lo, hi, agg="max") == pytest.approx(1.0)
    assert limit_margin_penalty(q, lo, hi, agg="sum") == pytest.approx(1.0)
    q2 = np.array([0.5, 0.5, 0.5, -0.02])        # 2% past: (0.04/0.02)^2 = 4
    assert limit_margin_penalty(q2, lo, hi, agg="max") == pytest.approx(4.0)
    with pytest.raises(ValueError):
        limit_margin_penalty(q, lo, hi, agg="median")


def test_clock_lift_targets_follow_the_contact_phase_convention():
    from rrp.envs.mujoco.legged_core import clock_lift_targets
    l, r = clock_lift_targets(0.25, 0.08)        # right swings in (0, 0.5)
    assert l == 0 and r == pytest.approx(0.08)
    l, r = clock_lift_targets(0.75, 0.08, frac=0.5)
    assert l == pytest.approx(0.04) and r == 0
    assert np.allclose(clock_lift_targets(0.0, 0.08), 0)


def test_yaw_progress_cap_is_wired():
    """With the cap at -0.5 the turn_lin term is the constant -0.5 x turn_lin on every turning command (yaw_lin_all)."""
    base = {"turn_lin": 1.0, "yaw_lin_all": 1.0}
    from rrp.bodies.legged import legged_body
    from rrp.envs.mujoco.legged_core import LeggedEnv
    envs = [LeggedEnv(lambda: legged_body("pquad4"), 2, 3, contact="v1", reward_overrides=ov, push=False)
            for ov in ({"yaw_progress_cap": -0.5, "turn_lin": 1.0, "yaw_lin_all": 1.0}, {"turn_lin": 0.0})]
    rng = np.random.default_rng(1)
    for _ in range(15):
        a = rng.normal(0, 0.2, (2, envs[0].b.n))
        turning = np.abs(envs[0].cmd[:, 2]) > 0.05
        assert np.allclose(envs[0].cmd, envs[1].cmd)
        r0 = envs[0].step(a)[2]
        r1 = envs[1].step(a)[2]
        assert np.allclose(r0 - r1, -0.5 * turning, atol=1e-9)
    assert base  # (documentation: the default cap 1.2 is covered by the golden streams)


def test_terrain_curriculum_gate():
    from rrp.harness.train.reward_schedule import TerrainCurriculum
    t = TerrainCurriculum(amp_max=0.1, warmup=10, after_alpha=0.5)
    assert t.update(5, dict(fall_rate=0.0, track_rel_err=0.1), alpha=1.0) == "skip"
    assert t.update(20, dict(fall_rate=0.0, track_rel_err=0.1), alpha=0.2) == "hold"        # alpha gate not reached
    assert t.update(25, dict(fall_rate=0.0, track_rel_err=0.1), alpha=0.6) == "advance" and t.level == pytest.approx(0.1)
    assert t.amp_m == pytest.approx(0.01)
    assert t.update(50, dict(fall_rate=0.3), alpha=0.6) == "backoff" and t.level == 0.0       # gait_v1 window: fall rate only
    assert t.state()["version"] == "terrain_curriculum_v1"


def test_terrain_env_rescales_the_heightfield_at_run_time():
    from rrp.bodies.legged import legged_body
    from rrp.envs.mujoco.legged_core import LeggedEnv
    env = LeggedEnv(lambda: legged_body("pquad4"), 2, 3, contact="v2", push=False,
                    terrain=dict(amp_max=0.08, seed=4, frac=0.5, half_m=4.0))
    assert env.meta["terrain"]["version"] == "bumps_v1" and env.b.floor2 >= 0
    assert env.terrain_amp == 0.0
    assert env.set_terrain_amp(1.0) == pytest.approx(0.04)
    assert env.model.hfield_size[env.hfield_id, 2] == pytest.approx(0.04)
    for _ in range(5):
        env.step(np.zeros((2, env.b.n)))
    flat = LeggedEnv(lambda: legged_body("pquad4"), 1, 3, contact="v2", push=False)
    assert flat.b.floor2 == -1 and "terrain" not in flat.meta


def test_ref_gait_clock_is_biped_only():
    from rrp.bodies.legged import legged_body
    from rrp.envs.mujoco.legged_core import LeggedEnv
    with pytest.raises(ValueError, match="bipeds"):
        LeggedEnv(lambda: legged_body("pquad4"), 1, 0, contact="v2", reward_overrides={"ref_gait": "clock", "ref_lift": 1.0})


@pytest.mark.menagerie
def test_ref_gait_clock_steps_on_a_biped():
    h, env = _stream("t1", "v2", {"ref_gait": "clock", "ref_lift": 1.0, "ref_contact": 0.5}, steps=10)
    h0, _ = _stream("t1", "v2", None, steps=10)
    assert h != h0 and env.cfg.options()["ref_gait"] == "clock"


def test_actuator_mode_resolution_and_speed_sources(monkeypatch):
    from rrp.bodies import actuator as A
    monkeypatch.delenv("RRP_ACTUATOR_MODE", raising=False)
    assert A.resolve_mode() == "ideal" == A.ACTUATOR_MODE_DEFAULT and A.legacy_mode_name(None) == "v1"
    assert A.resolve_mode("v1") == "ideal" and A.legacy_mode_name("v2") == "v2"
    monkeypatch.setenv("RRP_ACTUATOR_MODE", "v1lat")
    assert A.resolve_mode() == "v1lat"
    with pytest.raises(ValueError):
        A.resolve_mode("v3")
    assert set(A.speed_sources("t1", ["Left_Knee_Pitch"]).values()) == {"sourced_urdf"}
    assert set(A.speed_sources("pquad4", ["leg0_knee"]).values()) == {"estimate"}
    monkeypatch.setenv("RRP_ACTUATOR_LATENCY_MS", "12")
    assert A.resolve_latency_ms(3) == 12.0
    monkeypatch.delenv("RRP_ACTUATOR_LATENCY_MS")
    assert A.resolve_latency_ms(3) == A.resolve_latency_ms(3) and 0 <= A.resolve_latency_ms(3) <= 30


def test_legged_session_actuator_mode(monkeypatch):
    monkeypatch.delenv("RRP_ACTUATOR_MODE", raising=False)
    from rrp.core.action import NativeCommand
    from rrp.envs.mujoco.legged import LeggedSession, build_waypoint_contact
    ideal = LeggedSession(build_waypoint_contact("pquad4", 3), tracker_kind="cpg", seed=3)
    assert ideal.actuator_model is None and ideal.actuator_record() is None
    s = LeggedSession(build_waypoint_contact("pquad4", 3), tracker_kind="cpg", seed=3, actuator_mode="v1lat",
                      actuator_latency_ms=20.0)
    rec = s.actuator_record()
    assert rec["actuator_mode"] == "v1lat" and rec["latency_ms"] == 20.0 and rec["speed_source"] == "estimate"
    cmd = lambda: NativeCommand(controller_version=s.controller_version(), groups={"base_velocity": [0.2, 0.0, 0.1]},
                                source="scripted_teacher")
    for _ in range(3):
        s.step(cmd())
    snap = s.snapshot()
    s.step(cmd())
    q1 = s.data.qpos.copy()
    s.restore(snap)
    s.step(cmd())
    assert np.allclose(s.data.qpos, q1)
    # the ideal session differs (latency), i.e. the actuator is really in the loop
    for _ in range(4):
        ideal.step(NativeCommand(controller_version=ideal.controller_version(), groups={"base_velocity": [0.2, 0.0, 0.1]},
                                 source="scripted_teacher"))
    assert not np.allclose(ideal.data.qpos, q1)


def _ctx(tmp_path, stage="eval_r2", **opts):
    from rrp.core.runconfig import RunConfig, RunIndex
    from rrp.harness.pipelines.base import StageContext
    rc = RunConfig.model_validate(dict(
        schema_version="runconfig-1", family="legged", stage=stage, variant="semfix", seed=0, lineage="l", track="t",
        flags=dict(zero_prev_action=None, realizer_anchor=None, realizer_drop_qd=None, probe_lv_min=None, qd_dropout=None,
                   contact_version="contact_v2"), options=dict(body="go2", **opts)))
    return StageContext(rc=rc, index=RunIndex(), root=tmp_path)


def test_pipeline_threads_the_declared_actuator_mode(tmp_path, monkeypatch):
    monkeypatch.delenv("RRP_ACTUATOR_MODE", raising=False)
    from rrp.harness.pipelines.base import StageError
    from rrp.harness.pipelines.legged import check_rows_contact, physics_env
    e0 = physics_env(_ctx(tmp_path))
    assert "RRP_ACTUATOR_MODE" not in e0 and e0["RRP_CONTACT_MODEL"] == "contact_v2"
    c = _ctx(tmp_path, actuator_mode="v1lat", actuator_latency_ms=15)
    e = physics_env(c)
    assert e["RRP_ACTUATOR_MODE"] == "v1lat" and e["RRP_ACTUATOR_LATENCY_MS"] == "15.0"
    ok = dict(seed=1, contact_version="contact_v2", actuator_mode=dict(actuator_mode="v1lat"))
    check_rows_contact(c, [ok], "rows")
    with pytest.raises(StageError, match="actuator mode"):
        check_rows_contact(c, [dict(seed=1, contact_version="contact_v2")], "rows")      # an ideal row under a v1lat DAG
    check_rows_contact(_ctx(tmp_path), [dict(seed=1, contact_version="contact_v2")], "rows")   # undeclared: as before


def test_tracker_recipes_parse_and_pin_their_initial_actors():
    from rrp.harness.train import tracker_training as T
    from rrp.harness.train.tracker_recipes import CPU_RECIPES as TRACKER_RECIPES
    seen = {}
    T_train = T.train
    try:
        T.train = lambda a: seen.setdefault(a.body, (a, T.reward_overrides(a)))
        for name, rec in TRACKER_RECIPES.items():
            T.main(["--recipe", name, "--out", "/nonexistent/x"])
            if rec.get("init_actor"):
                assert len(rec["init_actor_sha256"]) == 64, name
    finally:
        T.train = T_train
    a, ov = seen["h1"]
    assert a.init_actor is None and ov["ref_gait"] == "clock" and a.recipe_record["name"] == "h1_clock_scratch"
    a, ov = seen["g1"]
    assert ov["yaw_progress_cap"] == 1.0 and ov["yaw_overshoot"] == -1.0
    assert seen["t1"][0].actuator == "v1lat" and seen["anymal_c"][0].terrain_curriculum == "gated"
    T.train = lambda a: seen.__setitem__("default", (a, T.reward_overrides(a)))
    try:
        T.main(["--body", "go2", "--out", "/nonexistent/x"])
    finally:
        T.train = T_train
    a, ov = seen["default"]
    assert ov == {} and a.actuator == "v1" and a.terrain_curriculum == "off" and not getattr(a, "recipe_record", None)
    meta = T.d126_meta(a, dict(), None)
    assert meta == {}


def test_tracker_recipe_dags_plan():
    from rrp.harness.dag import load_dag, plan_dag
    from rrp.harness.train.tracker_recipes import CPU_RECIPES as TRACKER_RECIPES
    for f in sorted((ROOT / "recipes/humanoid").glob("d126_tracker_*.yaml")):
        p = plan_dag(load_dag(f), source="t")
        tr = p.nodes["train"]
        assert tr.rc.stage == "train_tracker" and tr.rc.options["recipe"] in TRACKER_RECIPES and tr.placement == "peer"
        assert TRACKER_RECIPES[tr.rc.options["recipe"]]["body"] == tr.rc.options["body"]
        v = p.nodes["validate"]
        assert v.rc.stage == "validate_tracker" and "train" in v.deps and v.rc.inputs["actor"].endswith(":actor.pt")


def test_heldout_template_keeps_the_heldout_body_out_of_training():
    from rrp.harness.dag import load_dag, plan_dag
    p = plan_dag(load_dag(ROOT / "recipes/templates/legged_heldout.yaml"), source="t")
    nodes = p.nodes
    h = "anymal_c"
    for nid, n in nodes.items():
        assert n.placement == "peer", nid
        if n.rc.stage in ("train_rep",) or nid == "bc_zs":
            assert h not in n.rc.params["bodies"], nid
        if n.rc.stage == "heldout":
            assert n.rc.options["body"] == h and h not in n.rc.options["train_bodies"], nid
        if n.rc.stage == "collect":
            assert {"validate_a", "validate_b", "validate_h"} <= set(n.deps), nid
    assert h in nodes["bc_adapt"].rc.params["bodies"]
    for v in ("semfix", "nosem"):
        rf = nodes[f"refit_h@{v}.s0"]
        assert h in rf.rc.params["bodies"] and "collect_h" in rf.deps and rf.rc.params["dagger_present_bodies"] is True
        ad = nodes[f"heldout_refit@{v}.s0"]
        assert ad.rc.stage == "eval_r2" and ad.rc.tag == "adapt" and "realizer" in ad.rc.inputs     # ADAPTED, not zero-shot
    # the same H budget feeds both adapted methods (fair acquisition accounting)
    assert nodes["collect_h"].rc.options["seeds"] == "0-49"
    assert nodes["validate_h"].rc.options["tracker_sha256"] == "SET_IN_CHILD_DAG"
