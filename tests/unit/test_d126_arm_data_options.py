"""D-126 arm data options, all default OFF: limit-aware IK + teacher v2lim (#8), guarded descent-phase DART (#5),
richer task objects (#35). Red/green where a silent change would invalidate data: default paths unchanged, labels
bumped, options recorded."""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

ARM3 = """<mujoco><compiler angle="radian"/><worldbody>
<body name="l1"><joint name="j1" type="hinge" axis="0 0 1" range="-1.0 1.0"/><geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02"/>
<body name="l2" pos="0.3 0 0"><joint name="j2" type="hinge" axis="0 0 1" range="0 2.4"/><geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02"/>
<body name="l3" pos="0.3 0 0"><joint name="j3" type="hinge" axis="0 0 1" range="-2.0 2.0"/><geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02"/>
<site name="tip" pos="0.3 0 0"/></body></body></body></worldbody></mujoco>"""


def _ik():
    from rrp.bodies.ik import IKSolver
    m = mujoco.MjModel.from_xml_string(ARM3)
    return IKSolver(m, "tip", ["j1", "j2", "j3"]), np.zeros(m.nq)


def test_limit_aware_ik_keeps_margin_without_losing_reach():
    ik, q0 = _ik()
    rng = np.random.default_rng(1)
    plain_ok = lim_ok = plain_m = lim_m = 0
    for _ in range(80):
        r, a = rng.uniform(0.25, 0.85), rng.uniform(-0.8, 0.8)
        pos = np.array([r * np.cos(a), r * np.sin(a), 0.0])
        qi = rng.uniform(ik.lo, ik.hi)
        qa, ea = ik.solve(q0, qi, pos, None, iters=150, tol=2e-4)
        qa0, ea0 = ik.solve(q0, qi, pos, None, iters=150, tol=2e-4, limit_margin=0.0)
        assert np.array_equal(qa, qa0) and ea == ea0                 # margin 0 = the historical solve
        qb, eb = ik.solve(q0, qi, pos, None, iters=150, tol=2e-4, limit_margin=0.05)
        if ea < 2e-3:
            assert eb < 2e-3                                          # never loses a reachable target
        plain_ok += ea < 2e-3
        lim_ok += eb < 2e-3
        plain_m += ea < 2e-3 and ik.margin(qa) >= 0.02
        lim_m += eb < 2e-3 and ik.margin(qb) >= 0.02
    assert lim_ok >= plain_ok and lim_m >= plain_m + 8, (plain_ok, lim_ok, plain_m, lim_m)


def test_teacher_v2lim_is_a_version_bump():
    from rrp.core.provenance import parse_source
    from rrp.envs.mujoco.fixtures import make_pick_place_session
    from rrp.policies.teachers.arm_smooth import make_arm_teacher, teacher_source, TEACHER_VERSIONS
    assert TEACHER_VERSIONS["v2"] == "pick_place_v2_minjerk"
    assert parse_source(teacher_source("v2lim"), strict=True).detail == "pick_place_v2_minjerk_lim"
    s = make_pick_place_session(seed=3)
    with pytest.raises(ValueError, match="v2lim"):
        make_arm_teacher(s, "v2", ik_limit_margin=0.05)
    t2 = make_arm_teacher(s, "v2")
    assert t2._ikkw == {} and "ik_limit_margin" not in t2.diag
    tl = make_arm_teacher(make_pick_place_session(seed=3), "v2lim")
    assert tl._ikkw == {"limit_margin": 0.05} and tl.diag["version"] == "pick_place_v2_minjerk_lim"


def test_generate_options_default_off_and_checked():
    from rrp.harness.data.generate import _extra_options
    assert _extra_options({"items": [], "dart_safety": "phase", "teacher_version": "v2"}) is None
    with pytest.raises(ValueError):
        _extra_options({"dart_descent_sigma": 0.02, "dart_safety": "proximity"})
    ex = _extra_options({"dart_descent_sigma": 0.02, "dart_safety": "phase", "ik_limit_margin": 0.05,
                         "object_variation": {"shapes": ["cube", "cylinder"]}})
    assert ex == {"dart_descent_sigma": 0.02, "teacher_kw": {"ik_limit_margin": 0.05},
                  "object_variation": {"shapes": ["cube", "cylinder"]}}


def test_descent_dart_guarded_and_recorded():
    from rrp.harness.data.collect import collect_teacher_episode
    from rrp.envs.mujoco.fixtures import make_pick_place_session
    rec = collect_teacher_episode(make_pick_place_session(seed=2), max_steps=160, exec_noise=0.08, noise_seed=2,
                                  teacher_version="v2", dart_safety="phase", dart_descent_sigma=0.02)
    d = rec.public["meta"]["dart"]
    assert d["descent"]["sigma"] == 0.02 and d["descent"]["phases"] == ["descend"]
    desc = d["applied_by_phase"].get("descend", 0) + d["rejected_by_phase"].get("descend", 0) + \
        d.get("held_by_phase", {}).get("descend", 0)
    assert desc == d["ticks_by_phase"].get("descend", 0) > 0          # every descent tick went through the guard
    base = collect_teacher_episode(make_pick_place_session(seed=2), max_steps=40, exec_noise=0.08, noise_seed=2,
                                   teacher_version="v2", dart_safety="phase")
    assert "descent" not in base.public["meta"]["dart"]
    assert "descend" not in base.public["meta"]["dart"]["applied_by_phase"]      # v6dart behaviour: no descent noise
    with pytest.raises(ValueError):
        collect_teacher_episode(make_pick_place_session(seed=2), max_steps=5, exec_noise=0.08, teacher_version="v2",
                                dart_safety="proximity", dart_descent_sigma=0.02)


def _arrays(m):
    return {k: np.array(getattr(m, k)) for k in ("body_pos", "body_quat", "body_mass", "geom_size", "geom_friction",
                                                  "geom_type", "geom_pos")}


def test_object_spec_default_identical_and_variants(monkeypatch):
    from rrp.envs.mujoco.fixtures import fixture_robot
    from rrp.envs.mujoco.scenario import build_pick_place, sample_object_spec, OBJECT_VARIATION_ENV
    monkeypatch.delenv(OBJECT_VARIATION_ENV, raising=False)
    r = fixture_robot()
    a = build_pick_place(r, 11, n_distractors=2)
    b = build_pick_place(r, 11, n_distractors=2, object_spec={"shape": "cube"})
    assert "object_spec" not in a.meta
    for k, v in _arrays(a.model).items():                  # same RNG draws, placement and physics
        assert np.array_equal(v, _arrays(b.model)[k]), k
    monkeypatch.setenv("RRP_GRASP_CONTACT", "v2")          # object friction must survive the grasp contact model
    c = build_pick_place(r, 11, n_distractors=2, object_spec={"shape": "cylinder", "size": 0.02, "mass": 0.2,
                                                              "friction": 0.4})
    m = c.model
    assert m.geom_type[m.geom("cube_geom").id] == mujoco.mjtGeom.mjGEOM_CYLINDER
    assert abs(m.body_mass[m.body("cube").id] - 0.2) < 1e-9 and abs(m.geom_friction[m.geom("cube_geom").id][0] - 0.4) < 1e-9
    assert np.allclose(m.body_pos[m.body("cube").id][:2], a.model.body_pos[a.model.body("cube").id][:2])
    assert c.meta["object_spec"]["shape"] == "cylinder" and c.object("cube").descriptor == "red cube"
    d = build_pick_place(r, 11, object_spec={"shape": "box_tall", "descriptor_shape": True})
    assert d.object("cube").descriptor == "red box tall"
    assert next(e for e in d.task["entity_declarations"] if e["id"] == "cube")["descriptor"] == "red box tall"
    s1, s2 = sample_object_spec({"size": [0.015, 0.03], "shapes": ["cube", "cylinder"]}, 5), \
        sample_object_spec({"size": [0.015, 0.03], "shapes": ["cube", "cylinder"]}, 5)
    assert s1 == s2 and 0.015 <= s1["size"] <= 0.03
    monkeypatch.setenv(OBJECT_VARIATION_ENV, '{"mass": [0.1, 0.1]}')
    e = build_pick_place(r, 11)
    assert abs(e.model.body_mass[e.model.body("cube").id] - 0.1) < 1e-9
    with pytest.raises(ValueError):
        build_pick_place(r, 11, object_spec={"shape": "sphere"})
