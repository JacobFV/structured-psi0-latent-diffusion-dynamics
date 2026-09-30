"""armdiv (D-137): procedural arm generator v2, additional menagerie adapters, additive workbench keys."""
import numpy as np
import pytest

from rrp.bodies.generators_v2 import sample_arm_v2, procedural_arm_v2, solve_home, ArmGenerationError


def test_v2_sampling_is_deterministic_and_in_range():
    for s in range(30):
        a, b = sample_arm_v2(s), sample_arm_v2(s)
        assert a == b
        assert 5 <= len(a.axes) <= 8
        assert a.axes[0] == "z" and a.axes[1] == "y" and a.axes[-1] == "z" and a.axes[-2] in ("x", "y")
        assert len(a.lengths) == len(a.axes) == len(a.twists) == len(a.offsets) == len(a.ranges)
        assert 0.0 <= a.pedestal <= 0.15


def test_v2_draws_are_diverse():
    draws = [sample_arm_v2(s) for s in range(40)]
    assert len({len(d.axes) for d in draws}) >= 3
    assert len({d.axes for d in draws}) >= 10
    assert any(any(t != 0 for t in d.twists) for d in draws)
    assert any(any(o != (0.0, 0.0) for o in d.offsets) for d in draws)


def test_v2_home_reaches_tool_down_ready_pose():
    from rrp.bodies.generators_v2 import _build_spec, READY_FLANGE_TARGETS
    from rrp.bodies.ik import IKSolver
    ok = 0
    for s in range(8):
        p = sample_arm_v2(s)
        try:
            home = solve_home(p)
        except ArmGenerationError:
            continue
        ok += 1
        spec, joints = _build_spec(p)
        m = spec.compile()
        ik = IKSolver(m, "flange_tcp", joints)
        pos, R = ik.fk(m.qpos0.copy(), np.array(home))
        assert min(np.linalg.norm(pos - np.array(t)) for t in READY_FLANGE_TARGETS) < 4e-3
        assert R[2, 2] < -0.99                                   # flange z points down
        mod = procedural_arm_v2(p)
        assert mod.meta["synthetic"] is True and mod.meta["lineage"][0] == "procedural_arm_family/v2"
    assert ok >= 4


def test_legacy_workbench_keys_unchanged():
    from rrp.bodies.catalog import workbench_robots
    w = workbench_robots()
    assert w["parm5_pg2"]().robot_spec.spec_hash == "f88c4d7d34000d6d"
    assert w["parm7_tf3"]().robot_spec.spec_hash == "ea5a7499706e0d7f"


def test_armdiv_keys_build():
    from rrp.bodies.catalog import workbench_robots
    w = workbench_robots()
    assert "pa2s3_pg2" in w and "pa2s3_tf3" in w and "pa2s900000_pg2" in w
    r = w["pa2s3_tf3"]()
    assert len(r.meta["home"]) == len(sample_arm_v2(3).axes)


@pytest.mark.menagerie
def test_menagerie_v2_adapters_build_and_legacy_hash():
    from rrp.bodies.catalog import workbench_robots
    w = workbench_robots()
    if "gen3_pg2" not in w:
        pytest.skip("menagerie assets absent")
    assert w["xarm7_pg2"]().robot_spec.spec_hash == "9c80c56a324b6555"
    for k in ("gen3_pg2", "ur10e_tf3", "vx300s_pg2"):
        r = w[k]()
        grp = [g for c in r.robot_spec.controller_contracts for g in c.command_groups]
        assert [g.name for g in grp] == ["arm", "gripper"]
        # continuous joints received a declared range (IK clips to ranges)
        for j in r.robot_spec.joints:
            if j.type == "hinge":
                assert j.range is not None and j.range[1] > j.range[0]


def test_armdiv_sealed_guard():
    import json
    from rrp.bodies.armdiv import is_sealed_target
    split = json.load(open("research/splits/armdiv_v1.json"))
    declared = [r for t in split["targets"].values() for r in t["robots"]]
    assert all(is_sealed_target(k) for k in declared)
    for k in ("gen3_pg2", "gen3_tf3", "rizon4_pg2", "iiwa14_tf3", "pa2s900002_pg2", "pa2s900011_tf3", "xarm7_pg2"):
        assert is_sealed_target(k)
    for k in ("pa2s0_pg2", "ur10e_tf3", "vx300s_pg2", "parm6_tf3"):
        assert not is_sealed_target(k)
    pool = json.load(open("research/splits/armdiv_pool_v1.json"))
    assert not any(is_sealed_target(k) for k in pool["train_robots"])
    from rrp.harness.eval.ladder_cli import main
    with pytest.raises(SystemExit, match="target bodies"):
        main(["--route", "learned", "--robot", "gen3_pg2", "--n", "1", "--seed-start", "3000000", "--tag", "x",
              "--out", "/tmp/never", "--policy", "none.pt"])


def test_gate_treats_v2_procedural_arms_like_parm():
    from rrp.harness.eval.gates import _procedural
    assert _procedural("pa2s3_tf3") and _procedural("parm6_pg2")
    assert not _procedural("ur10e_pg2") and not _procedural("gen3_pg2")
