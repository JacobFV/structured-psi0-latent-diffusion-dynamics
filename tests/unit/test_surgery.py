import pytest
from rrp.morphology.fixtures import arm_with_port, gripper_module, three_finger_module
from rrp.morphology.surgery import attach, AttachmentError


def test_attachment_rebinds_names_and_preserves_positive_mass():
    a = attach(arm_with_port(), gripper_module(), port_id="wrist")
    assert len(a.joint_names) == len(set(a.joint_names))
    assert all(link.mass > 0 for link in a.rigid_links)
    with pytest.raises(AttachmentError):
        attach(arm_with_port(), gripper_module(), port_id="absent")


def test_mimic_joints_are_not_independent_controls():
    a = attach(arm_with_port(), gripper_module(), port_id="wrist")
    rs = a.robot_spec
    mimic = [j for j in rs.joints if j.mimic_of]
    assert len(mimic) == 1
    assert rs.independent_controls() == 6          # 5 arm + 1 gripper actuator
    assert rs.generalized_coordinates() == 7       # 5 arm + 2 finger slides
    groups = {g.name: g for g in rs.controller_contracts[0].command_groups}
    assert groups["arm"].width == 5 and groups["gripper"].width == 1
    tf = attach(arm_with_port(), three_finger_module(), port_id="wrist").robot_spec
    assert sum(1 for j in tf.joints if j.mimic_of) == 2 and tf.independent_controls() == 6


def test_different_modules_give_fresh_hashes_and_lineage():
    a = attach(arm_with_port(), gripper_module(), port_id="wrist").robot_spec
    b = attach(arm_with_port(), three_finger_module(), port_id="wrist").robot_spec
    c = attach(arm_with_port(), gripper_module(), port_id="wrist").robot_spec
    assert a.spec_hash != b.spec_hash and a.spec_hash == c.spec_hash
    assert any("procedural_gripper" in l for l in a.lineage)
    assert {x.kind for x in a.assemblies} == {"arm", "gripper"}


def test_structural_addresses_are_name_independent():
    a = attach(arm_with_port(name="armA"), gripper_module(name="gA"), port_id="wrist").robot_spec
    b = attach(arm_with_port(name="armB"), gripper_module(name="gB"), port_id="wrist").robot_spec
    assert [j.address for j in a.joints] == [j.address for j in b.joints]
    assert a.spec_hash == b.spec_hash   # presentation names do not change identity


def test_payload_kind_and_occupancy_checks():
    heavy = gripper_module(density=200000.0)
    with pytest.raises(AttachmentError, match="payload"):
        attach(arm_with_port(), heavy, port_id="wrist")
    once = attach(arm_with_port(), gripper_module(), port_id="wrist")
    with pytest.raises(AttachmentError, match="occupied"):
        attach(once, gripper_module(), port_id="wrist")
    m = gripper_module()
    m.meta["module_kind"] = "leg"
    with pytest.raises(AttachmentError, match="not allowed"):
        attach(arm_with_port(), m, port_id="wrist")


def test_physics_validation_reports_stability():
    a = attach(arm_with_port(), gripper_module(), port_id="wrist")
    v = a.validation
    assert v["ok"] and v["hold_max_qvel"] < 5 and v["max_initial_penetration"] < 0.01


def test_validator_detects_self_collision_with_world_welded_parent():
    import mujoco
    from rrp.morphology.surgery import validate_physics
    from rrp.morphology.generators import procedural_arm, ArmParams
    m = procedural_arm(ArmParams())
    m.spec.excludes[0].delete() if hasattr(m.spec.excludes[0], "delete") else m.spec.delete(m.spec.excludes[0])
    v = validate_physics(m.spec.compile(), steps=10)
    assert not v["ok"] and any("penetration" in f for f in v["failures"])
