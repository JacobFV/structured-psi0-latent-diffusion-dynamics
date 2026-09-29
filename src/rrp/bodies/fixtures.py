"""Small real model records for tests and the first vertical slice."""
from __future__ import annotations

from rrp.bodies.generators import ArmParams, GripperParams, procedural_arm, gripper_module as _gripper


def arm_with_port(**kw):
    return procedural_arm(ArmParams(**kw))


def gripper_module(**kw):
    return _gripper(GripperParams(**kw))


def three_finger_module(**kw):
    kw.setdefault("name", "tf3")
    return _gripper(GripperParams(kind="three_finger", **kw))


def cw_pointer_spec(lo: list[float], hi: list[float], hover_z: float, n_keys: int, rate_hz: float = 10.0):
    """ComputerWorld `cw_pointer` (docs/architecture.md section 5): a base link at the screen origin, slide joints x, y
    over the viewport [lo, hi] (m) and a slide z fixed at the hover height; one `tool` assembly. Command groups:
    pointer (x, y), button, wheel, key (index into the env's key vocabulary of size n_keys)."""
    from rrp.core.robot import (ActuatorSpec, AssemblySpec, CommandGroup, ControllerContract, FrameDef, JointSpec,
                                LinkSpec, RobotSpec, TypedEdge)
    ident = [1.0, 0.0, 0.0, 0.0]

    def link(a, name, parent):
        return LinkSpec(address=a, name=name, parent_joint=parent, mass=0.01, inertia_diag=[1e-6] * 3, com=[0, 0, 0],
                        pos_in_parent=[0, 0, 0], quat_in_parent_wxyz=ident)
    links = [link("p", "screen", None), link("p/0", "carriage_x", "p/j0"), link("p/0/0", "carriage_y", "p/j1"),
             link("p/0/0/0", "tip", "p/j2")]
    joints = [JointSpec(address="p/j0", name="x", type="slide", parent_link="p", child_link="p/0", axis=[1, 0, 0],
                        range=[lo[0], hi[0]], qpos_width=1, qvel_width=1),
              JointSpec(address="p/j1", name="y", type="slide", parent_link="p/0", child_link="p/0/0", axis=[0, 1, 0],
                        range=[lo[1], hi[1]], qpos_width=1, qvel_width=1),
              JointSpec(address="p/j2", name="z", type="slide", parent_link="p/0/0", child_link="p/0/0/0",
                        axis=[0, 0, 1], range=[hover_z, hover_z], qpos_width=1, qvel_width=1)]
    acts = [ActuatorSpec(address="p/a0", name="x", kind="position", joint="p/j0", ctrl_range=[lo[0], hi[0]]),
            ActuatorSpec(address="p/a1", name="y", kind="position", joint="p/j1", ctrl_range=[lo[1], hi[1]]),
            ActuatorSpec(address="p/a2", name="button", kind="general", joint=None, ctrl_range=[0, 1]),
            ActuatorSpec(address="p/a3", name="wheel", kind="general", joint=None),
            ActuatorSpec(address="p/a4", name="key", kind="general", joint=None, ctrl_range=[-1, n_keys - 1])]
    contract = ControllerContract(
        id="cw_pointer", version="cw_pointer.v1", kind="scripted", rate_hz=rate_hz, state_required=["qpos"],
        command_groups=[CommandGroup(name="pointer", width=2, units="m", semantic="cartesian_position",
                                     actuators=["p/a0", "p/a1"], lower=lo, upper=hi, hold="hold_last"),
                        CommandGroup(name="button", width=1, units="normalized", semantic="button",
                                     actuators=["p/a2"], lower=[0], upper=[1], hold="hold_last"),
                        CommandGroup(name="wheel", width=1, units="normalized", semantic="discrete",
                                     actuators=["p/a3"], lower=[-10], upper=[10], hold="zero_velocity"),
                        CommandGroup(name="key", width=1, units="normalized", semantic="discrete",
                                     actuators=["p/a4"], lower=[-1], upper=[n_keys - 1], hold="zero_velocity")])
    tool = AssemblySpec(id="tool", kind="tool", members=["p/0/0/0", "p/j0", "p/j1", "p/j2"],
                        frame=FrameDef(link="p/0/0/0", pos=[0, 0, 0], quat_wxyz=ident), capabilities=["push"])
    edges = [TypedEdge(src=j.parent_link, dst=j.child_link, type="parent_of") for j in joints]
    return RobotSpec(name="cw_pointer", family="pointer", asset_source={"kind": "computerworld", "key": "cw_pointer"},
                     lineage=["cw_pointer"], floating_base=False, links=links, joints=joints, actuators=acts,
                     sensors=[], assemblies=[tool], attachment_ports=[], controller_contracts=[contract],
                     typed_edges=edges, capability_tags=["pointer"], synthetic=True).with_hash()
