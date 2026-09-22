"""RobotSpec: typed morphology/controller graph. Static structure only (no dynamic state).

Node identity is a structural address derived from the kinematic tree (e.g. `r/0/1`),
never a learned robot-name table. Names are presentation metadata only.
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .base import Strict, content_hash

JointType = Literal["hinge", "slide", "ball", "free", "fixed"]
AssemblyKind = Literal["arm", "hand", "gripper", "leg", "base", "torso", "head", "tool", "wheel_base", "body"]
Capability = Literal["grasp", "support", "push", "locomote", "observe", "carry", "insert", "mobile_base"]


class LinkSpec(Strict):
    address: str
    name: str
    parent_joint: str | None
    mass: float = Field(gt=0)
    inertia_diag: list[float] = Field(min_length=3, max_length=3)
    com: list[float] = Field(min_length=3, max_length=3)
    pos_in_parent: list[float] = Field(min_length=3, max_length=3)
    quat_in_parent_wxyz: list[float] = Field(min_length=4, max_length=4)
    geom_features: list[float] = Field(default_factory=list)  # e.g. [n_geoms, bbox_x, bbox_y, bbox_z, volume]

    @model_validator(mode="after")
    def _positive_inertia(self):
        if any(i <= 0 for i in self.inertia_diag):
            raise ValueError(f"link {self.name}: inertia must be positive")
        return self


class JointSpec(Strict):
    address: str
    name: str
    type: JointType
    parent_link: str | None
    child_link: str
    axis: list[float] = Field(min_length=3, max_length=3)
    range: list[float] | None = None      # radians or meters; None = unlimited
    damping: float = 0.0
    armature: float = 0.0
    qpos_width: int = Field(ge=0)
    qvel_width: int = Field(ge=0)
    mimic_of: str | None = None           # equality/mimic constraint -> not independently actuated


class ActuatorSpec(Strict):
    address: str
    name: str
    kind: Literal["position", "velocity", "motor", "general"]
    joint: str | None
    tendon: str | None = None
    gear: float = 1.0
    ctrl_range: list[float] | None = None
    force_range: list[float] | None = None
    kp: float | None = None


class SensorSpec(Strict):
    address: str
    name: str
    kind: Literal["camera", "joint_pos", "joint_vel", "force", "torque", "touch", "imu", "gripper_width"]
    mount_link: str | None
    width: int = Field(ge=1)
    units: str
    rate_hz: float = Field(gt=0)


class FrameDef(Strict):
    """Explicit aggregate frame (palm/TCP/root), never an averaged rotation."""
    link: str
    pos: list[float] = Field(min_length=3, max_length=3)
    quat_wxyz: list[float] = Field(min_length=4, max_length=4)
    site: str | None = None


class AssemblySpec(Strict):
    id: str
    kind: AssemblyKind
    members: list[str]            # link/joint addresses
    frame: FrameDef
    capabilities: list[Capability]
    parent_assembly: str | None = None


class AttachmentPort(Strict):
    id: str
    host_link: str
    pos: list[float] = Field(min_length=3, max_length=3)
    quat_wxyz: list[float] = Field(min_length=4, max_length=4)
    max_payload_kg: float = Field(gt=0)
    max_module_extent_m: float = Field(gt=0)
    interface: Literal["flange_iso9409", "wrist_generic", "mount_plate", "leg_socket"]
    allowed_module_kinds: list[AssemblyKind]
    collision_exemptions: list[str] = Field(default_factory=list)
    occupied_by: str | None = None


class CommandGroup(Strict):
    name: str
    width: int = Field(ge=1)
    units: Literal["rad", "m", "rad/s", "m/s", "N", "Nm", "normalized", "mixed"]
    semantic: Literal["joint_position", "joint_velocity", "joint_torque", "gripper", "base_velocity",
                      "wholebody_command", "ee_pose"]
    actuators: list[str]                   # addresses owned by this group (exclusive)
    lower: list[float]
    upper: list[float]
    hold: Literal["hold_last", "hold_measured", "zero_velocity", "damp"] = "hold_measured"

    @model_validator(mode="after")
    def _widths(self):
        if len(self.lower) != self.width or len(self.upper) != self.width:
            raise ValueError(f"group {self.name}: bounds width mismatch")
        if any(l > u for l, u in zip(self.lower, self.upper)):
            raise ValueError(f"group {self.name}: lower > upper")
        return self


class ControllerContract(Strict):
    id: str
    version: str
    kind: Literal["joint_targets", "psi0_original", "psi0_sonic", "legged_tracker", "base_velocity", "scripted"]
    rate_hz: float = Field(gt=0)
    command_groups: list[CommandGroup]
    state_required: list[str]
    body_specific: bool = True
    validated_bodies: list[str] = Field(default_factory=list)   # spec hashes validated with this controller

    @model_validator(mode="after")
    def _exclusive_ownership(self):
        seen = set()
        for g in self.command_groups:
            for a in g.actuators:
                if a in seen:
                    raise ValueError(f"actuator {a} owned by more than one command group")
                seen.add(a)
        return self

    def total_width(self) -> int:
        return sum(g.width for g in self.command_groups)


class TypedEdge(Strict):
    src: str
    dst: str
    type: Literal["parent_of", "actuates", "mimics", "senses", "member_of", "attached_at", "coupled"]


class RobotSpec(Strict):
    schema_version: Literal["1.0"] = "1.0"
    spec_hash: str = ""
    name: str
    family: Literal["humanoid", "quadruped", "biped", "hexapod", "multipod", "arm", "dual_arm", "mobile_manipulator",
                    "end_effector", "fixture"]
    asset_source: dict
    lineage: list[str]                      # body/module lineage ids for split control
    geometry_hashes: list[str] = Field(default_factory=list)
    floating_base: bool
    links: list[LinkSpec]
    joints: list[JointSpec]
    actuators: list[ActuatorSpec]
    sensors: list[SensorSpec]
    assemblies: list[AssemblySpec]
    attachment_ports: list[AttachmentPort]
    controller_contracts: list[ControllerContract]
    typed_edges: list[TypedEdge]
    capability_tags: list[str]
    synthetic: bool = False

    @model_validator(mode="after")
    def _integrity(self):
        link_addr = {l.address for l in self.links}
        joint_addr = {j.address for j in self.joints}
        for j in self.joints:
            if j.child_link not in link_addr or (j.parent_link and j.parent_link not in link_addr):
                raise ValueError(f"joint {j.name} references unknown link")
            if j.mimic_of and j.mimic_of not in joint_addr:
                raise ValueError(f"joint {j.name} mimics unknown joint")
        act_addr = {a.address for a in self.actuators}
        for a in self.actuators:
            if a.joint and a.joint not in joint_addr:
                raise ValueError(f"actuator {a.name} drives unknown joint")
        for c in self.controller_contracts:
            for g in c.command_groups:
                for a in g.actuators:
                    if a not in act_addr:
                        raise ValueError(f"controller {c.id} group {g.name} owns unknown actuator {a}")
        mimicked = {j.address for j in self.joints if j.mimic_of}
        for a in self.actuators:
            if a.joint in mimicked:
                raise ValueError(f"actuator {a.name} drives a mimic (dependent) joint")
        return self

    def with_hash(self) -> "RobotSpec":
        """Physical/structural identity: names, sites, lineage and provenance are excluded
        (they are presentation/split metadata), so renaming never changes identity."""
        def strip(x):
            if isinstance(x, dict):
                return {k: strip(v) for k, v in x.items()
                        if k not in ("name", "site", "spec_hash", "lineage", "asset_source", "occupied_by")}
            if isinstance(x, list):
                return [strip(v) for v in x]
            return x
        self.spec_hash = content_hash(strip(self.model_dump(mode="json")))
        return self

    def independent_controls(self) -> int:
        return len(self.actuators)

    def generalized_coordinates(self) -> int:
        return sum(j.qpos_width for j in self.joints)
