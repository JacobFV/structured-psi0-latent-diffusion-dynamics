"""Public observation vs privileged truth. Separate types, separate buses, separate files.

PolicyObservation must never contain simulator object identity/poses, contacts, reward,
completion truth or labels. PrivilegedTruth is serialized independently and the public
transport refuses it (rrp.contracts.channels).
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import Strict, NDArray
from .refs import EntityRef
from .task import TaskDefinition


class NodeState(Strict):
    """Measured coordinates with masks, timestamps and units (dynamic, not morphology)."""
    joint_addresses: list[str]
    qpos: NDArray
    qvel: NDArray
    qpos_mask: NDArray
    effort: NDArray | None = None
    base_pose_estimate: NDArray | None = None     # [x y z qw qx qy qz] from declared estimator
    base_vel_estimate: NDArray | None = None
    timestamp: float
    units: dict = Field(default_factory=lambda: {"hinge": "rad", "slide": "m", "vel": "per_s"})


class ImageObs(Strict):
    camera: str
    height: int
    width: int
    encoding: Literal["rgb8", "depth_m"]
    pixels: NDArray | None = None       # None when streamed by reference
    ref: str | None = None
    timestamp: float


class SensorChannel(Strict):
    name: str
    kind: str
    values: NDArray
    mask: NDArray
    timestamp: float


class RuntimeEventView(Strict):
    """Observation-backed runtime status of an event (no hidden completion truth)."""
    event_id: str
    attempt: int
    status: Literal["pending", "ready", "active", "succeeded", "failed", "cancelled", "blocked"]
    reason_code: str | None = None
    rejection_count: int = 0
    last_rejection_reasons: list[str] = Field(default_factory=list)


class ReceiptView(Strict):
    event_id: str
    attempt: int
    output_name: str
    type: str
    version: int
    valid: bool
    value: dict
    covariance_diag: list[float] | None = None
    age_s: float = 0.0


class TaskInput(Strict):
    definition: TaskDefinition
    graph_version: int
    runtime_version: int
    events: list[RuntimeEventView]
    receipts: list[ReceiptView]
    interventions: list[str] = Field(default_factory=list)


class ObjectDescriptor(Strict):
    """Visible descriptor available to deployment (e.g. detector output), not sim identity."""
    slot: int
    descriptor: str
    bbox_xyxy: list[float] | None = None
    position_estimate: list[float] | None = None     # from perception, with covariance
    position_cov_diag: list[float] | None = None
    visible: bool
    bound_entity: EntityRef | None = None             # task binding supplied by the user/plan
    timestamp: float


class PredicateEstimate(Strict):
    """Output of a declared public estimator (perception/proprioception/contact sensing).
    known=False means unobserved/unknown -- never silently false."""
    predicate: str
    args: list[str]
    value: bool | float | None
    known: bool
    confidence: float = Field(ge=0, le=1)
    estimator: str
    timestamp: float


class PolicyObservation(Strict):
    kind: Literal["policy_observation"] = "policy_observation"
    observation_id: str
    sensor_time: float
    robot_spec_hash: str
    sensor_images: list[ImageObs]
    measured_node_state: NodeState
    declared_sensor_channels: list[SensorChannel]
    object_descriptors: list[ObjectDescriptor] = Field(default_factory=list)
    predicate_estimates: list[PredicateEstimate] = Field(default_factory=list)
    task_input: TaskInput | None = None
    belief_history: list[dict] = Field(default_factory=list)


class ContactTruth(Strict):
    body_a: str
    body_b: str
    pos: list[float]
    normal: list[float]
    force: float


class PrivilegedTruth(Strict):
    """Training/evaluation-only bus. Never an input to deployable policies."""
    kind: Literal["privileged_truth"] = "privileged_truth"
    observation_id: str
    sim_time: float
    object_poses: dict[str, list[float]]            # sim object name -> [x y z qw qx qy qz]
    object_entity_map: dict[str, str]               # task entity id -> sim object name
    contacts: list[ContactTruth]
    held_by: dict[str, list[str]]                   # manipulator -> objects
    predicates: dict[str, bool | float]
    event_completion_truth: dict[str, bool]
    reward: float | None = None
    labels: dict = Field(default_factory=dict)
