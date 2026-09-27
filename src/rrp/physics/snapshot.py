"""Full continuation snapshot contract: physics AND controller/runtime/sensor/belief/RNG state.

Restoring qpos/qvel alone is not a continuation; every component below must be present.
"""
from __future__ import annotations

from dataclasses import dataclass, field

REQUIRED_COMPONENTS = ("physics", "controller_state", "task_runtime", "sensor_filters", "entity_tracker",
                       "belief_state", "command_queue", "sampler_rng", "env_rng", "source_versions", "step_count")

PHYSICS_FIELDS = ("time", "qpos", "qvel", "act", "ctrl", "qacc_warmstart", "mocap_pos", "mocap_quat",
                  "qfrc_applied", "xfrc_applied")


class SnapshotError(ValueError):
    code = "snapshot_incomplete"


@dataclass
class Snapshot:
    components: dict = field(default_factory=dict)

    def validate(self) -> "Snapshot":
        missing = [c for c in REQUIRED_COMPONENTS if c not in self.components]
        if missing:
            raise SnapshotError(f"snapshot missing components: {missing}")
        ph = self.components["physics"]
        miss_ph = [f for f in PHYSICS_FIELDS if f not in ph]
        if miss_ph:
            raise SnapshotError(f"physics snapshot missing fields: {miss_ph}")
        return self
