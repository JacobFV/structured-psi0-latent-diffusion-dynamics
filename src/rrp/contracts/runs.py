from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import Strict

RunState = Literal["planned", "implementing", "test_failed", "verified", "running", "completed",
                   "failed_hypothesis", "blocked_external", "budget_exhausted", "failed", "cancelled"]


class ResourceManifest(Strict):
    node: Literal["host", "peer"]
    cpu_cores: float
    memory_bytes: int
    gpu: bool
    gpu_memory_bytes: int = 0
    max_seconds: int
    lease_id: str | None = None


class RunManifest(Strict):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    stage: Literal["ops_fixtures", "source_and_assets", "teacher_gui_vertical_slice", "codec_policy_development",
                   "vlm_semantic_integration", "primary_confirmation", "breadth_and_interventions",
                   "online_adaptation", "audit_and_release"]
    question: str
    kind: str
    method: str
    config: dict
    config_hash: str
    source_revision: str
    dataset_hashes: list[str] = Field(default_factory=list)
    seeds: list[int]
    resources: ResourceManifest
    state: RunState = "planned"
    sealed: bool = False
    controller_source: str | None = None       # teacher/scripted/learned labelling
    outputs: dict = Field(default_factory=dict)
    notes: str = ""
