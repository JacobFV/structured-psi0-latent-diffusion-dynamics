"""Layer 9: Pipeline(body_family) and its stage registry (W5). See rrp.pipelines.base."""
from rrp.harness.pipelines.base import MANIFEST, Pipeline, StageContext, StageError, read_stage_manifest, register

__all__ = ["MANIFEST", "Pipeline", "StageContext", "StageError", "read_stage_manifest", "register"]
