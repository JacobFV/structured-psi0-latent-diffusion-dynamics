"""Dual-arm (M=2) stages: SKELETON. The dual lineage never finished (naming.md: dualarm_latent_{sem,nosem}_v1), so no
parity target exists yet. Registered here: the stages whose existing entry points take a config dict directly.

TODO (W5 follow-up, in this order):
- collect: rrp.data.collect_dual (`python -m rrp.data.collect_dual --config`), task-specific configs in configs/data
  (support_insert / handover / assign).
- pack: `rrp latent pack-dual --config` (logic in rrp.cli.dual_latent.cmd_pack; move it to rrp.data.dual_latent first,
  pipelines may not import cli).
- train_rep / train_flow: rrp.training.latent_train on the dual pack (multi_m=2 path) - registered below.
- eval_r2 / edits: rrp.evaluation.dual_latent_eval and latent_semantic_edits.cmd_arm (`rrp latent arm-edits`).
- dagger_collect / refit: no dual DAgger exists yet.
"""
from __future__ import annotations

from rrp.pipelines import arm
from rrp.pipelines.base import StageContext, register


@register("dual", "train_rep", source="learned")
def train_rep(ctx: StageContext) -> dict:
    """Dual Stage A: the arm trainer on a multi-assembly pack (configs/latent/rep-dualarm_latent_*)."""
    return arm.train_rep(ctx)


@register("dual", "train_flow", source="learned")
def train_flow(ctx: StageContext) -> dict:
    """Dual system i: the arm flow trainer on a multi-assembly pack."""
    return arm._flow(ctx)
