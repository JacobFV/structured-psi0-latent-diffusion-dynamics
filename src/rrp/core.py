"""rrp.core: the STABLE public API of rrp for external packages (psi1z) — see docs/core_api.md.

Everything listed in `STABLE` is re-exported here lazily (PEP 562): `import rrp.core` needs only the core install
(numpy, pydantic); a name backed by torch (probes, realizer) or mujoco (arm system 0 runtime) imports its module on
first access, so those names need the `ml` / `sim` extras. Import from here, not from the defining modules: the
defining modules may move (W4-style moves keep shims, but this list is the contract), and a name removed or changed
incompatibly here bumps CORE_API_VERSION's major number.
"""
from __future__ import annotations

import importlib

CORE_API_VERSION = "1.0"

# name -> defining module. Grouped as in docs/core_api.md.
STABLE: dict[str, str] = {}


def _add(module: str, *names: str) -> None:
    for n in names:
        STABLE[n] = module


# 1. packet contract
_add("rrp.contracts.latent_action", "LATENT_SCHEMA", "LatentActionChunk", "AssemblyHandle", "EntityHandle",
     "check_packet")
_add("rrp.contracts.errors", "ControllerRejection", "StaleActionError")
_add("rrp.contracts.action", "NativeCommand")
# 2. system 0 / realizer base classes and compatibility IDs
_add("rrp.contracts.system0", "System0Base", "System0Stats")
_add("rrp.controllers.latent_realizer", "LatentRealizer", "LatentSystem0", "make_realizer", "bundle_versions",
     "is_fingerprinted", "REALIZER_RECURRENT_STATE")
_add("rrp.evaluation.legged_latent_eval", "legged_bundle_versions")
# 3. semantic probes and bounded-NLL losses (torch)
_add("rrp.models.latent_probes", "PacketProbe", "gaussian_nll", "probe_loss", "probe_metrics", "probe_loss_multi",
     "probe_metrics_multi", "ALL_QUERIES", "ENTITY_QUERIES", "ENTITY_MANIP_QUERIES", "ENTITY_EFFECT_QUERIES",
     "MANIP_QUERIES")
# 4. statistics
_add("rrp.evaluation.statistics", "wilson", "newcombe_diff", "boot_ci", "boot_diff", "paired_bootstrap_ci",
     "paired_permutation_test", "permutation_test", "mcnemar_exact", "success_counts")
# 5. packet-edit / causal test harness
_add("rrp.evaluation.edit_harness", "EditCondition", "EDIT_KINDS", "PacketSource", "restamp", "run_suite",
     "read_rows", "paired_effect", "matched_random", "orthogonal_matched", "probe_jacobian", "probe_guided_edit")
# 6. provenance, source labels, manifests
_add("rrp.contracts.provenance", "Provenance", "PhysicsProvenance", "CodeProvenance", "Source", "SourceLabel",
     "source_label", "parse_source", "make_provenance", "legacy_provenance", "read_provenance", "code_provenance",
     "physics_provenance", "weights_digest", "file_digest", "training_flags", "resolve_zero_prev_action",
     "MissingFlagError", "FEATURIZER_VERSION", "row_source", "stamp_source_label", "parse_legacy_source",
     "canonical_source_labels", "SOURCE_LABELS_ENV", "SOURCE_LABEL_VERSION")
_add("rrp.data.manifest", "write_manifest", "read_manifest", "dataset_provenance", "assert_disjoint_lineages")
# 7. RunConfig, pipeline registry, run-dag, family/robot extension hooks
_add("rrp.contracts.runconfig", "RunConfig", "Flags", "RunIndex", "RunConfigError", "PIPELINE_STAGES", "FLAG_NAMES",
     "overlay", "expand_matrix", "render", "register_family", "unregister_family", "families", "ensure_family",
     "load_family_plugins", "FAMILY_ENTRY_POINT_GROUP")
_add("rrp.pipelines.base", "Pipeline", "StageContext", "StageError", "register", "read_stage_manifest")
_add("rrp.orchestration.dag", "load_dag", "plan_dag", "Executor", "OpsRunner", "Ledger", "DagError")
_add("rrp.bodies.catalog", "register_robot", "unregister_robot", "workbench_robots", "ROBOT_ENTRY_POINT_GROUP")
# 8. ops broker client
_add("rrp.orchestration.broker", "ResourceBroker", "ResourceRequest", "Lease", "CapacityError", "LeaseError")
_add("rrp.orchestration.runtime", "make_broker", "run_leased", "ops_root", "node_role", "load_config")
# paths
_add("rrp.contracts.paths", "rrp_home", "is_checkout", "data_path", "package_data")

# pipeline stage registration is a decorator named `register`; expose an unambiguous alias too
STABLE["register_stage"] = "rrp.pipelines.base:register"

__all__ = ["CORE_API_VERSION", "STABLE", *STABLE]


def __getattr__(name: str):
    mod = STABLE.get(name)
    if mod is None:
        raise AttributeError(f"rrp.core has no attribute {name!r} (stable names: rrp.core.STABLE)")
    mod, _, attr = mod.partition(":")
    val = getattr(importlib.import_module(mod), attr or name)
    globals()[name] = val
    return val


def __dir__():
    return sorted(__all__)
