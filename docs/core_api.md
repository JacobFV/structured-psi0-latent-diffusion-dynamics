# rrp core API (W11, D-100)

rrp is both the multi-body research repo and an installable core for external packages (psi1z: Ψ₀ + structured
packets, Python 3.11). This page says what is stable, how to extend rrp from outside, and how the pin is bumped.

## install

```
# core only (numpy, pydantic): contracts, provenance, manifests, statistics, RunConfig/pipelines/run-dag, ops broker
pip install "rrp @ git+file:///home/brandonin/work/relational-robot-policy@<sha>"
# + torch (probes, realizer, edit gradients) / + mujoco (rrp's own sim bodies and runtimes)
pip install "rrp[ml,sim] @ git+file:///home/brandonin/work/relational-robot-policy@<sha>"
```

- Python `>=3.11,<3.13`. `pytest tests/unit` runs under both (see research/tracks/core.md for the verified envs
  `~/work/ext/venvs/rrpcore311` and `rrpcore312`).
- Extras: `sim` (mujoco), `ml` (torch), `service` (workbench app), `viz` (pillow, imageio, matplotlib: videos and
  plots), `vlm` (transformers, huggingface_hub: the VLM backbone), `dev` (pytest, jsonschema), `all`.
- The wheel ships the task specs as package data (`rrp/_data/tasks`); nothing else under the repo is packaged
  (configs/, artifacts/ and .cache/assets are data roots, see "paths").

## stable names: `rrp.core`

`from rrp.core import X` for every X in `rrp.core.STABLE` (the list below). `import rrp.core` needs only the core
install; names are re-exported lazily, so a torch/mujoco-backed name imports its module on first access (marked
[ml] / [sim]). `tests/unit/test_core_api.py` asserts every name imports, and that the unmarked ones import with
torch/mujoco/fastapi blocked.

| group | names | defined in |
|---|---|---|
| packet contract | `LATENT_SCHEMA`, `LatentActionChunk`, `AssemblyHandle`, `EntityHandle`, `check_packet`, `ControllerRejection`, `StaleActionError`, `NativeCommand` | contracts.latent_action / errors / action |
| system 0 | `System0Base`, `System0Stats` (acceptance protocol: receive / invalidate / stats); [ml] `LatentRealizer`, `LatentSystem0` (arm), `make_realizer`, `REALIZER_RECURRENT_STATE` | contracts.system0, controllers.latent_realizer |
| compatibility IDs | [ml] `bundle_versions`, `is_fingerprinted`, `legged_bundle_versions`; `weights_digest` | controllers.latent_realizer, evaluation.legged_latent_eval, contracts.provenance |
| probes, bounded NLL | [ml] `PacketProbe`, `gaussian_nll` (log-variance clamped to `[lv_min, 6]`), `probe_loss`, `probe_metrics`, `probe_loss_multi`, `probe_metrics_multi`, query tuples (`ALL_QUERIES`, ...) | models.latent_probes |
| statistics | `wilson`, `newcombe_diff`, `boot_ci`, `boot_diff`, `paired_bootstrap_ci`, `paired_permutation_test` (sign flip; exact up to 16 pairs), `permutation_test`, `mcnemar_exact`, `success_counts` | evaluation.statistics |
| edit / causal harness | `EditCondition`, `EDIT_KINDS`, `PacketSource` (protocol), `restamp`, `run_suite`, `read_rows`, `paired_effect`, `matched_random`, `orthogonal_matched`; [ml] `probe_jacobian`, `probe_guided_edit` | evaluation.edit_harness |
| provenance | `Provenance`, `PhysicsProvenance`, `CodeProvenance`, `Source`, `SourceLabel`, `source_label`, `parse_source`, `make_provenance`, `legacy_provenance`, `read_provenance`, `code_provenance`, `physics_provenance` [sim model arg], `file_digest`, `training_flags`, `resolve_zero_prev_action`, `MissingFlagError`, `FEATURIZER_VERSION` | contracts.provenance |
| manifests | `write_manifest`, `read_manifest`, `dataset_provenance`, `assert_disjoint_lineages` | data.manifest |
| run configs | `RunConfig`, `Flags`, `RunIndex`, `RunConfigError`, `PIPELINE_STAGES`, `FLAG_NAMES`, `overlay`, `expand_matrix`, `render` | contracts.runconfig |
| pipelines, run-dag | `Pipeline`, `StageContext`, `StageError`, `register_stage` (= `register`), `read_stage_manifest`; `load_dag`, `plan_dag`, `Executor`, `OpsRunner`, `Ledger`, `DagError` | pipelines.base, orchestration.dag |
| extension hooks | `register_family`, `unregister_family`, `families`, `ensure_family`, `load_family_plugins`, `FAMILY_ENTRY_POINT_GROUP`; [sim] `register_robot`, `unregister_robot`, `workbench_robots`, `ROBOT_ENTRY_POINT_GROUP` | contracts.runconfig, bodies.catalog |
| ops broker client | `ResourceBroker`, `ResourceRequest`, `Lease`, `CapacityError`, `LeaseError`, `make_broker`, `run_leased`, `ops_root`, `node_role`, `load_config` | orchestration.broker / runtime |
| paths | `rrp_home`, `is_checkout`, `data_path`, `package_data` | contracts.paths |

Everything else in rrp is internal: it may move or change without a version bump (moves keep a shim for a while,
W4). Do not import internals from psi1z; if psi1z needs something, add it to `rrp.core` in rrp first.

## extending rrp from outside (psi1z registers `g1_simple` without editing rrp)

### a body family for RunConfig / Pipeline / run-dag

```python
# psi1z/src/psi1z/rrp_plugin.py
def register():
    from rrp.core import register_family, register_stage
    register_family("g1_simple",
                    flag_spec={"train_rep": {"probe_lv_min": "latent.probe_lv_min"}},   # flags that apply, per stage
                    default_stage_flags={},                                            # every other stage: none
                    doc="Unitree G1 + Dex3 through the Psi0 action interface, SIMPLE benchmark")

    @register_stage("g1_simple", "eval_r2", source="learned")
    def eval_r2(ctx):            # ctx: StageContext (rc, native config, inputs, out dir, subprocess helpers)
        ...
        return {"outputs": {...}, "metrics": {...}}
```

and in psi1z's pyproject:

```toml
[project.entry-points."rrp.families"]
g1_simple = "psi1z.rrp_plugin:register"
```

- rrp loads the `rrp.families` entry points lazily, the first time an unknown family name is validated
  (`RunConfig`, `Pipeline(...)`, `rrp run-dag`, `python -m rrp.pipelines stages`). A module target registers on
  import; a callable target is called with no arguments. A plugin that fails to load raises (no silent skip).
- Or call `register()` directly before building RunConfigs (no packaging needed, e.g. in tests).
- Flags are the closed set `FLAG_NAMES`; an applicable flag must be stated in every RunConfig (no silent defaults),
  a non-applicable one must be None, exactly as for the built-in families. Stages are the fixed `PIPELINE_STAGES`.
  A new flag or stage name is a change in rrp (and a core API minor bump).
- Built-in family names cannot be replaced; re-registering the same spec is a no-op, a different one raises.

### a robot for rrp's MuJoCo workbench (sim)

`register_robot(key, factory)` or an entry point in the group `rrp.robots` whose target returns `{key: factory}`.
Keys must not shadow built-in robots. (psi1z's G1 runs in Isaac Sim through SIMPLE; it does not need this. It is
for bodies built with rrp's MuJoCo generators.)

### a system 0 for a new body

Subclass `System0Base` (acceptance protocol, compatibility IDs, rejection log) and implement `robot_spec_hash` and
`tick(...)`. Compute compatibility IDs from the weights actually loaded (`weights_digest`, `bundle_versions`), so a
retrained encoder or realizer can never silently accept old packets.

### an edit suite for a new body

Declare the conditions (`EditCondition` with a written prediction; include `replay`, an `irrelevant_control` of
matched norm and a `negative_control`), implement `run_one(seed, condition) -> row` with your simulator, run
`run_suite(...)` (JSONL, resumable with `done=`), and report `paired_effect(rows, metric, condition)`. Edited
packets go through `restamp(..., intervention=name)` so they are labelled `source="debug"`.

## paths (installed rrp)

rrp functions that touch repo-level data (artifacts/, configs/, tasks/, .cache/assets, ops state) resolve it from
`rrp_home()`: `$RRP_HOME`, else the rrp source checkout (when running from one; unchanged behaviour), else the
current directory. Task specs fall back to the packaged copy (`data_path`). `code_provenance()` of an installed rrp
records the commit it was installed from (pip/uv `direct_url.json`), not the git state of the consumer's directory.

The ops broker state is shared per node: `ops_root()` is `$RRP_OPS_ROOT`, else (on the host) the main rrp checkout
`~/work/relational-robot-policy` when it has `configs/resources.local.json`. psi1z jobs therefore lease from the SAME
broker as rrp's agents (AGENTS.md: one aggregate budget). Run leased jobs with
`python -m rrp.cli ops run --cpu X --mem Y --label L -- cmd` from any env that has rrp installed.

Known remaining cwd/checkout assumptions (not part of the stable API): see research/tracks/core.md "paths left".

## versioning and pin bumps

- `rrp.core.CORE_API_VERSION` = `MAJOR.MINOR`. Adding a name or an optional argument: minor. Removing or renaming a
  stable name, changing a signature incompatibly, or changing an on-disk format a stable function reads or writes
  (packet schema, manifest, provenance, RunConfig schema): major, recorded in research/decisions.md.
- psi1z pins rrp by full git sha (`rrp @ git+file://...@<sha>`), never a branch. Bump deliberately:
  1. change rrp (tests green under 3.11 and 3.12), merge to rrp main;
  2. in psi1z set the new sha in pyproject.toml, `uv pip install -e ~/work/psi1z` in the 3.11 env, run psi1z's tests;
  3. commit the bump in psi1z with the rrp sha and CORE_API_VERSION in the message.
- A psi1z result records the rrp commit it ran with (`code_provenance()` of the installed rrp), so the pin at the
  time of a result is recoverable.
