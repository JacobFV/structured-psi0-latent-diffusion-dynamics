# pipeline track (W5: unified pipeline + DAG orchestration, audit phases 3-4)

Owner: pipeline agent. Worktree `~/work/rrp-wt/pipeline`, branch `track/pipeline`. Peer code dir
`/dev/shm/rrp-brandonin/wt/pipeline` (RRP_PEER_REPO). Started 2026-09-26.

State: see the table at the end (RunConfig, pipelines, run-dag, DAG files: verified; parity: see "parity").

## API
### `rrp.contracts.runconfig`
- `RunConfig` (pydantic, strict, `schema_version="runconfig-1"`): family (arm|dual|legged), stage (12 pipeline stages +
  legacy-only kinds train_bc/train_policy/adapt/vlm/protocol), variant (sem|nosem|semfix|na), seed, lineage, track, tag,
  `inputs` (run ids), `flags`, `params` (native config, passed through unchanged), `options` (pipeline-wrapper args such
  as robots, seed sets, workers), note, `legacy`.
- `out` is DERIVED: `artifacts/runs/{track}/{lineage}/{stage}[-{tag}]_s{seed}`. Deviation from the audit template: the
  `tag` suffix (a lineage has six refits, four flows and eight DAgger rounds). New configs may not set out_dir.
- Flags (`Flags`): zero_prev_action, realizer_anchor, realizer_drop_qd, probe_lv_min, qd_dropout, contact_version; every
  field required (no default). `FLAG_SPEC[(family, stage)]` says which apply and where the existing code reads them
  (e.g. arm train_rep: `latent.probe_lv_min`; legged refit: top-level `qd_dropout`; arm evaluations: the ladder CLI
  `--prev-action zero|own`; contact_version is recorded only). Applicable = must be non-None; not applicable = None.
- Variant check for NEW configs: train_rep sem needs semantic weight > 0 and floor -8, semfix > -8, nosem weight 0;
  flows: nosem iff packet_semantic_weight 0.
- Run ids: `<store>/<name>[:file]` resolves to `artifacts/<store>/<name>/<file>`; `RunIndex` aliases
  (`configs/run_index.json`, 115 entries) name the six legacy arm lineages canonically, e.g.
  `arm/semfix/s2/refit-gendag3_noqd` -> `artifacts/runs/ladder_rz_sfjf2_gendag3_noqd`. Lists of buffers are
  `{runs: [...], files: [...]}` products (or plain ordered lists when the legacy list is not a product).
- `overlay` (deep merge, dotted keys, None deletes), `render` (`{expr}` templates: names, + - * // %), `expand_matrix`.
- Legacy: `load_legacy(path)`; `to_native()` returns the original dict exactly (key order too) for all 263 run configs
  under configs/ (excluded: resources.local.json, run_index.json). Flags a legacy file omits get the value the code
  always defaulted to (LEGACY_FLAG_DEFAULTS, each with the source line) and are listed in `legacy.absent_flags`, so they
  are not written back; zero_prev_action is never defaulted (W3 reader semantics unchanged).

### `rrp.pipelines`
- `Pipeline(family).stages()`, `.run(rc, index=, root=)`: checks inputs exist, calls the stage in the repo root,
  writes `<out>/pipeline_manifest.json` through `rrp.data.manifest.write_manifest` (hashed) with a W3 `Provenance`
  (source label, flags, versions, code revision) + RunConfig, config_hash, native config, input/output digests, metrics.
  `python -m rrp.pipelines run --config[-b64]` is the leased-job entry point; `python -m rrp.pipelines stages`.
- arm (all 12 stages): collect/pack (`rrp data generate|pack`), train_rep/probes/train_flow/flow_ft/refit (in-process
  `rrp.training.latent_train` functions, same calls as the CLI / scripts/ladder_refit.py; flow option
  `final_snapshot_alias` = the chains' F0 copy to snap_final_s20000.pt), dagger_collect (scripts/ladder.py
  --collect-dagger per body, modes bc|gen, genctx; BC expert only), eval_r1 (stateless-BC oracle, DIAGNOSTIC), eval_r2,
  heldout (refuses training bodies), edits (`rrp latent semantic-edits`, sharded). scripts/ladder.py keeps its logic in
  the script's main, so the rollout stages run it as a subprocess with identical arguments (moving it into
  rrp.evaluation is a follow-up). Outputs go to the stage's own dir, never to the shared ladder_v1/.
- legged (11 stages; no pack: legged trains from shards): collect, train_rep, probes (post-hoc), train_flow, flow_ft,
  dagger_collect, refit, eval_r1/eval_r2/heldout (legged_latent_eval over seed chunks + legged_ladder_summary.py, as
  scripts/legged_ladder.sh), edits (as legged_edit_suite.sh + effects, optional mirror effect). Training on a dataset
  whose manifests record a different contact version than flags.contact_version is refused.
- dual: skeleton (train_rep, train_flow via the arm trainers) + TODO list in `rrp/pipelines/dual.py`.
- Eval rows keep their legacy free-string source; the manifest records per-row canonical `Source` kinds
  (`parse_source`) under metrics.<robot>/s<seed>.source_kinds.

### `rrp run-dag` (`rrp.orchestration.dag`, `rrp.cli.dag`, YAML subset `rrp.orchestration.yamlmini`)
- DAG file: name, family, track, lineage/label_prefix templates, `matrix` (whole DAG per point, node ids
  `<node>@<variant>.<...>.s<seed>`), `axis_vars`, `vars`, `lists` ("$name"), `defaults` (placement, retries,
  resources, max_parallel, admission_timeout_s), `base` (shared RunConfig fields; flags must be stated), `nodes`
  (stage, tag, deps, resources {cpu, mem, gpu, gpu_mem, max_seconds, disk}, placement host|peer|auto, retries,
  only {axis: values}, per {axis: {value: overlay}}, config). Inputs "@node[:file]" add the dependency.
- Each node = one leased job via the broker: host `python -m rrp.cli ops run --detach`, peer `scripts/peer_run.sh
  --detach` (refuses unless RRP_PEER_REPO is `/dev/shm/rrp-brandonin/wt/<track>`). Completed iff rc 0 AND the manifest
  carries the node's config_hash. Bounded retries (default 0, D-061); broker capacity refusals are waited for
  (logged, bounded by admission_timeout_s) and are not attempts; a unit that vanishes without an rc (e.g. OOM kill)
  is rc -1 with the systemd Result recorded.
- Ledger: one JSON per DAG (`artifacts/runs/<track>/_dags/<name>/ledger.json`, atomic writes, flock). Resume: completed
  nodes skipped; running nodes re-adopted by lease id (rc file); outputs whose manifest has the same config_hash are
  adopted without a lease; a changed config of a node that already ran is refused (`--reset NODE`); failed/blocked
  only re-run with `--retry-failed`. It never stops a lease.
- `--dry-run` prints nodes, state, resources, placement, out dir, config hash, inputs and the launch command;
  `--show-config NODE` prints RunConfig + native config; `--only REGEX`, `--point variant=semfix,seed=2`.
- Cross-placement dependencies are refused by default (no automatic artifact transfer; the DAGs use one placement).

### DAG files
- `dags/arm_lineage.yaml`: stageA -> F0 -> Fft; bc1 -> rzbcdag1/rzbcdag1long -> bc2/bc3/gen1 -> rzbcdag2 -> rzgendag1
  -> gen2 -> rzgendag2 -> gen3/gdag1 -> rzgendag3/Fgdag1 -> gdag2 -> Fgdag2h -> finalevals, heldout, prog20k,
  proggdag1, orcbc, semedits; variant {sem, semfix, nosem} x seed {1, 2} = 150 nodes. Every planned native training
  config (Stage A, 4 flows, 6 refits) equals the committed legacy config of all six lineages (jointfix, sfjf, nsjf,
  sejf2, sfjf2, nsjf2) up to names/out dirs and explicitly stated code defaults (tests/unit/test_dag.py). Recipe quirk
  reproduced: gendag1 omits gen1/ur5e_tf3 and lists bc1, bc2, gen1, bc3.
- `dags/legged_fixrep.yaml`: variant {semfix, nosem} x body {go2, hexapod6} x seed {1, 2}: rep -> [nosem probe] -> flow
  -> R2 (snap_s4000, policy) -> z-edit and task-context edit suites with mirror effects; the 16 rep/flow configs
  equal configs/legged_fixsem/* (test).
- `dags/parity_arm.yaml` (below), `dags/smoke_legged.yaml` (tiny host smoke).

### W4-deferred duplicates
- packet builders: ONE builder `rrp.evaluation.packets.arm_packet`; `ladder.make_packet` delegates (difference: the
  old copy kept a strided z view, now contiguous; values identical; tests/unit/test_packets.py); `build_packet` is an
  alias; `dual_packet` moved unchanged (different semantics: masked assemblies).
- `_dev`: `rrp.contracts.workload.select_device(on_cap_error="raise"|"ignore")`; latent_train._dev = raise (old
  behaviour), controllers.bundles._dev = ignore (old legged behaviour); `training.adapt._device` stays (returns cap
  info, disables TF32).
- oracle wrappers NOT merged (different interfaces): arm R1 = `ladder.OraclePacketPolicy` (batched, via
  scripts/ladder.py), arm edits = `latent_semantic_edits.OracleSource` (per-call context edits), legged =
  `legged_latent_eval.OracleShadow`. The pipeline uses exactly one per (family, stage).
- prev-action featurizer install paths NOT merged: `ladder.install_prev_action` (wraps step/snapshot/restore) vs the
  inline PrevActionFeaturizer in `run_ladder` (recording done by the loop). The pipeline only reaches run_ladder
  (through scripts/ladder.py), so it has one path.
- Source enum for eval rows: recorded in the stage manifest (rows unchanged for comparability).

## parity (PENDING: filled in below when the runs finish)

## tests
`tests/unit/test_runconfig.py` (round-trip of all configs, flags, variant check, overlays/matrix/templates),
`test_dag.py` (arm DAG vs 6 lineages, legged DAG vs 16 configs, structure, fake-runner run/retry/block/resume/
admission/adoption/config change, OpsRunner rc parsing, YAML subset), `test_pipelines.py` (stage wiring with fakes: arm
refit/eval_r2/heldout, legged train_rep + contact-version refusal), `test_packets.py`. Layering test unchanged and green.

## remaining steps
1. Port new work to run-dag: W8 legged regeneration (contact v2: set base.flags.contact_version and the dataset input
   in a copy of dags/legged_fixrep.yaml; needs a v2 `collect` node per body).
2. Dual: implement collect/pack/eval stages (move `cli.dual_latent.cmd_pack` logic into rrp.data first).
3. Move scripts/ladder.py main into `rrp.evaluation` (the arm rollout stages then call it in-process); same for the
   legged summary/effects scripts.
4. Cross-placement artifact transfer (rsync pull/push per edge) if a DAG ever mixes host and peer.
5. Retire the chain scripts once no lease references them (`rrp ops status` on both nodes; labels a*_/ans_/asf_).
