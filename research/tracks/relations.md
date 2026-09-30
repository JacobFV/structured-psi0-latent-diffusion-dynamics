# track: relations (open) — relation-factor experiments (D-144)

The relation-factor registry (D-144, `docs/relations.md`, catalog `research/relations_catalog.md`) is built: token model,
`FactorSite` in the shared attention block, `ReadoutProbe`, `relgen` labels / parts / transforms, curriculum scheduler,
deploy guard, room panels. What is missing is the EXPERIMENT: no factor set has been trained against the `preset:arm`
control. That is this track. State: **planned** (recipes render and plan dry; nothing has run).

Per-unit history of the build (R1-R21, rel-geo, sweep-flags, closures) is in `research/decisions.md` D-144 and its
addenda; the unit notes `research/tracks/rel-*.md` and `sweep-flags.md` are closed-track material (moved to `.old/research/tracks/` by P5).
Their open items are collected below, so nothing needs the old notes.

## recipes

`recipes/templates/relations_factor.yaml`: one shared representation (stageA, semantic probes, trained once so every
factor set sees the same input information) + probes read once, then per factor set (`fset` axis) x seed: flow `F0` with
`params.policy.factors = [preset:arm, preset:<set>]`, the deployable route on the R2 dev bodies (`dev`) and the two
held-out bodies (`heldout`). `base` (preset:arm only) is the paired control at equal steps and seeds.

| instance | factor set vs `base` | resolves to (beyond the 17 `edge.*` + `msg.incidence`) |
|---|---|---|
| `recipes/relations/relations_geo.yaml` | `preset:geo` | `geo.pos3d geo.depth3d geo.orient geo.normal_align geo.above` |
| `recipes/relations/relations_ix.yaml` | `preset:ix` | `ix.contact ix.held_by ix.handover ix.support ix.force_flow` |
| `recipes/relations/relations_task.yaml` | `preset:task` | `task.next_contact` (`time.same_track` is `planned`, D-146 round 2) |

`preset:ui` (ComputerWorld widget tree) is a pointer-track factor set: it is a `factors:` var of the pointer recipes
(`recipes/pointer/`), not an arm instance. Legged / humanoid sets (`legged-r19`: `leg.foothold leg.com_support`) run through
`recipes/templates/legged_lineage.yaml` once a body set is chosen; no instance yet.

Relgen reaches the trainer (R2 RG, below): per set the F0 node takes the `relgen` shards through `inputs.relgen` and a
`params.curriculum`; the `base` control and the shared stageA take none. Factor lists are the ones the arm family can build:
`resolve(family="arm")` rejects `geo.normal_align` as `given` (no `normal` ctx field; it is `source: probe` here), `ix.handover`
(no `handover_pairs` label), so those two are not in the F0 lists (the full presets failed at model build, which no run had
reached before; `time.same_track` is already out of `preset:task`, D-146 round 2). State: planned for the real recipes, smoke on the peer (R2 RG note).

## resume

Host = git / editing / unit suite only. Everything below runs on the peer with `RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/relations`
after `ops/bin/peer_sync.sh push`.
```
rrp run-dag recipes/relations/relations_geo.yaml --dry-run          # 14 nodes: stageA, probes_rep, then F0/dev/heldout x (base, geo) x seeds 1, 2
rrp run-dag recipes/relations/relations_geo.yaml --max-parallel-gpu 1
rrp run-dag recipes/relations/relations_ix.yaml --max-parallel-gpu 1     # the base control (lineage relations-base) is the same run set in all three instances: it is skipped once complete
rrp run-dag recipes/relations/relations_task.yaml --max-parallel-gpu 1
```
Dry-run node list of one instance (`relations_geo`): `stageA`, `probes_rep` (global, lineage `relations-shared`), then for
each `fset` in base, geo and seed in 1, 2: `F0@<fset>.s<seed>` -> `dev@...` and `heldout@...` (out
`artifacts/runs/relations/relations-<fset>/<stage>[-dev]_s<seed>`). `base` runs are identical across the three instances
(same config hash): the second and third instance find them complete and skip them.

## open items (collected from the rel-* notes and sweep-flags)

1. ~~`relations_data` is not a stage~~ (R10) closed by R1 below: a registered stage (`arm`, `legged`, `pointer`, `psi0`,
   `relations`) and the trainer hook `relation_batches`; the arm trainers call the hook (R2 RG note); HL legged and the
   observation gap for non-arm families remain.
2. **Probe-sourced `edges:support-v1`** (rel-geo, R17): `ix.force_flow` reads the support graph from the PRIVILEGED label
   only (`nets/batch.py::support_edges`, a pure function, not on `RelCtx.edges`); the deployable half (from `ix.support`'s own
   bilinear pair score) needs a read hook in `relations/ops.py` (`BilinearOp` returns q/k features, `FieldReadouts` skips
   bilinear). Until then `ix.force_flow` cannot be deployed with a probe source.
3. **Pair readouts unconsumed** (R18): `task.next_contact`'s `ReadoutDef(address="pair", reads="tokens")` is metadata; neither
   `ReadoutProbe` nor `FieldReadouts` reads a pair address. `candidate_interaction_edges` is not wired into `RelCtx`.
4. **`ix.handover` is a per-frame proxy** (R16): "both hands on the object at once"; the episode-level release-then-grip
   event belongs in `relations_data` calling the label across a trajectory.
5. **Legged / humanoid foothold not on a live env** (R19): `terrain_steps` / `foothold_cell` entities are declarative; no live
   env's `state_view()` populates them (compose wiring, R11-adjacent). `warp_tracker_ppo.py`'s `extra_obs` display string
   still implies the height scan reaches the actor (one-line wording fix).
6. **`ui.drag_to` label** (R20): "focused widget = dragged widget" is a proxy; a mid-drag predicate from the `cw/drag_window`
   judge state would sharpen it.
7. **`base_cmd` reads the whole packet** (R5): the legacy restricted key-mask (locomotion knot tokens only) is not reproduced by
   `ReadoutProbe` (one key mask per call). Add a per-query key mask only if the restriction is a real requirement.
8. **Per-sample-conditioned query address** (R4): legged `body` probes use `address="asm"` + gather (M queries instead of 1);
   a shared `asm@index_field` primitive in `nets/probes.py` would remove the trick if R5/R6-style needs recur.
9. **`packet_semantic_weight` on-disk key** (rel-r2c / sweep-flags): four readers share one conversion point; retiring the key
   itself needs a registered `flow.packet_semantic` factor (`catalog.py`, `nets/flow.py`, `nets/probes.py`). Not required by
   anything today; `joint_adapt.py`'s flat read stays (stage `adapt` is a permanent legacy-only stage).
10. **Curriculum policy** (R11): `relgen/curriculum.py` ships the scheduler, promotion by probe competence and `rrp steer`; no
    run has used it. First use belongs to the geo -> ix -> task ladder once item 1 lands.

## F1: relation runtime contract (ready-f1; audit D2 / D5; docs/relations.md section 11)

Closes open items 2 and 3 above for the arm / dual nets (pair readouts are consumed; the probe-sourced support graph is
published by `ix.support`'s own pair logits) and makes "resolves" mean "runs".
- `relations.base`: `FamilyTokens` / `FAMILIES` / `register_family`; `resolve(..., family=, env_caps=, env=, training=)`
  refuses what the net cannot run (no applicable site, unfilled `given` field, label the collate cannot attach, readout
  the net does not implement, StateView caps, no scene part / transform for `mix > 0`); `estimates_loss` supervises the
  pair estimates and field probes a forward wrote (Gaussian NLL / bce / soft_ce, `(sum, count)` metrics);
  `stamp_versions` / `require_factors` for checkpoints. `gaussian_nll` lives here (`nets.probes` re-exports it).
- `relations.catalog`: families `arm`, `dual`, `psi0`, `pointer`, and `legged` / `humanoid` as their nets run today
  (the `act>knots` routing site + `probe.legged.*`); unit HL extends those two with `legged-rel-v1` and the ctx sites.
- `relations.ops`: every bilinear factor writes `rc.estimates[("pair", name)]`; closure / hop ops apply only at square self
  sites (`kin.ancestor` / `kin.sibling` crashed at `act>ctx` before); `site_field` gt path reads the factor's label name.
- `nets.batch`: field builders are Batch-only and vectorised; `relation_token_sets(family, batch, labels, deploy, fields,
  pad_ctx)`; labels are refused when `deploy=True`. `nets.flow` builds its ctx / act sets through it (the arm net had no
  fields at all before, so every `geo.*` / `id.*` factor was unrunnable there) and adds `estimates_loss` to `loss`.
- `relgen`: `load_families()`, raising `label_def` / `part_def` / `transform_def`, `rrp factors coverage` writing
  `artifacts/runs/relations/coverage/coverage.json` (factor x family x env: resolves / label runnable / part available /
  training resolve).
- Not done (lead questions): other checkpoint writers (`pointer.py`, `bundles.py`, `latent.py`, `bc.py`) still stamp by hand
  and do not `require_factors`; `harness/pipelines/relations.py` still uses `TRANSFORMS.get` (silent skip); no `@gt`
  EdgeSet builder for `ix.force_flow` source gt; the `dual` family is checked by the shared arm test only.
- Architect review fixes: the field builders now run inside the forward, so every tensor they create is placed on the
  batch's device (the branch built them on CPU: any GPU forward, default preset included, would have raised in
  `_global_kind`; test on the `meta` device); `nets.batch` was missing its `F` import (the image-token pad path raised
  NameError) and padded pair labels along one token dim only; `relations.base` is torch-free at import again
  (`rrp factors list`, the DAG parent); the `est_weight` loss argument is gone (`FactorSpec.weight` is the one knob);
  the env-caps literals of `rrp factors coverage` are now actually tested against the env sources.

## R1: `relations_data` stage + trainer hook (ready-r1; audit D1)

Closes open item 1. Nothing about the training loop changed in F0 (no new params or inputs); a trainer opts in with one line.
- Stage `relations_data` (`harness/pipelines/relations.py`, registered for `relations arm legged pointer psi0`, source
  `privileged_teacher`): resolves `params.factors` (or `policy.factors` / `latent.factors`), rolls the driver (`options.policy`,
  default `teacher:<task>`) over `options.seeds` / `n_episodes (+ seed_start)` on `options.env / task / body` through the ordinary
  `evaluate` with a `SnapshotCollector` rollout hook. The hook labels from `env.state_view()` at reset and every
  `snapshot_every` ticks (at most `max_snapshots` per episode), at capture time, so a lazy view is never labelled from a later
  state. Shards land under `<out>/relgen/<factor>/<version>/` (not `artifacts/relgen`: `schema.toml` forbids it); the shard id is
  `seed<seed>-<hash of the rows' provenance>`, so rerunning replaces its rows instead of doubling them. The stage manifest carries
  per-shard `manifest_hash`, row count and npz digests; a stage that produces no rows raises. DAG adoption (config hash + pins)
  is the resume-by-hash. The `relations_factor` template has a `relgen` node per `geo / ix / task` instance (dry-run planned for
  the three relations recipes).
- Shard IO (`write_shard`, `load_shard_rows`, `RelgenError`) lives in `harness/data/mix.py` (the data layer must not import pipelines: `test_layering`); the stage imports it from there.
- Names are checked: a factor label not in `LABELS` (except `probe.*`, whose labels are family-level readouts) or a `gen` that
  is neither a TRANSFORM nor a scene PART raises `RelgenError` (F1 note: the silent `TRANSFORMS.get` skip is gone).
- Trainer hook (`harness/data/mix.py`): `batches = relation_batches(rc, out_dir, main_batches)`. Needs `params.curriculum`
  (`shards`: the `relations_data` `<run>/relgen` dir(s); optional `factors`; any `SchedulerConfig` key such as `interval`,
  `share_min`, `share_max`) and `batch_size`. Each batch is `{"step", "main", "relgen", "counts", "active", "labels"}`; `labels`
  is ready for `batch.extra["relation_labels"]`. The trainer feeds `batches.observe(step, {factor: {"competence": ..}})` or
  `batches.observe_estimates(step, metrics_of_estimates_loss)`. Every decision appends `<out_dir>/schedule.jsonl` with the
  steers applied and metrics observed since the last one, so `Scheduler.replay_records(cfg, seed, records, steps, n)` reproduces
  the composition exactly (tested); `start_step > 0` resumes and re-decides from the earlier records.
- `rrp steer <run>` with no op lists `<run>/steer.jsonl` as pending / applied at step N / REJECTED; `rrp steer <run> <op>`
  appends as before. An unparseable line is logged as a rejected steer, not applied.
- `rrp run-dag` now loads the pipeline modules before planning (`cli/dag.py`), otherwise a stage a module registers on import
  (this one) is unknown to the recipe planner.
- OBSERVATION GAP: closed for the arm family by the R2 RG note below (shard rows now carry the featurizer input).
- Not run: no peer smoke of the real collector path (the host does no simulation); the tests drive `SnapshotCollector` with a
  fake env and the stage with a fake collector.

## R2 RG: relgen reaches the arm trainers (readiness round 2)

Closes the R1 observation gap for the arm family and the hook wiring (open item 1).
- Forwardable rows: a shard row stores the snapshot's `PolicyInput` arrays plus its ctx token entity ids (`relgen-shard-2`; a
  shard-1 manifest or a row without `policy_input` is refused, no reader for the old layout). `SnapshotCollector` takes the
  observation from `on_reset(obs)` / `on_step(step.observation)` and featurizes it once per episode with the env's
  `featurizer_for`; morph tokens map to assemblies through the inverse `feat.bindings`, scene tokens to `view.token_entity`,
  task/interact tokens carry no entity. `relations_data_stage` collects inside `apply_run_context(rc)` (the run's kinfeat).
  `label_episode` calls `transform_def(name)`, so an unknown generator name raises instead of being skipped.
- `collate_rows(rows, family)` (`harness/data/mix.py`) builds one Batch through `collate_inputs` and scatters each row's labels
  into the padded ctx layout (`extra["relation_labels"]["ctx"]`); a non-ctx label raises. `RelationBatches` (from
  `relation_batches(cfg, out_dir, specs, batch_size=, family=, main=, start_step=)`; None without `params.curriculum`) gives
  `draw(dev) -> (record, shard batch | None)` and `loss(rc, step)` = `estimates_loss` over the shard rows, then
  `observe_estimates(step, metrics)`. A scheduled factor whose net wrote no estimate (source `given`, or no readout) raises
  ("give the factor `source: probe`"): it would otherwise train at zero loss without anyone noticing. `inputs.relgen` without
  `params.curriculum`, and `prefetch`, raise.
- Trainers (`latent_train.py` stage A and flow, `behavior.py`): each step the main batch is the scheduler's `counts["main"]`
  rows of the trainer's own data (flow / stage A) and the factor loss is `(Bm*L_main + n_shard*L_factor) / B`; the shard rows go
  through the same context forward (`E.context` for stage A, `FlowPolicy.prepare(assembly_batch(b)).rc` for flow and BC) with NO
  action / flow loss, so the action loss of a mix row is zero and the factor loss is on. Logs gain `relgen`, `n_relgen` and the
  `probe_<factor>` terms; `schedule.jsonl` / `steer.jsonl` are in the run dir and `start_step` resumes.
- Recipes: `relations_factor.yaml` per set F0 has `inputs.relgen: '@relgen:relgen'` and `params.curriculum` (geo:
  depth3d, normal_align; ix: contact, held_by, support; task: next_contact; `interval` 1000 = the snapshot cadence so a resume
  loses no metrics, `ramp_steps` 20000). Stage A is not fed in the template (its E has only `preset:arm`); the trainer supports
  it through `encoder_factors`. `recipes/relations/relations_geo_smoke.yaml` is the plumbing smoke.
- Tests (`tests/unit/test_relgen_trainers.py`, `test_relgen_stage.py`; CPU, about 13 s): 3 steps each of stage A, flow and BC on
  a tiny pack plus real shards from fixture sessions: factor loss > 0 and differentiable, no labels in the action-loss batch,
  `schedule.jsonl` steps `[0, 2]`, `Scheduler.replay_records` reproduces the composition; the recipes feed F0 and the F0 lists
  build in the arm family.
- Peer smoke (`relations_geo_smoke`, `RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/r2-rg`): relgen on 6 teacher episodes x 3
  snapshots wrote 72 rows (geo.depth3d / normal_align / orient / pos3d 18 each; geo.above is not a producer) in 11 s; F0 (120
  steps, batch 16, width 64, the armdiv semfix stage-A representation, v7div pack) completed in 11 s with `schedule.jsonl` at
  steps 0..100 (shares 0.4 -> 0.16 as the full-world floor ramps) and the step-100 record `n_relgen 7`, `relgen 327.7`,
  `probe_geo.depth3d 329.2`, `probe_geo.normal_align -1.4`, flow 1.6. Plumbing only: a 120-step net is not a result. The first
  launch found that `rrp stage run` validates the RunConfig before loading the pipeline modules, so the stage `relations_data`
  is unknown to a leased job (`base.py`, not RG's): the smoke used a peer-side `sitecustomize` preloading
  `rrp.harness.pipelines.relations` (RRP_PEER_PYTHONPATH, not in the repo).
- Open: (1) the init-time factor loss (about 330 for depth3d) is two orders above the flow loss: a loss weight
  (`curriculum.weight`) is probably needed before the real run; (2) `observe_estimates` reads `<factor>_acc` only, so the
  field-estimate factors (depth3d, normal_align, pos3d, orient) give the scheduler no competence signal (`metrics` is empty in
  `schedule.jsonl`); (3) `FlowPolicy.loss` discards the `estimates_loss` metrics, so the trainer does a second
  prepare + estimates pass on the shard rows only (`nets/flow.py` is RC's); (4) `train_bc`'s arm.py config ignores `inputs`, so
  BC shards come through `params.curriculum.shards`; the BC main batch cannot be resized (shard share weights the loss only);
  (5) task / interact tokens carry no entity id; (6) legged / HD2 need their own collate (HL).
