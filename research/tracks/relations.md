# track: relations (open) — relation-factor experiments (D-144)

The relation-factor registry (D-144, `docs/relations.md`, catalog `research/relations_catalog.md`) is built: token model,
`FactorSite` in the shared attention block, `ReadoutProbe`, `relgen` labels / parts / transforms, curriculum scheduler,
deploy guard, room panels. What is missing is the EXPERIMENT: no factor set has been trained against the `preset:arm`
control. That is this track. State: **planned** (recipes render and plan dry; nothing has run).

Per-unit history of the build (R1-R21, rel-geo, sweep-flags, closures) is in `research/decisions.md` D-144 and its
addenda; the unit notes `research/tracks/rel-*.md` and `sweep-flags.md` are closed-track material (moved to `.old/` by P5).
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
| `recipes/relations/relations_task.yaml` | `preset:task` | `task.next_contact time.same_track` |

`preset:ui` (ComputerWorld widget tree) is a pointer-track factor set: it is a `factors:` var of the pointer recipes
(`recipes/pointer/`), not an arm instance. Legged / humanoid sets (`legged-r19`: `leg.foothold leg.com_support`) run through
`recipes/templates/legged_lineage.yaml` once a body set is chosen; no instance yet.

CAVEAT recorded in the DAG ledger: structural side only. `relations_data` (labelled shards + `harness/data/mix.py`) is a
library function, not a DAG stage (open item 1), so probe-sourced factors (`geo.depth3d`, `ix.*`, `task.next_contact`)
train without their label shards: their estimate heads start at zero and an `aug` factor is a no-op at step 0. Until item 1
lands, a positive result would show only what the structural inputs give; a null is NOT evidence against the factor.

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

1. **`relations_data` is not a stage** (R10): the shard writer / reader and `mix.mixed_batches` exist, but `"relations_data"`
   is not in `PIPELINE_STAGES` and no training stage reads `artifacts/relgen/` shards. Needs `core/runconfig.py`
   (stage list, `register_family` validation) + `harness/pipelines/base.py` registration + a `train_flow` option that mixes
   shards. Blocks every probe-sourced factor experiment above.
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
