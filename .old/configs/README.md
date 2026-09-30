# .old/configs — hand-written per-run JSON configs of the pre-schema layout (D-145 P2)

What it was: one JSON file per run stage (`configs/<group>/<stage>_<lineage>.json`), written or codemodded by hand and
read by the stage functions through `RunConfig.load_legacy`. A run's identity was a file name. Now a run config is
RENDERED from `recipes/templates/` + a thin `recipes/<track>/` instance (docs/architecture.md section 13.2); the
`load_legacy` / `classify_legacy` / `LEGACY_FLAG_DEFAULTS` / `LegacyInfo` machinery was deleted with these files.

Path rule: this directory mirrors the old path, so `configs/ladder/x.json` cited by a decision or a run's provenance
resolves as `.old/configs/ladder/x.json`. 274 files here; the other 6 files of the old `configs/` tree were kept live:

| old path | now |
|---|---|
| `configs/model/policy-small-structured.json`, `codec-small.json` | `recipes/presets/` (same names; read by `harness/train/baseline_campaign.py`) |
| `configs/eval/latent_slice1.json` | `recipes/presets/eval-latent_slice1.json` (byte-frozen: its sha256 is recorded in sealed eval results) |
| `configs/eval/primary.json` | `recipes/presets/eval-primary.json` (DRAFT_NOT_SEALED protocol; dangling `configs/` references reworded) |
| `configs/run_index.json` | `artifacts/run_index.json` (`RunIndex.load` default) |
| `configs/resources.local.json` | `ops/resources.local.json` (`rrp.ops.runtime.CONFIG`) |

Pinned elsewhere: the `latent` block and frozen `LatentConfig.version()` of the 18 configs that construct a
`LatentConfig` are in `tests/data/latent_versions.json` (keys = the path below `configs/` without `.json`), so bundle
compatibility ids of existing checkpoints stay pinned without reading this directory.

| group | files | what it was | cited by |
|---|---|---|---|
| `adapt/` | 22 | target-body adaptation runs on xarm7_pg2 (EXPO / GRPO, prefix / suffix / exact / dimnorm, secondary-eval and start-selection variants v1..v4) | frozen runs `artifacts/runs/adapt_*`; D-126 arm track |
| `data/` | 10 | teacher data collection specs (pick_place v1..v3dart, handover, support_insert, assign) | D-012, D-014, D-016..D-019, D-022 |
| `ladder/` | 126 | the arm lineage ladder: `flow_*`, `rz_*` (refit), `rep-*` for jointfix / bindv4 / sem / nosem / semfix, plus `armnosem/`, `armsemfix/`, `armseed2/{nsjf2,sejf2,sfjf2}/`, `armnosemabl/{nsqd,nszn,nszq}/` seed and ablation sets | D-047..D-068 (jointfix), D-090, D-096 (armseed2), D-144 (factors codemod), D-145; recipes `arm_lineage` / `arm_targets_*` render the same plans (goldens `recipe.*`) |
| `latent/` | 29 | first-generation `pack-*`, `rep-*`, `flow_*` configs (latent v1..v3, binding, binding_paired, dualarm, target packs) | D-043, D-045, D-051, D-059, D-080, D-144 |
| `legged_bc/` | 4 | legged BC positive controls (g1, go2, hexapod6, t1) | W8 legged8 (closed) |
| `legged_dagger/` | 6 | legged refit (`rz_*`) DAgger rounds on T1 | W8 legged8 (closed) |
| `legged_fixsem/` | 20 | legged fixed-semantic lineage (`rep_*`, `flow_*`, go2 / hexapod6 / lv4) | D-088, D-090, D-092, D-106, D-144 |
| `legged_latent/` | 32 | legged latent v1 / v2 (`rep_*`, `flow_*`, nosem / sem, four bodies) | D-141, D-144 |
| `model/` | 8 | unstructured / medium / si- policy sizes and the adapt-start policies | early baseline campaign (D-014..) |
| `t1_diag/` | 13 | T1 diagnostic reps / flows / refits (sem lv4, w0, ctl, genz) | D-085, D-087, D-144 |
| `vlm/` | 4 | legged VLM cache, QA and the two VLM training configs | D-060..D-092 (legged_vlm) |

Rerunning any of these needs the path `.old/configs/...` and only works from a checkout that still has the legacy
`RunConfig` reader (before commit "D-145 P2"); the ledgers and `config.json` files under `artifacts/runs/*/` are the
record of what each frozen run used.
