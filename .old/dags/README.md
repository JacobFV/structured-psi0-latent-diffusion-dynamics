# .old/dags — per-lineage DAGs of the pre-schema layout (D-145 P1)

What it was: one hand-copied `dags/*.yaml` per lineage, plus `dags/templates/` and `dags/overlays/arm_recipe/`
(`extends` fragments). Now: `recipes/templates/` + thin instances under `recipes/<track>/`; the rendered plans of the
kept ones are pinned by `recipe.*` goldens in `tests/data/golden.json` (recorded before the move).

Path mapping (this dir keeps the old path, so a decision citing `dags/x.yaml` resolves as `.old/dags/x.yaml`):

| old path | now |
|---|---|
| `dags/arm_lineage.yaml` | `recipes/templates/arm_lineage.yaml` |
| `dags/templates/{arm_grpo,arm_targets_bc,arm_targets_latent,dual_lineage}.yaml` | `recipes/templates/<same>.yaml` |
| `dags/templates/legged_v2_gated.yaml` / `legged_v2_heldout.yaml` | `recipes/templates/legged_lineage.yaml` / `legged_heldout.yaml` |
| `dags/templates/tracker_recipe_gated.yaml` | `recipes/templates/tracker_gated.yaml` |
| `dags/arm_lineage_v7div{,_kinfeat,_smoke}.yaml`, `dags/armdiv_{data,bc}_v7div*.yaml` | `recipes/armdiv/` (chain `v7div -> v6 -> v2 -> arm_lineage` flattened into the instance; data/bc got `templates/arm_collect.yaml` / `arm_bc.yaml`) |
| `dags/d126_tracker_*.yaml` | `recipes/humanoid/` |

Retired here (closed tracks, provenance only), all in this directory: `arm_lineage_v2`, `arm_lineage_v6` (armexpert /
arm v2, v6 lineages), `armexpert_v{4,5,6}dart`, `arm_targets_{d136_joint,v6_bc,v6_latent}` (armdiag / d126 arm
targets), `legged_v2_{anymal,anymal_smoke,go2,t1,t1sl}`, `legged_fixrep`, `smoke_legged` (W8 legged8), `parity_arm`
(the sfjf2 parity DAG), and `overlays/arm_recipe/*` (the `extends`-fragment demo; see its README).

Note: the `extends:` paths inside these retired files are not repaired (they pointed at `dags/templates/`); the templates now live under `recipes/templates/`.

Cited by: D-080..D-145 in `research/decisions.md` (lineage DAG names), `research/tracks/{armdiag,d126_arm,d126_legged,
legged8,sweep-flags,rel-r2c}.md`. Their reruns (`rrp run-dag dags/legged_v2_t1sl.yaml ...`) need the path `.old/dags/...`
and only work from a checkout that still has the retired-track code paths; the ledgers under `artifacts/runs/*/_dags/`
are the record.
