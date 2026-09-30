# recipes

A recipe is a `dag-1` DAG file (`harness/yamlmini.py` subset, loaded by `harness/dag.py::load_dag`) that plans a
lineage: a matrix of (variant x seed) points over pipeline stages. Run one with `rrp run-dag <path-or-name>`; a name is
resolved under `recipes/` (`rrp run-dag armdiv/arm_lineage_v7div --dry-run`).

- `templates/` — generic, per-family recipes (`arm_lineage`, `arm_collect`, `arm_bc`, `arm_grpo`, `arm_targets_*`,
  `dual_lineage`, `legged_lineage`, `legged_heldout`, `tracker_gated`, `relations_factor`). Never run bare; an instance extends one.
- `presets/` — shared parameter fragments read by code, not by a DAG: `policy-small-structured.json`, `codec-small.json`
  (the baseline campaign's model sizes), `eval-latent_slice1.json` (the sealed target-eval protocol; byte-frozen, its
  sha256 is recorded in eval results), `eval-primary.json` (draft protocol, unsealed).
- `<track>/` — thin instances: a header (`name schema track policy env task bodies factors`), `extends: ../templates/...`
  and only the overrides. `extends` may name a sibling instance (`*_kinfeat`, `*_smoke`) or a list, folded left:
  dicts merge, lists and scalars replace, `null` deletes.
  - `armdiv/` — the v7div lineage, its data collection and BC baselines (+ `kinfeat` and `smoke` variants).
  - `relations/` — the relation-factor experiment (D-144): `geo`, `ix`, `task` factor sets vs the `preset:arm` control on
    one shared representation (`research/tracks/relations.md`; planned, structural side only).
  - `humanoid/` — the four D-126 tracker recipes (`track: trackers` stays in the header because `rc.track` is part
    of the rendered digest; P4a re-homes it under humanoid).

Every recipe here renders to a pinned digest: `recipe.<name>` in `tests/data/golden.json`, checked by
`tests/unit/test_recipes.py` (which also dry-run plans every file). The pre-schema per-lineage DAG copies that these
replace are in the legacy area (index at the repo root), with the recorded path mapping.
