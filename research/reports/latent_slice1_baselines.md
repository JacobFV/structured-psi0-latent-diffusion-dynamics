# latent_slice1: baseline methods (direct action, action-only codec)

Status: RUNNING (2026-09-25). No result yet. This file is filled from
`python -m rrp.evaluation.latent_slice1_report` once cells finish. Every number will come from
`artifacts/runs/latent_slice1/<method>/seed<k>/eval/*.summary.json` (raw episode rows in `*.jsonl`).

Protocol: `configs/eval/latent_slice1.json` (SEALED_BEFORE_RESULTS, unchanged). Methods here:
`baseline_direct_action` (FlowPolicy latent_dim 1 + hidden-state auxiliaries) and `baseline_action_only_codec`
(frozen per-seed action codec + FlowPolicy on its latent). Both use the native ActionChunk path, and the controller
source is `learned:<checkpoint>`. These are BASELINES, not the corrected architecture.

Grid: seeds 1701/1702/1703 x targets xarm7_pg2/xarm7_tf3/panda_tf3 x SFT budgets 0/5/20/100 target demo episodes;
100 eval episodes per cell (scene seeds 2,000,000..), Wilson 95% CI; source competence on parm5s_tf3 + parm5l_pg2
(50 each). Budget 0 is the zero-shot transfer of the source-trained controller. Budgets > 0 fine-tune that
controller on target demos, with the same acquisition as the latent methods (same demo episodes, transitions,
150/300/600 optimizer updates). Fairness details and deviations: `research/tracks/baselines.md`.
