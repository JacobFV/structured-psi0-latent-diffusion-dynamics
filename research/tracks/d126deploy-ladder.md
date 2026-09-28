# d126deploy-ladder (D-126 sub-track: ladder / legged summaries into the library)

- `scripts/ladder.py` main()/render() -> `rrp.evaluation.ladder_cli` (`build_parser`, `main(argv=None)`, `render`); the
  script is a 1-line wrapper. New code paths: `rrp.contracts.workload.apply_cap` (not the rrp.ops.gpu shim).
- Seeds: evaluation may not import training, so ladder_cli uses `rrp.evaluation.robustness.feasible_arm_seeds`, whose body
  was already identical to `rrp.training.latent_grpo.feasible_seeds`; the latter now delegates to it (one definition).
- `caption` -> `rrp.evaluation.captions` (identical code; `scripts/render_episode.py` re-exports it, so
  render_causal_edit.py / research.bc_semantic_edits keep working).
- `scripts/legged_ladder_summary.py`, `legged_edit_effects.py`, `legged_mirror_effect.py` -> `rrp.evaluation.legged_summaries`
  (`ladder_summary_main`, `edit_effects_main`, `mirror_effect_main` + helpers). The wrappers prepend their checkout's
  `src` to sys.path (the old scripts needed no PYTHONPATH). Bootstrap CIs: one rng(0) stream per main() call, consumed in
  the old order (identical to the old per-process module-level rng for one invocation).
- `scripts/legged8_summary.py` stays a script: one-off W8 report (hard-coded headline/protocol text), not reusable.
- Pipelines unchanged (they still run the scripts).
- Parity: `tests/unit/test_ladder_cli_parity.py` runs the frozen pre-move scripts (tests/data/legacy_scripts, git 733b02a)
  and the new wrappers: ladder teacher route (panda_pg2, n=1, 30 steps; 2 arg sets) -> identical stdout, summary.json bytes,
  rows equal except `wall_s`; parser --help / guard exits identical; legged tools on synthetic rows -> identical stdout and
  output files byte for byte.
