# .old/tests — retired parity tests and their frozen scripts (D-145 P7)

| path | what it was | cited by |
|---|---|---|
| `tests/unit/test_ladder_cli_parity.py` | D-126 move parity: ran the frozen pre-move scripts and their `rrp suite ladder` / `legged-*` ports as subprocesses on identical inputs and compared stdout + output files | `research/tracks/d126_deploy.md`, `d126deploy-ladder.md` (D-126) |
| `tests/data/legacy_scripts/` | `ladder.py`, `legged_ladder_summary.py`, `legged_edit_effects.py`, `legged_mirror_effect.py`: the scripts frozen verbatim at git 733b02a | the test above |

Why retired: the ports were verified equal when the scripts were removed (D-126) and are pinned by `tests/data/golden.json`
(`test_golden.py`); the scripts themselves are no longer part of the repository schema. Not collected by pytest
(`testpaths = ["tests"]`). Moved in by D-145 unit P7.
