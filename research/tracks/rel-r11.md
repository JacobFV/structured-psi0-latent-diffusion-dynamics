# rel-r11 track (D-144 fanout unit R11: responsive, steerable scheduler + composition + interference)

Owner: Sonnet fanout agent. Worktree `~/work/rrp-wt/rel-r11`, branch `track/rel-r11` from `origin/main` @ `9be709dd`
(D-144 foundation + R1/R3/R9/R10 merged). Brief: `docs/relations.md` section 10, row R11 + its brief paragraph
(section 5.5 is the design this unit implements). Host only (git / editing / unit suite; CUDA hidden; no
training/simulation beyond unit-test fixtures): the row's acceptance is entirely unit-test fixtures (a simulated
competence curve, planted struggling/interfering factors, a replay check), so `scripts/peer_run.sh` was not needed.

## what changed (owned files only)
- `src/rrp/harness/data/relgen/curriculum.py` (body only -- the F4 dataclass shapes, steer grammar, `Scheduler.steer`
  / `.validate` / `._apply_steers` and the replay contract are untouched):
  - `Scheduler._raw_signal` / `._level_and_reasons`: per factor, replays every decision-interval window up to the
    current step from `self.metrics_log` (never cached on `self`, so `decide()` stays a pure function of its inputs
    -- required by the replay contract) and EMA-smooths (`cfg.ema`) competence / plateau / interference / attributed
    failures. A factor `observe()` never reports for is "unseen": it keeps level 1 and the old F4 placeholder's flat
    weight (this is what keeps `tests/unit/test_relgen.py`'s F4 tests, which never call `observe()`, passing
    unchanged). A gap window (no observation that interval) holds the EMA rather than decaying it toward 0.
  - Promotion: 2 consecutive intervals with EMA competence >= its threshold (`cfg.promote`, default 0.8, the
    doc's own example number). Demotion: the first interval it drops below `threshold - cfg.hysteresis`.
  - Drop-back: `_DROPBACK_INTERVALS` (3) consecutive intervals of high struggle score (`(1-c)+0.5p+i+e`) with
    positive interference drop the level by one and CAP it there (promotion skipped, not just delayed) until BOTH
    `_DROPBACK_COOLDOWN_INTERVALS` have elapsed AND the struggle score has actually recovered under the threshold.
    A fixed timer alone was tried first and produces a genuine infinite oscillation when interference stays high
    while competence stays high too (verified empirically, then fixed -- see "what I found empirically" below);
    recovery-gating is what actually makes "hysteresis prevents oscillation" (5.5) true rather than just slower.
  - `decide()`: shares are now `share_min + struggle_score` (falls back to the flat `1.0` baseline for an unseen
    factor) instead of a flat `1.0`, then clipped to `[share_min, share_max]` and renormalized exactly as F4 did,
    then a NEW rate-limit pass caps the per-decision absolute change to `cfg.max_step_change` from the previous
    share (frozen factors, which already carry their previous share forward, are exempt -- they don't move at all).
    `st.signals` is populated per factor (`{competence, plateau, interference, failures, struggle}`) and `st.reasons`
    collects this decision's promote/demote/drop-back events (only this decision's, not the whole replayed history
    -- see the empirical-bug note below).
  - `interference(records, f, g, metric)` / `max_interference(records, factors, metric)` / `interfering_pairs(...)`
    (module-level functions, not `Scheduler` methods): `interference(f, g) = metric_f(sets with f, without g) -
    metric_f(sets with f and g)` over caller-supplied eval records (5.5's formula, verbatim). A caller feeds
    `max_interference(...)`'s per-factor result into `observe(step, {f: {"interference": ...}})`, closing the loop
    into the struggle score above.
- `src/rrp/harness/data/relgen/__init__.py`: `compose(parts, env, rng)` body (the dataclasses / registries / other
  functions are the F4 foundation, untouched). `parts` is the desired ACTIVE SET (e.g. one of `Scheduler.sample`'s
  active-set entries), not a hand-picked list of `ScenePart`s -- matching 5.5's "compose picks the parts that cover
  each set". `_min_cover` finds the smallest-cardinality subset of `PARTS` (restricted to the env) whose `activates`
  realize the target (exact search below 12 candidates, deterministic greedy above); `compose` then closes
  `requires` transitively by repeatedly covering whatever is still missing (never silently dropping a requirement,
  raising `ComposeError` if nothing can supply it), rejects any chosen part whose `conflicts` collides with the
  final active set, builds every chosen part in deterministic (sorted-name) order, resolves a simple generic
  non-overlapping layout pass over any entities carrying `pos`/`radius`/`extent` (`_resolve_layout`), and merges +
  dedupes every part's task events (`_merge_events`).
- `src/rrp/cli/curriculum.py` (new): the two CLI surfaces the row names.
  - `register(sub)` adds the top-level `rrp steer <run> <op...>` command (same pattern as `cli.dag`'s `run-dag`,
    not a `(group, name)` `TOOLS` entry, because the doc's own grammar is `rrp steer <run> <op> ...`, not `rrp
    steer <subcommand>`): validates the op parses (`parse_steer`), stamps author/reason and appends it as one JSON
    line to `<run>/steer.jsonl`. Full semantic validation (unknown factor, bad bounds, ...) happens where it always
    did, in `Scheduler.validate` at the run's next `decide()` -- this command only writes the line, per 5.5
    ("picked up at the next decision interval; no restart, no code").
  - `suite_main(argv)`, registered as the `("suite", "relations-curriculum")` `rrp.cli.tools.TOOLS` entry: reads a
    run's `schedule.jsonl` (one `Scheduler.export`-written `ScheduleState` per decision) and prints the schedule
    history with its reasons, competence by factor x composition depth (from `ScheduleState.signals`), and (given
    `<run>/eval_records.jsonl` or `--records`) interfering pairs via `interfering_pairs`.
  - `src/rrp/cli/main.py` / `src/rrp/cli/tools.py`: minimal, disjoint edits only -- `main.py`'s command-module
    import tuple and registration loop gained `curriculum` (one module added, same list every other command module
    is already in); `tools.py`'s `TOOLS` dict gained exactly one new key,
    `("suite", "relations-curriculum")`. Nothing else in either file was touched.
- `tests/unit/test_curriculum.py` (new, 27 tests, all red before this change: `compose` raised `NotImplementedError`
  and the scheduler had no promotion/drop-back/interference/CLI code to exercise). `tests/unit/test_relgen.py` (F4's
  own tests: `test_allocate_is_exact`, `test_steer_grammar_and_validation`, `test_scheduler_replay_and_floor`,
  `test_readout_probe_equals_packet_probe`) is untouched and still green -- it is not this unit's file.

## what I found empirically (worth recording: two real design bugs the naive first cut had)
Both were caught by actually running small scripts against the implementation before finalizing the tests, not by
reasoning about the formulas on paper -- the EMA / cooldown interaction is not obvious from the doc prose alone.
1. **Reasons were replayed, not reported.** `_level_and_reasons` has to re-walk the whole trajectory from step 0
   on every `decide()` call (nothing is cached on `self`, to keep the replay contract a pure function of its
   inputs) -- but the first cut returned every historical promote/demote/drop-back message it walked past, so
   `st.reasons` (and the `rrp suite relations-curriculum` report) re-announced the SAME old event on every
   subsequent decision. Fixed by keeping only the last observed window's reasons.
2. **A fixed-duration drop-back cooldown oscillates forever under sustained interference.** First cut: drop back
   one level, hold for `N` intervals, then resume normal promotion/demotion. Empirically (see the throwaway script
   in the commit history / rerun it: hold a factor's competence high while its interference/plateau/failures stay
   maximally high forever), competence alone re-earns 2-consecutive-interval promotion the moment the cooldown
   timer expires, and 3 struggling intervals later it drops back again -- repeating indefinitely
   (`level: 2,2,2,2,2,2,2,1,1,1,2,2,1,1,1,2,2,1,1,1,...`). Fixed by making the cooldown a CAP (promotion skipped
   entirely, not just delayed) released only once BOTH the minimum cooldown has elapsed AND the struggle score has
   actually fallen back under threshold -- i.e., recovery-gated, not timer-gated. Re-running the same adversarial
   scenario after the fix: exactly one drop-back event, held at the lower level permanently (struggle never
   actually recovers in that synthetic scenario, which is the correct outcome).

## acceptance evidence (row R11, docs/relations.md section 10)
Command: `cd ~/work/rrp-wt/rel-r11 && export PYTHONPATH=$PWD/src:$PWD && .venv (main checkout)/bin/python -m pytest
tests/unit/test_curriculum.py -v` -> 27 passed. Full suite: `pytest tests/unit -q` -> see command output recorded
in the merge section below.
- **"`compose` closes requires, rejects conflicts, minimal cover"**: `test_compose_minimal_cover` (a single part
  covering both target dynamics beats combining two single-purpose parts), `test_compose_closes_requires_transitively`
  (a two-hop requires chain is closed automatically), `test_compose_rejects_conflicts` /
  `test_compose_unsatisfiable_requires_raises` / `test_compose_no_covering_part_raises` (`ComposeError` in each
  case), `test_compose_env_compatibility`, `test_compose_empty_active_set_is_a_noop`,
  `test_compose_merges_and_dedupes_events`, `test_compose_resolves_non_overlapping_layout`,
  `test_compose_is_deterministic`.
- **"default trajectory -> mostly full-world on a simulated competence curve"**:
  `test_default_trajectory_promotes_to_full_depth_and_mostly_full_world` (3 factors on a 60-interval rising
  competence curve all reach max composition depth; `full_world` reaches >= 0.75 and never decreases).
- **"a planted struggling factor gets boosted within bounds, no oscillation"**:
  `test_struggling_factor_share_rises_bounded_and_rate_limited` (share rises from baseline to the `share_max`
  ceiling in exactly `max_step_change`-sized steps, monotonically, never promoted) and
  `test_dropback_holds_the_level_for_its_cooldown_no_oscillation` (adversarial: competence stays high while
  interference stays maximal for 34 intervals past the promotion; exactly one drop-back, held permanently -- the
  empirical bug fix above, now red/green-pinned).
- **"replay reproduces sampling byte for byte with a steer log"**:
  `test_replay_reproduces_sampling_with_observed_metrics` (with `observe()` calls AND a boost steer in the log,
  unlike F4's own replay test which has neither).
- **"invalid steers rejected and logged"**: `test_invalid_steer_is_rejected_and_logged_not_applied` (this unit's
  own coverage of the row's acceptance line; the mechanism itself is F4's `validate`/`steer`, untouched).
- **"interference flags a planted pair"**: `test_interference_detects_a_planted_pair` (a clearly-interfering pair
  is flagged above threshold and ranked first; a non-interfering pair is not flagged), plus
  `test_interference_none_without_comparable_records` and
  `test_interference_feeds_the_struggle_score_via_observe` (the intended end-to-end loop: `max_interference` ->
  `observe` -> a larger share for the interfering factor).
- **`rrp steer` / `rrp suite relations-curriculum`**: `test_cli_steer_appends_a_json_line_cli_grammar`,
  `test_cli_steer_appends_a_json_line_json_form_with_author_reason`, `test_cli_steer_rejects_unparseable_op`,
  `test_cli_suite_relations_curriculum_reports_history_and_interference`,
  `test_cli_suite_relations_curriculum_handles_no_schedule_yet`, `test_cli_registers_top_level_steer_command`,
  `test_suite_relations_curriculum_is_a_tool_entry`. (`argparse.REMAINDER` swallows trailing flags, discovered by
  actually running the command, not just the mocked unit tests -- see `cmd_steer`'s docstring note: `--author` /
  `--reason` flags were dropped in favor of the op's own JSON form, which is the only way to set them.)
- **`schedule.jsonl` export**: `test_export_writes_one_jsonl_line_per_decision`.
- **pins / freezes still work under the new policy**: `test_pins_and_freezes_override_the_policy`.
- **an unobserved scheduler is unchanged from F4's placeholder**:
  `test_never_observed_factor_keeps_level_one_and_flat_baseline_share` (guards against a future edit accidentally
  changing the no-`observe()` baseline and silently breaking `test_relgen.py`'s F4 tests).

## merge fixup: `compose` test isolation (found by the merge's own rebase/rerun cycle)
While rebasing onto commits landed by other units mid-merge, `pytest tests/unit -q` picked up one failure:
`test_compose_closes_requires_transitively` (passed alone, failed in the full suite -- classic test-order-dependent
global-registry pollution). Cause: R16/R17 had by then merged real `PARTS` entries (`grasp_target` activating
"contact", `stack` activating "support"/"force_flow", both for `envs=("mujoco/arm", "mujoco/dual")`) registered as
an import-time side effect in `harness/data/relgen/{contact,support}.py`; some other test module imports one of
those at collection time, so by the time `test_curriculum.py`'s compose tests ran later in the full suite, the real
`PARTS` registry (process-global, no per-test reset) already had entries whose generic-sounding activation tags
("contact", "support") and env ("mujoco/arm") collided with the ones this unit's fixtures used, so `_min_cover`
picked a different (still-valid) cover than the test's hard-coded expectation. Fixed by namespacing every fixture
part name, activation tag and env string used by `compose`'s tests (`_ut11` suffix / `unit-test-env_ut11`), and by
making `parts_sandbox` save and restore (not just delete) whatever it shadows in `PARTS`, so it is now safe even if
a name collides with something real. Re-ran the full suite clean (below) before merging. This is exactly the kind
of thing "rerun the unit suite if the rebase brought new commits" is for.

## merge
Command sequence (executed after the unit suite above passed): `pytest tests/unit -q` (full suite, exit 0) ->
`git add -A && git commit` -> `git fetch origin && git rebase origin/main` -> re-run the unit suite if the rebase
brought new commits -> `git push origin HEAD:main`. Exact commands, exit codes and the final commit sha are recorded
by the lead / harness that ran this track (this file is written before the merge step, as instructed).
