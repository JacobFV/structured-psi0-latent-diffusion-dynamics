# track: rel-r9 (relation-factor fanout, unit R9: generic transforms)

Branch `track/rel-r9`, worktree `~/work/rrp-wt/rel-r9`, from `origin/main` @ `c5e7d45` (D-144 foundation). Spec:
`docs/relations.md` sections 2-5 and 10 (row R9), `research/relations_catalog.md` J "epistemic" row. Host only
(git / editing / unit suite; CUDA hidden); no peer smoke needed (the row's acceptance is pure unit math, no training).

## scope

Implemented `src/rrp/harness/data/relgen/transforms.py`: the six generic, factor-agnostic `TransformDef.fn`s named
in docs/relations.md 5.2, registered into `TRANSFORMS` under their doc names (`reveal`, `surprise`, `cf_swap`,
`noise`, `occlude`, `subsample`), each `(sample, rng, params) -> list[sample]`, pure (never mutates its `sample`
argument -- verified by a `_deep_eq` check against a pre-call deep copy in every purity test) and seed-deterministic
(only source of randomness is the `rng` argument). Did not touch `relations/base.py`, `relations/ops.py`, or
`harness/data/relgen/__init__.py` (only imported from it: `Sample`, `TransformDef`, `register_transform`) --
no operator/interface change was needed for this row.

- **`reveal(schedule)`**: Bayes posterior `q_t(e) ~ prior(e) * 1[e consistent with evidence <= t]` (doc 5.4, exact
  formula) over a toy candidate set + ordered exclusion `evidence`; `schedule` defaults to the evidence times.
- **`surprise(rate, after)`**: reuses `reveal`'s posterior machinery; once it collapses (`max q >= collapse` thr,
  default 0.999) onto one candidate, at each schedule step `>= collapse_t + after` draws `rng.random() < rate`; on
  the first hit it revises belief (the collapsed candidate is now known wrong, so the survivor set reopens to every
  OTHER candidate, or to `params["flip_survivors"]` if given) and records `switch_at` in the provenance + the label.
  `recovery_steps(target, estimate, switch_index, eps)` is a separate pure metric (steps from the switch until
  `KL(target_t || estimate_t) < eps`, or `None`) -- it consumes a `reveal`/`surprise` target stream and a caller-
  supplied model-estimate stream, so it stays a pure function with no model dependency here.
- **`cf_swap(field)`**: swaps two token slots of one token set (picked via `params["pair"]` or, by default, two
  valid slots drawn from `rng`) consistently across every field of that token set, every edge whose query/key axis
  is that set (rows/cols `i, j` swapped), and every arity-2 label over that set -- so a binding / id swap moves
  identity, position and every other attribute together, matching "today's `binding_cf` is `cf_swap('binding')`"
  (docs/relations.md 8.1).
- **`noise(field, sigma)`**: adds `Normal(0, sigma)` to a token-set field and writes `<field>.var = sigma**2`,
  masked by `<field>.valid` so invalid/null slots are untouched.
- **`occlude(p)`**: each valid token becomes a null identity independently with probability `p` (mask cleared, slot
  kept per the token model in section 2, not deleted).
- **`subsample(stride)`**: keeps every `stride`-th knot index and remaps `evidence` event times into the kept index
  space (an event between two kept indices attaches to the next kept index, so no event is silently dropped).

Sample/field shapes read by each transform are documented in the module docstring; no consumer exists on `main` yet
(R11/R18 are the eventual callers per the fanout table), so these shapes are this row's own design choice, kept
close to the `Sample = {"inputs", "labels", "provenance"}` alias already in `relgen/__init__.py`.

## tests (red then green)

`tests/unit/test_relgen_transforms.py`, 25 cases. Verified red first: moved the new module aside and confirmed
`pytest tests/unit/test_relgen_transforms.py` fails on `ModuleNotFoundError` (ImportError), then restored it and
iterated to green.

- `test_reveal_is_bayes_posterior` / `_respects_nonuniform_prior` / `_default_schedule_is_evidence_times`: hand-
  computed posteriors on a 3-candidate toy set against `docs/relations.md` 5.4's exact formula.
- `test_surprise_flips_after_collapse_deterministically` / `_rate_zero_never_flips`: collapse detection + the flip
  step + the post-flip posterior, hand-computed; `test_recovery_steps_counts_until_kl_below_eps` /
  `_never_converges_is_none`: the recovery metric on synthetic target/estimate streams.
- `cf_swap`: token fields + edge rows/cols + arity-2 label all swap consistently for an explicit pair, and an
  unspecified pair is reproducible from a fixed `rng` seed.
- `noise`: additive noise present only on valid slots, `.var` set correctly, seed-deterministic, `sigma=0` is a
  value no-op.
- `occlude`: `p=0` / `p=1` edge cases exact, `p=0.5` seed-deterministic, provenance's `occluded` list matches the
  actual mask.
- `subsample`: stride keeps the right indices, evidence times remap without dropping events, stride 1 is identity.
- Every transform: a purity check (`_deep_eq(sample, before)` after the call -- plain `==` raises on dict-of-ndarray,
  hence the helper) and a provenance-appended check.

## commands run

```
cd ~/work/rrp-wt/rel-r9 && export PYTHONPATH=$PWD/src:$PWD
~/work/relational-robot-policy/.venv/bin/python -m pytest tests/unit -q      # baseline (pre-change): 551 passed, 39 skipped, exit 0
~/work/relational-robot-policy/.venv/bin/python -m pytest tests/unit/test_relgen_transforms.py -q   # red: ModuleNotFoundError
~/work/relational-robot-policy/.venv/bin/python -m pytest tests/unit/test_relgen_transforms.py -q   # green: 25 passed
~/work/relational-robot-policy/.venv/bin/python -m pytest tests/unit -q      # full suite post-change: 576 passed, 39 skipped, exit 0
```

State: **completed** (implementation + tests green; merge to main pending per docs/relations.md 10's merge rule).
