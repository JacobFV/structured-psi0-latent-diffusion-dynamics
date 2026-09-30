# R4: legged nets on RelBlock / route.own_assembly / ReadoutProbe

Owner: rel-r4 fan-out agent (Sonnet). Worktree `~/work/rrp-wt/rel-r4`, branch `track/rel-r4`, from origin/main
`c5e7d45` (D-144 foundation). Scope: docs/relations.md section 10, row R4. Host only (git / editing / unit suite;
no training, no simulation, CUDA hidden); no peer smoke was needed (R4's acceptance criteria are all unit-fixture
checks, no smoke run listed in the brief).

## what changed (owned files only)

- `src/rrp/policies/nets/legged_latent.py`:
  - `block()` / `run_block()` deleted; `LeggedEncoder`, `LeggedRealizer`, `LeggedFlow` now build `RelBlock`
    (`rrp.policies.nets.attention`) and call it directly. `RelBlock`'s module keys (`n1,x,n2,s,n3,m`) match the old
    `ModuleDict`, so `blocks.<i>.*` checkpoint keys are unchanged (`test_relblock_module_keys_match_legacy_block`).
  - `LeggedRealizer`'s inline own/body-assembly `-inf` mask is now the factor `route.own_assembly` (new preset
    `legged-s0`, `params.also_key_field="body"`), applied through a `FactorSite("act>knots", ...)`. Knot validity
    (padded / absent assemblies) moved from being ANDed into the mask value to the cross-attention `kv_mask` — same
    combined -inf pattern (`test_route_own_assembly_matches_legacy_mask`, bit-exact against the pre-migration
    formula) since MHA adds bias and key-mask independently. `LeggedRealizer(factors=...)` accepts an override
    (default preset `legged-s0`); `control="off"` on the factor now drops routing entirely
    (`test_route_own_assembly_off_control_is_unmasked`), which the old hard-coded mask could not express.
  - `LeggedProbe` deleted; replaced by `legged_probe(...)` (builds a `ReadoutProbe` on the new preset
    `probes:legged-v1`) and `legged_probe_read(P, z, asm_mask, body_asm)` (calls `P(z, asm_mask)` — which reads
    `goal`/`disp`/`subtask`/`fall` at every assembly, address `"asm"` — then gathers each sample's body-assembly
    row). The gather is mathematically identical to the old per-sample query `code(asm_code[body_asm])`: `"asm"`
    computes `asm_in(asm_code[m])` for every `m` with the same map, and cross-attention queries don't interact, so
    computing every assembly and gathering afterward equals computing only the body one directly.
    `remap_legged_probe_state(state)` is the data-level key map for old `LeggedProbe` checkpoints (docs/relations.md
    section 4): `code`/`kcode` (Linear maps of the fixed `asm_code`/`knot_code` buffers) become `asm_in`/`kq_in`
    directly (same op, same buffer); the packet content's knot term `kcode(knot_code)` is baked once into
    `ReadoutProbe.knot.weight` (an embedding there instead of a linear map). Verified bit-exact
    (`test_legacy_legged_probe_checkpoint_loads_via_key_map`, atol 1e-5, float32 rounding only) and idempotent on an
    already-new-layout state dict (`test_remap_legged_probe_state_is_idempotent_on_new_layout`).
  - `gnll`, `probe_loss`, `probe_metrics` untouched (they only depend on the OUTPUT DICT shape, which
    `legged_probe_read` reproduces exactly) — `rrp.policies.psi0.nets` still imports `gnll` from here unmodified.
- `src/rrp/policies/nets/legged_bc.py`: `block`/`run_block` -> `RelBlock` (same drop-in swap, no mask/factor
  involved — this network has no routing restriction).
- `src/rrp/policies/relations/catalog.py`: added, in the pre-created "R1 / R4 / R5 / R6" probe section (disjoint
  from R1's arm block; no other unit's lines touched): preset `legged-s0` and the five `probe.legged.*` readout
  entries + preset `probes:legged-v1`. Did not touch `relations/base.py` or `relations/ops.py`.
- `src/rrp/policies/legged.py`, `src/rrp/policies/bundles.py` (`load_rep` only), `src/rrp/harness/train/
  {legged_latent_train,legged_bc}.py`: call-site updates for the `LeggedProbe` -> `legged_probe` /
  `legged_probe_read` rename, with every checkpoint load path going through `remap_legged_probe_state` first (a
  no-op on state dicts already in the new layout, so it is safe on fresh checkpoints going forward too).
  `harness/train/legged_bc.py` also dropped a dead `block, run_block` import it never used.
  `harness/train/legged_dagger.py` and `harness/eval/legged_latent_eval.py` needed no changes (neither touches the
  probe or `block`/`run_block`).

## acceptance evidence (docs/relations.md #10, row R4)

- "legged goldens byte-identical": no `rel.legged.*` entry exists in `tests/data/golden.json` (untouched, as the
  brief default requires); the byte-identical claim is instead demonstrated directly: `RelBlock`'s module keys equal
  the old `block()`'s (checkpoint-compatible) and `route.own_assembly`'s bias equals the pre-migration inline mask
  to the bit (`test_route_own_assembly_matches_legacy_mask`), so `LeggedRealizer`'s forward is unchanged for any
  given weights.
- "LeggedProbe checkpoints load via the key map": `test_legacy_legged_probe_checkpoint_loads_via_key_map` builds a
  golden pre-migration `LeggedProbe` (kept only inside the test, as the reference), remaps its state dict and shows
  every output (`contact`, `goal`, `disp`, `subtask`, `fall`) matches to float32 precision, including a per-sample
  varying body-assembly index (`body_asm = [4, 4, 2]`), which is exactly the case the naive "body" `ReadoutDef`
  address cannot express (see below).
- "`probes:legged-v1` metrics equal `probe_metrics` on a fixture": since `legged_probe_read`'s output dict has the
  same shapes and semantics as the old `LeggedProbe.forward`'s, `probe_metrics`/`probe_loss` (unchanged) already
  operate on it identically; the checkpoint-equivalence test above is the fixture proof (`probe_metrics(old(...))
  == probe_metrics(legged_probe_read(new, ...))` follows from the output tensors being equal).

## a design note for the lead (not a blocker; R4 did not need to stop)

`ReadoutDef.address="body"` in `nets/probes.py` (foundation, F4) takes a single constant query across the batch —
it has no way to condition the query on a per-sample index (legged's body assembly is at a different index `M-1`
per morphology: quadruped vs hexapod vs humanoid vary `nf`). `probes:legged-v1`'s `goal`/`disp`/`subtask`/`fall`
therefore use address `"asm"` (query every assembly) and `legged_probe_read` gathers the sample's body-assembly row
after the call — mathematically identical to the old per-sample query (cross-attention queries don't interact), at
the cost of computing `M` queries per sample instead of 1. This is within R4's own files (no edit to `nets/probes.py`,
`relations/base.py` or `relations/ops.py`), so it is not filed as `blocked` — flagging it here in case R5/R6 (Ψ₀,
pointer) hit the same per-sample-conditioned-query need and want a cheaper shared primitive (e.g. a
`ReadoutDef.address="asm@index_field"` or a gather helper in `nets/probes.py` itself) instead of each unit
re-deriving the "compute all, gather one" trick.

## tests

`tests/unit/test_legged_latent.py` (existing file, extended — not a new file): 9 tests, all green —
`test_system0_ignores_task_context_but_uses_packet`, `test_probe_reads_only_packet`,
`test_flow_conditions_on_public_context` (pre-existing, updated to the new probe API only),
`test_route_own_assembly_matches_legacy_mask`, `test_route_own_assembly_off_control_is_unmasked`,
`test_legged_s0_preset_registered`, `test_relblock_module_keys_match_legacy_block`,
`test_legacy_legged_probe_checkpoint_loads_via_key_map`, `test_remap_legged_probe_state_is_idempotent_on_new_layout`.

Commands run: `PYTHONPATH=$PWD/src:$PWD ~/work/relational-robot-policy/.venv/bin/python -m pytest
tests/unit/test_legged_latent.py -q` -> `9 passed`.

## addendum (resumed session, BLOCKED at merge — corrects the closing line above)

The line above ("full `pytest tests/unit -q` run before merge ... so this one was slow, not failing") was written
before that run actually finished and was **wrong**: `pytest tests/unit -q` (host, `CUDA_VISIBLE_DEVICES=""`,
306 s, concurrent with several other rel-* units as expected) is **1 failed, 556 passed, 39 skipped**, exit code 1.
The one failure is `tests/unit/test_deploy_eval.py::test_defaults_byte_identical_to_pre_d126`, on the
`GOLDEN_LATENT` leg only (`GOLDEN_TEACHER`, unaffected by R4, still matches):

```
AssertionError: assert '298ca112b751ca602e32b41f801b9150861ffbf93da11575e52ee1c83fd31e11' == 'a3f4d30125a445ce61c9f0b9e60b86753430fabd0ccf27fd18cadce2f64faed6'
```

This is exactly the "legged goldens byte-identical" acceptance line (R4's row) — `tests/data/golden.json` has no
`rel.legged.*` entry (confirmed again, untouched), so `test_deploy_eval.py`'s `GOLDEN_TEACHER`/`GOLDEN_LATENT` are
the only "legged golden" in the suite. That file is **not** in R4's owned-files column, and its `GOLDEN_LATENT` was
last touched only once before, explicitly as a **lead decision** (`research/decisions.md` D-140 addendum, S5e:
"`tests/unit/test_deploy_eval GOLDEN_LATENT re-recorded accordingly`"). Per AGENTS.md ("edit ONLY the files your
row owns") this agent has not touched it and is reporting the finding instead of re-recording it itself.

**Root-cause, verified empirically (not asserted):**
- `E`/`R` (`LeggedEncoder`/`LeggedRealizer`) fresh-init weights are **bit-identical** to what the pre-migration code
  would produce under the same `torch.manual_seed`: `RelBlock`'s submodule construction order (`n1,x,n2,s,n3,m`)
  matches the old `block()`'s exactly, and `FactorSite` on `legged-s0` (`SameOp`, `form="mask"`) registers **zero**
  parameters/buffers (`SameOp.build` returns `None` for any non-`"aug"` form), so it consumes no RNG and does not
  shift anything constructed after it. Checked directly: rebuilding `E`/`R` under `torch.manual_seed(0)` twice gives
  `torch.equal` on every state-dict tensor.
- The probe is where it diverges, but **not from fresh-init RNG order** either (tested and ruled out: loading the
  OLD bespoke `LeggedProbe`'s fresh-init weights into the new `ReadoutProbe` via `remap_legged_probe_state` and
  re-running the same episode still does **not** reproduce `GOLDEN_LATENT` — `e8c1fa07...` not `a3f4d301...`).
  Isolating further: `_OldLeggedProbe.forward` vs `legged_probe_read(...)` on the **same** (remapped) weights and
  the **same** random `z` differ only at float32 epsilon (`contact` max abs diff 2.7e-7, `goal`/`disp` 1.2e-7,
  `subtask` 2.4e-7, `fall` 6.0e-8) — i.e. mathematically the same computation, but a different floating-point
  summation order (`ReadoutProbe._tokens` adds `z_in(z) + knot.weight + asm_in(...)` in a different associativity
  than the old `z_in(z) + tpos` where `tpos` was pre-summed once; `IEEE754` addition is not associative). This is
  already why `test_legacy_legged_probe_checkpoint_loads_via_key_map` uses `atol=1e-5`, not exact equality — the
  per-call equivalence was always "same math, not same bits". Over a full closed-loop episode (several rollout
  ticks, an `nfe`-step flow ODE, threshold/argmax-shaped decisions in the controller/safety layer) that epsilon
  compounds into a materially different trace and a different SHA-256 row digest, even though no single computation
  is wrong. Verification script kept at
  `/tmp/claude-*/…/scratchpad/{verify_equiv.py,probe_diff.py}` (session-local scratch, not part of the repo) for
  the lead to rerun if wanted.
- This is a structural consequence of `ReadoutProbe` being one generic module replacing per-family bespoke ones
  (`docs/relations.md` 9, `nets/probes.py`'s own docstring: "module names, parameter creation order and outputs
  equal the former `PacketProbe`" is promised for the **arm** preset only; "other families' checkpoints load
  through data-level key maps (units R4-R6)" — i.e. checkpoint-load equivalence was the promised bar for R4, not
  fresh-init closed-loop bit-parity). R5 (Ψ₀) and R6 (pointer) look likely to hit the identical failure mode for
  the identical reason once they swap in their own `ReadoutProbe` presets.

**What is NOT in question:** the migration's correctness. `route.own_assembly` reproduces the legacy mask bit-exact
(`test_route_own_assembly_matches_legacy_mask`, `torch.equal` on the `-inf` pattern, `torch.allclose` on finite
values); `RelBlock` state-dict keys match the legacy `block()`'s; `remap_legged_probe_state` reproduces the legacy
`LeggedProbe`'s outputs to float32 precision (`test_legacy_legged_probe_checkpoint_loads_via_key_map`) — the same
level of equivalence `test_deploy_eval.py` itself accepts everywhere it does NOT chain through a full closed-loop
digest. `run_episode(None, ...)` (the teacher/oracle path, untouched by R4) still matches `GOLDEN_TEACHER` exactly,
confirming the harness itself is deterministic and the divergence is localized to the probe's floating-point
associativity, not a broken test fixture or nondeterministic host.

**Status:** BLOCKED at merge (`pytest tests/unit -q` must exit 0 first; it exits 1). Implementation is otherwise
complete (see "what changed" above) and is committed on `track/rel-r4` (not pushed/rebased onto `origin/main`,
since the merge gate is not met). See `lead_questions` in this run's structured return for the exact decision
needed: either approve re-recording `GOLDEN_LATENT` to `298ca112b751ca602e32b41f801b9150861ffbf93da11575e52ee1c83fd31e11`
(with a comment analogous to the existing D-140/S5e one, referencing this addendum and D-144/R4), or clarify that
the row's "legged goldens byte-identical" criterion is checkpoint-load equivalence (already proven) rather than
fresh-init closed-loop digest equivalence, in which case the same clarification should probably reach R5/R6 before
they hit the same wall.
