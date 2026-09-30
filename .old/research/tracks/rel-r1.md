# track: rel-r1 (relation-factor fanout, unit R1: arm/dual probes -> ReadoutProbe)

Branch `track/rel-r1`, worktree `~/work/rrp-wt/rel-r1`, from `origin/main` @ `c5e7d45` (D-144 foundation). Spec:
`docs/relations.md` sections 2-5 and 10 (row R1, brief in "briefs"), `research/relations_catalog.md`. Host only
(git / editing / unit suite; CUDA hidden); no peer smoke needed (the row's acceptance is pure unit math on CPU
fixtures, no training).

## scope

Replaced every R1-owned construction of `nets.latent_probes.PacketProbe` with `nets.probes.ReadoutProbe` configured
by the registry preset `probes:arm-packet-v1` (+ `probe.arm.goal_effect` when the legacy `probe` config carried
`goal_effect=True`), keeping the state-dict layout so old checkpoints load strictly:

- `policies/bundles.py`: `load_representation` (only function this unit owns in this shared file) now builds
  `ReadoutProbe` from `_readout_probe_specs(...)` and loads its saved `P` state dict through
  `_remap_probe_state_dict(...)`. Added both helpers here (used by every other R1 construction site) since they are
  data/config translation for the probe this file already owns, not a new file.
- `harness/train/latent_train.py` (probe lines only): `train_representation`'s fresh `P`, its checkpoint-resume
  `P.load_state_dict(...)`, and `fit_probes_on_frozen`'s fresh measurement `P` all go through the same two helpers.
- `harness/eval/latent_causal.py`: `load_probe(...)` (defines edit directions; same procedure for sem/nosem) rebuilt
  on `ReadoutProbe`; its `_anchor` query list updated `desired_delta` -> `observed_effect` (R1 brief: the output
  alias is dropped).
- `cli/latent.py` (`cmd_counterfactuals`) and `cli/dual_latent.py` (`cmd_eval`): the ad hoc post-hoc-probe loading
  blocks rebuilt the same way.
- `harness/eval/{hooks,ladder,dual_latent_eval,edit_harness}.py`, `viz/record.py`: audited, not touched. None of
  them construct a probe network directly -- they call `bundles.load_representation` / `latent_causal.load_probe`
  (already fixed above) or import the generic `probe_metrics` (unchanged; see below). No PacketProbe construction
  remains in the owned surface (`grep -rn "PacketProbe(" src/` -> only `policies/psi0/nets.py`, an unrelated class
  in a different module, out of scope).

Kept unchanged, with reasons recorded here per the "shared files, disjoint functions" rule:

- **`probe_loss` / `probe_metrics` (`nets/latent_probes.py`) were NOT renamed to `readout_loss` / `readout_metrics`.**
  They are generic dict-in/dict-out math over a probe's *output* keys, unaffected by which network produced that
  output, and are still imported unmodified by code outside R1's owned-file list: `harness.eval.latent_eval`
  (`PacketProbeHook`), `harness.train.joint_adapt`, `harness.eval.ladder`, `harness.eval.dual_latent_eval`
  (`DualPacketProbeHook`), and the golden/equivalence tests. Row R1's own acceptance criterion is stated *against*
  `probe_metrics` ("`probes:arm-packet-v1` metrics equal `probe_metrics` on a fixture batch"), so keeping it as the
  reference implementation is the literal acceptance target, not a shortcut. `nets/probes.py` already has the
  registry-generic `readout_loss` / `readout_metrics` (spec-driven, F4) for callers that want them; R1 does not
  force existing arm call sites onto them since that would touch out-of-scope files for no behavior change.
- **`nets/latent_probes.py` was NOT deleted**, although the table's "owns" column lists it "(delete)". Deleting it
  would remove `PacketProbe`, but `tests/unit/test_golden.py` (not R1-owned; no golden-strategy file is) constructs
  `PacketProbe` directly as the arm probe fixture for the byte-identical golden hashes (D-140 golden strategy,
  docs/relations.md 8.3) -- deleting the module would break the goldens, which is this row's *first* acceptance
  criterion ("goldens unchanged"). `probe_loss`/`probe_metrics` also stay depended on by out-of-scope files (above).
  Deferred to whichever later unit retires `test_golden.py`'s direct `PacketProbe` fixture / the other callers
  (tracked as an open question below, not a blocker for this row: R1's brief paragraph does not mention deletion,
  only the table's "owns" column does).

## tests

`tests/unit/test_relations_r1_probes.py` (new; red confirmed before the migration landed -- `ReadoutProbe` import
and `_readout_probe_specs`/`_remap_probe_state_dict` did not exist, `ImportError`):

1. `_readout_probe_specs` translates `goal_effect` -> `probe.arm.goal_effect` spec, drops `n_operators`, passes the
   rest through unchanged.
2. `_remap_probe_state_dict` renames only `heads.desired_delta.*` -> `heads.observed_effect.*`; already-renamed
   dicts are a no-op.
3. Acceptance #2 (old state dicts load strictly): a simulated pre-rename `PacketProbe` state dict fails
   `ReadoutProbe.load_state_dict` (strict) without the remap (RED, `RuntimeError`), loads strictly with it (GREEN);
   the real post-rename layout also loads strictly.
4. Acceptance #3 (metrics equal `probe_metrics`): on a seeded fixture batch (with and without `goal_effect`),
   `PacketProbe` and `ReadoutProbe(specs=probes:arm-packet-v1[+goal_effect])` built with the same seed produce
   identical `probe_metrics` / `probe_loss` (`torch.allclose` on the loss, exact equality on every metric and log
   key); `"desired_delta"` is present in the old output dict and absent from the new one (output alias dropped) while
   `"desired_delta_err_m"` stays as a metric key name (R1 brief: "do not change what any hook reports").
5. `policies.bundles.load_representation`'s source text names `ReadoutProbe(` and not `PacketProbe(` (its
   construction call site, R1-owned).

Acceptance #1 (goldens unchanged) is proved by *not* touching `nets/latent_probes.py`, `relations/base.py` or
`relations/ops.py`: `tests/unit/test_golden.py`'s byte-identical hashes are untouched by this unit's diff.

Full unit suite: see run log in the merge commit / structured result (`pytest tests/unit -q`, host CPU,
`PYTHONPATH=$PWD/src:$PWD`).

## open question for the lead

The fanout table's R1 "owns" column says `nets/latent_probes.py (delete)`. As implemented, the file is kept (see
above) because deleting it breaks `test_golden.py` (out of R1's scope) and orphans `probe_loss`/`probe_metrics`
callers in `latent_eval.py` / `joint_adapt.py` (also out of scope). Flagged to the lead rather than deleting
`relations/base.py`/`ops.py` territory or reaching into another unit's owned files; not treated as a blocker since
the brief paragraph itself never mentions deletion, only the table's "owns" column does, and R1's stated acceptance
criteria are all met without it.
