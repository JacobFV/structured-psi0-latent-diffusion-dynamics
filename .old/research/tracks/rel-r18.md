# track: rel-r18 (relation-factor fanout, unit R18: task / temporal / epistemic)

Branch `track/rel-r18`, worktree `~/work/rrp-wt/rel-r18`, from `origin/main` @ `b2d15d7c` (R7: state_view() on the
MuJoCo sessions -- the last commit on `main` before this unit started; R9 `a66959c5` and R12 `f38b689a`, this row's
two deps, were already in). Spec: `docs/relations.md` sections 2-5 and 10 (row R18, brief), `research/relations_catalog.md`
F "task, procedure and causality" and J "epistemic". Host only (git / editing / unit suite; CUDA hidden via
`tests/conftest.py`'s `CUDA_VISIBLE_DEVICES=""`); no peer smoke needed -- this row's acceptance (candidates on a
fixture, reveal/surprise math, the gate's causal effect) is pure unit-level and fixture-level testing, no training.

## scope

Implemented, in the three files this row owns:

- **`src/rrp/policies/relations/catalog.py` §R18** -- `task.next_contact` (`field="hidden"`, `op="bilinear"`,
  `form="aug"`, `sources=("probe", "gt")`, `label="next_contact"`, `gen=("reveal","surprise")`, `gates=("task",)`,
  `readout=ReadoutDef(query="next_contact", address="pair", out=1, loss="soft_ce", reads="hidden")`) and
  `time.same_track` (new `FieldDef("track_id", 1, "id", "public")`, `op="same"`, `form="aug"`, `sources=("given",)`).
  `sources` / `reads` match R16's `ix.*` (landed on `main` mid-unit, rebased in as the first real sibling
  `bilinear` entry): `control="gt"` is rejected on EVERY `bilinear` factor regardless of `sources`
  (`BilinearOp.controls()`, `relations/ops.py`, NOT edited by this unit, allows only on/off/zero/rewired -- a
  bilinear op reads hiddens directly, never resolves a field through `effective_source`); `"gt"` in `sources` is
  instead reached only via an explicit `FactorSpec(source="gt")` override, which the deploy guard
  (`assert_deployable`) still honors as privileged (tested). My first pass used `sources=("probe",)` /
  `reads="tokens"` before R16 landed, with no sibling `bilinear` entry yet to confirm the convention against;
  corrected once the rebase brought R16 in.
- **`src/rrp/harness/data/relgen/task.py`** (new file) -- the PRIVILEGED label `next_contact` (needs cap
  "contacts" only -- NOT "task_runtime": `MujocoStateView.CAPS` (R7, `envs/mujoco/session.py`) never declares that
  cap and `StateView` has no dedicated task-runtime accessor either, so this unit does not add one (outside its
  owned files; `envs/base.py` is F/R7's). "task runtime" in the docs table's `next_contact (task runtime + contacts)`
  is the candidate-EDGE side's job (public `PolicyInput`/featurizer data, `nets.batch.candidate_interaction_edges`),
  never this label's; the label itself only needs the privileged contact stream to know WHICH candidate is the
  ground truth). `next_contact_sample(candidates, view, ...)` builds a `relgen.Sample`-shaped dict from that ground
  truth, ready for R9's `TRANSFORMS["reveal"]` / `TRANSFORMS["surprise"]` UNMODIFIED (no operator/interface change
  needed for this row either). A design correction made while writing the reveal/surprise test: "nothing touching
  yet" must be an EMPTY evidence list (no information), never an evidence event that excludes every candidate (a
  contradiction `reveal`'s posterior falls back to "no information" for anyway, but for the wrong reason -- it
  would silently mask a genuine "not yet observed" state as "observed and rejected everything").
- **`src/rrp/policies/nets/batch.py`** (candidate edges only, appended after R12's functions; nothing else in this
  file touched) -- `candidate_interaction_edges(inputs, batch) -> EdgeSet`: `CAND_REL_VOCAB = ("graspable",
  "support", "destination")`, a soft PUBLIC `EdgeSet` over the SAME `ctx>ctx` site R12's `relation_token_sets`
  serves. `graspable`: manipulator touch/grip SENSOR tokens (`interact` bank, local kind 1 -- the featurizer's
  per-manipulator `declared_sensor_channels`) -> every valid scene entity token, uniform prior 1/N. `support` /
  `destination`: every valid scene entity token -> every OTHER valid scene entity token (excludes self), uniform
  prior. No affordance labels exist in these fixtures, so "candidate" is deliberately unfiltered by the `known` bit
  (an occluded object stays a valid interaction candidate -- resolving which one is this unit's whole epistemic
  point, docs 5.4). `task.next_contact` never reads this `EdgeSet` directly (a `bilinear` op has no `edges:*`
  field); it is the candidate pool `next_contact_sample` (relgen/task.py) turns into `reveal`/`surprise` targets.

Did NOT touch `relations/base.py`, `relations/ops.py`, `relgen/__init__.py`, or `flow.py` (wiring `task.next_contact`
/ `time.same_track` into a net's actual `FactorSite` calls -- passing `carries=("hidden",)` / `("track_id",)` at a
real attention site -- is that net's own unit, none of which exists on `main` yet; this unit only ADDS the
declarative registry entries and their data-gen halves, exactly as every sibling `bilinear` row (`ix.*`) does).

## tests

`tests/unit/test_relations_r18.py` (18 new; red on pre-R18 `main` -- every import at the top of the file does not
exist there):
- registry: both factors resolve; `compat_hash` sees the added factor; disallowed gate / disallowed `gt` control
  raise `FactorError` (bilinear's control set, `time.same_track`'s `given`-only sources).
- `task.next_contact`'s `ReadoutDef` fields (soft_ce / pair / tokens) and `gates == ("task",)`.
- **gate `task` changes the bias between two task texts** (this row's acceptance): with the gate ACTIVATED
  (`{"gate": "task"}` -- the registry only lists it as ALLOWED, not on by default) and every learned parameter
  randomized, two different `[B, dim]` "task text" summary vectors produce different query-side augmentations and
  different attention logits; the key side is untouched (the gate scales `phi_q` only, docs 3.4); same task text
  twice is byte-identical (determinism). A companion test pins the OTHER end: at the registry's zero-init defaults
  the augmentation is exactly zero regardless of the task text (`BilinearOp`'s `g` is zero-init; enabling the
  factor is a no-op at step 0, docs 3.2).
- `time.same_track`: exact equality of the `aug` q/k dot product against a hand-built track-id table, with
  `n_ids <= code_dim` overridden so the fixed random codes are exactly orthonormal (the registry default n_ids=64 >
  code_dim=16 is only approximate -- matches `SameOp`'s own documented condition, not a bug).
- **candidates on pick_place / dual fixtures** (acceptance): shape `[1, C, C, 3]`, `prov="public"`, every value in
  [0, 1], on BOTH the arm (`make_pick_place_session` + `featurizer_for`) and dual (`DualSession` +
  `MultiFeaturizer`) fixtures (mirrors R12's own dual test pattern, `test_relations_r12.py`). Graspable rows are
  exact uniform PRIOR DISTRIBUTIONS (sum to 1) over every scene entity, from every sensor token found. Support /
  destination rows exclude the row's own entity and hit exactly the other scene entities (built with
  `n_distractors=2` so the exclusion is non-trivial). A synthetic-`Batch` edge case (no sensor tokens, degenerate
  scene) never crashes and stays all-zero.
- `next_contact` label: a lightweight `StateView` stand-in (`_FakeView`, no simulator) isolates the label's contact
  logic (manipulator-manipulator and non-manipulator-non-manipulator contacts never count; only a
  manipulator<->non-manipulator pair does) from MuJoCo geometry; a structural test then runs the SAME label
  function against the REAL `state_view()` of both fixtures (`label_runs_in` gate, shape, validity).
- **reveal / surprise applied to its targets** (acceptance): `next_contact_sample` + R9's unmodified
  `TRANSFORMS["reveal"]` collapses onto the touched candidate when something is touching, and stays the uninformed
  uniform prior when nothing is (the bug this caught, see above); `TRANSFORMS["surprise"]` on a hand-built
  multi-step `evidence` list collapses at the expected step then -- with `rate=1.0` -- deterministically reopens to
  every OTHER candidate exactly `after` steps later, matching R9's `switch_at` bookkeeping exactly.

Full unit suite: `pytest tests/unit -q` -- see the merge section of the parent report for the exact pass count and
exit code recorded at merge time (re-run after every rebase per the fanout rules).

## for the lead

Nothing blocked; no operator/interface change needed. Two things worth a look when a later unit wires
`task.next_contact` / `time.same_track` into an actual net (this unit deliberately stopped short of that, matching
`ix.*`'s pattern of registry-first, net-wiring-later):
1. `task.next_contact`'s `ReadoutDef(address="pair", reads="tokens")` is metadata only today -- neither
   `nets.probes.ReadoutProbe` (packet addresses only) nor `relations.ops.FieldReadouts` (explicitly skips
   `op="bilinear"`) consumes a `pair`/`tokens` readout yet; docs section 4's "the bias IS the probe" note says the
   attention-time bilinear score itself becomes the pair-probe output, which needs a small reader wherever a net
   actually trains this factor.
2. `candidate_interaction_edges` is exposed as a pure function (`nets.batch`, not wired into `relation_token_sets`
   or `RelCtx.edges`) -- deliberately, since `task.next_contact` never reads an `EdgeSet` at attention time and
   wiring an UNCONSUMED site key into `flow.py` seemed like scope creep past "candidate edges only". Whoever trains
   `task.next_contact` end to end will want to call it directly (as this unit's tests do) rather than expect it on
   `RelCtx`.
