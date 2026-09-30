# track rel-r3: arm system 0 on RelBlock + route.own_assembly

Row: docs/relations.md section 10 (`R3`, brief below the table). Deps: F (foundation, c5e7d45, merged to main).
Owns: `src/rrp/policies/system0.py`, `src/rrp/policies/latent.py`.

## what changed

`LatentRealizer` (`policies/system0.py`):
- `self.blocks`: `nn.ModuleList[nn.ModuleDict(n1,x,n2,s,n3,m)]` (plain `MHA`) -> `nn.ModuleList[RelBlock(D, heads)]`
  (`rrp.policies.nets.attention.RelBlock`, foundation F3). `RelBlock` keeps the same submodule names
  (`n1,x,n2,s,n3,m`) and the same `MHA` internals (`q,k,v,o` Linears) in the same construction order, so parameter
  paths and the RNG draw sequence for weight init are byte-identical to the pre-R3 net (verified: golden hashes
  unchanged, `state_dict()` key set unchanged, `strict=True` cross-load between two freshly constructed instances).
- The inline `-inf` own-assembly mask (`node_asm[:,:,None] == knot_asm[None,None,:] & zmask...`) -> the
  `route.own_assembly` factor (`op=same, form=mask`, already registered by the foundation in `catalog.py`) through a
  `FactorSite(heads, D, "node>knot", specs, ("assembly_id",))` (`self.route`), resolved from a new preset `s0-arm`
  (`catalog.py`, one line: `register_preset("s0-arm", ["route.own_assembly"])`, added in its own section — the only
  catalog.py edit; `route.own_assembly` itself was already in the foundation's "legacy structure" section).
  `self.route` owns zero parameters and makes zero RNG draws (`SameOp.build` returns `None` for non-`aug` forms), so
  its placement in `__init__` cannot perturb the golden hashes either.
- New method `LatentRealizer.route_bias(zmask, node_mask, K, node_asm)` factors out the RelCtx construction (node /
  knot `TokenSet`s with an `assembly_id` field; invalid knot slots get sentinel id `-1` so `SameOp`'s `ik >= 0` check
  reproduces the old `& zmask` term exactly) so `forward()` and `AnchorLatentRealizer.forward()` share it.
- `policies/latent.py`: no change needed. It only constructs `LatentSystem0`/`DualLatentSystem0` around a realizer
  instance and never touches the private block/mask internals, so the R3 refactor is invisible to it. Left untouched
  (own it, but there was nothing to do inside the acceptance criteria).

## cross-cutting fix (outside the row's `owns`, done anyway — see lead_questions)

`policies/system0_anchor.py` (`AnchorLatentRealizer`, W12 ladder track) is NOT in R3's owns list, but it subclasses
`LatentRealizer` and had **copy-pasted** the pre-R3 private block/mask internals directly (`L["x"](L["n1"](x), ...)`
dict-style block access, `base.blocks[0]["x"].h` introspection in `from_base`) instead of calling
`super().forward()`. Swapping `blocks` to `RelBlock` broke it mechanically (`RelBlock` is not subscriptable) and
`tests/unit/test_contact_frames.py::test_anchor_realizer_zero_init_equals_base` failed red. Fixed minimally and
mechanically, no architecture change: `L["x"](...)` -> `L(x, tok, bias_x=bias, q_mask=node_mask)` (the `RelBlock`
call), `base.blocks[0]["x"].h` -> `base.blocks[0].x.h`, and its own duplicated own-assembly mask math replaced by a
call to the new shared `LatentRealizer.route_bias(...)` (removes the duplication rather than keeping two copies of
the same logic). Behavior unchanged (same test, now green; it already asserts zero-init equivalence to the base
realizer and the per-node anchor routing).

## tests (red/green, in existing files — no new test files; none listed under R3's `owns`)

`tests/unit/test_latent_contract.py` (existing file, two new tests):
- `test_realizer_param_keys_match_pre_relblock_layout` — asserts the exact pre-R3 `blocks.0.{n1,x,n2,n3,m}...` key
  set is `<=` the new `state_dict()` keys, `route.*` contributes no keys, and a fresh instance loads another's
  `state_dict()` with `strict=True` (the row's "realizer state dicts load strictly" criterion).
- `test_route_own_assembly_masks_other_assemblies_knots` — causal test of the factor itself: a single node owning
  assembly 0 is byte-unchanged (`torch.equal`) when assembly 1's `z` is perturbed, and changes when its own
  assembly's `z` is perturbed. This is the acceptance evidence for "system 0 reads only its own assembly's knots"
  going through the factor path instead of the deleted inline mask.

`tests/unit/test_golden.py` (existing; unmodified) already covers the row's "goldens unchanged" criterion:
`test_arm_latent_system_i_and_system0` and `test_latent_adapter_matches_old_path` both build a seeded
`LatentRealizer` and hash its outputs against `tests/data/golden.json` (untouched, per the brief).

`tests/unit/test_latent_boundary.py` (existing; unmodified) already round-trips a realizer `state_dict()` through a
separate process (`test_independent_consumer_in_separate_process`) — additional coverage of strict state-dict
loading.

## commands run (host only: git / editing / unit suite; no training/simulation)

```
export PYTHONPATH=$PWD/src:$PWD
~/work/relational-robot-policy/.venv/bin/python -m pytest tests/unit -q
```
Result: full `tests/unit` green (see the merge section of the parent task for the exact count / exit code recorded
at merge time). Focused re-runs during development: `test_golden.py`, `test_latent_contract.py`,
`test_latent_boundary.py`, `test_contact_frames.py`, `test_chunk_blend.py`, `test_latent_grpo.py` — all green.
No peer smoke was needed: R3's brief has no runtime/behavioral acceptance beyond the unit suite (no new training,
no new op/field), so nothing was dispatched to `scripts/peer_run.sh`.

## state: verified (implementation) -> merged (see commit reported to the lead)

No blockers; no operator/interface change needed (`relations/base.py` and `relations/ops.py` untouched, as required).
One judgment call, recorded above and in `lead_questions`: fixing `system0_anchor.py`'s mechanical breakage rather
than leaving `pytest tests/unit` red, since the merge gate requires exit code 0 and the fix does not change behavior
or touch the forbidden files.
