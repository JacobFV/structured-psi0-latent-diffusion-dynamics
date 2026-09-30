# rel-r13 track (D-144 fanout unit R13: geometry factors)

Owner: Sonnet fanout agent. Worktree `~/work/rrp-wt/rel-r13`, branch `track/rel-r13` from `origin/main` @
`f38b689a` (>= the required `c5e7d45` foundation commit; also includes R1-R12). Brief: `docs/relations.md` section
10, row R13 + its brief paragraph: "Declare the geometry entries (sqdiff+diff on `pos3d` and `cam_uvd`, rel_rot on
`orient`, align on normals, order along gravity), wire field readouts (`source=probe`, `readout_layer`) through the
foundation hook, write the guard and resolution tests, run the peer smoke." Deps: R12 (already on `main`).

Never touched `relations/base.py` / `relations/ops.py` (forbidden by the fanout rules) — the "foundation hook" for
field readouts, `rrp.policies.relations.ops.FieldReadouts` (+ `site_field` reading `rc.estimates`), already exists;
this unit's job is the catalog entries that wire INTO it, not the hook itself.

## what changed (owned files only)

- **`src/rrp/policies/relations/catalog.py`**, section `# ---- R13: geometry (geo.*)` (the foundation left this
  section pre-created, per-row, at the file's bottom): five `FactorDef`s + a convenience preset.
  - `geo.pos3d` — `sqdiff+diff` (PaPE) on `pos3d`, `form="aug"`, `sources=("given", "probe", "gt")`,
    `params={"p": 3, "frame": "world", "readout_layer": 0}`.
  - `geo.depth3d` — `sqdiff+diff` on `cam_uvd`, `sources=("probe", "given", "gt")` (default `probe`: the case this
    factor exists for is exactly the un-tracked / occluded token, where there is no `given` `cam_uvd` to read).
    This is the literal example `docs/relations.md` section 3.1 gives for a PaPE depth factor; used verbatim.
  - `geo.orient` — `rel_rot` on `orient` (assembly frames), `sources=("given", "probe", "gt")`.
  - `geo.normal_align` — `align` on `normal` (key-side normal / axis field), `sources=("given", "probe", "gt")`.
  - `geo.above` — `order` on `pos3d`, `form="bias"`, `params={"axis": (0,0,1), "margin": 0.01}` (world-up; `order`'s
    own code default is already `(0,0,1)`, declared explicitly here so the entry documents "along gravity" rather
    than relying on the operator's implicit default). `sources=("given", "gt")` only — a hard sign relation has no
    readout of its own (no `ReadoutDef`), so `probe` isn't offered.
  - Every `aug` entry (`pos3d`, `depth3d`, `orient`, `normal_align`) carries a `readout=ReadoutDef(..., reads="tokens")`
    and `params.readout_layer` — these are what `FieldReadouts` (built by `nets.flow.ContextEncoder.__init__` as
    `self.readouts`, called once per layer as `self.readouts.observe(li, "ctx", h, rc)`, already on `main` since the
    foundation) picks up for any resolved spec whose `effective_source() == "probe"`. `label` / `gen` name the
    LABELS / scene-part keys unit R14 (`harness/data/relgen/geometry.py`: labels `pos3d`, `cam_uvd`, `orient`,
    `contact_normal`; parts `table_objects`, `camera_depth`) will register — declared here for documentation
    continuity; nothing in R13 depends on R14's registries existing yet (unregistered `LABELS`/`TRANSFORMS` keys are
    inert strings until something looks them up, matching R10 and R12's precedent for forward-referenced names).
  - `register_preset("geo", [...])` — all five, for convenience (used throughout the unit's own tests; not required
    by the row's acceptance, not exercised by anything outside this file).
- **`tests/unit/test_relations_geo.py`** (new; not listed in the table row, but every unit writes red/green tests
  for its own acceptance criteria — same precedent as every prior R* unit's test file).

## acceptance evidence (row R13, docs/relations.md section 10)

Command: `cd ~/work/rrp-wt/rel-r13 && export PYTHONPATH=$PWD/src:$PWD && <main checkout>/.venv/bin/python -m pytest
tests/unit/test_relations_geo.py -v` -> **15 passed**. Full suite: `pytest tests/unit -q` -> **628 passed, 39
skipped** (pre-existing skips: Menagerie assets / optional extras / peer-only, same set R10/R12 recorded), **0
failed**, exit code 0.

| criterion | evidence |
|---|---|
| factors resolve on arm / dual | `test_resolve_on_arm` (`resolve(["preset:arm", "preset:geo"])`, all 5 names present, order preserved); `test_resolve_on_dual` (real `DualSession` + `MultiFeaturizer` fixture, `relation_token_sets` ctx set, `FactorSite` built against it, `.bias()` / `.augment()` run finite) |
| deploy guard red/green with `source=gt` | RED: `test_deploy_guard_red_blocks_gt_control`, `test_deploy_guard_red_blocks_gt_source` (`assert_deployable` raises `PrivilegedInput`); GREEN: `test_deploy_guard_green_default_and_off_gt_pass` (default specs, and an explicit `off` + `gt` combination, both pass); `test_deploy_guard_rejects_gt_where_sources_disallow_it` (registry-level validation: `geo.above` has no `probe` source, `resolve` rejects it before the guard is even reached) |
| field readout (`readout_layer`) wiring test | `test_field_readout_populates_estimate_only_at_its_layer`, `test_readout_layer_param_moves_the_hook` (the `params.readout_layer` override actually moves which forward layer populates `rc.estimates`), `test_site_field_probe_source_reads_the_readout_estimate_not_the_raw_field` (`site_field` returns exactly the tuple `FieldReadouts.observe` wrote, not the token set's raw field — there is deliberately no raw `cam_uvd` field on the test token set), `test_factor_site_augments_from_the_probe_estimate_end_to_end` (an end-to-end `FactorSite.augment` computed ENTIRELY from a probe estimate — no raw field ever exists on the token set — is finite and, once nudged off zero-init exactly as one optimizer step would, has a real gradient path back into `FieldReadouts`' own parameters) |
| `<=10 min peer smoke of arm flow with geo.depth3d source=probe (loss decreases, depth probe error logged)` | see "peer smoke" below |

Also covered (not literally required by the row, but load-bearing for the above): `test_geo_entries_registered_field_op_form` / `test_geo_entries_declare_readout_layer_and_probe_source_where_expected` pin field/op/form/`readout.reads`/`readout_layer` against the brief's exact list; `test_geo_aug_factors_are_zero_init` (docs 3.2's zero-bias-equivalence, the same property `test_relations.py` checks for the foundation's own ops, extended to these four); `test_geo_above_sign_matches_hand_computed_height_order` and `test_geo_factors_resolve_and_run_on_the_arm_fixture_ctx_set` run the entries on a real `make_pick_place_session` fixture through `relation_token_sets` (unit R12), matching R12's own test-file precedent for "on the arm fixture."

## field-readout wiring: what "the foundation hook" is, precisely

`rrp.policies.relations.ops.FieldReadouts` (foundation, `ops.py`, never touched by this unit): a `nn.Module` built
once per net from the resolved specs, filtering to `effective_source(s) == "probe"`; its `.observe(layer, set_name,
h, rc)` is already called once per context layer by `nets.flow.ContextEncoder.forward` (`self.readouts.observe(li,
"ctx", h, rc)`, foundation code, unmodified). It writes `rc.estimates[(set_name, field)] = (mu, var)` at the layer
named by each factor's own `params.readout_layer`. `ops.site_field` (also foundation) already reads from
`rc.estimates` instead of the token set's raw field whenever a factor's effective source is `"probe"`. R13's job was
registering entries that a) declare `readout_layer` / `reads="tokens"` correctly and b) actually resolve and run
through this existing machinery — proven directly (no `nets/flow.py` involved) by the four `test_field_readout_*` /
`test_factor_site_augments_*` tests above, and end-to-end (through the real `FlowPolicy`) by the peer smoke.

## peer smoke

`nets/flow.py`'s `CTX_CARRIES = ("edges:arm-rel-v1", "hidden")` (module-level tuple controlling which `FactorSite`s
a given site accepts) does not yet name any field (`pos3d`, `cam_uvd`, ...), and `nets/flow.py` is not in R13's
`owns` cell (docs/relations.md section 10's table), so this unit could not add `"cam_uvd"` to it in the repo. Without
that one name in the tuple, `geo.depth3d` would never reach the `ctx>ctx` attention site inside the real
`FlowPolicy`, even though the field-readout mechanism itself (proven above) works correctly. To still run the row's
literal "peer smoke of arm flow with `geo.depth3d` source=probe" against the REAL, unmodified `FlowPolicy`, the
smoke script extends `CTX_CARRIES` by one tuple entry **in its own process only** (`flow_mod.CTX_CARRIES =
flow_mod.CTX_CARRIES + ("cam_uvd",)` — a runtime attribute patch, not a source edit; nothing under `src/` changed).
See "lead question" below for the follow-up this implies.

Script (not a repo file — no new files beyond this unit's owned list — kept in this agent's own scratch dir, copied
straight to the peer's own copy of the worktree): `scratchpad/r13_geo_smoke.py` (full text reproduced at the bottom
of this file for the record). It:
1. builds one real `make_pick_place_session` fixture, featurizes + collates it (unit R7/R12 machinery);
2. computes the TRUE `cam_uvd` via `relation_token_sets(..., cameras=[(model, data, "front")])` (R12's real MuJoCo
   camera projection) — used only as this smoke's supervised label (standing in for unit R14's not-yet-written
   `cam_uvd` label; nothing under `src/` reads it — the model only ever sees the factor's own probe estimate);
3. builds `PolicyConfig(factors=["preset:arm", "geo.depth3d"], width=32, heads=2, ctx_layers=2, blocks=1,
   horizon=2)` and the real `FlowPolicy`;
4. trains `total = flow_loss (FlowPolicy.loss against a fixed random BC target) + depth-probe gaussian NLL (rc.estimates
   vs. the true cam_uvd)` with Adam, logging `total`, `flow`, `probe_nll`, `depth_rmse_m` (RMSE of the depth channel
   only, in meters) every ~1/12 of the run.

Reproduction:
```
cd ~/work/rrp-wt/rel-r13
export RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/rel-r13
scripts/peer_sync.sh push
scp <script> gb10-direct:$RRP_PEER_REPO/_r13_geo_smoke.py
scripts/peer_run.sh --cpu 2 --mem 2G --label r13_geo_smoke --max-seconds 480 -- PY _r13_geo_smoke.py 200
```
Result (peer `gb10-direct`, CPU only, lease `1790737664_85e905`, `rc=0`, `memory_peak=0.41G` — comfortably under the
reserved 2G, > 1.35x margin): **200 steps in 3.4s**.
```
step   0: total  3.3178  flow 0.0000  probe_nll  3.3178  depth_rmse_m 1.014480
step  48: total -2.7793  flow 1.6362  probe_nll -4.4155  depth_rmse_m 0.073092
step 112: total -5.1271  flow 2.7099  probe_nll -7.8370  depth_rmse_m 0.011775
step 176: total -7.3804  flow 1.1515  probe_nll -8.5319  depth_rmse_m 0.002406
step 199: total -6.9774  flow 1.2513  probe_nll -8.2286  depth_rmse_m 0.003840
RESULT total DOWN 3.3178 -> -6.9774; depth_rmse_m DOWN 1.0145 -> 0.0038   =>  PASS
```
`total` (loss) decreases (3.32 -> -6.98, monotone in trend, not step-to-step — Adam on one 6-token overfit batch);
`depth_rmse_m` (depth probe error) is logged every step-group and drops ~265x over the run (1.01 m -> 0.0038 m). This
is a single-fixture overfit smoke by design (a sanity check that the wiring trains at all, not a generalization
claim) — marked as such here; not presented as a trained deployable model. `flow` is the real `FlowPolicy.loss`
flow-matching term against a FIXED random target (this smoke has no teacher/BC dataset in scope); it does not need
to converge for the row's acceptance and visibly does not (noisy around ~1-2, expected for an unfittable random
target) — only `total` and `depth_rmse_m` are the claims made above, and both hold.

Broker: acquired via the peer's own lease broker (`scripts/peer_run.sh` -> `rrp.cli ops run`), the SAME shared
broker/ops-state as the main peer repo (`peer_sync.sh` keeps `artifacts/`, `.cache/`, and `ops/` state as symlinks
into `$P/repo`); nothing peer-side outside this unit's own `RRP_PEER_REPO` (`/dev/shm/rrp-brandonin/wt/rel-r13`) was
touched, and no other running peer job was interrupted (checked before the push, per `peer_sync.sh`'s own busy-check
plus a manual `pgrep`/`cwd` scan against that dir).

## lead question: `nets/flow.py`'s `CTX_CARRIES` needs one entry for geo.* to attach for real

Not blocking — every literal R13 acceptance bullet is met (resolution, deploy guard, field-readout wiring, and the
peer smoke, all above) — but flagging because it affects whoever next legitimately owns `nets/flow.py`: `CTX_CARRIES
= ("edges:arm-rel-v1", "hidden")` needs at least `"cam_uvd"` (and, for `geo.pos3d` / `geo.orient` / `geo.normal_align`
to attach too, `"pos3d"`, `"orient"`, `"normal"`) added for the `geo.*` `aug` factors to actually reach the `ctx>ctx`
attention site in the shipped `FlowPolicy` — today they resolve, deploy-guard correctly, and field-readout-wire
correctly (all proven above, none of which touches `CTX_CARRIES`), but `FactorSite._applies` silently drops them
from `ctx>ctx`'s `self.specs` until that tuple is extended, so enabling `"geo.depth3d"` in a real run config would
currently be a structural no-op inside `FlowPolicy` itself (the peer smoke works around this with a same-process
attribute patch, not a fix). `nets/flow.py` is not in R13's owned-files cell, so this unit did not touch it. No
`docs/relations.md` row obviously owns this either (R2 owns `nets/semantic_latent.py` etc., not `nets/flow.py`).
Suggest either amending R13's owned-files list to add this one tuple literal, or a short follow-up unit / lead
sign-off scoped to `nets/flow.py`'s `CTX_CARRIES` / `ACT_CARRIES`.

## resume steps (if interrupted before merge)

1. `cd ~/work/rrp-wt/rel-r13 && export PYTHONPATH=$PWD/src:$PWD`
2. `<main checkout>/.venv/bin/python -m pytest tests/unit -q` — must exit 0 before merging.
3. `git fetch origin && git rebase origin/main`; rerun the unit suite if the rebase brought new commits.
4. `git push origin HEAD:main` (retry fetch/rebase/test/push up to 5x on a race, per the fanout rules).

## `scratchpad/r13_geo_smoke.py` (full text, for the record -- not a repo file)

```python
"""R13 peer smoke (docs/relations.md section 10, row R13): "arm flow with geo.depth3d source=probe (loss decreases,
depth probe error logged)". NOT a repo file (no new files beyond R13's owned list) -- run standalone on the peer via
scripts/peer_run.sh from this unit's own peer dir. Trains the real `rrp.policies.nets.flow.FlowPolicy` (unmodified)
on one real arm pick_place fixture, factors=["preset:arm", "geo.depth3d"].

CTX_CARRIES in nets/flow.py (a file this unit does not own -- docs/relations.md section 10's R13 row does not list
it) does not yet name field carries, so `geo.depth3d` cannot reach the ctx>ctx attention site as written on main.
This smoke extends CTX_CARRIES IN THIS PROCESS ONLY (a plain attribute patch, nothing written to the repo) to
exercise the wiring end-to-end; research/tracks/rel-r13.md reports this as a follow-up for whichever unit next
touches nets/flow.py. Independent of that patch, the field-readout mechanism itself (FieldReadouts.observe +
site_field reading rc.estimates for source="probe") is the existing foundation hook R13's catalog entries wire into
-- see tests/unit/test_relations_geo.py for the unit-level proof that does not depend on this patch.
"""
import json
import math
import sys
import time

import torch

from rrp.envs.mujoco.fixtures import make_pick_place_session
from rrp.policies.features.featurizer import featurizer_for
from rrp.policies.nets.batch import collate_inputs, relation_token_sets
import rrp.policies.nets.flow as flow_mod
from rrp.policies.nets.flow import FlowPolicy, PolicyConfig
import rrp.policies.relations.catalog  # noqa: F401  (registers FACTORS / FIELDS)

N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
LOG_EVERY = max(1, N_STEPS // 12)

torch.manual_seed(0)
flow_mod.CTX_CARRIES = flow_mod.CTX_CARRIES + ("cam_uvd",)      # see module docstring above

s = make_pick_place_session(seed=3, n_distractors=1)
f = featurizer_for(s)
pi = f(s.observe())
batch = collate_inputs([pi])

sets = relation_token_sets([pi], batch, cameras=[(s.model, s.data, "front")])
true_uvd = sets["ctx"].field("cam_uvd")[0]
uvd_valid = sets["ctx"].field("cam_uvd.valid")[0]
n_valid = int(uvd_valid.sum())
print(f"[r13-smoke] fixture: {batch.node_mask.shape[1]} ctx tokens, {n_valid} with known cam_uvd", flush=True)
assert n_valid > 0, "fixture has no cam_uvd-valid tokens: cannot run the depth-probe smoke"

cfg = PolicyConfig(width=32, heads=2, ctx_layers=2, blocks=1, horizon=2, aux=False,
                   factors=["preset:arm", "geo.depth3d"])
policy = FlowPolicy(cfg)
print(f"[r13-smoke] policy params: {sum(p.numel() for p in policy.parameters())}", flush=True)

N = batch.node_feats.shape[1]
target = torch.randn(1, cfg.horizon, N, cfg.latent_dim, generator=torch.Generator().manual_seed(0))
valid = batch.node_mask[:, None, :].expand(1, cfg.horizon, N)

opt = torch.optim.Adam(policy.parameters(), lr=3e-3)


def probe_loss_and_rmse():
    _, _, rc = policy.context(batch)
    mu, var = rc.estimates[("ctx", "cam_uvd")]
    mu, var = mu[0], var[0]
    m = uvd_valid.float()
    nll = (0.5 * ((true_uvd - mu) ** 2 / var + var.log() + math.log(2 * math.pi)).sum(-1) * m).sum() / m.sum().clamp(min=1)
    with torch.no_grad():
        depth_rmse = (((mu[:, 2] - true_uvd[:, 2]) ** 2 * m).sum() / m.sum().clamp(min=1)).sqrt()
    return nll, float(depth_rmse)


t0 = time.time()
log = []
for step in range(N_STEPS):
    opt.zero_grad()
    flow_loss, flogs = policy.loss(batch, target, valid, generator=torch.Generator().manual_seed(step))
    p_loss, depth_rmse = probe_loss_and_rmse()
    total = flow_loss + p_loss
    total.backward()
    opt.step()
    if step % LOG_EVERY == 0 or step == N_STEPS - 1:
        row = dict(step=step, total=float(total.detach()), flow=flogs["flow"], probe_nll=float(p_loss.detach()),
                  depth_rmse_m=depth_rmse)
        log.append(row)
        print("[r13-smoke]", json.dumps(row), flush=True)

elapsed = time.time() - t0
print(f"[r13-smoke] elapsed {elapsed:.1f}s for {N_STEPS} steps", flush=True)
first, last = log[0], log[-1]
ok_total = last["total"] < first["total"]
ok_depth = last["depth_rmse_m"] < first["depth_rmse_m"]
print(f"[r13-smoke] RESULT total {'DOWN' if ok_total else 'UP/FLAT'} {first['total']:.4f} -> {last['total']:.4f}; "
     f"depth_rmse_m {'DOWN' if ok_depth else 'UP/FLAT'} {first['depth_rmse_m']:.4f} -> {last['depth_rmse_m']:.4f}",
     flush=True)
print("[r13-smoke] PASS" if (ok_total and ok_depth) else "[r13-smoke] FAIL", flush=True)
```
