# track pointer: ComputerWorld environment and the pointer policy (D-140, D-142; was research/tracks/cworld.md)

Owner decision D-140; design `docs/architecture.md` section 5. Status: **verified, merged after S3**. Code:
`src/rrp/envs/computerworld.py` (env, mapping, cw/* setups + judges), `src/rrp/bodies/fixtures.py:cw_pointer_spec`,
`src/rrp/tasks/spec.py` (cw/* TaskSpecs), `src/rrp/policies/teachers/computerworld.py` (teacher:cw/*),
`tests/unit/test_computerworld.py`. The pre-S3 prototype (proto/cworld, commit c305296) is deleted.

## install (exact; no Rust build needed)
PyPI publishes an aarch64 abi3 wheel for 0.2.0, so no Rust build is needed on either machine.
```sh
mkdir -p ~/work/ext && git clone https://github.com/JacobFV/computerworld ~/work/ext/computerworld   # read-only reference (HEAD 35753083)
python3 -m pip download computerworld==0.2.0 --no-deps -d ~/work/ext/wheels
#   computerworld-0.2.0-cp39-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
#   sha256 b6360952f6ec2c54921ccc6d71d99577842dc30c01b482de33849e976b748b8a (40 MB; never committed)
uv pip install --python ~/work/relational-robot-policy/.venv/bin/python --target ~/work/ext/cw-site --no-deps \
    ~/work/ext/wheels/computerworld-0.2.0-*.whl
PYTHONPATH=~/work/ext/cw-site python -c "import computerworld as c; print(c.__version__, c.engine_version)"   # 0.2.0 0.2.0
```
The shared host `.venv` is not modified: the wheel sits in its own `--target` dir and is added to `PYTHONPATH`. At S3 it
becomes the optional extra `computerworld = ["computerworld==0.2.0"]`. Pin the exact version. CW snapshots and pixels
are only valid within one engine version, and a git-HEAD build (0.2.0+246 commits) is a different engine from the wheel.
The world definitions under `worlds/` at HEAD are also not guaranteed to load in 0.2.0. Our env builds its own
one-machine world inline, so it does not use them.

## run the tests
```sh
cd <worktree>; PYTHONPATH=$PWD/src:$PWD:$HOME/work/ext/cw-site ~/work/relational-robot-policy/.venv/bin/python -m pytest -q tests/unit/test_computerworld.py
```
Tests marked `computerworld` (rollout of every teacher, idle timeouts, failure reasons, determinism, truth/observation
split) skip when the wheel is absent; mapping, depth, occlusion, slots, button edges, body and negotiation always run.

## what is there
- `ScreenFrame`: at 1 mm/px, (u, v) maps to ((u-W/2)s, (H/2-v)s) with z toward the viewer. Default viewport 960x640,
  the CW console default. `depth="stack"` (default) sets z = dense rank of the scene's z-layers × dz (2 mm), and
  `depth="constant"` sets z = 0.
- `scene_widgets` / `SlotRegistry` / `descriptors`: every semantic node becomes an `ObjectDescriptor` (descriptor =
  role; attributes are label/value/state/disabled/window/interaction once S3 adds the field). Occluded nodes are kept
  with `visible=False`: a node counts as occluded when a higher opaque box/rounded_box/backdrop/image fully covers its
  clipped box. Slots stay stable within an episode. A vanished widget leaves a `descriptor="null"` slot, and new
  widgets are appended. Task bindings set `bound_entity` (e.g. `textbox:Name` → `field_name`).
- `cw_pointer` RobotSpec: slide x, y (viewport range) and slide z fixed at hover (0.05 m). qpos = pointer (x, y, z).
- Action groups at 10 Hz:
  - `pointer`: cartesian_position, 2-wide, m, absolute.
  - `button`: ≥ 0.5 is down; only edges emit `down`/`up`, so a click takes two ticks.
  - `wheel`: notches × 120 px.
  - `key`: discrete over `KEY_VOCAB` (14 named keys + 95 printable characters; -1 = none).
  - Within a tick the order is move, button, wheel, key.
  - Rejections are returned as codes: `bad_group:*`, `key_out_of_vocab`, `cw_refused:<family>.<op>:<code>`.
- Tasks (judge → outcome + failure_reason):

  | task | setup | success | failure / timeout reasons |
  |---|---|---|---|
  | `cw/calc_sum` | calculator open | history shows `a + b = a+b` | `wrong_result`, `no_result` |
  | `cw/open_type` | empty desktop | editor text == word | `wrong_text`, `app_not_open`, `incomplete_text` |
  | `cw/drag_window` | calculator open | title bar moved by (dx, dy) ± 6 px, button up | `window_closed`, `off_target` |
  | `cw/fill_form` | browser on the simulated `form.internal` static site | submitted query == name/email | `wrong_value:<field>`, `not_submitted` |

  Drag goals are sampled from the feasible range: CW clamps windows below the 32 px top bar and right of the 64 px dock.
- `ScriptedTeacher(task)`: source `scripted_teacher`, privileged (it reads the full scene through the env). It moves the
  pointer at 60 px/tick toward the widget's centre, then clicks, drags or types one symbol per tick.
- `truth()` gives every widget's pose, including occluded ones, plus the goal and the CW `state_hash`. `reset(seed)`
  restores a cached per-seed initial snapshot, so it is deterministic. `snapshot`/`restore` wrap `world.snapshot()`
  and also carry pointer/slot state. Provenance records the engine version, package version, adapter version, world
  digest, theme and viewport.

## findings about CW 0.2.0 that correct the design doc
1. **Scene node ids are not stable across revisions.** The Files launcher gets a new id after a window opens, and CW's
   own docs say not to retain node ids across layout changes. Slot identity is therefore keyed on
   `interaction|role|label` plus an occurrence index.
2. There is no explicit `state` in the node semantics (only role/label/value/disabled/focusable). The "focused" state
   comes from `scene.focus.interaction`. The Text Editor's value appears only in the `semantic.v1` observation
   (`editor-text`), not in the scene node, so the judge reads the observation.
3. Stack depth ranks z-layers rather than nodes, because a desktop has ~100 nodes but ~5–10 layers. Within a layer,
   occlusion follows insertion order, as CW's hit test does.
4. Every themed-desktop GUI action needs `application.v1`, and browser content needs `browser.v1`. The mouse pointer
   is the last scene node (z 2e6, no semantics) and is never an occluder.

## smoke evidence (host, ~2 s total, vibe-check scale)
Scripted teacher, seeds 0–19: calc_sum 20/20, open_type 20/20, fill_form 20/20. drag_window: 50/50 on seeds 0–49
after the feasibility fix, 16/20 before it (the 4 failures were targets behind the dock). The idle policy times out on
every task with the reasons above. Speed: `scene()` ~1 ms, `step` ~0.7 ms, snapshot+restore ~0.01 ms, a 640x480 frame
~85 ms.

## open items
- `rrp matrix` row for computerworld (S5, same track): the arm/legged declines are already asserted in
  tests/unit/test_computerworld.py; the accepted pointer policies are the track `pointer` below.
- Demo video on the peer (D-115: no rendering on the host): one teacher success and one failure per task.
- cw/* tasks have no task graph yet (the env judges success); `task_graph` capability is not declared.

## pointer policy (track `pointer`, branch track/pointer; owner request 2026-09-29 "engineer or train a pointer policy with/as a system 0")
Code: `src/rrp/policies/pointer/` (spec, packet, features, probe, checkpoint, runtime), `src/rrp/policies/nets/pointer.py`,
`src/rrp/harness/train/pointer/` (split, collect, data, losses, train, diagnostics, video), `tests/unit/test_pointer.py`. Peer: code dir `/dev/shm/rrp-brandonin/wt/pointer`,
CW wheel extracted to `/home/brandonin/work/ext/cw-site` on the peer (the peer venv has no pip: `python -m zipfile -e
<wheel> ~/work/ext/cw-site`, same sha256 as above); `RRP_PEER_PYTHONPATH=/home/brandonin/work/ext/cw-site
ops/bin/peer_run.sh ...` appends it to the job's PYTHONPATH.

### step 1: engineered system 0 + teacher oracle packets (verified 2026-09-29)
- Packet contract for the pointer: M = 1 assembly (`tool`), the latent contract's knots (0.1, 0.3, 0.5, 0.7) s,
  validity 0.8 s. At 10 Hz knot k covers the ticks ending in (t_{k-1}, t_k]: 1 tick for k = 0, 2 (early, late) after,
  so one packet spans 7 ticks.
- Engineered encoding `cw_pointer_eng.v1`, z[4, 1, 14]: per knot two tick slots `[x, y, depth, button, key, wheel,
  flag]` (target pointer xy in m; stack depth of the widget under the target; button ±1; (key index + 1)/|vocab|, 0 =
  none; wheel notches; 1 = planned). `EngineeredSystem0` (label `scripted:cw_pointer_eng.v1`; NOT learned) moves toward
  the slot target from the measured pointer by ≤ 60 px/tick (the teachers' speed), sets the button, emits key/wheel;
  unplanned slot / expired / no packet = hold.
- `TeacherOracleSource` (ORACLE, privileged): runs `teacher:cw/*` on a shadow twin env ahead of the real episode and
  encodes its next 7 commands; counts shadow/real state-hash divergences. `make_pointer_oracle` = registry
  `pointer_oracle` (source `oracle`) = `LatentStackPolicy(oracle source, make_s0=EngineeredSystem0)`, replan 4 ticks.
  `LatentStackPolicy` gained `make_s0` / `requires` and an env clock (`env.time`, graph version 0 off MuJoCo).
- Result (`rrp eval`, peer, seeds 0–99 per task, `artifacts/runs/pointer_s1/`): oracle packets -> engineered system 0
  100/100 on each of cw/calc_sum, cw/open_type, cw/drag_window, cw/fill_form, with exactly the teacher's control-step
  totals (2860, 1729, 766, 3508), i.e. the packet round trip is lossless; teacher 100/100 each on the same seeds.
  Command:
  `ops/bin/peer_run.sh --cpu 2 --mem 4G --label pointer_s1_eval -- PY -m rrp.cli eval --policy pointer_oracle --env computerworld --task cw/<t> --body cw_pointer --seeds 0:100 --out artifacts/runs/pointer_s1/oracle_eng_<t>.jsonl`

### step 2: learned route (completed; D-142)
- Split `research/splits/cworld_pointer_v1.json` (commit 34fbc94, before any demo): held-out calc pairs (9), words (4),
  names (4); dev 50 seeds/task (in-distribution), sealed_id 100/task, sealed_heldout 50/task (calc, type, form).
  Adapter `cw_env.v2` widens the word/name pools to 24 each so typing must copy characters from the instruction.
- Public inputs of every learned pointer policy (`rrp.policies.pointer.public_features`): descriptor slots (label chars,
  role, bound task entity, box, depth, visible/focused/disabled/focusable), instruction characters, pointer x/y + button,
  tick, and an efference copy of the policy's own executed events (button edges with position, typed keys) — CW does not
  expose typed text in the scene, so typing progress is otherwise unobservable. The same for system i and BC.
- Demos (`rrp train pointer collect`, peer, `artifacts/datasets/pointer_v1/`, not committed): 6000 teacher episodes per
  task (545k ticks; calc 176,821, type 103,973, drag 47,542, form 216,391), 0 teacher failures, held-out variants and
  eval seeds skipped (split guard `check_no_leak` at load). DART: N(0, 25 px) on intermediate move ticks of 50% of
  episodes (never on the arriving tick: the teacher's goto would not terminate).
- Models (`artifacts/runs/pointer_v1/`): E (encoder: context + 7-tick demo chunk → z[4, 1, 16]), R (learned system 0:
  z + phase + measured pointer/button → pointer step, button, key), P (packet probe: target widget slot over the
  descriptors, target position relative to the pointer, tick phase idle/move/press/release/drag/type), flow system i
  (context → z), BC (context → 7-tick chunk). semfix = probe loss on z during E/R/P training (Gaussian NLL log-variance
  ≥ −4) + packet semantic loss through the frozen P in the flow (w 0.5, τ ≥ 0.6); nosem = both weights 0. `eng` = flow
  trained on the engineered encoding, realized by the SCRIPTED engineered system 0. Capacity: flow 2.1 M, BC 2.0 M,
  E + R 1.7 M parameters. rep 20k / flow 30k / BC 50k steps, batch 512, task-balanced sampling, bf16 on the peer GPU.
- Representation (val episodes of the training seeds, 20k steps): reconstruction 0.15 px (semfix) / 0.16 px (nosem),
  button and key 100%; semfix's joint probe on E means: slot 100%, phase 100%, relative target 1.5 px.
- R1 rung (ORACLE: teacher chunk → frozen E → LEARNED system 0; `pointer_oracle={"representation": ...}`), dev seeds:
  semfix 200/200, nosem 200/200 (50 per task). The learned system 0 realizes encoded packets without loss.
- Dev (50 seeds/task): latent semfix 179/200, nosem 184/200, eng 95/200, BC 200/200. Probes and probe-guided edits:
  D-142. Pre-registration 7d4b34ad (pushed to origin/track/pointer before any sealed run).
- SEALED (once; D-142): sealed_id teacher 400/400, oracle→engineered 400, BC 398, semfix 372, nosem 368, eng 185;
  sealed_heldout: every learned method 0/50 on unseen words and 0/50 on unseen names, 45–50/50 on unseen calc pairs.
- Rebase onto the D-144 relation-factor main: frozen checkpoints give identical per-seed outcomes and step counts on the
  old and new code (fill_form dev seeds 500000–5, semfix), so the merged code reproduces the evaluated policies.
  Outcomes depend on batch composition (one flow-noise generator per policy): sealed/dev rows used batch 16.
- Resume / reuse (P4c, D-145): the whole lineage is a recipe, `recipes/templates/pointer_lineage.yaml` (instances
  `recipes/pointer/pointer_v1.yaml`, gate `pointer_smoke.yaml`; family `pointer`, stages in
  `src/rrp/harness/pipelines/pointer.py`). Peer only, from the peer code dir, with the wheel dir on the peer path:
  `RRP_PEER_PYTHONPATH=$HOME/work/ext/cw-site RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/pointer rrp run-dag recipes/pointer/pointer_v1.yaml`
  (`--dry-run` first; `--point variant=nosem` selects one factor set; the smoke recipe writes its own lineage). Dry-run nodes
  of pointer_v1: global `collect`, `bc`, `eval_bc`, `flow_eng`, `eval_eng`, `eval_oracle`; per variant (semfix, nosem) at s1
  `rep`, `flow`, `eval_dev`, `eval_r1`, `probes`, `edits`. Checkpoints and data of the completed run stay in the peer store
  `artifacts/runs/pointer_v1/`, `artifacts/datasets/pointer_v1/`; evaluating them needs no recipe:
  `rrp eval --policy 'pointer_latent={"flow": "artifacts/runs/pointer_v1/flow_semfix.pt"}' --env computerworld --task
  cw/<t> --body cw_pointer --seeds ...`; `pointer_bc={"checkpoint": ".../bc.pt"}`; `pointer_oracle` (optionally
  `{"representation": ".../rep_semfix.pt"}` for the learned system 0). The sealed splits (sealed_id, sealed_heldout) were
  consumed once (D-142); a recipe evaluates `dev` (`vars.seed_set`). Not in the recipe: videos (`rrp train pointer
  video`) and the split file (`rrp train pointer split`, done).
- `rrp matrix` (artifacts/runs/pointer_v1/matrix.jsonl): pointer_latent (semfix, nosem), pointer_bc, pointer_oracle accepted
  on computerworld × all four cw/* tasks; teacher:pick_place declined (needs gripper, joint_position).
- Open: typing generalization needs a copy mechanism; the engineered key field needs a discrete code for a learned
  system i; one training seed per model.
- Readiness K2 (D-146): `Demos.batch` now supplies `wpos3d` / `wcamuvd` (stored per widget table by `collect`; packs from
  before D-144's geometry get zeros with `wgeo_ok` False, which `_relctx` forwards as `pos3d.valid` / `cam_uvd.valid`), so
  `rrp train pointer rep|flow|bc` no longer KeyErrors on the first step; `tests/unit/test_pointer_train_smoke.py` runs two
  steps of each trainer on a synthetic pack. Already-collected packs (`artifacts/datasets/pointer_v1/`) are unchanged and
  read as zero geometry; re-collect to train with real geometry.
- Readiness C0 (audit D24, pointer module split): `policies/pointer.py` is now the package `policies/pointer/` and the
  `torch` nets (`UICtx`, `PointerEncoder`, `PointerRealizer`, `PointerFlow`, `PointerBC`) live at module level in
  `policies/nets/pointer.py`; `import rrp.policies.pointer` (the scripted / registry route) stays torch-free and
  `nets()` / the checkpoint loaders import them lazily. `harness/train/pointer.py` is a package with one training loop,
  `train.fit`, that `rep`, `flow`, `bc`, `probe` and the `edit` probe share. Public import paths and the `rrp train pointer`
  entry point are unchanged; `policies.pointer.STEP_M` is gone (use `PointerGeometry.step_m`).
  - Geometry: `PointerGeometry.from_spec(env.spec)` (screen px, m/px, control Hz) replaces the hard-coded 0.1 s tick,
    the `ScreenFrame()` default, `px = [480, 320]` and `STEP_M`. Trainers read it from the env once (`pointer_geometry()`);
    `collect` stamps it into new packs (`geom`) and `Demos` refuses a pack of another geometry (old packs carry none and
    are taken as the spec's). The float32 `half` (features) and float64 `half_m` (trainers) differ by 1 ulp exactly as
    before, on purpose: it keeps frozen checkpoints and features bit-identical.
  - Bound entities: `features.check_bound_ids` raises `ValueError` when two different bound entity ids of one scene hash
    to the same bucket (`bound_id` is unchanged, N_BOUND = 16); before, they silently merged. No scene of the four cw/* tasks
    collides (checked on 20 seeds each).
  - Frozen-checkpoint parity (old single-file code vs the split, CPU, seeded): E / R / S / BC outputs, S loss, SHA256 of the
    public features of every cw/* task, and 2-step `rep` / `flow` / `flow --target eng` / `bc` log rows, weight digests and
    saved versions all compare equal; the net sums are pinned in `test_frozen_pointer_nets_give_identical_outputs`.
    `edit`'s probe now trains under bf16 autocast on CUDA like the other probes (it did not before); a diagnostic, CPU
    unchanged.
  - Not touched (D-146 item 7): the recorded `cworld_pointer_v1` split and version strings.
- Readiness C1 (audit D21 / D22, pointer training correctness). Re-checked on origin/main 6f8e4910 before editing: all five
  findings still held (collect called `public_features` with no table, so packs lacked z-layer / parent / focus rank / UI
  edges and a `preset:ui` net trained on zeros while rollout fed real edges; the nets took no `factors` from the trainers
  and `save_checkpoint` stamped the empty-preset hash regardless; `--seed` also seeded the train / validation split; flow
  noise came from one shared generator, so a packet depended on the batch and the rollout order).
  - One featurizer path: `collect_episode` calls `public_features(..., table=env_widget_table(env))` exactly as the three
    live call sites do (`table` is now a required keyword of `public_features`; `widget_features` without one yields the
    same keys with empty UI values). Packs gain `wzlayer` / `wparent` / `wfocusrank` / `wuiedges` (edges bit-packed per
    table; `data.stack_tables` / `unpack_edges`; `collect.write_pack`), and `widget_features` emits `wgeo_ok`, so
    `Demos.batch` and a live batch have identical key sets, dtypes and (tick 0) values (test). Old packs load with empty UI
    fields flagged `wui` False (`Demos.has_ui`); UI factors refuse them (`FactorError`), default factors still train.
  - `--factors <relations.resolve items>` on `rep` / `flow` / `bc`: resolved with `family="pointer", training=True`, stored
    in `arch[<net>]["factors"]`, the net is built with it, `save_checkpoint(factors=specs)` stamps `versions["factors"]`
    (`stamp_versions`, so a flow no longer inherits its representation's hash) and `load_pointer_bundle` rebuilds from the
    saved specs and `require_factors` (a missing / different hash is refused; `allow_factor_mismatch` for ablations). `mix > 0`
    is refused (no relgen scene mix in these trainers). `assert_deployable` guards `PointerSystemI` / `PointerBCPolicy`.
    Default factors are byte-identical to before (frozen-nets sums unchanged; old checkpoints load).
  - `ui.drag_to` (probe source) is supervised, not left untrained: `nets.pointer.drag_to_label` builds its pair label from
    the public batch fields (focused widget -> nearest other widget by `wpos3d`, `relgen.ui.drag_to_fn`'s definition), the
    trainers run `relations.estimates_loss` on the forward's `RelCtx` (`UICtx.record_rc` / `last_rc`) and log
    `fx_probe_ui.drag_to`. Not done here: scene-mix (`mix`) and gt-source labels from relgen (C3 / R1 own those).
  - `--split-seed` (default 0) seeds the train / validation split; `--seed` seeds torch / numpy only, so a seed sweep keeps
    one held-out set. Both are recorded in the checkpoint config.
  - Flow noise: `PointerFlow.sample(b, nfe, noise=None)` replaces `generator=`; `PointerSystemI` feeds
    `flow_noise(policy seed, env.seed, packet index)` per env row (fresh CPU generator from a hash of the triple, counters
    restart at `reset`), so z is identical at batch 1 and batch N and an episode replays exactly.
  - Tests: `tests/unit/test_pointer_train_smoke.py` (UI fields in the batch, split seed, factor stamp / load / refusal,
    drag_to term in all three trainers, collect == live featurizer, batch-1 == batch-N, noise triple) and
    `tests/unit/test_pointer.py` (`ui.same_window` / `ui.label_for` each change attention, `drag_to_label`). The frozen-nets
    golden is unchanged (the sample call moved to `noise=`, same numbers).
- Readiness C3 (audit D7 / D1: pointer recipes and ComputerWorld scene parts). Re-checked on origin/main b98e1150 first: the
  pointer stages and `pointer_v1` existed, but the lineage had one training seed, no way to name the split (the stages always used
  `cworld_pointer_v1`), no factor list passed through (`--factors` never reached the trainers), no sealed evaluation at all
  (`options.seed_set` took `sealed_*` and simply ran it, again), and `relgen` had no ComputerWorld part for `compose`.
  - Stages (`pipelines/pointer.py`): `options.split` names the split file and every stage passes it to the trainers as
    `--split` (the no-leak guard and the eval seeds come from one file; a missing file is refused before any job: declare the
    split first); list params are repeated values (`params.factors: [preset:ui]` -> `--factors preset:ui`; empty = absent);
    `options.seed_sets` evaluates several sets in one node. Sealed guard: `sealed_id` / `sealed_heldout` run at most once per
    (split, seed set, task, method = policy kind + digest of the frozen checkpoint paths), logged as start / done in
    `artifacts/runs/pointer/sealed_log.jsonl` (superseded by PC below: now `core.sealed.SealedSplit`). A crash leaves the attempt open; a bad result never re-opens it. The
    consumed `cworld_pointer_v1` (D-142) refuses its sealed sets outright. Dev is unguarded. `PIPELINE_VERSION` is unchanged
    (F3 owns `base.py`); `pointer_v1` / `pointer_smoke` keep their config hashes (the template did not change).
  - Recipes (all on `research/splits/cworld_pointer_v2.json` (C2: procedural strings; the eval stages pass the split's `env_kw` as `--env-kw`)):
    `pointer_seeds` (variant semfix|nosem x seeds 1,2,3; BC and the engineered flow per seed at variant nosem, shared demos in
    lineage `pointer-v2-data`; `eval_oracle` global), `pointer_ui` (same graph, rep and flow with `preset:ui`, no baselines; same
    collect run), `pointer_copy` (variant nosem x seeds 1,2,3: eng.v2 discrete key with the free head, eng.v2 with the copy head, BC with the copy head; same collect run), `pointer_sealed` / `pointer_ui_sealed` (frozen checkpoints of those lineages on both sealed sets, nothing
    trained). Dry-run node counts: pointer_seeds 50, pointer_ui 37, pointer_copy 19, pointer_sealed 19, pointer_ui_sealed 12. `rrp run-dag recipes/pointer/<name>.yaml --dry-run`; peer
    only, as pointer_v1. Resume: nodes are adopted by config hash; a sealed node that crashed needs `SealedSplit.record_infrastructure_failure` first (PC).
  - Scene parts (`relgen/ui.py`): `cw_viewport` (activates `viewport`; width, height, m_per_px of `make_env(scene=...)`) and
    `cw_depth` (activates `zstack`; `depth` stack|constant, `dz`), env `computerworld`, no entities; each has a `vary` that
    changes only its own scene keys (decoupling pairs). `compose({"viewport","zstack"}, "computerworld", rng)` returns the kwargs.
    Nothing calls them in training yet.
  - Roadmap and pre-registration (written before any of these runs; dev = the v2 dev seeds, 50 per task, per-task success out of 50,
    three training seeds per arm, reported as mean and per-seed values; a difference counts only when it has the same sign in at
    least 2 of 3 paired seeds and exceeds the larger within-arm seed range; otherwise the result is "no measurable difference").
    Sealed sets are evaluated once per arm and seed after the dev table is written down, for every arm, whatever dev says.
    | row | question | arms | metric and rule | status |
    |---|---|---|---|---|
    | P-SEEDS | Is D-142's semfix vs nosem vs eng ordering stable over training seeds? | pointer_seeds: semfix, nosem, eng, BC | dev and sealed_id success per task; rule above; no claim about semfix on a single seed | planned |
    | P-UI | Do the ui.* relation factors help vs none? | pointer_ui vs pointer_seeds (same variant, same demos) | dev per-task success (expected: drag_window, fill_form, open_type); UI wins if mean gain >= 4 points on >= 2 tasks and no task loses more than the seed range | planned |
    | P-COPY | Does a copy mechanism fix unseen-word typing (0/50 heldout words and names, D-142)? | pointer_copy: eng.v2 flow with the copy key head vs the free head, and BC with the copy head vs pointer_seeds' BC (the latent-route realizer cannot copy, so learned-realizer arms are not in this row) | sealed_heldout open_type and fill_form success (procedural strings); copy wins if >= 10/50 on both with the same dev success elsewhere | planned (dev first; sealed once per arm and seed) |
    | P-KEY | Does a discrete key code let a learned system i type? | pointer_copy flow_eng (cw_pointer_eng.v2, free head) vs pointer_seeds eng (PC: stated explicitly as cw_pointer_eng.v1, free head; the pipeline default is now v2 + copy), same demos and seeds | dev open_type / fill_form success; wins if the dev gain exceeds the seed range | planned |
    | P-CURR | Does relgen scene variation (viewport, z-stack) make the pointer robust to frame changes? | mix of `cw_viewport` / `cw_depth` scenes at 0, 0.25 vs none | success on a viewport not seen in training and on constant-depth scenes; wins if the shifted-frame success gap to in-frame closes by half | blocked: the trainers do not call `relation_batches`, the `ui.*` catalog entries name no scene part in `gen` (so `mix > 0` is refused), and the data path has to render scenes with the composed kwargs |
- Readiness round 2, unit PC (pointer closure). Re-checked on origin/main 6d5ed2ab first: the gap held except checkpoint
  stamping (`stamp_versions` / `require_factors` were already wired; `test_factors_are_threaded_stamped_and_hash_checked_on_load`
  covers it). Changes:
  - Pipeline (`pipelines/pointer.py`): `SPLIT_PATH_V2` imported from the split module (one definition); `env_kw`
    (`strings=procedural`) reaches every job as `--env-kw k=v`; eval and train defaults are `cw_pointer_eng.v2` + key head `copy`
    (`V2_EVAL_DEFAULTS`; pointer_v1 / P-SEEDS state v1 / free explicitly); sealed guard is `SealedSplit("cworld_pointer_v2")`
    (cell = body, method, train_seed, scenes, task, seed_set; 7 cells per method), the local run-once log is gone; `options.max_seeds`
    (dev only) bounds the smoke; local `_env` keeps the inherited PYTHONPATH (see lead_questions: `StageContext.env()` drops it).
  - Collect (`train/pointer/collect.py`) runs on `harness.rollout` (`DemoTeacher`, source scripted_teacher, privileged, + `DemoRecorder`);
    digests identical to the old collector (c2c83c9c713bc7ce, 30 rows; 81dfc0a7eb242537, 19 rows). The pack stores the teacher drag
    target (`ep_drag`, `ep_drags`).
  - Trainers call `relation_batches` (`train/pointer/relmix.py`, `RelStream`); `ui.drag_to` label v2 (`relgen/ui.py`,
    `DRAG_TO_VERSION = "2"`) is the TEACHER's drag target, needs cap `teacher_drag`, and `factor_loss` refuses a drag_to head without
    labels (the old net-proxy label is removed). Mixed relgen shards are refused by name while `load_shard_rows` returns `inputs == {}`.
  - Recipes: `--curriculum` flag on rep / flow / bc; `pointer_smoke` = 4 episodes, 20 steps, v2 split, `max_seeds: 2`, `preset:ui`.
  - Tests (red then green): `test_pointer_pc.py` (9), `test_pointer_c3.py` (sealed rewritten), `test_relations_r20.py` (drag),
    `test_pointer_train_smoke.py`. Unit suite: 1418 passed, 100 skipped, 2 deselected.
  - Peer smoke 2026-09-30 (`pointer_smoke`, own RRP_PEER_REPO, one lease at a time, stopped): 11/11 nodes completed after one
    `--retry-failed` (first attempt: ModuleNotFoundError computerworld, from the PYTHONPATH bug above). Collect: 4 episodes, 128 ticks,
    teacher_failures 0, `ep_drag` (4, 2) and `ep_drags` (4,) in the pack. Eval (dev, 2 seeds, 20-step models): bc 0/2, eng 0/2
    (timeout), i.e. a wiring check, not a result. Raw: peer `artifacts/runs/pointer/pointer-smoke-*`.
  - P-CURR stays blocked on the RG redesign (`track/r2-rg`, unmerged): shards carry no `inputs`. No real training run was started;
    sealed v2 evaluation needs a written decision first (T8: dev tables only).
