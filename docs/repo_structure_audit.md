# repository structure audit (2026-09-26, origin/main fab0670, read-only)


The main problem is that three pipelines grew side by side: arm, dual-arm and legged. Each has its own collect, train, DAgger, eval and checkpoint format, and each is started a different way. Around them sit 89 shell scripts that copy each other. The unit tests mostly pass: 143 pass and 2 fail, and both failures are environment problems, not code bugs. I changed nothing, and the audit worktree has been removed.

## 1. Top-level layout

**`src/rrp`** has 139 modules, about 28.3k lines, in 12 subpackages plus 7 `cli*.py` files at the top.
- `control/` (18 files) mixes several unrelated things:
  - IK and joint-target controllers
  - scripted teachers (arm, dual, legged)
  - legged RL training: `tracker_training`, `tracker_nets`, `legged_core.LeggedEnv`, `legged_vec`
  - system 0 (`latent_realizer`)
- The legged system 0 lives in a different place again: `model/legged_latent.py:LeggedRealizer`.
- `learning/legged_t1_diag.py` (602 lines) is a one-off diagnosis sitting in library code.

**`scripts/`** holds 134 files: 89 `.sh`, 42 `.py`, plus `demo/` (14 files), `dev/` and `analysis/`.
- 82 of them hardcode specific `artifacts/runs/<name>` paths, so they are one-off.
- 44 hardcode `/dev/shm/rrp-brandonin`, and 37 hardcode `$HOME/work` or `/home/brandonin`.
- **Chain drivers:** `arm_lineage_chain.sh` says in its own header that it is a "copy of armnosem_chain.sh". `arm_lineage_hybrid.sh` and `arm_seed2_host.sh` (232 lines) are further copies. The same `need()`/`node()` DAG-with-marker-files code is repeated in 4 scripts, and the 13-body list is repeated in 5 files.
- **Versioned families:** `latent_chain{,_v2}`, `binding_chain_v3/v4`, and about 30 `legged_t1_*`, `t1_diag_*`, `legged_r1/r2_*`, `legged_fix{rep,sem}_*`.
- **Ad-hoc checkers:** `ladder_peek`, `locpeek`, `sumpeek`, `t0_check`.
- `scripts/ladder.py` has the same name as `rrp.evaluation.ladder`.

**`configs/`** holds 264 JSON files in 13 directories with five naming schemes:
- `flow_X`/`rep-X` (hyphen) against `rep_X` (underscore) in the legged directories.
- `pack-target-…`, and `rz_*` for system-0 refits.
- Lineage codes: `sfjf`, `nsjf`, `sejf2`, `jointfix`, `b1fix`, `bindv4`.
- Suffixes: `_v1/_v2s3/_s1/_lv4/_long/_noqd/_gdag2h/_pfA-D`.
- Axis names disagree: arm uses `semfix`, legged uses `fixsem` or `fixrep`, and `t1_diag` uses `sem_lv4`.

Other config problems:
- 202 of 264 configs are never referenced by exact filename. Many are built dynamically (for example `$C/flow_${LIN}.json`), so treat this as an upper bound on dead configs.
- Sibling configs differ only in name, seed and paths. For example, `armsemfix/flow_sfjf.json` and `armseed2/sfjf2/flow_sfjf2.json` differ in 4 fields. That is templating done by copy-paste.
- `configs/ladder/rz_jointfix_bcdag1.json` lists 13 DAgger `.npz` paths inline.

**`research/`**:
- `tracks/` holds 2757 lines, including a `BRIEF.md` that is not a track.
- `tracks/ladder/sprint_final/` holds 55 raw JSON/JSONL result files, and `tracks/ladder/armnosem/ladder_v1/<body>/` holds per-body raw output. Raw results belong in `artifacts/`.
- `reports/*.txt` are also raw probe output.
- `registry.jsonl` was last touched 2026-09-25 12:00, and its last entry still says `latent_chain_v2` is `"running"`.
- `artifacts/requirements.json` was last touched 09-24.
- Both files are required by AGENTS.md and both are stale.

**`artifacts/`** has 2600 tracked files, about 179 MB, including:
- 242 mp4 files
- 30 DAgger `.npz` shards in `runs/ladder_dagger_bc1`
- 39 `.genctx.pkl` files of about 11 MB each (`runs/ladder_dagger_gdag1`)
- 3 `.pt` files that were force-added even though `*.pt` is gitignored

This conflicts with "never commit datasets". On top of that, `docs/demo/` (442 files, 24 MB) duplicates 176 blobs from `artifacts/`, via `docs/demo/raw/artifacts`. `.gitignore` also has duplicate entries (`.cache`, `.venv`, `artifacts/datasets`), and `ops/legged/sup_*.log` logs are committed.

**`tests/`**: `tests/test_binding_aug.py` sits at the root, outside `unit/` and `integration/`.

## 2. Code architecture

**Parallel pipelines**

| stage | arm | dual | legged |
|---|---|---|---|
| collect | `data/collect.py`, `generate.py` | `data/collect_dual.py`, `dual_pairs.py` | `data/legged_collect.py`, `legged_latent_collect.py` |
| dataset | `learning/packed.PackedChunkDataset` → `latent_train.LatentData` | `learning/dual_latent` (`meta.json`) | `legged_latent_train.LeggedData` |
| train | `latent_train.py` (representation, flow, SFT, probes, refit: 766 lines) | via `cli_dual_latent` | `legged_latent_train.py`, `legged_bc.py` |
| DAgger/refit | `evaluation/ladder.save_dagger` + `scripts/ladder_refit.py` → `latent_train.refit_realizer` | – | `learning/legged_dagger.py` (collect, gate, refit) |
| eval | `evaluation/ladder.py`, `latent_eval`, `latent_causal`, `latent_semantic_edits` | `dual_latent_eval.py` | `legged_latent_eval.py` (643 lines) |
| entry | `rrp latent …` or `scripts/*.py` | `rrp latent *-dual` | `python -m rrp.learning.legged_*` (21 `__main__` modules) |

**Inconsistent abstractions**
- Sessions: `LeggedSession(Session)` and `DualSession(Session)` in `sim/`, but legged RL training uses a separate physics path, `control/legged_core.LeggedEnv`. There is no shared protocol: the only `Protocol` in the codebase is in `tasks/runtime`.
- Teachers: `PickPlaceTeacher`, `DualTeacherBase` and three subclasses, `WaypointTeacher`, plus four more teacher-like classes defined inside evaluation code (`ShadowTeacher`, `ShiftedGoalTeacher`, `TeacherSource`, `_DualTeacher`).
- Oracle routes are implemented three times: `ladder.OraclePacketPolicy`, `latent_semantic_edits.OracleSource`, `legged_latent_eval.OracleShadow`.
- Packet builders are implemented three times: `ladder.make_packet`, `build_packet`, `dual_packet`.
- Trackers: `LearnedTracker`, `CPGTracker`, and a separate `sim/sensors.ObjectTracker` that shares the name but not the meaning.

**Copy-pasted helpers**
- `wilson()` in `evaluation/statistics` and `evaluation/ladder`, plus two copies in scripts.
- `_featurizer` in `ladder` and in `latent_semantic_edits`.
- `_dev` three times, `_seeds` three times.
- The featurizer version string exists twice: `data/collect.FEATURIZER_VERSION` and `learning/behavior.FEAT_VERSION`, both `"feat-v2"`.

**God-modules (lines, and what they mix)**
- `latent_semantic_edits` (918): arm and dual edits, teachers, sources, metrics.
- `latent_train` (766): 6 unrelated training entry points.
- `ladder` (727): featurizer patching, teachers, policies, metering, DAgger I/O.
- `dual_teachers` (721).
- `legged_latent_eval` (643): adapters, BC, oracle, video.
- `cli_latent` (541): about 15 commands with business logic inline.

**Layering**
- Import cycles between subpackages: `control↔model`, `control↔learning`, `control↔sim`, `control↔morphology`, `data↔sim`, `data↔model`, `learning↔evaluation`, `learning↔policy`, `policy↔service`, `morphology↔sim`.
- Concrete bad dependencies:
  - `sim.native` and `sim.legged` depend on `control`
  - `sim.dual` imports `data.features_multi`
  - `model.batch` and `model.attention` import `data.features`
  - `model.system2` imports `sim`
  - `control.latent_realizer` imports `learning.packed`
  - `policy.registry` imports `service.sessions` and `learning.latent_train`
  - `learning.adapt` and `learning.sft` import `evaluation`

**CLI chain**
`cli` → `cli_ext` → `cli_ml` → `cli_train`, `cli_latent`, `cli_adapt` → `cli_dual_latent`. Each link is wrapped in `try/except ImportError: pass` (`cli.py:231`, `cli_ext.py:48`, `cli_ml.py:36-46`), so a real import bug makes subcommands silently disappear.

**Dead or unreferenced code**
`evaluation/bc_semantic_edits`, `latent_slice1_report`, `system2_eval` and `learning/qa_train` are referenced only in `research/*.md`.

**Config handling**
- 223 `cfg.get("key", default)` calls across 23 files. Defaults are spread through the code, so changing a default in code silently changes what old configs do.
- 12 separate `@dataclass` config types.
- 141 distinct top-level JSON keys, with no schema.
- Only 1 config is versioned.
- 81 configs have no `out_dir`; 74 have an `out_dir` basename that differs from `name`, because code adds `ladder_`/`legged_` prefixes.

## 3. Contracts and provenance

- **No physics or contact version in datasets or checkpoints.**
  - `physics_version` and `contact_version` appear nowhere.
  - `mujoco.__version__` is recorded only in `sim/native.py:530` (snapshots) and `tracker_validation`.
  - Dataset manifests store the config and status counts only.
  - Legged `impratio`/`cone` overrides (`morphology/legged.py:88,456`) are not recorded anywhere.
- **Three manifest formats:** `write_manifest` (hashed) for arm and dual, a hand-written `manifest.json` without a hash in `legged_collect.py:123`, and `meta.json` files in packed data and dual.
- **Two checkpoint formats:**
  - `learning/checkpoint.save_checkpoint`: versions, RNG, sha sidecar.
  - Legged `_save(**kw)`: no versions, no digest.
- **System-0 compatibility IDs:**
  - Arm uses weight-fingerprinted `bundle_versions` (`control/latent_realizer.py:187`).
  - Legged uses `latent_space_version=f"legged-ls-{name}"` and a constant `REALIZER_COMPAT="legged-rz-osc-v1"` (`evaluation/legged_latent_eval.py:41`). A legged system-0 refit therefore cannot be told apart from the original.
- **Source labels are inconsistent:**
  - `contracts/action.Source` includes both `teacher` and `scripted_teacher`, and a `privileged_teacher` value, while there is also a separate `privileged_teacher=True` flag.
  - `latent_action` uses `target_encoder_oracle`, while ladder and eval code use the bare string `"oracle"`.
  - Sources are also written as free strings (`learned:{ckpt}`, `"encoded teacher targets…"`, `cli_latent.py:227`).
  - Legged uses a separate `tracker_source` ("scripted_controller" / "learned_tracker").
- **`zero_prev_action`** (the B-1 fix) is set per config (115 configs) and read with a `False` default in 7 files. It is not stored in the packed metadata, and the legged and dual paths never look at it. A config that leaves it out silently reproduces the contaminated behaviour.

## 4. Experiment orchestration

There are four ways to launch a job:
1. `rrp ops run` on the host.
2. `scripts/peer_run.sh`: ssh plus `ops run`, used by 10 scripts.
3. Chain scripts that call `python3 -m rrp.cli ops run` directly on the peer: 14 scripts, each with `cd $P/wt/ladder` hardcoded.
4. `nohup`/systemd units, per the registry.

Other orchestration problems:
- Chain state is kept as `.done`/`.failed` marker files in `artifacts/runs/*_state` or `.slot_done_*`, with a different mechanism in each script.
- Output naming is inconsistent: `artifacts/runs` prefixes include `legged` (58), `ladder` (26), `acceptance` (19), `t1diag` (13) and `t1` (6), against the documented rule `<track>_…`.
- The host has 152 committed run directories; the peer store has 359.

**Running on the peer right now** (from `/dev/shm/rrp-brandonin/wt/ladder`):
- `scripts/ladder_dagger_collect.sh`
- `scripts/ladder_refit.py configs/ladder/armseed2/sfjf2/…`
- `rrp.cli latent train-flow --config configs/ladder/armseed2/sfjf2/…`

These are the arm seed-2 lineages `nsjf2`, `sejf2` and `sfjf2`. Those script, config and module paths must not change until the jobs finish.

## 5. Docs consistency

- **Resource limits disagree:** AGENTS.md says "≤50% of free, host GPU off"; CLAUDE.md, README and STATUS say 80% and host GPU allowed (D-033). `docs/handoff/AGENTS.md` differs from the root AGENTS.md.
- **README is stale:**
  - It says active work is on `correction/controller-facing-latent` and `main` is pre-correction; CLAUDE.md says the correction branch is merged into `main`.
  - It says "private remote"; CLAUDE.md says the repo is public.
  - It sends readers to AGENTS.md for the testing policy, which is actually in CLAUDE.md.
  - "system-0 feedback 20 Hz" is stated for everything, but the legged path runs at 50 Hz.
  - It says `rrp adapt grpo` is on the old path, but latent GRPO exists.
  - It does not mention the ladder, the legged or DAgger pipelines, or the `python -m` entry points.
- **STATUS is stale:** its header reads "Updated 2026-09-25 14:00" although the file was committed 09-26, and its track table lacks ladder, legged and t1.
- **BRIEF** still describes the 09-25 sprint.

## 6. Tests

I ran `pytest tests/unit`. With `-x` it stops at the first failure; run in full it gives **143 passed, 2 failed**, and the slowest test takes 2 s. Both failures are in `test_prev_action_col.py` (`KeyError 'panda_pg2'`). They need the Menagerie assets in `.cache/assets` and are not skipped when those are absent, so they fail in any fresh checkout. `addopts=-q` combined with `-q` hides the summary line.

Coverage: 74 modules, about 16.1k lines, are never imported directly by any test. These include `latent_train`, `ladder`, `latent_semantic_edits`, every legged train, eval and DAgger module, `data/features` (the definition of what a policy may see), `sim/legged` and `service/sessions`.

## Target structure

```
rrp/
  contracts/      packet, observation, action, Source enum, Provenance (physics, featurizer, bundle, git)
  physics/        mujoco options, contact version, snapshot; PHYSICS_VERSION
  bodies/         morphology (arms, grippers, legged, aloha, menagerie import), catalog registry
  tasks/          graphs, runtime, receipts, scenarios (single/dual/legged)
  envs/           Env protocol; ArmEnv, DualEnv, LeggedEnv (sim session + vectorized pool)
  features/       featurizers + FEATURIZER_VERSION (one place)
  teachers/       Teacher protocol; arm/dual/legged scripted, BC expert, shadow/oracle wrappers
  controllers/    joint targets, IK, trackers (learned/CPG), realizers (system 0, arm + legged)
  models/         encoder, flow (system i), probes, codec/baselines
  data/           collect → shards → pack, one manifest writer with Provenance
  training/       representation, flow, bc, refit, dagger, sft, grpo (body-agnostic loops)
  evaluation/     rollouts, ladder routes, edits/causal, latency, statistics
  pipelines/      Pipeline(body_family) : collect/train/dagger/eval stages + stage registry
  orchestration/  ops (broker/cgroup/watchdog), DAG runner (replaces *_chain.sh markers), peer transport
  cli/            one Typer/argparse tree: rrp {ops,data,train,dagger,eval,run-dag}; no silent ImportError
  research/       diagnostics (t1_diag, peek scripts) — allowed to import anything; nothing imports it
```

Layering rule: contracts → physics → bodies → tasks → envs → features → teachers/controllers/models → data → training/evaluation → pipelines → orchestration/cli. Add an import-linter check to CI to keep it that way.

**Config schema.** Use pydantic `RunConfig`:
- Fields: `schema_version`, `family` (arm/dual/legged), `stage`, `lineage`, `seed`, `inputs` (by run id, not path), `out` (derived as `artifacts/runs/{track}/{lineage}/{stage}_s{seed}`), `flags` (such as `zero_prev_action`, required with no default), `note`.
- Configs inherit from a base, so seed and lineage variants become short overlays or matrix entries in a single DAG file instead of copied JSON.

**Naming.** Use `variant ∈ {sem, nosem, semfix}` × `seed s{n}` × `stage ∈ {rep, flow, flowft, rz, dagN-{bc|gen}}`. Keep a table mapping old codes (`sfjf`, `nsjf`, `sejf2`, `fixsem`, `jointfix`) to new names in `research/naming.md`.

**Scripts.** Replace the chain `.sh` files with `rrp run-dag dags/<lineage>.yaml`: nodes, dependencies and resources declared once, and state kept in a JSON ledger. Keep `peer_{run,sync,bootstrap}.sh` and `fetch_menagerie.sh`. Move the peek, diagnostic and one-off scripts to `research/scripts/<date>/`, frozen.

## Migration plan

**Phase 0 (now, zero risk)**
- Freeze `scripts/`, `configs/ladder/armseed2/**` and `rrp.learning.latent_train`, `rrp.evaluation.ladder` and `rrp.cli latent` until the peer `wt/ladder` jobs finish.
- Fix the Menagerie skip in the two failing tests.
- Deduplicate `.gitignore`; stop committing `.npz`, `.pkl`, `.pt` and `docs/demo/raw` (use git-lfs or leave them on the peer store).
- Refresh `registry.jsonl`, STATUS and README.

**Phase 1 (additive)**
- Add `contracts/provenance.py`, `physics` versioning, the `Source` enum and a single manifest writer.
- Write provenance for new runs, and backfill legacy runs with a sidecar marked `legacy=true` instead of rewriting them.
- Give legged checkpoints weight fingerprints; keep reading bare `_save` checkpoints and treat them as `unfingerprinted`.

**Phase 2 (move with shims)**
- Create the new packages and leave re-export stubs at the old paths (`rrp/learning/latent_train.py: from rrp.training.representation import *`), with a DeprecationWarning.
- Checkpoints mostly store `state_dict`s and plain dicts. The `.pt` files are not in git; they live in the peer store. So loading them depends on key and config names, not class import paths. Still run a load test over all `*.pt` files on the peer before removing any shim, because `weights_only=False` can pickle arbitrary objects.

**Phase 3 (unify pipelines)**
- Build `Pipeline(family)` with arm first.
- Check parity: the new arm pipeline must reproduce `ladder_rz_*` gate metrics with fixed seeds.
- Then port legged, then dual.

**Phase 4 (orchestration)**
- Introduce `rrp run-dag`, and convert completed lineages to DAG files only for provenance.
- Remove chain scripts once no lease references them. Check with `rrp ops status` on both nodes.

**Phase 5**
- Remove the shims after one full cycle of no peer jobs using old paths (check with `grep` over the `/dev/shm/rrp-brandonin/wt/*` process cmdlines).

**Risks**
- The peer `wt/<track>` directories are synced copies, so new code reaches them only through `peer_sync.sh push`. Never push into `wt/ladder` or `repo` while chains are running.
- Re-deriving `out_dir` from a new naming scheme would orphan the chain marker files. Keep the old paths as aliases, for example a symlink map.
- Configs that leave out `zero_prev_action` must not change meaning when the flag becomes required. Migrate them by writing the old default explicitly.