# strategy (adopted 2026-09-26, D-094)

Inputs: `docs/robot_training_considerations.md` (problem checklist, D-093), `docs/repo_structure_audit.md` (structure audit),
`research/reports/evidence_matrix.md` (current evidence). This file says what we do, in what order, who owns it, and how each step
is judged done. Update the status column when a gate passes, citing the decision or track note.

## 1. Goal and principles
Goal: a clean, trustworthy platform on which the central claim (a semantically supervised latent packet from system i gives
causal task control through system 0 across bodies) is tested on physically credible behaviour, with fair baselines.

Principles:
1. **Physics credibility before new claims.** No new legged results on contact v1. Every dataset and model records its physics version.
2. **One pipeline, many bodies.** Arm, dual and legged share contracts, provenance, config schema, pipeline stages and orchestration.
3. **Additive then subtractive.** New structure lands next to the old one with shims; old paths are removed only once nothing uses them.
4. **Running experiments are never disturbed.** In-flight peer jobs run from synced copies (`/dev/shm/rrp-brandonin/wt/<track>`, `repo`).
   Never sync into a copy that has running jobs. Main can change freely.
5. **Gates, not vibes.** Every workstream step has a measurable acceptance gate (tests, parity numbers, validation metrics, videos).
6. **Honest labels.** Existing contract rules stand (sources, privileged info, fair budgets, no fake success).

## 2. Workstreams

| id | workstream | owner | depends on | status |
|---|---|---|---|---|
| W1 | Physics realism: contact v2, reward schedule, slip gate; then actuator realism and latency | contact agent | — | round complete (D-101/103/107): installed anymal_c, go2 (floor), t1 w8d (sourced limits); sourced torque limits default; permanent standing; g1 (overspins: cap yaw-progress reward) and h1 (retrain from scratch) parked; actuator dynamics/latency opt-in |
| W2 | Repo hygiene (phase 0) | hygiene agent | — | verified 2026-09-26 (main 3c796fd; tests 152 pass / 3 skip on a fresh clone; 78 dataset/weight files, 391 MB, untracked; copy in ~/work/rrp-data/git-untracked-2026-09-26) |
| W3 | Provenance and contracts (phase 1) | provenance agent | — | verified 2026-09-26 (main 510f052; 165 unit tests pass; arm/dual/legged smoke manifests carry full provenance; 22 configs made explicit zero_prev_action) |
| W4 | Package restructure with shims (phase 2) | restructure agent | W2, W3 | completed 2026-09-27 (P6 f1db76d: 338 unit tests, parity byte-identical, host audit 391 / peer audit post-P6 0 errors; legacy packages are shims only) |
| W5 | Unified pipeline + DAG orchestration (phases 3–4) | pipeline agent | W4; arm seed 2 finished | arm verified 2026-09-26 (D-096: R2/R1 parity row-identical; refit bit-identical to step 2900); legged smoke only; dual skeleton; ladder.py main → rrp.evaluation pending |
| W6 | Robustness sweeps + motion-quality gates | robustness agent | W1 v2 trackers | harness + metrics verified (D-108); legged seeds 1–2 sweep next; arm grasp-contact realism opened |
| W7 | Arm expert: smooth scripted trajectories, then GRPO fine-tuning with anchor evals | arm agent | arm seed 2 finished | teacher v2 + grasp v2 verified (D-097/D-110); v4dart lineage set stopped; v5dart + BC expert + lineage set regenerating; re-eval of old arm routes under grasp v2 |
| W8 | Legged regeneration on contact v2 (data → Stage A → flow → R2 → edits) | legged agent | W1 gate, W3 | anymal_c done (D-105, robustness D-112); go2 running; t1 rerun from collection on the sourced-limit w8d tracker (sha 36e91467) |
| W9 | Open claims: held-out target bodies; one loco-manipulation task | later | W5, W8 | planned |
| R0 | Arm seed-2 replication (running experiment) | arm agent | — | completed 2026-09-26 (D-095) |

### W1 Physics realism (running)
- **Scope:**
  - Contact v2 (elliptic cone, impratio, noslip, compliant sole, friction/restitution/mass randomization).
  - Slope stick/slide unit test.
  - Staged reward schedule (permanent / shaping-prior / natural-objective terms, gated α).
  - Slip-ratio and duty-factor gate.
  - Retrain trackers, humanoids first; v1 vs v2 videos.
  - Then: armature, joint friction and damping, motor torque–speed limits, randomized actuation latency 0–30 ms.
- **Gate:** slip ratio < 0.15 and tracking and no-fall at v1 levels or better on t1, h1, g1, go2, anymal_c, hexapod6; reviewed videos.
- **Status:** running.

### W2 Repo hygiene (phase 0)
- **Scope:**
  - Skip asset-dependent tests when the menagerie assets are absent.
  - Deduplicate `.gitignore`.
  - Stop tracking datasets (`.npz`, `.pkl`, `.pt`) and `ops/*.log`: untrack only; history is left as is.
  - Reconcile README / AGENTS / CLAUDE / STATUS / BRIEF with reality (resource rule D-033/D-086, public repo, current pipelines, rates).
  - Refresh `research/registry.jsonl` and `artifacts/requirements.json`.
  - Record the naming map (`research/naming.md`: sfjf/nsjf/sejf2/fixsem/jointfix … → variant × seed × stage).
- **Gate:** `pytest tests/unit` green on a fresh checkout without assets. No tracked dataset files. Docs consistent (checklist in the PR/commit).

### W3 Provenance and contracts (phase 1)
- **Scope:**
  - `rrp/contracts/provenance.py`: physics version incl. contact version and MuJoCo version, featurizer version (one constant),
    bundle fingerprint, git sha, flags such as `zero_prev_action`.
  - A single manifest writer used by arm, dual and legged collection.
  - A `Source` enum replacing free strings (scripted_teacher, privileged, oracle, learned:<ckpt>, bc, random, mock).
  - Weight-fingerprinted legged checkpoints and legged system-0 compatibility IDs.
  - Legacy runs readable, marked `legacy=true`.
  - `zero_prev_action` becomes explicit: configs that omit it are migrated by writing the old default.
- **Gate:** unit tests for each piece. A new tiny arm, dual and legged run each produce manifests with full provenance. Old checkpoints still load.

### W4 Package restructure (phase 2)
- **Scope:** the target layout in the audit
  (`contracts, physics, bodies, tasks, envs, features, teachers, controllers, models, data, training, evaluation, pipelines, orchestration, cli, research`),
  moved with re-export shims and a DeprecationWarning.
  - Remove the silent `try/except ImportError` in the CLI chain.
  - Break the import cycles; add an import-layering check to the tests.
  - Move one-off diagnostics (`learning/legged_t1_diag.py`, peek scripts) to `research/`.
  - Deduplicate helpers (wilson, featurizer, seeds, dev, oracle wrappers, packet builders).
- **Gate:** all tests green. Import-layer check passes. A load test of every `*.pt` on the peer store succeeds. The demo page builds unchanged.

### W5 Unified pipeline + orchestration (phases 3–4)
- **Scope:**
  - `Pipeline(family)` with stages collect / pack / train-rep / train-flow / dagger / refit / eval / edits.
  - A config schema (pydantic `RunConfig`, schema version, required flags, derived output paths, overlays and matrices).
  - `rrp run-dag dags/<lineage>.yaml` replaces the chain shell scripts (resources, dependencies, retries, JSON state ledger, host/peer placement).
  - Arm first, with a parity check, then legged, then dual.
- **Gate:** the new arm pipeline reproduces an existing lineage's gate metrics with fixed seeds (within seed noise). The chain scripts
  are retired once no lease references them.

### W6 Robustness sweeps + motion-quality gates
- **Scope:**
  - Evaluate any policy over a physics grid (friction, mass/CoM, PD gains, latency, pushes, terrain) and report break-points.
  - Motion-quality metrics (slip ratio, CoT, jerk, contact forces, joint-limit margin) in every eval.
  - Trackers and datasets must pass the gates.
- **Gate:** sweep reports for BC and latent routes on one arm and one legged body.

### W7 Arm expert and RL fine-tuning
- **Scope:**
  - Smooth, time-parameterized (minimum-jerk) scripted trajectories. Teacher success must not drop.
  - Regenerate arm data only with the gate passed.
  - Then GRPO with a success reward on the latent route and on BC under matched budgets, with anchor evals to catch forgetting.
- **Gate:** teacher jerk and success report. GRPO improves R2 success beyond seed noise without anchor regression, or an honest failed_hypothesis.

### W8 Legged regeneration on contact v2
- **Scope:** recollect with v2 trackers, then Stage A (sem / nosem / fixed sem), flows, R2, context and z edit suites, 3 seeds. Re-test D-088/090/092.
- **Gate:** the same statistical protocol as D-090. Results replace v1 results in the evidence matrix. v1 results are kept, labelled.

## 3. Sequencing and resources
1. **Now:** W1 (host CPU); W2 and W3 (code only, host); R0 (peer).
2. **After R0 completes (~2–3 h):** record D-095. Then W7 smoothing (host CPU), W4 restructure (code), and W5 design.
3. **After the W1 gate:** W6 sweeps on the peer CPU; W8 regeneration on the peer GPU (the biggest compute consumer, several GPU-days).
4. **After W4/W5:** W8 runs through the new pipeline where possible. W9 afterwards.

Compute: peer GPU for training and flows; peer CPU for simulation evals; host CPU for tracker PPO and code/test work. Host memory is
constrained by an external process, so keep host jobs ≤24 GiB. The utilization watchdog stays on.

Conflict rules:
- Each agent owns a worktree and a set of paths.
- W4 does not move files W1 or W3 are editing until they merge.
- W3 owns `contracts/`; W1 registers its contact version through W3's API.
  If W1 lands first, W1 adds a minimal `physics_version` field that W3 absorbs.


### Host: no heavy compute (D-115, user). All heavy work on the peer.

### Current peer allocation (2026-09-27 08:00, lead; revise here)
The peer's memory watchdog stopped every agent's jobs at 07:50 because the peer was oversubscribed. Until revised:
- W10 (Isaac): 2 GPU leases.
- W8: 3 GPU leases while the arm set awaits grasp-v2 data (then back to 2), ≤8 CPU.
- Arm v2 (W7): 2 GPU leases, ≤10 CPU.
- Contact (W1): peer CPU only when load < 15.
Every lease declares its measured peak memory + 20%, so the broker, not the emergency watchdog, does admission control.
Peer disk is 97% used (69 GB free): no new large downloads without checking.

## 4. Supervision
- **Lead (main session):**
  - Assigns scopes and checks gates against raw outputs.
  - Records decisions, keeps STATUS, this table and the evidence matrix current.
  - Keeps the machines busy (utilization watchdog), cleans up finished agents and waiters.
- **Agents:**
  - Report with the exact commands, raw paths and numbers.
  - Merge verified work to main with rebase.
  - Never touch other agents' leases, other users' processes, or other projects' files.


| R1 | Arm nosem recipe ablation (z-noise 0; keep qd; both), seed 1 | arm agent | R0 | completed (D-099): recipe does not explain nosem failure |
| W10 | Ψ₀ / SIMPLE benchmark in [psi1z](https://github.com/JacobFV/psi1z) (D-100, D-119; see docs/related_repos.md): probe → Ψ₀ baseline → Ψ₀ + our structure vs Ψ₀ direct expert | psi0 agent | W11 for packaging | step 0 done (D-104/D-109: stack reproduces TabletopGraspMP; XMovePick not); step 1 done (D-120: 3/6 reproduce: TabletopGraspMP, BendPickMP, HandoverTeleop); step 2 running on those (≤40 GPU-h) |
| W11 | rrp as an installable core for psi1z: Python 3.11 support, public API doc, core tests under 3.11 | core agent | W4, W5 | verified 2026-09-26 (main b7dc677: rrp.core API 1.0; py3.11 + 3.12 venvs 318 pass / 4 skip each; lead .venv 321 pass; register_family / rrp.families entry points; psi1z pinned at b7dc677, no remote) |
| W12 | Feature-centric coordination: anchor-relative (pose) packet targets, contact-event-aligned knots, orientation-drift / contact-sequence / re-anchoring metrics, anchor and sequence edits; dual-arm support_insert first | w12 agent | arm v6 lineages (for compute) | phase A running (D-122) |
