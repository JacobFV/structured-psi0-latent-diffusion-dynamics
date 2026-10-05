# project status: structured-psi0-latent-diffusion-dynamics (formerly relational robot policy)

Updated 2026-09-30 (D-146 readiness round 2, unit X2). Overall: **code for the pre-training campaign merged; no run has
started.** Every progress state below uses one vocabulary: planned, implementing, test_failed, verified, running,
completed, failed_hypothesis, blocked_external, budget_exhausted (`rrp.core.runs.RunState`; `verified` and `completed`
name the recorded command or artifact). `parked` is a scope tag, not a state. The repository layout is `schema.toml`
(docs/architecture.md section 13); legacy material is under `.old/` (index `.old/README.md`); the previous long-form
status, evidence summary and work log are `.old/STATUS.md`.

Where to read: plan `docs/strategy.md` (D-094), readiness plan and ledger `research/readiness.md`, open questions
`docs/experiments_roadmap.md`, what is shown `research/reports/evidence_matrix.md`, why `research/decisions.md` (latest
entries D-144..D-146), rules `AGENTS.md`.

## tracks (`schema.toml [tracks]`; note `research/tracks/<track>.md`, recipes `recipes/<track>/`)
`state` is the progress vocabulary (also the `State:` line of the track note). The `schema.toml` open / paused flag is a
separate scheduling switch that decides which recipe trees may be tracked; it stays as is until the campaign opens.

| track | workstream | state | schema flag | next step | decisions |
|---|---|---|---|---|---|
| relations | relation-factor experiments | planned | open | T9 re-planned on the v6 semfix lineage, PRE-REGISTERED (`research/tracks/relations.md` "T9 v6 pre-registration"; `recipes/relations/relations_v6.yaml`); T9-on-v8div moot (armdiv G3 FAIL); v3dart comparison halted (floor) | D-144, D-147 |
| humanoid | W13 humanoid transfer | running | open | D-147 campaign: T1 h_steps/h_gap excluded; T2 gait ub_v1 t1/g1/h1; T3 under exception; T4 re-gate: h_reach/h_squat_pick pass; sigma-0 T5 VOID; DART pilot t1 h_turn zero-shot 97/100 (learned) -> full DART rebuild of h_walk/h_turn/h_reach/h_squat_pick, every arm + LOO, running on the peer since 10-04 21:59 (`~/work/rrp-data/campaign/STATUS.md`) | D-138, D-139, D-146, D-147 |
| armdiv | W7 training-arm diversity | failed_hypothesis | closed (G3) | T6 phase B: v8div semfix lineages done; G3 lineage gate FAIL 362/480 = 0.754 < 0.835 (v6 425/480); STOP per signed G4 (no nosem / kinfeat / sealed / v6ref) (`research/tracks/armdiv.md`) | D-137, D-146, D-147 |
| psi0 | W10 Psi0 / SIMPLE | planned | paused | T7: label recording, stage A with the D-141 fix, packet-use gate (structured arm was 0/20 vs direct 19/20) | D-141, D-146 |
| pointer | ComputerWorld pointer policy | planned | paused | T8: v2 re-collect, rep, flow / bc x 3 seeds, dev tables (`recipes/pointer/`) | D-142, D-146 |

Closed tracks (W1 contact v2, W2 to W6 hygiene / provenance / restructure / pipeline / robustness, W7 arm expert, W8
legged regeneration, W9 target bodies, W11, W12 and the earlier ladder / acceptance / baselines / binding lines): notes in
`.old/research/tracks/`, decisions in `research/decisions.md`, raw results in `artifacts/runs/` (frozen run names listed in
`schema.toml`). Their DAG templates stay as generic family recipes in `recipes/templates/`. The loopback workbench is
retired (D-145 addendum): the visualization room (`viz/room`, D-131) is the only viz surface. The dual arm is parked
(D-146): collect, pack and evals on an external checkpoint stay; no training, refit, DAgger or BC node.

## readiness ledger (`research/readiness.md`; main 037c73ac)
State `verified` here means: merged to main through the unit suite (`pytest tests/unit -m "not slow"` green at the merge)
with the commit below as evidence. It is not the final integration check, which has not run.

| unit | scope | state | evidence (commit) |
|---|---|---|---|
| round 1 (32 units) | F0 F1 F2 F3 K1 K2 H5 A3 RP1 HT HJ HL A1 R1 P1 P2 C0 A2 RP2 U1 H7 H4 C1 P3 RP3 U2 HX C2 C3 U3 H6 RP4 | verified | `main 4af47836` |
| G0 | fresh-checkout green, slow marker | verified | `032358a3` |
| RC | relation registry closure | verified | `24859b51` |
| FS | factor stamping at every loader | verified | `2a196429` |
| SL | split-agnostic sealed guard, sealed-log CLI | verified | `ea1335a1` |
| TK | humanoid task / teacher closure | verified | `d1ae3002` |
| HS1 | public range ring, sensor and env fixes | verified | `30c0ea58` |
| HD1 | humanoid collect on rollout, upper targets, terrain | verified | `288b3779` |
| RG | relgen shards reach the arm trainers | verified | `3fb304fd` |
| AR | armdiv closure (v6 reference recipe, pin procedure) | verified | `da85efec` |
| DP | dual parking finished | verified | `1026b134` |
| RL | last private step loops on rollout | verified | `c2b9c901` |
| HS2 | wholebody tracker training, morph_v2 shared, install, sealed guard | verified | `4316a26c` |
| HD2 | task-agnostic legged / humanoid trainers, upper supervision | verified | `3e2e3f47` |
| HP | legged policy from checkpoint factors, upper check | verified | `86337a45` |
| PS | Psi0 closure (gate stage, labels stage) | verified | `a07c8f22` |
| PC | pointer closure (v2 split, SealedSplit, relgen) | verified | `1decd724` |
| RP5 | recorder as rollout hooks | verified | `51ff3c75` |
| HA | adapting trainers, humanoid family stages, native sealed cells | verified | `bd8adc37` |
| HR | humanoid recipes for every task and tracker run | verified | `26e59cf9`, `9b2be4c3` |
| X1 | residual dedupe and lints (`harness.hooks`) | verified | `037c73ac` |
| X2 | docs truth pass (this file, README, docs, track states, `tests/unit/test_docs_truth.py`) | verified | `tests/unit/test_docs_truth.py`; merge commit: `git log --grep "X2:"` |

## campaign-opening runs (section B of the readiness plan; none started)
All `planned`; nothing starts before the final integration check is green. Order and gates are in
`research/readiness.md` section B.

| run | what | state | recipes |
|---|---|---|---|
| T0 | peer store check: pinned trackers and v7div pack / BC 1701 hashes | planned | `ops/bin/`, `recipes/humanoid/`, `recipes/armdiv/` |
| T1 | terrain-scan trackers `h_steps`, `h_gap` (+ `range_ring`) | planned | `recipes/humanoid/` |
| T2 | wholebody trackers `*_ub` with the amplitude / payload ramp | planned | `recipes/humanoid/` |
| T3 | shared `morph_v2` tracker (legs + upper) | planned | `recipes/humanoid/` |
| T4 | humanoid collect for every `h_*` task, teacher quality table | planned | `recipes/humanoid/` |
| T5 | humanoid lineages (`preset:legged` vs `legged-none`), dev transfer tables | planned | `recipes/humanoid/` |
| T6 | armdiv: BC 1702 / 1701 pins, v6 reference evals, G4 signature, v8div lineages | failed_hypothesis | `recipes/armdiv/arm_lineage_v8div.yaml`; G3 FAIL, `research/tracks/armdiv.md` |
| T7 | Psi0: labels, stage A, packet-use gate, structured head, eval | planned | `recipes/psi0/` |
| T8 | pointer v2 lineages, dev tables | planned | `recipes/pointer/` |
| T9 | relation-factor experiments on the v6 semfix lineage (none / geo / ix / task x 2 seeds), GPU nodes raced peer vs host (`relations_v6_host.yaml`), interference table; v8div attempt moot (G3 FAIL), v3dart attempt halted (floor) | running | `recipes/relations/relations_v6*.yaml` |

## known open code gaps (not owned by any merged unit)
- `LearnedTracker` cannot yet load the shared `morph_v2` tracker checkpoint; T3's gate needs it. State: planned.
- The gap trainer writes no `range_ring` actor meta, so the T1 `h_gap` run with the ring cannot be gated. State: planned.

## running work
Nothing is running. Before starting anything: `rrp ops status` on both nodes and `systemctl --user list-units 'rrp-*'`;
attach to owned jobs, never duplicate. After a host reboot restart the watchdog (AGENTS.md).

## resolved environment
- repo `~/work/relational-robot-policy` (GitHub JacobFV/structured-psi0-latent-diffusion-dynamics; host `Dell-gb10-1`, aarch64 GB10); parallel agents use `~/work/rrp-wt/<track>`.
- peer `gb10-direct` (all of it, D-008/D-026); workspace `/dev/shm/rrp-brandonin/{repo,venv,cache,bin}` (RAM-backed); shared repo dir is for the lead only, every other agent uses `wt/<track>`.
- host venv `.venv` (CPU only); peer venv `/dev/shm/rrp-brandonin/venv` (torch cu130, mujoco). Host broker caps and the peer admission limits: `ops/resources.local.json`; disk reserve 100 GB (D-086).
- watchdogs: `rrp-watchdog-host.service`, `rrp-watchdog-peer.service` (user services in `rrp-control.slice`).

## how to run anything (always under a lease)
```bash
cd ~/work/relational-robot-policy
PYTHONPATH=src python3 -m rrp.cli ops status
PYTHONPATH=src python3 -m rrp.cli ops run --cpu 1 --mem 2G --label NAME -- <cmd>        # host: light jobs only
export RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/<track>; ops/bin/peer_sync.sh push       # peer: own dir, then
ops/bin/peer_run.sh --gpu --gpu-mem 12G --cpu 4 --mem 24G --label <track>_x --max-seconds N --detach -- PY -m rrp.cli run-dag recipes/<track>/<name>.yaml
```

## known limits (details: evidence matrix, `.old/STATUS.md`)
- Legged results through D-092 use contact v1; contact v2 trackers exist for anymal_c, go2 and t1 (t1 under a labelled D-112 gate exception, D-113 / D-139).
- Arm results before D-110 used grasp v1; v6 (teacher v2 + grasp v2.1) is the current arm baseline (D-134). The latent route does not transfer to a new arm and never beats BC there (D-135).
- Host: no heavy compute (D-115, D-127). Peer: disk about 97% used; memory pressure sheds leases (declare at least 1.35 x peak, D-117).

## resume
Read this file, `docs/strategy.md`, the latest entries of `research/decisions.md` and the track note; then the note's
RESUME `rrp run-dag` lines (idempotent ledgers under `artifacts/runs/<track>/_dags/`).
