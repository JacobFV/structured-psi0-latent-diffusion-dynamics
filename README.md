# structured-psi0-latent-diffusion-dynamics

> Public research repo (github.com/JacobFV/structured-psi0-latent-diffusion-dynamics), shared openly. Formerly
> `relational-robot-policy` (renamed 2026-09-25, D-030). The Python package and CLI are still `rrp`; the working
> checkout on the GB10 hosts is still `~/work/relational-robot-policy`.

Research system for **morphology-general robot control with a structured, semantic latent packet**: a ψ₀-inspired stack
in which a planner ("system i") emits a continuous latent packet `z` that carries the task's meaning (which object, which
manipulator, what relation should change, relative geometry, contact, uncertainty), and a fast embodiment-specific
controller ("system 0") realizes that same `z` online from the robot's own proprioception and touch. Simulation is
native MuJoCo on two NVIDIA GB10 machines.

> **Status (2026-09-26):** research code, not a product. On deployable routes (no teacher, oracle or BC at run time) the
> latent route is competent on legged go2, hexapod6 and the t1 humanoid, and partly competent on arms (below plain BC).
> Task-context edits reach behaviour through the generated packet (goal and binding on arms, goal and halt on legs).
> Semantic supervision of the packet is essential on the arm and adds a halt channel on legs (D-089..D-092). All legged
> results so far use contact model v1, whose trackers skate (D-093); they are being redone on contact v2.
> What is and is not shown: [`research/reports/evidence_matrix.md`](research/reports/evidence_matrix.md).
> Plan and workstreams: [`docs/strategy.md`](docs/strategy.md). Problem checklist for physically credible training:
> [`docs/robot_training_considerations.md`](docs/robot_training_considerations.md). Running work: [`STATUS.md`](STATUS.md).

---

## architecture

```text
system ii (optional VLM) + supplied task graph (events, roles, dependencies)
        │  public observations, task/event structure, morphology
        ▼
system i   typed context (morphology · scene · task/events · interaction) → rectified-flow sampling of z
        │
        ▼
latent packet z[knots, assemblies, 64]   ← the SAME tensor is (a) supervised for semantics (probes),
        │                                   (b) transmitted, (c) consumed by system 0
        ▼
system 0   z + morphology + current joint state + touch/IMU + elapsed phase → native joint targets
        ▼
joint-target tracking in MuJoCo
```

- **Semantics live on the packet.** A probe reads `z` with opaque handles only (never scene features or labels). Evidence
  of meaning comes from probes and causal edits of the received packet.
- **System 0 never sees the task.** Task meaning arrives only through `z`; sensor feedback arrives every tick.
- **Clocks differ by body family:**

  | family | system-i replan | system 0 | physics |
  |---|---|---|---|
  | arm (single, dual) | 0.4 s, 4 knots × 0.2 s | 20 Hz (50 ms) | 500 Hz |
  | legged / humanoid | 0.4 s (every 20 ticks), 0.8 s packet horizon | 50 Hz (20 ms) | body-specific substeps |

- Semantic vs no-semantic ("nosem") variants are capacity-matched: nosem sets the semantic loss weights to 0. The
  bounded-NLL fix (`latent.probe_lv_min: -4`, D-085) is part of the current sem recipe ("semfix"/"fixsem").
- Design contract: [`research/corrections/controller-facing-semantic-latent.md`](research/corrections/controller-facing-semantic-latent.md).
  Original assignment (historical): [`docs/handoff/`](docs/handoff/).

## pipelines

Three pipelines currently exist side by side (unifying them is W4/W5 in `docs/strategy.md`). Lineage codes in config and
run names (`sfjf`, `nsjf2`, `fixsem`, `gendag3_noqd`, …) are decoded in [`research/naming.md`](research/naming.md).

- **Arm ladder** (pick_place, 13 source bodies). Scripted teacher data → pack → Stage A representation (encoder E,
  system 0 R, probes P) → system-i flow → system-0 refits with DAgger labelled by a stateless learned BC expert,
  including states visited with generated packets (`rz_*`) → generator DAgger for the flow (`gdag*`). Evaluation
  routes on matched seeds: R0 teacher, R1 oracle packet E(expert chunk) (DIAGNOSTIC), R2 generated packet (deployable),
  plus a BC positive control and task-context edit suites. Driver: `scripts/arm_lineage_chain.sh`.
- **Legged** (go2, hexapod6, t1; others blocked at the positive control). PPO body trackers → tick-level teacher data →
  Stage A → flow → stateless-BC-expert DAgger and system-0 refit → R1/R2 ladder → context and z edit suites.
- **Dual arm** (support_insert, handover; M=2 assemblies): scripted teachers, paired arm-assignment data, pack and
  training smoke only; learned dual-arm control is not shown (D-043).

## entry points

`rrp` = `PYTHONPATH=src .venv/bin/python -m rrp.cli`; wrap anything heavy in `rrp ops run` (see resources).

| goal | command |
|---|---|
| generate / pack arm data | `rrp data generate --config configs/data/…`; `rrp data pack --config … --out artifacts/packed/<name>` |
| arm Stage A / flow / probes | `rrp latent train-representation --config …`; `rrp latent train-flow --config …`; `rrp latent fit-probes …` |
| arm system-0 refit | `python scripts/ladder_refit.py configs/ladder/<…>/rz_<…>.json` |
| arm ladder evaluation (R0/R1/R2) | `python scripts/ladder.py --route {teacher,oracle,generated} --robot panda_pg2 --n 30 --out …` |
| arm task-context edits | `rrp latent semantic-edits --route {teacher,oracle,generated,bc} …` |
| arm lineage end to end | `LIN=<lineage> bash scripts/arm_lineage_chain.sh` (peer) |
| legged trackers (PPO) | `python -m rrp.control.tracker_training …` |
| legged data / Stage A / flow | `python -m rrp.data.legged_latent_collect …`; `python -m rrp.learning.legged_latent_train …` |
| legged BC control / DAgger | `python -m rrp.learning.legged_bc …`; `python -m rrp.learning.legged_dagger {collect,gate,refit} …` |
| legged ladder / edits | `bash scripts/legged_ladder.sh BODY TAG PAR ROUTES …`; `python -m rrp.evaluation.legged_latent_eval …` |
| dual arm | `rrp latent {pair-index-dual,pack-dual,evaluate-dual,teacher-ref-dual} …` |
| packet-policy GRPO | `rrp latent grpo …` (latent path; on hold until a competent base, D-042) |
| latency | `rrp latent latency --checkpoint … --out …` |
| labelled video | `scripts/render_episode.py`, `scripts/render_legged_episode.py`, `scripts/render_dual_episode.py` |
| demo page | `scripts/demo/refresh.sh` (builds `docs/demo/` from raw results) |
| workbench UI (loopback) | `rrp workbench --port 8765` |

The old direct-action path (`rrp train policy`, `rrp train codec`, `rrp adapt grpo|expo`) is kept for baselines only.

## quickstart

Requirements: Linux aarch64 or x86_64, Python 3.12, [`uv`](https://github.com/astral-sh/uv), a systemd user session
(the broker uses user cgroups), Node 20+ only for the UI.

```bash
git clone https://github.com/JacobFV/structured-psi0-latent-diffusion-dynamics.git && cd structured-psi0-latent-diffusion-dynamics
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[sim,service,ml,dev]'   # CPU torch is fine for tests
.venv/bin/python -m pytest tests/unit -q                               # ~10 s; Menagerie/data tests skip if absent
scripts/fetch_menagerie.sh                                             # pinned third-party robot assets (~1.7 GB)

# one-time: measure free capacity and create the enforced project slice + watchdog
PYTHONPATH=src python3 -m rrp.cli ops init --role host
PYTHONPATH=src python3 -m rrp.cli ops start-watchdog
PYTHONPATH=src python3 -m rrp.cli ops run --cpu 2 --mem 4G --label my_job -- <command>
PYTHONPATH=src python3 -m rrp.cli ops stop --owned-only --lease <lease_id>   # never stop other people's jobs
```

Datasets, packed data and checkpoints are not in git; they live on the peer store and a host mirror (`~/work/rrp-data`).

## repository map

```text
src/rrp/
  contracts/   typed schemas: RobotSpec, PolicyObservation vs PrivilegedTruth, TaskDefinition, LatentActionChunk
  ops/         resource broker, cgroup enforcement, watchdog, peer discovery
  morphology/  procedural arms/grippers/legged bodies, Menagerie importers, module surgery, contact model versions
  sim/         MuJoCo sessions (arm, dual, legged), sensors, privileged truth, snapshots
  tasks/       event-hypergraph compiler, runtime guards, receipts
  control/     IK, joint-target control, scripted teachers, legged trackers (PPO), latent_realizer (arm system 0)
  data/        featurizer (the only definition of what a policy may see), collection, packing (arm, dual, legged)
  model/       context banks + flow, semantic_latent (encoder), latent_probes, legged_latent, codec (baseline)
  learning/    latent_train (arm Stage A/flow/refit), legged_* (legged training, BC, DAgger), GRPO/EXPO
  evaluation/  ladder, closed-loop evals, semantic edits, latency, statistics, campaigns
  policy/, service/   system-i runner, workbench backend
ui/            React + TypeScript workbench
configs/       data / latent / ladder / legged / eval configs
scripts/       chain drivers, peer transport (peer_run/sync/bootstrap), renderers, demo builder
research/      decisions.md (append-only), naming.md, tracks/, reports/ (evidence_matrix.md), registry.jsonl
artifacts/     small raw results (JSON/JSONL), receipts, labelled videos
docs/          strategy.md, robot_training_considerations.md, architecture.md, demo/, handoff/
tests/         unit/, integration/, browser/, gpu/
```

## rules that shape the results

- **Public vs privileged.** Policies see only `PolicyObservation`. Simulator truth is used only as training labels,
  rewards and evaluation.
- **Labels.** scripted_teacher (privileged) / oracle (teacher- or expert-encoded packet: diagnostic, not deployable) /
  bc / learned:<ckpt>. Every number comes from a saved raw output; failures are kept.
- **Splits are sealed before results** (`research/splits/`, `configs/eval/latent_slice1.json`).
- **Resources.** Host: ≤80% of currently free CPU and memory, host GPU allowed, ≥100 GB disk kept free; peer: all of it
  (D-026, D-033, D-086). Details and the peer workflow: [`AGENTS.md`](AGENTS.md). No cloud spend,
  no physical robots.

## development conventions

- **Branch:** `main` is the integration branch; parallel agents work in `track/<track>` worktrees and rebase onto main
  (`research/tracks/BRIEF.md`). Commit small, push often.
- **Testing:** light at this research stage (policy in [`AGENTS.md`](AGENTS.md)): test loss/likelihood math, leakage,
  resource safety and split leakage; otherwise prefer a short real run.
- **Decisions:** anything that changes a plan, gate or interpretation goes into `research/decisions.md` with evidence.

## where to look next

- current state and workstreams → [`STATUS.md`](STATUS.md), [`docs/strategy.md`](docs/strategy.md)
- what is shown → [`research/reports/evidence_matrix.md`](research/reports/evidence_matrix.md)
- why → [`research/decisions.md`](research/decisions.md); names → [`research/naming.md`](research/naming.md)
- agent rules → [`AGENTS.md`](AGENTS.md)

## related repository: psi1z (Ψ₀ + structured packets)
The Ψ₀ / SIMPLE benchmark (workstream W10) lives in [JacobFV/psi1z](https://github.com/JacobFV/psi1z) (not public; local `~/work/psi1z`). It installs this repo
as a library (`rrp.core`, pinned by git sha) and never copies its code; shared pieces change here first. Full map, dependency rules
and the decision crosswalk (rrp D-xxx ↔ psi1z P-xxx): [docs/related_repos.md](docs/related_repos.md). Current W10 status: D-120 in
[research/decisions.md](research/decisions.md) and psi1z's [research/notes.md](https://github.com/JacobFV/psi1z/blob/main/research/notes.md).
Open questions and planned experiments (what is still uncertain but in scope): [docs/experiments_roadmap.md](docs/experiments_roadmap.md).

## checkpoints and rollouts
Public (unlicensed; all rights reserved) on Hugging Face: [jacob-valdez/rrp-checkpoints](https://huggingface.co/jacob-valdez/rrp-checkpoints) — arm v1 routes, v2/v6 BC experts, legged8 routes, gait trackers, and 160 recorded evaluation rollouts (`rollouts/viz_replays`). `MANIFEST.tsv` there lists sha256 per file. Weights are never committed to this repo.
