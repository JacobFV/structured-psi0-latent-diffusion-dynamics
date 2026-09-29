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

## one repo, three interfaces (D-140)

Every approach is a **policy** and every simulator an **environment** behind one interface each; a **task** says which
environments it exists in and how an episode is judged; one **harness** (`rrp.harness.rollout`) runs any
policy × env × task that `negotiate()` accepts and records declined pairs with their reasons. Design, interfaces and the
moved-path table: [`docs/architecture.md`](docs/architecture.md).

| kind | registered names |
|---|---|
| environments (`rrp.envs.base.make_env`) | `mujoco/arm`, `mujoco/dual`, `mujoco/legged` (`control="base_velocity"` through the embedded tracker, or `"legs"` joint targets), `warp/legged` (batched GPU), `simple` (Ψ₀ SIMPLE, Isaac Sim), `computerworld` (UI world as 3D, in progress) |
| policies (`rrp.policies.base.make_policy`) | `bc`, `latent` (system i + system 0; sem / nosem / semfix variants), `legged_latent`, `legged_bc`, `oracle` (privileged diagnostic), `teacher:<task>` (scripted_teacher), `psi0_direct`, `psi0_structured`, `psi0_replay` |
| tasks (`rrp.tasks.spec.TASKS`) | pick_place, reach_pose, support_insert, handover, assign_left/right, pivot_against_surface, carry_tray_level, waypoint_contact, loco_pick, foothold_steps, h_steps, h_gap, locomotion, `simple/<Task>` |

Lineage codes in config and run names (`sfjf`, `nsjf2`, `fixsem`, `gendag3_noqd`, …) are decoded in
[`research/naming.md`](research/naming.md). The training lineages (arm ladder, legged, dual arm) run as pipeline stages
(`rrp.harness.pipelines`, families arm / dual / legged) from DAG files in `dags/`; evaluation routes on matched seeds are
R0 teacher, R1 oracle packet E(expert chunk) (DIAGNOSTIC), R2 generated packet (deployable), plus a BC positive control
and task-context edit suites.

## entry points

`rrp` = `PYTHONPATH=src .venv/bin/python -m rrp.cli`; wrap anything heavy in `rrp ops run` (see resources).

| goal | command |
|---|---|
| evaluate any policy × env × task | `rrp eval --policy NAME[=JSON] --env ENV --task TASK --body BODY --seeds a:b --out F` (`harness.eval.evaluate` over `harness.rollout`, family hooks from `harness.eval.hooks`; JSONL rows + Wilson summary) |
| compatibility matrix (n/a with reasons) | `rrp matrix …` |
| a whole lineage (collect → pack → Stage A → flow → DAgger / refit → eval → edits) | `rrp run-dag dags/<lineage>.yaml` (resumable JSON ledger; stage list: `rrp stage list`) |
| generate / pack arm data | `rrp data generate --config configs/data/…`; `rrp data pack --config … --out artifacts/packed/<name>` |
| arm Stage A / flow / probes | `rrp latent train-representation --config …`; `rrp latent train-flow --config …`; `rrp latent fit-probes …` |
| arm ladder evaluation (R0/R1/R2) | `rrp suite ladder --route {teacher,oracle,generated} --robot panda_pg2 --n 30 --out …` |
| arm task-context edits | `rrp latent semantic-edits --route {teacher,oracle,generated,bc} …` |
| evaluation suites / audits | `rrp suite {ladder,legged,robustness,target,tracker-validation,privileged-audit,…} …` |
| legged trackers (GPU PPO) | `rrp train tracker-warp --recipe <recipe> --out …` |
| Ψ₀ fine-tunes | `rrp train psi0 …` (SIMPLE eval needs the Isaac venv: `scripts/psi0_ext.sh`) |
| labelled video | `scripts/render_episode.py`, `scripts/render_legged_episode.py`, `scripts/render_dual_episode.py` |
| demo page | `scripts/demo/refresh.sh` (builds `docs/demo/` from raw results) |
| visualization room | `cd viz/room && npm run snapshot` (exporter `rrp viz export`) |
| workbench UI (loopback) | `rrp workbench --port 8765` |

## quickstart

Requirements: Linux aarch64 or x86_64, Python 3.12, [`uv`](https://github.com/astral-sh/uv), a systemd user session
(the broker uses user cgroups), Node 20+ only for the UI.

```bash
git clone https://github.com/JacobFV/structured-psi0-latent-diffusion-dynamics.git && cd structured-psi0-latent-diffusion-dynamics
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[sim,service,ml,dev]'   # CPU torch is fine for tests
PYTHONPATH=src:. .venv/bin/python -m pytest tests/unit -q              # ~2 min; Menagerie/data tests skip if absent
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
src/rrp/          layers import only downward (tests/unit/test_layering.py)
  core/           contracts: observation vs privileged truth, NativeCommand / ActionChunk, latent packet + system-0
                  protocol, RobotSpec (morphology graph), TaskDefinition, provenance + source labels, RunConfig, paths
  ops/            resource broker, leases, cgroup enforcement, watchdog, telemetry, peer discovery
  bodies/         procedural + Menagerie arms, grippers, aloha, legged, humanoids, G1 (g1_simple), IK, physics versions
  tasks/          task-graph compiler/runtime/receipts, TaskSpec registry (spec.py)
  envs/           base.py (Env, EnvSpec, capabilities, make_env); mujoco/ (sessions, scenes, sensors, trackers);
                  warp/ (batched GPU legged envs); simple/ (Ψ₀ SIMPLE)
  policies/       base.py (Policy, negotiate, make_policy); features/ (featurizers: the only definition of what a
                  policy may see); nets/; bc, latent, system0, legged, oracle, packets; teachers/ (scripted_teacher);
                  psi0/
  harness/        rollout.py (the one episode loop), eval/ (evaluate.py: evaluate / matrix; hooks.py; suites), train/,
                  data/, pipelines/, dag.py (run-dag)
  viz/            room exporter, recorder/replay, workbench service
  cli/            the `rrp` command (tools.py: `rrp <group> <tool>` for data / train / suite / stage / viz tools)
ui/               React + TypeScript workbench
viz/room/         visualization room (exporter output in viz/data)
configs/, dags/   run configs and lineage DAGs (provenance of every run)
scripts/          peer transport (peer_run/sync/bootstrap), asset fetch, renderers, demo builder, paused-track drivers
research/         decisions.md (append-only; appendix P = former psi1z), naming.md, tracks/, reports/, splits/, registry.jsonl
artifacts/        small raw results (JSON/JSONL), receipts, labelled videos
docs/             architecture.md, strategy.md, experiments_roadmap.md, robot_training_considerations.md, demo/, handoff/
tests/            unit/ (incl. test_golden.py: byte-identity of featurizers, teachers, physics, policies), integration/, browser/, gpu/
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

## the Ψ₀ line (W10)
Ψ₀ direct and Ψ₀ + structure on Ψ₀'s SIMPLE benchmark are policies (`rrp.policies.psi0`) on the `simple` env
(`rrp.envs.simple`, Isaac Sim 5.1 in its own venv: `scripts/psi0_ext.sh`, extra `rrp[psi0]`). The separate psi1z repo is
retired (D-140); its decisions are appendix P of [research/decisions.md](research/decisions.md), its notes and results
[research/tracks/psi0.md](research/tracks/psi0.md).
Open questions and planned experiments (what is still uncertain but in scope): [docs/experiments_roadmap.md](docs/experiments_roadmap.md).

## checkpoints and rollouts
Public (unlicensed; all rights reserved) on Hugging Face: [jacob-valdez/rrp-checkpoints](https://huggingface.co/jacob-valdez/rrp-checkpoints) — arm v1 routes, v2/v6 BC experts, legged8 routes, gait trackers, and 160 recorded evaluation rollouts (`rollouts/viz_replays`). `MANIFEST.tsv` there lists sha256 per file. Weights are never committed to this repo.
