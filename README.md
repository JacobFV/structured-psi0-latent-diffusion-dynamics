# structured-psi0-latent-diffusion-dynamics

> Public research repo, shared openly. Formerly `relational-robot-policy` (renamed 2026-09-25). The Python package/CLI is still `rrp`, and the working checkout on the GB10 hosts is still `~/work/relational-robot-policy`.

Research system for **morphology-general robot control with a structured, semantic action latent**:
a ψ₀-inspired stack where a planner ("system i") emits a continuous latent action packet `z` that carries the task's
meaning (which object, which manipulator, what relation should change, relative geometry, contact frames,
uncertainty), and a fast embodiment-specific controller ("system 0") realizes that same `z` online from the robot's
own proprioception and touch sensing. Simulation is native MuJoCo; everything runs on two NVIDIA GB10 machines.

> **Status (2026-09-21):** research code, not a product. The corrected architecture (below) is being implemented on
> branch `correction/controller-facing-latent`; the earlier direct-action policy is kept only as a baseline.
> No learned policy is competent yet — see [`STATUS.md`](STATUS.md) and
> [`research/decisions.md`](research/decisions.md) for the honest current state.

---

## architecture in one picture

```text
system ii (optional VLM: ψ₀ System-II / Qwen3-VL) + supplied task graph (events, roles, dependencies)
        │  observations, task/event structure, morphology            ← public inputs only
        ▼
system i   cached typed context (morphology · scene · task/events · interaction) + flow sampling
        │
        ▼
LatentActionChunk  z[knots=4, assemblies=M, 64]     ← the SAME tensor is (a) supervised for semantics,
        │                                              (b) transmitted, (c) consumed by system 0
        ▼
system 0   z + morphology + current joint state + touch/grip sensors + elapsed phase → joint targets, every 50 ms
        ▼
joint-target tracker (500 Hz physics)  →  MuJoCo
```

- **Semantics live on the packet.** `probe(received_packet, query, opaque_handle)` answers visible / looking_at /
  focused_on / held_by / acting_on / relative position / desired change / manipulator subtask. The probe never sees
  scene features or labels, only `z` and opaque handles.
- **System 0 never sees the task.** No task graph, instruction, object estimates or planner hidden state reach it.
  Task meaning arrives only through `z`; current sensor feedback arrives every tick.
- **Four clocks:** system-i replan 0.4 s · latent knots 4 × 0.2 s · system-0 feedback 20 Hz · physics 500 Hz.
- Design contract and audit: [`research/corrections/controller-facing-semantic-latent.md`](research/corrections/controller-facing-semantic-latent.md).
  Original assignment: [`docs/handoff/`](docs/handoff/).

## quickstart

Requirements: Linux aarch64 or x86_64, Python 3.12, [`uv`](https://github.com/astral-sh/uv), systemd user session
(the resource broker uses user cgroups), Node 20+ only for the UI.

```bash
git clone https://github.com/JacobFV/structured-psi0-latent-diffusion-dynamics.git && cd structured-psi0-latent-diffusion-dynamics
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[sim,service,ml,dev]'   # CPU torch is fine for tests
scripts/fetch_menagerie.sh                                             # pinned third-party robot assets (~1.7 GB)

# one-time: measure free capacity and create the enforced project slice + watchdog
PYTHONPATH=src python3 -m rrp.cli ops init --role host
PYTHONPATH=src python3 -m rrp.cli ops start-watchdog

# fast checks (CPU, ~1 min)
.venv/bin/python -m pytest tests/unit -q
```

Everything non-trivial runs **inside a broker lease** (a systemd unit with CPU/memory caps and heartbeats):

```bash
PYTHONPATH=src python3 -m rrp.cli ops run --cpu 2 --mem 4G --label my_job -- <command>
PYTHONPATH=src python3 -m rrp.cli ops status
PYTHONPATH=src python3 -m rrp.cli ops stop --owned-only --lease <lease_id>   # never stop other people's jobs
```

## common tasks

| goal | command |
|---|---|
| validate a robot asset / attachment | `rrp assets validate --robot panda_tf3` |
| validate a task graph | `rrp task validate tasks/support_and_insert.json` |
| generate teacher demonstrations | `rrp data generate --config configs/data/pick_place_primary_v3dart.json` |
| pack a dataset for training (memory-mapped) | `rrp data pack --config configs/latent/rep-latent_sem_v1.json --out artifacts/packed/<name>` |
| train latent representation (encoder + system 0 + probes) | `rrp latent train-representation --config configs/latent/rep-latent_sem_v1.json` |
| train system-i flow over the latent | `rrp latent train-flow --config configs/latent/flow_latent_sem_v1.json` |
| measurement probes / metadata-only control | `rrp latent fit-probes --representation … --packed-dir … --out … [--metadata-only]` |
| closed-loop evaluation (packet probes scored online) | `rrp latent evaluate --checkpoint … --robots panda_pg2 --episodes 20 --out …` |
| held-packet disturbance test | `rrp latent disturbance --checkpoint … --robots panda_pg2 --out …` |
| latency (NFE sweep, system-0 deadlines) | `rrp latent latency --checkpoint … --out …` |
| campaign cell (sealed protocol) | `rrp latent cell --method latent_sem --seed 1701 --base-flow-config configs/latent/flow_latent_sem_v1.json` |
| render a labelled demo video | `scripts/render_episode.py --robot panda_pg2 --seeds 3000001 --source learned --checkpoint …` |
| interactive workbench (loopback only) | `rrp workbench --port 8765` then open `http://127.0.0.1:8765/?token=$(cat ops/workbench-token)` |

(`rrp` = `PYTHONPATH=src .venv/bin/python -m rrp.cli`; wrap heavy commands in `ops run`.)

Baselines (old path, kept for comparison only): `rrp train policy --config configs/model/policy-small-structured.json`
(direct actions + hidden-state auxiliaries) and `rrp train codec --config configs/model/codec-small.json`
(action-only codec). `rrp adapt grpo|expo` holds the verified flow-SDE GRPO / EXPO-FT-inspired code (currently wired
to the old action path; being re-targeted to latent packets).

## repository map

```text
src/rrp/
  contracts/   typed schemas: RobotSpec, PolicyObservation vs PrivilegedTruth, TaskDefinition,
               ActionChunk (native, old path), LatentActionChunk (the system-i → system-0 packet)
  ops/         resource broker, cgroup enforcement, watchdog, peer discovery (see "resources")
  morphology/  procedural arms/grippers/legged bodies, MuJoCo Menagerie importers, module surgery + validation
  sim/         MuJoCo sessions, sensors + object tracker, privileged truth, snapshots, dual-arm + legged scenes
  tasks/       event-hypergraph compiler, runtime guards, receipts/provenance, versioned graph edits
  control/     joint-target controller, IK, scripted teachers, legged trackers, ψ₀ contracts,
               latent_realizer.py (system 0)
  data/        featurizer (the only definition of what a policy may see), collection, generation, packing
  model/       context banks + flow policy, semantic_latent.py (target encoder), latent_probes.py,
               codec.py (baseline), attention with typed structural bias
  learning/    latent_train.py (representation, flow, SFT, probes), behavior.py (baselines), GRPO/EXPO
  policy/      latent_runner.py (system i), runner.py (old native path), workbench policy registry
  evaluation/  closed-loop evaluators, latency, statistics, identifiability audit, campaigns, release gate
  service/     FastAPI + WebSocket workbench backend
ui/            React + TypeScript workbench (three.js scene, React Flow task graph, Packet inspector)
tasks/         supplied task graphs (pick_place, reach_pose, support_and_insert, handover, waypoint_contact)
configs/       data / model / latent / eval / adapt / vlm configs; resources.local.json (machine-specific)
research/      decisions.md (append-only decision log), corrections/, splits/, methods/, reports/, registry.jsonl
artifacts/     receipts, assets ledgers, videos (large datasets/checkpoints are not committed)
docs/handoff/  the original assignment package (preserved verbatim)
tests/         unit/, integration/, browser/ (Playwright), gpu/
```

## data and information boundaries

- **Public vs privileged.** Policies see only `PolicyObservation` (proprioception, declared sensors, tracked object
  estimates with uncertainty, the supplied task graph and its runtime status). Simulator truth goes on a separate
  `PrivilegedTruth` bus and is used only as training labels, rewards and evaluation.
- **Teachers are labelled.** Scripted teachers use privileged state and are always marked `scripted_teacher`; they
  produce demonstrations, never "learned" results.
- **Splits are frozen before results.** Held-out arm family (xArm7), held-out attachment combination (Panda +
  three-finger); see `research/splits/` and the sealed protocol `configs/eval/latent_slice1.json`.

## resources and machines

- **Host** (shared): at most 50% of currently free CPU; memory at most 80% of free, GPU authorized (user decision
  D-027). Enforced by the broker plus a watchdog that sheds our own jobs under external pressure.
- **Peer** (`gb10-direct`, dedicated): unrestricted per user instruction (D-026), with a whole-project 100 GiB
  memory ceiling to protect the OS. The code/venv workspace is RAM-backed (`/dev/shm/rrp-brandonin`); datasets and
  packs live on `~/rrp-peer-data`. Rebuild after a reboot with `scripts/peer_bootstrap.sh`; sync code with
  `scripts/peer_sync.sh push`.
- No cloud spend, no public services (loopback only), no physical robots.

## development conventions

- **Branch:** active work is on `correction/controller-facing-latent`; `main` holds the pre-correction line (tag
  `pre-correction-272c689`). Commit small, push often (private remote).
- **Testing:** research stage, so keep it light (see `AGENTS.md`). Test math that defines losses/likelihoods, information
  leakage, resource safety and split leakage. Otherwise prefer a short real run over a new unit suite.
- **Honesty rules:** every reported number comes from a saved raw run; failed experiments are preserved; interventions
  and teacher/privileged sources are labelled everywhere (UI included).
- **Decisions:** anything that changes a plan, threshold, gate or interpretation goes into `research/decisions.md`
  with evidence.

## where to look next

- current state and resume steps → [`STATUS.md`](STATUS.md)
- why things are the way they are → [`research/decisions.md`](research/decisions.md)
- the correction driving current work → [`research/corrections/controller-facing-semantic-latent.md`](research/corrections/controller-facing-semantic-latent.md)
- agent operating rules → [`AGENTS.md`](AGENTS.md)
- original assignment and acceptance criteria → [`docs/handoff/`](docs/handoff/)
