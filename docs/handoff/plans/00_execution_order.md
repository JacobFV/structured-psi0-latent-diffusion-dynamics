# implementation and research execution order

> for agentic workers: execute the four subsystem plans task by task, using installed subagent-driven-development or executing-plans skills where available. the user's execution method is autonomous local implementation. routine review gates are internal; do not wait for new approval after each task.

**goal:** deliver the complete relational robot research system and an executed, audited campaign.

**architecture:** source-locked simulation/task/observation contracts feed a cached multi-bank flow policy and controller adapters; the workbench and batch workers use the same runtime; a resource broker admits every significant job.

**tech stack:** Python, PyTorch, native MuJoCo, typed schemas, FastAPI/WebSockets, React/TypeScript, graph editor, optional verified acceleration. exact package versions are pinned after the ARM64/GB10 audit.

**spec:** `docs/00_scope_and_acceptance.md` through `docs/10_autonomy_and_recovery.md`, with `reference/original-design-v0.1.md` retained for provenance.

## global constraints

Shared host ≤50% of currently free resources, aggregated across all descendants. No unverified host GPU work. No sudo/global changes, paid services, arbitrary network discovery, remote publication, or physical actuation. No privileged deployment inputs. Every claim has raw evidence. Candidate assets and code-only methods are not executed-policy results.

## review focus

- dynamic node/command dimensions, passive/mimic joints and controller ownership must stay aligned.
- stale graph/model/controller versions must invalidate KV and queued control, including during reconnect.
- high-level stage order is not functional composition; outputs require actual typed provenance.
- missing or occluded sensor observations must not become false predicates or privileged truth.
- resource/SSH interruption must preserve other jobs and avoid duplicate training after resume.

## dependency sequence

P01–P06 (`01_ops_and_runtime.md`) establish schemas, resource safety, graph/runtime and information boundaries.
P07–P12 (`02_simulation_and_workbench.md`) establish physics, teachers, assets/surgery, full GUI and replay.
P13–P19 (`03_models_and_learning.md`) establish codec, flow model, semantics/VLM, training and real adaptation.
P20–P25 (`04_campaign_and_delivery.md`) establish controlled data splits, actual confirmation/breadth, latency, independent analysis and final delivery.

The first vertical slice follows P01,P02,P03,P04,P07,P08,P10,P11: resource-safe teacher-driven arm plus editable graph. Preserve teacher labeling. Continue through ALL remaining tasks. GUI/asset work can overlap peer model runs only under broker leases. Each task is broken into its own red-green-review-commit steps; expensive runs are separately budgeted and registered.

## target repository map

```text
src/rrp/
  contracts/       typed schemas, references, observations, actions, run manifests
  ops/             discovery, telemetry, aggregate broker, cgroup backend, watchdog, jobs
  morphology/      import, graph encoding metadata, attachment ports, surgery, validation
  sim/             native backend, sensors, truth bus, snapshots, render service
  control/         native tracking, original psi/sonic contracts, legged trackers
  tasks/           compiler, role bindings, guards, runtime, receipts, retries, interventions
  data/            teacher collection, manifests, chunking, normalization, feature cache
  model/           body/scene/event encoders, KV bank, factorized flow, codec, heads
  learning/        behavior, auxiliary losses, SFT, flow-SDE GRPO, edit/off-policy baseline
  evaluation/      splits, episode runner, latency, interference tests, statistics
  service/         HTTP/WebSocket sessions, commands, graphs, probes, authentication
  cli.py           doctor, ops, assets, task, workbench, data, train, adapt, evaluate, campaign
ui/src/
  scene/ graph/ inspector/ playback/ resources/ transport/
tests/
  unit/ integration/ gpu/ browser/ support.py conftest.py
configs/
  resources.local.json robots.json campaign.json model/ data/ eval/ adapt/
research/
  methods/ registry.jsonl decisions.md sources.lock.json splits/ reports/
artifacts/
  datasets/ checkpoints/ episodes/ metrics/ figures/ video/ release/
```

The package's `config/` files are templates/input policies. The implementation writes resolved executable configurations into its own `configs/`, preserving the protected resource fields. Actual CLI commands and tests are created by these plans; they do not already exist in the handoff.

## first-hours operating rhythm

Read/preserve the handoff; audit machines and existing work; write status/registry; build/test aggregate resource enforcement; create primitive schemas/task runtime; run the first tiny native simulation. Do not spend the first session cloning every model or installing a distributed framework. Maintain a short progress brief once a working artifact exists. Do not promise the whole campaign fits before Saturday.
