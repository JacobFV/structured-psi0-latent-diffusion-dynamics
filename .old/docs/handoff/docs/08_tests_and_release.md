# verification matrix and release gate

## tier 0 — ordinary CPU tests, no downloads

Contract schema tests, typed errors, unknown/null states, task guards/cycles/resource ownership, event-output identity/version/role order, stale edit conflicts, unit transforms, attachment namespacing and positive inertia, mask/permutation logic, resource budgeting, aggregate leases, watchdog and restart/idempotency tests. Use small deterministic fixtures. No model weights or real hardware required.

## tier 1 — numerical and native-physics integration

Primitive MuJoCo arm/leg fixtures; finite stepping; joint-limit and target tracking; mimic joints and actuator groups; full snapshot round trip; teacher task; action codec overfit/reconstruction; forward/backward flow; independent flow/action time conventions; masked all-null context sentinel; cached/uncached outputs; gradient routing; frozen QA decoder with live projector gradients; analytic GRPO transition math; bad variance/time-grid rejection.

Use double/float reference math where needed. Shared-weight node-order tests permute every relevant input and noise tensor. Test `n=1`, odd n, variable batch padding, empty optional banks, missing channels, long graph sequences and incompatible controller output widths.

## tier 2 — service and real browser

Spin a real backend and MuJoCo fixture. Verify HTTP/WebSocket schema, origin/auth, idempotency, reconnect, queue bounds, edit transaction, stale-action cancellation, actual robot movement, task execution request, probe, replay and control-mode labels. No fixture that only changes a screenshot without stepping physics. Record a short browser evidence video.

## tier 3 — actual peer GPU

Architecture/driver audit; small tensor kernel; real pretrained image/text features; actual codec/policy training step; useful tiny learning test; actual VLM-backed rollout; resource enforcement evidence; sampled memory/latency; single- and grouped-rollout GRPO; off-policy comparator; deterministic checkpoint reload within backend tolerance. Do not turn mandatory GPU failure into a skip while claiming the model is validated.

## tier 4 — campaign and scientific audit

Registered conditions, three-seed primary runs, actual per-episode outputs, censored failure handling, source/controller/asset manifests, failed hypotheses retained, reproducible plots, and claim-to-artifact links. Randomly select raw episodes/checkpoint to rerun, not only the best demo. Verify no future data or exact simulator truths entered deployment policy input.

## proposed operator commands to IMPLEMENT

```bash
uv run rrp doctor --role host --write artifacts/host.json
uv run rrp ops verify --config configs/resources.local.json
uv run rrp assets audit --catalog configs/robots.json
uv run rrp assets validate --robot fixture_arm
uv run rrp task validate tasks/support_and_insert.json
uv run rrp workbench --host 127.0.0.1 --port 8765
uv run rrp data generate --config configs/data/dev.json
uv run rrp train codec --config configs/model/codec-small.json
uv run rrp train policy --config configs/model/policy-small.json
uv run rrp evaluate --config configs/eval/development.json
uv run rrp adapt sft --config configs/adapt/sft.json
uv run rrp adapt grpo --config configs/adapt/grpo.json
uv run rrp adapt expo --config configs/adapt/expo.json
uv run rrp campaign run --config configs/campaign.json --resume
uv run rrp analyze --registry research/registry.jsonl
uv run rrp replay --episode artifacts/selected_episode.json
uv run rrp release verify --requirements docs/handoff/REQUIREMENTS_TRACEABILITY.md
uv run rrp ops stop --owned-only
```

These command names are the intended final interface, not commands already implemented by this package. The agent must create actual runnable configs and test each relevant command with `--help` plus a bounded fixture. Real configs are resolved from the supplied templates; no literal placeholder paths may survive in final reproduction instructions.

## release artifact contract

`artifacts/release/` contains a README with exact start/stop/resume commands, source and asset locks, checkpoints or local indexed paths+hashes, data manifests, primary metrics/figures, a full report, a collaborator brief, UI demo video, test/browser receipts, resource accounting, completion matrix and unresolved blocker ledger. Private keys/tokens/user data and third-party restricted assets are excluded.

Each requirement has `status`, `test_command`, `artifact_paths`, `source_revision`, and `limitations`. The gate validator rejects “complete” with missing evidence, fabricated paths or unexecuted test commands. A failing research hypothesis is compatible with research completion; an unimplemented GRPO stub is not. Workbench labels match actual controller/model execution mode.

Verify from a fresh isolated environment or documented clean worktree under resource limits. The handoff is not complete just because existing in-memory imports still work. Save final local commits, shut down owned jobs, and list any intentionally retained services with exact ownership and endpoints. Do not stop unrelated jobs.
