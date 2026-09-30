# autonomy, coordination, restart and failure handling

## one owner for budgets and evidence

The lead agent owns the resource broker, integration branch, run registry and final claims. Use independent workers for contracts/runtime, UI, simulation/assets, model/math tests, and analysis only when dependencies are explicit. All workers share one budget and request leases. Subagent count is not permission for more GPU owners or CPU threads.

Start with at most two concurrent implementation workers plus the coordinator, and adjust down when the host's measured free CPU/memory is low. GPU jobs are serialized by default on the peer. A reviewer can inspect math/tests while another worker trains; no unleased local model server for reviews. Available external agent infrastructure is used only within the existing authorized environment, not a new paid API.

Each task ends with changed files, test commands/results, artifact paths, source version and remaining risks. The lead integrates only after its acceptance gate. Prefer small local commits and a clean task boundary over giant generated files. Use installed planning/TDD/debugging skills if available, but the user's autonomy request replaces routine approval prompts with self-review gates.

## persistent state

Keep `STATUS.md` concise and current: goal, resolved paths/peer, active leases/job IDs, completed tasks, failed tests, current experiment, next exact command, pending blockers and remaining budgets. `ops/` stores leases/telemetry/process ownership. `research/` stores immutable experiment registrations, decisions, seeds/splits, metrics indexes and analysis. `artifacts/` stores dataset/checkpoint/video manifests. Do not put secrets in any of them.

Write checkpoints atomically with checksum plus source/config/model/optimizer/RNG state and data cursor. Keep the previous known-good checkpoint until replacement verification succeeds. Register local and remote artifact paths explicitly; never assume that a file exists on both nodes. Sync only owned project artifacts and avoid overwriting newer remote results.

A restarted agent first reads status and broker state, checks whether an owned job is still running, and attaches rather than starting a duplicate. PID reuse requires start-time/cgroup/ownership validation. If the broker disappeared, workers stop at lease expiry. If SSH drops, do not assume the remote job died; inspect ownership/state before resubmitting.

## failure taxonomy and response

Infrastructure: compatibility, import, GPU/driver, renderer or transport failure. Reproduce minimally, inspect official support, isolate environment, fix with a regression check. Do not globally upgrade or guess CUDA flags until it “runs.”

Contract: unit, mask, identity, role order, stale cache, controller mismatch or privileged leakage. Stop contaminated runs; fix the boundary; label earlier results invalid rather than quietly mixing them with corrected data.

Identifiability: required conditions map to identical public tensors. Add the missing legitimate observation/history or reduce the claim. Privileged labels cannot repair indistinguishable inference inputs.

Optimization: correct interface but failure to fit/learn. Check losses, gradients, normalizations, teacher coverage and small overfit tests. Use bounded development variants; preserve all failures.

Hypothesis: implementation learns yet structured method does not beat simpler alternatives. Report the result, retain the working simpler method, and continue remaining requirements and discriminating ablations. Do not redefine success after seeing the final test.

External blocker: unavailable peer credentials, forbidden installation, inaccessible weights/assets or license. Record exact dependency and continue independent tasks. Ask only when required information cannot be safely discovered. Do not fall back to paid cloud or physical robots.

## stop and continue rules

Do not stop at a proposal, code scaffold, first video, or first negative learning result. Do not continue an unchanged failing job indefinitely. Each new expensive run needs a specific hypothesis, cost profile and remaining-budget check. Emergency memory/disk/thermal protection always overrides the experiment.

The first full campaign has finite ceilings. When reached, leave a complete resumable status and deliver actual evidence; mark incomplete mandatory execution. Do not silently renew beyond the configured ceiling. A later user extension can resume exactly, without relearning which runs occurred.

## final response from the local agent

Report where the repository/workbench live, exact launch command, model/controller scope, tests actually run, training/checkpoints actually produced, primary result with limitations, resource totals, remaining gaps, and how to resume. Include a brief collaborator-facing narrative suitable for the meeting. Do not include machine secrets or claim readiness for real hardware. Close only owned jobs and list any retained service.
