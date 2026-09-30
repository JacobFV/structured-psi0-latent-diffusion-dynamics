# campaign, analysis and delivery implementation plan

> for agentic workers: use installed subagent-driven-development or executing-plans to implement each task. write and run the failing test before implementation. the user has selected autonomous execution; review internally and proceed after evidence-backed gates.

**goal:** execute and audit the research, then deliver reproducible code and operating artifacts

**architecture:** implement the boundaries in `../docs/01_architecture.md` and `../docs/02_interfaces.md`; preserve simulator/controller/public-observation separation.

**tech stack:** Python/PyTorch/MuJoCo and React/TypeScript as appropriate, with verified local dependency locks.

**spec:** `../docs/00_scope_and_acceptance.md` and the relevant technical document linked by task.

## global constraints

The parent resource budget is aggregate ≤50% host free resources. All significant work requires a lease. No paid services, global changes, reference-repo mutation, or physical actuation. Commands below are target implementation commands, not a claim that product code ships in this handoff.

## review focus

Test malformed/empty inputs, version/provenance conflicts, variable dimensions and masks, interrupted/restarted jobs, and leakage or false completion. Extend the test examples with the full cases specified in `../docs/08_tests_and_release.md`.

---
## P20: split generator, experiment registry and campaign scheduler

**files:** create `src/rrp/evaluation/{splits,registry,campaign}.py`, `research/splits/`

**interfaces:** immutable registered runs, source/body/module split hashes, scheduler using broker leases

- [ ] write the following failing test in `tests/unit/test_registry.py` (create any explicitly imported fixture helper in the same task; it must not substitute for the implementation under test):

```python
import pytest
from rrp.evaluation.registry import ExperimentRegistry

def test_sealed_protocol_cannot_be_mutated(tmp_path):
    registry = ExperimentRegistry(tmp_path / "runs.jsonl")
    registry.register("run1", {"seed": 7, "method": "base"}, sealed=True)
    with pytest.raises(ValueError):
        registry.register("run1", {"seed": 8, "method": "base"}, sealed=True)
```

- [ ] run `uv run pytest tests/unit/test_registry.py -q`; confirm failure comes from the missing behavior, not an unrelated environment/import issue.
- [ ] implement the behavior below and its additional failure cases.

Resolve executable configs from templates. Freeze eligibility independent of policy score. Produce separate development/confirmation/breadth/online stages rather than a huge Cartesian sweep. Preflight every run with projected resource usage and stop/restart semantics. Preserve config/content hashes and append-only failures. Implement `rrp campaign run --resume` that checks existing jobs/artifacts before submission.

- [ ] rerun that test file and all upstream contract tests; perform the specified integration check under a lease. save the command/output and artifact path in the task ledger.
- [ ] self-review against the spec, have an independent reviewer inspect high-risk changes when available, then make a local scoped commit. do not claim downstream experiments passed merely because this task's unit test passed.

## P21: primary three-seed transfer experiment

**files:** create `src/rrp/evaluation/{runner,success,retention}.py`, `configs/eval/primary.json`

**interfaces:** actual per-episode success/failure/retention outputs for matched methods at registered budgets

- [ ] write the following failing test in `tests/unit/test_success_accounting.py` (create any explicitly imported fixture helper in the same task; it must not substitute for the implementation under test):

```python
from rrp.evaluation.success import summarize_attempts

def test_timeouts_and_refusals_stay_in_denominator():
    result = summarize_attempts(["success", "timeout", "refused", "failure"])
    assert result.attempted == 4
    assert result.successes == 1
    assert result.success_rate == 0.25
```

- [ ] run `uv run pytest tests/unit/test_success_accounting.py -q`; confirm failure comes from the missing behavior, not an unrelated environment/import issue.
- [ ] implement the behavior below and its additional failure cases.

Execute the registered two-task, two-heldout-condition, three-seed primary comparison with 100 episodes per cell/checkpoint. Restore source initialization independently for targets/methods. Report actual private evaluator versus public completion discrepancies. Record all attempted episodes, source retention, controller work and resources. Do not promote exploratory tuning runs to confirmation. If resources prevent a cell, mark it blocked/censored and retain exact resume command.

- [ ] rerun that test file and all upstream contract tests; perform the specified integration check under a lease. save the command/output and artifact path in the task ledger.
- [ ] self-review against the spec, have an independent reviewer inspect high-risk changes when available, then make a local scoped commit. do not claim downstream experiments passed merely because this task's unit test passed.

## P22: breadth and mechanistic counterfactuals

**files:** create `src/rrp/evaluation/{breadth,interventions,identifiability}.py`, `configs/eval/stress.json`

**interfaces:** coverage ledger plus actual input/behavior intervention results across required families

- [ ] write the following failing test in `tests/unit/test_identifiability.py` (create any explicitly imported fixture helper in the same task; it must not substitute for the implementation under test):

```python
import numpy as np
from rrp.evaluation.identifiability import distinguishability

def test_identical_inputs_are_reported_before_training():
    result = distinguishability(np.array([1.,2.]), np.array([1.,2.]),
                                different_required_action=True)
    assert result.status == "unidentifiable_under_current_input"
```

- [ ] run `uv run pytest tests/unit/test_identifiability.py -q`; confirm failure comes from the missing behavior, not an unrelated environment/import issue.
- [ ] implement the behavior below and its additional failure cases.

Run actual learned-policy evaluations on every required feasible family, retaining failed competence. Test node permutations/larger graphs, role and frame outputs, wrong/stale provenance, failure memory, contact-anchor removal, capability shifts and functional composition. Compare raw public tensors before interpretation. Stress results may be one seed but must be labeled exploratory. Asset/model loading alone never satisfies breadth execution.

- [ ] rerun that test file and all upstream contract tests; perform the specified integration check under a lease. save the command/output and artifact path in the task ledger.
- [ ] self-review against the spec, have an independent reviewer inspect high-risk changes when available, then make a local scoped commit. do not claim downstream experiments passed merely because this task's unit test passed.

## P23: latency, memory and adaptation tradeoff measurements

**files:** create `src/rrp/evaluation/{latency,profiling,frontier}.py`, `configs/eval/latency.json`

**interfaces:** synchronized latency components, resource footprint and adaptation curves with explicit units

- [ ] write the following failing test in `tests/unit/test_censoring.py` (create any explicitly imported fixture helper in the same task; it must not substitute for the implementation under test):

```python
from rrp.evaluation.frontier import first_sustained_crossing

def test_unreached_success_is_censored():
    r = first_sustained_crossing(budgets=[0,10,50], success=[0.1,0.3,0.7],
                                 threshold=0.8)
    assert r.censored and r.budget is None
```

- [ ] run `uv run pytest tests/unit/test_censoring.py -q`; confirm failure comes from the missing behavior, not an unrelated environment/import issue.
- [ ] implement the behavior below and its additional failure cases.

Profile naive/cached/factorized attention and actual VLM/context/codec/controller paths, warm/cold, multiple NFE and n/h sizes on matched hardware/load. Include graph-bias and projection/MLP cost. Execute registered SFT/GRPO/off-policy adaptation budgets and retention. Count unique environment steps versus replay/prefix reuse. Do not convert hardware memory capacity or FP4 marketing throughput into measured BF16 training performance.

- [ ] rerun that test file and all upstream contract tests; perform the specified integration check under a lease. save the command/output and artifact path in the task ledger.
- [ ] self-review against the spec, have an independent reviewer inspect high-risk changes when available, then make a local scoped commit. do not claim downstream experiments passed merely because this task's unit test passed.

## P24: analysis programs, independent audit and research report

**files:** create `src/rrp/evaluation/{statistics,plots,audit}.py`, `research/reports/`

**interfaces:** raw metrics -> deterministic tables, intervals/figures, claim map and audit receipts

- [ ] write the following failing test in `tests/unit/test_analysis.py` (create any explicitly imported fixture helper in the same task; it must not substitute for the implementation under test):

```python
from rrp.evaluation.statistics import success_counts

def test_analysis_recomputes_counts_from_episode_rows():
    rows = [{"success": True}, {"success": False}, {"success": False}]
    assert success_counts(rows) == (1, 3)
```

- [ ] run `uv run pytest tests/unit/test_analysis.py -q`; confirm failure comes from the missing behavior, not an unrelated environment/import issue.
- [ ] implement the behavior below and its additional failure cases.

Generate every headline table from raw rows. Include paired and per-seed uncertainty, censored curves and failures. Independently reconstruct a selected result and one adverse counterfactual from hashes/checkpoints. Verify input leakage/splits and account for failed runs. Write related work, scope of learned versus supplied structure, outcome, limitations and practical recommended configuration. A negative hypothesis is documented, not edited out. Preserve unfinished tests as such.

- [ ] rerun that test file and all upstream contract tests; perform the specified integration check under a lease. save the command/output and artifact path in the task ledger.
- [ ] self-review against the spec, have an independent reviewer inspect high-risk changes when available, then make a local scoped commit. do not claim downstream experiments passed merely because this task's unit test passed.

## P25: operator docs, release verification and final delivery

**files:** create `src/rrp/evaluation/release.py`, `artifacts/release/`, `README.md`, `docs/operator.md`

**interfaces:** complete proof-linked requirement matrix, working start/stop/resume workflow, report/brief/video/checkpoint index

- [ ] write the following failing test in `tests/unit/test_release_gate.py` (create any explicitly imported fixture helper in the same task; it must not substitute for the implementation under test):

```python
import pytest
from rrp.evaluation.release import validate_release

def test_missing_evidence_cannot_be_called_complete():
    with pytest.raises(ValueError):
        validate_release({"status":"complete", "requirements":[
            {"id":"grpo", "status":"verified", "evidence_paths":[]} ]})
```

- [ ] run `uv run pytest tests/unit/test_release_gate.py -q`; confirm failure comes from the missing behavior, not an unrelated environment/import issue.
- [ ] implement the behavior below and its additional failure cases.

Run the whole mandatory test matrix and one clean-environment reproduction under leases. Resolve every command/path in final docs to real existing artifacts. Produce collaborator brief, real GUI/model demo, source/asset/checkpoint/data manifests, budget totals, claims and blockers. Verify cleanup affects only owned jobs, list retained services, and write exact resumption steps if blocked/exhausted. Finish with local commits and an honest overall status; no unapproved remote publication.

- [ ] rerun that test file and all upstream contract tests; perform the specified integration check under a lease. save the command/output and artifact path in the task ledger.
- [ ] self-review against the spec, have an independent reviewer inspect high-risk changes when available, then make a local scoped commit. do not claim downstream experiments passed merely because this task's unit test passed.
