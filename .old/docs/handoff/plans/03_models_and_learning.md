# model and learning implementation plan

> for agentic workers: use installed subagent-driven-development or executing-plans to implement each task. write and run the failing test before implementation. the user has selected autonomous execution; review internally and proceed after evidence-backed gates.

**goal:** implement real transferable action learning and valid fine-tuning algorithms

**architecture:** implement the boundaries in `../docs/01_architecture.md` and `../docs/02_interfaces.md`; preserve simulator/controller/public-observation separation.

**tech stack:** Python/PyTorch/MuJoCo and React/TypeScript as appropriate, with verified local dependency locks.

**spec:** `../docs/00_scope_and_acceptance.md` and the relevant technical document linked by task.

## global constraints

The parent resource budget is aggregate ≤50% host free resources. All significant work requires a lease. No paid services, global changes, reference-repo mutation, or physical actuation. Commands below are target implementation commands, not a claim that product code ships in this handoff.

## review focus

Test malformed/empty inputs, version/provenance conflicts, variable dimensions and masks, interrupted/restarted jobs, and leakage or false completion. Extend the test examples with the full cases specified in `../docs/08_tests_and_release.md`.

---
## P13: action codec with reconstructible targets

**files:** create `src/rrp/model/{codec,command_groups}.py`, `src/rrp/learning/codec.py`

**interfaces:** native chunks + public body/state -> latent targets; latent + public body/state -> native commands

- [ ] write the following failing test in `tests/unit/test_codec.py` (create any explicitly imported fixture helper in the same task; it must not substitute for the implementation under test):

```python
import torch
from rrp.model.codec import masked_reconstruction_loss

def test_padding_does_not_train_the_codec():
    pred = torch.tensor([1., 999.], requires_grad=True)
    target = torch.tensor([2., -999.])
    mask = torch.tensor([True, False])
    loss = masked_reconstruction_loss(pred, target, mask)
    loss.backward()
    assert loss.item() == 1.
    assert pred.grad[1].item() == 0.
```

- [ ] run `uv run pytest tests/unit/test_codec.py -q`; confirm failure comes from the missing behavior, not an unrelated environment/import issue.
- [ ] implement the behavior below and its additional failure cases.

Implement graph-conditioned encoder/decoder and physical-effect loss; the decoder has no future labels or demonstrated-action input. Unit-test heterogeneous command groups and empty masks (explicit error or configured zero contribution). Overfit a tiny batch, then run held-out reconstruction and teacher-latent rollout gates. Add shuffled-latent controls. Freeze codec checkpoints for later flow/RL runs. Keep a direct-action target path.

- [ ] rerun that test file and all upstream contract tests; perform the specified integration check under a lease. save the command/output and artifact path in the task ledger.
- [ ] self-review against the spec, have an independent reviewer inspect high-risk changes when available, then make a local scoped commit. do not claim downstream experiments passed merely because this task's unit test passed.

## P14: factorized flow model and cached context

**files:** create `src/rrp/model/{flow,attention,context,cache,morphology}.py`

**interfaces:** ContextCache + noisy z + τ -> valid-coordinate velocity; `Policy.prepare/sample` -> versioned ActionChunk

- [ ] write the following failing test in `tests/unit/test_flow_math.py` (create any explicitly imported fixture helper in the same task; it must not substitute for the implementation under test):

```python
import torch
from rrp.model.flow import interpolate_target

def test_flow_endpoints_and_velocity():
    noise, data = torch.tensor([2.]), torch.tensor([5.])
    z0, velocity = interpolate_target(noise, data, tau=0.)
    z1, _ = interpolate_target(noise, data, tau=1.)
    assert torch.equal(z0, noise)
    assert torch.equal(z1, data)
    assert torch.equal(velocity, data-noise)
```

- [ ] run `uv run pytest tests/unit/test_flow_math.py -q`; confirm failure comes from the missing behavior, not an unrelated environment/import issue.
- [ ] implement the behavior below and its additional failure cases.

Implement temporal/entity factorization, packed typed cross-attention, separate graph/node/time conditioning and a dense correctness reference. Add zero-bias, mask, node-permutation, null-bank, finite-gradient and cached/uncached numerical tests. Cache keys include all source/context versions and reject stale reuse. Profile projection/MLP/bias-construction overhead, not only attention complexity. Sample a real chunk and decode it through the controller fixture.

- [ ] rerun that test file and all upstream contract tests; perform the specified integration check under a lease. save the command/output and artifact path in the task ledger.
- [ ] self-review against the spec, have an independent reviewer inspect high-risk changes when available, then make a local scoped commit. do not claim downstream experiments passed merely because this task's unit test passed.

## P15: canonical semantics, event representations and QA

**files:** create `src/rrp/model/{scene,events,roles,frames,belief,probes,qa}.py`, `src/rrp/learning/auxiliary.py`

**interfaces:** four clean typed banks and query heads tied to canonical objects/event roles; differentiable auxiliary losses

- [ ] write the following failing test in `tests/unit/test_semantics.py` (create any explicitly imported fixture helper in the same task; it must not substitute for the implementation under test):

```python
from rrp.model.frames import ContactFrameValidity

def test_lost_contact_is_not_a_certain_anchor():
    validity = ContactFrameValidity(max_age_seconds=1.0)
    result = validity.evaluate(contact_present=False, age_seconds=2.0,
                               object_motion_observed=False, slip_probability=0.9)
    assert not result.valid
    assert result.reason is not None
```

- [ ] run `uv run pytest tests/unit/test_semantics.py -q`; confirm failure comes from the missing behavior, not an unrelated environment/import issue.
- [ ] implement the behavior below and its additional failure cases.

Implement all object/relation, manipulator-task, event/delta, geometry/contact and uncertainty objectives. Create separate canonical identity versus prompt relation readouts. Use role sets/incidence rather than averaging away composition. QA frozen decoder permits gradients into slots/projection; verify with a tiny local frozen-transformer fixture plus actual pretrained integration later. Add counterfactual probe/action tests, contact tangent ambiguity, stale output frames, unknown versus false, capability-aware invariance and per-head calibration. Add explicit gradient tests from system-i semantic readouts into action-expert blocks, not only system-ii/clean-context encoders, and control interventions on those readouts. No auxiliary decoder is compulsory on the sampling path.

- [ ] rerun that test file and all upstream contract tests; perform the specified integration check under a lease. save the command/output and artifact path in the task ledger.
- [ ] self-review against the spec, have an independent reviewer inspect high-risk changes when available, then make a local scoped commit. do not claim downstream experiments passed merely because this task's unit test passed.

## P16: real VLM adapter and behavior training

**files:** create `src/rrp/model/backbone.py`, `src/rrp/learning/{behavior,trainer,checkpoint}.py`, `src/rrp/data/features.py`

**interfaces:** actual pinned pretrained image/text features, reproducible behavioral training and atomic checkpoints

- [ ] write the following failing test in `tests/unit/test_checkpoint_contract.py` (create any explicitly imported fixture helper in the same task; it must not substitute for the implementation under test):

```python
import pytest
from rrp.learning.checkpoint import require_compatible_versions

def test_codec_mismatch_is_not_silently_loaded():
    with pytest.raises(ValueError):
        require_compatible_versions(saved={"codec": "a", "robot": "r"},
                                    requested={"codec": "b", "robot": "r"})
```

- [ ] run `uv run pytest tests/unit/test_checkpoint_contract.py -q`; confirm failure comes from the missing behavior, not an unrelated environment/import issue.
- [ ] implement the behavior below and its additional failure cases.

Audit/load actual ψ₀ system-ii weights, tokenizer and image processor in the peer environment. Keep meaningful visual hidden states and direct morphology inputs. Record fallback differences if genuinely necessary. Implement behavior/auxiliary training, reproducible batches, checkpoints/RNG/cursors and bounded resource-aware runs. Run tiny fit, source validation and actual VLM-backed learned-policy control, then medium/target-scale profiling. A feature stub or imported class without training does not pass.

- [ ] rerun that test file and all upstream contract tests; perform the specified integration check under a lease. save the command/output and artifact path in the task ledger.
- [ ] self-review against the spec, have an independent reviewer inspect high-risk changes when available, then make a local scoped commit. do not claim downstream experiments passed merely because this task's unit test passed.

## P17: compatible swaps and supervised adaptation

**files:** create `src/rrp/learning/{swap_alignment,sft}.py`, `src/rrp/evaluation/adaptation.py`

**interfaces:** nested target-demo budgets; separately frozen source initializations; role/effect alignment without erasing capabilities

- [ ] write the following failing test in `tests/unit/test_adaptation.py` (create any explicitly imported fixture helper in the same task; it must not substitute for the implementation under test):

```python
from rrp.evaluation.adaptation import nested_budget_indices

def test_adaptation_budgets_are_nested():
    selected = nested_budget_indices(n=120, budgets=[5,20,100], seed=3)
    assert set(selected[5]) <= set(selected[20]) <= set(selected[100])
    assert len(selected[100]) == 100
```

- [ ] run `uv run pytest tests/unit/test_adaptation.py -q`; confirm failure comes from the missing behavior, not an unrelated environment/import issue.
- [ ] implement the behavior below and its additional failure cases.

Implement source-only normalization, independent target/method reset, selected-module fine-tuning and source-task retention. Align semantic effects/roles across compatible swaps while retaining morphology/capabilities for control. Test infeasible swaps separately. Log every demo control transition and optimizer update. Measure zero-shot before target adaptation, and distinguish target-controller calibration cost from policy updates.

- [ ] rerun that test file and all upstream contract tests; perform the specified integration check under a lease. save the command/output and artifact path in the task ledger.
- [ ] self-review against the spec, have an independent reviewer inspect high-risk changes when available, then make a local scoped commit. do not claim downstream experiments passed merely because this task's unit test passed.

## P18: flow-SDE GRPO and full-state branching

**files:** create `src/rrp/learning/{flow_sde,grpo,branching}.py`, `research/methods/grpo-derivation.md`

**interfaces:** recorded stochastic paths + old likelihoods + group returns -> mathematically justified clipped policy updates

- [ ] write the following failing test in `tests/unit/test_grpo_density.py` (create any explicitly imported fixture helper in the same task; it must not substitute for the implementation under test):

```python
import math
import torch
from rrp.learning.flow_sde import gaussian_transition_log_prob

def test_gaussian_log_density_is_not_flow_mse():
    x = torch.tensor([[0.]], dtype=torch.float64)
    lp = gaussian_transition_log_prob(x, mean=x, std=torch.ones_like(x),
                                      valid=torch.ones_like(x, dtype=torch.bool))
    assert torch.allclose(lp, torch.tensor([-0.5*math.log(2*math.pi)],
                                          dtype=torch.float64))
```

- [ ] run `uv run pytest tests/unit/test_grpo_density.py -q`; confirm failure comes from the missing behavior, not an unrelated environment/import issue.
- [ ] implement the behavior below and its additional failure cases.

Follow the learning document's full derivation/tests, including endpoint treatment, exact masked dimension sums, identity ratios, old-policy versions and shared-prefix gradient exclusions. Branch complete simulator/controller/runtime/belief state; count prefix work once and gradient reuse separately. Codec/controller fixed during updates. Run synthetic scalar-control learning, then actual robot suffix adaptation with measured outcomes. Do not relabel deterministic action MSE as a log-probability or claim GRPO helps before results.

- [ ] rerun that test file and all upstream contract tests; perform the specified integration check under a lease. save the command/output and artifact path in the task ledger.
- [ ] self-review against the spec, have an independent reviewer inspect high-risk changes when available, then make a local scoped commit. do not claim downstream experiments passed merely because this task's unit test passed.

## P19: off-policy edit/base comparison

**files:** create `src/rrp/learning/{expo,replay_buffer,critics}.py`, `research/methods/expo-port.md`

**interfaces:** versioned replay records and verified source-derived edit/base-policy objectives

- [ ] write the following failing test in `tests/unit/test_replay.py` (create any explicitly imported fixture helper in the same task; it must not substitute for the implementation under test):

```python
import pytest
from rrp.learning.replay_buffer import ReplayBuffer, ReplayRecord

def test_incompatible_controller_replay_is_rejected():
    replay = ReplayBuffer(controller_version="c1", capacity=32)
    with pytest.raises(ValueError):
        replay.append(ReplayRecord.fixture(controller_version="c2"))
```

- [ ] run `uv run pytest tests/unit/test_replay.py -q`; confirm failure comes from the missing behavior, not an unrelated environment/import issue.
- [ ] implement the behavior below and its additional failure cases.

Read and document exact EXPO-FT objectives and departures before implementation. Store base proposal/edit/executed command and versions separately. Include critic/target/update math tests, transition/reset handling, replay limits and base-policy update accounting. Implement an actual short learning run; compare equal new experience and separately reported compute with GRPO. A frozen-base residual alone is a separately named baseline, not the full referenced method.

- [ ] rerun that test file and all upstream contract tests; perform the specified integration check under a lease. save the command/output and artifact path in the task ledger.
- [ ] self-review against the spec, have an independent reviewer inspect high-risk changes when available, then make a local scoped commit. do not claim downstream experiments passed merely because this task's unit test passed.
