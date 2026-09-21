import pytest
from rrp.evaluation.registry import ExperimentRegistry


def test_sealed_protocol_cannot_be_mutated(tmp_path):
    registry = ExperimentRegistry(tmp_path / "runs.jsonl")
    registry.register("run1", {"seed": 7, "method": "base"}, sealed=True)
    with pytest.raises(ValueError):
        registry.register("run1", {"seed": 8, "method": "base"}, sealed=True)


def test_idempotent_and_append_only_states(tmp_path):
    r = ExperimentRegistry(tmp_path / "runs.jsonl")
    a = r.register("x", {"a": 1})
    assert r.register("x", {"a": 1}) == a
    r.update("x", "running")
    r.update("x", "failed", reason="oom")
    assert r.state("x") == "failed" and len(r.rows()) == 3
    with pytest.raises(KeyError):
        r.update("nope", "running")
