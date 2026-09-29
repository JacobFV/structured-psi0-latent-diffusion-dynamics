import pytest
from rrp.harness.eval.release import validate_release


def test_missing_evidence_cannot_be_called_complete():
    with pytest.raises(ValueError):
        validate_release({"status": "complete", "requirements": [
            {"id": "grpo", "status": "verified", "evidence_paths": []}]})
