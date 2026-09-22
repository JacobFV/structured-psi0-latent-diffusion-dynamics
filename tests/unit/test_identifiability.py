import numpy as np
from rrp.evaluation.identifiability import distinguishability


def test_identical_inputs_are_reported_before_training():
    result = distinguishability(np.array([1., 2.]), np.array([1., 2.]), different_required_action=True)
    assert result.status == "unidentifiable_under_current_input"
