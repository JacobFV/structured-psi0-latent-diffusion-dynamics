"""Feed-forward stepping reference: the deployed tracker adds exactly what the training env adds (same phase convention)."""
import warnings

import pytest

import numpy as np

from rrp.bodies.legged import legged_body, standalone_model
from rrp.envs.mujoco.legged_core import LeggedBinding


@pytest.mark.menagerie                      # builds the t1 Menagerie body
def test_ref_offset_shape_and_phase():
    warnings.filterwarnings("ignore")
    m, _, meta = standalone_model(legged_body("t1"), contact="v2")
    b = LeggedBinding(m, meta)
    L, R = b.pitch_idx()
    o = b.ref_offset(0.25, [0.15, 0, 0], 0.2)          # sin = 1: right leg swings
    assert np.allclose(o[R], [-0.2, 0.4, -0.2]) and np.allclose(o[L], 0)
    o = b.ref_offset(0.75, [0, 0, 0.3], 0.2)          # sin = -1: left leg swings (turn command also counts)
    assert np.allclose(o[L], [-0.2, 0.4, -0.2]) and np.allclose(o[R], 0)
    assert not b.ref_offset(0.25, [0, 0, 0], 0.2).any()
    assert not b.ref_offset(0.25, [0.5, 0, 0], 0.2).any()           # normal walking: no reference


def test_tracker_matches_env_phase():
    """LearnedTracker computes the offset with the phase BEFORE advancing, as LeggedEnv.step does."""
    import inspect
    from rrp.envs.mujoco import legged_tracker
    src = inspect.getsource(legged_tracker.LearnedTracker.act)
    assert src.index("ref_offset(self.phase") < src.index("self.phase = (self.phase")
