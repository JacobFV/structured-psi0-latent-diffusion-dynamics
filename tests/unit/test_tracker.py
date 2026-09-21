import numpy as np
from rrp.sim.sensors import ObjectTracker, DetectorConfig


def test_tracker_follows_moving_object_and_grows_uncertainty_when_unseen():
    tr = ObjectTracker(["red cube"], DetectorConfig())
    for k in range(50):
        tr.update(k * 0.05, {0: np.array([0.4, 0.0, 0.02])})
    for k in range(20):   # object carried 20 cm while visible
        tr.update(2.5 + k * 0.05, {0: np.array([0.4 + 0.01 * (k + 1), 0.0, 0.02])})
    assert abs(tr.tracks[0].mean[0] - 0.6) < 0.01
    v0 = max(tr.tracks[0].var)
    for k in range(10):
        tr.update(4 + k * 0.05, {0: None})
    assert max(tr.tracks[0].var) > v0 and not tr.tracks[0].visible
    d = tr.descriptors(5.0)[0]
    assert d.visible is False and d.position_estimate is not None   # unknown != absent: belief retained
