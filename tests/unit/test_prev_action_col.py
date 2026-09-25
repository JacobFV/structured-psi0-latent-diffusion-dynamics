"""Bug B-1 guard: the prev-action node column must be where PREV_ACTION_COL says, and the load-time zeroing must
hit exactly that column on node rows (a silent mismatch here invalidates every closed-loop result)."""
import numpy as np

from rrp.learning.packed import PREV_ACTION_COL


def test_prev_action_col_matches_featurizer_layout():
    from rrp.morphology.catalog import workbench_robots
    from rrp.sim.scenario import BUILDERS
    from rrp.sim.native import Session
    from rrp.data.collect import featurizer_for
    s = Session(BUILDERS["pick_place"](workbench_robots()["panda_pg2"](), 3_000_000, n_distractors=0), seed=3_000_000)
    f = featurizer_for(s)
    assert f.static_dim + 2 == PREV_ACTION_COL
    pi = f(s.observe())
    assert np.all(pi.act_node_feats[:, PREV_ACTION_COL] == 0)      # deployment writes 0 (D-021)
