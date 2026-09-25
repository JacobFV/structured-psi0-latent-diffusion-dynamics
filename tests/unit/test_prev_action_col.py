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



def test_load_time_zeroing_hits_prev_action_col_not_col2():
    """D-045: the D-021 load-time fix must zero PREV_ACTION_COL on node rows (it used to zero static column 2)."""
    from rrp.learning.data import zero_prev_action_input
    from rrp.morphology.catalog import workbench_robots
    from rrp.sim.scenario import BUILDERS
    from rrp.sim.native import Session
    from rrp.data.collect import featurizer_for
    s = Session(BUILDERS["pick_place"](workbench_robots()["panda_pg2"](), 3_000_000, n_distractors=0), seed=3_000_000)
    pi = featurizer_for(s)(s.observe())
    n = pi.act_node_feats.shape[0]
    pi.act_node_feats = pi.act_node_feats.copy(); pi.act_node_feats[:, PREV_ACTION_COL] = 0.7   # teacher prev cmd
    pi.tokens = dict(pi.tokens); pi.tokens["morph"] = pi.tokens["morph"].copy()
    pi.tokens["morph"][:n, PREV_ACTION_COL] = 0.7
    before = pi.act_node_feats.copy(), pi.tokens["morph"].copy()
    got = zero_prev_action_input(pi)
    assert np.all(got.act_node_feats[:, PREV_ACTION_COL] == 0) and np.all(got.tokens["morph"][:n, PREV_ACTION_COL] == 0)
    others = [c for c in range(before[0].shape[1]) if c != PREV_ACTION_COL]
    assert np.array_equal(got.act_node_feats[:, others], before[0][:, others])          # incl. static column 2
    assert np.array_equal(got.tokens["morph"][n:], before[1][n:])                        # non-node morph rows untouched
    assert np.all(pi.act_node_feats[:, PREV_ACTION_COL] == 0.7)                          # input not mutated
