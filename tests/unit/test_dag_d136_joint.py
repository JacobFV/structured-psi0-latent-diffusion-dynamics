"""D-136 joint adaptation: update counts matched to BC SFT (SFT_STEPS). (The D-136 DAG moved to the legacy area with the closed armdiag track.)"""
from rrp.harness.pipelines.arm import SFT_STEPS
from rrp.harness.train.joint_adapt import split_steps


def test_split_steps_total_matches_budget():
    for s in SFT_STEPS.values():
        f, r = split_steps(s, "split")
        assert f + r == s and abs(f - r) <= 1
    assert split_steps(600, "joint") == (600, 600)          # one optimizer: each update touches both modules
