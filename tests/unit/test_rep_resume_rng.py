"""Stage-A resume restores every RNG (python batch sampler, torch CPU/CUDA), so a resumed run draws the same sequence."""
import random

import torch

from rrp.training.latent_train import _restore_rng, _rng_state


def test_rng_roundtrip_continues_sequence():
    rng = random.Random(1706); torch.manual_seed(1706)
    [rng.random() for _ in range(5)]; torch.randn(3)
    st = _rng_state(rng)
    want = ([rng.random() for _ in range(3)], torch.randn(4))
    rng2 = random.Random(0); torch.manual_seed(0)
    assert _restore_rng(st, rng2)
    got = ([rng2.random() for _ in range(3)], torch.randn(4))
    assert got[0] == want[0] and torch.equal(got[1], want[1])


def test_old_checkpoint_without_rng_is_flagged():
    assert _restore_rng({"sched": {}}, random.Random(0)) is False
