"""Legged trainers' resume state covers numpy batch sampling, torch CPU and (when present) torch CUDA RNGs (W8)."""
import numpy as np
import torch

from rrp.training.legged_latent_train import restore_rng, rng_state


def test_legged_rng_roundtrip_continues_sequence():
    rng = np.random.default_rng(7); torch.manual_seed(7)
    rng.integers(0, 100, 5); torch.randn(3)
    if torch.cuda.is_available():
        torch.randn(3, device="cuda")
    st = rng_state(rng)
    want = (rng.integers(0, 1000, 4).tolist(), torch.randn(4),
            torch.randn(4, device="cuda").cpu() if torch.cuda.is_available() else None)
    rng2 = np.random.default_rng(0); torch.manual_seed(0)
    assert restore_rng(st, rng2)
    got = (rng2.integers(0, 1000, 4).tolist(), torch.randn(4),
           torch.randn(4, device="cuda").cpu() if torch.cuda.is_available() else None)
    assert got[0] == want[0] and torch.equal(got[1], want[1])
    if torch.cuda.is_available():
        assert torch.equal(got[2], want[2])


def test_checkpoint_without_cuda_state_is_flagged_when_cuda_present():
    rng = np.random.default_rng(0)
    st = dict(rng=rng.bit_generator.state, torch_rng=torch.get_rng_state())
    assert restore_rng(st, np.random.default_rng(1)) is (not torch.cuda.is_available())
