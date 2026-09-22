import math

import pytest
import torch

from rrp.learning.replay_buffer import ReplayBuffer, ReplayRecord
from rrp.learning.critics import squashed_log_prob, td_target, min_of_random_pair


def test_incompatible_controller_replay_is_rejected():
    replay = ReplayBuffer(controller_version="c1", capacity=32)
    with pytest.raises(ValueError):
        replay.append(ReplayRecord.fixture(controller_version="c2"))


def test_capacity_and_robot_version():
    replay = ReplayBuffer(controller_version="c1", capacity=2, robot_spec_hash="r1")
    for _ in range(3):
        replay.append(ReplayRecord.fixture())
    assert len(replay) == 2 and replay.evicted == 1
    with pytest.raises(ValueError):
        replay.append(ReplayRecord.fixture(robot_spec_hash="r2"))


def test_critic_target_and_edit_density_math():
    y = td_target(torch.tensor([0.5, 1.0]), torch.tensor([3.0, 8.0]), torch.tensor([0.0, 1.0]), 0.9,
                  torch.tensor([2.0, 5.0]))
    assert y.tolist() == pytest.approx([0.5 + 0.9 ** 3 * 2.0, 1.0])       # terminal: no bootstrap
    q = torch.tensor([[1.0, 5.0], [2.0, 3.0]])
    assert min_of_random_pair(q).tolist() == [1.0, 3.0]
    # tanh-Gaussian density integrates to 1 (1-D numeric check on y in (-1, 1))
    mu, ls = torch.tensor([0.3], dtype=torch.float64), torch.tensor([math.log(0.8)], dtype=torch.float64)
    ys = torch.linspace(-0.9999, 0.9999, 200001, dtype=torch.float64)
    u = torch.atanh(ys)[:, None]
    dens = squashed_log_prob(u, mu, ls).exp()
    assert torch.trapezoid(dens, ys).item() == pytest.approx(1.0, abs=2e-3)
