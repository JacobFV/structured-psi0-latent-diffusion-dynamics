import torch
from rrp.model.flow import interpolate_target, masked_mse
from rrp.model.codec import masked_reconstruction_loss
import pytest


def test_flow_endpoints_and_velocity():
    noise, data = torch.tensor([2.]), torch.tensor([5.])
    z0, velocity = interpolate_target(noise, data, tau=0.)
    z1, _ = interpolate_target(noise, data, tau=1.)
    assert torch.equal(z0, noise)
    assert torch.equal(z1, data)
    assert torch.equal(velocity, data - noise)


def test_padding_does_not_train_the_codec():
    pred = torch.tensor([1., 999.], requires_grad=True)
    target = torch.tensor([2., -999.])
    mask = torch.tensor([True, False])
    loss = masked_reconstruction_loss(pred, target, mask)
    loss.backward()
    assert loss.item() == 1.
    assert pred.grad[1].item() == 0.


def test_empty_mask_is_explicit():
    with pytest.raises(ValueError):
        masked_mse(torch.zeros(3), torch.zeros(3), torch.zeros(3, dtype=torch.bool))
    assert masked_reconstruction_loss(torch.zeros(3), torch.ones(3), torch.zeros(3, dtype=torch.bool),
                                      empty="zero").item() == 0.0
