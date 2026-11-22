import math

import pytest
import torch

from uav_control.distributions import (
    diagonal_gaussian, gaussian_entropy, gaussian_log_prob, sample_actions,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_log_density_and_entropy_from_scalar_formula(dtype):
    mean = torch.tensor([[0.0, 1.0], [2.0, -1.0]], dtype=dtype)
    log_std = torch.tensor([0.0, math.log(2.0)], dtype=dtype)
    actions = torch.tensor([[1.0, 3.0], [2.0, -1.0]], dtype=dtype)
    # Independent scalar normal densities: squared standardized errors [1,1], [0,0].
    constant = math.log(2.0 * math.pi) + math.log(2.0)
    expected = torch.tensor([-constant - 1.0, -constant], dtype=dtype)
    torch.testing.assert_close(gaussian_log_prob(actions, mean, log_std), expected)
    joint_entropy = 1.0 + constant
    torch.testing.assert_close(gaussian_entropy(mean, log_std), mean.new_full((2,), joint_entropy))
    torch.testing.assert_close(gaussian_entropy(mean, log_std, reduction="mean"),
                               mean.new_full((2,), joint_entropy / 2))


@pytest.mark.parametrize("shape", [(1,), (1, 1), (7, 4), (2, 3, 5)])
@pytest.mark.parametrize("full_log_std", [False, True])
def test_sampling_shapes_repeatability_and_raw_action_ownership(shape, full_log_std):
    mean = torch.full(shape, 10.0, requires_grad=True)
    log_std = torch.zeros(shape if full_log_std else (shape[-1],), requires_grad=True)
    torch.manual_seed(71)
    first = sample_actions(mean, log_std)
    torch.manual_seed(71)
    second = sample_actions(mean, log_std)
    assert first.shape == mean.shape
    assert not first.requires_grad
    assert torch.equal(first, second)
    assert (first > 1).all()  # No environment clipping to [-1,1].
    assert gaussian_log_prob(first, mean, log_std).shape == shape[:-1]
    deterministic = sample_actions(mean, log_std, deterministic=True)
    assert torch.equal(deterministic, mean)
    assert not deterministic.requires_grad
    deterministic.zero_()
    assert (mean == 10).all()  # Returning means must not alias model outputs.


def test_log_std_clamps_before_exp_and_preserves_gradients():
    mean = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
    log_std = torch.tensor([-1000.0, 0.0, 1000.0], dtype=torch.float64, requires_grad=True)
    distribution = diagonal_gaussian(mean, log_std)
    torch.testing.assert_close(distribution.base_dist.scale,
                               torch.tensor([[math.exp(-20), 1, math.exp(2)]], dtype=torch.float64))
    logp = gaussian_log_prob(mean.detach(), mean, log_std)
    assert torch.isfinite(logp).all()
    logp.sum().backward()
    torch.testing.assert_close(log_std.grad, torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64))


def test_policy_gradients_use_fixed_actions():
    mean = torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)
    log_std = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    actions = torch.tensor([[1.0, 2.0]], dtype=torch.float64, requires_grad=True)
    gaussian_log_prob(actions, mean, log_std).sum().backward()
    torch.testing.assert_close(mean.grad, torch.tensor([[1.0, 2.0]], dtype=torch.float64))
    torch.testing.assert_close(log_std.grad, torch.tensor([0.0, 3.0], dtype=torch.float64))
    assert actions.grad is None
    log_std.grad = None
    gaussian_entropy(mean, log_std).sum().backward()
    torch.testing.assert_close(log_std.grad, torch.ones(2, dtype=torch.float64))


@pytest.mark.parametrize("mean,log_std,error", [
    (torch.tensor(0.0), torch.zeros(1), ValueError),
    (torch.zeros(0, 2), torch.zeros(2), ValueError),
    (torch.zeros(2, 3), torch.zeros(2, 1), ValueError),
    (torch.zeros(2, 3), torch.zeros(1, 3), ValueError),
    (torch.zeros(2, 3), torch.zeros(3, dtype=torch.float64), ValueError),
    (torch.zeros(2, 3, dtype=torch.int64), torch.zeros(3), TypeError),
    (torch.zeros(2, 3, dtype=torch.float16), torch.zeros(3, dtype=torch.float16), TypeError),
    (torch.tensor([[float('nan')]]), torch.zeros(1), ValueError),
    (torch.zeros(1, 1), torch.tensor([float('inf')]), ValueError),
])
def test_invalid_parameters(mean, log_std, error):
    with pytest.raises(error):
        diagonal_gaussian(mean, log_std)


@pytest.mark.parametrize("bounds", [(-1000, 2), (-20, 1000), (2, -20), (float('nan'), 2)])
def test_unsafe_or_unordered_bounds(bounds):
    with pytest.raises(ValueError):
        diagonal_gaussian(torch.zeros(2, 3), torch.zeros(3),
                          min_log_std=bounds[0], max_log_std=bounds[1])


def test_invalid_actions_and_entropy_reduction():
    mean, log_std = torch.zeros(2, 3), torch.zeros(3)
    with pytest.raises(ValueError):
        gaussian_log_prob(torch.zeros(2, 1), mean, log_std)
    with pytest.raises(ValueError):
        gaussian_log_prob(torch.full_like(mean, float('inf')), mean, log_std)
    with pytest.raises(ValueError):
        gaussian_log_prob(torch.full_like(mean, 1e30), mean, log_std)
    with pytest.raises(ValueError):
        gaussian_entropy(mean, log_std, reduction="invalid")
    with pytest.raises(TypeError):
        sample_actions(mean, log_std, deterministic=1)
