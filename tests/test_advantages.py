import pytest
import torch

from uav_control.advantages import generalized_advantage_estimate as gae


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hand_calculated_continuing_trajectory(dtype):
    rewards = torch.tensor([[1.0], [2.0], [3.0]], dtype=dtype, requires_grad=True)
    values = torch.tensor([[0.5], [1.0], [1.5]], dtype=dtype, requires_grad=True)
    next_values = torch.tensor([[1.0], [1.5], [2.0]], dtype=dtype, requires_grad=True)
    flags = torch.zeros(3, 1, dtype=torch.bool)
    result = gae(rewards, values, flags, flags, next_values=next_values, gamma=0.9, gae_lambda=0.8)
    # deltas = [1.4, 2.35, 3.3]; trace multiplier = 0.72.
    torch.testing.assert_close(result.advantages, rewards.new_tensor([[4.80272], [4.726], [3.3]]))
    torch.testing.assert_close(result.returns, rewards.new_tensor([[5.30272], [5.726], [4.8]]))
    assert not result.advantages.requires_grad and not result.returns.requires_grad
    assert torch.equal(rewards.detach(), rewards.new_tensor([[1], [2], [3]]))


@pytest.mark.parametrize("bootstrap,expected_timeout", [(False, 10.0), (True, 15.0)])
def test_parallel_boundaries_and_final_state_bootstrap(bootstrap, expected_timeout):
    rewards = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    values = torch.zeros_like(rewards)
    terminated = torch.tensor([[False, False], [True, False], [False, True]])
    truncated = torch.tensor([[False, True], [False, False], [False, False]])
    next_values = torch.tensor([[0.0, 9999.0], [float('nan'), 0.0], [4.0, float('nan')]])
    final_values = torch.full_like(values, float('nan'))
    final_values[0, 1] = 5.0
    result = gae(rewards, values, terminated, truncated, next_values=next_values,
                 truncation_values=final_values, bootstrap_truncations=bootstrap,
                 gamma=1.0, gae_lambda=1.0)
    # Env 0 resets after t=1; env 1 truncates after t=0 then terminates after t=2.
    # The reset observation's value 9999 must never enter the timeout target.
    expected = torch.tensor([[3.0, expected_timeout], [2.0, 50.0], [7.0, 30.0]])
    torch.testing.assert_close(result.advantages, expected)
    torch.testing.assert_close(result.returns, expected)


def test_terminal_wins_over_timeout_and_needs_no_bootstrap():
    result = gae(torch.tensor([[2.0]]), torch.tensor([[0.5]]),
                 torch.tensor([[True]]), torch.tensor([[True]]), bootstrap_truncations=True)
    assert result.advantages.item() == 1.5
    assert result.returns.item() == 2.0


def test_lambda_zero_is_one_step_td_and_gamma_zero_ignores_bootstrap():
    r, v = torch.tensor([[1.0], [2.0]]), torch.tensor([[0.5], [1.0]])
    flags = torch.zeros(2, 1, dtype=torch.bool)
    one_step = gae(r, v, flags, flags, next_values=torch.tensor([[1.0], [3.0]]),
                   gamma=0.5, gae_lambda=0.0)
    torch.testing.assert_close(one_step.advantages, torch.tensor([[1.0], [2.5]]))
    zero_discount = gae(r, v, flags, flags, gamma=0.0)
    torch.testing.assert_close(zero_discount.advantages, r - v)
    torch.testing.assert_close(zero_discount.returns, r)


def test_missing_or_nonfinite_required_bootstrap():
    r, v = torch.ones(1, 1), torch.zeros(1, 1)
    no, yes = torch.tensor([[False]]), torch.tensor([[True]])
    with pytest.raises(ValueError, match="next_values is required"):
        gae(r, v, no, no)
    with pytest.raises(ValueError, match="truncation_values is required"):
        gae(r, v, no, yes, bootstrap_truncations=True, next_values=v)
    with pytest.raises(ValueError, match="finite"):
        gae(r, v, no, no, next_values=torch.full_like(v, float('nan')))
    with pytest.raises(ValueError, match="finite"):
        gae(r, v, no, yes, bootstrap_truncations=True,
            truncation_values=torch.full_like(v, float('nan')))
    # Disabling timeout bootstrap explicitly needs neither final nor next values.
    assert gae(r, v, no, yes).returns.item() == 1.0


@pytest.mark.parametrize("field,bad_value,error", [
    ("rewards", torch.zeros(2, 1, 1), ValueError),
    ("rewards", torch.zeros(0, 1), ValueError),
    ("rewards", torch.ones(2, 1, dtype=torch.int64), TypeError),
    ("values", torch.zeros(2), ValueError),
    ("values", torch.zeros(2, 1, dtype=torch.float64), ValueError),
    ("values", torch.full((2, 1), float('inf')), ValueError),
    ("terminated", torch.zeros(2, 1), TypeError),
    ("truncated", torch.zeros(2, dtype=torch.bool), ValueError),
    ("next_values", torch.zeros(1), ValueError),
    ("truncation_values", torch.zeros(2, 1, dtype=torch.float64), ValueError),
    ("gamma", -0.1, ValueError),
    ("gamma", 1.1, ValueError),
    ("gae_lambda", float('nan'), ValueError),
    ("gae_lambda", 1.1, ValueError),
    ("bootstrap_truncations", 1, TypeError),
])
def test_invalid_inputs(field, bad_value, error):
    kwargs = dict(rewards=torch.ones(2, 1), values=torch.zeros(2, 1),
                  terminated=torch.zeros(2, 1, dtype=torch.bool),
                  truncated=torch.zeros(2, 1, dtype=torch.bool), next_values=torch.zeros(2, 1))
    kwargs[field] = bad_value
    with pytest.raises(error):
        gae(**kwargs)


def test_overflow_is_reported():
    large = torch.full((2, 1), torch.finfo(torch.float32).max)
    flags = torch.zeros(2, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="finite"):
        gae(large, torch.zeros_like(large), flags, flags, next_values=large, gamma=1.0)
