import math

import pytest
import torch

from uav_control.losses import clipped_policy_loss, ppo_loss, probability_ratio, value_loss


@pytest.mark.parametrize("ratio,advantage,expected_loss,expected_grad", [
    (1.5, 2.0, -2.4, 0.0),   # Positive advantage: cap beneficial increase.
    (0.5, 2.0, -1.0, -1.0),  # Positive advantage: keep harmful decrease.
    (0.5, -2.0, 1.6, 0.0),   # Negative advantage: cap beneficial decrease.
    (1.5, -2.0, 3.0, 3.0),   # Negative advantage: keep harmful increase.
    (1.0, 2.0, -2.0, -2.0),
])
def test_clipped_surrogate_signs_and_gradients(ratio, advantage, expected_loss, expected_grad):
    new = torch.tensor([math.log(ratio)], dtype=torch.float64, requires_grad=True)
    old = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    adv = torch.tensor([advantage], dtype=torch.float64, requires_grad=True)
    result = clipped_policy_loss(new, old, adv)
    assert result.loss.item() == pytest.approx(expected_loss)
    result.loss.backward()
    assert new.grad.item() == pytest.approx(expected_grad)
    assert old.grad is None and adv.grad is None
    assert not result.approx_kl.requires_grad and not result.clip_fraction.requires_grad


def test_ratio_and_diagnostics_against_scalar_calculations():
    ratios = [0.5, 1.0, 1.5]
    new = torch.tensor([math.log(r) for r in ratios], dtype=torch.float64)
    old = torch.zeros(3, dtype=torch.float64)
    torch.testing.assert_close(probability_ratio(new, old), torch.tensor(ratios, dtype=torch.float64))
    result = clipped_policy_loss(new, old, torch.ones_like(new))
    assert result.loss.item() == pytest.approx(-0.9)
    assert result.clip_fraction.item() == pytest.approx(2 / 3)
    expected_kl = sum(r - 1 - math.log(r) for r in ratios) / 3
    assert result.approx_kl.item() == pytest.approx(expected_kl)
    identical = clipped_policy_loss(old, old, torch.ones_like(old))
    assert identical.approx_kl.item() == 0
    assert identical.clip_fraction.item() == 0


def test_value_mse_prediction_clipping_and_target_detachment():
    values = torch.tensor([2.0, 0.5], dtype=torch.float64, requires_grad=True)
    returns = torch.tensor([0.0, 1.0], dtype=torch.float64, requires_grad=True)
    old = torch.tensor([1.0, 0.5], dtype=torch.float64, requires_grad=True)
    plain = value_loss(values, returns)
    assert plain.item() == pytest.approx((4 + 0.25) / 2)
    plain.backward()
    torch.testing.assert_close(values.grad, torch.tensor([2.0, -0.5], dtype=torch.float64))
    assert returns.grad is None
    values.grad = None
    clipped = value_loss(values, returns, old_values=old, clip_range=0.2)
    # Prediction 2.0 is clipped to 1.2: MSE uses 1.44, not max(1.44, 4).
    assert clipped.item() == pytest.approx((1.44 + 0.25) / 2)
    clipped.backward()
    torch.testing.assert_close(values.grad, torch.tensor([0.0, -0.5], dtype=torch.float64))
    assert old.grad is None and returns.grad is None


def test_combined_loss_coefficients_and_gradients():
    new = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    values = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    entropy = torch.tensor([2.0, 4.0], dtype=torch.float64, requires_grad=True)
    result = ppo_loss(new, torch.zeros_like(new), torch.tensor([1.0, 3.0], dtype=torch.float64),
                      values, torch.zeros_like(values), entropy=entropy,
                      value_coefficient=0.5, entropy_coefficient=0.1)
    assert result.policy.item() == -2
    assert result.value.item() == 2.5
    assert result.entropy_loss.item() == pytest.approx(-0.3)
    assert result.total.item() == pytest.approx(-1.05)
    result.total.backward()
    torch.testing.assert_close(new.grad, torch.tensor([-0.5, -1.5], dtype=torch.float64))
    torch.testing.assert_close(values.grad, torch.tensor([0.5, 1.0], dtype=torch.float64))
    torch.testing.assert_close(entropy.grad, torch.full((2,), -0.05, dtype=torch.float64))


def test_zero_coefficients_and_optional_entropy():
    x = torch.zeros(1)
    result = ppo_loss(x, x, torch.ones(1), x, x, value_coefficient=0.0)
    assert result.total.item() == -1.0
    assert result.entropy_loss.item() == 0.0
    with pytest.raises(ValueError, match="entropy is required"):
        ppo_loss(x, x, x, x, x, entropy_coefficient=0.1)


@pytest.mark.parametrize("bad", [torch.zeros(2, 1), torch.zeros(0),
                                  torch.tensor([float('nan')]), torch.tensor([float('inf')])])
def test_invalid_log_probability_shapes_and_values(bad):
    with pytest.raises(ValueError):
        probability_ratio(bad, torch.zeros_like(bad))


@pytest.mark.parametrize("difference", [1000.0, -1000.0])
def test_ratio_overflow_and_underflow_rejected(difference):
    with pytest.raises(ValueError):
        probability_ratio(torch.tensor([difference]), torch.zeros(1))


@pytest.mark.parametrize("clip", [-0.1, 1.0, float('inf')])
def test_invalid_policy_clip(clip):
    with pytest.raises(ValueError):
        clipped_policy_loss(torch.zeros(1), torch.zeros(1), torch.ones(1), clip_range=clip)


def test_invalid_shapes_dtypes_value_clipping_and_coefficients():
    x = torch.zeros(2)
    with pytest.raises(ValueError):
        probability_ratio(x, torch.zeros(1))
    with pytest.raises(ValueError):
        probability_ratio(x, x.double())
    with pytest.raises(TypeError):
        probability_ratio(x.long(), x.long())
    with pytest.raises(ValueError):
        clipped_policy_loss(x, x, torch.zeros(2, 1))
    with pytest.raises(ValueError, match="old_values is required"):
        value_loss(x, x, clip_range=0.2)
    with pytest.raises(ValueError):
        value_loss(x, torch.zeros(2, 1))
    with pytest.raises(ValueError):
        value_loss(x, x, old_values=x, clip_range=-0.1)
    with pytest.raises(ValueError):
        ppo_loss(x, x, x, torch.zeros(1), torch.zeros(1))
    with pytest.raises(ValueError):
        ppo_loss(x, x, x, x, x, entropy=torch.zeros(2, 1))
    for keyword in ('value_coefficient', 'entropy_coefficient'):
        with pytest.raises(ValueError):
            ppo_loss(x, x, x, x, x, **{keyword: -0.1})


def test_small_log_ratio_kl_and_large_value_errors():
    new = torch.tensor([1e-8], dtype=torch.float64)
    result = clipped_policy_loss(new, torch.zeros_like(new), torch.ones_like(new))
    assert result.approx_kl.item() == pytest.approx(5e-17, rel=1e-7, abs=1e-24)
    with pytest.raises(ValueError, match="finite"):
        value_loss(torch.tensor([1e30]), torch.zeros(1))


def test_double_precision_autograd_gradcheck():
    new = torch.tensor([0.02, -0.03], dtype=torch.float64, requires_grad=True)
    values = torch.tensor([0.1, -0.1], dtype=torch.float64, requires_grad=True)
    old = torch.zeros(2, dtype=torch.float64)
    advantages = torch.tensor([1.0, -2.0], dtype=torch.float64)
    returns = torch.tensor([0.5, 0.2], dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda logp, v: ppo_loss(logp, old, advantages, v, returns).total,
        (new, values),
    )
