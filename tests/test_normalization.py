import math

import pytest
import torch

from uav_control.normalization import RunningNormalizer, normalize_advantages


def test_population_moments_with_prior_and_split_batches():
    normalizer = RunningNormalizer(1, clip=None)
    normalizer.update(torch.tensor([[1.0], [3.0]]))
    assert normalizer.count.item() == 3
    assert normalizer.mean.item() == pytest.approx(4 / 3)
    assert normalizer.variance.item() == pytest.approx(17 / 9)
    split = RunningNormalizer(1, clip=None)
    split.update(torch.tensor([[1.0]]))
    split.update(torch.tensor([[3.0]]))
    torch.testing.assert_close(split.mean, normalizer.mean)
    torch.testing.assert_close(split.variance, normalizer.variance)


def test_transforms_frozen_statistics_and_input_gradients():
    normalizer = RunningNormalizer(1, clip=None, initial_count=0)
    normalizer.update(torch.tensor([[1.0], [3.0]], dtype=torch.float64))
    x = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64, requires_grad=True)
    before = {name: tensor.clone() for name, tensor in normalizer.state_dict().items()}
    normalizer.train()
    normalized = normalizer.normalize(x)
    torch.testing.assert_close(normalized, torch.tensor([[-1], [0], [1]], dtype=torch.float64), atol=2e-8, rtol=0)
    torch.testing.assert_close(normalizer.denormalize(normalized), x)
    normalized.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x) / (1 + 1e-8))
    for name, tensor in before.items():
        assert torch.equal(normalizer.state_dict()[name], tensor)


def test_singletons_clipping_and_large_offsets():
    scaler = RunningNormalizer(2, initial_count=0)
    scaler.update(torch.tensor([[3.0, 4.0]]))
    torch.testing.assert_close(scaler.normalize(torch.tensor([[3.0, 4.0]])), torch.zeros(1, 2))
    torch.testing.assert_close(scaler.normalize(torch.tensor([[4.0, 3.0]])), torch.tensor([[5.0, -5.0]]))
    stable = RunningNormalizer(1, initial_count=0, clip=None)
    stable.update(torch.tensor([[1e12 - 1], [1e12 + 1]], dtype=torch.float64))
    assert stable.mean.item() == 1e12 and stable.variance.item() == 1
    assert stable.normalize(torch.tensor([[1e12]], dtype=torch.float64)).item() == 0
    # Denormalization deliberately does not clip predictions to +/-5.
    assert stable.denormalize(torch.tensor([[10.0]], dtype=torch.float64)).item() > 1e12 + 9


def test_global_advantage_standardization():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64, requires_grad=True)
    expected = (x.detach() - 2.5) / (math.sqrt(5 / 3) + 1e-8)
    torch.testing.assert_close(normalize_advantages(x), expected)
    assert not normalize_advantages(x).requires_grad
    assert normalize_advantages(torch.ones(1)).item() == 0
    assert torch.equal(normalize_advantages(torch.ones(4)), torch.zeros(4))


def test_invalid_inputs_do_not_update_statistics():
    scaler = RunningNormalizer(2)
    for bad in (torch.ones(2), torch.ones(2, 3), torch.zeros(0, 2), torch.full((2, 2), float('nan'))):
        with pytest.raises(ValueError):
            scaler.update(bad)
    assert scaler.count.item() == 1
    with pytest.raises(ValueError):
        RunningNormalizer(2, epsilon=0)
    with pytest.raises(ValueError):
        scaler.float().normalize(torch.ones(1, 2))
