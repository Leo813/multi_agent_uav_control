"""Explicit running population statistics, with no implicit train/eval updates."""

import torch
from torch import nn

from ._validation import finite, floating, scalar


class RunningNormalizer(nn.Module):
    """Float64 moments over [..., features]; transformations retain input dtype.

    Initial count=1, mean=0, variance=1 is a configurable prior. Statistics only
    change through update(), never through normalize(), denormalize(), or train().
    Keep statistics in float64 when moving this module between devices.
    """

    def __init__(self, features: int, *, device: torch.device | str = "cpu",
                 epsilon: float = 1e-8, clip: float | None = 5.0, initial_count: float = 1.0):
        super().__init__()
        if type(features) is not int or features <= 0:
            raise ValueError("features must be a positive integer")
        scalar(epsilon, "epsilon", minimum=0)
        scalar(initial_count, "initial_count", minimum=0)
        if epsilon == 0:
            raise ValueError("epsilon must be positive")
        if clip is not None:
            scalar(clip, "clip", minimum=0)
            if clip == 0:
                raise ValueError("clip must be positive or None")
        self.features, self.epsilon, self.clip = features, epsilon, clip
        self.register_buffer("mean", torch.zeros(features, dtype=torch.float64, device=device))
        self.register_buffer("variance", torch.ones(features, dtype=torch.float64, device=device))
        self.register_buffer("count", torch.tensor(float(initial_count), dtype=torch.float64, device=device))

    def _check(self, data: torch.Tensor) -> None:
        floating(data, "normalizer data")
        if data.ndim < 2 or data.shape[-1] != self.features:
            raise ValueError("Normalizer input must have shape [..., features] with a sample axis")
        if data.device != self.mean.device:
            raise ValueError("Normalizer input and statistics must share a device")
        if any(buffer.dtype != torch.float64 for buffer in self.buffers()):
            raise ValueError("Normalizer statistics must remain float64")

    @torch.no_grad()
    def update(self, data: torch.Tensor) -> None:
        self._check(data)
        samples = data.detach().reshape(-1, self.features).double()
        batch_variance, batch_mean = torch.var_mean(samples, dim=0, correction=0)
        n = samples.shape[0]
        total = self.count + n
        offset = batch_mean - self.mean
        new_mean = self.mean + offset * (n / total)
        new_variance = (self.count * self.variance + n * batch_variance
                        + offset.square() * self.count * n / total) / total
        finite(new_mean, "running mean")
        finite(new_variance, "running variance")
        self.mean.copy_(new_mean)
        self.variance.copy_(new_variance.clamp_min(0))
        self.count.copy_(total)

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        self._check(data)
        result = (data.double() - self.mean) / (self.variance.sqrt() + self.epsilon)
        if self.clip is not None:
            result = result.clamp(-self.clip, self.clip)
        result = result.to(data.dtype)
        finite(result, "normalized data")
        return result

    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """Unclipped inverse, using the same epsilon as normalize()."""
        self._check(data)
        result = (data.double() * (self.variance.sqrt() + self.epsilon) + self.mean).to(data.dtype)
        finite(result, "denormalized data")
        return result


def normalize_advantages(advantages: torch.Tensor, *, epsilon: float = 1e-8) -> torch.Tensor:
    """Global sample-std normalization; a singleton/constant batch returns zeros."""
    floating(advantages, "advantages")
    scalar(epsilon, "epsilon", minimum=0)
    if epsilon == 0:
        raise ValueError("epsilon must be positive")
    targets = advantages.detach()
    if targets.numel() == 1:
        return torch.zeros_like(targets)
    result = (targets - targets.mean()) / (targets.std(correction=1) + epsilon)
    finite(result, "normalized advantages")
    return result
