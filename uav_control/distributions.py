"""Unsquashed diagonal Gaussian policies; no environment action transforms.

Means/actions have shape [..., A], with A > 0. Log standard deviations have
shape [A] or exactly the mean shape. Log probabilities sum the action axis and
have shape [...]. Inputs share a device and float32/float64 dtype.
"""

import math
from typing import Literal

import torch
from torch.distributions import Independent, Normal

from ._validation import finite, floating, matching, scalar


def diagonal_gaussian(mean: torch.Tensor, log_std: torch.Tensor, *,
                      min_log_std: float = -20.0,
                      max_log_std: float = 2.0) -> Independent:
    """Construct a distribution, bounding log_std before exponentiation.

    Only per-action or full-shape log_std is accepted; accidental broadcasting
    across batch axes is rejected. Custom bounds must keep variance representable.
    """
    floating(mean, "mean")
    floating(log_std, "log_std")
    if mean.ndim < 1:
        raise ValueError("mean must have an action axis")
    if log_std.shape not in (mean.shape, torch.Size([mean.shape[-1]])):
        raise ValueError("log_std must have shape [A] or match mean exactly")
    if mean.device != log_std.device or mean.dtype != log_std.dtype:
        raise ValueError("mean and log_std must share dtype and device")
    scalar(min_log_std, "min_log_std")
    scalar(max_log_std, "max_log_std")
    limits = torch.finfo(mean.dtype)
    if not (0.5 * math.log(limits.tiny) <= min_log_std <= max_log_std
            <= 0.5 * math.log(limits.max) - 1.0):
        raise ValueError("log_std bounds must be ordered and keep variance representable")
    std = log_std.clamp(min_log_std, max_log_std).exp()
    return Independent(Normal(mean, std), 1)


def sample_actions(mean: torch.Tensor, log_std: torch.Tensor, *,
                   deterministic: bool = False, min_log_std: float = -20.0,
                   max_log_std: float = 2.0) -> torch.Tensor:
    """Return detached policy actions, or a detached copy of the mean.

    Samples use PyTorch's device RNG (torch.manual_seed controls CPU tests).
    No clipping or squashing is performed. Store these raw actions for PPO;
    pass a separate tensor to any later environment transform.
    """
    if not isinstance(deterministic, bool):
        raise TypeError("deterministic must be bool")
    distribution = diagonal_gaussian(mean, log_std, min_log_std=min_log_std,
                                     max_log_std=max_log_std)
    actions = mean.detach().clone() if deterministic else distribution.sample()
    finite(actions, "sampled actions")
    return actions


def gaussian_log_prob(actions: torch.Tensor, mean: torch.Tensor,
                      log_std: torch.Tensor, *, min_log_std: float = -20.0,
                      max_log_std: float = 2.0) -> torch.Tensor:
    """Joint log density of fixed, raw actions; gradients go to policy parameters."""
    distribution = diagonal_gaussian(mean, log_std, min_log_std=min_log_std,
                                     max_log_std=max_log_std)
    matching(actions, mean, "actions")
    result = distribution.log_prob(actions.detach())
    finite(result, "log probabilities (action/scale combination may be too extreme)")
    return result


def gaussian_entropy(mean: torch.Tensor, log_std: torch.Tensor, *,
                     reduction: Literal["sum", "mean"] = "sum",
                     min_log_std: float = -20.0,
                     max_log_std: float = 2.0) -> torch.Tensor:
    """Entropy per batch entry: joint sum, or per-action mean for skrl scaling.

    There is no reduction over batch axes. A later mean over the batch with
    reduction='mean' matches skrl 1.4.2's Gaussian entropy regularizer.
    """
    if reduction not in ("sum", "mean"):
        raise ValueError("reduction must be 'sum' or 'mean'")
    distribution = diagonal_gaussian(mean, log_std, min_log_std=min_log_std,
                                     max_log_std=max_log_std)
    result = distribution.entropy()
    if reduction == "mean":
        result = result / mean.shape[-1]
    finite(result, "entropy")
    return result
