"""PPO objectives to minimize, independent of networks and environment APIs.

All sample tensors are [B], with B > 0, on one device and float32/float64 dtype.
Old-policy statistics, advantages, returns, and old values are fixed targets.
Functions never normalize advantages or values and never update parameters.
"""

from typing import NamedTuple

import torch

from ._validation import finite, floating, matching, scalar


class PolicyLoss(NamedTuple):
    loss: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor


class PPOLoss(NamedTuple):
    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    entropy_loss: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor


def probability_ratio(new_log_prob: torch.Tensor, old_log_prob: torch.Tensor) -> torch.Tensor:
    """exp(new - old.detach()), with no silent clamping of likelihood ratios.

    Nonfinite inputs, overflow, and underflow to zero are rejected. Policy clipping
    belongs to the surrogate objective, not to this density ratio.
    """
    floating(new_log_prob, "new_log_prob", ndim=1)
    matching(old_log_prob, new_log_prob, "old_log_prob")
    ratio = (new_log_prob - old_log_prob.detach()).exp()
    finite(ratio, "probability ratio (log probability difference may overflow)")
    if (ratio == 0).any():
        raise ValueError("probability ratio underflowed to zero")
    return ratio


def clipped_policy_loss(new_log_prob: torch.Tensor, old_log_prob: torch.Tensor,
                        advantages: torch.Tensor, *, clip_range: float = 0.2) -> PolicyLoss:
    """Negative mean clipped surrogate, plus detached KL and clip diagnostics.

    approx_kl = mean(exp(d) - 1 - d), d = new_log_prob - old_log_prob.
    clip_fraction counts ratios outside [1-clip_range, 1+clip_range], regardless
    of advantage sign. Diagnostics do not impose early stopping or schedules.
    """
    scalar(clip_range, "clip_range", minimum=0.0)
    if clip_range >= 1:
        raise ValueError("clip_range must be less than 1")
    ratio = probability_ratio(new_log_prob, old_log_prob)
    matching(advantages, new_log_prob, "advantages")
    targets = advantages.detach()
    clipped = ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
    loss = -torch.minimum(ratio * targets, clipped * targets).mean()
    finite(loss, "policy loss")
    with torch.no_grad():
        difference = new_log_prob - old_log_prob
        approx_kl = (torch.expm1(difference) - difference).mean()
        outside = (ratio < 1.0 - clip_range) | (ratio > 1.0 + clip_range)
        clip_fraction = outside.to(ratio.dtype).mean()
        finite(approx_kl, "approximate KL")
    return PolicyLoss(loss, approx_kl, clip_fraction)


def value_loss(values: torch.Tensor, returns: torch.Tensor, *,
               old_values: torch.Tensor | None = None,
               clip_range: float | None = None) -> torch.Tensor:
    """Mean squared error, optionally clipping predictions around old_values.

    Clipping follows skrl 1.4.2's prediction-clipping convention. This is NOT the
    maximum of clipped and unclipped squared errors. There is no factor of 1/2.
    """
    floating(values, "values", ndim=1)
    matching(returns, values, "returns")
    if old_values is not None:
        matching(old_values, values, "old_values")
    predicted = values
    if clip_range is not None:
        scalar(clip_range, "clip_range", minimum=0.0)
        if old_values is None:
            raise ValueError("old_values is required for clipped value loss")
        reference = old_values.detach()
        predicted = reference + (values - reference).clamp(-clip_range, clip_range)
    loss = (predicted - returns.detach()).square().mean()
    finite(loss, "value loss")
    return loss


def ppo_loss(new_log_prob: torch.Tensor, old_log_prob: torch.Tensor,
             advantages: torch.Tensor, values: torch.Tensor, returns: torch.Tensor, *,
             old_values: torch.Tensor | None = None, entropy: torch.Tensor | None = None,
             ratio_clip: float = 0.2, value_clip: float | None = None,
             value_coefficient: float = 1.0, entropy_coefficient: float = 0.0) -> PPOLoss:
    """Aggregate policy + value_coefficient*MSE - entropy_coefficient*mean(entropy).

    Entropy is [B], already reduced over actions by the caller; choose the mean
    action entropy to match skrl's regularizer, or joint entropy deliberately.
    A positive entropy coefficient requires entropy. Returned value is unscaled;
    entropy_loss is the signed, scaled contribution to total.
    """
    scalar(value_coefficient, "value_coefficient", minimum=0.0)
    scalar(entropy_coefficient, "entropy_coefficient", minimum=0.0)
    policy = clipped_policy_loss(new_log_prob, old_log_prob, advantages, clip_range=ratio_clip)
    matching(values, new_log_prob, "values")
    critic = value_loss(values, returns, old_values=old_values, clip_range=value_clip)
    if entropy is not None:
        matching(entropy, new_log_prob, "entropy")
    elif entropy_coefficient != 0:
        raise ValueError("entropy is required when entropy_coefficient is positive")
    entropy_loss = (new_log_prob.new_zeros(()) if entropy_coefficient == 0
                    else -entropy_coefficient * entropy.mean())
    total = policy.loss + value_coefficient * critic + entropy_loss
    finite(total, "total PPO loss")
    return PPOLoss(total, policy.loss, critic, entropy_loss, policy.approx_kl, policy.clip_fraction)
