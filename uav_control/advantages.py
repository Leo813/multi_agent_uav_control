"""Time-major GAE with explicit transition boundaries and bootstrap inputs."""

from typing import NamedTuple

import torch

from ._validation import finite, floating, matching, scalar


class GAEOutput(NamedTuple):
    advantages: torch.Tensor
    returns: torch.Tensor


@torch.no_grad()
def generalized_advantage_estimate(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    *,
    next_values: torch.Tensor | None = None,
    truncation_values: torch.Tensor | None = None,
    bootstrap_truncations: bool = False,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> GAEOutput:
    """Return detached, unnormalized advantages and returns, each [T, N].

    rewards/values are float32 or float64 [T, N]; values describe pre-action
    states. Boolean terminated/truncated describe the result of that action.
    All tensors must share shape and device; floating inputs also share dtype.

    next_values[t,n] is V(s_next) for continuing transitions, including the
    rollout's last row. It is required when gamma > 0 and any transition continues.
    A truncated transition never reads next_values: when bootstrap_truncations
    is True, truncation_values[t,n] must supply the final PRE-RESET state value.
    Missing required values raise. Terminal transitions never bootstrap, even
    if truncated is also True. Both flags stop the recursive advantage trace.

    Optional bootstrap tensors have shape [T, N]. Unused entries may be NaN;
    only entries actually used must be finite. With gamma=0 no bootstrap is needed.
    Rewards must be unbootstrapped (no prior time-limit reward correction).
    """
    floating(rewards, "rewards", ndim=2)
    matching(values, rewards, "values")
    for name, flag in (("terminated", terminated), ("truncated", truncated)):
        if not isinstance(flag, torch.Tensor) or flag.dtype != torch.bool:
            raise TypeError(f"{name} must be a boolean tensor")
        if flag.shape != rewards.shape or flag.device != rewards.device:
            raise ValueError(f"{name} must match rewards shape and device")
    scalar(gamma, "gamma", minimum=0.0, maximum=1.0)
    scalar(gae_lambda, "gae_lambda", minimum=0.0, maximum=1.0)
    if not isinstance(bootstrap_truncations, bool):
        raise TypeError("bootstrap_truncations must be bool")

    continuing = ~(terminated | truncated)
    timeout_bootstrap = truncated & ~terminated & bootstrap_truncations
    bootstrap = torch.zeros_like(values)
    for name, tensor, mask in (("next_values", next_values, continuing),
                               ("truncation_values", truncation_values, timeout_bootstrap)):
        if tensor is not None:
            matching(tensor, values, name, check_finite=False)
        if gamma != 0 and mask.any():
            if tensor is None:
                raise ValueError(f"{name} is required for the requested bootstrap")
            finite(tensor[mask], name)
            bootstrap = torch.where(mask, tensor, bootstrap)

    deltas = rewards + gamma * bootstrap - values
    advantages = torch.empty_like(values)
    carry = torch.zeros_like(values[0])
    for t in range(rewards.shape[0] - 1, -1, -1):
        carry = deltas[t] + gamma * gae_lambda * torch.where(continuing[t], carry, 0.0)
        advantages[t] = carry
    returns = advantages + values
    finite(advantages, "advantages")
    finite(returns, "returns")
    return GAEOutput(advantages, returns)
