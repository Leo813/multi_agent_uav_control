"""Configurable feedforward Gaussian actor and scalar critic."""

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn

from ._validation import floating, scalar
from .distributions import gaussian_entropy, gaussian_log_prob, sample_actions


@dataclass(frozen=True)
class NetworkConfig:
    observation_dim: int = 15
    action_dim: int = 4
    hidden_sizes: tuple[int, ...] = (64, 64)
    activation: str = "elu"
    shared: bool = True
    initial_log_std: float = 0.0
    trainable_log_std: bool = True
    min_log_std: float = -20.0
    max_log_std: float = 2.0

    def __post_init__(self):
        for size in (self.observation_dim, self.action_dim, *self.hidden_sizes):
            if type(size) is not int or size <= 0:
                raise ValueError("Network dimensions must be positive integers")
        if not isinstance(self.hidden_sizes, tuple):
            raise TypeError("hidden_sizes must be a tuple")
        if self.activation not in ("elu", "relu", "tanh"):
            raise ValueError("activation must be elu, relu, or tanh")
        if type(self.shared) is not bool or type(self.trainable_log_std) is not bool:
            raise TypeError("shared and trainable_log_std must be bool")
        for name in ("initial_log_std", "min_log_std", "max_log_std"):
            scalar(getattr(self, name), name)
        # Safe for both supported tensor dtypes; narrower than their extreme limits.
        if not -40 <= self.min_log_std <= self.initial_log_std <= self.max_log_std <= 40:
            raise ValueError("log_std bounds must be ordered within [-40, 40] and contain the initial value")


class NetworkOutput(NamedTuple):
    mean: torch.Tensor
    log_std: torch.Tensor
    values: torch.Tensor


class PolicyStep(NamedTuple):
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor


class ActionEvaluation(NamedTuple):
    log_probs: torch.Tensor
    entropy: torch.Tensor
    values: torch.Tensor


class ActorCritic(nn.Module):
    """Shared hidden trunk by default, with distinct linear policy/value heads.

    Inputs are already-preprocessed [B, O]. Values are in the critic's output
    coordinates; normalization belongs to PPOLearner. No recurrent state or cache.
    Move the model to its final device/dtype BEFORE constructing the learner.
    """

    def __init__(self, config: NetworkConfig | None = None):
        super().__init__()
        self.config = config or NetworkConfig()
        c = self.config
        activations = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}

        def trunk():
            layers = []
            width = c.observation_dim
            for next_width in c.hidden_sizes:
                layers.extend((nn.Linear(width, next_width), activations[c.activation]()))
                width = next_width
            return nn.Sequential(*layers)

        self.policy_trunk = trunk()
        self.value_trunk = None if c.shared else trunk()
        width = c.hidden_sizes[-1] if c.hidden_sizes else c.observation_dim
        self.policy_head = nn.Linear(width, c.action_dim)
        self.value_head = nn.Linear(width, 1)
        self.log_std = nn.Parameter(torch.full((c.action_dim,), float(c.initial_log_std)),
                                    requires_grad=c.trainable_log_std)

    def forward(self, observations: torch.Tensor) -> NetworkOutput:
        floating(observations, "observations", ndim=2)
        if observations.shape[1] != self.config.observation_dim:
            raise ValueError("Incorrect observation dimension")
        if observations.device != self.log_std.device or observations.dtype != self.log_std.dtype:
            raise ValueError("Observations must match model dtype and device")
        features = self.policy_trunk(observations)
        critic_features = features if self.value_trunk is None else self.value_trunk(observations)
        return NetworkOutput(self.policy_head(features), self.log_std,
                             self.value_head(critic_features).squeeze(-1))

    @property
    def distribution_bounds(self) -> dict[str, float]:
        return {"min_log_std": self.config.min_log_std, "max_log_std": self.config.max_log_std}

    @torch.no_grad()
    def act(self, observations: torch.Tensor, *, deterministic: bool = False) -> PolicyStep:
        mean, log_std, values = self(observations)
        actions = sample_actions(mean, log_std, deterministic=deterministic, **self.distribution_bounds)
        return PolicyStep(actions, gaussian_log_prob(actions, mean, log_std, **self.distribution_bounds), values)

    def evaluate_actions(self, observations: torch.Tensor, actions: torch.Tensor) -> ActionEvaluation:
        mean, log_std, values = self(observations)
        return ActionEvaluation(
            gaussian_log_prob(actions, mean, log_std, **self.distribution_bounds),
            gaussian_entropy(mean, log_std, reduction="mean", **self.distribution_bounds), values,
        )
