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
    critic_observation_dim: int | None = None
    critic_hidden_sizes: tuple[int, ...] | None = None

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
        if self.critic_observation_dim is not None:
            if type(self.critic_observation_dim) is not int or self.critic_observation_dim <= 0:
                raise ValueError("critic_observation_dim must be a positive integer")
            if self.shared:
                raise ValueError("Centralized critic inputs require separate trunks")
        if self.critic_hidden_sizes is not None:
            if self.shared or not isinstance(self.critic_hidden_sizes, tuple):
                raise ValueError("critic_hidden_sizes requires separate trunks and a tuple")
            if any(type(n) is not int or n <= 0 for n in self.critic_hidden_sizes):
                raise ValueError("Critic widths must be positive integers")
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


class PolicySample(NamedTuple):
    actions: torch.Tensor
    log_probs: torch.Tensor


class ActionEvaluation(NamedTuple):
    log_probs: torch.Tensor
    entropy: torch.Tensor
    values: torch.Tensor


class ActorCritic(nn.Module):
    """Shared hidden trunk by default, with distinct linear policy/value heads.

    Inputs are already-preprocessed [B, O], with explicit [B, S] critic_states
    when configured. sample_policy() uses only local observations. Values are in
    critic coordinates; normalization belongs to the learner. No recurrent cache.
    Move the model to its final device/dtype BEFORE constructing the learner.
    """

    def __init__(self, config: NetworkConfig | None = None):
        super().__init__()
        self.config = config or NetworkConfig()
        c = self.config
        activations = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}

        def trunk(width, widths):
            layers = []
            for next_width in widths:
                layers.extend((nn.Linear(width, next_width), activations[c.activation]()))
                width = next_width
            return nn.Sequential(*layers)

        critic_dim = c.critic_observation_dim or c.observation_dim
        critic_widths = c.hidden_sizes if c.critic_hidden_sizes is None else c.critic_hidden_sizes
        self.policy_trunk = trunk(c.observation_dim, c.hidden_sizes)
        self.value_trunk = None if c.shared else trunk(critic_dim, critic_widths)
        width = c.hidden_sizes[-1] if c.hidden_sizes else c.observation_dim
        self.policy_head = nn.Linear(width, c.action_dim)
        self.value_head = nn.Linear(critic_widths[-1] if critic_widths else critic_dim, 1)
        self.log_std = nn.Parameter(torch.full((c.action_dim,), float(c.initial_log_std)),
                                    requires_grad=c.trainable_log_std)

    def _check_input(self, data: torch.Tensor, dimension: int, name: str) -> None:
        floating(data, name, ndim=2)
        if data.shape[1] != dimension:
            raise ValueError(f"Incorrect {name} dimension")
        if data.device != self.log_std.device or data.dtype != self.log_std.dtype:
            raise ValueError(f"{name} must match model dtype and device")

    def policy_mean(self, observations: torch.Tensor) -> torch.Tensor:
        """Actor-only evaluation; neither requires nor evaluates critic inputs."""
        self._check_input(observations, self.config.observation_dim, "observations")
        return self.policy_head(self.policy_trunk(observations))

    def predict_values(self, inputs: torch.Tensor) -> torch.Tensor:
        """Critic-only evaluation on local PPO observations or centralized states."""
        dimension = self.config.critic_observation_dim or self.config.observation_dim
        self._check_input(inputs, dimension, "critic inputs")
        trunk = self.policy_trunk if self.value_trunk is None else self.value_trunk
        return self.value_head(trunk(inputs)).squeeze(-1)

    def forward(self, observations: torch.Tensor, *,
                critic_states: torch.Tensor | None = None) -> NetworkOutput:
        self._check_input(observations, self.config.observation_dim, "observations")
        if self.config.critic_observation_dim is not None:
            if critic_states is None:
                raise ValueError("Centralized critic states are required")
            values = self.predict_values(critic_states)
            if values.shape[0] != observations.shape[0]:
                raise ValueError("Actor and critic batches must align")
            return NetworkOutput(self.policy_mean(observations), self.log_std, values)
        if critic_states is not None:
            raise ValueError("This network does not accept centralized critic states")
        features = self.policy_trunk(observations)
        critic_features = features if self.value_trunk is None else self.value_trunk(observations)
        return NetworkOutput(self.policy_head(features), self.log_std,
                             self.value_head(critic_features).squeeze(-1))

    @property
    def distribution_bounds(self) -> dict[str, float]:
        return {"min_log_std": self.config.min_log_std, "max_log_std": self.config.max_log_std}

    @torch.no_grad()
    def act(self, observations: torch.Tensor, *, deterministic: bool = False,
            critic_states: torch.Tensor | None = None) -> PolicyStep:
        mean, log_std, values = self(observations, critic_states=critic_states)
        actions = sample_actions(mean, log_std, deterministic=deterministic, **self.distribution_bounds)
        return PolicyStep(actions, gaussian_log_prob(actions, mean, log_std, **self.distribution_bounds), values)

    @torch.no_grad()
    def sample_policy(self, observations: torch.Tensor, *, deterministic: bool = False) -> PolicySample:
        mean = self.policy_mean(observations)
        actions = sample_actions(mean, self.log_std, deterministic=deterministic, **self.distribution_bounds)
        return PolicySample(actions, gaussian_log_prob(actions, mean, self.log_std, **self.distribution_bounds))

    def policy_log_probs(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return gaussian_log_prob(actions, self.policy_mean(observations), self.log_std,
                                 **self.distribution_bounds)

    def evaluate_actions(self, observations: torch.Tensor, actions: torch.Tensor, *,
                         critic_states: torch.Tensor | None = None) -> ActionEvaluation:
        mean, log_std, values = self(observations, critic_states=critic_states)
        return ActionEvaluation(
            gaussian_log_prob(actions, mean, log_std, **self.distribution_bounds),
            gaussian_entropy(mean, log_std, reduction="mean", **self.distribution_bounds), values,
        )
