"""Single-agent PPO inference, target preparation, and optimization only."""

from dataclasses import dataclass

import torch

from ._validation import finite, scalar
from .advantages import generalized_advantage_estimate
from .losses import ppo_loss
from .networks import ActorCritic, PolicyStep
from .normalization import RunningNormalizer, normalize_advantages
from .rollout import RolloutBuffer, minibatch_indices
from .schedules import adaptive_learning_rate


@dataclass(frozen=True)
class PPOConfig:
    horizon: int = 24
    learning_epochs: int = 5
    mini_batches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 5e-4
    ratio_clip: float = 0.2
    value_clip: float | None = 0.2
    max_grad_norm: float = 1.0
    value_coefficient: float = 1.0
    entropy_coefficient: float = 0.0
    reward_scale: float = 0.01
    bootstrap_truncations: bool = False
    normalize_observations: bool = True
    normalize_values: bool = True
    normalize_advantages: bool = True
    observation_clip: float | None = 5.0
    normalization_epsilon: float = 1e-8
    shuffle_minibatches: bool = False
    adaptive_lr: bool = True
    kl_threshold: float = 0.016
    min_lr: float = 1e-6
    max_lr: float = 1e-2
    kl_factor: float = 2.0
    lr_factor: float = 1.5

    def __post_init__(self):
        for name in ("horizon", "learning_epochs", "mini_batches"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("bootstrap_truncations", "normalize_observations", "normalize_values",
                     "normalize_advantages", "shuffle_minibatches", "adaptive_lr"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        for name in ("gamma", "gae_lambda"):
            scalar(getattr(self, name), name, minimum=0, maximum=1)
        scalar(self.ratio_clip, "ratio_clip", minimum=0)
        if self.ratio_clip >= 1:
            raise ValueError("ratio_clip must be less than 1")
        for name in ("learning_rate", "max_grad_norm", "reward_scale", "normalization_epsilon",
                     "kl_threshold", "min_lr", "max_lr"):
            scalar(getattr(self, name), name, minimum=0)
            if getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive")
        for name in ("value_coefficient", "entropy_coefficient"):
            scalar(getattr(self, name), name, minimum=0)
        if self.value_clip is not None:
            scalar(self.value_clip, "value_clip", minimum=0)
        if self.observation_clip is not None:
            scalar(self.observation_clip, "observation_clip", minimum=0)
            if self.observation_clip == 0:
                raise ValueError("observation_clip must be positive or None")
        for name in ("kl_factor", "lr_factor"):
            scalar(getattr(self, name), name, minimum=1)
            if getattr(self, name) == 1:
                raise ValueError(f"{name} must exceed 1")
        if self.min_lr > self.max_lr:
            raise ValueError("min_lr must not exceed max_lr")
        if self.adaptive_lr and not self.min_lr <= self.learning_rate <= self.max_lr:
            raise ValueError("Adaptive learning rate must start within its bounds")


@dataclass(frozen=True)
class PPOBatch:
    """Owned, detached [B, ...] tensors. Treat contents as read-only.

    observations are already normalized exactly once. old_values and returns
    share the same value-normalization snapshot. actions remain raw samples.
    raw_observations are retained solely for the post-update statistics update.
    reward_returns retain scaled-reward units for the value statistics update.
    """
    observations: torch.Tensor
    raw_observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    reward_returns: torch.Tensor


@dataclass(frozen=True)
class UpdateDiagnostics:
    total_loss: float
    policy_loss: float
    value_loss: float
    entropy_loss: float
    approx_kl: float
    clip_fraction: float
    grad_norm: float  # Mean norm BEFORE clipping.
    learning_rate: float  # Rate for the next epoch/update.
    optimizer_steps: int
    samples: int


class PPOLearner:
    """Compose an ActorCritic with on-policy, single-device PPO optimization.

    External code collects rollouts with act()/predict_values(), then calls
    prepare_batch() and optimize() once (or update()). No environment is created.
    Inference is blocked while a prepared batch is pending. Move model device/
    dtype before construction, not after; normalizer statistics remain float64.
    """

    def __init__(self, model: ActorCritic, config: PPOConfig | None = None):
        self.model, self.config = model, config or PPOConfig()
        self.device, self.dtype = model.log_std.device, model.log_std.dtype
        if self.dtype not in (torch.float32, torch.float64):
            raise TypeError("PPO supports float32 or float64 models")
        if any(p.device != self.device or p.dtype != self.dtype for p in model.parameters()):
            raise ValueError("All model parameters must share dtype and device")
        c = self.config
        self.observation_normalizer = (RunningNormalizer(
            model.config.observation_dim, device=self.device, epsilon=c.normalization_epsilon,
            clip=c.observation_clip) if c.normalize_observations else None)
        self.value_normalizer = (RunningNormalizer(
            1, device=self.device, epsilon=c.normalization_epsilon, clip=None)
            if c.normalize_values else None)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=c.learning_rate)
        self._pending: PPOBatch | None = None

    def _ready_for_collection(self):
        if self._pending is not None:
            raise RuntimeError("Optimize the pending batch before collecting another rollout")

    def _observations(self, observations: torch.Tensor) -> torch.Tensor:
        if self.observation_normalizer is None:
            return observations
        return self.observation_normalizer.normalize(observations)

    def _reward_values(self, values: torch.Tensor) -> torch.Tensor:
        if self.value_normalizer is None:
            return values
        return self.value_normalizer.denormalize(values.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def act(self, observations: torch.Tensor, *, deterministic: bool = False) -> PolicyStep:
        """Raw [N,O] -> raw actions [N,A], log_probs [N], scaled-reward values [N]."""
        self._ready_for_collection()
        step = self.model.act(self._observations(observations), deterministic=deterministic)
        values = self._reward_values(step.values)
        finite(values, "predicted values")
        return PolicyStep(step.actions, step.log_probs, values)

    @torch.no_grad()
    def predict_values(self, observations: torch.Tensor) -> torch.Tensor:
        """Values in scaled-reward units; no sampling and no statistics updates."""
        self._ready_for_collection()
        values = self._reward_values(self.model(self._observations(observations)).values)
        finite(values, "predicted values")
        return values

    def new_buffer(self, num_envs: int) -> RolloutBuffer:
        return RolloutBuffer(self.config.horizon, num_envs, self.model.config.observation_dim,
                             self.model.config.action_dim, device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def prepare_batch(self, buffer: RolloutBuffer) -> PPOBatch:
        """Scale raw rewards ONCE, run GAE, normalize targets, and flatten T then N.

        Collection must use this learner's unchanged policy and normalizers. A
        buffer can be prepared once per fill. Statistics stay frozen until after
        optimization. The returned batch owns all data independently.
        """
        self._ready_for_collection()
        if buffer.prepared:
            raise ValueError("This rollout has already been prepared; clear and refill it")
        if (buffer.horizon, buffer.observation_dim, buffer.action_dim) != (
                self.config.horizon, self.model.config.observation_dim, self.model.config.action_dim):
            raise ValueError("Buffer dimensions must match learner configuration")
        size = buffer.horizon * buffer.num_envs
        if size % self.config.mini_batches:
            raise ValueError("Rollout size must be divisible by mini_batches")
        rollout = buffer.snapshot()
        if rollout.observations.device != self.device or rollout.observations.dtype != self.dtype:
            raise ValueError("Rollout must match learner dtype and device")
        c = self.config
        targets = generalized_advantage_estimate(
            rollout.rewards * c.reward_scale, rollout.values, rollout.terminated, rollout.truncated,
            next_values=rollout.next_values, truncation_values=rollout.truncation_values,
            gamma=c.gamma, gae_lambda=c.gae_lambda, bootstrap_truncations=c.bootstrap_truncations,
        )
        advantages = (normalize_advantages(targets.advantages, epsilon=c.normalization_epsilon)
                      if c.normalize_advantages
                      else targets.advantages)
        raw = rollout.observations.reshape(size, -1)
        observations = self._observations(raw).clone()
        returns, old_values = targets.returns, rollout.values
        reward_returns = returns.reshape(size).clone()
        if self.value_normalizer is not None:
            # Keep collection coordinates, including the critic clipping center.
            returns = self.value_normalizer.normalize(returns.unsqueeze(-1)).squeeze(-1)
            old_values = self.value_normalizer.normalize(old_values.unsqueeze(-1)).squeeze(-1)
        batch = PPOBatch(observations, raw, rollout.actions.reshape(size, -1),
                         rollout.log_probs.reshape(size), old_values.reshape(size),
                         advantages.reshape(size), returns.reshape(size), reward_returns)
        buffer.prepared = True
        self._pending = batch
        return batch

    def optimize(self, batch: PPOBatch, *, generator: torch.Generator | None = None) -> UpdateDiagnostics:
        """Optimize one pending batch; fixed targets are never modified.

        Sequential minibatches by default; optional fresh permutation per epoch.
        Scheduler uses that epoch's mean pre-step approximate KL. If an update
        raises, earlier optimizer steps are not rolled back; do not retry the batch.
        """
        if batch is not self._pending:
            raise ValueError("Batch must be this learner's pending, unconsumed batch")
        self._pending = None  # Consume before any optimizer step; no silent retries.
        c = self.config
        size = batch.advantages.numel()
        totals = torch.zeros(7, device=self.device, dtype=torch.float64)
        steps = 0
        for _ in range(c.learning_epochs):
            epoch_kl = torch.zeros((), device=self.device, dtype=torch.float64)
            for indices in minibatch_indices(size, c.mini_batches, device=self.device,
                                              shuffle=c.shuffle_minibatches, generator=generator):
                current = self.model.evaluate_actions(batch.observations[indices], batch.actions[indices])
                losses = ppo_loss(
                    current.log_probs, batch.old_log_probs[indices], batch.advantages[indices],
                    current.values, batch.returns[indices], old_values=batch.old_values[indices],
                    entropy=current.entropy, ratio_clip=c.ratio_clip, value_clip=c.value_clip,
                    value_coefficient=c.value_coefficient, entropy_coefficient=c.entropy_coefficient,
                )
                self.optimizer.zero_grad(set_to_none=True)
                losses.total.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.max_grad_norm,
                                                          error_if_nonfinite=True)
                self.optimizer.step()
                with torch.no_grad():
                    totals += torch.stack((losses.total, losses.policy, losses.value, losses.entropy_loss,
                                           losses.approx_kl, losses.clip_fraction, grad_norm)).detach()
                    epoch_kl += losses.approx_kl
                steps += 1
            if c.adaptive_lr:
                rate = adaptive_learning_rate(
                    self.optimizer.param_groups[0]["lr"], max(0.0, (epoch_kl / c.mini_batches).item()),
                    threshold=c.kl_threshold, min_lr=c.min_lr, max_lr=c.max_lr,
                    kl_factor=c.kl_factor, lr_factor=c.lr_factor,
                )
                for group in self.optimizer.param_groups:
                    group["lr"] = rate
        for parameter in self.model.parameters():
            finite(parameter, "updated model parameters")
        if self.observation_normalizer is not None:
            # Only future rollouts see these new statistics.
            self.observation_normalizer.update(batch.raw_observations)
        if self.value_normalizer is not None:
            self.value_normalizer.update(batch.reward_returns.unsqueeze(-1))
        return UpdateDiagnostics(*(totals / steps).tolist(), self.optimizer.param_groups[0]["lr"], steps, size)

    def update(self, buffer: RolloutBuffer, *, generator: torch.Generator | None = None) -> UpdateDiagnostics:
        return self.optimize(self.prepare_batch(buffer), generator=generator)
