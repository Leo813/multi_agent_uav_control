"""Owned time-major storage; no environment stepping or reset assumptions."""

from dataclasses import dataclass, fields
from collections.abc import Iterator, Mapping

import torch

from ._validation import matching


@dataclass(frozen=True)
class Rollout:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    log_probs: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    next_values: torch.Tensor
    truncation_values: torch.Tensor


class RolloutBuffer:
    """Append N transitions at a time; snapshot() returns independent copies.

    Rewards are RAW environment rewards. Values/next_values/truncation_values
    are in scaled-reward units, returned by PPOLearner value inference. Bootstrap
    entries may be absent/NaN only where GAE will not use them. Call clear() before
    refilling a full buffer. Prepared rollouts cannot be prepared a second time.
    """

    def __init__(self, horizon: int, num_envs: int, observation_dim: int, action_dim: int, *,
                 device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32):
        if any(type(n) is not int or n <= 0 for n in (horizon, num_envs, observation_dim, action_dim)):
            raise ValueError("Buffer dimensions must be positive integers")
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("Buffer dtype must be float32 or float64")
        self.horizon, self.num_envs = horizon, num_envs
        self.observation_dim, self.action_dim = observation_dim, action_dim
        self.position, self.prepared = 0, False
        shape = (horizon, num_envs)
        options = dict(device=device, dtype=dtype)
        self._data = Rollout(
            torch.empty(*shape, observation_dim, **options), torch.empty(*shape, action_dim, **options),
            *(torch.empty(shape, **options) for _ in range(3)),
            *(torch.empty(shape, device=device, dtype=torch.bool) for _ in range(2)),
            *(torch.full(shape, float("nan"), **options) for _ in range(2)),
        )

    @torch.no_grad()
    def add(self, *, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
            values: torch.Tensor, log_probs: torch.Tensor, terminated: torch.Tensor,
            truncated: torch.Tensor, next_values: torch.Tensor | None = None,
            truncation_values: torch.Tensor | None = None) -> None:
        supplied = self._validate_transition(
            observations=observations, actions=actions, rewards=rewards, values=values,
            log_probs=log_probs, terminated=terminated, truncated=truncated,
            next_values=next_values, truncation_values=truncation_values)
        self._append_transition(supplied)

    def _validate_transition(self, *, observations, actions, rewards, values, log_probs,
                             terminated, truncated, next_values=None, truncation_values=None):
        """Validate without mutation so a multi-agent row can be checked atomically."""
        if self.position == self.horizon:
            raise ValueError("Buffer is full; clear it before adding more transitions")
        supplied = dict(observations=observations, actions=actions, rewards=rewards, values=values,
                        log_probs=log_probs, terminated=terminated, truncated=truncated,
                        next_values=next_values, truncation_values=truncation_values)
        # Validate the entire transition before writing any part of it.
        for name, tensor in supplied.items():
            if tensor is None:
                if name not in ("next_values", "truncation_values"):
                    raise TypeError(f"{name} must be a tensor")
                continue
            matching(tensor, getattr(self._data, name)[self.position], name,
                     check_finite=name not in ("next_values", "truncation_values"))
        return supplied

    @torch.no_grad()
    def _append_transition(self, supplied):
        for name, tensor in supplied.items():
            target = getattr(self._data, name)[self.position]
            if tensor is None:
                target.fill_(float("nan"))
            else:
                target.copy_(tensor.detach())
        self.position += 1

    def snapshot(self) -> Rollout:
        if self.position != self.horizon:
            raise ValueError("A complete rollout is required")
        return Rollout(*(getattr(self._data, f.name).clone() for f in fields(Rollout)))

    def clear(self) -> None:
        self.position, self.prepared = 0, False


def _agent_keys(data: Mapping, agents: tuple[str, ...], name: str) -> None:
    if not isinstance(data, Mapping) or set(data) != set(agents):
        raise ValueError(f"{name} must contain exactly the agents {agents}")


@dataclass(frozen=True)
class MultiAgentRollout:
    states: torch.Tensor
    agents: dict[str, Rollout]


class MultiAgentRolloutBuffer:
    """Synchronized two-agent rows with one owned [T,N,S] centralized state.

    Each field passed to add() except states is a mapping keyed by agent name.
    Individual rewards and boundaries are never combined. All fields for both
    agents are validated before appending any data. Snapshot tensors are owned.
    """

    def __init__(self, horizon: int, num_envs: int, agent_dimensions: Mapping[str, tuple[int, int]],
                 state_dim: int, *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32):
        if not isinstance(agent_dimensions, Mapping) or len(agent_dimensions) != 2:
            raise ValueError("Exactly two agent dimension entries are required")
        if any(not isinstance(name, str) or not name for name in agent_dimensions):
            raise ValueError("Agent names must be nonempty strings")
        if type(state_dim) is not int or state_dim <= 0:
            raise ValueError("state_dim must be a positive integer")
        self.agent_dimensions = dict(agent_dimensions)
        self.agents = tuple(agent_dimensions)
        self._buffers = {name: RolloutBuffer(horizon, num_envs, *dimensions, device=device, dtype=dtype)
                         for name, dimensions in agent_dimensions.items()}
        self.horizon, self.num_envs, self.state_dim = horizon, num_envs, state_dim
        self._states = torch.empty(horizon, num_envs, state_dim, device=device, dtype=dtype)
        self.position, self.prepared = 0, False

    @torch.no_grad()
    def add(self, *, states: torch.Tensor, observations: Mapping, actions: Mapping, rewards: Mapping,
            values: Mapping, log_probs: Mapping, terminated: Mapping, truncated: Mapping,
            next_values: Mapping | None = None, truncation_values: Mapping | None = None) -> None:
        if self.position == self.horizon:
            raise ValueError("Buffer is full; clear it before adding more transitions")
        matching(states, self._states[self.position], "states")
        fields = dict(observations=observations, actions=actions, rewards=rewards, values=values,
                      log_probs=log_probs, terminated=terminated, truncated=truncated)
        for name, data in (("next_values", next_values), ("truncation_values", truncation_values)):
            fields[name] = {agent: None for agent in self.agents} if data is None else data
        for name, data in fields.items():
            _agent_keys(data, self.agents, name)
        validated = {agent: self._buffers[agent]._validate_transition(
            **{name: data[agent] for name, data in fields.items()}) for agent in self.agents}
        for agent in self.agents:
            self._buffers[agent]._append_transition(validated[agent])
        self._states[self.position].copy_(states.detach())
        self.position += 1

    def snapshot(self) -> MultiAgentRollout:
        if self.position != self.horizon:
            raise ValueError("A complete rollout is required")
        return MultiAgentRollout(self._states.clone(), {a: b.snapshot() for a, b in self._buffers.items()})

    def clear(self) -> None:
        for buffer in self._buffers.values():
            buffer.clear()
        self.position, self.prepared = 0, False


def minibatch_indices(batch_size: int, mini_batches: int, *, device: torch.device | str = "cpu",
                      shuffle: bool = False, generator: torch.Generator | None = None
                      ) -> Iterator[torch.Tensor]:
    """Partition all samples exactly once. Require equal-size minibatches.

    Sequential time-major flattening is the default. Optional shuffling uses a
    device-local permutation; a supplied generator must match that device.
    """
    if type(batch_size) is not int or type(mini_batches) is not int:
        raise TypeError("Batch sizes must be integers")
    if batch_size <= 0 or mini_batches <= 0 or batch_size % mini_batches:
        raise ValueError("Batch size must be positive and divisible by mini_batches")
    if type(shuffle) is not bool:
        raise TypeError("shuffle must be bool")
    order = (torch.randperm(batch_size, device=device, generator=generator) if shuffle
             else torch.arange(batch_size, device=device))
    yield from order.split(batch_size // mini_batches)
