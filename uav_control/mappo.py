"""Two independent local policies with agent-specific centralized critics."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from ._validation import finite, scalar
from .networks import ActorCritic, NetworkConfig, PolicySample
from .normalization import RunningNormalizer
from .ppo import PPOBatch, PPOConfig, UpdateDiagnostics, _optimize_ppo, _prepare_ppo_batch
from .rollout import MultiAgentRolloutBuffer, _agent_keys


AGENTS = ("Translation", "Yaw")


@dataclass(frozen=True)
class MAPPOConfig(PPOConfig):
    """Per-agent PPO settings with UAV MAPPO defaults and state preprocessing."""
    horizon: int = 16
    learning_epochs: int = 8
    mini_batches: int = 2
    learning_rate: float = 3e-4
    kl_threshold: float = 0.008
    normalize_states: bool = True
    state_clip: float | None = 5.0

    def __post_init__(self):
        super().__post_init__()
        if type(self.normalize_states) is not bool:
            raise TypeError("normalize_states must be bool")
        if self.state_clip is not None:
            scalar(self.state_clip, "state_clip", minimum=0)
            if self.state_clip == 0:
                raise ValueError("state_clip must be positive or None")


@dataclass(frozen=True)
class MAPPOBatch:
    """Owned detached tensors; treat mappings and tensors as read-only."""
    agents: dict[str, PPOBatch]
    critic_states: dict[str, torch.Tensor]
    raw_states: torch.Tensor


class MAPPOLearner:
    """Synchronous Translation/Yaw learner; no environment or centralized actor.

    Model device/dtype must be finalized before construction. Per-agent Adam
    optimizers cover separate actor and critic parameters, as in skrl 1.4.2.
    One prepared batch is consumed by one update across both agents.
    """

    def __init__(self, models: Mapping[str, ActorCritic] | None = None,
                 configs: Mapping[str, MAPPOConfig] | None = None):
        if models is None:
            models = {name: ActorCritic(NetworkConfig(obs, act, (32, 32), shared=False,
                                                     critic_observation_dim=30))
                      for name, obs, act in (("Translation", 21, 4), ("Yaw", 9, 1))}
        if configs is None:
            configs = {a: MAPPOConfig() for a in AGENTS}
        _agent_keys(models, AGENTS, "models")
        _agent_keys(configs, AGENTS, "configs")
        if any(not isinstance(m, ActorCritic) for m in models.values()):
            raise TypeError("Models must be ActorCritic instances")
        if any(not isinstance(c, MAPPOConfig) for c in configs.values()):
            raise TypeError("Configs must be MAPPOConfig instances")
        self.models, self.configs = dict(models), dict(configs)
        first = self.models[AGENTS[0]]
        self.device, self.dtype = first.log_std.device, first.log_std.dtype
        self.state_dim = first.config.critic_observation_dim
        self.horizon = self.configs[AGENTS[0]].horizon
        if self.dtype not in (torch.float32, torch.float64):
            raise TypeError("MAPPO supports float32 or float64 models")
        storage = set()
        for agent in AGENTS:
            model = self.models[agent]
            if model.config.shared or model.config.critic_observation_dim is None:
                raise ValueError("MAPPO requires separate actors and explicit centralized critics")
            if model.config.critic_observation_dim != self.state_dim:
                raise ValueError("Both critics must accept the same centralized state dimension")
            if self.configs[agent].horizon != self.horizon:
                raise ValueError("Both agents must share one rollout horizon")
            for parameter in model.parameters():
                if parameter.device != self.device or parameter.dtype != self.dtype:
                    raise ValueError("All MAPPO parameters must share dtype and device")
                address = parameter.untyped_storage().data_ptr()
                if address in storage:
                    raise ValueError("Agents must not share parameter storage")
                storage.add(address)
        self.optimizers = {a: torch.optim.Adam(m.parameters(), lr=self.configs[a].learning_rate)
                           for a, m in self.models.items()}
        self.observation_normalizers, self.state_normalizers, self.value_normalizers = {}, {}, {}
        for agent in AGENTS:
            c = self.configs[agent]
            options = dict(device=self.device, epsilon=c.normalization_epsilon)
            self.observation_normalizers[agent] = (RunningNormalizer(
                self.models[agent].config.observation_dim, clip=c.observation_clip, **options)
                if c.normalize_observations else None)
            self.state_normalizers[agent] = (RunningNormalizer(self.state_dim, clip=c.state_clip, **options)
                                             if c.normalize_states else None)
            self.value_normalizers[agent] = (RunningNormalizer(1, clip=None, **options)
                                             if c.normalize_values else None)
        self._pending: MAPPOBatch | None = None

    def _ready_for_collection(self):
        if self._pending is not None:
            raise RuntimeError("Optimize the pending MAPPO batch before collecting again")

    def _agent(self, agent: str):
        if agent not in AGENTS:
            raise ValueError(f"Unknown agent: {agent}")

    @torch.no_grad()
    def act(self, agent: str, observations: torch.Tensor, *, deterministic: bool = False) -> PolicySample:
        """Decentralized action/log-probability selection from this agent's [N,O]."""
        self._ready_for_collection()
        self._agent(agent)
        scaler = self.observation_normalizers[agent]
        inputs = observations if scaler is None else scaler.normalize(observations)
        return self.models[agent].sample_policy(inputs, deterministic=deterministic)

    @torch.no_grad()
    def predict_values(self, agent: str, states: torch.Tensor) -> torch.Tensor:
        """Centralized [N,S] -> agent values [N] in scaled-reward units."""
        self._ready_for_collection()
        self._agent(agent)
        scaler = self.state_normalizers[agent]
        inputs = states if scaler is None else scaler.normalize(states)
        values = self.models[agent].predict_values(inputs)
        scaler = self.value_normalizers[agent]
        if scaler is not None:
            values = scaler.denormalize(values.unsqueeze(-1)).squeeze(-1)
        finite(values, "predicted values")
        return values

    def new_buffer(self, num_envs: int) -> MultiAgentRolloutBuffer:
        dimensions = {a: (self.models[a].config.observation_dim, self.models[a].config.action_dim) for a in AGENTS}
        return MultiAgentRolloutBuffer(self.horizon, num_envs, dimensions, self.state_dim,
                                       device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def prepare_batch(self, buffer: MultiAgentRolloutBuffer) -> MAPPOBatch:
        self._ready_for_collection()
        if buffer.prepared:
            raise ValueError("This rollout has already been prepared; clear and refill it")
        dimensions = {a: (self.models[a].config.observation_dim, self.models[a].config.action_dim) for a in AGENTS}
        if (buffer.horizon != self.horizon or buffer.state_dim != self.state_dim
                or buffer.agent_dimensions != dimensions):
            raise ValueError("Buffer dimensions must match MAPPO models/configuration")
        size = buffer.horizon * buffer.num_envs
        if any(size % c.mini_batches for c in self.configs.values()):
            raise ValueError("Rollout size must be divisible by each agent's mini_batches")
        rollout = buffer.snapshot()
        if rollout.states.device != self.device or rollout.states.dtype != self.dtype:
            raise ValueError("Rollout must match learner dtype and device")
        raw_states = rollout.states.reshape(size, self.state_dim)
        batches, states = {}, {}
        # Validate and prepare BOTH agents before consuming the buffer or changing state.
        for agent in AGENTS:
            batches[agent] = _prepare_ppo_batch(rollout.agents[agent], self.configs[agent],
                                               self.observation_normalizers[agent], self.value_normalizers[agent])
            scaler = self.state_normalizers[agent]
            states[agent] = (raw_states if scaler is None else scaler.normalize(raw_states)).clone()
        result = MAPPOBatch(batches, states, raw_states)
        buffer.prepared = True
        self._pending = result
        return result

    def optimize(self, batch: MAPPOBatch, *, generator: torch.Generator | None = None
                 ) -> dict[str, UpdateDiagnostics]:
        """Independent per-agent updates; update all statistics only after both finish.

        A failed update may leave partial optimizer changes; batch retry is forbidden.
        """
        if batch is not self._pending:
            raise ValueError("Batch must be this learner's pending, unconsumed batch")
        self._pending = None
        reports = {}
        for agent in AGENTS:
            reports[agent] = _optimize_ppo(self.models[agent], self.optimizers[agent], batch.agents[agent],
                                          self.configs[agent], critic_states=batch.critic_states[agent],
                                          generator=generator)
        for agent in AGENTS:
            for scaler, data in ((self.observation_normalizers[agent], batch.agents[agent].raw_observations),
                                 (self.state_normalizers[agent], batch.raw_states),
                                 (self.value_normalizers[agent], batch.agents[agent].reward_returns.unsqueeze(-1))):
                if scaler is not None:
                    scaler.update(data)
        return reports

    def update(self, buffer: MultiAgentRolloutBuffer, *, generator: torch.Generator | None = None
               ) -> dict[str, UpdateDiagnostics]:
        return self.optimize(self.prepare_batch(buffer), generator=generator)
