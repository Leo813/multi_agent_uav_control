"""Persistent tensor rollout collection and shared interaction bookkeeping."""

from dataclasses import dataclass

import torch

from ._validation import matching
from .environments import MultiObservation, MultiStepResult, StepResult
from .mappo import AGENTS, MAPPOLearner
from .ppo import PPOLearner
from .rollout import _agent_keys


@dataclass(frozen=True)
class Episode:
    agent: str
    environment: int
    reward: float
    length: int


class EnvironmentSession:
    """Owned current observations and raw episode totals, persistent across rollouts."""

    def __init__(self, learner, environment):
        if not isinstance(learner, (PPOLearner, MAPPOLearner)):
            raise TypeError("A PPO or MAPPO learner is required")
        self.learner, self.environment = learner, environment
        self.multi = isinstance(learner, MAPPOLearner)
        self.agents = AGENTS if self.multi else ('policy',)
        self.num_envs = environment.num_envs
        if type(self.num_envs) is not int or self.num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if torch.device(environment.device) != learner.device or environment.dtype != learner.dtype:
            raise ValueError("Environment and learner must match device/dtype")
        self.current = None
        self.vector_steps = 0
        self.closed = False
        self.failed = False
        self.rewards = {a: torch.zeros(self.num_envs, device=learner.device, dtype=learner.dtype) for a in self.agents}
        self.lengths = {a: torch.zeros(self.num_envs, device=learner.device, dtype=torch.int64) for a in self.agents}

    def _tensor(self, value, shape, name, *, boolean=False, finite_required=True):
        reference = torch.empty(shape, device=self.learner.device,
                                dtype=torch.bool if boolean else self.learner.dtype)
        matching(value, reference, name, check_finite=finite_required)
        return value.detach().clone()

    def copy_observation(self, value):
        n = self.num_envs
        if self.multi:
            if not isinstance(value, MultiObservation):
                raise TypeError("MAPPO requires MultiObservation")
            _agent_keys(value.observations, AGENTS, 'observations')
            return MultiObservation({a: self._tensor(value.observations[a],
                (n, self.learner.models[a].config.observation_dim), a) for a in AGENTS},
                self._tensor(value.states, (n, self.learner.state_dim), 'states'))
        return self._tensor(value, (n, self.learner.model.config.observation_dim), 'observations')

    def initialize(self):
        if self.closed or self.failed:
            raise RuntimeError("Session is closed or failed; create a new session")
        if self.current is None:
            self.current = self.copy_observation(self.environment.reset())

    @torch.no_grad()
    def step(self, *, deterministic=False, collect=False):
        """One real vector step; never reset again implicitly or scale rewards."""
        self.learner._ready_for_collection()
        self.initialize()
        previous = self.current
        try:
            if self.multi:
                samples = {a: self.learner.act(a, previous.observations[a], deterministic=deterministic)
                           for a in AGENTS}
                values = {a: self.learner.predict_values(a, previous.states) for a in AGENTS} if collect else None
                result = self.environment.step({a: s.actions.clone() for a, s in samples.items()})
                if not isinstance(result, MultiStepResult):
                    raise TypeError("MAPPO step must return MultiStepResult")
                rewards, terminated, truncated = result.rewards, result.terminated, result.truncated
                for name, data in (('rewards', rewards), ('terminated', terminated), ('truncated', truncated)):
                    _agent_keys(data, AGENTS, name)
            else:
                # Evaluation uses actor-only inference, even for the single-agent learner.
                if collect:
                    sample = self.learner.act(previous, deterministic=deterministic)
                    values = {'policy': sample.values}
                else:
                    sample = self.learner.model.sample_policy(self.learner._observations(previous),
                                                              deterministic=deterministic)
                    values = None
                samples = {'policy': sample}
                result = self.environment.step(sample.actions.clone())
                if not isinstance(result, StepResult):
                    raise TypeError("PPO step must return StepResult")
                rewards, terminated, truncated = ({'policy': result.rewards}, {'policy': result.terminated},
                                                   {'policy': result.truncated})
            following = self.copy_observation(result.observations)
            rewards = {a: self._tensor(rewards[a], (self.num_envs,), 'rewards') for a in self.agents}
            terminated = {a: self._tensor(terminated[a], (self.num_envs,), 'terminated', boolean=True) for a in self.agents}
            truncated = {a: self._tensor(truncated[a], (self.num_envs,), 'truncated', boolean=True) for a in self.agents}
            transition = None
            if collect:
                next_values, final_values = {}, {}
                for a in self.agents:
                    c = self.learner.configs[a] if self.multi else self.learner.config
                    predictor = (lambda x, agent=a: self.learner.predict_values(agent, x)) if self.multi else self.learner.predict_values
                    next_input = following.states if self.multi else following
                    next_values[a] = torch.full_like(rewards[a], float('nan'))
                    continuing = ~(terminated[a] | truncated[a])
                    if c.gamma and continuing.any():
                        next_values[a][continuing] = predictor(next_input[continuing])
                    final_values[a] = torch.full_like(rewards[a], float('nan'))
                    needed = truncated[a] & ~terminated[a]
                    if c.gamma and c.bootstrap_truncations and needed.any():
                        if self.multi:
                            if result.final_states is None or result.final_masks is None:
                                raise ValueError(f"{a}: truncation bootstrap requires final pre-reset states and mask")
                            _agent_keys(result.final_states, AGENTS, 'final_states')
                            _agent_keys(result.final_masks, AGENTS, 'final_masks')
                            final, mask = result.final_states[a], result.final_masks[a]
                        else:
                            final, mask = result.final_observations, result.final_mask
                        if final is None or mask is None:
                            raise ValueError(f"{a}: truncation bootstrap requires final pre-reset observations and mask")
                        final = self._tensor(final, next_input.shape, 'final inputs', finite_required=False)
                        mask = self._tensor(mask, (self.num_envs,), 'final mask', boolean=True)
                        if (needed & ~mask).any():
                            raise ValueError(f"{a}: missing valid final pre-reset input for a truncation")
                        final_values[a][needed] = predictor(final[needed])
                transition = dict(observations=previous.observations if self.multi else previous,
                                  actions={a: s.actions for a, s in samples.items()}, rewards=rewards,
                                  log_probs={a: s.log_probs for a, s in samples.items()}, values=values,
                                  terminated=terminated, truncated=truncated,
                                  next_values=next_values, truncation_values=final_values)
                if self.multi:
                    transition['states'] = previous.states
                else:
                    transition = {k: v['policy'] if isinstance(v, dict) else v for k, v in transition.items()}
            episodes = []
            for a in self.agents:
                self.rewards[a].add_(rewards[a])
                self.lengths[a].add_(1)
                done = terminated[a] | truncated[a]
                indices = done.nonzero().flatten()
                for index, reward, length in zip(indices.tolist(), self.rewards[a][indices].tolist(),
                                                  self.lengths[a][indices].tolist()):
                    episodes.append(Episode(a, index, reward, length))
                self.rewards[a][done] = 0
                self.lengths[a][done] = 0
            self.current = following
            self.vector_steps += 1
            return transition, episodes
        except Exception:
            self.failed = True  # Environment may already have advanced; never silently retry.
            raise

    def state_dict(self):
        """Optional data-only exact interaction snapshot, between complete updates."""
        self.initialize()
        if not callable(getattr(self.environment, 'state_dict', None)):
            raise ValueError("Environment does not support exact interaction snapshots")
        observation = ({'observations': self.current.observations, 'states': self.current.states}
                       if self.multi else self.current)
        return dict(environment=self.environment.state_dict(), current=observation,
                    rewards=self.rewards, lengths=self.lengths, vector_steps=self.vector_steps,
                    num_envs=self.num_envs)

    def load_state_dict(self, state):
        if self.current is not None or self.closed or self.failed:
            raise RuntimeError("Restore interaction into a fresh session")
        required = {'environment', 'current', 'rewards', 'lengths', 'vector_steps', 'num_envs'}
        if not isinstance(state, dict) or set(state) != required:
            raise ValueError("Invalid interaction snapshot fields")
        if (type(state['num_envs']) is not int or state['num_envs'] != self.num_envs
                or type(state['vector_steps']) is not int or state['vector_steps'] < 0):
            raise ValueError("Incompatible interaction counters/environment count")
        saved_current = (MultiObservation(
            {a: t.to(self.learner.device) for a, t in state['current']['observations'].items()},
            state['current']['states'].to(self.learner.device)) if self.multi
            else state['current'].to(self.learner.device))
        current = self.copy_observation(saved_current)
        _agent_keys(state['rewards'], self.agents, 'episode rewards')
        _agent_keys(state['lengths'], self.agents, 'episode lengths')
        rewards = {a: self._tensor(state['rewards'][a].to(self.learner.device), (self.num_envs,), 'episode rewards')
                   for a in self.agents}
        lengths = {}
        for a in self.agents:
            value = state['lengths'][a].to(self.learner.device)
            matching(value, self.lengths[a], 'episode lengths')
            if (value < 0).any():
                raise ValueError("Episode lengths must be nonnegative")
            lengths[a] = value.clone()
        if not callable(getattr(self.environment, 'load_state_dict', None)):
            raise ValueError("Environment does not support exact interaction restoration")
        restored = self.copy_observation(self.environment.load_state_dict(state['environment']))
        pairs = ([(current.states, restored.states)] + [(current.observations[a], restored.observations[a]) for a in AGENTS]
                 if self.multi else [(current, restored)])
        if any(not torch.equal(left, right) for left, right in pairs):
            raise ValueError("Restored environment observations do not match the saved interaction")
        self.current, self.rewards, self.lengths = current, rewards, lengths
        self.vector_steps = state['vector_steps']

    def close(self):
        if not self.closed:
            self.closed = True
            self.environment.close()


class PPOCollector(EnvironmentSession):
    def __init__(self, learner: PPOLearner, environment):
        if not isinstance(learner, PPOLearner):
            raise TypeError("PPOCollector requires PPOLearner")
        super().__init__(learner, environment)

    def collect(self):
        buffer = self.learner.new_buffer(self.num_envs)
        episodes = []
        for _ in range(self.learner.config.horizon):
            transition, completed = self.step(collect=True)
            buffer.add(**transition)
            episodes.extend(completed)
        return buffer, episodes


class MAPPOCollector(EnvironmentSession):
    def __init__(self, learner: MAPPOLearner, environment):
        if not isinstance(learner, MAPPOLearner):
            raise TypeError("MAPPOCollector requires MAPPOLearner")
        super().__init__(learner, environment)

    def collect(self):
        buffer = self.learner.new_buffer(self.num_envs)
        episodes = []
        for _ in range(self.learner.horizon):
            transition, completed = self.step(collect=True)
            buffer.add(**transition)
            episodes.extend(completed)
        return buffer, episodes
