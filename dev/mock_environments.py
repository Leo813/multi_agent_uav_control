"""Deterministic CPU tensor fixtures, not UAV dynamics or a simulator replacement."""

import torch

from uav_control._validation import matching
from uav_control.environments import MultiObservation, MultiStepResult, StepResult
from uav_control.mappo import AGENTS


class _Mock:
    def __init__(self, num_envs, *, multi, dtype=torch.float32, expose_final=True):
        if type(num_envs) is not int or num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Mock dtype must be float32 or float64")
        self.num_envs, self.multi, self.dtype = num_envs, multi, dtype
        self.device = torch.device('cpu')
        self.expose_final = expose_final
        self.agents = AGENTS if multi else ('policy',)
        self.x = torch.zeros(num_envs, 2 if multi else 1, dtype=dtype)
        self.ages = {a: torch.zeros(num_envs, dtype=torch.int64) for a in self.agents}
        self.indices = torch.arange(num_envs)
        self.limits = {a: 2 + i + self.indices % 3 for i, a in enumerate(self.agents)}
        self.reset_calls = self.steps = 0
        self.closed = False
        self.obs = ({'Translation': torch.zeros(num_envs, 21, dtype=dtype),
                     'Yaw': torch.zeros(num_envs, 9, dtype=dtype)} if multi else torch.zeros(num_envs, 15, dtype=dtype))
        self.states = torch.zeros(num_envs, 30, dtype=dtype) if multi else None
        self.last_actions = None

    def _refresh(self):
        for agent in self.agents:
            obs = self.obs[agent] if self.multi else self.obs
            obs.zero_()
            obs[:, :self.x.shape[1]] = self.x
            obs[:, 2] = self.ages[agent]
            obs[:, 3] = self.indices.to(self.dtype) / 10
        if self.multi:
            self.states.copy_(torch.cat([self.obs[a] for a in AGENTS], -1))
            return MultiObservation(self.obs, self.states)
        return self.obs

    def reset(self):
        if self.closed:
            raise RuntimeError("Mock is closed")
        self.reset_calls += 1
        self.x.zero_()
        for age in self.ages.values():
            age.zero_()
        return self._refresh()

    def step(self, actions):
        if self.closed:
            raise RuntimeError("Mock is closed")
        action_map = actions if self.multi else {'policy': actions}
        if set(action_map) != set(self.agents):
            raise ValueError("Wrong mock action keys")
        for a in self.agents:
            dim = 1 if a == 'Yaw' else 4
            matching(action_map[a], torch.empty(self.num_envs, dim, dtype=self.dtype), 'actions')
            action_map[a].clamp_(-0.5, 0.5)  # Deliberate mutation tests independent execution tensors.
        self.last_actions = {a: x.clone() for a, x in action_map.items()}
        translation = action_map[self.agents[0]].sum(-1)
        yaw = action_map['Yaw'].squeeze(-1) if self.multi else torch.zeros_like(translation)
        self.x[:, 0].add_(0.1 + 0.01 * translation + 0.005 * yaw)
        if self.multi:
            self.x[:, 1].add_(0.2 + 0.01 * yaw + 0.005 * translation)
        for age in self.ages.values():
            age.add_(1)
        self.steps += 1
        rewards = {a: (i + 1) - self.x[:, i].square() for i, a in enumerate(self.agents)}
        ended = {a: self.ages[a] >= self.limits[a] for a in self.agents}
        terminated = {a: ended[a] & ((self.indices + i) % 2 == 0) for i, a in enumerate(self.agents)}
        truncated = {a: ended[a] & ~terminated[a] for a in self.agents}
        self._refresh()
        final = self.states.clone() if self.multi else self.obs.clone()
        for i, a in enumerate(self.agents):
            self.x[ended[a], i] = 0  # Partial reset of this objective's component in one coupled system.
            self.ages[a][ended[a]] = 0
        observations = self._refresh()
        if self.multi:
            return MultiStepResult(observations, rewards, terminated, truncated,
                {a: final for a in AGENTS} if self.expose_final else None,
                {a: ended[a].clone() for a in AGENTS} if self.expose_final else None)
        return StepResult(observations, rewards['policy'], terminated['policy'], truncated['policy'],
                          final if self.expose_final else None, ended['policy'].clone() if self.expose_final else None)

    def state_dict(self):
        return dict(kind='mappo' if self.multi else 'ppo', num_envs=self.num_envs,
                    dtype=str(self.dtype), expose_final=self.expose_final,
                    x=self.x.clone(), ages={a: x.clone() for a, x in self.ages.items()},
                    steps=self.steps, reset_calls=self.reset_calls)

    def load_state_dict(self, state):
        signature = {k: self.state_dict()[k] for k in ('kind', 'num_envs', 'dtype', 'expose_final')}
        if any(state[k] != v for k, v in signature.items()):
            raise ValueError("Incompatible mock environment configuration")
        matching(state['x'], self.x, 'mock state')
        if set(state['ages']) != set(self.agents):
            raise ValueError("Wrong mock agent state")
        for a in self.agents:
            matching(state['ages'][a], self.ages[a], 'mock ages')
            if ((state['ages'][a] < 0) | (state['ages'][a] >= self.limits[a])).any():
                raise ValueError("Invalid mock ages")
        for name in ('steps', 'reset_calls'):
            if type(state[name]) is not int or state[name] < 0:
                raise ValueError("Invalid mock counters")
        self.x.copy_(state['x'])
        for a in self.agents:
            self.ages[a].copy_(state['ages'][a])
        self.steps, self.reset_calls = state['steps'], state['reset_calls']
        return self._refresh()

    def close(self):
        self.closed = True


class SingleAgentMock(_Mock):
    def __init__(self, num_envs=4, **kwargs):
        super().__init__(num_envs, multi=False, **kwargs)


class MultiAgentMock(_Mock):
    def __init__(self, num_envs=4, **kwargs):
        super().__init__(num_envs, multi=True, **kwargs)
