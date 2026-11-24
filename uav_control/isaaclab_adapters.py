"""Tensor-only adapters for the verified Isaac Lab 2.1.0 direct-environment API."""

import torch

from .environments import MultiObservation, MultiStepResult, StepResult


AGENTS = ("Translation", "Yaw")


class _Adapter:
    def __init__(self, environment, *, bootstrap_truncations=False):
        if bootstrap_truncations:
            raise ValueError("Isaac Lab 2.1.0 step() does not supply final pre-reset inputs; "
                             "bootstrap_truncations must be False")
        self.environment = environment
        self.native = environment.unwrapped
        self.num_envs = self.native.num_envs
        self.device = torch.device(self.native.device)
        self.dtype = torch.float32
        self.closed = False

    def _tensor(self, value, shape, *, boolean=False):
        dtype = torch.bool if boolean else self.dtype
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.dtype != dtype or value.device != self.device):
            raise ValueError(f"Native tensor must have shape {shape}, dtype {dtype}, device {self.device}")
        return value

    def _flags(self, value):
        return self._tensor(value, (self.num_envs,), boolean=True)

    def close(self):
        if not self.closed:
            self.closed = True
            self.environment.close()


class IsaacLabSARLAdapter(_Adapter):
    """Native observations are already post-reset; never reset again in step()."""
    def _observations(self, observations):
        return self._tensor(observations["policy"], (self.num_envs, 15))

    def reset(self):
        observations, _extras = self.environment.reset()
        return self._observations(observations)

    def step(self, actions):
        actions = self._tensor(actions, (self.num_envs, 4)).detach().clone()
        observations, rewards, terminated, truncated, _extras = self.environment.step(actions)
        return StepResult(self._observations(observations),
                          self._tensor(rewards, (self.num_envs,)),
                          self._flags(terminated), self._flags(truncated))


class IsaacLabMARLAdapter(_Adapter):
    """One physical step, separate rewards/flags, Translation-then-Yaw critic state."""
    def __init__(self, environment, *, bootstrap_truncations=False):
        super().__init__(environment, bootstrap_truncations=bootstrap_truncations)
        if tuple(self.native.possible_agents) != AGENTS or self.native.cfg.state_space != -1:
            raise ValueError("Expected Translation/Yaw order and native state_space = -1")

    def _mapping(self, values, dimensions=None, *, boolean=False):
        if set(values) != set(AGENTS):
            raise ValueError("Expected exactly Translation and Yaw")
        return {a: self._tensor(values[a], (self.num_envs,) if dimensions is None
                                else (self.num_envs, dimensions[a]), boolean=boolean) for a in AGENTS}

    def _observations(self, observations):
        local = self._mapping(observations, {"Translation": 21, "Yaw": 9})
        # v2.1.0 state() concatenates obs_dict in possible_agents order. The
        # constructor checks that order and state_space; no CPU round trip.
        states = self._tensor(self.native.state(), (self.num_envs, 30))
        return MultiObservation(local, states)

    def reset(self):
        observations, _extras = self.environment.reset()
        return self._observations(observations)

    def step(self, actions):
        owned = {a: v.detach().clone() for a, v in
                 self._mapping(actions, {"Translation": 4, "Yaw": 1}).items()}
        observations, rewards, terminated, truncated, _extras = self.environment.step(owned)
        # Native reset = all(terminated) OR all(truncated). Thus every physical
        # reset bounds both trajectories. Preserve individual flags, including
        # boundaries which do not reset the physical environment.
        return MultiStepResult(self._observations(observations), self._mapping(rewards),
                               self._mapping(terminated, boolean=True),
                               self._mapping(truncated, boolean=True))
