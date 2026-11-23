"""Bounded synchronous training lifecycle; no simulator imports or factories."""

from dataclasses import asdict, dataclass
import random

import torch

from .collection import MAPPOCollector, PPOCollector
from .mappo import MAPPOLearner


def seed_everything(seed: int):
    """Seed before constructing models/environments; CPU repeatability is tested."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Progress:
    updates: int = 0
    vector_steps: int = 0
    transitions: int = 0

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in asdict(self).values()):
            raise ValueError("Training counters must be nonnegative integers")


def episode_summary(episodes, agents):
    result = {}
    for agent in agents:
        completed = [e for e in episodes if e.agent == agent]
        count = len(completed)
        result[agent] = dict(episodes=count,
                            mean_reward=sum(e.reward for e in completed) / count if count else None,
                            mean_length=sum(e.length for e in completed) / count if count else None)
    return result


class Trainer:
    """Own an environment until close()/context exit; run() adds complete updates.

    One vector step represents num_envs system transitions, also for MAPPO (not
    twice as many). Counters cover completed updates; failed updates cannot be
    checkpointed or retried on this Trainer. Repeated run() calls retain state.
    """

    def __init__(self, learner, environment):
        self.learner = learner
        try:
            self.collector = (MAPPOCollector if isinstance(learner, MAPPOLearner) else PPOCollector)(learner, environment)
        except Exception:
            environment.close()
            raise
        self.progress = Progress()
        self.at_boundary = True

    def run(self, updates: int):
        if type(updates) is not int or updates < 0:
            raise ValueError("updates must be a nonnegative integer")
        if not self.at_boundary or self.collector.closed or self.collector.failed:
            raise RuntimeError("Trainer is closed or failed")
        history = []
        try:
            for _ in range(updates):
                self.at_boundary = False
                buffer, episodes = self.collector.collect()
                diagnostics = self.learner.update(buffer)
                self.progress.updates += 1
                self.progress.vector_steps += buffer.horizon
                self.progress.transitions += buffer.horizon * buffer.num_envs
                self.at_boundary = True
                metrics = ({a: asdict(d) for a, d in diagnostics.items()} if isinstance(diagnostics, dict)
                           else {'policy': asdict(diagnostics)})
                history.append(dict(progress=asdict(self.progress), optimization=metrics,
                                    rollout=episode_summary(episodes, self.collector.agents)))
            return history
        except Exception:
            self.close()
            raise

    def close(self):
        self.collector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
