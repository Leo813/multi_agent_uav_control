"""Finite actor-only evaluation with frozen learner state."""

import torch

from .collection import EnvironmentSession
from .training import episode_summary


@torch.no_grad()
def evaluate(learner, environment, *, max_steps: int, max_episodes: int | None = None,
             deterministic: bool = True):
    """Own/close a separate evaluation environment, including on errors.

    max_episodes counts completed agent episodes (both agents for MAPPO). A
    vector step may finish several episodes and overshoot that limit. max_steps
    always bounds execution, even if no episodes end. Incomplete episodes are
    not included in means. Evaluation never calls a critic or an optimizer.
    """
    session = None
    try:
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if max_episodes is not None and (type(max_episodes) is not int or max_episodes <= 0):
            raise ValueError("max_episodes must be a positive integer or None")
        if type(deterministic) is not bool:
            raise TypeError("deterministic must be bool")
        session = EnvironmentSession(learner, environment)
        episodes = []
        for _ in range(max_steps):
            _, completed = session.step(deterministic=deterministic)
            episodes.extend(completed)
            if max_episodes is not None and len(episodes) >= max_episodes:
                break
        return dict(vector_steps=session.vector_steps, transitions=session.vector_steps * session.num_envs,
                    agents=episode_summary(episodes, session.agents))
    finally:
        if session is not None:
            session.close()
        else:
            environment.close()
