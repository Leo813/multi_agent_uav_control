"""Tensor-only vector environment contracts; adapters own reset/execution semantics."""

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class MultiObservation:
    observations: dict[str, torch.Tensor]
    states: torch.Tensor


@dataclass(frozen=True)
class StepResult:
    observations: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    final_observations: torch.Tensor | None = None
    final_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class MultiStepResult:
    observations: MultiObservation
    rewards: dict[str, torch.Tensor]
    terminated: dict[str, torch.Tensor]
    truncated: dict[str, torch.Tensor]
    final_states: dict[str, torch.Tensor] | None = None
    final_masks: dict[str, torch.Tensor] | None = None


class VectorEnvironment(Protocol):
    """step() returns next-action-ready observations, including partial resets.

    Returned tensors may alias environment memory. Actions may be mutated by the
    adapter. Flags describe the action just executed. Final inputs have full batch
    shape and a boolean validity mask; only valid rows need be finite. The adapter
    must resolve its native autoreset timing before returning. No extra reset or
    physics step is inferred by the workflow.
    """
    num_envs: int
    device: torch.device
    dtype: torch.dtype

    def reset(self) -> torch.Tensor: ...
    def step(self, actions: torch.Tensor) -> StepResult: ...
    def close(self) -> None: ...


class MultiAgentEnvironment(Protocol):
    """Same timing contract, with individual flags/rewards and explicit shared state.

    Both Translation and Yaw remain present in all rows. If a physical reset
    invalidates another agent's trajectory, the adapter must mark that boundary
    too. Final centralized states and validity masks are indexed by agent.
    """
    num_envs: int
    device: torch.device
    dtype: torch.dtype

    def reset(self) -> MultiObservation: ...
    def step(self, actions: dict[str, torch.Tensor]) -> MultiStepResult: ...
    def close(self) -> None: ...
