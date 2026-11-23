# PPO/MAPPO training workflow

This milestone composes the existing learners with tensor environment interfaces,
persistent collectors, bounded training, checkpoint/resume, and evaluation. The
CPU fixtures under `dev/` verify software behavior; they implement no UAV physics.
Core modules import neither Gymnasium nor Isaac Lab/Sim. Dependencies and learning
algorithms are unchanged.

## Environment and collector contract

`environments.py` defines `VectorEnvironment` and `MultiAgentEnvironment` protocols.
Each provides `num_envs`, `device`, `dtype`, `reset()`, `step(actions)`, and
`close()`. All observations/rewards use the learner's float32/float64 dtype and
device; flags and validity masks are boolean. Empty/misaligned data are rejected.

| Interface | Reset / next observations | Actions | Outcomes |
| --- | --- | --- | --- |
| PPO | [N,O] tensor (UAV default O=15) | [N,A] (default A=4) | `StepResult`: observations, rewards, terminated, truncated |
| MAPPO | `MultiObservation(observations, states)` | Translation/Yaw mapping | `MultiStepResult`: next MultiObservation and per-agent rewards/flags |

For MAPPO, local observation/action dimensions default to 21/4 and 9/1, shared
states to [N,30]. States are supplied explicitly; workflow code never reconstructs
arbitrary global state. The future UAV adapter supplies Translation-then-Yaw
state order. Both agents remain present in every row. Rewards and boundaries are
preserved independently, and centralized critics never supply actor inputs.

**Every step returns observations ready for the next action.** An adapter must
perform or reconcile its native partial resets before returning. The workflow
does not call `reset()` on episode boundaries, introduce an extra physics step,
or infer Isaac Lab behavior from Isaac Gym. If a system reset invalidates another
agent's trajectory, its boundary must also be represented by the adapter.

`StepResult.final_observations` and `final_mask` optionally describe final
pre-reset inputs, shaped [N,O] and [N]. MAPPO uses `final_states` and `final_masks`
keyed by agent, shaped [N,S] and [N]. Invalid/unneeded rows may be NaN. For an
enabled nonzero-discount timeout bootstrap, every truncated-but-not-terminated
row must have valid final input. Missing inputs raise an error; post-reset
observations are never substituted. Termination takes precedence, and no final
input is required when timeout bootstrapping is disabled or gamma is zero.

`PPOCollector.collect()` / `MAPPOCollector.collect()` return `(buffer, episodes)`.
They initialize once, keep owned current observations/states across horizons,
sample raw policy actions/log probabilities, and send independent action copies
to the environment. Environment-side action mutation cannot alter stored policy
samples. Only continuing rows receive next-state value estimates; only eligible
timeout rows receive final-state values. All values use scaled-reward units,
while collected rewards remain raw. Existing learners scale rewards once and
freeze normalizers through collection and optimization.

Completed `Episode` records contain agent, environment row, raw reward sum, and
length. Incomplete episode totals persist across rollout boundaries. Tensor data
stay on-device; completed episode summaries and existing learner diagnostics
transfer small scalar data to the host. A step failure marks the session failed:
the environment may already have advanced, so retrying the partial rollout is
not supported.

## Training and evaluation lifecycle

```python
from uav_control.training import Trainer, seed_everything
from uav_control.checkpoints import save_checkpoint

seed_everything(42)  # before constructing learner and environment
# learner and environment are supplied by application code
with Trainer(learner, environment) as trainer:
    reports = trainer.run(updates=2)
    save_checkpoint("outputs/run.pt", trainer, include_environment=False)
```

`run(updates)` adds complete collect/prepare/optimize cycles. Repeated calls
retain the environment and episode meters. Reports contain progress, per-agent
optimization diagnostics, and raw completed-episode mean rewards/lengths for
that update (None if no episode completed). Context exit or training failure
closes the environment; save checkpoints before closing. The caller should use
a separate environment for evaluation.

Counters cover successfully completed updates: `updates`, `vector_steps`, and
`transitions`. A horizon T adds T vector steps and T*N system transitions for
both PPO and MAPPO, with no agent-count multiplier. The CLI uses update counts;
it does not reinterpret the historical `timesteps: 38400` as 38,400 individual
environment transitions. Seeding covers Python and PyTorch; CPU continuation is
tested, universal GPU determinism is not claimed.

`evaluate(learner, environment, max_steps=..., max_episodes=None)` defaults to
deterministic actor-only inference under `torch.no_grad()`. It never calls a
critic, optimizer, or normalizer update. Partial resets use the same contract.
Results contain steps/transitions and per-agent completed-episode count, mean
raw return, and mean length. `max_steps` is always finite; an optional episode
limit counts agent episodes and may be exceeded by simultaneous completions
within one vector step. Incomplete episodes are excluded from means. Evaluation
always closes its environment, including on failure.

## Checkpoint contents and resume

`save_checkpoint(path, trainer, include_environment=False)` writes format version
1 at a complete update boundary, using a temporary file and atomic replacement.
It stores plain configuration dictionaries, normalization-semantics version,
network/Adam state dictionaries (including current learning rates), all enabled
normalizer moments, training counters, Python/torch CPU RNGs, and initialized
CUDA RNG states. MAPPO stores each agent independently. Tensors are copied to
CPU; no environment object or executable custom class is serialized.

`load_checkpoint(path, device="cpu", expected_learner=None)` creates a new learner
from validated configurations, loads on CPU, then places model/optimizer state
on the requested device. It validates keys, model tensor shapes/dtypes, Adam
moments/settings, normalizers, counters, and RNGs. An optional expected learner
enforces configuration equality. Loading leaves the caller's CPU RNG unchanged;
resumption restores saved RNGs after constructing/restoring interaction.

Loading explicitly uses `weights_only=True`, rejects legacy and TorchScript
archives, and never falls back to unrestricted loading. Only trusted local
project checkpoints are supported: restricted loading in the pinned PyTorch
2.5.1 is not a security boundary for hostile files. See the upstream
[serialization guidance](https://docs.pytorch.org/docs/stable/notes/serialization)
and [affected-version advisory](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6).
No package upgrade is performed by this workflow.

`resume_training(loaded, environment, exact=False)` restores learner state,
historical counters, and RNGs, then starts a fresh environment on first collection.
It does not claim trajectory continuation. New environment counts are allowed;
subsequent transitions add the new N at each vector step.

With `include_environment=True`, the environment must implement data-only
`state_dict()` and `load_state_dict(state) -> current_observation`. The checkpoint
also stores current observations/states, partial episode returns/lengths, and
session step count. `resume_training(..., exact=True)` requires a compatible
snapshot/device/environment and restores it without another reset. Restored
observations must equal the saved interaction snapshot. Exact continuation is
verified for CPU mocks at successful update boundaries, with no pending rollout
or batch. There is no mid-update recovery or automatic rollback after failures.
Future Isaac Lab simulator/RNG restoration needs independent implementation and
validation. Cross-device loading is explicit; exact cross-device continuation is
rejected. CUDA resume requires matching CUDA RNG state/device count and remains
untested.

## Development mocks and commands

`dev/mock_environments.py` contains CPU fixtures with predictable additive tensor
transitions, raw rewards, staggered per-row terminations/timeouts, partial resets,
and optional final inputs. Step deliberately clips supplied actions in-place to
test ownership. MAPPO has one two-component system coupled to both agents'
actions, separate local views/rewards, and component-specific episode boundaries.
It is not a pair of unrelated environments and does not approximate UAV physics.
Snapshots restore its state, ages, and counters. `dev/` is excluded from package
discovery; production collectors do not import it.

Run `python -m dev.mock_workflow --help` from the repository root. Options select
PPO/MAPPO, environment count, seed, additional updates, checkpoint output,
resume/exact resume, or finite evaluation. See the [README commands](../README.md#mock-workflow-verification).
CLI output is JSON tagged `mock_only`; generated checkpoints belong under the
ignored `outputs/` directory. The CLI uses the existing CPU environment and
does not install packages or initialize a simulator.

## Verification boundary

The complete suite runs with `.venv\Scripts\python.exe -B -m pytest -q` on Windows,
Python 3.10.11, torch 2.5.1+cpu, pytest 8.4.2: **188 tests passed**, including all
151 existing tests and 37 new workflow tests. Tests cover action/input ownership,
persistent collection, individual boundaries/rewards, required final states,
analytic timeout values, normalization, successive updates, checkpoint round
trips/rejection, exact resume, frozen deterministic evaluation, and both CLI
workflows. Resume checks compare actual model parameters, Adam moments/LRs,
normalizers, counters, episode meters, subsequent observations, reports, and RNGs.
All 151 existing PPO/MAPPO tests remain regression requirements. Optional NumPy
is absent and generates a PyTorch warning; no test needs it.

No existing learning algorithm was changed. Real task migration, Isaac Lab
adapters/launchers, native reset/final-state semantics, CUDA execution, Ubuntu,
and UAV research performance remain unvalidated.
