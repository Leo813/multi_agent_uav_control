# Project guidance

## Objective and identity

Build an independent PyTorch PPO/MAPPO training system for the existing UAV
research tasks using Isaac Lab's GPU-parallel simulation. Replacing the skrl
training pipeline is the primary work; task refactoring is secondary.

- Project: Multi-Agent UAV Control Project.
- Repository: https://github.com/Leo813/multi_agent_uav_control
- Distribution: `multi-agent-uav-control`; import package: `uav_control`.
- Keep the flat package layout and root-level `pyproject.toml`. Prefer focused
  modules; add subpackages, scripts, configurations, and tests only when needed.
- Primary platform: Windows 11; secondary target: Ubuntu.
- Initial simulator target: Isaac Lab 2.1.0 with Isaac Sim 4.5.0 and Python 3.10.

## Architecture

- Keep core learning modules free of Isaac Lab, Isaac Sim, and simulator startup
  imports. Importing `uav_control` must not initialize a simulator.
- Use PyTorch directly for networks, Gaussian distributions, PPO optimization,
  GAE, batched rollout storage, normalization, and schedules. Do not depend on
  skrl, RL-Games, or RSL-RL, or recreate a general-purpose RL framework.
- Keep the learning algorithm separate from rollout collection, environment
  integration, and training/evaluation lifecycle management.
- Use a minimal adapter for `DirectRLEnv` and `DirectMARLEnv`. Preserve tensor
  shapes, device placement, and batch semantics, including CUDA tensors.
- MARL means two policies controlling one quadcopter: Translation and Yaw. Support
  their different observation/action dimensions and centralized critic inputs;
  decentralized actors must use only their assigned observations.
- Specify action sampling versus executed actions, normalization updates,
  termination/truncation masks, final observations, and autoreset behavior before
  implementing the corresponding learning logic. Do not inherit Isaac Gym's
  timeout or reset conventions without checking Isaac Lab 2.1.0.
- Isaac Lab and Isaac Sim own physics, articulation management, and simulation
  parallelism. Do not implement a replacement simulator or CUDA runtime.

## Reference material and task preservation

The primary task references are `quadcopter_SARL.py` and `quadcopter_MARL.py` in:
https://github.com/YuHaoTsai/IsaacLab/tree/main/source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter_MARL

Use the original task configurations, skrl configurations, and relevant skrl
behavior as references for essential learning semantics. Inspect and record the
upstream revision when adapting code. Preserve applicable copyright and license
notices. Do not choose a final project license without user review.

The draft paper explains the research objective and methodology; it is not an
immutable implementation specification. New results must come from the new
implementation. Do not claim performance or reproducibility without evidence.

Preserve task definitions as closely as possible. Limit migration changes to
external packaging/registration, official API compatibility, removal of framework
patch dependencies, trainer integration, and essential runtime/platform fixes.
Support required behavior such as the fork's `quat_apply_inverse()` locally or
through verified official APIs; never patch the installed Isaac Lab framework.

Do not proactively change rewards, observations, actions, wind models, or training
conditions. Record nonblocking issues for later review, including discrepancies
between the paper and code. Keep historical wind datasets, alternative unfinished
architectures, notebook checkpoints, and unrelated fork content out of this repo.

Architectural reference: https://github.com/Leo813/hexapod_proj
Inspect `src/hexapod_rl/`, `train_agent.py`, and `docs/ppo-implementation.md` for
algorithm/simulator separation, adapters, workflow, and logical verification.
Do not copy its Isaac Gym interfaces, hyperparameters, or RL-Games-specific
learning conventions into this Isaac Lab project without justification.

## Incremental development and verification

Work in reviewable milestones:

1. Package and repository foundation.
2. Independent PPO components with deterministic numerical checks.
3. Single-agent PPO learner with synthetic rollout verification.
4. Two-policy MAPPO with centralized value estimation.
5. Mock vectorized-environment training, checkpoint/resume, and evaluation.
6. Minimal integration of the existing UAV tasks.
7. Real simulator, CUDA training, and evaluation validation.
8. Task refinements and experimental procedures.

Isaac Lab/Isaac Sim absence does not block simulator-independent development.
Use static checks, deterministic numerical unit tests, and lightweight mock
environments. Test GAE boundaries, policy likelihoods and clipping, normalization,
buffer ownership, per-agent routing, resets/truncations, and checkpoint state as
those components are introduced. Keep network sizes and hyperparameters
configurable. Do not add empty modules for every future component.

Separate software verification from real simulation validation in every report.
Do not claim Isaac Lab integration, CUDA execution, or training works until the
corresponding execution has been tested. Report commands, results, and untested
parts; keep documentation concise and focused on implemented functionality and
the next milestone.

## Current PPO learner milestone and review boundaries

Milestone 3 composes the existing Gaussian distributions, GAE/returns, and PPO
losses into a single-agent learner. Networks, rollout storage, running statistics,
and optimization stay in focused modules directly under `uav_control/`. Validate
with deterministic numerical and synthetic rollout tests. Do not add adapters,
environment-driven training loops, MAPPO, configuration pipelines, checkpointing,
simulation evaluation, or migrate tasks during this milestone.

Request approval before installing packages. Use an ignored repository-local
`.venv` for CPU tests; never modify existing Python/Conda environments, CUDA,
drivers, or system settings. Do not install simulators or run the official
template generator for this milestone. Use the template as a compatibility
reference, not a required directory layout.

Document raw policy actions separately from executed actions. Keep GAE boundary
semantics explicit and require final pre-reset values for optional truncation
bootstrapping. Follow the documented skrl prediction-clipping value loss rather
than importing the Hexapod implementation's maximum-of-two-errors convention.
The public contracts and reference revisions are in `docs/ppo-core.md` and
`docs/ppo-learner.md`. Collect raw observations, raw action samples, and raw
rewards; the learner scales rewards once. Both normalizers stay frozen through
collection and optimization, then update once for the next rollout. Prepared
targets remain fixed; preserve the documented differences from skrl when changing
normalization or minibatch behavior.

Implement this milestone on `feature/ppo-learner`, created from clean, updated `main`.
Stop after implementation, verification, and reporting; leave changes uncommitted
for review.

Use the existing Git repository and remote. Do not create a nested repository,
vendor Isaac Lab, or modify its installation. Leave commits, pushes, and pull
requests for explicit user authorization.
