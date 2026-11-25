# Project guidance

## Purpose and architecture

- Build independent PyTorch PPO/MAPPO for the original UAV research tasks using
  Isaac Lab's GPU-parallel simulation. MARL means Translation and Yaw policies
  controlling one quadcopter, with local actor inputs and centralized critics.
- Distribution: `multi-agent-uav-control`; import package: `uav_control`.
  Keep the flat package layout and root `pyproject.toml`; add focused modules
  only when needed. Do not depend on skrl, RL-Games, or RSL-RL.
- Target Python 3.10, Isaac Lab 2.1.0, and Isaac Sim 4.5.0 on Windows 11;
  Ubuntu is a secondary target. Installation and usage belong in `README.md`.
- Keep learning algorithms, rollout collection, environment adapters, and
  training/evaluation lifecycle separate. Ordinary package imports must not
  initialize a simulator. Isaac Lab/Sim own physics and simulation parallelism.

## Native integration and learning contracts

- Use the official AppLauncher before importing task implementations or
  registering tasks. Root `train_agent.py` and `uav-train` share one launcher;
  SARL selects PPO and MARL selects Translation/Yaw MAPPO.
- Preserve tensor shapes, dtype, device placement, and batch semantics in
  adapters. Isaac Lab 2.1.0 resets internally before returning observations;
  do not add adapter resets or extra physics steps. Native MARL resets when
  all agents terminate OR all agents time out. Preserve individual rewards and
  boundaries; centralized state order is Translation then Yaw.
- Store raw Gaussian samples and old log probabilities; execution actions must
  be independently owned. Collect raw observations and individual raw rewards,
  scale rewards once in the learner, freeze normalizers through collection and
  optimization, then update statistics once for the next rollout.
- Native timeout bootstrapping stays disabled: final pre-reset observations and
  states are unavailable. Reject incompatible requests/checkpoints; never use
  post-reset inputs as terminal values. Native resume restores learner state
  into a fresh environment, not an exact simulator trajectory.
- Preserve the contracts in `docs/ppo-core.md`, `docs/ppo-learner.md`,
  `docs/mappo-learner.md`, `docs/training-workflow.md`, and
  `docs/isaaclab-integration.md` when changing related code.

## Research sources and task preservation

- Original tasks come from
  [YuHaoTsai/IsaacLab](https://github.com/YuHaoTsai/IsaacLab/tree/bdd31df85cc47e07715442cdc3fe0057c798fb35/source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter_MARL).
  Preserve `uav_control/tasks/provenance.json`, its source fingerprints, original
  copyright notices, and `LICENSE.isaaclab`. Record revisions when adapting
  upstream code; review the original configurations and relevant learning
  semantics rather than importing another framework's conventions.
- Preserve task class names, configurations, physics, observations, actions,
  rewards, wind, resets, visualization, and scenarios unless a change is
  explicitly requested. Report inherited issues before changing behavior.
- Retain `CRAZYFLIE_CFG.replace(...)` and Isaac Lab asset resolution. Do not
  vendor USD assets or patch installed NVIDIA framework sources. Keep required
  fork compatibility local, including `quat_apply_inverse()`.
- The draft paper provides research context, not an immutable implementation
  specification. New performance and reproducibility claims need evidence from
  this implementation. Keep historical datasets and unrelated fork content out
  of the repository. Do not select a root project license without user review.

## Verification and work boundaries

- Use the existing CPU `.venv` for numerical tests and native-interface stand-ins.
  Preserve `dev/` mock workflows and the full regression suite. Check GAE
  boundaries, likelihoods/clipping, normalization, storage ownership, per-agent
  routing, resets/truncations, and checkpoint state when changing those areas.
- Separate CPU software verification from native simulation validation. Report
  commands, results, and untested parts; mocks do not establish Isaac Lab/CUDA
  execution, asset loading, performance, or Ubuntu compatibility.
- Request approval before installing packages. Do not modify existing Python or
  Conda environments, simulator installations, CUDA, drivers, or system settings
  without explicit authorization. Preserve prepared PyTorch/CUDA builds through
  prerequisite checks and `--no-deps --no-build-isolation` package installation.
  Do not run the template generator or simulator experiments unless requested.
- Inspect Git status before editing and preserve unrelated user work. Use the
  existing repository/remote and a focused branch; update main only through a
  fast-forward-only workflow. Do not create nested repositories or delete local
  environments, checkpoints, datasets, or research outputs during cleanup.
- Stop after implementation, verification, and reporting. Leave changes
  uncommitted for review; do not commit, push, open a pull request, or merge
  changes automatically.
