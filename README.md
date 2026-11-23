# Multi-Agent UAV Control Project

Independent PyTorch PPO and MAPPO for quadcopter navigation and stable hovering
under wind disturbances, using NVIDIA Isaac Lab for GPU-parallel simulation.

**Status:** Independent PPO and two-agent MAPPO support persistent rollout
collection, bounded training, checkpoint/resume, and deterministic evaluation.
CPU mock environments verify the workflow end to end. MAPPO retains separate
Translation/Yaw actors and centralized critics. UAV task migration, Isaac Lab
integration, CUDA execution, and research performance remain unvalidated.

See [PPO learner](docs/ppo-learner.md) and [MAPPO learner](docs/mappo-learner.md)
for configuration, collection/update interfaces, normalization choices, and
validation results. See [training workflow](docs/training-workflow.md) for the
environment contract, checkpoint semantics, and mock verification. The underlying
[PPO mathematical foundations](docs/ppo-core.md) specify action likelihoods,
boundary semantics, and loss conventions.

## Research objective

The project will compare two ways to control one quadcopter:

- **SARL / PPO:** one policy commands collective thrust and three control moments.
- **MARL / MAPPO:** separate Translation and Yaw policies cooperate on the same
  quadcopter, with centralized value estimation during training and decentralized
  policy execution.

The original task definitions and skrl configurations in
[YuHaoTsai/IsaacLab](https://github.com/YuHaoTsai/IsaacLab/tree/main/source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter_MARL)
are the starting references, especially `quadcopter_SARL.py` and
`quadcopter_MARL.py`. The draft paper provides research context; new training and
evaluation results will be generated for this implementation.

## Intended architecture

- **Learning:** simulator-independent PyTorch modules for configurable networks,
  Gaussian policies, PPO/MAPPO updates, GAE, rollout storage, normalization, and
  learning-rate schedules.
- **Integration:** a small adapter for Isaac Lab's `DirectRLEnv` and
  `DirectMARLEnv` tensor interfaces, including per-agent observations, centralized
  critic inputs, termination, truncation, and reset behavior.
- **Workflow:** training, checkpoint/resume, and evaluation procedures composed
  around the learning modules and environment adapter.

Isaac Lab and Isaac Sim own physics and simulation parallelism. The learning
code will not depend on skrl, RL-Games, or RSL-RL. Existing UAV task behavior will
be preserved, with changes limited to necessary integration and runtime fixes.

This separation follows the broad approach in
[hexapod_proj](https://github.com/Leo813/hexapod_proj/blob/main/docs/ppo-implementation.md),
with interfaces and learning behavior designed for Isaac Lab and the UAV tasks.

## Package and compatibility target

- Distribution: `multi-agent-uav-control`; Python import: `uav_control`.
- Python 3.10 or newer for the package foundation.
- Initial simulation target: Isaac Lab 2.1.0, Isaac Sim 4.5.0, and Python 3.10.
- Primary development platform: Windows 11; secondary target: Ubuntu.

Install Isaac Lab and Isaac Sim separately when simulation work begins, following
the [versioned installation guide](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/pip_installation.html).
This repository is an external project and does not modify NVIDIA's framework.
The official external-project template is a compatibility reference; this
package uses a flat layout and a root-level `pyproject.toml`.

The learning modules require PyTorch 2.5.1. Install the appropriate PyTorch
build first; CPU-only PyTorch is sufficient for the numerical tests. An editable
installation with test dependencies, in a separately prepared environment, uses:

```console
python -m pip install -e ".[test]"
```

This installs the package, not the Isaac stack. The build backend is setuptools.
Run the numerical tests from the repository root with `python -m pytest -q`.
From the repository root, a dependency-free source import check is:

```console
python -B -c "import uav_control; print(uav_control.__file__)"
```

## Development sequence

1. Establish the package foundation.
2. Implement and numerically verify independent PPO components.
3. Compose and verify the single-agent PPO learner.
4. Extend to two-policy MAPPO with centralized value estimation.
5. Verify training and evaluation with lightweight vectorized mock environments (current milestone).
6. Integrate the existing UAV tasks through minimal compatibility code.
7. Run real Isaac Lab simulation, CUDA training, and evaluation.
8. Review task refinements and experimental procedures.

Static checks, deterministic numerical tests, and mock-environment tests precede
simulator testing. Passing software tests is not evidence of simulation
compatibility or research performance. See [AGENTS.md](AGENTS.md) for development
constraints and verification expectations.

License selection is pending review.

## Mock workflow verification

These CPU commands run from the repository root using the existing `.venv`.
They are **not Isaac Lab UAV training commands**. Use `--help` for options.

```powershell
.venv\Scripts\python.exe -m dev.mock_workflow --algorithm ppo --num-envs 4 --updates 2 --checkpoint outputs/mock/ppo.pt
.venv\Scripts\python.exe -m dev.mock_workflow --algorithm ppo --num-envs 4 --updates 1 --resume outputs/mock/ppo.pt --exact-resume
.venv\Scripts\python.exe -m dev.mock_workflow --algorithm ppo --evaluate --resume outputs/mock/ppo.pt --eval-steps 20
.venv\Scripts\python.exe -m dev.mock_workflow --algorithm mappo --num-envs 4 --updates 2 --checkpoint outputs/mock/mappo.pt
.venv\Scripts\python.exe -m dev.mock_workflow --algorithm mappo --evaluate --resume outputs/mock/mappo.pt --eval-steps 20
```

`--updates` adds complete updates. Output reports vector steps and transitions;
one vector step with N environments is N transitions, including MAPPO.
`--exact-resume` restores mock interaction state and requires matching environment
configuration. Without it, a restored learner starts a fresh environment while
retaining training counters. Load only trusted project-generated checkpoints.
Generated outputs are ignored by Git; `dev/` is not an installed simulation dependency.
