# Multi-Agent UAV Control Project

Independent PyTorch PPO and MAPPO for one quadcopter under wind disturbances.
Isaac Lab supplies physics and GPU-parallel simulation; the project supplies
the original UAV tasks, tensor adapters, learning algorithms, and training workflow.

**Status:** Milestone 6 implements task migration and native launch orchestration.
CPU numerical and stand-in integration tests pass. Native Isaac Lab execution,
Crazyflie asset loading, CUDA training, Ubuntu, and research performance remain
unverified. The next milestone is real simulator validation.

## Architecture

- `uav_control/tasks/`: original SARL/MARL task definitions and minimal compatibility helpers.
- `uav_control/isaaclab_adapters.py`: native tensor interfaces and reset semantics.
- `uav_control/`: simulator-independent PPO/MAPPO, collectors, checkpointing, and evaluation.
- `train_agent.py`: unified native launcher; installed equivalent: `uav-train`.
- `dev/`: CPU mock workflow for software verification.

| Selection | Learner | Local observations / actions | Gymnasium ID |
| --- | --- | --- | --- |
| SARL | PPO | 15 / 4 | `UAV-Control-Quadcopter-SARL-v0` |
| MARL | MAPPO | Translation 21 / 4; Yaw 9 / 1 | `UAV-Control-Quadcopter-MARL-v0` |

MARL means two policies controlling **one quadcopter**. Separate critics receive
the 30-dimensional Translation-then-Yaw state. Actors use only their assigned
observations. Importing `uav_control` does not load a simulator.

## Prepared Isaac Lab environment

Initial target: **Windows 11, Python 3.10, Isaac Lab 2.1.0, Isaac Sim 4.5.0**,
with the GPU driver and CUDA-enabled PyTorch build required by that installation.
Ubuntu is a secondary target. Prepare the simulator separately using NVIDIA's
[versioned installation guide](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/pip_installation.html).
This repository is an external package; it does not modify NVIDIA's framework.

Both tasks retain `isaaclab_assets.CRAZYFLIE_CFG.replace(...)`. Isaac Lab resolves
`{ISAAC_NUCLEUS_DIR}/Robots/Crazyflie/cf2x.usd`; that asset must be accessible
in the prepared environment. No robot USD is bundled. Package availability alone
does not establish that the asset is reachable or cached.

## Install this package

Use the **Python interpreter prepared for Isaac Lab**, not the repository's CPU
test `.venv`. From the repository root, inspect prerequisites before installing:

```console
python -c "import sys, torch, numpy, gymnasium; from importlib.metadata import version; print(sys.version); print('Isaac Lab', version('isaaclab')); print('torch', torch.__version__, 'CUDA build', torch.version.cuda, 'available', torch.cuda.is_available()); print('NumPy', numpy.__version__, 'Gymnasium', gymnasium.__version__); print('setuptools', version('setuptools'), 'wheel', version('wheel'))"
python -m pip install --no-deps --no-build-isolation -e .
```

Check Python 3.10, Isaac Lab 2.1.0, PyTorch 2.5.1 with the prepared CUDA build,
NumPy <2, Gymnasium, and existing setuptools >=64 / wheel. Confirm Isaac Sim
4.5.0 in the simulator installation as well. The tasks need NumPy; pandas,
matplotlib, and SciPy are not required by the migrated task code.

The PyTorch pin matches Isaac Lab 2.1.0's own requirement. `--no-deps` prevents
pip from replacing the prepared PyTorch/CUDA stack; it also skips all dependency
installation and validation. `--no-build-isolation` uses existing build tools.
Resolve missing prerequisites separately before installation. These installation
commands are documented for the prepared environment and have **not been executed
there**. Wheel and source-package builds were checked without installing packages.

For a binary Isaac Lab setup, invoke its Python through the external checkout:
`C:\path\to\IsaacLab\isaaclab.bat -p train_agent.py ...` on Windows, or
`/path/to/IsaacLab/isaaclab.sh -p train_agent.py ...` on Ubuntu. Use that same
`-p -m pip ...` interpreter for installation. In an activated compatible Python
environment, the `python` commands below are sufficient.

## Native training and evaluation

These are **target commands, not runtime-verified simulator runs**:

```console
python train_agent.py --task SARL --headless --num_envs 64 --updates 10 --checkpoint-out outputs/sarl.pt
python train_agent.py --task MARL --headless --num_envs 64 --updates 10 --checkpoint-out outputs/marl.pt
python train_agent.py --task MARL --headless --num_envs 64 --updates 10 --checkpoint outputs/marl.pt --checkpoint-out outputs/marl-resumed.pt
python train_agent.py --task MARL --mode evaluate --checkpoint outputs/marl.pt --num_envs 64 --eval-steps 1000 --headless
```

`--task` selects the environment, adapter, learner, and existing research defaults.
`--updates` counts additional complete updates. `--seed` defaults to 0.
Omitting `--num_envs` preserves the original task's 4096 environments.
Official AppLauncher options, including `--device cuda:0`, are also available.

After installation, use `uav-train` with the same arguments from any directory.
It invokes the same implementation as `train_agent.py`; task imports and
registration occur after AppLauncher initializes Isaac Sim. Relative checkpoint
paths resolve from the current directory. Use trusted project checkpoints.

Resume restores learner state, counters, and RNGs into a fresh native environment;
it does not restore simulator trajectories. CUDA resume requires saved CUDA RNG
states matching the available device count; CPU mock checkpoints cannot directly
resume CUDA training under the current workflow. Evaluation uses frozen actors.
Time-limit bootstrapping is disabled; incompatible checkpoints are rejected
because native step outputs lack final pre-reset observations/states.

Legacy CSV logging is off by default. To opt in, specify
`--legacy-telemetry-dir outputs/legacy`. It is synchronous diagnostic output,
with Windows-safe filenames; it is not a validated research data export.

## Mock workflow verification

These commands use the existing CPU test `.venv` and run no simulator:

```powershell
.venv\Scripts\python.exe -B -m pytest -q
.venv\Scripts\python.exe -m dev.mock_workflow --algorithm ppo --num-envs 4 --updates 2 --checkpoint outputs/mock/ppo.pt
.venv\Scripts\python.exe -m dev.mock_workflow --algorithm ppo --num-envs 4 --updates 1 --resume outputs/mock/ppo.pt --exact-resume
.venv\Scripts\python.exe -m dev.mock_workflow --algorithm mappo --num-envs 4 --updates 2 --checkpoint outputs/mock/mappo.pt
.venv\Scripts\python.exe -m dev.mock_workflow --algorithm mappo --evaluate --resume outputs/mock/mappo.pt --eval-steps 20
```

See [integration contracts and verification](docs/isaaclab-integration.md),
[training workflow](docs/training-workflow.md), [PPO learner](docs/ppo-learner.md),
[MAPPO learner](docs/mappo-learner.md), and [numerical foundations](docs/ppo-core.md).
The draft paper supplies research context; new performance claims require new
experiments. [AGENTS.md](AGENTS.md) records development boundaries.

Project license selection remains pending. Migrated NVIDIA-derived sources
retain their notices and [upstream BSD-3-Clause terms](uav_control/tasks/LICENSE.isaaclab).
