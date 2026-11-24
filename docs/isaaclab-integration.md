# Isaac Lab integration — Milestone 6

The original UAV tasks are packaged as external Isaac Lab tasks and connected to
the existing independent learners. This document records source inspection and
CPU verification, not native simulator compatibility or research results.

## Sources and migration boundary

The source revision is
[`YuHaoTsai/IsaacLab@bdd31df85cc47e07715442cdc3fe0057c798fb35`](https://github.com/YuHaoTsai/IsaacLab/tree/bdd31df85cc47e07715442cdc3fe0057c798fb35/source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter_MARL).
The original Git blobs are `f838b6de78335250323f049d1ac61c7a9f97e4b8`
(SARL) and `cca68159dfb68eb477b64f8c7fa18bcde6f46a02` (MARL).
[provenance.json](../uav_control/tasks/provenance.json) records every inspected
reference URL/blob and original class fingerprints. SARL's source lacked a final
newline; the migrated file adds one. No historical datasets or unrelated task
variants are included. Source notices and the upstream BSD-3-Clause license are
retained; they do not select a license for the rest of this project.

Exact executable task-file changes, in both files unless stated otherwise:

| Location | Change and reason |
| --- | --- |
| Imports | Route `quat_apply_inverse` to local `_compat.py`; add `write_legacy_csv`. Official 2.1.0 lacks the fork's helper. |
| Imports | Remove unused matplotlib/SciPy imports and SARL's unused `get_checkpoint_path`; remove pandas after replacing CSV serialization. NumPy remains for the original inertia matrix. |
| Configuration | Add `legacy_telemetry_dir: str \| None = None`; original task configuration fields are unchanged. |
| `_pre_physics_step` | Gate two telemetry append groups on the optional directory. Dynamics, force computations, wind sampling, RNG calls, and action interpretation are unchanged. |
| `_get_observations` | Gate noisy-drag telemetry appends. Observation computations and RNG calls are unchanged. |
| `_reset_idx` | Gate the existing environment-zero telemetry/export block. Replace nine DataFrame constructions with the same column dictionaries, and nine `to_csv` calls with the local writer. Physical resets and reward logging are unchanged. |

The CSV helper creates only the explicitly selected directory, replaces Windows
filename-invalid characters (including `:`), and writes with Python's standard
library. Legacy observation and action sample counts can differ, including on
initial reset; unequal columns are padded with empty cells rather than failing a
DataFrame construction. This does not imply sample alignment. I/O failures emit
a warning instead of stopping training. Logging remains synchronous when opted
in and can be expensive. Default training performs no legacy append/export work.

Static class fingerprints compare all original configuration/window/environment
class ASTs after excluding only the new telemetry setting/gates and the legacy
CSV reset block. This checks that the other class code remains unchanged;
it is not a proof of numerical equivalence inside a simulator.

## Quaternion and articulation compatibility

The local scripted helper preserves the fork's implementation, scalar-first
quaternions, reshape behavior, and output shape. For unit quaternions its result
matches both an independent inverse rotation matrix and the official 2.1.0
`quat_rotate_inverse` formula. For non-unit quaternions the difference
`fork - official = 2 * (1 - ||q||²) * v` is generally nonzero. Keeping the local
helper avoids silently assuming normalization beyond the original implementation.
Tests cover singleton, batched, and multidimensional inputs in float32/float64.

Official 2.1.0 articulation sources contain the task's `root_link_state_w`,
`root_link_pos_w`, `root_com_lin_vel_b`, `root_com_ang_vel_b`,
`write_root_link_pose_to_sim`, and `write_root_com_velocity_to_sim` APIs.
No framework patch or additional articulation API replacement was introduced.
Correct behavior with live physics still requires native execution.

Both task configs retain `CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")`.
The official [quadcopter asset configuration](https://github.com/isaac-sim/IsaacLab/blob/v2.1.0/source/isaaclab_assets/isaaclab_assets/robots/quadcopter.py)
references `{ISAAC_NUCLEUS_DIR}/Robots/Crazyflie/cf2x.usd`. No USD, custom asset
loader, changed physical properties, or altered robot initial configuration is
included. Asset access/caching has not been verified.

## Registration, startup, and installation

`register_tasks()` installs lazy Gymnasium entry points, is idempotent for matching
registrations, and rejects ID collisions:

| ID | Environment / configuration in `uav_control.tasks` |
| --- | --- |
| `UAV-Control-Quadcopter-SARL-v0` | `quadcopter_SARL:QuadcopterEnvSARL` / `QuadcopterEnvCfgSARL` |
| `UAV-Control-Quadcopter-MARL-v0` | `quadcopter_MARL:QuadcopterEnvMARL` / `QuadcopterEnvCfgMARL` |

The launcher parses project options and official AppLauncher arguments, starts
Isaac Sim, then imports/registers tasks and constructs the selected config. It
overrides only requested environment count, device, seed, and optional telemetry
directory. Task selection fixes the learner/adapter combination. PPO uses the
existing 15/4 network defaults; MAPPO uses existing 21/4 and 9/1 local networks
and separate 30-input critics. Checkpoints may retain configurable network widths
and learner settings but must match task dimensions, float32, and disabled timeout
bootstrapping. Models move to the requested device before optimizers are created.

`Trainer`, `evaluate`, `save_checkpoint`, `load_checkpoint`, and `resume_training`
are reused without algorithm changes. Environment and application cleanup run on
normal completion and workflow failures. Output is the existing JSON metrics.
Checkpointing is explicit via `--checkpoint-out`, at a complete update boundary;
it excludes native interaction snapshots. Resume starts fresh physical episodes.
The existing CUDA RNG/device-count constraint also applies to learner-only resume;
there is no automatic CPU-to-CUDA resume fallback.

Root `train_agent.py` and installed `uav-train` share `uav_control.launcher:main`.
No source-root path injection is used. Ordinary imports of the base package,
registration module, and launcher load no torch, Gymnasium, Isaac Lab, or simulator.
The native command itself requires the prepared simulator environment, including
for its full help because official AppLauncher defines the simulator flags.

The flat package layout follows the official external project's editable-package
and string-registration approach without copying the template's framework or
training backends. No template generator was run. The retained `torch==2.5.1`
requirement matches [Isaac Lab 2.1.0 setup.py](https://github.com/isaac-sim/IsaacLab/blob/v2.1.0/source/isaaclab/setup.py),
which also requires NumPy <2. A CUDA local-version build satisfies the public
PyTorch version pin. Use `--no-deps --no-build-isolation` with already-present
dependencies/build tools, as described in the README; this skips dependency
resolution and must not be mistaken for a prerequisite check. No new packages
were installed for this milestone, and no prepared Isaac Lab installation was
available to test editable installation or native entry-point execution.

## Adapter and boundary contracts

Both adapters are torch-only modules. Native tensors remain on the native device;
shape/dtype/device mismatches fail explicitly. Actions are detached and cloned
before a native step, so in-place task interpretation cannot overwrite stored
Gaussian samples. Adapters do not clip, normalize, scale rewards, advance physics
extra times, or manually reset episodes. Returned tensors may alias environment
memory as allowed by the existing collector contract.

| Native API | Workflow API |
| --- | --- |
| SARL `reset() -> ({"policy": [N,15]}, extras)` | `[N,15]` tensor |
| SARL five-tuple `step([N,4])` | `StepResult`, scalar raw rewards `[N]`, separate boolean termination/truncation `[N]` |
| MARL `reset() -> (agent_observations, extras)` | `MultiObservation`, local `[N,21]` / `[N,9]` and state `[N,30]` |
| MARL five-tuple `step({Translation:[N,4], Yaw:[N,1]})` | `MultiStepResult`, separate raw rewards and flags for each agent |

For `state_space=-1`, official `DirectMARLEnv.state()` concatenates `obs_dict`
in `cfg.possible_agents` order. Native `possible_agents` is initialized from that
same list. The adapter requires Translation then Yaw and calls `state()` after
reset/step. States go to critics through the existing collector; actor calls use
only local observations. Both policies share exactly one native step.

Source-inspected behavior at **v2.1.0**:

- [DirectRLEnv](https://github.com/isaac-sim/IsaacLab/blob/v2.1.0/source/isaaclab/isaaclab/envs/direct_rl_env.py)
  computes dones/rewards, resets done rows, and then computes returned observations.
- [DirectMARLEnv](https://github.com/isaac-sim/IsaacLab/blob/v2.1.0/source/isaaclab/isaaclab/envs/direct_marl_env.py)
  resets with `all(terminated) OR all(time_out)` across agents, then computes
  observations. It does not use `any(agent_done)`. The migrated task supplies the
  same physical termination/time-out masks for both agents, so physical resets
  bound both trajectories. The adapter preserves individual flags and rewards.
- Extras do not supply a guaranteed final pre-reset observation/state for these
  tasks. All workflow final-input fields remain `None`; post-reset observations
  are next-action-ready inputs, never final bootstrap inputs.
- Continuing transitions bootstrap normally; terminations and timeouts bootstrap
  zero under the existing default. Explicit adapter requests and checkpoint
  configurations enabling timeout bootstrapping fail with a clear error.

The adapter is specific to these tasks and this source contract; arbitrary patched
native reset logic or later Isaac Lab versions need a new audit.

## Verification and remaining work

Detected test environment: Windows, Python 3.10.11, torch 2.5.1+cpu, pytest 8.4.2.
Isaac Lab/Isaac Sim were not found in the project environment or inspected Conda
environments. The CPU test environment lacks NumPy, producing one existing
PyTorch warning; the numerical/stand-in tests do not require NumPy. This environment
cannot import or run the full native tasks. No simulator experiment was attempted.

Executed from the repository root:

```powershell
.venv\Scripts\python.exe -B -m pytest -q tests/test_task_sources.py tests/test_isaaclab_adapters.py tests/test_native_launcher.py
.venv\Scripts\python.exe -B -m pytest -q
git diff --check
```

Results: **36 new targeted tests passed; 224 total tests passed**, preserving all
188 prior regressions. Tests cover source/config preservation, quaternion math,
Crazyflie reference, no vendored assets, registrations, shapes, ordering, distinct
rewards/flags, action ownership, internal-reset handling, unavailable final inputs,
collector optimization, checkpoint/resume/evaluation through launcher stand-ins,
startup order, cleanup, and simulator-free imports. No `dev/` behavior or existing
learning/workflow implementation changed.

Wheel and sdist were built with already-installed bundled Python 3.12.14,
setuptools 84.2.1, and wheel 0.48.0, without installation or dependency downloads:

```python
from setuptools.build_meta import build_sdist, build_wheel
build_sdist("dist")
build_wheel("dist")
```

Archive checks verify task sources, compatibility helper, license, provenance,
launcher, console metadata, dependency pin, source launcher/docs, and exclusion
of vendored USD/frameworks. Importing the wheel and loading its console callable
from outside the source directory checks packaging independently of the root
launcher. These are artifact checks, not an installed Isaac Lab runtime test.

Milestone 7 must verify application/task startup, actual config/registration
resolution, Crazyflie USD loading, native articulation/force behavior, singleton
and multi-environment tensor shapes, resets and state ordering, CUDA PPO/MAPPO
updates, checkpoint resume/evaluation, cleanup, asset access, performance, and
Ubuntu portability. No specific static API blocker remains identified, but source
inspection cannot establish that native execution succeeds.

Preserved nonblocking research issues for later review: fixed world goal/reset
coordinates do not incorporate environment origins; SARL/MARL differ in wind
means, distance reward expressions, goal bonuses, and force limiting; original
per-environment matrix loops may limit GPU throughput. These are not revised in
this milestone. Legacy telemetry alignment and its performance remain diagnostic
limitations. The draft paper is context, not evidence for new results.

## Change inventory

Modified: `AGENTS.md`, `README.md`, `pyproject.toml`, `docs/training-workflow.md`.

Added: `MANIFEST.in`, `train_agent.py`, `uav_control/launcher.py`,
`uav_control/isaaclab_adapters.py`, `uav_control/tasks/__init__.py`,
`quadcopter_SARL.py`, `quadcopter_MARL.py`, `_compat.py`, `telemetry.py`,
`LICENSE.isaaclab`, `provenance.json` (the last six under `uav_control/tasks/`),
`tests/test_task_sources.py`, `tests/test_isaaclab_adapters.py`,
`tests/test_native_launcher.py`, and this document.

Work is on `feature/isaaclab-integration`, from clean updated main
`909676508ce40bd16c731409af9246e330621c62`. Changes remain unstaged/uncommitted
for review. Build products and temporary reference snapshots are ignored.
