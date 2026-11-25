# Isaac Lab integration

The external UAV tasks target **Isaac Lab 2.1.0, Isaac Sim 4.5.0, and Python 3.10**.
Compatibility has been inspected at source level and checked with CPU tensor
tests and native-interface stand-ins. Native simulator execution remains
unverified. See the [README](../README.md) for prerequisites, installation, and
training/evaluation commands.

## Sources and compatibility

The original research tasks are
[`quadcopter_SARL.py`](https://github.com/YuHaoTsai/IsaacLab/blob/bdd31df85cc47e07715442cdc3fe0057c798fb35/source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter_MARL/quadcopter_SARL.py)
and
[`quadcopter_MARL.py`](https://github.com/YuHaoTsai/IsaacLab/blob/bdd31df85cc47e07715442cdc3fe0057c798fb35/source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter_MARL/quadcopter_MARL.py)
at revision `bdd31df85cc47e07715442cdc3fe0057c798fb35`.
[provenance.json](../uav_control/tasks/provenance.json) records source URLs, Git
blobs, and original class fingerprints used by the preservation tests. Original
notices and [BSD-3-Clause terms](../uav_control/tasks/LICENSE.isaaclab) are retained;
they do not select a license for the rest of this project.

Task adaptations are limited to local quaternion compatibility, removal of unused
imports, and optional standard-library CSV telemetry in place of pandas output.
The local `quat_apply_inverse()` preserves the fork's scalar-first quaternion
formula. It agrees with official `quat_rotate_inverse()` for unit quaternions
but differs for non-unit inputs; unit-quaternion equivalence alone does not
justify replacing it.
Class-preservation tests exclude the documented telemetry setting, gates, and
CSV reset block; they do not establish simulator-level numerical equivalence.

Both configs retain `CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")`.
The [official asset configuration](https://github.com/isaac-sim/IsaacLab/blob/v2.1.0/source/isaaclab_assets/isaaclab_assets/robots/quadcopter.py)
resolves `{ISAAC_NUCLEUS_DIR}/Robots/Crazyflie/cf2x.usd`. No USD or framework patch
is bundled. Asset availability and live articulation behavior require native
validation.

## Startup and task registration

Root `train_agent.py` and installed `uav-train` invoke `uav_control.launcher:main`.
The launcher starts Isaac Sim through the official AppLauncher before importing
task implementations and registering environments. Ordinary imports of
`uav_control`, `uav_control.tasks`, and the launcher remain simulator-free.

`register_tasks()` uses lazy Gymnasium entry points, accepts matching existing
registrations, and rejects conflicting IDs. Environment/configuration classes
live in `uav_control.tasks.quadcopter_SARL` and `quadcopter_MARL` respectively.

| Gymnasium ID | Environment / configuration | Learner | Observations / actions |
| --- | --- | --- | --- |
| `UAV-Control-Quadcopter-SARL-v0` | `QuadcopterEnvSARL` / `QuadcopterEnvCfgSARL` | PPO | 15 / 4 |
| `UAV-Control-Quadcopter-MARL-v0` | `QuadcopterEnvMARL` / `QuadcopterEnvCfgMARL` | MAPPO | Translation: 21 / 4; Yaw: 9 / 1 |

Both MARL critics receive a 30-dimensional state: Translation observations then
Yaw observations. Actors use only their assigned local observations. Native
learners must use float32 and matching task dimensions; models move to the
selected device before optimizer construction.

## Adapter and episode-boundary contracts

Adapters preserve native device placement and reject shape/dtype/device
mismatches. They clone execution actions to protect stored Gaussian samples;
they do not clip actions, scale rewards, normalize observations, reset episodes,
or advance physics beyond the native step. Returned tensors may alias native
memory; collectors own copies needed for learning.

| Native API (`N` environments) | Workflow API |
| --- | --- |
| SARL `reset() -> ({"policy": [N,15]}, extras)` | `[N,15]` tensor |
| SARL five-tuple `step([N,4])` | `StepResult`, raw rewards `[N]`, separate boolean termination/truncation `[N]` |
| MARL `reset() -> (agent_observations, extras)` | `MultiObservation`, local `[N,21]` / `[N,9]`, state `[N,30]` |
| MARL five-tuple `step({Translation:[N,4], Yaw:[N,1]})` | `MultiStepResult`, separate raw rewards and flags for each agent |

The version-specific source contracts are:

- [DirectRLEnv](https://github.com/isaac-sim/IsaacLab/blob/v2.1.0/source/isaaclab/isaaclab/envs/direct_rl_env.py)
  computes dones/rewards, resets completed rows, then computes returned observations.
- [DirectMARLEnv](https://github.com/isaac-sim/IsaacLab/blob/v2.1.0/source/isaaclab/isaaclab/envs/direct_marl_env.py)
  resets on `all(terminated) OR all(time_out)`, then computes observations. The UAV
  task gives both agents the same physical boundaries. Rewards remain separate.
- With `state_space=-1`, `state()` concatenates `obs_dict` in `possible_agents`
  order. The adapter requires Translation then Yaw and obtains state after
  reset/step. Both policies share one native step.
- These tasks supply no guaranteed final pre-reset observations/states. Workflow
  final-input fields remain `None`; returned post-reset inputs are for the next
  action, never terminal bootstrap values. Native timeout bootstrapping stays
  disabled, and incompatible requests/checkpoints are rejected.

See [training workflow](training-workflow.md) for collection, reward scaling,
normalization, checkpoint, and evaluation contracts. Native checkpoints exclude
interaction snapshots: resume restores learner state into fresh physical
episodes. CUDA resume requires saved RNG states matching the available device
count; CPU mock checkpoints cannot directly resume CUDA training.

## Limitations and validation

Preserved task behavior needs review before research results are interpreted:

- Fixed world start/goal coordinates overwrite environment-origin offsets.
  Collision filtering must be checked natively; overlap alone does not prove
  cross-environment collisions.
- MARL's observed basis vectors are cached before physics and are not refreshed
  on reset. Both tasks retain thrust/moment history across resets for smoothing.
- Full-batch resets randomize episode ages, including every reset at `num_envs=1`.
  SARL/MARL also differ in wind means, distance shaping, bonuses, and motor limits.
- Per-environment motor-force loops and synchronous optional telemetry can limit
  throughput. Legacy CSV columns may differ in length and are padded with blanks;
  this does not establish sample alignment. I/O errors warn rather than stop training.
- The native CLI requires the prepared simulator environment even for help;
  supply `--task SARL --help` or `--task MARL --help`. Bare `--help` encounters the
  official helper's preliminary argument parse before task validation completes.

CPU tests cover source preservation, quaternion math, registration, tensor
ownership/shapes, state ordering, individual rewards/boundaries, and launcher
checkpoint/resume/evaluation with stand-ins. They do not execute UAV physics.
Native validation must cover startup, Crazyflie loading, articulation/wrench
behavior, singleton and parallel resets, collision isolation, visualization,
CUDA learning and evaluation, checkpoint resume, cleanup, and performance.
Ubuntu execution and research reproducibility also remain unverified. Later
Isaac Lab versions or patched native reset logic require a new compatibility review.
