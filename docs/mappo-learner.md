# Two-agent MAPPO learner

`uav_control.mappo.MAPPOLearner` coordinates two independent policy/value pairs
named `Translation` and `Yaw`. It consumes tensor rollouts, creates no environment,
and reuses PPO target preparation, losses, optimization, and scheduling.

## Networks and configuration

| Agent | Local actor input | Raw Gaussian action | Centralized critic input | Value |
| --- | --- | --- | --- | --- |
| Translation | [N,21] | [N,4] | [N,30] | [N] |
| Yaw | [N,9] | [N,1] | [N,30] | [N] |

Each actor and critic has its own 32/32 ELU MLP. No parameters are shared between
agents or between actor and critic. Each actor has a trainable state-independent
log_std initialized to zero and clamped to [-20,2] for Gaussian evaluation.
Mean heads are unsquashed; deterministic actions are means. Critic heads are
scalar. Default parameter counts are 3,977 (Translation) and 3,491 (Yaw).

`ActorCritic(NetworkConfig(..., shared=False, critic_observation_dim=S))` permits
different local and critic inputs. `hidden_sizes` controls actor widths;
`critic_hidden_sizes` overrides critic widths, otherwise using the same widths.
Existing activation, log_std, and dtype options remain available. Set models to
their final device/dtype before learner construction. All models and rollouts
must share one device and float32 or float64 dtype.

`MAPPOConfig` inherits PPO settings with defaults: horizon 16, epochs 8,
minibatches 2, learning rate 0.0003, KL threshold 0.008; gamma 0.99, lambda 0.95,
policy/value clips 0.2, gradient norm limit 1, value/entropy coefficients 1/0,
reward scale 0.01, timeout bootstrap disabled. All three preprocessing paths
are enabled: local observations, shared states, and values. State clipping
defaults to 5 (`state_clip=None` disables it).

Supply mappings of models and configs keyed by both agent names to override
defaults. The horizon and centralized-state dimension must agree; other settings
may differ per agent. No team reward or team-value estimate is constructed.

## Inference and updates

```python
from uav_control.mappo import MAPPOLearner

learner = MAPPOLearner()  # default CPU UAV architecture
buffer = learner.new_buffer(num_envs=8)
# External collector obtains raw local observations and raw centralized state.
# translation = learner.act("Translation", translation_obs)
# yaw = learner.act("Yaw", yaw_obs, deterministic=True)  # evaluation only
# values = learner.predict_values("Translation", shared_state)
# After collection: batch = learner.prepare_batch(buffer)
# diagnostics = learner.optimize(batch)  # or learner.update(buffer)
```

`act(agent, local_observations, deterministic=False)` returns detached
`PolicySample(actions [N,A], log_probs [N])`. It never evaluates a critic or
requires another agent's observation. Stochastic selection is for collection;
deterministic selection is for evaluation. `predict_values(agent, states)` returns
detached values in **scaled-reward units** without sampling or updating moments.

At network level, `sample_policy()` and differentiable `policy_log_probs()` take
only local inputs. `predict_values()` takes critic inputs only.
`evaluate_actions(local_inputs, actions, critic_states=states)` returns log
probabilities, mean per-coordinate entropy, and critic values for training.
Network inputs are already preprocessed; learner inference inputs are raw.

`prepare_batch()` computes GAE/returns independently using each agent's rewards,
values, and boundaries. `MAPPOBatch.agents` maps to existing `PPOBatch` objects;
`critic_states` maps to independently normalized [B,S] inputs, and `raw_states`
retains the owned raw [B,S] state. Treat tensors and mappings as read-only.

`optimize()` uses the shared PPO implementation once for each agent in
Translation/Yaw order. Each agent has one Adam optimizer over its separate actor
and critic parameters, joint loss/backward, and combined gradient norm clipping,
matching skrl's optimizer organization. Optimizers, schedules, rewards, and
advantages are independent across agents. Returned diagnostics are keyed by
agent; each contains losses, KL, clip fraction, gradient norm, next LR, steps,
and unique sample count. Checkpointing, logging, and environment loops belong to
the separate [training workflow](training-workflow.md).

## Storage and normalization contract

`MultiAgentRolloutBuffer.add()` accepts `states [N,S]` and mappings for
`observations`, `actions`, `rewards`, `values`, `log_probs`, `terminated`,
`truncated`, optional `next_values`, and optional `truncation_values`. Every
provided mapping must contain exactly both agents; optional bootstrap entries
may individually be `None`. All fields for both agents are validated before
appending a row. Invalid input leaves the append position unchanged.

Storage owns detached local observations [T,N,O_agent], actions [T,N,A_agent],
individual scalar fields [T,N], and one shared state [T,N,S]. `snapshot()`
requires a full buffer and returns independent copies; `clear()` permits refill.
Store raw environment rewards and raw policy actions/log probabilities. Copy
observations, states, and actions before mutable simulator buffers can change;
use a separate action tensor for execution transformations.

Flattening uses `b=t*N+n`. The same indices select each agent's local data and
centralized states. Every sample appears once per epoch; exact minibatch
divisibility is required. Sequential order is default, with optional device-local
shuffling. Agents may use different minibatch partitions because their updates
are independent. No rollout tensor transfers to CPU are required; eager checks
may synchronize CUDA, and scheduler/diagnostic scalars reach the host.

Each agent has its own local-observation, centralized-state, and value normalizer.
The state normalizers see the same raw states; identical settings produce equal
moments, but their state is not shared. Preserve the PPO normalization contract:
freeze all moments through collection and both agents' optimization; afterward,
update each normalizer once from its raw local observations, raw shared states,
or scaled-reward returns. Scale rewards exactly once during preparation. Compute
advantages in scaled-reward units and standardize separately per agent. Initial
policy ratios remain one and critic clipping centers align with predictions.

One batch may be pending. Inference and further preparation are blocked until
optimization. Each fill/batch is consumed once. Callers must not mutate models,
normalizers, or prepared data, mix stale rollouts, or collect asynchronously.
Failures during optimization can leave partial parameter updates; neither agent
is rolled back, and the batch cannot be retried. Recover from a known state.

## References and deliberate differences

UAV references are pinned to
[`bdd31df85cc47e07715442cdc3fe0057c798fb35`](https://github.com/YuHaoTsai/IsaacLab/tree/bdd31df85cc47e07715442cdc3fe0057c798fb35/source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter_MARL):
`quadcopter_MARL.py` blob `cca68159dfb68eb477b64f8c7fa18bcde6f46a02` and
`agents/skrl_mappo_cfg.yaml` blob `a88144fa2ed1e4c2ab8437d3b7373e0e9ded1d4f`.
The task returns distinct rewards. Its `possible_agents` order is Translation,
Yaw; `state_space=-1` yields the 30-dimensional concatenation under
[Isaac Lab 2.1.0 DirectMARLEnv.state()](https://github.com/isaac-sim/IsaacLab/blob/v2.1.0/source/isaaclab/isaaclab/envs/direct_marl_env.py)
(blob `1a95f5b0e83fa063c465bbeccef7fe006b621e19`). The learner accepts an explicit
state tensor; the native adapter supplies it in that order.

[skrl 1.4.2 MAPPO](https://github.com/Toni-SM/skrl/blob/1.4.2/skrl/multi_agents/torch/mappo/mappo.py)
(blob `2cf0a78ca52967c8bbe48782aabb1a1bdf76eaec`) is the chosen behavior reference,
not a verified historical runtime. Separate policies/critics, per-agent Adam
and preprocessors, prediction-clipped MSE, and epoch-level KL scheduling follow
that reference. Implementation source was independently written.

Differences retain the [PPO choices](ppo-learner.md#reference-choices-and-differences):
frozen normalization through optimization, population running moments,
unclipped value transformations with consistent inverse epsilon, exact
minibatch coverage, optional shuffling, and explicit pre-reset timeout values.
Moment updates are not PopArt and can change predictions between rollouts.
No reward correction using pre-action values is applied. Clipping is controlled
by each agent's `value_clip`, including `None` to disable it; no KL early stopping,
mixed precision, or distributed execution is implemented.

## Verification

Numerical checks cover
different agent dimensions/rewards/boundaries, decentralized execution,
actor/critic gradient isolation, analytic Gaussian likelihoods and Adam steps,
normalization snapshots, atomic storage/preparation, minibatch alignment,
independent updates, and consecutive synthetic MAPPO updates. Run
`python -m pytest -q` in the prepared CPU test environment for the full suite,
including PPO regressions.

Adapters must supply aligned local observations and centralized states for the
**same pre-action timestep** and preserve per-agent rewards and flags. Collectors
compute per-agent next values in scaled-reward units. Continuing transitions
need `next_values`, including the final rollout row. Optional truncation
bootstrapping needs each critic evaluated on the final centralized state before
reset; post-reset values are invalid. Termination overrides truncation. Keep
bootstrapping disabled if final states are unavailable. See [GAE](ppo-core.md#gae-and-episode-boundaries).

The [training workflow](training-workflow.md) supplies collection/lifecycle,
persistence, and evaluation verified against CPU mocks. See
[Isaac Lab integration](isaaclab-integration.md) for native adapters, task
limitations, and disabled native timeout bootstrapping. CUDA, Ubuntu,
Isaac Lab/Sim execution, and research performance remain untested.
