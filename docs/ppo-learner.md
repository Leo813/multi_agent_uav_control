# Single-agent PPO learner

The learner performs inference and optimization on supplied tensors. It creates
no environment and contains no rollout collection loop. All modules are flat
under `uav_control/`; importing the base package does not load PyTorch.

## Configuration and architecture

`networks.NetworkConfig` defaults to 15 observations, four actions, a shared
64/64 ELU trunk, distinct linear mean/value heads, and a trainable,
state-independent log-standard-deviation vector initialized to zero. This
implements the shared hidden representation intended by `models.separate: False`
in the UAV configuration. It has 5,513 parameters. Linear layers retain PyTorch's
default initialization; there are no recurrent layers, dropout, or output bounds.

Observation/action dimensions, hidden widths (including no hidden layers), ELU /
ReLU / tanh, shared versus separate trunks, and fixed versus trainable log_std
are configurable. Gaussian log_std bounds default to [-20, 2]. Stochastic actions
are raw unsquashed samples; deterministic actions are means. Both use the
existing distribution functions. Entropy regularization averages action
coordinates, matching the selected skrl reference.

`ppo.PPOConfig` supplies the following configurable defaults:

| Setting | Default |
| --- | --- |
| Horizon / epochs / minibatches | 24 / 5 / 4 |
| Discount / GAE lambda | 0.99 / 0.95 |
| Adam learning rate | 0.0005 |
| Policy / value clipping | 0.2 / 0.2 (`value_clip=None` disables value clipping) |
| Gradient norm limit | 1.0 |
| Value / entropy coefficients | 1.0 / 0.0 |
| Raw reward multiplier | 0.01 |
| Truncation bootstrap | Disabled |
| Observation / value / advantage normalization | Enabled independently |
| Observation clip / normalization epsilon | 5.0 / 1e-8 |
| Minibatch shuffling | Disabled; optional fresh permutation each epoch |
| Adaptive LR / KL threshold | Enabled / 0.016 |
| LR bounds / KL factor / LR factor | [1e-6, 0.01] / 2.0 / 1.5 |

Adam uses PyTorch defaults: betas (0.9, 0.999), epsilon 1e-8, no weight decay or
AMSGrad. Float32 and float64 models are supported. Move the model to its final
device/dtype before constructing the learner. Normalizer buffers remain float64.

## Interfaces and rollout ownership

```python
from uav_control.networks import ActorCritic, NetworkConfig
from uav_control.ppo import PPOLearner, PPOConfig

model = ActorCritic(NetworkConfig())  # CPU; choose device/dtype here
learner = PPOLearner(model, PPOConfig())
buffer = learner.new_buffer(num_envs=8)
# An external collector must fill buffer using the contract below.
# batch = learner.prepare_batch(buffer)
# diagnostics = learner.optimize(batch)  # alternatively: learner.update(buffer)
```

`learner.act(raw_observations, deterministic=False)` accepts `[N,O]` and returns
`PolicyStep(actions [N,A], log_probs [N], values [N])`, detached from autograd.
Values are in **scaled-reward units**, with value normalization inverted.
`predict_values(raw_observations)` uses the same units without sampling or
changing statistics. `model.evaluate_actions(normalized_observations, raw_actions)`
is differentiable and returns log probabilities, per-coordinate mean entropy,
and values in normalized critic coordinates.

`buffer.add(...)` appends one transition for every environment. Supply raw
`observations [N,O]`, raw `actions [N,A]`, and `[N]` tensors for `rewards`,
`values`, `log_probs`, boolean `terminated` and `truncated`, plus optional
`next_values` and `truncation_values`. All floating tensors must match the
learner's dtype/device. Store **raw environment rewards**; do not scale them in
the collector. All three value fields use scaled-reward units.

Storage is owned and detached: observations `[T,N,O]`, actions `[T,N,A]`, all
other fields `[T,N]`. `snapshot()` requires a full rollout and returns independent
copies. `clear()` permits refilling. Preserve raw observations and actions before
an environment can mutate them; pass a separate action tensor through execution
clipping/transformation. Likelihoods always refer to the stored raw samples.

Flattening uses `b = t*N + n`. Equal-size minibatches cover every sample exactly
once per epoch; `T*N` must divide evenly by the minibatch count. Optional shuffle
uses device-local indices and an optional matching-device `torch.Generator`.
No rollout tensors move to CPU. Eager validation can synchronize CUDA;
scheduler scalars and final diagnostics transfer to the host.

Boundary flags describe the result of each action. For continuing transitions,
provide `next_values`, including the last rollout row. Termination takes
precedence over truncation. Optional timeout bootstrapping requires
`truncation_values` from the final observation **before reset**. Unused bootstrap
entries may be absent/NaN. See [GAE semantics](ppo-core.md#gae-and-episode-boundaries).

## Preparation, normalization, and optimization

Keep policy parameters and both normalizers unchanged while collecting one
rollout. `prepare_batch()` scales rewards once, reuses GAE, normalizes advantages
globally with sample standard deviation, and transforms raw observations and
value targets using the **same frozen statistics as collection**. Advantages
and returns are computed in scaled-reward units before target normalization.
A singleton or constant advantage batch becomes zero.

The returned `PPOBatch` owns detached flattened tensors. `observations` are
already normalized; `actions` and `old_log_probs` are fixed policy samples.
`old_values` and `returns` share the collection value-normalization snapshot.
`raw_observations` and `reward_returns` retain original observation and
scaled-reward target units for later statistics updates. Treat every field as
read-only. Preparation does not update normalizers; initial policy ratios are
one and critic predictions match the old-value clipping centers, up to rounding.

`optimize()` reevaluates stored actions, uses the existing PPO objective, then
performs backward, gradient clipping, and Adam steps for every minibatch/epoch.
Value clipping clips predictions before MSE; there is no maximum-of-two-errors
rule or one-half factor. Targets remain fixed across epochs. After each epoch,
mean pre-step approximate KL drives the schedule: above `2*threshold`, divide LR
by 1.5; below `threshold/2`, multiply it by 1.5; otherwise retain it. Clamp to the
configured LR bounds. There is no KL early stopping.

After successful optimization, update observation statistics once from raw
observations and value statistics once from scaled-reward returns. The next
rollout uses these new moments. This is **not PopArt**: network weights and Adam
moments are not rebased, so changing statistics changes predictions in original
units across rollout boundaries. Statistical updates themselves are not bounded
by the PPO KL metric. Evaluation must freeze statistics and avoid `update()`.

`RunningNormalizer` merges float64 population moments with an initial prior
(count 1, mean 0, variance 1). Normalization uses `(x-mean)/(sqrt(variance)+eps)`;
only observations clip by default. Values and returns are unclipped. Inversion
uses the same epsilon without clipping. `train()`/`eval()` never change moments;
only explicit `update()` does. No normalization is repeated on prepared tensors.

Diagnostics report means over all equal-size minibatch steps: total/policy/value
and weighted entropy loss, approximate KL, clip fraction, and pre-clipping
gradient norm; also next learning rate, optimizer-step count, and unique sample
count. They are Python scalars, not a training logger.

Only one batch may be pending. Inference/preparation is blocked until that batch
is optimized. A buffer can be prepared once per fill and a batch optimized once.
The caller must not mix stale rollouts or mutate model/normalizer state between
collection and optimization; no asynchronous collector is supported. Failed
updates are not transactional: earlier optimizer steps may have occurred and the
batch remains consumed. Recover from a known state rather than retrying it.

## Reference choices and differences

UAV task and PPO configuration reference:
[`bdd31df85cc47e07715442cdc3fe0057c798fb35`](https://github.com/YuHaoTsai/IsaacLab/tree/bdd31df85cc47e07715442cdc3fe0057c798fb35/source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter_MARL).
Configuration blob: `6bbf7680075b393b36ea7732055bfd21f5edc92b`.
skrl **1.4.2** is a selected behavior reference, not a verified historical runtime.
Its versioned [shared model](https://github.com/Toni-SM/skrl/blob/1.4.2/skrl/utils/model_instantiators/torch/shared.py),
[PPO](https://github.com/Toni-SM/skrl/blob/1.4.2/skrl/agents/torch/ppo/ppo.py),
[scaler](https://github.com/Toni-SM/skrl/blob/1.4.2/skrl/resources/preprocessors/torch/running_standard_scaler.py),
and [scheduler](https://github.com/Toni-SM/skrl/blob/1.4.2/skrl/resources/schedulers/torch/kl_adaptive.py)
implementations were inspected. All new implementation code is independently written.

- Shared architecture, optimizer settings, prediction-clipping loss, entropy
  reduction, and epoch-level KL scheduling follow those references.
- skrl updates observation moments in first-epoch minibatches and value moments
  separately on old values and returns before optimization. Here both are frozen
  through optimization and updated once afterward. This preserves collection
  likelihoods and value clipping coordinates before the first gradient step.
- Running moments use population variance, including singleton batches, rather
  than skrl's sample batch variance. Transformations retain float64 arithmetic
  until casting to the input dtype. Value targets and inverse predictions are
  unclipped, with a consistent inverse epsilon; skrl's scaler clips to +/-5 and
  omits epsilon during inversion.
- Exact divisibility replaces skrl's possible leftover-sample omission. Fresh
  per-epoch shuffling is optional; sequential minibatches remain the default.
- Raw policy actions and explicit pre-reset bootstrap values retain the
  mathematical-foundations contract; no timeout reward correction is applied.

The Hexapod checkout at `f0035b4fb93b030c13911dfb8970304b79db0d2d` informed
separation of learner, collector, adapter, and lifecycle through `src/hexapod_rl/`,
`train_agent.py`, and `docs/ppo-implementation.md`. Its Isaac Gym interfaces,
RL-Games objective, hyperparameters, and timeout conventions were not adopted.

## Verification

Run `python -m pytest -q` in the prepared CPU test environment for the full suite,
including [MAPPO checks](mappo-learner.md#verification).
PPO tests include analytic gradients, a hand-calculated first Adam step,
known running moments and GAE targets, clipping-center regression, zero-update
behavior, detached storage/targets, minibatch coverage, schedule boundaries, and
synthetic consecutive rollouts with normalization/shuffling enabled or disabled.

Network configuration, distributions, GAE/loss functions, owned storage,
normalizers, and the schedule are reused by the [MAPPO learner](mappo-learner.md).
The single-agent learner retains its original local observation input. Network
configuration now also supports explicit centralized critic dimensions and
independent critic widths when `shared=False`. PPO and MAPPO call the same target
preparation and optimization helpers; the mathematical functions are unchanged.

The [training workflow](training-workflow.md) supplies collection, lifecycle,
checkpoint/resume, and evaluation verified with CPU mocks. The
[Isaac Lab integration](isaaclab-integration.md) documents the implemented task
adapters and source-level reset/action contracts. CUDA, Isaac Lab/Sim execution,
Ubuntu, and research performance remain untested.
