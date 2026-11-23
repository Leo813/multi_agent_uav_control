# PPO mathematical foundations

These functions implement distribution and objective mathematics only. There
are no networks, optimizer steps, rollout collectors, or simulator imports.

## Interfaces

Inputs use float32 or float64 tensors on a common device. Shapes are checked
without implicit squeezing or broadcasting, except the stated per-action
`log_std` broadcast. `B` is a flattened sample batch, `T` a rollout length,
`N` an environment count, and `A` an action dimension. Dimensions must be positive.

| Module / function | Inputs | Output |
| --- | --- | --- |
| `distributions.diagonal_gaussian` | `mean [..., A]`, `log_std [A]` or `[..., A]` | PyTorch `Independent(Normal, 1)` |
| `distributions.sample_actions` | Same; optional `deterministic=True` | Detached raw actions `[..., A]` |
| `distributions.gaussian_log_prob` | Actions and mean `[..., A]`, log_std as above | Joint log probability `[...]` |
| `distributions.gaussian_entropy` | Mean/log_std as above; `reduction="sum"` or `"mean"` | Entropy `[...]` |
| `advantages.generalized_advantage_estimate` | Rewards/values `[T, N]`, boolean terminated/truncated `[T, N]`, optional bootstrap tensors `[T, N]` | `GAEOutput(advantages, returns)`, both `[T, N]` |
| `losses.probability_ratio` | New/old joint log probabilities `[B]` | Ratio `[B]` |
| `losses.clipped_policy_loss` | New/old log probabilities and advantages `[B]` | Scalar loss, approximate KL, clip fraction |
| `losses.value_loss` | Values/returns `[B]`, optional old values `[B]` | Scalar MSE |
| `losses.ppo_loss` | Log probabilities, advantages, values, returns `[B]`; optional old values/entropy `[B]` | `PPOLoss` with total, component losses, and diagnostics |

Base package import remains lightweight. Import the mathematical functions from
their modules, for example `from uav_control.advantages import generalized_advantage_estimate`.

## Gaussian actions

The policy is an unsquashed diagonal Gaussian. Log standard deviations are
clamped to `[-20, 2]` by default before exponentiation. Bounds are configurable
but must preserve representable variances in the input dtype. Nonfinite inputs
are rejected rather than repaired. Log density and joint entropy sum over the
last (action) axis. Batch axes are retained, including singleton batches.

Sampling returns detached actions; deterministic selection returns an owned,
detached copy of the mean. Log-probability evaluation treats supplied actions as
fixed samples, while retaining gradients through the mean and log_std. Future
collectors must store raw samples and their old log probabilities before passing
a separate action tensor through any environment clipping or transformation.

`gaussian_entropy(..., reduction="sum")` is joint entropy. Use `"mean"` followed
by a batch mean to match skrl 1.4.2's per-coordinate entropy regularization.
Both original UAV configurations set the entropy coefficient to zero.

## GAE and episode boundaries

`values[t,n]` is the value before action `t`. Flags describe that action's
result, not the state at the start of the next stored step. Define
`continuing = ~(terminated | truncated)` and select a bootstrap value `b`:

| Transition | Bootstrap value | Continue advantage trace? |
| --- | --- | --- |
| Continuing, including the last rollout row | `next_values[t,n]` | Yes (zero carry beyond rollout) |
| Terminated, even if also truncated | Zero | No |
| Truncated, `bootstrap_truncations=False` (default) | Zero | No |
| Truncated only, `bootstrap_truncations=True` | `truncation_values[t,n]` | No |

```text
delta[t] = rewards[t] + gamma * b[t] - values[t]
advantage[t] = delta[t] + gamma * gae_lambda * continuing[t] * advantage[t+1]
return[t] = advantage[t] + values[t]
```

The recurrence runs backward with a zero carry beyond the rollout. Defaults are
`gamma=0.99`, `gae_lambda=0.95`. Outputs are detached and unnormalized; no reward
scaling, advantage standardization, or value normalization is performed here.

For continuing transitions, callers provide `next_values`, including the final
rollout bootstrap. For timeouts with bootstrapping enabled, callers must provide
`truncation_values` evaluated on the **final observation before reset**. Values
of returned post-reset observations are not a substitute. Missing required
inputs raise an error; termination takes precedence if both flags are true.
Unused entries in bootstrap tensors may be NaN. `gamma=0` needs no bootstrap.
Do not apply a separate timeout reward correction before calling this function.

This contract makes no assumptions about Isaac Lab autoreset behavior. A future
adapter must obtain final values correctly or leave timeout bootstrapping
disabled, as in the reference configurations. No Isaac Gym reset conventions
are inherited.

## PPO losses

Losses are means over `[B]` and are minimized. The policy loss uses
`-mean(min(ratio * advantage, clipped_ratio * advantage))`, with
`ratio = exp(new_log_prob - old_log_prob)` and default clip range `0.2`.

Value loss is MSE with no factor of one half. With `clip_range` supplied,
predictions are first clipped around detached `old_values`, then compared with
returns. This is skrl's convention, rather than taking the maximum of two errors.
Predictions outside the clip interval have zero value gradient. The standalone
default is unclipped; pass `0.2` to match the UAV configs' enabled value clipping.

`ppo_loss` combines policy loss, `value_coefficient * value_loss`, and
`-entropy_coefficient * mean(entropy)`. Coefficients default to `1.0` and `0.0`.
Positive entropy regularization requires an explicit entropy tensor; its action
reduction is the caller's choice. Old log probabilities, advantages, returns,
and old values are detached inside the loss functions.

Diagnostics are detached scalar tensors: approximate KL is
`mean(expm1(log_ratio) - log_ratio)`; clip fraction counts ratios strictly outside
the clip interval, independent of advantage sign. Neither triggers optimization
or early stopping. Overflow, ratios underflowing to zero, invalid shapes/dtypes,
and nonfinite results raise errors instead of silently changing the objective.

These are eager mathematical APIs. Validation uses tensor reductions that may
synchronize a CUDA device; CUDA performance, mixed precision, and compilation
have not been validated. No tensors are silently copied to CPU.

## Reference scope

The UAV settings reviewed are at upstream revision
[`bdd31df85cc47e07715442cdc3fe0057c798fb35`](https://github.com/YuHaoTsai/IsaacLab/commit/bdd31df85cc47e07715442cdc3fe0057c798fb35),
in `source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter_MARL/agents/`:

- `skrl_ppo_cfg.yaml`: blob `6bbf7680075b393b36ea7732055bfd21f5edc92b`.
- `skrl_mappo_cfg.yaml`: blob `a88144fa2ed1e4c2ab8437d3b7373e0e9ded1d4f`.

Algorithm conventions were checked against the versioned
[skrl 1.4.2 PPO documentation](https://skrl.readthedocs.io/en/1.4.2/api/agents/ppo.html)
and [Gaussian documentation](https://skrl.readthedocs.io/en/1.4.2/api/models/gaussian.html).
Entropy aggregation was also checked against that release's Gaussian API and
PPO update behavior. This is a chosen reference version, not a claim about the
exact skrl installation used for the historical experiments. All new code is
independently written; skrl is not a dependency.

GAE follows the positive discounted trace in the
[GAE paper](https://arxiv.org/abs/1506.02438), with explicit boundary masks and
final-state timeout bootstrapping. The latter is an opt-in contract, not a claim
of equivalence with skrl's optional timeout reward correction. Architecture and
interface choices do not imply historical result reproduction.

## Verification

Use Python 3.10 with CPU PyTorch 2.5.1 and pytest 8.x in an isolated environment.
From the repository root, run `python -m pytest -q` (or
`.venv\Scripts\python.exe -m pytest -q` in Windows PowerShell).

Tests use scalar Gaussian formulas, hand-calculated trajectories and clipping
cases, known analytic gradients, and finite-difference gradient checking. They
cover parallel boundaries, required bootstrap values, detached targets, numerical
edge cases, and invalid inputs. CPU software tests do not validate Isaac Lab,
CUDA execution, or a complete PPO training procedure.
