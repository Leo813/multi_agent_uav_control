from dataclasses import fields, replace

import pytest
import torch

from uav_control.networks import ActorCritic, NetworkConfig
from uav_control.ppo import PPOBatch, PPOConfig, PPOLearner
from uav_control.schedules import adaptive_learning_rate


def small_learner(**kwargs):
    torch.manual_seed(5)
    config = replace(PPOConfig(), horizon=2, mini_batches=2, learning_epochs=2, **kwargs)
    return PPOLearner(ActorCritic(NetworkConfig(3, 2, (8,))), config)


def synthetic_rollout(learner, n=2):
    """Explicit synthetic transitions, including a timeout followed by a reset."""
    buffer = learner.new_buffer(n)
    for t in range(learner.config.horizon):
        observations = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) / 10 + t
        step = learner.act(observations)
        terminated = torch.full((n,), t == learner.config.horizon - 1)
        truncated = torch.full((n,), t == 0)
        next_values = learner.predict_values(observations + 0.1)
        final_values = learner.predict_values(observations + 0.2)
        buffer.add(observations=observations, actions=step.actions, rewards=torch.arange(n).float() + t + 1,
                   values=step.values, log_probs=step.log_probs, terminated=terminated, truncated=truncated,
                   next_values=next_values, truncation_values=final_values)
    return buffer


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("shuffle", [True, False])
def test_synthetic_rollout_update_and_fixed_targets(normalize, shuffle):
    learner = small_learner(normalize_observations=normalize, normalize_values=normalize,
                            shuffle_minibatches=shuffle, bootstrap_truncations=True)
    buffer = synthetic_rollout(learner)
    raw_before = buffer.snapshot()
    parameters = [p.detach().clone() for p in learner.model.parameters()]
    batch = learner.prepare_batch(buffer)
    assert batch.observations.shape == (4, 3) and batch.actions.shape == (4, 2)
    # Initial policy ratios remain one: observation statistics did not change.
    current = learner.model.evaluate_actions(batch.observations, batch.actions)
    torch.testing.assert_close(current.log_probs, batch.old_log_probs)
    torch.testing.assert_close(current.values, batch.old_values)
    saved = {f.name: getattr(batch, f.name).clone() for f in fields(PPOBatch)}
    assert not any(tensor.requires_grad for tensor in saved.values())
    if normalize:
        assert learner.observation_normalizer.count.item() == 1
        assert learner.value_normalizer.count.item() == 1
    with pytest.raises(RuntimeError, match="pending"):
        learner.act(torch.zeros(1, 3))
    with pytest.raises(RuntimeError):
        learner.prepare_batch(buffer)
    diagnostics = learner.optimize(batch, generator=torch.Generator().manual_seed(11))
    assert diagnostics.optimizer_steps == 4 and diagnostics.samples == 4
    assert all(torch.isfinite(torch.tensor(v)) for v in vars(diagnostics).values())
    assert any(not torch.equal(before, after) for before, after in zip(parameters, learner.model.parameters()))
    assert all(torch.isfinite(p).all() for p in learner.model.parameters())
    grad_norm = torch.stack([p.grad.square().sum() for p in learner.model.parameters() if p.grad is not None]).sum().sqrt()
    assert grad_norm <= learner.config.max_grad_norm + 1e-5
    for name, tensor in saved.items():
        assert torch.equal(getattr(batch, name), tensor)
    after = buffer.snapshot()
    assert torch.equal(raw_before.rewards, after.rewards)
    assert torch.equal(raw_before.actions, after.actions)
    assert torch.equal(raw_before.log_probs, after.log_probs)
    if normalize:
        assert learner.observation_normalizer.count.item() == 5
        assert learner.value_normalizer.count.item() == 5
    with pytest.raises(ValueError):
        learner.optimize(batch)
    with pytest.raises(ValueError, match="already"):
        learner.prepare_batch(buffer)
    # After statistics change, the next rollout still starts with aligned values.
    later = learner.prepare_batch(synthetic_rollout(learner))
    reevaluated = learner.model.evaluate_actions(later.observations, later.actions)
    torch.testing.assert_close(reevaluated.log_probs, later.old_log_probs)
    torch.testing.assert_close(reevaluated.values, later.old_values, atol=1e-6, rtol=1e-5)
    again = learner.optimize(later)
    assert again.optimizer_steps == 4


def test_reward_scaling_gae_and_one_value_normalization_snapshot():
    config = PPOConfig(horizon=2, mini_batches=1, normalize_advantages=False, gamma=1, gae_lambda=1,
                       reward_scale=0.5)
    learner = PPOLearner(ActorCritic(NetworkConfig(1, 1, ())), config)
    buffer = learner.new_buffer(1)
    for t, (reward, value) in enumerate(((2.0, 0.25), (4.0, 0.5))):
        buffer.add(observations=torch.zeros(1, 1), actions=torch.zeros(1, 1), rewards=torch.tensor([reward]),
                   values=torch.tensor([value]), log_probs=torch.zeros(1),
                   terminated=torch.tensor([t == 1]), truncated=torch.tensor([False]),
                   next_values=torch.tensor([0.5 if t == 0 else float('nan')]))
    batch = learner.prepare_batch(buffer)
    # Rewards [1,2], values [.25,.5] give returns [3,2] and advantages [2.75,1.5].
    torch.testing.assert_close(batch.advantages, torch.tensor([2.75, 1.5]))
    scaler = learner.value_normalizer
    assert scaler.count.item() == 1
    torch.testing.assert_close(scaler.denormalize(batch.returns[:, None]).squeeze(-1), torch.tensor([3.0, 2.0]))
    torch.testing.assert_close(scaler.denormalize(batch.old_values[:, None]).squeeze(-1), torch.tensor([0.25, 0.5]))
    assert torch.equal(buffer.snapshot().rewards[:, 0], torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(batch.reward_returns, torch.tensor([3.0, 2.0]))
    learner.optimize(batch)
    assert scaler.count.item() == 3
    assert scaler.mean.item() == pytest.approx(5 / 3)
    assert scaler.variance.item() == pytest.approx(17 / 9)


def test_known_first_adam_update():
    model = ActorCritic(NetworkConfig(1, 1, ())).double()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    config = PPOConfig(horizon=1, learning_epochs=1, mini_batches=1, learning_rate=0.01,
                       adaptive_lr=False, max_grad_norm=100, reward_scale=1, value_clip=None,
                       normalize_advantages=False, normalize_observations=False, normalize_values=False)
    learner = PPOLearner(model, config)
    obs = torch.ones(1, 1, dtype=torch.float64)
    actions = torch.ones_like(obs)
    evaluation = model.evaluate_actions(obs, actions)
    buffer = learner.new_buffer(1)
    buffer.add(observations=obs, actions=actions, rewards=torch.ones(1, dtype=torch.float64),
               values=evaluation.values, log_probs=evaluation.log_probs,
               terminated=torch.tensor([True]), truncated=torch.tensor([False]))
    report = learner.update(buffer)
    assert report.policy_loss == pytest.approx(-1)
    assert report.value_loss == pytest.approx(1)
    # dL/d policy weight = -1, dL/d value weight = -2. Adam's first bias-corrected step.
    assert model.policy_head.weight.item() == pytest.approx(0.01 / (1 + 1e-8), abs=1e-12)
    assert model.value_head.weight.item() == pytest.approx(0.02 / (2 + 1e-8), abs=1e-12)
    assert model.log_std.item() == 0


def test_value_normalization_does_not_shift_initial_clipping_center():
    model = ActorCritic(NetworkConfig(1, 1, ()))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    learner = PPOLearner(model, PPOConfig(horizon=1, learning_epochs=1, mini_batches=1,
                                         reward_scale=1, normalize_observations=False))
    obs = torch.ones(1, 1)
    step = learner.act(obs)
    buffer = learner.new_buffer(1)
    buffer.add(observations=obs, actions=step.actions, rewards=torch.tensor([100.0]),
               values=step.values, log_probs=step.log_probs,
               terminated=torch.tensor([True]), truncated=torch.tensor([False]))
    batch = learner.prepare_batch(buffer)
    # Updating statistics from the large return here would move old_values far
    # outside the +/-0.2 clipping interval around the unchanged critic output.
    assert learner.value_normalizer.count.item() == 1
    assert batch.old_values.item() == 0
    report = learner.optimize(batch)
    assert report.value_loss == pytest.approx(10000)
    assert model.value_head.bias.item() > 0
    assert learner.value_normalizer.count.item() == 2


def test_zero_objective_preserves_parameters_and_scheduler_cadence():
    model = ActorCritic(NetworkConfig(1, 1, ()))
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    learner = PPOLearner(model, PPOConfig(horizon=1, learning_epochs=2, mini_batches=1,
                                         normalize_values=False, normalize_observations=False))
    before = [p.detach().clone() for p in model.parameters()]
    obs = torch.ones(1, 1)
    step = learner.act(obs)
    buffer = learner.new_buffer(1)
    buffer.add(observations=obs, actions=step.actions, rewards=torch.zeros(1), values=step.values,
               log_probs=step.log_probs, terminated=torch.tensor([True]), truncated=torch.tensor([False]))
    report = learner.update(buffer)
    assert report.total_loss == 0 and report.grad_norm == 0
    assert report.learning_rate == pytest.approx(0.0005 * 1.5**2)
    assert all(torch.equal(p, q) for p, q in zip(before, model.parameters()))


def test_predict_values_and_deterministic_inference_do_not_update_stats_or_rng():
    learner = small_learner()
    learner.value_normalizer.update(torch.tensor([[2.0], [4.0]]))
    obs = torch.ones(1, 3)
    before = torch.get_rng_state()
    values = learner.predict_values(obs)
    expected = learner.value_normalizer.denormalize(
        learner.model(learner.observation_normalizer.normalize(obs)).values[:, None]).squeeze(-1)
    torch.testing.assert_close(values, expected)
    step = learner.act(obs, deterministic=True)
    torch.testing.assert_close(step.values, values)
    assert torch.equal(torch.get_rng_state(), before)
    assert learner.observation_normalizer.count.item() == 1
    assert learner.value_normalizer.count.item() == 3


@pytest.mark.parametrize("bootstrap", [True, False])
def test_timeout_requires_final_values_only_when_enabled(bootstrap):
    learner = small_learner(bootstrap_truncations=bootstrap)
    buffer = learner.new_buffer(2)
    for _ in range(2):
        buffer.add(observations=torch.zeros(2, 3), actions=torch.zeros(2, 2), rewards=torch.ones(2),
                   values=torch.zeros(2), log_probs=torch.zeros(2),
                   terminated=torch.zeros(2, dtype=torch.bool), truncated=torch.ones(2, dtype=torch.bool),
                   next_values=torch.full((2,), 999.0))
    if bootstrap:
        with pytest.raises(ValueError, match="truncation_values"):
            learner.prepare_batch(buffer)
        assert learner.value_normalizer.count.item() == 1 and not buffer.prepared
    else:
        batch = learner.prepare_batch(buffer)
        torch.testing.assert_close(learner.value_normalizer.denormalize(batch.returns[:, None]),
                                   torch.full((4, 1), 0.01), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("kl,expected", [(0, 0.00075), (0.008, 0.0005),
                                        (0.016, 0.0005), (0.032, 0.0005), (0.04, 0.0005 / 1.5)])
def test_adaptive_lr_scalar_reference(kl, expected):
    assert adaptive_learning_rate(0.0005, kl) == pytest.approx(expected)
    assert adaptive_learning_rate(1e-6, 1) == 1e-6
    assert adaptive_learning_rate(1e-2, 0) == 1e-2


@pytest.mark.parametrize("kwargs", [dict(horizon=0), dict(mini_batches=0), dict(gamma=2),
                                     dict(learning_rate=-1), dict(ratio_clip=1), dict(reward_scale=0),
                                     dict(value_clip=-1), dict(max_grad_norm=float('nan')),
                                     dict(normalize_values=1), dict(min_lr=1), dict(lr_factor=1)])
def test_invalid_configuration(kwargs):
    with pytest.raises((ValueError, TypeError)):
        PPOConfig(**kwargs)


def test_incomplete_or_incompatible_buffer():
    learner = small_learner()
    with pytest.raises(ValueError, match="complete"):
        learner.prepare_batch(learner.new_buffer(2))
    wrong_horizon = PPOLearner(ActorCritic(NetworkConfig(3, 2, (8,))), PPOConfig(horizon=3))
    with pytest.raises(ValueError, match="dimensions"):
        learner.prepare_batch(wrong_horizon.new_buffer(2))
    with pytest.raises(ValueError):
        adaptive_learning_rate(0.001, float('nan'))
