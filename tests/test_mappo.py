from dataclasses import fields, replace
import math

import pytest
import torch

from uav_control.mappo import AGENTS, MAPPOConfig, MAPPOLearner
from uav_control.networks import ActorCritic, NetworkConfig
from uav_control.ppo import PPOBatch, PPOLearner
from uav_control.rollout import MultiAgentRolloutBuffer, minibatch_indices


def small_learner(*, normalize=False, dtype=torch.float32, **kwargs):
    torch.manual_seed(41)
    models = {a: ActorCritic(NetworkConfig(o, d, (4,), shared=False, critic_observation_dim=5,
                                          critic_hidden_sizes=(6,))).to(dtype=dtype)
              for a, o, d in (("Translation", 3, 2), ("Yaw", 2, 1))}
    config = replace(MAPPOConfig(horizon=2, learning_epochs=2, mini_batches=2,
                                 normalize_observations=normalize, normalize_states=normalize,
                                 normalize_values=normalize), **kwargs)
    return MAPPOLearner(models, {a: config for a in AGENTS})


def transition(learner, t=0, n=2):
    obs = {a: torch.arange(n * m.config.observation_dim, dtype=learner.dtype).reshape(
        n, m.config.observation_dim) / 10 + t for a, m in learner.models.items()}
    states = torch.cat([obs[a] for a in AGENTS], dim=-1)
    steps = {a: learner.act(a, obs[a]) for a in AGENTS}
    return dict(states=states, observations=obs, actions={a: steps[a].actions for a in AGENTS},
                log_probs={a: steps[a].log_probs for a in AGENTS},
                values={a: learner.predict_values(a, states) for a in AGENTS},
                rewards={a: torch.arange(1, n + 1, dtype=learner.dtype) * (i + 1) + t
                         for i, a in enumerate(AGENTS)},
                terminated={a: torch.full((n,), t == learner.horizon - 1) for a in AGENTS},
                truncated={a: torch.zeros(n, dtype=torch.bool) for a in AGENTS},
                next_values={a: learner.predict_values(a, states + 0.2) for a in AGENTS})


def rollout(learner, n=2):
    buffer = learner.new_buffer(n)
    for t in range(learner.horizon):
        buffer.add(**transition(learner, t, n))
    return buffer


def test_default_architecture_and_reference_configuration():
    learner = MAPPOLearner()
    for agent, obs, act, parameters in (("Translation", 21, 4, 3977), ("Yaw", 9, 1, 3491)):
        model = learner.models[agent]
        assert sum(p.numel() for p in model.parameters()) == parameters
        assert model.config.critic_observation_dim == 30 and not model.config.shared
        assert [x.out_features for x in model.policy_trunk if isinstance(x, torch.nn.Linear)] == [32, 32]
        assert [x.out_features for x in model.value_trunk if isinstance(x, torch.nn.Linear)] == [32, 32]
        assert sum(isinstance(x, torch.nn.ELU) for x in model.policy_trunk) == 2
        sample = learner.act(agent, torch.ones(3, obs))
        assert sample.actions.shape == (3, act) and sample.log_probs.shape == (3,)
        assert learner.predict_values(agent, torch.ones(3, 30)).shape == (3,)
        config = learner.configs[agent]
        assert (config.horizon, config.learning_epochs, config.mini_batches) == (16, 8, 2)
        assert (config.learning_rate, config.kl_threshold, config.reward_scale) == (3e-4, 0.008, 0.01)
        assert not config.bootstrap_truncations
    assert learner.optimizers['Translation'] is not learner.optimizers['Yaw']


def test_actor_only_inference_and_analytic_log_probabilities():
    learner = MAPPOLearner()
    for agent in AGENTS:
        model = learner.models[agent]
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
        obs = torch.ones(2, model.config.observation_dim)
        rng = torch.get_rng_state()
        step = learner.act(agent, obs, deterministic=True)
        assert torch.equal(rng, torch.get_rng_state())
        assert torch.equal(step.actions, torch.zeros_like(step.actions))
        expected = -model.config.action_dim * math.log(2 * math.pi) / 2
        torch.testing.assert_close(step.log_probs, torch.full((2,), expected))
        # A broken critic cannot affect actor-only execution, even for a singleton.
        with torch.no_grad():
            model.value_head.weight.fill_(float('nan'))
        torch.testing.assert_close(learner.act(agent, obs[:1], deterministic=True).log_probs,
                                   step.log_probs[:1])
        with pytest.raises(TypeError):
            learner.act(agent, obs, critic_states=torch.ones(2, 30))


def test_separate_actor_critic_gradients_and_centralized_information():
    learner = small_learner()
    model = learner.models['Yaw']
    obs, states = torch.ones(2, 2), torch.ones(2, 5)
    actions = model.policy_mean(obs).detach() + 1
    (-model.policy_log_probs(obs, actions).mean()).backward()
    assert model.policy_head.weight.grad.abs().sum() > 0
    assert model.log_std.grad is not None
    assert all(p.grad is None for p in model.value_trunk.parameters())
    assert model.value_head.weight.grad is None
    model.zero_grad(set_to_none=True)
    model.predict_values(states).square().mean().backward()
    assert model.value_head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.policy_trunk.parameters())
    assert model.policy_head.weight.grad is None and model.log_std.grad is None
    # Critic responds to Translation information; Yaw policy still sees only obs.
    old_actions = learner.act('Yaw', obs, deterministic=True).actions
    old_values = learner.predict_values('Yaw', states)
    states[:, :3] += 7
    assert not torch.equal(learner.predict_values('Yaw', states), old_values)
    torch.testing.assert_close(learner.act('Yaw', obs, deterministic=True).actions, old_actions)


@pytest.mark.parametrize('normalize', [False, True])
@pytest.mark.parametrize('shuffle', [False, True])
def test_full_two_agent_updates_fixed_targets_and_normalizers(normalize, shuffle):
    learner = small_learner(normalize=normalize, shuffle_minibatches=shuffle)
    buffer = rollout(learner)
    snapshot = buffer.snapshot()
    before = {a: [p.detach().clone() for p in m.parameters()] for a, m in learner.models.items()}
    batch = learner.prepare_batch(buffer)
    saved = {a: {f.name: getattr(b, f.name).clone() for f in fields(PPOBatch)} for a, b in batch.agents.items()}
    states = {a: s.clone() for a, s in batch.critic_states.items()}
    for a in AGENTS:
        current = learner.models[a].evaluate_actions(batch.agents[a].observations, batch.agents[a].actions,
                                                     critic_states=batch.critic_states[a])
        torch.testing.assert_close(current.log_probs, batch.agents[a].old_log_probs)
        torch.testing.assert_close(current.values, batch.agents[a].old_values)
        assert all(not t.requires_grad for t in saved[a].values())
        if normalize:
            for scalers in (learner.observation_normalizers, learner.state_normalizers, learner.value_normalizers):
                assert scalers[a].count.item() == 1
    with pytest.raises(RuntimeError):
        learner.act('Yaw', torch.ones(1, 2))
    with pytest.raises(RuntimeError):
        learner.predict_values('Yaw', torch.ones(1, 5))
    reports = learner.optimize(batch, generator=torch.Generator().manual_seed(11))
    for a in AGENTS:
        report = reports[a]
        assert report.optimizer_steps == 4 and report.samples == 4
        assert all(math.isfinite(v) for v in vars(report).values())
        model = learner.models[a]
        assert any(not torch.equal(p, q) for p, q in zip(before[a], model.parameters()))
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert model.policy_head.weight.grad.abs().sum() > 0
        assert model.value_head.weight.grad.abs().sum() > 0
        grad_norm = sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None).sqrt()
        assert grad_norm <= learner.configs[a].max_grad_norm + 1e-5
        for name, original in saved[a].items():
            assert torch.equal(getattr(batch.agents[a], name), original)
        assert torch.equal(batch.critic_states[a], states[a])
        assert torch.equal(buffer.snapshot().agents[a].actions, snapshot.agents[a].actions)
        if normalize:
            for scalers in (learner.observation_normalizers, learner.state_normalizers, learner.value_normalizers):
                assert scalers[a].count.item() == 5
    with pytest.raises(ValueError):
        learner.optimize(batch)
    with pytest.raises(ValueError):
        learner.prepare_batch(buffer)
    later = learner.prepare_batch(rollout(learner))
    for a in AGENTS:
        evaluation = learner.models[a].evaluate_actions(later.agents[a].observations, later.agents[a].actions,
                                                        critic_states=later.critic_states[a])
        torch.testing.assert_close(evaluation.log_probs, later.agents[a].old_log_probs)
        torch.testing.assert_close(evaluation.values, later.agents[a].old_values, atol=1e-6, rtol=1e-5)
    learner.optimize(later)


def test_individual_rewards_boundaries_and_hand_calculated_returns():
    learner = small_learner(gamma=1, gae_lambda=1, reward_scale=0.5,
                            normalize_advantages=False, bootstrap_truncations=True)
    buffer = learner.new_buffer(2)
    for t in range(2):
        step = transition(learner, t)
        step.update(rewards={'Translation': torch.tensor([[2., 4.], [6., 8.]][t]),
                             'Yaw': torch.tensor([[10., 20.], [30., 40.]][t])},
                    values={'Translation': torch.tensor([[0.25, 0.5], [1., 2.]][t]),
                            'Yaw': torch.tensor([[1., 2.], [3., 4.]][t])},
                    terminated={'Translation': torch.tensor([[False, True], [True, False]][t]),
                                'Yaw': torch.tensor([[False, False], [True, True]][t])},
                    truncated={'Translation': torch.zeros(2, dtype=torch.bool),
                               'Yaw': torch.tensor([[True, False], [False, False]][t])},
                    next_values={'Translation': torch.tensor([[1., 999.], [float('nan'), 5.]][t]),
                                 'Yaw': torch.tensor([[999., 4.], [float('nan'), float('nan')]][t])},
                    truncation_values={'Translation': None, 'Yaw': torch.tensor([7., float('nan')])})
        buffer.add(**step)
    batch = learner.prepare_batch(buffer)
    torch.testing.assert_close(batch.agents['Translation'].returns, torch.tensor([4., 2., 3., 9.]))
    torch.testing.assert_close(batch.agents['Translation'].advantages, torch.tensor([3.75, 1.5, 2., 7.]))
    torch.testing.assert_close(batch.agents['Yaw'].returns, torch.tensor([12., 30., 15., 20.]))
    torch.testing.assert_close(batch.agents['Yaw'].advantages, torch.tensor([11., 28., 12., 16.]))
    assert buffer.snapshot().agents['Yaw'].rewards[0, 0] == 10


def test_normalizer_statistics_are_distinct_and_update_once():
    learner = small_learner(normalize=True)
    buffer = rollout(learner)
    batch = learner.prepare_batch(buffer)
    expected_states_mean = batch.raw_states.double().sum(0) / 5
    expected_values_means = {a: b.reward_returns.double().sum() / 5 for a, b in batch.agents.items()}
    learner.optimize(batch)
    for a in AGENTS:
        torch.testing.assert_close(learner.state_normalizers[a].mean, expected_states_mean)
        torch.testing.assert_close(learner.value_normalizers[a].mean.squeeze(), expected_values_means[a])
        torch.testing.assert_close(learner.observation_normalizers[a].mean,
                                   batch.agents[a].raw_observations.double().sum(0) / 5)
    assert learner.state_normalizers['Translation'] is not learner.state_normalizers['Yaw']
    assert not torch.equal(learner.value_normalizers['Translation'].mean, learner.value_normalizers['Yaw'].mean)


def test_independent_optimizers_match_analytic_first_adam_steps():
    models = {a: ActorCritic(NetworkConfig(1, d, (), shared=False, critic_observation_dim=2)).double()
              for a, d in (('Translation', 2), ('Yaw', 1))}
    for model in models.values():
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
    config = MAPPOConfig(horizon=1, learning_epochs=1, mini_batches=1, learning_rate=0.01,
                         normalize_observations=False, normalize_states=False, normalize_values=False,
                         normalize_advantages=False, reward_scale=1, value_clip=None,
                         adaptive_lr=False, max_grad_norm=100)
    learner = MAPPOLearner(models, {a: config for a in AGENTS})
    buffer = learner.new_buffer(1)
    obs = {a: torch.ones(1, 1, dtype=torch.float64) for a in AGENTS}
    actions = {a: torch.ones(1, m.config.action_dim, dtype=torch.float64) for a, m in models.items()}
    buffer.add(states=torch.ones(1, 2, dtype=torch.float64), observations=obs, actions=actions,
               log_probs={a: m.policy_log_probs(obs[a], actions[a]) for a, m in models.items()},
               values={a: torch.zeros(1, dtype=torch.float64) for a in AGENTS},
               rewards={'Translation': torch.tensor([1.], dtype=torch.float64),
                        'Yaw': torch.tensor([-2.], dtype=torch.float64)},
               terminated={a: torch.tensor([True]) for a in AGENTS},
               truncated={a: torch.tensor([False]) for a in AGENTS})
    reports = learner.update(buffer)
    for a, reward in (('Translation', 1), ('Yaw', -2)):
        expected_actor = 0.01 * reward / (abs(reward) + 1e-8)
        expected_critic = 0.02 * reward / (2 * abs(reward) + 1e-8)
        torch.testing.assert_close(models[a].policy_head.weight,
                                   torch.full_like(models[a].policy_head.weight, expected_actor), atol=1e-12, rtol=0)
        torch.testing.assert_close(models[a].value_head.weight,
                                   torch.full_like(models[a].value_head.weight, expected_critic), atol=1e-12, rtol=0)
        assert reports[a].policy_loss == pytest.approx(-reward)
        assert reports[a].value_loss == pytest.approx(reward**2)


def test_translation_rewards_cannot_change_yaw_updates():
    first = small_learner(normalize=True, normalize_advantages=False)
    second = small_learner(normalize=True, normalize_advantages=False)
    buffers = [first.new_buffer(2), second.new_buffer(2)]
    for t in range(2):
        step = transition(first, t)
        buffers[0].add(**step)
        step['rewards']['Translation'] = -5 * step['rewards']['Translation']
        buffers[1].add(**step)
    reports = [learner.update(buffer) for learner, buffer in zip((first, second), buffers)]
    assert reports[0]['Yaw'] == reports[1]['Yaw']
    for left, right in zip(first.models['Yaw'].parameters(), second.models['Yaw'].parameters()):
        assert torch.equal(left, right)
    assert any(not torch.equal(left, right) for left, right in zip(
        first.models['Translation'].parameters(), second.models['Translation'].parameters()))
    for learner in (first, second):
        optimizer_parameters = [{id(p) for group in learner.optimizers[a].param_groups for p in group['params']}
                                for a in AGENTS]
        assert optimizer_parameters[0].isdisjoint(optimizer_parameters[1])


def test_agent_specific_settings_and_scheduler_cadence():
    base = small_learner()
    models = base.models
    with torch.no_grad():
        for model in models.values():
            for parameter in model.parameters():
                parameter.zero_()
    configs = {'Translation': replace(base.configs['Translation'], learning_epochs=1, mini_batches=1),
               'Yaw': replace(base.configs['Yaw'], learning_epochs=3, mini_batches=2,
                              learning_rate=1e-4, normalize_states=True)}
    learner = MAPPOLearner(models, configs)
    buffer = learner.new_buffer(2)
    for t in range(2):
        step = transition(learner, t)
        for rewards in step['rewards'].values():
            rewards.zero_()
        buffer.add(**step)
    reports = learner.update(buffer)
    assert reports['Translation'].optimizer_steps == 1
    assert reports['Yaw'].optimizer_steps == 6
    assert reports['Translation'].learning_rate == pytest.approx(3e-4 * 1.5)
    assert reports['Yaw'].learning_rate == pytest.approx(1e-4 * 1.5**3)
    assert learner.state_normalizers['Translation'] is None
    assert learner.state_normalizers['Yaw'].count.item() == 5
    assert all(p.eq(0).all() for model in models.values() for p in model.parameters())


def test_invalid_critic_inputs_and_uneven_rollouts():
    learner = small_learner(horizon=1, mini_batches=2)
    model = learner.models['Yaw']
    with pytest.raises(ValueError, match='required'):
        model(torch.zeros(1, 2))
    with pytest.raises(ValueError, match='align'):
        model(torch.zeros(1, 2), critic_states=torch.zeros(2, 5))
    with pytest.raises(ValueError, match='dimension'):
        learner.predict_values('Yaw', torch.zeros(2, 4))
    with pytest.raises(ValueError, match='divisible'):
        learner.prepare_batch(rollout(learner, n=1))
    with pytest.raises(ValueError):
        NetworkConfig(shared=False, critic_observation_dim=0)
    with pytest.raises(ValueError):
        NetworkConfig(shared=False, critic_hidden_sizes=(0,))


@pytest.mark.parametrize('bootstrap', [False, True])
def test_missing_yaw_final_values_and_atomic_preparation(bootstrap):
    learner = small_learner(normalize=True, bootstrap_truncations=bootstrap)
    buffer = learner.new_buffer(2)
    for t in range(2):
        step = transition(learner, t)
        step['terminated']['Yaw'].zero_()
        step['truncated']['Yaw'].fill_(True)
        buffer.add(**step)
    if bootstrap:
        with pytest.raises(ValueError, match='truncation_values'):
            learner.prepare_batch(buffer)
        assert not buffer.prepared and learner._pending is None
        assert learner.value_normalizers['Translation'].count == 1
        learner.act('Translation', torch.zeros(1, 3))
    else:
        batch = learner.prepare_batch(buffer)
        torch.testing.assert_close(batch.agents['Yaw'].reward_returns, torch.tensor([.02, .04, .03, .05]))


def test_invalid_models_configuration_and_buffers():
    learner = small_learner()
    with pytest.raises(ValueError):
        MAPPOLearner({'Translation': learner.models['Translation']})
    with pytest.raises(ValueError, match='storage'):
        MAPPOLearner({a: learner.models['Translation'] for a in AGENTS})
    with pytest.raises(ValueError, match='horizon'):
        MAPPOLearner(learner.models, {'Translation': MAPPOConfig(), 'Yaw': MAPPOConfig(horizon=3)})
    with pytest.raises(ValueError):
        MAPPOLearner({a: ActorCritic() for a in AGENTS})
    with pytest.raises(ValueError):
        MAPPOLearner({'Translation': learner.models['Translation'], 'Yaw': learner.models['Yaw'].double()})
    with pytest.raises(ValueError):
        PPOLearner(learner.models['Translation'])
    with pytest.raises(ValueError):
        MAPPOConfig(state_clip=0)
    with pytest.raises(TypeError):
        MAPPOConfig(normalize_states=1)
    with pytest.raises(ValueError):
        NetworkConfig(critic_observation_dim=30)
    with pytest.raises(ValueError):
        learner.act('typo', torch.zeros(1, 3))
    with pytest.raises(ValueError):
        learner.prepare_batch(learner.new_buffer(2))
    wrong = MultiAgentRolloutBuffer(2, 2, {'Translation': (3, 2), 'Yaw': (2, 1)}, 6)
    with pytest.raises(ValueError, match='dimensions'):
        learner.prepare_batch(wrong)


@pytest.mark.parametrize('kind', ['missing_agent', 'extra_agent', 'shape', 'dtype', 'flags', 'state', 'bootstrap_keys'])
def test_rollout_rejects_inconsistent_data_without_partial_append(kind):
    learner = small_learner()
    buffer = learner.new_buffer(2)
    step = transition(learner)
    if kind == 'missing_agent':
        step['actions'].pop('Yaw')
    elif kind == 'extra_agent':
        step['rewards']['Team'] = torch.ones(2)
    elif kind == 'shape':
        step['observations']['Yaw'] = torch.zeros(3, 2)
    elif kind == 'dtype':
        step['actions']['Yaw'] = step['actions']['Yaw'].double()
    elif kind == 'flags':
        step['terminated']['Yaw'] = torch.zeros(2)
    elif kind == 'state':
        step['states'] = torch.zeros(2, 4)
    else:
        step['next_values'].pop('Yaw')
    with pytest.raises((ValueError, TypeError)):
        buffer.add(**step)
    assert buffer.position == 0
    buffer.add(**transition(learner))
    buffer.add(**transition(learner, 1))
    assert buffer.snapshot().states.shape == (2, 2, 5)


def test_rollout_ownership_alignment_and_minibatches():
    learner = small_learner()
    buffer = learner.new_buffer(2)
    step = transition(learner)
    original = step['states'].clone()
    buffer.add(**step)
    for value in step.values():
        for tensor in value.values() if isinstance(value, dict) else (value,):
            tensor.zero_()
    buffer.add(**transition(learner, 1))
    saved = buffer.snapshot()
    torch.testing.assert_close(saved.states[0], original)
    torch.testing.assert_close(saved.agents['Translation'].observations[0], original[:, :3])
    torch.testing.assert_close(saved.agents['Yaw'].observations[0], original[:, 3:])
    saved.states.zero_()
    saved.agents['Translation'].observations.zero_()
    torch.testing.assert_close(buffer.snapshot().states[0], original)
    batch = learner.prepare_batch(buffer)
    seen = []
    for ids in minibatch_indices(4, 2, shuffle=True, generator=torch.Generator().manual_seed(13)):
        seen.extend(ids.tolist())
        torch.testing.assert_close(batch.raw_states[ids, :3], batch.agents['Translation'].observations[ids])
        torch.testing.assert_close(batch.raw_states[ids, 3:], batch.agents['Yaw'].observations[ids])
    assert sorted(seen) == list(range(4))
    buffer.clear()
    assert buffer.position == 0 and not buffer.prepared
    with pytest.raises(ValueError):
        buffer.snapshot()
