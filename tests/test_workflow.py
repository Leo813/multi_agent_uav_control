from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest
import torch

from dev.mock_environments import MultiAgentMock, SingleAgentMock
from uav_control.checkpoints import configuration, load_checkpoint, resume_training, save_checkpoint
from uav_control.collection import MAPPOCollector, PPOCollector
from uav_control.evaluation import evaluate
from uav_control.mappo import AGENTS, MAPPOConfig, MAPPOLearner
from uav_control.networks import ActorCritic, NetworkConfig
from uav_control.ppo import PPOConfig, PPOLearner
from uav_control.training import Trainer, seed_everything


def make_learner(multi=False, *, seed=19, normalize=True, bootstrap=False, horizon=4, shuffle=False):
    seed_everything(seed)
    common = dict(horizon=horizon, learning_epochs=2, mini_batches=2, bootstrap_truncations=bootstrap,
                  normalize_observations=normalize, normalize_values=normalize, shuffle_minibatches=shuffle)
    if multi:
        models = {a: ActorCritic(NetworkConfig(o, d, (8,), shared=False, critic_observation_dim=30))
                  for a, o, d in (('Translation', 21, 4), ('Yaw', 9, 1))}
        return MAPPOLearner(models, {a: MAPPOConfig(normalize_states=normalize, **common) for a in AGENTS})
    return PPOLearner(ActorCritic(NetworkConfig(hidden_sizes=(8,))), PPOConfig(**common))


def make_env(multi, n=3, **kwargs):
    return (MultiAgentMock if multi else SingleAgentMock)(n, **kwargs)


def assert_tree_equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for x, y in zip(left, right):
            assert_tree_equal(x, y)
    else:
        assert left == right


def learning_state(learner):
    # Public state dictionaries; independent of checkpoint serialization helpers.
    if isinstance(learner, MAPPOLearner):
        return {a: dict(model=learner.models[a].state_dict(), optimizer=learner.optimizers[a].state_dict(),
                        obs=None if learner.observation_normalizers[a] is None else learner.observation_normalizers[a].state_dict(),
                        states=None if learner.state_normalizers[a] is None else learner.state_normalizers[a].state_dict(),
                        values=None if learner.value_normalizers[a] is None else learner.value_normalizers[a].state_dict())
                for a in AGENTS}
    return dict(model=learner.model.state_dict(), optimizer=learner.optimizer.state_dict(),
                obs=None if learner.observation_normalizer is None else learner.observation_normalizer.state_dict(),
                values=None if learner.value_normalizer is None else learner.value_normalizer.state_dict())


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return type(value)((k, clone_tree(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return type(value)(clone_tree(v) for v in value)
    return value


@pytest.mark.parametrize('multi', [False, True])
def test_collection_copies_raw_actions_observations_and_keeps_alignment(multi):
    learner = make_learner(multi, normalize=False)
    models = learner.models if multi else {'policy': learner.model}
    with torch.no_grad():
        for model in models.values():
            model.policy_head.weight.zero_()
            model.policy_head.bias.fill_(5)
            model.log_std.fill_(-20)
    env = make_env(multi)
    collector = (MAPPOCollector if multi else PPOCollector)(learner, env)
    try:
        buffer, episodes = collector.collect()
        snapshot = buffer.snapshot()
        per_agent = snapshot.agents if multi else {'policy': snapshot}
        for a, data in per_agent.items():
            assert data.actions.eq(5).all()
            assert env.last_actions[a].eq(0.5).all()
            assert not data.observations.requires_grad
            model = models[a]
            logp = model.policy_log_probs(data.observations.reshape(-1, model.config.observation_dim),
                                          data.actions.reshape(-1, model.config.action_dim))
            torch.testing.assert_close(logp, data.log_probs.flatten())
            assert data.observations[0, :, 0].eq(0).all()
        if multi:
            torch.testing.assert_close(snapshot.states, torch.cat([per_agent[a].observations for a in AGENTS], -1))
            assert not torch.equal(per_agent['Translation'].rewards, per_agent['Yaw'].rewards)
            assert not torch.equal(per_agent['Translation'].terminated, per_agent['Yaw'].terminated)
        else:
            # Four clipped actions sum to 2: first x=0.12, first reward=1-0.12^2.
            torch.testing.assert_close(snapshot.rewards[0], torch.full((3,), 1 - .12**2))
            assert snapshot.terminated[1, 0] and snapshot.truncated[2, 1]
            assert snapshot.observations[2, 0, 0] == 0  # row 0 reset while row 1 continued
        last = clone_tree(collector.current.observations if multi else collector.current)
        learner.update(buffer)
        following, _ = collector.collect()
        for a, data in (following.snapshot().agents if multi else {'policy': following.snapshot()}).items():
            torch.testing.assert_close(data.observations[0], last[a] if multi else last)
        assert env.reset_calls == 1 and collector.vector_steps == 8
        assert episodes
    finally:
        collector.close()


@pytest.mark.parametrize('multi', [False, True])
@pytest.mark.parametrize('bootstrap', [False, True])
def test_final_pre_reset_values_not_post_reset_values(multi, bootstrap):
    learner = make_learner(multi, normalize=False, bootstrap=bootstrap, horizon=4)
    # Linear centralized/local values V(x)=x[0] with no normalization.
    if multi:
        for a in AGENTS:
            model = ActorCritic(NetworkConfig(21 if a == 'Translation' else 9, 4 if a == 'Translation' else 1,
                                               (), shared=False, critic_observation_dim=30))
            learner.models[a] = model
    else:
        learner.model = ActorCritic(NetworkConfig(hidden_sizes=()))
    models = learner.models.values() if multi else (learner.model,)
    with torch.no_grad():
        for model in models:
            model.value_head.weight.zero_()
            model.value_head.weight[0, 0] = 1
            model.value_head.bias.zero_()
    # No optimizer use after these test-only critic replacements.
    env = make_env(multi)
    collector = (MAPPOCollector if multi else PPOCollector)(learner, env)
    buffer, _ = collector.collect()
    data = buffer.snapshot().agents['Translation'] if multi else buffer.snapshot()
    assert data.truncated[2, 1] and not data.terminated[2, 1]
    assert data.next_values[2, 1].isnan()
    assert data.next_values[1, 0].isnan()  # true termination never bootstraps
    if bootstrap:
        expected = data.observations[2, 1, 0] + .1 + .01 * data.actions[2, 1].clamp(-.5, .5).sum()
        if multi:
            expected += .005 * buffer.snapshot().agents['Yaw'].actions[2, 1].clamp(-.5, .5).sum()
        torch.testing.assert_close(data.truncation_values[2, 1], expected)
        next_observation = data.observations[3, 1, 0]
        assert next_observation == 0  # terminal value was not taken from this reset observation
    else:
        assert data.truncation_values.isnan().all()
    collector.close()


@pytest.mark.parametrize('multi', [False, True])
@pytest.mark.parametrize('mode', ['absent', 'invalid_mask'])
def test_required_final_information_failure_closes_training(multi, mode):
    learner = make_learner(multi, bootstrap=True)
    env = make_env(multi, expose_final=mode != 'absent')
    if mode == 'invalid_mask':
        step = env.step
        def invalid(actions):
            result = step(actions)
            if multi:
                for mask in result.final_masks.values():
                    mask.zero_()
            else:
                result.final_mask.zero_()
            return result
        env.step = invalid
    trainer = Trainer(learner, env)
    with pytest.raises(ValueError, match='final pre-reset'):
        trainer.run(1)
    assert env.closed and not trainer.at_boundary


@pytest.mark.parametrize('multi', [False, True])
def test_disabled_timeout_bootstrap_does_not_need_final_information(multi):
    learner = make_learner(multi)
    env = make_env(multi, expose_final=False)
    with Trainer(learner, env) as trainer:
        reports = trainer.run(2)
        assert trainer.progress.updates == 2
        assert trainer.progress.vector_steps == 8
        assert trainer.progress.transitions == 24
        assert env.reset_calls == 1
        for report in reports:
            for diagnostics in report['optimization'].values():
                assert all(torch.isfinite(torch.tensor(v)) for v in diagnostics.values())
    assert env.closed


@pytest.mark.parametrize('multi', [False, True])
@pytest.mark.parametrize('normalize', [False, True])
def test_checkpoint_roundtrip_and_exact_continuation(tmp_path, multi, normalize):
    learner = make_learner(multi, normalize=normalize, bootstrap=True, shuffle=True)
    env = make_env(multi)
    path = tmp_path / 'resume.pt'
    with Trainer(learner, env) as original:
        original.run(1)
        save_checkpoint(path, original, include_environment=True)
        saved_state = clone_tree(learning_state(learner))
        expected_reports = original.run(2)
        expected_state = clone_tree(learning_state(learner))
        expected_env = env.state_dict()
        expected_observations = clone_tree(original.collector.current.observations if multi else original.collector.current)
        expected_rewards = clone_tree(original.collector.rewards)
        expected_lengths = clone_tree(original.collector.lengths)
        expected_progress = asdict(original.progress)
        rng = torch.get_rng_state().clone()
        python_next = random.random()
        # Deliberately perturb both RNGs before loading/resuming.
        torch.rand(7)
        random.random()
        loaded = load_checkpoint(path, device='cpu', expected_learner=learner)
        assert_tree_equal(learning_state(loaded.learner), saved_state)
        assert configuration(loaded.learner) == configuration(learner)
        restored_env = make_env(multi)
        with resume_training(loaded, restored_env, exact=True) as resumed:
            assert restored_env.reset_calls == 1
            actual_reports = resumed.run(2)
            assert_tree_equal(actual_reports, expected_reports)
            assert_tree_equal(learning_state(resumed.learner), expected_state)
            assert_tree_equal(restored_env.state_dict(), expected_env)
            assert_tree_equal(resumed.collector.current.observations if multi else resumed.collector.current,
                              expected_observations)
            assert_tree_equal(resumed.collector.rewards, expected_rewards)
            assert_tree_equal(resumed.collector.lengths, expected_lengths)
            assert asdict(resumed.progress) == expected_progress
            assert torch.equal(torch.get_rng_state(), rng)
            assert random.random() == python_next


@pytest.mark.parametrize('multi', [False, True])
def test_learner_only_resume_starts_new_interaction(tmp_path, multi):
    path = tmp_path / 'learner.pt'
    with Trainer(make_learner(multi), make_env(multi)) as trainer:
        trainer.run(1)
        save_checkpoint(path, trainer)
    loaded = load_checkpoint(path)
    with pytest.raises(ValueError, match='snapshot'):
        resume_training(loaded, make_env(multi), exact=True)
    env = make_env(multi, n=2)
    with resume_training(loaded, env) as resumed:
        assert resumed.collector.current is None
        resumed.run(1)
        assert asdict(resumed.progress) == dict(updates=2, vector_steps=8, transitions=20)
        assert env.reset_calls == 1 and env.steps == 4


@pytest.mark.parametrize('multi', [False, True])
def test_deterministic_evaluation_is_actor_only_and_frozen(tmp_path, multi):
    path = tmp_path / 'trained.pt'
    with Trainer(make_learner(multi), make_env(multi)) as trainer:
        trainer.run(1)
        save_checkpoint(path, trainer)
    learner = load_checkpoint(path).learner
    before = clone_tree(learning_state(learner))
    def forbidden(*args, **kwargs):
        raise AssertionError('Evaluation must not invoke value prediction or optimization')
    learner.predict_values = forbidden
    learner.update = forbidden
    models = learner.models.values() if multi else (learner.model,)
    for model in models:
        model.value_head.forward = forbidden
    rng = torch.get_rng_state().clone()
    env1, env2 = make_env(multi), make_env(multi)
    first = evaluate(learner, env1, max_steps=9)
    second = evaluate(learner, env2, max_steps=9)
    assert first == second
    assert first['transitions'] == 27
    assert all(v['episodes'] > 0 and v['mean_length'] >= 2 for v in first['agents'].values())
    assert torch.equal(torch.get_rng_state(), rng)
    assert_tree_equal(learning_state(learner), before)
    assert env1.closed and env2.closed
    limited = evaluate(learner, make_env(multi), max_steps=100, max_episodes=1)
    assert limited['vector_steps'] == 2


@pytest.mark.parametrize('corruption', ['version', 'algorithm', 'shape', 'dtype', 'config', 'normalizer', 'optimizer', 'counter', 'rng'])
def test_invalid_checkpoints_rejected(tmp_path, corruption):
    path = tmp_path / 'invalid.pt'
    with Trainer(make_learner(), make_env(False)) as trainer:
        trainer.run(1)
        save_checkpoint(path, trainer)
    data = torch.load(path, weights_only=True)
    if corruption == 'version':
        data['format_version'] = 99
    elif corruption == 'algorithm':
        data['configuration']['algorithm'] = 'unknown'
    elif corruption == 'shape':
        data['agents']['policy']['model']['log_std'] = torch.zeros(1)
    elif corruption == 'dtype':
        data['agents']['policy']['model']['log_std'] = torch.zeros(4, dtype=torch.float64)
    elif corruption == 'config':
        data['configuration']['networks']['policy'].pop('action_dim')
    elif corruption == 'normalizer':
        data['agents']['policy']['normalizers']['value']['variance'].fill_(-1)
    elif corruption == 'optimizer':
        data['agents']['policy']['optimizer']['state'][0]['exp_avg'] = torch.zeros(999)
    elif corruption == 'counter':
        data['progress']['updates'] = -1
    else:
        data['rng']['torch'] = torch.zeros(3)
    torch.save(data, path)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        load_checkpoint(path)


def test_checkpoint_boundaries_expected_configuration_and_legacy_rejection(tmp_path):
    path = tmp_path / 'boundary.pt'
    learner = make_learner()
    with Trainer(learner, make_env(False)) as trainer:
        trainer.run(1)
        save_checkpoint(path, trainer, include_environment=True)
        with pytest.raises(ValueError, match='configuration'):
            load_checkpoint(path, expected_learner=make_learner(horizon=8))
        with pytest.raises(ValueError, match='environment count'):
            resume_training(load_checkpoint(path), SingleAgentMock(2), exact=True)
        trainer.at_boundary = False
        with pytest.raises(ValueError, match='boundary'):
            save_checkpoint(path, trainer)
    legacy = tmp_path / 'legacy.pt'
    torch.save({'a': torch.ones(1)}, legacy, _use_new_zipfile_serialization=False)
    with pytest.raises(ValueError, match='ZIP'):
        load_checkpoint(legacy)


@pytest.mark.parametrize('multi', [False, True])
def test_termination_wins_when_both_flags_are_set(multi):
    learner = make_learner(multi, bootstrap=True)
    env = make_env(multi, expose_final=False)
    original = env.step
    def both(actions):
        result = original(actions)
        if multi:
            for a in AGENTS:
                result.terminated[a].logical_or_(result.truncated[a])
        else:
            result.terminated.logical_or_(result.truncated)
        return result
    env.step = both
    with Trainer(learner, env) as trainer:
        trainer.run(1)  # No final inputs required for true terminations.


@pytest.mark.parametrize('multi', [False, True])
def test_unused_final_rows_may_be_nan(multi):
    learner = make_learner(multi, bootstrap=True)
    env = make_env(multi)
    original = env.step
    def masked(actions):
        result = original(actions)
        if multi:
            # Create separate tensors because different agents have different masks.
            for a in AGENTS:
                result.final_states[a] = result.final_states[a].clone()
                result.final_states[a][~result.final_masks[a]] = float('nan')
        else:
            result.final_observations[~result.final_mask] = float('nan')
        return result
    env.step = masked
    with Trainer(learner, env) as trainer:
        trainer.run(1)


def test_float64_checkpoint_and_evaluation_numeric_episode_summary(tmp_path):
    learner = PPOLearner(ActorCritic(NetworkConfig(hidden_sizes=())).double(),
                          PPOConfig(horizon=4, mini_batches=2, learning_epochs=1))
    path = tmp_path / 'double.pt'
    with Trainer(learner, SingleAgentMock(2, dtype=torch.float64)) as trainer:
        trainer.run(1)
        save_checkpoint(path, trainer)
    restored = load_checkpoint(path).learner
    assert restored.dtype == torch.float64
    assert_tree_equal(learning_state(restored), learning_state(learner))
    # Zero deterministic action: x=.1,.2 => rewards .99+.96=1.95 at length 2.
    with torch.no_grad():
        restored.model.policy_head.weight.zero_()
        restored.model.policy_head.bias.zero_()
    result = evaluate(restored, SingleAgentMock(1, dtype=torch.float64), max_steps=2)
    assert result['agents']['policy']['episodes'] == 1
    assert result['agents']['policy']['mean_reward'] == pytest.approx(1.95)
    assert result['agents']['policy']['mean_length'] == 2


@pytest.mark.parametrize('multi', [False, True])
def test_cli_train_resume_and_evaluate(tmp_path, multi):
    algorithm = 'mappo' if multi else 'ppo'
    path = tmp_path / f'{algorithm}.pt'
    prefix = [sys.executable, '-B', '-m', 'dev.mock_workflow', '--algorithm', algorithm, '--num-envs', '2']
    def run(*args):
        result = subprocess.run(prefix + list(args), cwd=Path(__file__).resolve().parents[1],
                                text=True, capture_output=True, timeout=90)
        assert result.returncode == 0, result.stderr
        return [json.loads(line) for line in result.stdout.splitlines()]
    first = run('--updates', '1', '--checkpoint', str(path))[-1]
    resumed = run('--updates', '1', '--resume', str(path), '--exact-resume')[-1]
    evaluated = run('--evaluate', '--resume', str(path), '--eval-steps', '5')[-1]
    assert first['mock_only'] and first['progress']['updates'] == 1
    assert resumed['progress']['updates'] == 2
    assert evaluated['evaluation']['vector_steps'] == 5
