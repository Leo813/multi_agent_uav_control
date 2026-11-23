import pytest
import torch

from uav_control.networks import ActorCritic, NetworkConfig


def test_original_sarl_shared_architecture_and_inference():
    torch.manual_seed(3)
    model = ActorCritic()
    assert sum(p.numel() for p in model.parameters()) == 5513
    assert model.value_trunk is None
    assert [m.out_features for m in model.policy_trunk if isinstance(m, torch.nn.Linear)] == [64, 64]
    assert sum(isinstance(m, torch.nn.ELU) for m in model.policy_trunk) == 2
    obs = torch.arange(45, dtype=torch.float32).reshape(3, 15) / 45
    output = model(obs)
    deterministic = model.act(obs, deterministic=True)
    torch.testing.assert_close(deterministic.actions, output.mean)
    assert deterministic.values.shape == (3,)
    assert deterministic.actions.shape == (3, 4)
    assert not deterministic.actions.requires_grad
    torch.manual_seed(12)
    first = model.act(obs)
    torch.manual_seed(12)
    second = model.act(obs)
    assert torch.equal(first.actions, second.actions)
    assert not torch.equal(first.actions, deterministic.actions)
    torch.testing.assert_close(model.evaluate_actions(obs, first.actions).log_probs, first.log_probs)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("activation", ["elu", "relu", "tanh"])
def test_configurable_dimensions_and_gradients(shared, activation):
    torch.manual_seed(4)
    model = ActorCritic(NetworkConfig(9, 1, (8, 4), activation=activation, shared=shared)).double()
    obs = torch.ones(2, 9, dtype=torch.float64)
    output = model(obs)
    result = model.evaluate_actions(obs, output.mean.detach() + 0.7)
    loss = -result.log_probs.mean() + (result.values - 2).square().mean() - 0.1 * result.entropy.mean()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert model.policy_head.weight.grad.abs().sum() > 0
    assert model.value_head.weight.grad.abs().sum() > 0
    assert model.log_std.grad.abs().sum() > 0


def test_shared_trunk_receives_both_gradients_and_fixed_std():
    torch.manual_seed(9)
    model = ActorCritic(NetworkConfig(2, 1, (3,), trainable_log_std=False))
    output = model(torch.ones(1, 2))
    policy_grad = torch.autograd.grad(output.mean.sum(), model.policy_trunk[0].weight, retain_graph=True)[0]
    value_grad = torch.autograd.grad(output.values.sum(), model.policy_trunk[0].weight)[0]
    assert policy_grad.abs().sum() > 0 and value_grad.abs().sum() > 0
    assert not model.log_std.requires_grad
    linear = ActorCritic(NetworkConfig(2, 1, ()))
    assert linear(torch.zeros(1, 2)).values.shape == (1,)


@pytest.mark.parametrize("kwargs", [dict(observation_dim=0), dict(action_dim=True),
                                     dict(hidden_sizes=(0,)), dict(activation="unknown"),
                                     dict(shared=1), dict(initial_log_std=50)])
def test_invalid_network_config(kwargs):
    with pytest.raises((ValueError, TypeError)):
        NetworkConfig(**kwargs)


def test_invalid_observations():
    model = ActorCritic()
    for obs in (torch.zeros(15), torch.zeros(2, 14), torch.zeros(2, 15, dtype=torch.float64)):
        with pytest.raises(ValueError):
            model(obs)
