from dataclasses import fields

import pytest
import torch

from uav_control.rollout import Rollout, RolloutBuffer, minibatch_indices


def transition(n=2):
    return dict(observations=torch.ones(n, 3), actions=torch.ones(n, 1), rewards=torch.ones(n),
                values=torch.zeros(n), log_probs=torch.zeros(n),
                terminated=torch.zeros(n, dtype=torch.bool), truncated=torch.zeros(n, dtype=torch.bool),
                next_values=torch.ones(n), truncation_values=torch.zeros(n))


def test_storage_ownership_detachment_and_snapshot_independence():
    buffer = RolloutBuffer(2, 2, 3, 1)
    step = transition()
    step['actions'].requires_grad_()
    buffer.add(**step)
    with torch.no_grad():
        for tensor in step.values():
            tensor.zero_()
    buffer.add(**step)
    saved = buffer.snapshot()
    assert saved.observations.shape == (2, 2, 3)
    assert saved.actions.shape == (2, 2, 1)
    assert saved.rewards.shape == saved.values.shape == saved.terminated.shape == (2, 2)
    assert saved.actions[0].eq(1).all() and not saved.actions.requires_grad
    saved.actions.fill_(5)
    assert buffer.snapshot().actions[0].eq(1).all()
    with pytest.raises(ValueError, match="full"):
        buffer.add(**step)
    buffer.prepared = True
    buffer.clear()
    assert buffer.position == 0 and not buffer.prepared
    with pytest.raises(ValueError, match="complete"):
        buffer.snapshot()


@pytest.mark.parametrize("shuffle", [False, True])
def test_minibatches_cover_every_time_major_sample(shuffle):
    ids = torch.arange(24).reshape(4, 3, 2)
    flat = ids.reshape(12, 2)
    generator = torch.Generator().manual_seed(32)
    parts = list(minibatch_indices(12, 4, shuffle=shuffle, generator=generator))
    order = torch.cat(parts)
    assert len(parts) == 4 and all(len(part) == 3 for part in parts)
    assert torch.equal(order.sort().values, torch.arange(12))
    torch.testing.assert_close(torch.cat([flat[i] for i in parts]).sort(dim=0).values, flat)
    if not shuffle:
        assert torch.equal(order, torch.arange(12))
    repeat = list(minibatch_indices(12, 4, shuffle=shuffle, generator=torch.Generator().manual_seed(32)))
    assert torch.equal(order, torch.cat(repeat))
    with pytest.raises(ValueError):
        list(minibatch_indices(10, 4))


@pytest.mark.parametrize("field,bad", [("observations", torch.zeros(2, 2)),
                                      ("actions", torch.zeros(2, 1, dtype=torch.float64)),
                                      ("terminated", torch.zeros(2)),
                                      ("rewards", None),
                                      ("values", torch.full((2,), float('nan')))])
def test_invalid_transition_is_atomic(field, bad):
    buffer = RolloutBuffer(1, 2, 3, 1)
    step = transition()
    step[field] = bad
    with pytest.raises((ValueError, TypeError)):
        buffer.add(**step)
    assert buffer.position == 0


def test_missing_bootstraps_recorded_explicitly():
    buffer = RolloutBuffer(1, 2, 3, 1)
    step = transition()
    step.pop('next_values')
    step.pop('truncation_values')
    buffer.add(**step)
    result = buffer.snapshot()
    assert result.next_values.isnan().all() and result.truncation_values.isnan().all()
    for field in fields(Rollout):
        assert getattr(result, field.name).device.type == 'cpu'
