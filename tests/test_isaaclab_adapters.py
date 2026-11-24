"""Native-interface stand-ins, not Isaac Lab execution or UAV dynamics."""

from types import SimpleNamespace

import pytest
import torch

from uav_control.isaaclab_adapters import AGENTS, IsaacLabMARLAdapter, IsaacLabSARLAdapter
from uav_control.launcher import make_learner
from uav_control.training import Trainer


class NativeStandIn:
    """Scripted native API with v2.1.0 reset timing and all/all MARL reset rule."""
    def __init__(self, multi=False, num_envs=2):
        self.multi, self.num_envs = multi, num_envs
        self.device = torch.device("cpu")
        self.unwrapped = self
        self.possible_agents = list(AGENTS)
        self.cfg = SimpleNamespace(state_space=-1)
        self.reset_calls = self.steps = self.close_calls = 0
        self.obs = {a: torch.zeros(num_envs, width) for a, width in
                    (("Translation", 21), ("Yaw", 9))} if multi else {"policy": torch.zeros(num_envs, 15)}

    def reset(self):
        self.reset_calls += 1
        return self.obs, {"ignored": True}

    def state(self):
        return torch.cat([self.obs[a] for a in self.possible_agents], -1)

    def step(self, actions):
        self.steps += 1
        self.received = actions
        for action in actions.values() if self.multi else [actions]:
            action.clamp_(-0.5, 0.5)
        # Even rows terminate, odd rows time out. Every third step is continuing.
        index = torch.arange(self.num_envs)
        terminated = (index % 2 == 0) & (self.steps % 3 != 0)
        truncated = (index % 2 == 1) & (self.steps % 3 != 0)
        rewards = torch.arange(self.num_envs, dtype=torch.float32) + 2
        for i, observation in enumerate(self.obs.values()):
            observation.fill_(self.steps + i + 1)
            observation[terminated | truncated] = -10-i  # Next episode, not final inputs.
        if self.multi:
            term = {a: terminated.clone() for a in AGENTS}
            trunc = {a: truncated.clone() for a in AGENTS}
            self.reset_buf = (term[AGENTS[0]] & term[AGENTS[1]]) | (trunc[AGENTS[0]] & trunc[AGENTS[1]])
            return self.obs, {"Translation": rewards, "Yaw": -rewards}, term, trunc, {}
        self.reset_buf = terminated | truncated
        return self.obs, rewards, terminated, truncated, {}

    def close(self):
        self.close_calls += 1


@pytest.mark.parametrize("num_envs", [1, 2, 5])
@pytest.mark.parametrize("multi", [False, True])
def test_native_reset_step_ownership_and_boundaries(multi, num_envs):
    native = NativeStandIn(multi, num_envs)
    adapter = (IsaacLabMARLAdapter if multi else IsaacLabSARLAdapter)(native)
    obs = adapter.reset()
    actions = ({a: torch.full((num_envs, d), 3.0) for a, d in (("Translation", 4), ("Yaw", 1))}
               if multi else torch.full((num_envs, 4), 3.0))
    result = adapter.step(actions)
    assert native.reset_calls == native.steps == 1
    assert adapter.device == torch.device("cpu")
    if multi:
        assert obs.states.shape == (num_envs, 30)
        torch.testing.assert_close(result.observations.states[:, :21], native.obs["Translation"])
        torch.testing.assert_close(result.observations.states[:, 21:], native.obs["Yaw"])
        assert result.final_states is result.final_masks is None
        torch.testing.assert_close(result.rewards["Translation"], -result.rewards["Yaw"])
        for a in AGENTS:
            assert (actions[a] == 3).all() and (native.received[a] == 0.5).all()
            assert (result.terminated[a] | result.truncated[a]).all()
    else:
        assert obs.shape == (num_envs, 15)
        assert result.final_observations is result.final_mask is None
        assert (actions == 3).all() and (native.received == 0.5).all()
        assert (result.observations == -10).all()
        torch.testing.assert_close(result.rewards, torch.arange(num_envs, dtype=torch.float32)+2)
        assert result.terminated[0] and not result.truncated[0]
        if num_envs > 1:
            assert result.truncated[1] and not result.terminated[1]
    adapter.close()
    adapter.close()
    assert native.close_calls == 1


@pytest.mark.parametrize("multi", [False, True])
def test_missing_final_inputs_rejected(multi):
    with pytest.raises(ValueError, match="final pre-reset"):
        (IsaacLabMARLAdapter if multi else IsaacLabSARLAdapter)(NativeStandIn(multi), bootstrap_truncations=True)


def test_marl_requires_native_concatenation_order():
    native = NativeStandIn(True)
    native.possible_agents.reverse()
    with pytest.raises(ValueError, match="Translation/Yaw"):
        IsaacLabMARLAdapter(native)
    native.possible_agents.reverse()
    native.cfg.state_space = 30
    with pytest.raises(ValueError, match="state_space"):
        IsaacLabMARLAdapter(native)


def test_individual_marl_flags_are_not_collapsed():
    native = NativeStandIn(True)
    base_step = native.step
    def step(actions):
        obs, rewards, term, trunc, extras = base_step(actions)
        term["Yaw"].zero_()
        trunc["Yaw"].zero_()
        native.reset_buf.zero_()  # all-term OR all-timeout: no physical reset.
        return obs, rewards, term, trunc, extras
    native.step = step
    result = IsaacLabMARLAdapter(native).step({"Translation": torch.zeros(2, 4), "Yaw": torch.zeros(2, 1)})
    assert result.terminated["Translation"][0]
    assert result.truncated["Translation"][1]
    assert not result.terminated["Yaw"].any() and not result.truncated["Yaw"].any()


@pytest.mark.parametrize("bad", [torch.zeros(2, 3), torch.zeros(2, 4, dtype=torch.float64)])
def test_action_contract_rejects_bad_tensors(bad):
    native = NativeStandIn()
    with pytest.raises(ValueError, match="Native tensor"):
        IsaacLabSARLAdapter(native).step(bad)
    assert native.steps == 0


@pytest.mark.parametrize("task", ["SARL", "MARL"])
def test_adapter_collectors_complete_learning_update(task):
    learner = make_learner(task, "cpu")
    native = NativeStandIn(task == "MARL")
    adapter = (IsaacLabMARLAdapter if task == "MARL" else IsaacLabSARLAdapter)(native)
    models = learner.models.values() if task == "MARL" else [learner.model]
    before = [next(model.parameters()).detach().clone() for model in models]
    with Trainer(learner, adapter) as trainer:
        report = trainer.run(1)[0]
        assert report["progress"]["updates"] == 1
        assert trainer.progress.vector_steps == native.steps
        assert native.reset_calls == 1
        if task == "MARL":
            assert report["rollout"]["Translation"]["mean_reward"] > 0
            assert report["rollout"]["Yaw"]["mean_reward"] < 0
    models = learner.models.values() if task == "MARL" else [learner.model]
    assert all(not torch.equal(old, next(model.parameters())) for old, model in zip(before, models))
    assert native.close_calls == 1
