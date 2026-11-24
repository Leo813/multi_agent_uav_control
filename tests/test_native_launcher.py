"""Startup/registration orchestration tests with no Isaac modules installed."""

from dataclasses import replace
import sys
from types import ModuleType, SimpleNamespace

import pytest

from uav_control import launcher
from uav_control.tasks import TASKS, register_tasks


class FakeAppLauncher:
    @staticmethod
    def add_app_launcher_args(parser):
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--device", default="cuda:0")


def test_registration_lazy_idempotent_and_collision_safe(monkeypatch):
    gym = ModuleType("gymnasium")
    gym.registry = {}
    def register(**kwargs):
        gym.registry[kwargs["id"]] = SimpleNamespace(**kwargs)
    gym.register = register
    monkeypatch.setitem(sys.modules, "gymnasium", gym)
    register_tasks()
    register_tasks()
    assert set(gym.registry) == {s.id for s in TASKS.values()}
    for task, spec in TASKS.items():
        assert gym.registry[spec.id].entry_point == spec.entry_point
        assert gym.registry[spec.id].kwargs == {"env_cfg_entry_point": spec.config_entry_point}
        assert gym.registry[spec.id].disable_env_checker
        assert f"uav_control.tasks.quadcopter_{task}" not in sys.modules
    gym.registry[TASKS["SARL"].id].entry_point = "another:Task"
    with pytest.raises(ValueError, match="Conflicting"):
        register_tasks()


@pytest.mark.parametrize("task", ["SARL", "MARL"])
def test_options_and_automatic_learner_selection(task):
    parser = launcher.build_parser(FakeAppLauncher)
    args = parser.parse_args(["--task", task, "--headless", "--num_envs", "64", "--updates", "10",
                              "--seed", "7", "--checkpoint-out", "outputs/test.pt"])
    launcher._validate_args(parser, args)
    assert args.headless and args.num_envs == 64 and args.updates == 10 and args.seed == 7
    learner = launcher.make_learner(task, "cpu")
    launcher.validate_learner(task, learner)
    with pytest.raises(ValueError, match="selected task"):
        launcher.validate_learner("MARL" if task == "SARL" else "SARL", learner)
    if task == "SARL":
        learner.config = replace(learner.config, bootstrap_truncations=True)
    else:
        learner.configs["Yaw"] = replace(learner.configs["Yaw"], bootstrap_truncations=True)
    with pytest.raises(ValueError, match="final pre-reset"):
        launcher.validate_learner(task, learner)


@pytest.mark.parametrize("arguments", [["--mode", "evaluate"], ["--num_envs", "0"],
                                      ["--seed", "-1"], ["--updates", "-1"]])
def test_invalid_cli_before_simulation(arguments):
    parser = launcher.build_parser(FakeAppLauncher)
    with pytest.raises(SystemExit):
        launcher._validate_args(parser, parser.parse_args(["--task", "SARL", *arguments]))


@pytest.mark.parametrize("failure", [False, True])
def test_startup_order_and_application_cleanup(monkeypatch, failure):
    events = []
    class App(FakeAppLauncher):
        def __init__(self, args):
            events.append("app started")
            self.app = SimpleNamespace(close=lambda: events.append("app closed"))
    package, module = ModuleType("isaaclab"), ModuleType("isaaclab.app")
    module.AppLauncher = App
    monkeypatch.setitem(sys.modules, "isaaclab", package)
    monkeypatch.setitem(sys.modules, "isaaclab.app", module)
    def run(args):
        assert events == ["app started"]
        events.append("workflow")
        if failure:
            raise RuntimeError("workflow failure")
    monkeypatch.setattr(launcher, "_run", run)
    if failure:
        with pytest.raises(RuntimeError, match="workflow failure"):
            launcher.main(["--task", "MARL"])
    else:
        launcher.main(["--task", "SARL"])
    assert events == ["app started", "workflow", "app closed"]


@pytest.mark.parametrize("task", ["SARL", "MARL"])
def test_launcher_train_checkpoint_resume_and_evaluate(monkeypatch, tmp_path, task):
    from test_isaaclab_adapters import NativeStandIn
    from uav_control.checkpoints import load_checkpoint

    gym = ModuleType("gymnasium")
    gym.registry = {}
    gym.register = lambda **kw: gym.registry.update({kw["id"]: SimpleNamespace(**kw)})
    created = []
    def make(identifier, *, cfg):
        assert identifier == TASKS[task].id
        assert cfg.scene.num_envs == 2 and cfg.seed == 23 and cfg.sim.device == "cpu"
        native = NativeStandIn(task == "MARL")
        created.append(native)
        return native
    gym.make = make
    monkeypatch.setitem(sys.modules, "gymnasium", gym)
    module_name, cfg_name = TASKS[task].config_entry_point.split(":")
    module = ModuleType(module_name)
    setattr(module, cfg_name, lambda: SimpleNamespace(scene=SimpleNamespace(num_envs=4096), sim=SimpleNamespace()))
    monkeypatch.setitem(sys.modules, module_name, module)
    parser = launcher.build_parser(FakeAppLauncher)
    checkpoint = tmp_path / "trained.pt"
    args = parser.parse_args(["--task", task, "--device", "cpu", "--num_envs", "2", "--seed", "23",
                              "--updates", "1", "--checkpoint-out", str(checkpoint)])
    launcher._run(args)
    assert load_checkpoint(checkpoint).progress.updates == 1
    args.checkpoint = checkpoint
    launcher._run(args)
    assert load_checkpoint(checkpoint).progress.updates == 2
    args.mode, args.checkpoint_out, args.eval_steps = "evaluate", None, 4
    launcher._run(args)
    assert len(created) == 3 and all(n.close_calls == 1 and n.reset_calls == 1 for n in created)
    assert created[-1].steps == 4


def test_environment_closed_when_adapter_construction_fails(monkeypatch):
    from test_isaaclab_adapters import NativeStandIn
    native = NativeStandIn(True)
    native.possible_agents.reverse()
    gym = ModuleType("gymnasium")
    gym.registry = {}
    gym.register = lambda **kw: gym.registry.update({kw["id"]: SimpleNamespace(**kw)})
    gym.make = lambda *a, **kw: native
    monkeypatch.setitem(sys.modules, "gymnasium", gym)
    module_name, cfg_name = TASKS["MARL"].config_entry_point.split(":")
    module = ModuleType(module_name)
    setattr(module, cfg_name, lambda: SimpleNamespace(scene=SimpleNamespace(), sim=SimpleNamespace()))
    monkeypatch.setitem(sys.modules, module_name, module)
    args = launcher.build_parser(FakeAppLauncher).parse_args(["--task", "MARL", "--device", "cpu"])
    with pytest.raises(ValueError, match="Translation/Yaw"):
        launcher._run(args)
    assert native.close_calls == 1
