"""Unified native launcher. Simulator/task imports stay inside the startup path."""

import argparse
import importlib
import json
from pathlib import Path


def build_parser(app_launcher):
    parser = argparse.ArgumentParser(description="Independent UAV PPO/MAPPO with Isaac Lab 2.1.0")
    parser.add_argument("--task", choices=("SARL", "MARL"), required=True)
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--num_envs", type=int, default=None, help="Override original task environment count")
    parser.add_argument("--updates", type=int, default=10, help="Additional complete learner updates")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, help="Trusted checkpoint for resume or evaluation")
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--legacy-telemetry-dir", type=Path, help="Opt in to synchronous legacy CSV output")
    app_launcher.add_app_launcher_args(parser)
    return parser


def _validate_args(parser, args):
    if args.num_envs is not None and args.num_envs <= 0:
        parser.error("--num_envs must be positive")
    if args.updates < 0 or args.eval_steps <= 0 or not 0 <= args.seed < 2**32:
        parser.error("Require updates >= 0, eval-steps > 0, and 0 <= seed < 2**32")
    if args.mode == "evaluate" and args.checkpoint is None:
        parser.error("Evaluation requires --checkpoint")
    if args.mode == "evaluate" and args.checkpoint_out is not None:
        parser.error("--checkpoint-out is only supported for training")


def make_learner(task, device):
    """Use existing research defaults; move models before constructing optimizers."""
    from .mappo import MAPPOLearner
    from .networks import ActorCritic, NetworkConfig
    from .ppo import PPOLearner

    if task == "SARL":
        return PPOLearner(ActorCritic(NetworkConfig()).to(device))
    if task == "MARL":
        return MAPPOLearner({a: ActorCritic(NetworkConfig(obs, act, (32, 32), shared=False,
                               critic_observation_dim=30)).to(device)
                             for a, obs, act in (("Translation", 21, 4), ("Yaw", 9, 1))})
    raise ValueError(f"Unknown task: {task}")


def validate_learner(task, learner):
    """Reject incompatible checkpoints before constructing the native environment."""
    import torch
    from .mappo import MAPPOLearner
    from .ppo import PPOLearner

    if task == "SARL" and isinstance(learner, PPOLearner):
        models = [(learner.model, learner.config, 15, 4, None)]
    elif task == "MARL" and isinstance(learner, MAPPOLearner):
        models = [(learner.models[a], learner.configs[a], obs, act, 30)
                  for a, obs, act in (("Translation", 21, 4), ("Yaw", 9, 1))]
    else:
        raise ValueError("Checkpoint learner does not match the selected task")
    for model, config, obs, act, state in models:
        c = model.config
        if (c.observation_dim, c.action_dim, c.critic_observation_dim) != (obs, act, state):
            raise ValueError("Checkpoint network dimensions do not match the selected task")
        if config.bootstrap_truncations:
            raise ValueError("Native timeout bootstrapping requires unavailable final pre-reset inputs")
    if learner.dtype != torch.float32:
        raise ValueError("Native UAV tasks require float32 learner tensors")


def _run(args):
    # Called only after AppLauncher has initialized Isaac Sim.
    import gymnasium as gym
    from .checkpoints import load_checkpoint, resume_training, save_checkpoint
    from .evaluation import evaluate
    from .isaaclab_adapters import IsaacLabMARLAdapter, IsaacLabSARLAdapter
    from .tasks import TASKS, register_tasks
    from .training import Trainer, seed_everything

    seed_everything(args.seed)
    register_tasks()
    spec = TASKS[args.task]
    module, name = spec.config_entry_point.split(":")
    cfg = getattr(importlib.import_module(module), name)()
    if args.num_envs is not None:
        cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.legacy_telemetry_dir = (str(args.legacy_telemetry_dir.expanduser().resolve())
                                if args.legacy_telemetry_dir is not None else None)
    loaded = load_checkpoint(args.checkpoint, device=args.device) if args.checkpoint else None
    learner = loaded.learner if loaded else make_learner(args.task, args.device)
    validate_learner(args.task, learner)
    native = gym.make(spec.id, cfg=cfg).unwrapped
    adapter = None
    try:
        adapter = (IsaacLabSARLAdapter if args.task == "SARL" else IsaacLabMARLAdapter)(native)
        if args.mode == "evaluate":
            print(json.dumps(evaluate(learner, adapter, max_steps=args.eval_steps)))
        else:
            trainer = resume_training(loaded, adapter, exact=False) if loaded else Trainer(learner, adapter)
            with trainer:
                for _ in range(args.updates):
                    print(json.dumps(trainer.run(1)[0]))
                if args.checkpoint_out is not None:
                    save_checkpoint(args.checkpoint_out, trainer, include_environment=False)
    finally:
        if adapter is not None:
            adapter.close()
        else:
            native.close()


def main(argv=None):
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as error:
        raise SystemExit("Use a prepared Isaac Lab 2.1.0 / Isaac Sim 4.5.0 Python 3.10 environment; "
                         "the CPU test .venv cannot launch the simulator.") from error
    parser = build_parser(AppLauncher)
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    application = AppLauncher(args).app
    try:
        _run(args)
    finally:
        application.close()
