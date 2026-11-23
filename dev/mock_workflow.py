"""Mock-only workflow verification: python -m dev.mock_workflow --help."""

import argparse
import json

from uav_control.checkpoints import load_checkpoint, resume_training, save_checkpoint
from uav_control.evaluation import evaluate
from uav_control.mappo import MAPPOLearner
from uav_control.networks import ActorCritic
from uav_control.ppo import PPOLearner
from uav_control.training import Trainer, seed_everything

from .mock_environments import MultiAgentMock, SingleAgentMock


def main(argv=None):
    parser = argparse.ArgumentParser(description='CPU mock verification only; not Isaac Lab UAV training.')
    parser.add_argument('--algorithm', choices=('ppo', 'mappo'), default='ppo')
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--updates', type=int, default=2, help='Additional complete updates')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint', help='Output checkpoint, preferably under outputs/')
    parser.add_argument('--resume', help='Load a trusted local checkpoint')
    parser.add_argument('--exact-resume', action='store_true', help='Restore mock interaction as well as learning state')
    parser.add_argument('--evaluate', action='store_true', help='Evaluation only; load --resume if provided')
    parser.add_argument('--eval-steps', type=int, default=20)
    args = parser.parse_args(argv)
    if args.num_envs <= 0 or args.updates < 0 or args.eval_steps <= 0:
        parser.error('num-envs/eval-steps must be positive and updates nonnegative')
    if args.exact_resume and (not args.resume or args.evaluate):
        parser.error('--exact-resume requires training with --resume')
    if args.evaluate and args.checkpoint:
        parser.error('--checkpoint is a training output; use --resume for evaluation input')
    seed_everything(args.seed)
    loaded = load_checkpoint(args.resume) if args.resume else None
    learner = loaded.learner if loaded else (MAPPOLearner() if args.algorithm == 'mappo' else PPOLearner(ActorCritic()))
    if isinstance(learner, MAPPOLearner) != (args.algorithm == 'mappo'):
        parser.error('Checkpoint algorithm does not match --algorithm')
    factory = MultiAgentMock if args.algorithm == 'mappo' else SingleAgentMock
    env = factory(args.num_envs, dtype=learner.dtype)
    if args.evaluate:
        print(json.dumps(dict(mock_only=True, evaluation=evaluate(learner, env, max_steps=args.eval_steps))))
    else:
        trainer = resume_training(loaded, env, exact=args.exact_resume) if loaded else Trainer(learner, env)
        with trainer:
            for report in trainer.run(args.updates):
                print(json.dumps(dict(mock_only=True, **report)))
            if args.checkpoint:
                save_checkpoint(args.checkpoint, trainer, include_environment=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
