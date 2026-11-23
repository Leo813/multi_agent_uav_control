"""Versioned state-dictionary checkpoints for trusted local training artifacts."""

from dataclasses import asdict, dataclass, fields
import math
import os
from pathlib import Path
import random
import tempfile
import zipfile

import torch

from .mappo import AGENTS, MAPPOConfig, MAPPOLearner
from .networks import ActorCritic, NetworkConfig
from .ppo import PPOConfig, PPOLearner
from .training import Progress, Trainer


FORMAT_VERSION = 1
SEMANTICS = 'frozen-normalizers-population-variance-v1'


def _parts(learner):
    if isinstance(learner, MAPPOLearner):
        return {a: (learner.models[a], learner.optimizers[a], learner.configs[a],
                    {'observation': learner.observation_normalizers[a], 'state': learner.state_normalizers[a],
                     'value': learner.value_normalizers[a]}) for a in AGENTS}
    if not isinstance(learner, PPOLearner):
        raise TypeError("Expected PPO or MAPPO learner")
    return {'policy': (learner.model, learner.optimizer, learner.config,
                        {'observation': learner.observation_normalizer, 'value': learner.value_normalizer})}


def configuration(learner):
    return dict(algorithm='mappo' if isinstance(learner, MAPPOLearner) else 'ppo',
                dtype=str(learner.dtype), semantics=SEMANTICS,
                networks={a: asdict(m.config) for a, (m, _, _, _) in _parts(learner).items()},
                learners={a: asdict(c) for a, (_, _, c, _) in _parts(learner).items()})


def _data_only(value):
    """Owned CPU tensors and built-in containers/scalars only; no executable objects."""
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.is_quantized:
            raise ValueError("Checkpoint tensors must be ordinary dense tensors")
        if not torch.isfinite(value).all():
            raise ValueError("Checkpoint tensors must be finite")
        return value.detach().cpu().clone()
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, dict):
        if any(type(k) not in (str, int) for k in value):
            raise ValueError("Invalid checkpoint mapping key")
        return {k: _data_only(v) for k, v in value.items()}
    if type(value) in (list, tuple):
        return type(value)(_data_only(v) for v in value)
    raise ValueError(f"Unsupported checkpoint data type: {type(value).__name__}")


def _rng_state():
    return dict(python=random.getstate(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def save_checkpoint(path, trainer: Trainer, *, include_environment=False):
    """Save only at a successful update boundary. Environment state is opt-in."""
    learner = trainer.learner
    if not trainer.at_boundary or trainer.collector.failed or learner._pending is not None:
        raise ValueError("Checkpoint requires a completed update boundary and no pending batch")
    if trainer.collector.closed:
        raise ValueError("Save before closing the trainer")
    agents = {a: dict(model=m.state_dict(), optimizer=o.state_dict(),
                      normalizers={name: None if scaler is None else scaler.state_dict()
                                   for name, scaler in scalers.items()})
              for a, (m, o, _, scalers) in _parts(learner).items()}
    interaction = trainer.collector.state_dict() if include_environment else None
    payload = _data_only(dict(format_version=FORMAT_VERSION, configuration=configuration(learner), agents=agents,
                             progress=asdict(trainer.progress), rng=_rng_state(), device=str(learner.device),
                             torch_version=str(torch.__version__),
                             interaction=interaction))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.tmp', delete=False) as file:
        temporary = Path(file.name)
    try:
        torch.save(payload, temporary)  # Modern ZIP state dictionaries; no module instances.
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"Invalid {name} fields")


def _validate_module_state(saved, expected):
    _keys(saved, expected, 'module state')
    for name, reference in expected.items():
        value = saved[name]
        if not isinstance(value, torch.Tensor) or value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(f"Incompatible tensor: {name}")


def _validate_optimizer(saved, optimizer):
    _keys(saved, ('state', 'param_groups'), 'optimizer')
    if not isinstance(saved['param_groups'], list) or len(saved['param_groups']) != 1:
        raise ValueError("Expected one Adam parameter group")
    group = saved['param_groups'][0]
    expected = optimizer.state_dict()['param_groups'][0]
    _keys(group, expected, 'optimizer group')
    if group['params'] != expected['params']:
        raise ValueError("Optimizer parameter ordering differs")
    for key in expected:
        if key not in ('params', 'lr') and group[key] != expected[key]:
            raise ValueError(f"Incompatible Adam setting: {key}")
    if type(group['lr']) not in (int, float) or group['lr'] <= 0:
        raise ValueError("Invalid adaptive learning rate")
    if not isinstance(saved['state'], dict) or not set(saved['state']).issubset(group['params']):
        raise ValueError("Invalid optimizer state keys")
    for key, parameter in zip(group['params'], optimizer.param_groups[0]['params']):
        state = saved['state'].get(key)
        if state is None:
            continue
        _keys(state, ('step', 'exp_avg', 'exp_avg_sq'), 'Adam moments')
        for name in ('exp_avg', 'exp_avg_sq'):
            value = state[name]
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise ValueError("Incompatible Adam moment tensor")
        step = state['step']
        if not isinstance(step, torch.Tensor) or step.ndim != 0 or step.item() < 0 or not float(step.item()).is_integer():
            raise ValueError("Invalid Adam step counter")
        if (state['exp_avg_sq'] < 0).any():
            raise ValueError("Negative Adam second moment")


@dataclass
class LoadedCheckpoint:
    learner: object
    progress: Progress
    interaction: dict | None
    rng: dict
    original_device: str


def load_checkpoint(path, *, device='cpu', expected_learner=None) -> LoadedCheckpoint:
    """Load a trusted project artifact on CPU, then explicitly place the learner.

    Restricted unpickling is mandatory. Legacy/TorchScript archives are rejected;
    there is no unrestricted fallback. PyTorch 2.5.1 is not a security boundary
    for hostile files, so checkpoints must come from a trusted local workflow.
    """
    with open(path, 'rb') as file:
        if file.read(4) != b'PK\x03\x04' or not zipfile.is_zipfile(file):
            raise ValueError("Only modern project ZIP checkpoints are supported")
        file.seek(0)
        with zipfile.ZipFile(file) as archive:
            if any(name.endswith('/constants.pkl') for name in archive.namelist()):
                raise ValueError("TorchScript checkpoints are not supported")
        file.seek(0)
        payload = _data_only(torch.load(file, map_location='cpu', weights_only=True))
    _keys(payload, ('format_version', 'configuration', 'agents', 'progress', 'rng', 'device',
                    'torch_version', 'interaction'), 'checkpoint')
    if type(payload['format_version']) is not int or payload['format_version'] != FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint version")
    cfg = payload['configuration']
    _keys(cfg, ('algorithm', 'dtype', 'semantics', 'networks', 'learners'), 'configuration')
    if cfg['algorithm'] not in ('ppo', 'mappo') or cfg['semantics'] != SEMANTICS:
        raise ValueError("Unsupported learner or normalization semantics")
    if cfg['dtype'] not in ('torch.float32', 'torch.float64'):
        raise ValueError("Unsupported checkpoint dtype")
    if expected_learner is not None and cfg != configuration(expected_learner):
        raise ValueError("Checkpoint configuration does not match the expected learner")
    names = AGENTS if cfg['algorithm'] == 'mappo' else ('policy',)
    for key in ('networks', 'learners'):
        _keys(cfg[key], names, key)
    _keys(payload['agents'], names, 'agents')
    model_configs, configs = {}, {}
    config_type = MAPPOConfig if cfg['algorithm'] == 'mappo' else PPOConfig
    for a in names:
        _keys(cfg['networks'][a], [f.name for f in fields(NetworkConfig)], 'network configuration')
        _keys(cfg['learners'][a], [f.name for f in fields(config_type)], 'learner configuration')
        model_configs[a] = NetworkConfig(**cfg['networks'][a])
        configs[a] = config_type(**cfg['learners'][a])
    dtype = torch.float32 if cfg['dtype'] == 'torch.float32' else torch.float64
    # Constructing a loaded policy must not unexpectedly consume the caller's CPU RNG.
    with torch.random.fork_rng(devices=[]):
        models = {a: ActorCritic(c).to(device=device, dtype=dtype) for a, c in model_configs.items()}
        learner = (MAPPOLearner(models, configs) if cfg['algorithm'] == 'mappo'
                   else PPOLearner(models['policy'], configs['policy']))
    for a, (model, optimizer, c, scalers) in _parts(learner).items():
        state = payload['agents'][a]
        _keys(state, ('model', 'optimizer', 'normalizers'), 'agent state')
        _validate_module_state(state['model'], model.state_dict())
        model.load_state_dict(state['model'], strict=True)
        _validate_optimizer(state['optimizer'], optimizer)
        rate = state['optimizer']['param_groups'][0]['lr']
        if c.adaptive_lr and not c.min_lr <= rate <= c.max_lr:
            raise ValueError("Adaptive learning rate is outside configured bounds")
        optimizer.load_state_dict(state['optimizer'])
        _keys(state['normalizers'], scalers, 'normalizers')
        for name, scaler in scalers.items():
            saved = state['normalizers'][name]
            if scaler is None:
                if saved is not None:
                    raise ValueError("Disabled normalizer has checkpoint state")
                continue
            _validate_module_state(saved, scaler.state_dict())
            if (saved['variance'] < 0).any() or saved['count'].item() <= 0:
                raise ValueError("Invalid running moments")
            scaler.load_state_dict(saved, strict=True)
    _keys(payload['progress'], ('updates', 'vector_steps', 'transitions'), 'progress')
    progress = Progress(**payload['progress'])
    _keys(payload['rng'], ('python', 'torch', 'cuda'), 'RNG')
    random.Random().setstate(payload['rng']['python'])
    torch.Generator().set_state(payload['rng']['torch'])
    if payload['rng']['cuda'] is not None:
        if not isinstance(payload['rng']['cuda'], list) or any(
                not isinstance(t, torch.Tensor) or t.dtype != torch.uint8 or t.ndim != 1 for t in payload['rng']['cuda']):
            raise ValueError("Invalid CUDA RNG state")
    if payload['interaction'] is not None and not isinstance(payload['interaction'], dict):
        raise ValueError("Invalid interaction state")
    return LoadedCheckpoint(learner, progress, payload['interaction'], payload['rng'], payload['device'])


def resume_training(checkpoint: LoadedCheckpoint, environment, *, exact=False) -> Trainer:
    """Restore counters/RNG; exact=True additionally requires restorable interaction."""
    trainer = Trainer(checkpoint.learner, environment)
    try:
        trainer.progress = Progress(**asdict(checkpoint.progress))
        if exact:
            if checkpoint.interaction is None:
                raise ValueError("Exact resume requires an environment interaction snapshot")
            if checkpoint.original_device != str(checkpoint.learner.device):
                raise ValueError("Exact continuation across different devices is not supported")
            trainer.collector.load_state_dict(checkpoint.interaction)
        cuda = checkpoint.rng['cuda']
        if checkpoint.learner.device.type == 'cuda':
            if cuda is None or len(cuda) != torch.cuda.device_count():
                raise ValueError("Matching CUDA RNG states are required to resume CUDA training")
            torch.cuda.set_rng_state_all(cuda)
        random.setstate(checkpoint.rng['python'])
        torch.set_rng_state(checkpoint.rng['torch'])
        return trainer
    except Exception:
        trainer.close()
        raise
