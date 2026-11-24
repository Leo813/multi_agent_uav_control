"""Explicit external-task registration. Import task implementations after AppLauncher."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    id: str
    entry_point: str
    config_entry_point: str


TASKS = {
    task: TaskSpec(
        f"UAV-Control-Quadcopter-{task}-v0",
        f"uav_control.tasks.quadcopter_{task}:QuadcopterEnv{task}",
        f"uav_control.tasks.quadcopter_{task}:QuadcopterEnvCfg{task}",
    ) for task in ("SARL", "MARL")
}


def register_tasks():
    """Call deliberately after simulator startup; registration uses lazy strings."""
    import gymnasium as gym

    for spec in TASKS.values():
        kwargs = {"env_cfg_entry_point": spec.config_entry_point}
        if spec.id in gym.registry:
            existing = gym.registry[spec.id]
            if existing.entry_point != spec.entry_point or existing.kwargs != kwargs:
                raise ValueError(f"Conflicting task registration: {spec.id}")
            continue
        gym.register(id=spec.id, entry_point=spec.entry_point,
                     disable_env_checker=True, kwargs=kwargs)
