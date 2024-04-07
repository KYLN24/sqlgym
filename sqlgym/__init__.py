from gymnasium.envs.registration import register

from .sqlgym import SqlGymEnv

__version__ = "0.1.0"


register(
    id="SQLGymEnv-v0",
    entry_point=SqlGymEnv,
    max_episode_steps=300,
)
