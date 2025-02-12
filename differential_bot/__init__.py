from .diff_bot import *
import gymnasium as gym
from . import agents, diff_bot_rlenv_cfg

gym.register(
    id="Isaac-diffbot-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": diff_bot_rlenv_cfg.DiffBotRLEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
) 

