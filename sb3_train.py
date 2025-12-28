import time
from typing import Any
from datetime import datetime
from collections import defaultdict
import torch
import os
import yaml
import pickle
import numpy as np

import gymnasium as gym
import omni_bot.envs 

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import FlattenObservation


# Helper functions for saving
def dump_yaml(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f)

def dump_pickle(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)

# recursively process the yaml config file for the RL agent
def process_sb3_cfg(cfg: dict, num_envs: int):
    def update_dict(hyperparams: dict[str, Any], depth: int) -> dict[str, Any]:
        for key, value in hyperparams.items():
            if isinstance(value, dict):
                update_dict(value, depth + 1)
            if isinstance(value, str):
                if value.startswith("nn."):
                    hyperparams[key] = getattr(torch.nn, value[3:])
            if depth == 0:
                if key in ["learning_rate", "clip_range", "clip_range_vf"]:
                    if isinstance(value, str):
                        _, initial_value = value.split("_")
                        initial_value = float(initial_value)
                        hyperparams[key] = lambda progress_remaining: progress_remaining * initial_value
                    elif isinstance(value, (float, int)):
                        # negative value: ignore (ex: for clipping)
                        if value < 0:
                            continue
                        hyperparams[key] = constant_fn(float(value))
                    else:
                        raise ValueError(f"Invalid value for {key}: {hyperparams[key]}")
        
        # Convert to a desired batch_size (n_steps=2048 by default for SB3 PPO)
        if "n_minibatches" in hyperparams:
            hyperparams["batch_size"] = (hyperparams.get("n_steps", 2048) * num_envs) // hyperparams["n_minibatches"]
            del hyperparams["n_minibatches"]
        
        return hyperparams
    return update_dict(cfg, 0)

# custom callback function to render frame at regular intervals
class RenderCallback(BaseCallback):
    def __init__(self, render_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.render_freq = render_freq
    
    def _on_step(self):
        env = self.model.get_env()
        if self.n_calls % self.render_freq == 0:
            while hasattr(env, "envs"):
                env = env.envs[0]  # unwrap DummyVecEnv
            if hasattr(env, "render"):
                env.render()  # now calls OmniBotEnv.render("human")
        return True


env_cfg = defaultdict() # environment configuration dictionary
def main():

    # adding some global arguments
    max_iterations = None  # maximum training iterations
    env_cfg["num_envs"] = num_envs = 2  # number of parallel environments
    
    # load the config file
    with open('config/sb3_ppo_cfg.yaml', 'r') as file:
        agent_cfg = yaml.safe_load(file)

    if max_iterations is not None:
        agent_cfg["n_timesteps"] = max_iterations * agent_cfg["n_steps"] * num_envs 
    env_cfg["seed"] = agent_cfg["seed"]
    env_cfg["device"] = agent_cfg["device"]

    # directory settings for logging and saving models
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join("logs/sb3", "OmniBot-v0")) # "OmniBot-v0" is task/env name
    log_dir = os.path.join(log_root_path, run_info)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # ------------------------------------------------
    # some args
    log_interval = agent_cfg.get("log_interval", 1000)
    num_envs = agent_cfg.get("num_envs", 2)
    max_iterations = agent_cfg.get("max_iterations", None)
    
    agent_cfg = process_sb3_cfg(agent_cfg, env_cfg["num_envs"])
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    # Create vectorized environment (dummy vectorized environment hence single core processing)
    env = make_vec_env('OmniBot-v0', n_envs=num_envs, wrapper_class=FlattenObservation, env_kwargs={"render_mode": "human"}) # wrapped to DummyVecEnv 
    # Normalize environment if needed
    norm_keys = {"normalize_input", "normalize_value", "clip_obs"}
    norm_args = {}
    for key in norm_keys:
        if key in agent_cfg:
            norm_args[key] = agent_cfg.pop(key)

    env = VecNormalize(
        venv=env,
        training=True,
        norm_obs=norm_args.get("normalize_input", True), # normalizing observation
        norm_reward=norm_args.get("normalize_value", False), # not normalizing reward
        clip_obs=norm_args.get("clip_obs", 100.0), # clipping observation to avoid large value
        gamma=agent_cfg.get("gamma", 0.99), # discount factor
        clip_reward=np.inf,  # do not clip reward
    )
    
    # Create PPO agent
    agent = PPO(policy=policy_arch, env=env, verbose=1, tensorboard_log=log_dir, **agent_cfg)

    # Callbacks (save every 1000 timesteps)
    checkpoint_callback = CheckpointCallback(
        save_freq=1000, save_path=log_dir, name_prefix="omnibot_model_v0", verbose=2
    )
    # logging and checkpoint callback every log_interval timesteps
    callbacks = [checkpoint_callback]

    # Train the agent
    agent.learn(total_timesteps=n_timesteps, callback=callbacks, progress_bar=True)

    # Save final model
    agent.save(os.path.join(log_dir, "model"))
    env.save(os.path.join(log_dir, "params", "vec_env_stats.pkl"))

    print(f"[INFO]: Model saved at {os.path.join(log_dir, 'model.zip')}")

    env.close()


if __name__ == "__main__":
    main()
