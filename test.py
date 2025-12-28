import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import torch
import numpy as np
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback


from imitation.algorithms import bc

log_dir = "./cartpole_expert_demos/test2"

max_iterations = None
num_envs = 2 # 2 (trained)
# env = gym.make('CartPole-v1', render_mode='human')
env = make_vec_env('CartPole-v1', n_envs=num_envs, wrapper_class=FlattenObservation, env_kwargs={"render_mode": "human"})
observation_space_info = env.observation_space
action_space_info = env.action_space

# method-1 of recording expert demos (using standard DRL algos)
# using PPO (here)
def train_expert(env: VecEnv):

    print("Training Expert Agent")


    agent_cfg = {
        "seed": 42,
        "policy": "MlpPolicy",
        "batch_size": 128,
        "n_steps": 512,
        "gamma": 0.99,
        "gae_lambda": 0.9,
        "n_epochs": 20,
        "ent_coef": 0.0,
        "max_grad_norm": 0.5,
        "vf_coef": 0.5,
        "learning_rate": 3e-5,
        "clip_range": 0.4,
        "sde_sample_freq": -1,
        "use_sde": False,
        "device": "cuda:0",
        "policy_kwargs":{
            "log_std_init": -1,
            "ortho_init": False,
            "activation_fn": torch.nn.ReLU,
            "net_arch":{
                "pi": [256, 256],
                "vf": [256, 256]
            }
        }
    }

    if max_iterations is not None:
        n_timesteps = max_iterations*agent_cfg['n_steps']*num_envs

    n_timesteps = 1e6
    # env = VecNormalize(
    #     env,
    #     training=True,
    #     norm_obs=True,
    #     norm_reward=False,
    #     clip_obs=100.0,
    #     gamma=agent_cfg['gamma'],
    #     clip_reward=500.0,
    # )
    policy_kwargs = agent_cfg.pop("policy_kwargs")
    expert_agent = PPO(
        policy=agent_cfg.pop('policy'),
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
        **agent_cfg
        
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=1000, save_path=log_dir, name_prefix="cartpole_expert_model_no_vecnorm", verbose=2
    )
    callbacks = [checkpoint_callback]


    expert_agent.learn(total_timesteps=n_timesteps, callback=callbacks, progress_bar=True)

    return expert_agent


# expert_agent = train_expert(env=env)


def inference(algo, model_path):
    env = make_vec_env(
        'CartPole-v1',
        n_envs=1,
        wrapper_class=FlattenObservation,
        env_kwargs={"render_mode": "human"},
    )

    model = PPO.load(model_path)
    obs = env.reset()
    done = np.array([False, False])

    total_reward = 0
    steps = 0

    print("\n======== TESTING SINGLE EPISODE ========\n")

    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        total_reward += reward[0]
        steps += 1

        env.render()
        time.sleep(0.01)

    print("\n======== EPISODE FINISHED ========\n")
    print(f"Total steps survived: {steps}")
    print(f"Total reward: {total_reward}")
    print("Max episode length is 500. If steps >= 500 → policy is excellent.")


def evaluate_expert():
    pass


inference(algo=PPO, model_path=f"{log_dir}/cartpole_expert_model_no_vecnorm_132000_steps.zip")