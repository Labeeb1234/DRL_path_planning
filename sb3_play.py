import pickle as pkl
import omni_bot.envs
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import FlattenObservation

model = PPO.load("./logs/sb3/OmniBot-v0/2025-09-14_21-28-37/model_20000_steps.zip")
env = make_vec_env('OmniBot-v0', n_envs=1, wrapper_class=FlattenObservation,  env_kwargs={"render_mode": "human"})

# load the normalized vecenv stats along with model as training was done in vecnormalized env

with open('./logs/sb3/OmniBot-v0/2025-09-14_21-28-37/params/env.pkl', 'rb') as fp:
    env_data = pkl.load(fp)

print(env_data)




