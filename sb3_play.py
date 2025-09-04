
import omni_bot.envs
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import FlattenObservation


model = PPO.load("logs/sb3/OmniBot-v0/2025-09-01_00-08-19/model_104000_steps.zip")
env = make_vec_env('OmniBot-v0', n_envs=1, wrapper_class=FlattenObservation,  env_kwargs={"render_mode": "human"})
obs = env.reset()
done = False
while not done:
    for _ in range(10000):
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, terminated, info = env.step(action)
        done = terminated 
        env.render()




