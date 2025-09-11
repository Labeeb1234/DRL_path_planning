import omni_bot.envs
import gymnasium as gym
import numpy as np
import torch
import time


def main():
    env = gym.make("OmniBot-v0", render_mode="human")
    state, _  = env.reset()
    episodes = 10000
    for episode in range(episodes):
        print(f"episde: {episode}")
        state, _ = env.reset()
        done = False
        while not done:
            print(f"current state: {state}\n")
            action = np.array([1.0, 0.0, 0.0]) 
            new_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            state = new_state
            time.sleep(0.1) # extra delay of 0.1s to 60Hz physics stepping of an episode


if __name__ == "__main__":
    main()
