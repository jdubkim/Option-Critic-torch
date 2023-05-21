import gym
import gym_minigrid
import numpy as np
import torch


def make_env(env_name):
    env = gym.make(env_name)
    if 'MiniGrid' in env_name:
        env = gym_minigrid.wrappers.RGBImgPartialObsWrapper(env)
    return env


def to_tensor(obs):
    obs = np.asarray(obs)
    obs = torch.from_numpy(obs).float()
    return obs
