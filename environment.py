import gym
from interface_environment import IEnvironment
import numpy as np


class Environment(IEnvironment):
    def __init__(self):
        self.env = gym.make('MsPacman-v0')
        print("actions [0-8]:", self.env.env.get_action_meanings())

    def close(self):
        return self.env.close()

    def render(self):
        return self.env.render()

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    @property
    def num_actions(self):
        return self.env.action_space.n
