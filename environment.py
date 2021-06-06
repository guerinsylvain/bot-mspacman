import gym
from interface_environment import IEnvironment
import numpy as np
from PIL import Image
from PIL.Image import ANTIALIAS


class Environment(IEnvironment):
    def __init__(self):
        self._env = gym.make('MsPacman-v0')
        print("actions [0-8]:", self._env.env.get_action_meanings())

    def close(self):
        return self._env.close()

    def process_image(self, observation):
        # transform (210, 160, 3) RGB pic into (85,80) grayscaled pic
        # the unusefull bottom part of the pic is removed
        img = Image.fromarray(observation)
        img = img.convert('L')
        img = img.crop((0, 1, 160, 172))
        img = img.resize((85, 80), resample=ANTIALIAS)
        return np.array(img)

    def render(self):
        return self._env.render()

    def reset(self):
        obs = self._env.reset()
        processed_obs = self.process_image(obs)
        return processed_obs

    def step(self, action):
        next_obs, reward, done, _ = self._env.step(action)
        processed_next_obs = self.process_image(next_obs)
        return processed_next_obs, reward, done

    @property
    def num_actions(self):
        return self._env.action_space.n
