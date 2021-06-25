import gym
from environment_interface import IEnvironment
import numpy as np
from PIL import Image
from PIL.Image import ANTIALIAS

class Environment(IEnvironment):
    def __init__(self, frameskip, render):
        self._env = gym.make('MsPacman-v0')
        self._env.frameskip = frameskip
        self._env.env.frameskip = frameskip
        self._obs_size = (85, 80, 1) 
        self._render = render
        self._lives = 3
        self._score = 0
        print("actions [0-8]:", self._env.env.get_action_meanings())

    def close(self):
        return self._env.close()

    def process_image(self, observation):
        # transform (210, 160, 3)  RGB pic into (85,80, 1) grayscaled pic
        # the unusefull bottom part of the pic is removed
        img = Image.fromarray(observation)
        img = img.convert('L')
        img = img.crop((0, 1, 160, 172))
        img = img.resize((80, 85), resample=ANTIALIAS)
        img = np.array(img) / 255
        img = np.expand_dims(img, axis = 2)
        return np.array(img)

    def render(self):
        if self._render:
            self._env.render()

    def reset(self):
        obs = self._env.reset()
        processed_obs = self.process_image(obs)
        self._lives = 3
        self._score = 0
        return processed_obs

    def step(self, action):
        next_obs, reward, done, info = self._env.step(action)
        self._score = self._score + reward
        lives = info["ale.lives"]
        if lives < self._lives:
            reward = reward - 500
        self._lives = lives
        processed_next_obs = self.process_image(next_obs)
        return processed_next_obs, reward, done, self._score

    @property
    def num_actions(self):
        # ['NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT']
        return self._env.action_space.n

    @property
    def obs_size(self):
        return self._obs_size
