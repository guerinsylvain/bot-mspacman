from IAgent import IAgent
import numpy as np


class AgentRandom(IAgent):
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def choose_action(self, observation):
        return np.random.choice(range(self.num_actions))

    def gather_experience(self, prev_observation, action, reward, new_observation):
        pass

    def learn(self):
        pass
