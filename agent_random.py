from interface_agent import IAgent
import numpy as np

class AgentRandom(IAgent):
    def __init__(self, num_actions):
        self._num_actions = num_actions

    def choose_action(self, observation, explore=True):
        return np.random.choice(range(self._num_actions))

    def gather_experience(self, intial_obs, action, reward, next_obs, done):
        pass

    def learn(self):
        pass

    def loadModel(self, file_name):
        pass

    def saveModel(self, num_episodes):
        pass
