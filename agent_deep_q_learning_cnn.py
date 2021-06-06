from interface_agent import IAgent
import numpy as np
from replay_memory import ReplayMemory


class AgentDeepQLearningCNN(IAgent):
    def __init__(self, num_actions):
        self._num_actions = num_actions
        self._replay_memory = ReplayMemory(capacity=50000)

    def choose_action(self, observation):
        return np.random.choice(range(self._num_actions))

    def gather_experience(self, prev_observation, action, reward, new_observation, done):
        self._replay_memory.write(
            prev_observation, action, reward, new_observation, done)
        return

    def learn(self):
        pass
