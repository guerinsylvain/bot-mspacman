from interface_agent import IAgent
from network_cnn import NetworkCNN
import numpy as np
from replay_memory import ReplayMemory

class AgentDeepQLearningCNN(IAgent):
    def __init__(self, obs_size, num_actions, epsilon=1, epsilon_decay=0.996, epsilon_min=1):
        self._epsilon = epsilon  # exploration rate
        self._epsilon_decay = epsilon_decay
        self._epsilon_min = epsilon_min
        self._num_actions = num_actions
        self._replay_memory = ReplayMemory(capacity=50000)
        self._policy_network = NetworkCNN(obs_size, num_actions)
        self._target_network = NetworkCNN(obs_size, num_actions)

    def choose_action(self, observation, explore=True):
        if explore and np.random.rand() <= self._epsilon:
            return np.random.choice(range(self._num_actions))
        else:
            q_compute = self.policy_network.compute([observation],
                                                    batch_size=1)
            return np.argmax(q_compute[0])

    def gather_experience(self, prev_observation, action, reward, new_observation, done):
        self._replay_memory.write(prev_observation, action, reward, new_observation, done)
        return

    def learn(self):
        pass

    def loadModel(self, file_name):
        self._policy_network.load_model(file_name)
        self._target_network.load_model(file_name)
        return

    def saveModel(self, num_episodes):
        self._policy_network.save_model(f"policy_network_model_{num_episodes}.h5")
        return
