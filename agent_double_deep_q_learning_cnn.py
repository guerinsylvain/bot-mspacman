from interface_agent import IAgent
from network_cnn import NetworkCNN
import numpy as np
from replay_memory import ReplayMemory

class AgentDoubleDeepQLearningCNN(IAgent):
    def __init__(self, obs_size, num_actions, epsilon=1, epsilon_decay=0.996, epsilon_min=1, sample_size = 200, num_epochs = 1, gamma = 0.95):
        self._epsilon = epsilon  # exploration rate
        self._epsilon_decay = epsilon_decay
        self._epsilon_min = epsilon_min
        self._gamma = gamma #discount rate
        self._obs_size = obs_size
        self._learn_count = 0
        self._num_actions = num_actions
        self._num_epochs = num_epochs
        self._replay_memory = ReplayMemory(capacity=50000)
        self._policy_network = NetworkCNN(obs_size, num_actions)
        self._target_network = NetworkCNN(obs_size, num_actions)
        self._target_network.weights = self._policy_network.weights
        self._target_network_update_rate = 5
        self._sample_size = sample_size

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
        # An important element of DQN is a target network, a technique introduced to stabilize learning. 
        # A target network is a copy of the action-value function (or Q-function) that is held constant 
        # to serveas a stable target for learning for some fixed number of timesteps.

        # The difference between Q-learning and DQN is that you have replaced an exact value function with a function approximator.
        # With Q-learning you are updating exactly one state/action value at each timestep, whereas with DQN you are updating many, which you understand. 
        # The problem this causes is that you can affect the action values for the very next state you will be in instead of guaranteeing them to be stable as they are in Q-learning.
        # This happens basically all the time with DQN when using a standard deep network (bunch of layers of the same size fully connected). 
        # The effect you typically see with this is referred to as "catastrophic forgetting" and it can be quite spectacular. 
        # If you are doing something like moon lander with this sort of network (the simple one, not the pixel one) and track the rolling average score over the last 100 games or so,
        # you will likely see a nice curve up in score, then all of a sudden it completely craps out starts making awful decisions again even as your alpha gets small.
        # This cycle will continue endlessly regardless of how long you let it run.
        # Using a stable target network as your error measure is one way of combating this effect. 
        # Conceptually it's like saying, "I have an idea of how to play this well, I'm going to try it out for a bit until I find something better" as opposed to saying 
        # "I'm going to retrain myself how to play this entire game after every move". 
        # By giving your network more time to consider many actions that have taken place recently instead of updating all the time, 
        # it hopefully finds a more robust model before you start using it to make actions.
        
        batch = self._replay_memory.sample(self._sample_size)
        batch_size = len(batch)        
        
        initial_obs = np.asarray([exp.initial_obs for exp in batch])
        q_values = np.asarray(self._policy_network.compute(initial_obs, batch_size = batch_size))

        next_obs = np.asarray([exp.next_obs for exp in batch])
        target_q_values = np.array(self._target_network.compute(next_obs, batch_size = batch_size))

        x_batch = np.zeros([np.shape(batch)[0], self._obs_size[0], self._obs_size[1], self._obs_size[2]]).astype(np.float32)
        y_batch = np.zeros([np.shape(batch)[0], self._num_actions]).astype(np.float32)

        for i in range(np.shape(batch)[0]):
            x_batch[i]  = batch[i].initial_obs
            for j in range(self._num_actions):
                if j == batch[i].action:
                    if batch[i].done:
                        y_batch[i,j] = batch[i].reward
                    else:
                        y_batch[i,j] = batch[i].reward + self._gamma * np.max(target_q_values[i])
                else:
                    y_batch[i,j] = q_values[i,j] 

        history = self._policy_network.train(train_samples=x_batch, 
                                            train_labels=y_batch, 
                                            num_epochs=self._num_epochs,
                                            batch_size = batch_size)
        accuracy = history.history['accuracy'][-1]

        self._learn_count +=1
        if (self._learn_count % self._target_network_update_rate) == 0:
            self._target_network.weights = self._policy_network.weights
            self._learn_count = 0

        if self._epsilon > self._epsilon_min:
            self._epsilon *= self._epsilon_decay                  
        
        return accuracy

    def loadModel(self, file_name):
        self._policy_network.load_model(file_name)
        self._target_network.load_model(file_name)
        return

    def saveModel(self, num_episodes):
        self._policy_network.save_model(f"policy_network_model_{num_episodes}.h5")
        return
