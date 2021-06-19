from agent_random import AgentRandom
from agent_double_deep_q_learning_screenshot_cnn import AgentDoubleDeepQLearningCNN
from environment import Environment
import numpy as np

# parameters
env_render = False
env_frameskip = 20
agent_epsilon=1.0
agent_epsilon_decay=0.998
agent_epsilon_min=0.01
agent_sample_size = 200
agent_num_epochs = 1
agent_discount_rate = 0.95
num_episodes = 100

# inits
env = Environment(env_frameskip, env_render)
history = []
agent = AgentRandom(env.num_actions)
agent = AgentDoubleDeepQLearningCNN(env.obs_size, env.num_actions, epsilon=agent_epsilon, epsilon_decay=agent_epsilon_decay, epsilon_min=agent_epsilon_min, sample_size = agent_sample_size, num_epochs = agent_num_epochs, discount_rate = agent_discount_rate)
agent.loadModel('policy_network_model_230.h5')

# training
for i in range(num_episodes):
    done = False
    episodic_reward = 0
    obs = env.reset()
    while not done:
        env.render()
        action = agent.choose_action(obs, explore=False)
        next_obs, reward, done = env.step(action)
        episodic_reward += reward
        obs = next_obs
        print(f'Episode number: {i:0>4d}, reward: {episodic_reward}', end = '\r')

    history.append(episodic_reward)
    print("Episode number:", i, 'Episode Reward:', episodic_reward)

print(np.mean(history))
env.close()
