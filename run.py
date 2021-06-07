from agent_random import AgentRandom
from agent_double_deep_q_learning_cnn import AgentDoubleDeepQLearningCNN
from environment import Environment
import numpy as np

# parameters
render_env = False
num_episodes = 50

# inits
env = Environment()
history = []
# agent = AgentRandom(env.num_actions)
agent = AgentDoubleDeepQLearningCNN(env.obs_size, env.num_actions)
agent.loadModel('policy_network_model_1330.h5')

# training
for i in range(num_episodes):
    done = False
    episodic_reward = 0
    obs = env.reset()
    while not done:
        if render_env:
            env.render()

        action = agent.choose_action(obs)
        next_obs, reward, done = env.step(action)
        episodic_reward += reward
        obs = next_obs

    history.append(episodic_reward)
    print("Episode number:", i, 'Episode Reward:', episodic_reward)

print(np.mean(history))
env.close()
