from agent_random import AgentRandom
from agent_deep_q_learning_cnn import AgentDeepQLearningCNN
from environment import Environment

# parameters
render_env = True
num_episodes = 10

# inits
env = Environment()
history = []
# agent = AgentRandom(env.num_actions)
agent = AgentDeepQLearningCNN(env.obs_size, env.num_actions)

# training
for episode in range(num_episodes):
    done = False
    episodic_reward = 0
    obs = env.reset()
    while not done:
        if render_env:
            env.render()

        action = agent.choose_action(obs)
        next_obs, reward, done = env.step(action)
        agent.gather_experience(obs, action, reward, next_obs, done)
        agent.learn()

        episodic_reward += reward
        obs = next_obs
        print(f'Episode number: {episode:0>4d}, reward: {episodic_reward}', end = '\r')

    history.append(episodic_reward)
    if (episode % 1) == 0:
        agent.saveModel(episode)
    print(f'Episode number:", {episode:0>4d}, reward: {episodic_reward}')

env.close()
