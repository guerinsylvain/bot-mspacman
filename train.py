from agent_random import AgentRandom
from agent_double_deep_q_learning_cnn import AgentDoubleDeepQLearningCNN
from environment import Environment

# parameters
render_env = False
num_episodes = 15000
global_steps = 0
start_steps = 1000
steps_train = 4

# inits
env = Environment()
history = []
# agent = AgentRandom(env.num_actions)
agent = AgentDoubleDeepQLearningCNN(env.obs_size, env.num_actions)

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
        if done or (global_steps % steps_train == 0 and global_steps > start_steps):
            agent.learn()

        episodic_reward += reward
        global_steps += 1
        obs = next_obs
        print(f'Episode number: {episode:0>4d}, steps: {global_steps:0>4d}, exploration rate: {agent.exploration_rate:.2f}, reward: {episodic_reward}', end = '\r')

    history.append(episodic_reward)
    if (episode % 10) == 0:
        agent.saveModel(episode)
    print(f'Episode number:", {episode:0>4d}, steps: {global_steps:0>4d}, exploration rate: {agent.exploration_rate:.2f}, reward: {episodic_reward}')

env.close()
