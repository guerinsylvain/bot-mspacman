from agent_random import AgentRandom
from agent_double_deep_q_learning_screenshot_cnn import AgentDoubleDeepQLearningCNN
from environment import Environment

# parameters
env_render = False
env_frameskip = 20
agent_epsilon=1.0
agent_epsilon_decay=0.998
agent_epsilon_min=0.01
agent_sample_size = 200
agent_num_epochs = 1
agent_discount_rate = 0.95
num_episodes = 15000
global_steps = 0
start_steps = 1000
steps_train = 5

# inits
env = Environment(env_frameskip, env_render)
history = []
# agent = AgentRandom(env.num_actions)
agent = AgentDoubleDeepQLearningCNN(env.obs_size, env.num_actions, epsilon=agent_epsilon, epsilon_decay=agent_epsilon_decay, epsilon_min=agent_epsilon_min, sample_size = agent_sample_size, num_epochs = agent_num_epochs, discount_rate = agent_discount_rate)

# training
for episode in range(num_episodes):
    done = False
    episodic_reward = 0
    obs = env.reset()
    while not done:
        env.render()
        action = agent.choose_action(obs)
        next_obs, reward, done, score = env.step(action)
        agent.gather_experience(obs, action, reward, next_obs, done)
        if done or (global_steps % steps_train == 0 and global_steps > start_steps):
            agent.learn()

        episodic_reward += reward
        global_steps += 1
        obs = next_obs
        print(f'Episode number: {episode:0>4d}, steps: {global_steps:0>4d}, exploration rate: {agent.exploration_rate:.2f}, reward: {episodic_reward:5.0f}, score: {score:5.0f}', end = '\r')

    history.append(episodic_reward)
    if (episode % 10) == 0:
        agent.saveModel(episode)
    print(f'Episode number:", {episode:0>4d}, steps: {global_steps:0>4d}, exploration rate: {agent.exploration_rate:.2f}, reward: {episodic_reward:5.0f}, score: {score:5.0f}')

env.close()
