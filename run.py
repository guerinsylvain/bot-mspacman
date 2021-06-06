from agent_random import AgentRandom
from environment import Environment
import numpy as np

# parameters
render_env = False
num_episodes = 50

# inits
env = Environment()
history = []
agent = AgentRandom(env.num_actions)

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
