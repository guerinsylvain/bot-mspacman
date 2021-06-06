import gym
from agent_random import AgentRandom
import numpy as np

# parameters
render_env = False
num_episodes = 50

# inits
env = gym.make('MsPacman-v0')
print("actions [0-8]:", env.env.get_action_meanings())
history = []
agent = AgentRandom(env.action_space.n)

# training
for i in range(num_episodes):
    done = False
    episodic_reward = 0
    obs = env.reset()
    while not done:
        if render_env:
            env.render()

        action = agent.choose_action(obs)
        next_obs, reward, done, _ = env.step(action)
        episodic_reward += reward
        obs = next_obs

    history.append(episodic_reward)
    print("Episode number:", i, 'Episode Reward:', episodic_reward)

print(np.mean(history))
env.close()
