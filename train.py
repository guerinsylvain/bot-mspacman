import gym

# parameters
num_episodes = 10

env = gym.make('MsPacman-v0')
print("actions [0-8]:", env.env.get_action_meanings())
history = []

for i in range(num_episodes):
    done = False
    episodic_reward = 0
    env.reset()
    while not done:
        env.render()
        action = env.action_space.sample()
        next_obs, reward, done, _  = env.step(action)  
        episodic_reward += reward
    
    history.append(episodic_reward)
    print("Episode number:", i, 'Episode Reward:', episodic_reward)
print(history)
env.close()
