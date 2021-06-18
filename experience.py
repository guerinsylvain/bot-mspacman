class Experience:
    def __init__(self, intial_obs, action, reward, next_obs, done):
        self.initial_obs = intial_obs
        self.action = action
        self.reward = reward
        self.next_obs = next_obs
        self.done = done
