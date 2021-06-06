from experience import Experience
import random


class ReplayMemory:
    def __init__(self, capacity):
        self.__memory = []
        self.__push_count = 0
        self.__capacity = capacity
        self.__last_exp = None

    def write(self, intial_obs, action, reward, next_obs, done):
        experience = Experience(intial_obs, action, reward, next_obs, done)
        self.__last_exp = experience
        if len(self.__memory) < self.__capacity:
            self.__memory.append(experience)
        else:
            self.__memory[self.__push_count % self.__capacity] = experience
        self.__push_count += 1

    def sample(self, batch_size):
        if len(self.__memory) > batch_size:
            sample = random.sample(self.__memory, batch_size-1)
            # always add the last experience to the sample
            sample.append(self.__last_exp)
            return sample
        else:
            return random.sample(self.__memory, len(self.__memory))
