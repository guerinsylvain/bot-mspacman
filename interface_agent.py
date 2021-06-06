import abc

class IAgent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def choose_action(self, observation, explore=True):
        pass

    @abc.abstractmethod
    def gather_experience(self, intial_obs, action, reward, next_obs, done):
        pass

    @abc.abstractmethod
    def learn(self):
        pass

    @abc.abstractmethod
    def loadModel(self, file_name):
        pass

    @abc.abstractmethod
    def saveModel(self, num_episodes):
        pass
