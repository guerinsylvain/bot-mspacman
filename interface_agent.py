import abc


class IAgent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def choose_action(self, observation):
        pass

    @abc.abstractmethod
    def gather_experience(self, prev_observation, action, reward, new_observation):
        pass

    @abc.abstractmethod
    def learn(self):
        pass
