import abc


class IEnvironment(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def close(self):
        pass

    @abc.abstractmethod
    def render(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def step(self, action):
        pass

    @property
    def num_actions(self):
        pass
