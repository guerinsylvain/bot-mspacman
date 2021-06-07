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

    @abc.abstractproperty
    @property
    def num_actions(self):
        pass

    @abc.abstractproperty
    @property
    def obs_size(self):
        pass
