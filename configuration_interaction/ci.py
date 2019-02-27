import abc


class ConfigurationInteraction(metaclass=abc.ABCMeta):
    def __init__(self, system, verbose=False):
        self.system = system
        self.verbose = verbose

    @abc.abstractmethod
    def setup_ci_space(self):
        pass
