import abc


class ConfigurationInteraction(metaclass=abc.ABCMeta):
    def __init__(self, system, verbose=False, np=None):
        self.verbose = verbose

        if np is None:
            import numpy as np

        self.np = np

        self.system = system

        self.n = self.system.n
        self.l = self.system.l
        self.m = self.system.m
        self.o = self.system.o
        self.v = self.system.v

    @abc.abstractmethod
    def setup_ci_space(self):
        pass
