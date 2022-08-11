from abc import ABC, abstractmethod


class AbstractAgent(ABC):

    @abstractmethod
    def __init__(self, config_file):
        pass

    @abstractmethod
    def build_env(self, env_file, attacks_file=None):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def save_results(self):
        pass

    @abstractmethod
    def save_model(self):
        pass