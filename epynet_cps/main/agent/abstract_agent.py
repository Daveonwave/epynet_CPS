from abc import ABC, abstractmethod


class AbstractAgent(ABC):

    @abstractmethod
    def __init__(self, config_file):
        pass

    @abstractmethod
    def build_env(self,
                  env_file,
                  demand_patterns_file,
                  attacks_train_file=None,
                  attacks_test_file=None,
                  output_file_path=None):
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