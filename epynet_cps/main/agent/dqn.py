import yaml
import pickle
import pandas as pd
from pathlib import Path
from datetime import datetime
from mushroom_rl.core import Core
from mushroom_rl.algorithms.value import DQN
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import LinearParameter, Parameter
from mushroom_rl.utils.replay_memory import ReplayMemory
from mushroom_rl.utils.callbacks import CollectDataset
from torch.optim.adam import Adam
from torch.nn import functional as F
from .abstract_agent import AbstractAgent

from . import nn
from .rl_env import WaterNetworkEnvironment
from .logger import InfoLogger

file_path = Path("agent/")
results_path = 'agent/logs/'
logger = InfoLogger(str(datetime.now().strftime("%d-%b-%Y_%H:%M:%S")), results_path)


class DeepQNetwork(AbstractAgent):
    """
    # TODO: change env without recreate the agent is not possible with DQN, maybe saving the tuned weights of NN
    """
    def __init__(self, agent_config_file):
        with open(file_path / agent_config_file, 'r') as fin:
            self.model_configs = yaml.safe_load(fin)
            self.output_file_path = None

        # Creating the epsilon greedy policy
        self.epsilon_train = LinearParameter(value=1., threshold_value=.01, n=300000)
        self.epsilon_test = Parameter(value=0)
        self.epsilon_random = Parameter(value=1)
        self.pi = EpsGreedy(epsilon=self.epsilon_random)

        # Callbacks
        self.dataset = CollectDataset()

        self.agent = None
        self.env = None
        self.core = None
        self.optimizer = None
        self.replay_buffer = None

        self.scores = []
        self.results = None
        self.intermediate_results = []

    def build_env(self, env_file, demand_patterns_file, attacks_train_file=None, attacks_test_file=None, output_file_path=None):
        """
        Build the current experiment model passing the configurations of both environment and attacks.
        :param env_file:
        :param demand_patterns_file:
        :param attacks_train_file:
        :param attacks_test_file:
        :param output_file_path:
        """
        self.env = WaterNetworkEnvironment(env_config_file=env_file,
                                           demand_patterns_file=demand_patterns_file,
                                           attacks_train_file=attacks_train_file,
                                           attacks_test_file=attacks_test_file,
                                           logger=logger
                                           )

        self.output_file_path = output_file_path

        if not self.model_configs['agent']['load']:
            # Create the optimizer dictionary
            self.optimizer = dict()
            self.optimizer['class'] = Adam
            self.optimizer['params'] = self.model_configs['optimizer']

            # Set parameters of neural network taken by the torch approximator
            nn_params = dict(hidden_size=self.model_configs['nn']['hidden_size'])

            # Create the approximator from the neural network we have implemented
            approximator = TorchApproximator

            # Set parameters of approximator
            approximator_params = dict(
                network=nn.NN10Layers,
                input_shape=self.env.info.observation_space.shape,
                output_shape=(self.env.info.action_space.n,),
                n_actions=self.env.info.action_space.n,
                optimizer=self.optimizer,
                loss=F.smooth_l1_loss,
                batch_size=16,
                use_cuda=False,
                **nn_params
            )

            # Build replay buffer
            self.replay_buffer = ReplayMemory(initial_size=self.model_configs['agent']['initial_replay_memory'],
                                              max_size=self.model_configs['agent']['max_replay_size'])

            self.agent = DQN(mdp_info=self.env.info,
                             policy=self.pi,
                             approximator=approximator,
                             approximator_params=approximator_params,
                             batch_size=self.model_configs['agent']['batch_size'],
                             target_update_frequency=self.model_configs['agent']['target_update_frequency'],
                             replay_memory=self.replay_buffer,
                             initial_replay_size=self.model_configs['agent']['initial_replay_memory'],
                             max_replay_size=self.model_configs['agent']['max_replay_size']
                             )
            self.core = Core(self.agent, self.env, callbacks_fit=[self.dataset])
        else:
            self.agent = DQN.load(file_path / self.model_configs['agent']['load_as'])
            self.core = Core(self.agent, self.env, callbacks_fit=[self.dataset])
            self.pi = self.core.agent.policy

    def fill_replay_buffer(self):
        """

        :return:
        """
        self.pi.set_epsilon(self.epsilon_random)
        # self.core.learn(n_episodes=1, n_steps_per_fit=self.hparams['agent']['initial_replay_memory'])

        if self.replay_buffer.size < self.model_configs['agent']['initial_replay_memory']:
            # Fill replay memory with random data
            self.core.learn(n_steps=self.model_configs['agent']['initial_replay_memory'] - self.replay_buffer.size,
                            n_steps_per_fit=self.model_configs['agent']['initial_replay_memory'], render=False)

    def learn(self):
        """

        :return:
        """
        self.env.on_eval = False
        logger.training_phase()

        if self.replay_buffer is None:
            # Build replay buffer
            self.replay_buffer = ReplayMemory(initial_size=self.model_configs['agent']['initial_replay_memory'],
                                              max_size=self.model_configs['agent']['max_replay_size'])
            self.pi.set_epsilon(self.epsilon_random)
            self.core.learn(n_steps=self.model_configs['agent']['initial_replay_memory'] - self.replay_buffer.size,
                            n_steps_per_fit=self.model_configs['agent']['initial_replay_memory'], render=False)

        self.pi.set_epsilon(self.epsilon_train)
        self.core.learn(n_episodes=self.model_configs['learning']['train_episodes'],
                        n_steps_per_fit=self.model_configs['learning']['train_frequency'],
                        render=False)
        logger.end_phase()

    def evaluate(self, get_data=False, collect_qs=False):
        """

        :param get_data:
        :param collect_qs:
        :return:
        """
        self.env.on_eval = True
        self.pi.set_epsilon(self.epsilon_test)
        logger.evaluation_phase()

        self.agent.approximator.model.network.collect_qs_enabled(collect_qs)

        dataset = self.core.evaluate(n_episodes=1, render=False)
        self.scores.append(logger.get_stats(dataset))
        logger.end_phase()

        df_dataset = None
        qs_list = None

        if get_data:
            df_dataset = pd.DataFrame(dataset, columns=['current_state', 'action', 'reward', 'next_state',
                                                        'absorbing_state', 'last_step'])
            df_dataset['timestamp'] = self.env.wn.times
            df_dataset = df_dataset[['timestamp', 'current_state', 'action', 'reward', 'next_state', 'absorbing_state',
                                     'last_step']]

        if collect_qs:
            qs_list = self.agent.approximator.model.network.retrieve_qs()
            self.agent.approximator.model.network.collect_qs_enabled(False)

        return df_dataset, qs_list

    def save_results(self, do_save=False):
        """
        Save results: dataset and q_values.
        """
        if do_save:
            # Check if "results" directory exists
            Path(self.output_file_path).mkdir(parents=True, exist_ok=True)
            with open(self.output_file_path / "dataset", 'wb') as fp:
                pickle.dump(self.results, fp)
            print(">>> Results saved!")
        else:
            print(self.results)

    def save_model(self):
        """
        Save the model created
        """
        if self.model_configs['model']['save_model']:
            file_name = self.model_configs['model']['save_as'] + ".msh"
            Path(self.output_file_path).mkdir(parents=True, exist_ok=True)
            self.agent.save(path=self.output_file_path / file_name, full_save=True)
            print(">>> Model saved: ", file_name)

    def run(self):
        """
        Run schedule of the current training and testing session.
        # TODO: implement with schedule and option chosen from config file
        """
        if not self.model_configs['agent']['load']:
            self.fill_replay_buffer()

        self.results = {'train': [], 'eval': []}
        n_epochs = self.model_configs['learning']['epochs']

        for epoch in range(0, n_epochs):
            logger.print_epoch(epoch + 1)
            self.learn()

            # Evaluation at the end of the epoch (skip for last epoch)
            if epoch < n_epochs - 1:
                for seed in self.env.test_seeds:
                    self.env.curr_seed = seed
                    self.evaluate(get_data=False, collect_qs=False)
                    self.intermediate_results.append({'seed': seed,
                                                      'dsr': self.env.dsr,
                                                      'updates': self.env.total_updates,
                                                      'epoch': epoch})

        for seed in self.env.test_seeds:
            self.env.curr_seed = seed
            dataset, qs = self.evaluate(get_data=self.model_configs['results']['dataset'],
                                        collect_qs=self.model_configs['results']['q_values'])
            # TODO: add attacks to results
            res = {'dsr': self.env.dsr,
                   'updates': self.env.total_updates,
                   'seed': seed,
                   'dataset': dataset,
                   'q_values': qs,
                   'overflow': self.env.total_overflow,
                   'flow': self.env.total_flow}

            self.results['eval'].append(res)

            # Create and saves nodes and links reports
            if self.model_configs['results']['save_results']:
                for sensor in self.env.sensor_plcs:
                    sensor.save_ground_data(self.output_file_path, seed)
                    sensor.save_altered_data(self.output_file_path, seed)

        self.results['intermediate_results'] = self.intermediate_results
        self.results['training_patterns'] = [pattern['type'][0] for pattern in self.env.training_conditions]

        self.save_results(do_save=self.model_configs['results']['save_results'])
        self.save_model()


