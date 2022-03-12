import pandas as pd
import yaml
import random
from mushroom_rl.core import Core
from mushroom_rl.algorithms.value import DQN
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import LinearParameter, Parameter
from mushroom_rl.utils.replay_memory import ReplayMemory
from mushroom_rl.utils.callbacks import CollectDataset, CollectMaxQ
from torch.optim.adam import Adam
from torch.nn import functional as F

from scripts.dqn import nn
from scripts.dqn.env import WaterNetworkEnvironment
from scripts.dqn.logger import InfoLogger

results_path = '../../results/DQN'
config_file = 'anytown.yaml'

logger = InfoLogger(config_file[:-5], results_path)


class DQNAgent:
    """

    """
    def __init__(self):
        with open(config_file, 'r') as fin:
            self.hparams = yaml.safe_load(fin)

        self.env = WaterNetworkEnvironment(
            town=self.hparams['env']['town'],
            state_vars=self.hparams['env']['state_vars'],
            action_vars=self.hparams['env']['action_vars'],
            duration=self.hparams['env']['duration'],
            hyd_step=self.hparams['env']['hyd_step'],
            pattern_step=self.hparams['env']['pattern_step'],
            pattern_files=self.hparams['env']['patterns'],
            seed=self.hparams['env']['seed'],
            update_every=self.hparams['env']['update_every'],
            bounds=self.hparams['env']['bounds'],
            logger=logger,
            show_plot=None
        )

        # Creating the epsilon greedy policy
        self.epsilon_train = LinearParameter(value=1., threshold_value=.01, n=300000)
        self.epsilon_test = Parameter(value=0)
        self.epsilon_random = Parameter(value=1)
        self.pi = EpsGreedy(epsilon=self.epsilon_random)

        if not self.hparams['agent']['load']:
            # Create the optimizer dictionary
            self.optimizer = dict()
            self.optimizer['class'] = Adam
            self.optimizer['params'] = self.hparams['optimizer']

            # Set parameters of neural network taken by the torch approximator
            nn_params = dict(hidden_size=self.hparams['nn']['hidden_size'])

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
                batch_size=0,
                use_cuda=False,
                **nn_params
            )

            # Build replay buffer
            self.replay_buffer = ReplayMemory(initial_size=self.hparams['agent']['initial_replay_memory'],
                                              max_size=self.hparams['agent']['max_replay_size'])

            self.agent = DQN(mdp_info=self.env.info,
                             policy=self.pi,
                             approximator=approximator,
                             approximator_params=approximator_params,
                             batch_size=self.hparams['agent']['batch_size'],
                             target_update_frequency=self.hparams['agent']['target_update_frequency'],
                             replay_memory=self.replay_buffer,
                             initial_replay_size=self.hparams['agent']['initial_replay_memory'],
                             max_replay_size=self.hparams['agent']['max_replay_size']
                             )

        else:
            self.agent = DQN.load(self.hparams['agent']['load_as'])

        # Callbacks
        self.dataset = CollectDataset()
        self.core = Core(self.agent, self.env, callbacks_fit=[self.dataset])

        self.scores = []

    def fill_replay_buffer(self):
        """

        :return:
        """
        self.pi.set_epsilon(self.epsilon_random)
        #self.core.learn(n_episodes=1, n_steps_per_fit=self.hparams['agent']['initial_replay_memory'])

        if self.replay_buffer.size < self.hparams['agent']['initial_replay_memory']:
            # Fill replay memory with random data
            self.core.learn(n_steps=self.hparams['agent']['initial_replay_memory'] - self.replay_buffer.size,
                            n_steps_per_fit=self.hparams['agent']['initial_replay_memory'], render=False)

    def learn(self):
        """

        :return:
        """
        self.env.on_eval = False
        self.pi.set_epsilon(dqn.epsilon_train)
        logger.training_phase()
        self.core.learn(n_episodes=self.hparams['learning']['train_episodes'],
                        n_steps_per_fit=self.hparams['learning']['train_frequency'],
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

        dataset = self.core.evaluate(n_episodes=1, render=True)
        self.scores.append(logger.get_stats(dataset))
        logger.end_phase()

        df_dataset = None
        qs_list = None

        if get_data:
            df_dataset = pd.DataFrame(dataset, columns=['current_state', 'action', 'reward', 'next_state',
                                                        'absorbing_state', 'last_step'])
        if collect_qs:
            qs_list = self.agent.approximator.model.network.retrieve_qs()
            self.agent.approximator.model.network.collect_qs_enabled(False)

        return df_dataset, qs_list


if __name__ == '__main__':

    dqn = DQNAgent()

    if not dqn.hparams['agent']['load']:
        # Not loaded agent, we need to train
        n_epochs = dqn.hparams['learning']['epochs']

        logger.experiment_summary("\n\tObservation space: TIME - DAY - T41_level - T42_level - J20_pressure\n"
                                  "\tEpisode: 1 week\n"
                                  "\tHydraulic_step: 5 min\n"
                                  "\tUpdate every: step\n"
                                  "\tEpochs: " + str(n_epochs) + "\n"
                                  "\tTrain episodes per epochs: " + str(dqn.hparams['learning']['train_episodes']))

        dqn.fill_replay_buffer()
        # dqn.evaluate(get_data=False, collect_qs=False)

        results = {'train': [], 'eval': []}

        for epoch in range(1, n_epochs + 1):
            print(dqn.epsilon_train.get_value())
            logger.print_epoch(epoch)
            dqn.learn()
            #_, qs = dqn.evaluate(get_data=False, collect_qs=False)
            #results['train'].append(qs)

        #dqn.agent.save('saved_models/overflow_double_train_set.msh', full_save=True)

    else:
        results = {'eval': []}

    seeds = [0, 1, 2, 3]
    for seed in seeds:
        dqn.env.seed = seed
        dataset, qs = dqn.evaluate(get_data=True, collect_qs=True)
        res = {'dsr': dqn.env.dsr, 'updates': dqn.env.total_updates, 'seed': seed, 'dataset': dataset, 'q_values': qs,
               'attacks': dqn.env.attacks, 'T41_ground': dqn.env.t41_ground, 'T42_ground': dqn.env.t42_ground}
        results['eval'].append(res)

    import pickle

    with open('../../results/DQN/anytown/uninformed_agent_attacks', 'wb') as fp:
        pickle.dump(results, fp)

    #dqn.env.wn.create_df_reports()
    #dqn.env.wn.df_nodes_report.to_csv("../df_report.csv")
