import yaml
from pathlib import Path
from physical_process import WaterDistributionNetwork
from agent.dqn import DeepQNetwork

config_folder = Path("../experiments")

class Runner:

    def __init__(self, experiment_file: Path, agent_disabled: bool):
        """

        """
        with open(experiment_file, 'r') as fin:
            self.exp_configs = yaml.safe_load(fin)
        self.agent_disabled = agent_disabled
        self.agent = None

    def run(self):
        """
        Launcher of the simulation with split the cases with and without a RL agent
        """
        if self.agent_disabled:
            for exp in self.exp_configs['experiments']:
                self.run_no_agent_experiment(exp)
        else:
            for exp in self.exp_configs['experiments']:
                self.build_rl_agent()
                self.run_agent_experiment(exp)

    def run_no_agent_experiment(self, config_files):
        """
        Run a single no agent experiment.
        """
        env_config_path = Path(config_folder / self.exp_configs['experiment_folder'] / config_files['env'])

        with open(env_config_path, 'r') as fin:
            env_settings = yaml.safe_load(fin)

        wn = WaterDistributionNetwork(env_settings['town'] + '.inp')
        wn.set_time_params(duration=env_settings['duration'], hydraulic_step=env_settings['hyd_step'])
        #TODO: set all the configuration needed before starting the experiment like tanks level and demand patterns
        wn.run()

        wn.create_df_reports()
        where = Path(config_folder / self.exp_configs['experiment_folder'] / config_files['output_folder'])
        wn.save_csv_reports(where_to_save=where, save_links=False, save_nodes=False)

    def run_agent_experiment(self, config_files):
        """
        Run a single experiment with RL agent.
        """
        config_files = [Path(config_folder / self.exp_configs['experiment_folder'] / config_files[key])
                        if config_files[key] else None for key in config_files.keys()]
        self.agent.build_env(*config_files)
        self.agent.run()

    def build_rl_agent(self):
        """
        Build the RL agent specified in the experiment configuration file
        """
        agent_list = [
            DeepQNetwork
            # meta-learner
        ]

        # We build the agent object from the class name
        agent_class = globals()[self.exp_configs['agent_class']]

        if agent_class in agent_list:
            self.agent = agent_class(self.exp_configs['agent_config'])
        else:
            raise Exception('Agent not implemented or existent!')