import pandas as pd
import numpy as np
import random
import yaml
from mushroom_rl.core.environment import Environment, MDPInfo
from mushroom_rl.utils.spaces import Discrete, Box
from main.agent import objFunction
from main.physical_process_new import WaterDistributionNetwork
from main.network_ics.generic_plc import SensorPLC, ActuatorPLC


class WaterNetworkEnvironment(Environment):

    def __init__(self, config_file):
        """
        :param town:
        :param state_vars:
        :param action_vars:
        :param duration:
        :param hyd_step:
        :param pattern_step:
        :param pattern_files:
        """
        with open(config_file, 'r') as fin:
            env_config = yaml.safe_load(fin)

        self.town = env_config['town']
        self.state_vars = env_config['state_vars']
        self.action_vars = env_config['action_vars']

        self.duration = env_config['duration']
        self.hyd_step = env_config['hyd_step']
        self.pattern_step = env_config['pattern_step']

        # Demand patterns
        self.patterns_train = env_config['pattern_files']['train']
        self.patterns_train_full_range_csv = env_config['pattern_files']['train_full_range']
        self.patterns_train_low_csv = env_config['pattern_files']['train_low']
        self.patterns_test_csv = env_config['pattern_files']['test']

        self.demand_moving_average = None
        self.seed = env_config['seed']
        self.on_eval = False

        self.wn = WaterDistributionNetwork(self.town + '.inp')
        self.wn.set_time_params(duration=self.duration, hydraulic_step=self.hyd_step, pattern_step=self.pattern_step)

        self.curr_time = None
        self.timestep = None
        self.done = False
        self.total_updates = 0
        self.dsr = 0

        # Initialize ICS component
        self.sensor_plcs = []
        self.actuator_plcs = []
        self.plc_vars = {}
        self.build_ics_devices()

        # Two possible values for each pump: 2 ^ n_pumps
        action_space = Discrete(2 ** len(self.action_vars))

        # Current state
        self._state = None

        # Bounds for observation space
        # TODO: edit this
        lows = np.array([bounds[key]['min'] for key in bounds.keys()])
        highs = np.array([bounds[key]['max'] for key in bounds.keys()])

        # Observation space
        observation_space = Box(low=lows, high=highs, shape=(len(bounds),))

        # TODO: what is horizon?
        mdp_info = MDPInfo(observation_space, action_space, gamma=0.99, horizon=1000000)
        super().__init__(mdp_info)

    def reset(self, state=None):
        """
        Called at the beginning of each episode
        :param state:
        :return:
        """
        self.wn.reset()

        if self.on_eval:
            if self.patterns_test_csv:
                junc_demands = pd.read_csv(self.patterns_test_csv)

                if self.seed is not None and self.seed < len(junc_demands.columns):
                    col = junc_demands.columns.values[self.seed]
                else:
                    col = random.choice(junc_demands.columns.values)
                print("col: ", col)
                self.wn.set_demand_pattern('junc_demand', junc_demands[col], self.wn.junctions)

        else:
            if self.patterns_train:
                # Set pattern file choosing randomly between full range or low demand pattern
                junc_demands = pd.read_csv(self.patterns_train)
                col = random.choice(junc_demands.columns.values)
                print("col: ", col)
                self.wn.set_demand_pattern('junc_demand', junc_demands[col], self.wn.junctions)

        if 'demand_SMA' in self.state_vars.keys():
            self.demand_moving_average = junc_demands[col].rolling(window=6, min_periods=1).mean()
            # self.demand_moving_average = junc_demands[col].ewm(alpha=0.1, adjust=False).mean()

        self.wn.init_simulation()
        self.curr_time = 0
        self.timestep = 1
        self.seed = None
        self.wn.solved = False
        self.done = False
        self.total_updates = 0
        self._state = self.build_current_state(reset=True)
        return self._state

    def step(self, action):
        """

        :param action:
        :return:
        """
        n_updates = 0

        # Actuator PLCs apply the action into the physical process
        for plc in self.actuator_plcs:
            n_updates += plc.apply(action)

        # n_updates = self.wn.update_actuators_status(new_status_dict)
        self.total_updates += n_updates

        # Simulate the next hydraulic step
        # TODO: understand if we want to simulate also intermediate steps (not as DHALSIM)
        self.timestep = self.wn.simulate_step(self.curr_time)
        self.curr_time += self.timestep
        while self.curr_time % self.hyd_step != 0 and self.timestep != 0:
            self.timestep = self.wn.simulate_step(self.curr_time)
            self.curr_time += self.timestep

        readings = {}
        for sensor in self.sensor_plcs:
            readings[sensor.name] = sensor.apply()

        # Retrieve current state and reward from the chosen action
        self._state = self.build_current_state(readings=readings)
        reward = self.compute_reward(n_updates)

        info = None

        if self.timestep == 0:
            self.done = True
            self.wn.solved = True
            self.dsr = self.evaluate()
            #reward = self.compute_reward(n_actuators_updates=0)
            self.logger.results(self.dsr, self.total_updates)

        return self._state, reward, self.done, info

    def render(self):
        pass

    def build_ics_devices(self):
        """
        Create instances of acturators and sensors
        # TODO: implement
        """
        pass

    def build_current_state(self, readings, reset=False):
        """
        Build current state list, which can be used as input of the nn saved_models
        :param readings:
        :param reset:
        :return:
        """
        state = []

        # TODO: understand how to initialize the reset state
        if reset:
            for var in self.state_vars.keys():
                if var == 'time':
                    state.append(0)
                elif var == 'day':
                    state.append(1)
                elif var == 'demand_SMA':
                    state.append(0)
                elif var == 'attack_flag':
                    state.append(0)
                else:
                    state.append(0)
        else:
            seconds_per_day = 3600 * 24
            days_per_week = 7
            current_hour = 0    # TODO: modify current hour

            for var in self.state_vars.keys():
                if var == 'time':
                    state.append(self.curr_time % seconds_per_day)
                elif var == 'day':
                    state.append(((self.curr_time // seconds_per_day) % days_per_week) + 1)
                elif var == 'demand_SMA':
                    state.append(self.demand_moving_average.iloc[current_hour])
                elif var == 'under_attack':
                    attack_flag = 0
                    for plc in self.plc_vars.keys():
                        if readings[plc]['attack_flag']:
                            attack_flag = 1
                            break
                    state.append(attack_flag)
                else:
                    for plc in self.plc_vars.keys():
                        if var in self.plc_vars[var]:
                            state.append(readings[plc][var])

        state = [np.float32(i) for i in state]
        return state

    def check_overflow(self):
        """
        Check if the we have an overflow problem in the tanks. We have an overflow if after one hour we the tank is
        still at the maximum level.
        :return: penalty value
        """
        penalty = 1
        risk_percentage = 0.9

        for tank in self.wn.tanks:
            if tank.results['pressure'][-1] > tank.maxlevel * risk_percentage:
                out_bound = tank.results['pressure'][-1] - (tank.maxlevel * risk_percentage)
                # Normalization of the out_bound pressure
                multiplier = out_bound / (tank.maxlevel - tank.maxlevel * risk_percentage)
                return penalty * multiplier
        return 0
        #     if len(tank.results['pressure']) > 5:
        #         last_levels = tank.results['pressure'][-5:-1]
        #         is_overflow = True
        #
        #         for level in last_levels:
        #             if level != tank.maxlevel:
        #                 is_overflow = False
        #                 break
        #
        #         if is_overflow:
        #             print("OVERFLOW!!!")
        #             return 50
        #         else:
        #             return 0
        # return 0

    def compute_reward(self, n_actuators_updates):
        """
        TODO: understand how to compute reward
        Compute the reward for the current step. It depends on the step_DSR and on the number of actuators updates.
        :param n_actuators_updates:
        :return:
        """
        if self.done:
            if self.dsr < 0.9:
                return -100
            else:
                return 0
            #    return 10 ** (1/self.dsr)

        overflow_penalty = self.check_overflow()
        #if not self.on_eval:
        #    print("OVERFLOW RISK: ", overflow_penalty)
        dsr_ratio = objFunction.step_supply_demand_ratio(self.wn)
        # TODO: edit this
        if self.update_every:
            return dsr_ratio - overflow_penalty
        else:
            reward = -n_actuators_updates/2 + dsr_ratio - overflow_penalty
            return reward

    def evaluate(self):
        """

        :return:
        """
        return objFunction.supply_demand_ratio(self.wn)

    def get_state(self):
        return self._state

