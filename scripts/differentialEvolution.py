import time
import datetime
from tqdm import tqdm
import csv
import pandas as pd
import numpy as np
from scripts import network
from scripts import objFunction


class DifferentialEvolution:

    def __init__(self, wn: network.WaterDistributionNetwork, obj_func, objfunc_params, vars_list, n_pop, n_generations,
                 CR, F):
        """
        Initializes differential evolution and simulation handling parameters
        :param wds: water distribution network
        :param obj_func: function to maximize
        :param objfunc_params: parameters that have to be passed to the objective function
        :param vars_list: control variables ids (pumps and valves)
        :param n_pop: population amount
        :param n_generations: number of desired generations
        :param CR: cross-probability factor for recombination
        :param F: mutation factor
        :param info: descriptive annotation to specify the result file
        """
        self.wn = wn
        self.obj_func = obj_func
        self.objfunc_params = objfunc_params
        self.vars_list = vars_list
        self.n_pop = n_pop
        self.n_generations = n_generations
        self.CR = CR
        self.F = F

        self.duration = 0
        self.hyd_step = 0
        self.update_interval = 0
        self.n_updates = 0

        self.best_candidate = {'config': {}, 'value': float('-inf')}
        self.population = []
        self.last_population = []
        self.results_file = ''
        self.save_results = False
        self.start_time = 0

    def populate(self):
        """
        Gets the population among which we want to apply differential evolution
        :return: list of lists with length equal to n_updates * n_actuators
        """
        # Floor division between duration and update_interval gives the number of update we will perform
        self.n_updates = self.duration // self.update_interval
        population = [np.random.randint(2, size=self.n_updates * len(self.vars_list)) for i in range(self.n_pop)]
        return population

    def run(self, duration=86400, hyd_step = 600, update_interval=14400, n_simulations=1, keep_same_pop=False,
            save_results=False, info=''):
        """
        Wrapper method of the real run to set the configuration of the current DE simulation
        :param duration: duration of the simulation
        :param hyd_step: hydraulic step
        :param update_interval: time interval between each update
        :param n_simulations: number of successive simulations (e.g. 7 if we want to simulate a week)
        :param keep_same_pop: to keep the best population from the previous simulation
        :param save_results: flag which specifies if save or not results
        :param info: descriptive annotation on saved file

        How it works with patterns: we need to assign our demand patters of a week in the main function, then they will
        be taken 24 at time, since we consider multipliers last 1 hour.
        If we want a different pattern step we have to change a bit the function and generate appropriate patterns.
        """
        self.duration = duration
        self.hyd_step = hyd_step
        self.update_interval = update_interval
        self.save_results = save_results
        if self.save_results:
            self.create_output_file(info)
        self.start_time = time.time()

        patterns_uid = set([pattern.uid for pattern in self.wn.junctions.pattern])

        # If we want to run multiple serial simulations (for example, a week with one day duration each)
        if n_simulations > 1:
            for i in range(n_simulations):
                if self.save_results:
                    self.write_file('DAY ' + str(i + 1) + '\n')
                print('DAY ' + str(i + 1) + ':')

                # Keep the last computed population from the day before
                if keep_same_pop:
                    if not self.last_population:
                        self.last_population = self.populate()
                    self.__run(self.last_population)
                    self.last_population = self.population
                else:
                    pop = self.populate()
                    self.__run(pop)

                if i < n_simulations - 1:
                    for uid in patterns_uid:
                        # TODO: we should allow to modify the pattern step (here is 1 hour)
                        self.wn.set_demand_pattern(uid, self.wn.patterns[uid].values[24:])

                self.best_candidate = {'config': {}, 'value': 0}
                if self.save_results:
                    self.write_file('---------------------------------\n')
        # Runs only one simulation (for example, just one day)
        else:
            pop = self.populate()
            self.__run(pop)

    def __run(self, pop):
        """
        Runs one simulation of the differential evolution algorithm
        :param pop: population of the current simulation
        :return:
        """
        self.wn.set_time_params(duration=self.duration, hydraulic_step=self.hyd_step)

        self.population = pop
        # while iteration < max_iterations:
        for i in range(self.n_generations):
            # for individual in population:
            if self.save_results:
                self.write_file("Generation " + str(i + 1) + ':\n')
            print("> STARTING GENERATION " + str(i + 1) + ": ...")

            # TODO: pbar = progress bar, delete all prints
            for k, target in enumerate(self.population):
                mutant = self.mutatation(k)
                candidate = self.recombination(target, mutant)
                if self.selection(target, candidate):
                    self.population[k] = candidate
                print(" ----- Elapsed time [{}] -----".format(datetime.timedelta(seconds=time.time() - self.start_time)))
                # TODO: DE stopping conditions

            print("> GENERATION DONE!")
            print("> ------------------------------------------------")
            print("> Best generation member: " + str(self.best_candidate['config']))
            print("> Best generation value: " + str(self.best_candidate['value']))
            print("> ------------------------------------------------")

            if self.save_results:
                self.write_file("> Best candidate: " + str(self.best_candidate['config']) + '\n')
                self.write_file("> Best candidate value: " + str(self.best_candidate['value']) + '\n')

        print('\n')
        print("Best simulation candidate: " + str(self.best_candidate['config']))
        print("Best simulation candidate value: " + str(self.best_candidate['value']))
        print('\n')

    def mutatation(self, k):
        """
        Compute the new mutant by using binary mutation as explained in Gong-Tuson paper on BDE.
        # TODO: check if iBDE mutation explained by Doerr-Zheng works better
        :param k: target index of the population list
        :return: mutated candidate
        """
        pop_range = [idx for idx in range(self.n_pop) if idx != k]
        idx_selection = np.random.choice(pop_range, 3, replace=False)
        [a, b, c] = [self.population[i] for i in idx_selection ]

        # Rand/1 strategy
        mutant = []
        for j in range(len(a)):
            rand_var = np.random.uniform(0, 1)
            if b[j] != c[j] and rand_var < self.F:
                mutant.append(1 - a[j])
            else:
                mutant.append(a[j])
        return mutant

    def recombination(self, target, mutant):
        """
        Binomial recombination step. TODO: check with exponential
        :param target: target (or parent) candidate
        :param mutant: mutated (or child) candidate
        :return: recombined candidate
        """
        # Binomial strategy
        cross_vars = np.random.rand(len(target)) <= self.CR
        candidate = np.where(cross_vars, mutant, target)
        return candidate

    def selection(self, target, candidate):
        """
        Selects which candidate is the better one after the evaluation phase
        :param target: target (or parent) candidate
        :param candidate: mutated and recombined (or child) candidate
        :return: flag to understand if keep the old candidate or substitute him with the new one
        """
        parent = {}
        child = {}
        idx = 0
        # Builds candidates as dictionaries of pumps
        for pump_id in self.vars_list:
            parent[pump_id] = target[idx * self.n_updates : idx * self.n_updates + self.n_updates]
            child[pump_id] = candidate[idx * self.n_updates : idx * self.n_updates + self.n_updates]
            idx += 1

        parent_value = self.evaluation(parent)
        print(">>> Parent: candidate = {}, value = {}".format(parent, parent_value))
        child_value = self.evaluation(child)
        print(">>> Child: candidate = {}, value = {}".format(child, child_value))

        update_pop = False
        # We maximize the objective function
        if child_value >= parent_value:
            update_pop = True
            # Update the best candidate with the new candidate
            if child_value > self.best_candidate['value']:
                self.best_candidate['config'] = child
                self.best_candidate['value'] = child_value
                print('    -> NEW BEST CANDIDATE FOUND!')
        else:
            # Update the best candidate with the old target
            if parent_value > self.best_candidate['value']:
                self.best_candidate['config'] = parent
                self.best_candidate['value'] = parent_value
                print('    -> NEW BEST CANDIDATE FOUND!')
        print('    --> Substitution: {}'.format(update_pop))
        return update_pop

    def evaluation(self, candidate):
        """
        Wrapper method which runs the simulation and evaluation with the current candidate
        :param candidate: current candidate to be evaluated
        :return: objective function value
        """
        self.simulate_episode(candidate)
        if self.objfunc_params:
            return self.obj_func(self.wn, **self.objfunc_params)
        else:
            return self.obj_func(self.wn)

    def simulate_episode(self, candidate):
        """
        Run a complete simulation of the entire duration to evaluate the current candidate
        :param candidate: dictionary of pumps with, as values, the list of their updates
        """
        self.wn.init_simulation()
        curr_time = 0
        update_time = 0
        timestep = 1

        # Initialize status of pumps with the first element in the update list of the candidate
        init_status = {pump_id: candidate[pump_id][0] for pump_id in candidate.keys()}
        self.wn.update_actuators_status(init_status)

        # timestep becomes 0 the last hydraulic step
        while timestep > 0:
            # Check if it's needed an update (second condition is to avoid the last update on curr_time == duration)
            if update_time >= self.update_interval and not (curr_time > self.duration - self.update_interval):
                update_index = curr_time // self.update_interval
                new_status = {pump_id: candidate[pump_id][update_index] for pump_id in candidate.keys()}
                # new status= {'PU1':1, 'PU2':0}
                self.wn.update_actuators_status(new_status)
                update_time -= self.update_interval

            # Changed simulate_step method since it doesn't update status by itself
            timestep, state = self.wn.simulate_step(curr_time=curr_time)
            curr_time += timestep
            update_time += timestep

        self.wn.ep.ENcloseH()
        self.wn.solved = True
        # self.wds.create_df_reports()

    def create_output_file(self, info):
        """
        Creates new results file
        :param info: descriptive annotation to specify the kind of simulation
        """
        results_folder = "../results/DE/"
        filename = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p_") + info + '.txt'
        self.results_file = results_folder + filename

    def write_file(self, what_to_write):
        """
        Writes on the results file one row at time
        :param what_to_write: string to write
        """
        with open(self.results_file, 'a') as fd:
            fd.write(what_to_write)


if __name__ == '__main__':
    objfunc_params = {
        'target_nodes': ['1', '2', '3'],
        'nodes_band': {'1': [50, 100],
                       '2': [60, 90],
                       '3': [40, 60]
                       }
    }
    objfunc_params = None

    DE_params = {
        'vars_list': ['78', '79'],
        # 'vars_list': ['PU1', 'PU2'],
        # 'vars_list': ['X_Pump_1', 'X_Pump_2', 'X_Pump_3', 'X_Pump_4', 'X_Pump_5'],
        'n_pop': 15,
        'n_generations': 5,
        'CR': 0.7,
        'F': 0.5
    }

    net = network.WaterDistributionNetwork("anytown_pd.inp")
    de = DifferentialEvolution(net, objFunction.supply_demand_ratio, objfunc_params, **DE_params)

    pattern_csv = "../demand_patterns/demands_anytown.csv"
    junc_demands = pd.read_csv(pattern_csv)

    net.set_demand_pattern('junc_demand', junc_demands['130'], net.junctions)

    n_sim = 1
    duration = 24 * 3600 * 7    # 1 day
    hyd_step = 600              # 10 min
    update_interval = 3600 * 4  # 4 hours

    de.run(duration, hyd_step, update_interval, n_simulations=n_sim, save_results=True, keep_same_pop=False,
           info='week_130')
