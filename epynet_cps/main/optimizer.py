from pathlib import Path
import yaml
import skopt
from physical_process import WaterDistributionNetwork
from agent.dqn import DeepQNetwork

config_folder = Path("../experiments")
optimization_file = "main_optimization.yaml"

with open(optimization_file, 'r') as fin:
    exp_configs = yaml.safe_load(fin)


# Creating list containing parameters which have to be optimized
targets = exp_configs['targets']
PARAMS = []
for key in targets.keys():
    PARAMS.append(skopt.space.Real(targets[key]['min'], targets[key]['max'], name=key, prior='uniform'))


# Build RL agent
agent = None
sim_counter = 0
agent_list = [
        DeepQNetwork
        # meta-learner
    ]
agent_class = globals()[exp_configs['agent_class']]

if agent_class in agent_list:
    agent = agent_class(exp_configs['agent_config'])
else:
    raise Exception('Agent not implemented or existent!')


@skopt.utils.use_named_args(PARAMS)
def objective(**params):
    config_files = exp_configs['experiment']
    config_files = [Path(config_folder / exp_configs['experiment_folder'] / config_files[file])
                    if config_files[file] else None for file in config_files.keys()]
    global sim_counter
    output_file = Path(config_folder / exp_configs['experiment_folder'] / ("results/opt_" + str(sim_counter)))
    agent.build_env(*config_files, output_file_path=output_file)
    sim_counter += 1

    agent.env.set_reward_weights(params.values())
    agent.run()

    # Objective function which has to be maximized: avg_DSR - avg_overflow
    mean_dsr = sum([agent.results['eval'][i]['dsr'] for i in range(4)]) / 4
    # Average overflow penalty normalized wrt max penalty (=1) at each step, so the length of the list.
    # Then averaged again over each test simulation
    mean_overflow = sum([(sum(agent.results['eval'][i]['overflow']) /
                          len(agent.results['eval'][i]['overflow']) /
                          len(agent.results['eval'][i]['overflow'])) for i in range(4)]) / 4
    print(mean_overflow)

    return -1 * (mean_dsr - mean_overflow)


# Run optimization
results = skopt.forest_minimize(objective, PARAMS, n_calls=20, n_random_starts=10)
best_auc = -1.0 * results.fun
best_params = results.x

output_results_file = Path(config_folder / exp_configs['experiment_folder'] / "results/optimization_results.txt")
print('best result: ', best_auc)
print('best parameters: ', best_params)
print(results)

with open(output_results_file, 'w') as fin:
    fin.write(str(results))

