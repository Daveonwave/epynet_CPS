import argparse
from runner import Runner
from pathlib import Path

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error(arg + " does not exist.")
    else:
        return arg


def main():
    parser = argparse.ArgumentParser(description='Executes Epynet_CPS based on a config file')
    parser.add_argument(dest="experiment_file",
                        help="config file and its path", metavar="FILE",
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument('-n', '--no-agent', dest='agent_disabled', action='store_true',
                        help="to run only physical experiments without the RL agent")
    parser.add_argument('-o', '--optimization', dest='optimization_exp', action='store_true',
                        help="to run only physical experiments without the RL agent")

    args = parser.parse_args()

    experiment_file = Path(args.experiment_file)
    agent_disabled = args.agent_disabled
    optimization_exp = args.optimization_exp

    if agent_disabled:
        print("AGENT DISABLED")
    else:
        print("RUN WITH AGENT")

    assert optimization_exp and not agent_disabled, f"It's not possible to optimize without agent running..."

    if optimization_exp:
        os.system("optimizer.py")
    else:
        runner = Runner(experiment_file, agent_disabled)
        runner.run()


if __name__ == '__main__':
    import os
    os.chdir('../main')

    main()
