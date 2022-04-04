import argparse
import os.path
from pathlib import Path


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error(arg + " does not exist.")
    else:
        return arg


class Runner:

    def __init__(self, config_file, output_folder):
        self.config_file = config_file
        self.output_folder = output_folder

    # TODO: launch the simulation with agent and without...
    def run(self):
        pass


def main():
    parser = argparse.ArgumentParser(description='Executes Epynet_CPS based on a config file')
    parser.add_argument(dest="config_file",
                        help="config file and its path", metavar="FILE",
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument('-o', '--output', dest='output_folder', metavar="FOLDER",
                        help='folder where output files will be saved', type=str)

    args = parser.parse_args()

    config_file = Path(args.config_file)
    output_folder = Path(args.output_folder if args.output_folder else "output")

    runner = Runner(config_file, output_folder)
    runner.run()


if __name__ == '__main__':
    main()