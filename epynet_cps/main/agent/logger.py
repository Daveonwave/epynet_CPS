from mushroom_rl.utils.dataset import compute_metrics
from mushroom_rl.core import Logger


class InfoLogger:
    def __init__(self, folder_name=None, folder_path=None):
        # TODO: create better namings and check dir existence
        self.logger = None
        if folder_path and folder_path:
            self.logger = Logger(log_name=folder_name, results_dir=folder_path, log_console=True)

    def experiment_summary(self, info: str):
        if self.logger:
            self.logger.strong_line()
            self.logger.info(info)
            self.logger.strong_line()
        else:
            print(info)

    def fill_replay_memory(self):
        if self.logger:
            self.logger.info('Filling replay memory')
            self.logger.weak_line()

    def print_epoch(self, epoch):
        if self.logger:
            self.logger.epoch_info(epoch=epoch)
            self.logger.weak_line()

    def get_stats(self, dataset):
        score = compute_metrics(dataset)
        # self.logger.info('min_reward: %f, max_reward: %f, mean_reward: %f, games_completed: %d')
        return score

    def training_phase(self):
        if self.logger:
            self.logger.info('Learning...')
            self.logger.weak_line()
        else:
            print('Learning...')

    def evaluation_phase(self):
        if self.logger:
            self.logger.info('Evaluation..')
            self.logger.weak_line()
        else:
            print('Evaluating...')

    def end_phase(self):
        if self.logger:
            self.logger.strong_line()

    def log_results(self, seed, dsr, n_updates):
        if self.logger:
            self.logger.info('Demand pattern: ' + str(seed))
            self.logger.info('DSR: ' + str(dsr))
            self.logger.info('Total updates: ' + str(n_updates))
            self.logger.weak_line()
        else:
            print('DSR: ' + str(dsr))
            print('Total updates: ' + str(n_updates))
