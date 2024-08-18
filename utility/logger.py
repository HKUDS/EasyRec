import os
import logging
import datetime

def get_local_time():
    return datetime.datetime.now().strftime('%b-%d-%Y_%H-%M-%S')

class EasyrecEmbedderTrainingLogger(object):
    def __init__(self, model_args, data_args, training_args):
        self.eval_steps = training_args.eval_steps
        base_model = model_args.model_name_or_path
        log_dir_path = './log'
        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)
        output_model = training_args.output_dir.split('/')[-1]
        self.logger = logging.getLogger('train_logger')
        self.logger.setLevel(logging.INFO)
        log_file = logging.FileHandler('{}/{}.log'.format(log_dir_path, output_model), 'a', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        log_file.setFormatter(formatter)
        self.logger.addHandler(log_file)
        self.cnt = 0

    def log(self, message, save_to_log=True, print_to_console=True):
        if int(os.environ['LOCAL_RANK']) == 0:
            self.logger.info(message)
            print(message)

    def log_eval(self, eval_result, save_to_log=True, print_to_console=True):
        if int(os.environ['LOCAL_RANK']) == 0:
            self.cnt += 1
            message = 'Step {:6d} '.format(self.eval_steps * self.cnt)
            message += '['
            for metric in eval_result:
                message += '{}: {:.4f} '.format(metric, eval_result[metric])
            message += '] '
            if save_to_log:
                self.logger.info(message)
            if print_to_console:
                print(message)