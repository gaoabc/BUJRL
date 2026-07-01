import os
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(log_dir, self.timestamp)
        os.makedirs(self.run_dir, exist_ok=True)

        self.writer = SummaryWriter(self.run_dir)
        self.setup_logging()

    def setup_logging(self):
        self.logger = logging.getLogger("BUJRL")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        file_handler = logging.FileHandler(os.path.join(self.run_dir, "log.txt"))
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag, value_dict, step):
        self.writer.add_scalars(tag, value_dict, step)

    def log_image(self, tag, image, step):
        self.writer.add_image(tag, image, step)

    def log_histogram(self, tag, values, step):
        self.writer.add_histogram(tag, values, step)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def close(self):
        self.writer.close()