import os
import logging
import datetime


def setup_logger(name, log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, '{}_{}.log'.format(datetime.datetime.now().strftime('%m%d%H%M'), name))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)
