
# ------------------------------------------------------
# Файл конфигурации путей и преременных проекта
# ------------------------------------------------------

import os
import logging

from pathlib import Path
from dotenv import load_dotenv


load_dotenv()

PATH_TRAIN_RAW = os.getenv('PATH_TRAIN_RAW')
PATH_TRAIN_FINAL = os.getenv('PATH_TRAIN_RAW')
PATH_TRAIN_PROCESSED = os.getenv('PATH_TRAIN_PROCESSED')
PATH_TRAIN_FINAL = os.getenv('PATH_TRAIN_FINAL')

PATH_TRAIN_ADD_RAW = os.getenv('PATH_TRAIN_ADD_RAW')
PATH_TRAIN_ADD_FINAL = os.getenv('PATH_TRAIN_ADD_FINAL')
PATH_LOG = os.getenv("PATH_LOG")

base_logs_path = Path(PATH_LOG)

paths = [
    PATH_TRAIN_RAW,
    PATH_TRAIN_PROCESSED,
    PATH_TRAIN_FINAL,
    PATH_TRAIN_ADD_RAW,
    PATH_TRAIN_ADD_FINAL,
    PATH_LOG
]


if __name__ == '__main__':
    # Проверка существования дирректорий
    [os.mkdir(path) for path in paths if not os.path.isdir(path)]

    logging.basicConfig(
        level = logging.INFO,
        filename =  base_logs_path.joinpath('logs.log'),
        filemode = "w",
        format = "%(asctime)s %(levelname)s %(message)s"
    )

    logging.info("CONFIG COMPLETE")