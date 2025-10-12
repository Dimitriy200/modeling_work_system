
# ------------------------------------------------------
# Файл конфигурации путей и преременных проекта
# ------------------------------------------------------

import os
import logging

from dotenv import load_dotenv


logging.basicConfig(level=logging.INFO,
                    # filename=os.path.join(os.path.abspath("preprocess-data"),"src", "prepData", "logs", "prep_data_logs.log" ),
                    filemode="w",
                    format="%(asctime)s %(levelname)s %(message)s")

load_dotenv()

PATH_TRAIN_RAW = os.getenv('PATH_TRAIN_RAW')
PATH_TRAIN_FINAL = os.getenv('PATH_TRAIN_RAW')
PATH_TRAIN_PROCESSED = os.getenv('PATH_TRAIN_PROCESSED')
PATH_TRAIN_FINAL = os.getenv('PATH_TRAIN_FINAL')

PATH_TRAIN_ADD_RAW = os.getenv('PATH_TRAIN_ADD_RAW')
PATH_TRAIN_ADD_FINAL = os.getenv('PATH_TRAIN_ADD_FINAL')

paths = [
    PATH_TRAIN_RAW,
    PATH_TRAIN_PROCESSED,
    PATH_TRAIN_FINAL,
    PATH_TRAIN_ADD_RAW,
    PATH_TRAIN_ADD_FINAL
]


if __name__ == '__main__':
    # Проверка существования дирректорий
    [os.mkdir(path) for path in paths if not os.path.isdir(path)]
