# ======================================================
#   Файл конфигурации путей и преременных проекта
# ======================================================

import os
import logging
import pickle
import json
import sys
import pandas as pd

from dotenv import load_dotenv

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from typing import Dict, List, Any

from pathlib import Path
parent_dir = Path(__file__).parent
sys.path.append(str(parent_dir))


load_dotenv()

PATH_TRAIN_RAW = os.getenv('PATH_TRAIN_RAW')
PATH_TRAIN_FINAL = os.getenv('PATH_TRAIN_RAW')
PATH_TRAIN_PROCESSED = os.getenv('PATH_TRAIN_PROCESSED')
PATH_TRAIN_FINAL = os.getenv('PATH_TRAIN_FINAL')

PATH_TRAIN_ADD_RAW = os.getenv('PATH_TRAIN_ADD_RAW')
PATH_TRAIN_ADD_FINAL = os.getenv('PATH_TRAIN_ADD_FINAL')
PATH_LOG = os.getenv("PATH_LOG")
PATH_SKALERS = Path("skalers")

base_logs_path = Path(PATH_LOG)

paths = [
    PATH_TRAIN_RAW,
    PATH_TRAIN_PROCESSED,
    PATH_TRAIN_FINAL,
    PATH_TRAIN_ADD_RAW,
    PATH_TRAIN_ADD_FINAL,
    PATH_LOG,
    PATH_SKALERS
]

[os.mkdir(path) for path in paths if not os.path.isdir(path)]

logging.basicConfig(
    level = logging.INFO,
    filename =  base_logs_path.joinpath('logs.log'),
    filemode = "w",
    format = "%(asctime)s %(levelname)s %(message)s"
)
main_logger = logging.getLogger(__name__)


if __name__ == '__main__':
    # Проверка существования дирректорий. Создаем если нет
    logging.info("CONFIG COMPLETE")
