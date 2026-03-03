# ======================================================
# Тест pipeline
# ======================================================

import pandas as pd
import numpy as np
import logging

from pathlib import Path
from ..src.pipeline.pipeline import Pipeline
from ..src.preprocessing.scaler import Scaler
from ..src.preprocessing.load_data_first import LoadData
from ..src.config import (
    PATH_SKALERS,
    PATH_TRAIN_RAW,
    PATH_LOG
    )


logging.basicConfig(
    level = logging.INFO,
    filename = Path(PATH_LOG).joinpath('tests_ppeline_logs.log'),
    filemode = "w",
    format = "%(asctime)s %(levelname)s %(message)s"
)


# ======================================================
# 1 Подготовка Scaler
# ======================================================
scaler_manager = Scaler()
loader = LoadData()

raw_df = loader.data_raw_load(PATH_TRAIN_RAW)
logging.info(f"Загруенные данные:{raw_df}")

scaler = scaler_manager.fit_scaler()

# ======================================================
# 1 Тестирование Pipeline
# ======================================================
pipeline = Pipeline(PATH_TRAIN_RAW, )